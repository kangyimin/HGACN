import argparse
import json
import os
import time

import numpy as np
import torch

from data_preprocess import (
    load_and_construct_hypergraphs,
    load_psichic_attention,
)
from model import HGACN
from graphsaint_sampler import random_walk_subgraph, build_subgraph_edges
from train import union_ragged, build_edge_incidence
from attribution import minmax_norm, combine_attention, grad_score, select_edge_index
from viz_drug_rdkit import render_drug_atom_importance
from viz_protein_pdb import write_residue_scores_to_pdb, export_topk_residues_csv
from export_html import export_html_report


def _build_id_list(mapping):
    max_idx = max(mapping.values()) if mapping else -1
    id_list = [None] * (max_idx + 1)
    for k, v in mapping.items():
        if 0 <= v < len(id_list):
            id_list[v] = k
    return id_list


def _load_run_config(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists("run_config.json"):
        with open("run_config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_ids(args, dataset_dir):
    with open(os.path.join(dataset_dir, "drug_to_idx.pkl"), "rb") as f:
        drug_to_idx = json.load(f) if f.name.endswith(".json") else __import__("pickle").load(f)
    with open(os.path.join(dataset_dir, "protein_to_idx.pkl"), "rb") as f:
        prot_to_idx = json.load(f) if f.name.endswith(".json") else __import__("pickle").load(f)
    drug_id_list = _build_id_list(drug_to_idx)
    prot_id_list = _build_id_list(prot_to_idx)

    drug_idx = args.drug_id
    prot_idx = args.prot_id
    if drug_idx is None and args.smiles:
        drug_idx = drug_to_idx.get(args.smiles)
    if prot_idx is None and args.fasta:
        prot_idx = prot_to_idx.get(args.fasta)
    if drug_idx is None or prot_idx is None:
        raise ValueError("Missing drug_id/prot_id or smiles/fasta not found in mapping.")
    return int(drug_idx), int(prot_idx), drug_id_list, prot_id_list

def _prepare_subgraph(ctx, drug_idx, prot_idx, seed=0, use_random_walk=False):
    atom_seed = union_ragged(ctx["drug_atom_ptr"], ctx["drug_atom_nodes"], np.array([drug_idx], dtype=np.int64))
    res_seed = union_ragged(ctx["prot_res_ptr"], ctx["prot_res_nodes"], np.array([prot_idx], dtype=np.int64))
    if atom_seed.size == 0 or res_seed.size == 0:
        raise ValueError("Empty atom/residue seed nodes for this pair.")

    if use_random_walk:
        rng = np.random.default_rng(seed)
        atom_sub = random_walk_subgraph(
            ctx["atom_indptr"], ctx["atom_indices"], atom_seed,
            ctx["walk_length"], ctx["num_walks"], ctx["max_atoms"], rng
        )
        res_sub = random_walk_subgraph(
            ctx["res_indptr"], ctx["res_indices"], res_seed,
            ctx["walk_length"], ctx["num_walks"], ctx["max_res"], rng
        )
    else:
        atom_sub = atom_seed
        res_sub = res_seed

    if atom_sub.size == 0 or res_sub.size == 0:
        raise ValueError("Empty subgraph after sampling.")

    atom_edge_index, atom_edge_weight = build_subgraph_edges(
        ctx["atom_indptr"], ctx["atom_indices"], ctx["atom_data"], atom_sub
    )
    res_edge_index, res_edge_weight = build_subgraph_edges(
        ctx["res_indptr"], ctx["res_indices"], ctx["res_data"], res_sub
    )

    return atom_sub, res_sub, atom_edge_index, atom_edge_weight, res_edge_index, res_edge_weight


def _edge_logit_from_repr(model, drug_node_repr, prot_node_repr,
                          alpha_atom, alpha_res, drug_inv, prot_inv,
                          drug_ids, prot_ids, edge_drug_id, edge_prot_id):
    device = drug_node_repr.device
    num_drug = int(drug_ids.numel())
    num_prot = int(prot_ids.numel())
    drug_repr = drug_node_repr.new_zeros((num_drug, drug_node_repr.size(1)))
    prot_repr = prot_node_repr.new_zeros((num_prot, prot_node_repr.size(1)))
    drug_repr.index_add_(0, drug_inv, drug_node_repr * alpha_atom.view(-1, 1))
    prot_repr.index_add_(0, prot_inv, prot_node_repr * alpha_res.view(-1, 1))
    d_pos = torch.searchsorted(drug_ids, torch.tensor([edge_drug_id], device=device))
    p_pos = torch.searchsorted(prot_ids, torch.tensor([edge_prot_id], device=device))
    d_pos = int(d_pos.item())
    p_pos = int(p_pos.item())
    edge_drug = drug_repr[d_pos:d_pos + 1]
    edge_prot = prot_repr[p_pos:p_pos + 1]
    return model._edge_logits(edge_drug, edge_prot).view(-1)


def _integrated_gradients(model, drug_node_repr, prot_node_repr,
                          alpha_atom, alpha_res, drug_inv, prot_inv,
                          drug_ids, prot_ids, edge_drug_id, edge_prot_id,
                          steps=16):
    device = drug_node_repr.device
    baseline_d = torch.zeros_like(drug_node_repr, device=device)
    baseline_p = torch.zeros_like(prot_node_repr, device=device)
    total_grad_d = torch.zeros_like(drug_node_repr, device=device)
    total_grad_p = torch.zeros_like(prot_node_repr, device=device)

    for i in range(steps):
        t = float(i + 1) / float(steps)
        d_in = baseline_d + t * (drug_node_repr - baseline_d)
        p_in = baseline_p + t * (prot_node_repr - baseline_p)
        d_in.requires_grad_(True)
        p_in.requires_grad_(True)
        logit = _edge_logit_from_repr(
            model, d_in, p_in, alpha_atom, alpha_res,
            drug_inv, prot_inv, drug_ids, prot_ids, edge_drug_id, edge_prot_id
        )
        logit.sum().backward()
        if d_in.grad is not None:
            total_grad_d = total_grad_d + d_in.grad.detach()
        if p_in.grad is not None:
            total_grad_p = total_grad_p + p_in.grad.detach()
    avg_grad_d = total_grad_d / float(steps)
    avg_grad_p = total_grad_p / float(steps)
    ig_d = (drug_node_repr - baseline_d) * avg_grad_d
    ig_p = (prot_node_repr - baseline_p) * avg_grad_p
    return ig_d, ig_p


def _map_scores_to_entity(node_scores, sub_nodes, node_to_entity, entity_id, orig_pos):
    node_scores = node_scores.detach().cpu().numpy()
    sub_nodes = np.asarray(sub_nodes, dtype=np.int64)
    node_to_entity = node_to_entity.detach().cpu().numpy()
    mask = node_to_entity == int(entity_id)
    if not np.any(mask):
        return np.zeros((0,), dtype=np.float32)
    nodes = sub_nodes[mask]
    scores = node_scores[mask]
    if orig_pos is None:
        return scores
    orig = orig_pos[nodes]
    size = int(orig.max()) + 1 if orig.size else 0
    out = np.zeros((size,), dtype=np.float32)
    for pos, s in zip(orig.tolist(), scores.tolist()):
        if pos >= 0 and pos < size:
            out[pos] = max(out[pos], float(s))
    return out


def _topk_list(scores, topk=20):
    if scores is None or len(scores) == 0:
        return []
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)
    order = order[: min(int(topk), scores.size)]
    return [{"idx": int(i), "score": float(scores[i])} for i in order]

def load_context(args):
    data_root = os.path.join(os.getcwd(), "data")
    data = load_and_construct_hypergraphs(
        args.dataset, data_root, node_level=args.node_level,
        psichic_attention_path=args.psichic_attention, add_self_loop=not args.no_self_loop,
        protein_feat_mode=args.protein_feat_mode,
        esm_special_tokens=args.esm_special_tokens,
        esm_norm=args.esm_norm,
        esm_strict=args.esm_strict,
        reuse_cache=args.reuse_cache,
        prune_strategy_tag=args.prune_strategy_tag,
        atom_topk=args.atom_topk,
        atom_randk=args.atom_randk,
        res_topk=args.res_topk,
        res_randk=args.res_randk,
        randk_seed=args.randk_seed,
        randk_weight_mode=args.randk_weight_mode,
        prior_floor=args.prior_floor,
        use_knn_graph=args.use_knn_graph,
        drug_knn_k=args.drug_knn_k,
        prot_knn_k=args.prot_knn_k,
        knn_metric=args.knn_metric,
        knn_symmetric=args.knn_symmetric,
        knn_weight_temp=args.knn_weight_temp,
        knn_setting=args.knn_setting,
    )
    (train_edges, val_edges, test_edges,
     num_drugs, num_prots,
     H_atom, H_residue, G_atom, G_residue,
     features_atom, features_residue,
     atom_to_drug, residue_to_protein,
     atom_attn, residue_attn,
     atom_orig_pos, residue_orig_pos,
     drug_atom_ptr, drug_atom_nodes,
     prot_res_ptr, prot_res_nodes,
     prot_esm_missing, prot_esm_unreliable,
     drug_knn_edge_index, drug_knn_edge_weight,
     prot_knn_edge_index, prot_knn_edge_weight) = data

    from scipy import sparse
    if not sparse.isspmatrix_csr(G_atom):
        G_atom = G_atom.tocsr()
    if not sparse.isspmatrix_csr(G_residue):
        G_residue = G_residue.tocsr()

    ctx = {
        "features_atom": features_atom,
        "features_residue": features_residue,
        "atom_to_drug": atom_to_drug,
        "residue_to_protein": residue_to_protein,
        "atom_attn": atom_attn,
        "residue_attn": residue_attn,
        "atom_orig_pos": atom_orig_pos,
        "residue_orig_pos": residue_orig_pos,
        "drug_atom_ptr": drug_atom_ptr,
        "drug_atom_nodes": drug_atom_nodes,
        "prot_res_ptr": prot_res_ptr,
        "prot_res_nodes": prot_res_nodes,
        "prot_esm_missing": prot_esm_missing,
        "prot_esm_unreliable": prot_esm_unreliable,
        "drug_knn_edge_index": drug_knn_edge_index,
        "drug_knn_edge_weight": drug_knn_edge_weight,
        "prot_knn_edge_index": prot_knn_edge_index,
        "prot_knn_edge_weight": prot_knn_edge_weight,
        "atom_indptr": G_atom.indptr.astype(np.int64, copy=False),
        "atom_indices": G_atom.indices.astype(np.int64, copy=False),
        "atom_data": G_atom.data.astype(np.float32, copy=False),
        "res_indptr": G_residue.indptr.astype(np.int64, copy=False),
        "res_indices": G_residue.indices.astype(np.int64, copy=False),
        "res_data": G_residue.data.astype(np.float32, copy=False),
        "num_drugs": num_drugs,
        "num_prots": num_prots,
        "walk_length": args.explain_walk_length,
        "num_walks": args.explain_num_walks,
        "max_atoms": args.explain_max_atoms,
        "max_res": args.explain_max_res,
    }
    return ctx

def explain_pair(args, ctx, drug_idx, prot_idx, drug_id_list, prot_id_list):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_random_walk = bool(args.explain_use_random_walk)
    atom_sub, res_sub, atom_edge_index, atom_edge_weight, res_edge_index, res_edge_weight = _prepare_subgraph(
        ctx, drug_idx, prot_idx, seed=args.explain_seed, use_random_walk=use_random_walk
    )
    atom_sub_t = torch.from_numpy(atom_sub).to(device=device, dtype=torch.long)
    res_sub_t = torch.from_numpy(res_sub).to(device=device, dtype=torch.long)

    atom_feat_sub = torch.from_numpy(ctx["features_atom"]).to(device=device, dtype=torch.float32)[atom_sub_t]
    res_feat_sub = torch.from_numpy(ctx["features_residue"]).to(device=device, dtype=torch.float32)[res_sub_t]
    sub_atom_to_drug = torch.from_numpy(ctx["atom_to_drug"]).to(device=device, dtype=torch.long)[atom_sub_t]
    sub_res_to_prot = torch.from_numpy(ctx["residue_to_protein"]).to(device=device, dtype=torch.long)[res_sub_t]
    atom_attn_tensor = torch.from_numpy(ctx["atom_attn"]).to(device=device, dtype=torch.float32)[atom_sub_t]
    res_attn_tensor = torch.from_numpy(ctx["residue_attn"]).to(device=device, dtype=torch.float32)[res_sub_t]

    edge_pairs = np.asarray([[drug_idx, prot_idx]], dtype=np.int64)
    attn_db = load_psichic_attention(args.psichic_attention) if args.psichic_attention else None
    pair_cache = {}
    atom_edge_ptr, atom_edge_nodes, atom_edge_prior, _ = build_edge_incidence(
        edge_pairs,
        atom_sub,
        ctx["drug_atom_ptr"],
        ctx["drug_atom_nodes"],
        ctx["atom_orig_pos"],
        attn_db,
        drug_id_list,
        prot_id_list,
        ctx["atom_attn"],
        is_atom_side=True,
        alpha_eps=args.alpha_eps,
        cache=pair_cache,
    )
    res_edge_ptr, res_edge_nodes, res_edge_prior, _ = build_edge_incidence(
        edge_pairs,
        res_sub,
        ctx["prot_res_ptr"],
        ctx["prot_res_nodes"],
        ctx["residue_orig_pos"],
        attn_db,
        drug_id_list,
        prot_id_list,
        ctx["residue_attn"],
        is_atom_side=False,
        alpha_eps=args.alpha_eps,
        cache=pair_cache,
    )
    atom_edge_ptr_t = torch.from_numpy(atom_edge_ptr).to(device=device, dtype=torch.long)
    atom_edge_nodes_t = torch.from_numpy(atom_edge_nodes).to(device=device, dtype=torch.long)
    atom_edge_prior_t = torch.from_numpy(atom_edge_prior).to(device=device, dtype=torch.float32)
    res_edge_ptr_t = torch.from_numpy(res_edge_ptr).to(device=device, dtype=torch.long)
    res_edge_nodes_t = torch.from_numpy(res_edge_nodes).to(device=device, dtype=torch.long)
    res_edge_prior_t = torch.from_numpy(res_edge_prior).to(device=device, dtype=torch.float32)

    atom_edge_index_t = torch.from_numpy(atom_edge_index).to(device=device, dtype=torch.long)
    atom_edge_weight_t = torch.from_numpy(atom_edge_weight).to(device=device, dtype=torch.float32)
    res_edge_index_t = torch.from_numpy(res_edge_index).to(device=device, dtype=torch.long)
    res_edge_weight_t = torch.from_numpy(res_edge_weight).to(device=device, dtype=torch.float32)

    edge_index_t = torch.tensor([[drug_idx, prot_idx]], dtype=torch.long, device=device)

    drug_deg = None
    prot_deg = None
    dataset_dir = os.path.join(os.getcwd(), "data", args.dataset)
    deg_dir = os.path.join(dataset_dir, "processed_atomic")
    drug_deg_path = os.path.join(deg_dir, "drug_deg.npy")
    prot_deg_path = os.path.join(deg_dir, "prot_deg.npy")
    if os.path.exists(drug_deg_path):
        drug_deg = torch.from_numpy(np.load(drug_deg_path).astype(np.float32)).to(device=device)
    if os.path.exists(prot_deg_path):
        prot_deg = torch.from_numpy(np.load(prot_deg_path).astype(np.float32)).to(device=device)

    prot_esm_missing = None
    if ctx["prot_esm_missing"] is not None:
        prot_esm_missing = torch.from_numpy(ctx["prot_esm_missing"].astype(np.float32)).to(device=device)
    prot_esm_unreliable = None
    if ctx.get("prot_esm_unreliable") is not None:
        prot_esm_unreliable = torch.from_numpy(ctx["prot_esm_unreliable"].astype(np.float32)).to(device=device)

    drug_knn_edge_index = None
    drug_knn_edge_weight = None
    prot_knn_edge_index = None
    prot_knn_edge_weight = None
    if ctx["drug_knn_edge_index"] is not None:
        drug_knn_edge_index = torch.from_numpy(ctx["drug_knn_edge_index"]).to(device=device, dtype=torch.long)
        drug_knn_edge_weight = torch.from_numpy(ctx["drug_knn_edge_weight"]).to(device=device, dtype=torch.float32)
    if ctx["prot_knn_edge_index"] is not None:
        prot_knn_edge_index = torch.from_numpy(ctx["prot_knn_edge_index"]).to(device=device, dtype=torch.long)
        prot_knn_edge_weight = torch.from_numpy(ctx["prot_knn_edge_weight"]).to(device=device, dtype=torch.float32)

    model = HGACN(
        drug_feat_dim=atom_feat_sub.size(1),
        prot_feat_dim=res_feat_sub.size(1),
        hidden_dim=args.hidden,
        out_dim=args.out_dim,
        gat_top_k_sparse=args.gat_top_k_sparse,
        gat_dense_top_k=args.gat_dense_top_k,
        interaction=args.interaction_head,
        use_hyperedge_head=args.use_hyperedge_head,
        alpha_refine=not args.no_alpha_refine,
        alpha_eps=args.alpha_eps,
        prior_eps=args.prior_eps,
        alpha_temp=args.alpha_temp,
        use_residual=not args.no_residual,
        use_coldstart_gate=not args.no_coldstart_gate,
        prior_mix_mode=args.prior_mix_mode,
        prior_mix_lambda=args.prior_mix_lambda,
        prior_mix_learnable=args.prior_mix_learnable,
        prior_mix_conditional=args.prior_mix_conditional,
        prior_mix_features=args.prior_mix_features,
        prior_smoothing=args.prior_smoothing,
        pool_topk=args.pool_topk,
        pool_randk=args.pool_randk,
        beta_mix=args.beta_mix,
        randk_weight_mode=args.randk_weight_mode,
        prior_floor=args.prior_floor,
        moe_enable=bool(args.moe_enable),
        expert_A=bool(args.expert_A),
        expert_B=bool(args.expert_B),
        expert_C=bool(args.expert_C),
        use_knn_graph=bool(args.use_knn_graph),
        cold_deg_th_drug=args.cold_deg_th_drug,
        cold_deg_th_prot=args.cold_deg_th_prot,
        mp_gate_mode=args.mp_gate_mode,
        mp_gate_deg_only=args.mp_gate_deg_only,
        mp_gate_use_attn_entropy=bool(args.mp_gate_use_attn_entropy),
        mp_gate_use_prior_entropy=bool(args.mp_gate_use_prior_entropy),
        mp_gate_use_esm_missing=bool(args.mp_gate_use_esm_missing),
        bottleneck_drop=args.bottleneck_drop,
        use_bottleneck_gate=bool(args.use_bottleneck_gate),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    explain_cfg = {
        "aux_to_cpu": args.explain_aux_to_cpu,
        "atom_orig_pos": ctx["atom_orig_pos"],
        "residue_orig_pos": ctx["residue_orig_pos"],
        "drug_atom_ptr": ctx["drug_atom_ptr"],
        "prot_res_ptr": ctx["prot_res_ptr"],
        "drug_id_list": drug_id_list,
        "prot_id_list": prot_id_list,
        "knn_topk": args.knn_topk,
    }
    methods = [m.strip() for m in args.explain_methods.split(",") if m.strip()]
    if "grad" in methods or "ig" in methods:
        explain_cfg["aux_to_cpu"] = False
        explain_cfg["keep_grad"] = True
        explain_cfg["need_node_repr"] = True
        explain_cfg["retain_node_grad"] = True

    with torch.no_grad():
        edge_logits, aux = model(
            atom_feat_sub, res_feat_sub, None, None,
            drug_node_to_entity=sub_atom_to_drug,
            protein_node_to_entity=sub_res_to_prot,
            drug_node_weight=atom_attn_tensor,
            protein_node_weight=res_attn_tensor,
            atom_prior=atom_attn_tensor,
            res_prior=res_attn_tensor,
            edge_index=edge_index_t,
            drug_edge_index=atom_edge_index_t,
            drug_edge_weight=atom_edge_weight_t,
            drug_num_nodes=atom_feat_sub.size(0),
            drug_edge_ptr=atom_edge_ptr_t,
            drug_edge_nodes=atom_edge_nodes_t,
            drug_edge_psichic=atom_edge_prior_t,
            prot_edge_index=res_edge_index_t,
            prot_edge_weight=res_edge_weight_t,
            prot_num_nodes=res_feat_sub.size(0),
            prot_edge_ptr=res_edge_ptr_t,
            prot_edge_nodes=res_edge_nodes_t,
            prot_edge_psichic=res_edge_prior_t,
            drug_degree=drug_deg,
            prot_degree=prot_deg,
            prot_esm_missing=prot_esm_missing,
            prot_esm_unreliable=prot_esm_unreliable,
            drug_knn_edge_index=drug_knn_edge_index,
            drug_knn_edge_weight=drug_knn_edge_weight,
            prot_knn_edge_index=prot_knn_edge_index,
            prot_knn_edge_weight=prot_knn_edge_weight,
            return_aux=True,
            explain_cfg=explain_cfg,
        )
    edge_index_filtered = aux.get("edge_index_filtered")
    edge_pos = select_edge_index(edge_index_filtered, drug_idx, prot_idx)
    if edge_pos is None:
        raise RuntimeError("Target edge not found in filtered edge list.")
    logit = float(edge_logits[edge_pos].item())
    prob = float(torch.sigmoid(edge_logits[edge_pos]).item())

    gate_w = None
    if aux.get("gate_weights") is not None:
        gate_w = aux["gate_weights"][edge_pos].detach().cpu().numpy().tolist()
    alpha_atom = aux.get("atom_attention")
    alpha_res = aux.get("residue_attention")
    alpha_atom_learn = aux.get("atom_attention_learned")
    alpha_res_learn = aux.get("residue_attention_learned")
    if alpha_atom is not None and alpha_atom_learn is not None:
        alpha_atom_mix = combine_attention(alpha_atom, alpha_atom_learn, gate_w)
    else:
        alpha_atom_mix = alpha_atom
    if alpha_res is not None and alpha_res_learn is not None:
        alpha_res_mix = combine_attention(alpha_res, alpha_res_learn, gate_w)
    else:
        alpha_res_mix = alpha_res

    atom_scores_A = None
    res_scores_A = None
    atom_scores_B = None
    res_scores_B = None
    if alpha_atom is not None:
        atom_scores_A = _map_scores_to_entity(
            alpha_atom, atom_sub, sub_atom_to_drug, drug_idx, ctx["atom_orig_pos"]
        )
    if alpha_res is not None:
        res_scores_A = _map_scores_to_entity(
            alpha_res, res_sub, sub_res_to_prot, prot_idx, ctx["residue_orig_pos"]
        )
    if alpha_atom_learn is not None:
        atom_scores_B = _map_scores_to_entity(
            alpha_atom_learn, atom_sub, sub_atom_to_drug, drug_idx, ctx["atom_orig_pos"]
        )
    if alpha_res_learn is not None:
        res_scores_B = _map_scores_to_entity(
            alpha_res_learn, res_sub, sub_res_to_prot, prot_idx, ctx["residue_orig_pos"]
        )

    atom_scores_attn = None
    res_scores_attn = None
    if alpha_atom_mix is not None:
        atom_scores_attn = _map_scores_to_entity(
            alpha_atom_mix, atom_sub, sub_atom_to_drug, drug_idx, ctx["atom_orig_pos"]
        )
    if alpha_res_mix is not None:
        res_scores_attn = _map_scores_to_entity(
            alpha_res_mix, res_sub, sub_res_to_prot, prot_idx, ctx["residue_orig_pos"]
        )

    prior_conf_atom = None
    prior_conf_res = None
    prior_ent_atom_group = aux.get("prior_entropy_atom_group")
    prior_ent_res_group = aux.get("prior_entropy_res_group")
    drug_ids_aux = aux.get("drug_ids")
    prot_ids_aux = aux.get("prot_ids")
    if prior_ent_atom_group is not None and drug_ids_aux is not None and drug_ids_aux.numel():
        dpos = torch.searchsorted(drug_ids_aux, torch.tensor([drug_idx], device=drug_ids_aux.device))
        dpos = int(dpos.item())
        if dpos < int(drug_ids_aux.numel()) and int(drug_ids_aux[dpos].item()) == int(drug_idx):
            prior_conf_atom = float((1.0 - prior_ent_atom_group[dpos]).clamp(0.0, 1.0).item())
    if prior_ent_res_group is not None and prot_ids_aux is not None and prot_ids_aux.numel():
        ppos = torch.searchsorted(prot_ids_aux, torch.tensor([prot_idx], device=prot_ids_aux.device))
        ppos = int(ppos.item())
        if ppos < int(prot_ids_aux.numel()) and int(prot_ids_aux[ppos].item()) == int(prot_idx):
            prior_conf_res = float((1.0 - prior_ent_res_group[ppos]).clamp(0.0, 1.0).item())

    atom_scores_grad = None
    res_scores_grad = None
    if "grad" in methods:
        model.zero_grad()
        edge_logits, aux = model(
            atom_feat_sub, res_feat_sub, None, None,
            drug_node_to_entity=sub_atom_to_drug,
            protein_node_to_entity=sub_res_to_prot,
            drug_node_weight=atom_attn_tensor,
            protein_node_weight=res_attn_tensor,
            atom_prior=atom_attn_tensor,
            res_prior=res_attn_tensor,
            edge_index=edge_index_t,
            drug_edge_index=atom_edge_index_t,
            drug_edge_weight=atom_edge_weight_t,
            drug_num_nodes=atom_feat_sub.size(0),
            drug_edge_ptr=atom_edge_ptr_t,
            drug_edge_nodes=atom_edge_nodes_t,
            drug_edge_psichic=atom_edge_prior_t,
            prot_edge_index=res_edge_index_t,
            prot_edge_weight=res_edge_weight_t,
            prot_num_nodes=res_feat_sub.size(0),
            prot_edge_ptr=res_edge_ptr_t,
            prot_edge_nodes=res_edge_nodes_t,
            prot_edge_psichic=res_edge_prior_t,
            drug_degree=drug_deg,
            prot_degree=prot_deg,
            prot_esm_missing=prot_esm_missing,
            prot_esm_unreliable=prot_esm_unreliable,
            drug_knn_edge_index=drug_knn_edge_index,
            drug_knn_edge_weight=drug_knn_edge_weight,
            prot_knn_edge_index=prot_knn_edge_index,
            prot_knn_edge_weight=prot_knn_edge_weight,
            return_aux=True,
            explain_cfg=explain_cfg,
        )
        edge_index_filtered = aux.get("edge_index_filtered")
        edge_pos = select_edge_index(edge_index_filtered, drug_idx, prot_idx)
        target_logit = edge_logits[edge_pos]
        target_logit.backward()
        d_repr = aux.get("drug_node_repr")
        p_repr = aux.get("prot_node_repr")
        d_grad = d_repr.grad if d_repr is not None else None
        p_grad = p_repr.grad if p_repr is not None else None
        score_d = grad_score(d_repr, d_grad, mode="grad_x_input")
        score_p = grad_score(p_repr, p_grad, mode="grad_x_input")
        if score_d is not None:
            atom_scores_grad = _map_scores_to_entity(
                score_d, atom_sub, sub_atom_to_drug, drug_idx, ctx["atom_orig_pos"]
            )
        if score_p is not None:
            res_scores_grad = _map_scores_to_entity(
                score_p, res_sub, sub_res_to_prot, prot_idx, ctx["residue_orig_pos"]
            )

    atom_scores_ig = None
    res_scores_ig = None
    if "ig" in methods:
        alpha_atom_fix = aux.get("atom_attention_learned")
        alpha_res_fix = aux.get("residue_attention_learned")
        if alpha_atom_fix is None:
            alpha_atom_fix = aux.get("atom_attention")
        if alpha_res_fix is None:
            alpha_res_fix = aux.get("residue_attention")
        if alpha_atom_fix is not None and alpha_res_fix is not None:
            drug_inv = aux.get("drug_inv")
            prot_inv = aux.get("prot_inv")
            drug_ids = aux.get("drug_ids")
            prot_ids = aux.get("prot_ids")
            d_repr = aux.get("drug_node_repr")
            p_repr = aux.get("prot_node_repr")
            if d_repr is not None and p_repr is not None:
                ig_d, ig_p = _integrated_gradients(
                    model, d_repr, p_repr, alpha_atom_fix, alpha_res_fix,
                    drug_inv, prot_inv, drug_ids, prot_ids, drug_idx, prot_idx,
                    steps=args.explain_ig_steps
                )
                atom_scores_ig = _map_scores_to_entity(
                    ig_d.abs().sum(dim=1), atom_sub, sub_atom_to_drug, drug_idx, ctx["atom_orig_pos"]
                )
                res_scores_ig = _map_scores_to_entity(
                    ig_p.abs().sum(dim=1), res_sub, sub_res_to_prot, prot_idx, ctx["residue_orig_pos"]
                )

    esm_unreliable_flag = None
    if prot_esm_unreliable is not None:
        esm_unreliable_flag = bool(prot_esm_unreliable[prot_idx].item() > 0)

    out = {
        "pair": {
            "drug_id": int(drug_idx),
            "prot_id": int(prot_idx),
            "smiles": drug_id_list[drug_idx] if drug_idx < len(drug_id_list) else None,
            "fasta": prot_id_list[prot_idx] if prot_idx < len(prot_id_list) else None,
        },
        "logit": logit,
        "prob": prob,
        "experts": {
            "gate": gate_w,
            "logitA": float(aux.get("expert_logits")[edge_pos, 0].item()) if aux.get("expert_logits") is not None else None,
            "logitB": float(aux.get("expert_logits")[edge_pos, 1].item()) if aux.get("expert_logits") is not None else None,
            "logitC": float(aux.get("expert_logits")[edge_pos, 2].item()) if aux.get("expert_logits") is not None else None,
        },
        "evidence": {
            # These entropies are normalized by logK in model aux.
            "attn_entropy_atom_norm": float(aux.get("attn_entropy_atom_group")[edge_pos].item())
            if aux.get("attn_entropy_atom_group") is not None
            else None,
            "attn_entropy_res_norm": float(aux.get("attn_entropy_res_group")[edge_pos].item())
            if aux.get("attn_entropy_res_group") is not None
            else None,
            "prior_entropy_atom_norm": float(aux.get("prior_entropy_atom_group")[edge_pos].item())
            if aux.get("prior_entropy_atom_group") is not None
            else None,
            "prior_entropy_res_norm": float(aux.get("prior_entropy_res_group")[edge_pos].item())
            if aux.get("prior_entropy_res_group") is not None
            else None,
            "prior_conf_atom": prior_conf_atom,
            "prior_conf_res": prior_conf_res,
            "mp_alpha_atom_mean": float(aux.get("mp_alpha_atom").mean().item()) if aux.get("mp_alpha_atom") is not None else None,
            "mp_alpha_res_mean": float(aux.get("mp_alpha_res").mean().item()) if aux.get("mp_alpha_res") is not None else None,
            "is_cold_drug": bool(aux.get("is_cold_drug_edge")[edge_pos].item()) if aux.get("is_cold_drug_edge") is not None else None,
            "is_cold_prot": bool(aux.get("is_cold_prot_edge")[edge_pos].item()) if aux.get("is_cold_prot_edge") is not None else None,
            "is_cold_pair": bool(aux.get("is_cold_pair")[edge_pos].item()) if aux.get("is_cold_pair") is not None else None,
            "esm_unreliable": esm_unreliable_flag,
        },
        "uncertainty": {},
    }

    gate = out["experts"]["gate"]
    if gate:
        gate = np.asarray(gate, dtype=np.float32)
        gate_ent = float(-(gate * np.log(gate + 1e-12)).sum())
    else:
        gate_ent = 0.0
    attn_ent = out["evidence"].get("attn_entropy_atom_norm") or 0.0
    prior_ent = out["evidence"].get("prior_entropy_atom_norm") or 0.0
    out["uncertainty"] = {
        "gate_entropy": gate_ent,
        "attn_entropy": float(attn_ent),
        "prior_entropy": float(prior_ent),
        "score": float(0.4 * gate_ent + 0.3 * attn_ent + 0.3 * prior_ent),
    }

    scores = {
        "attn": {"atom": atom_scores_attn, "res": res_scores_attn},
        "grad": {"atom": atom_scores_grad, "res": res_scores_grad},
        "ig": {"atom": atom_scores_ig, "res": res_scores_ig},
    }
    out["scores"] = {}
    for k, v in scores.items():
        if v["atom"] is None and v["res"] is None:
            continue
        out["scores"][k] = {
            "atom_raw": v["atom"].tolist() if v["atom"] is not None else None,
            "res_raw": v["res"].tolist() if v["res"] is not None else None,
            "atom_norm": minmax_norm(v["atom"]).tolist() if v["atom"] is not None else None,
            "res_norm": minmax_norm(v["res"]).tolist() if v["res"] is not None else None,
            "top_atoms": _topk_list(minmax_norm(v["atom"]), topk=args.explain_topk) if v["atom"] is not None else [],
            "top_residues": _topk_list(minmax_norm(v["res"]), topk=args.explain_topk) if v["res"] is not None else [],
        }

    out["experts"]["top_atoms_A"] = _topk_list(minmax_norm(atom_scores_A), topk=args.explain_topk) if atom_scores_A is not None else []
    out["experts"]["top_residues_A"] = _topk_list(minmax_norm(res_scores_A), topk=args.explain_topk) if res_scores_A is not None else []
    out["experts"]["top_atoms_B"] = _topk_list(minmax_norm(atom_scores_B), topk=args.explain_topk) if atom_scores_B is not None else []
    out["experts"]["top_residues_B"] = _topk_list(minmax_norm(res_scores_B), topk=args.explain_topk) if res_scores_B is not None else []

    if aux.get("drug_knn_neighbors") is not None or aux.get("prot_knn_neighbors") is not None:
        out["knn_retrieval"] = {
            "drug": aux.get("drug_knn_neighbors", {}).get(int(drug_idx), {"ids": [], "weights": []}),
            "protein": aux.get("prot_knn_neighbors", {}).get(int(prot_idx), {"ids": [], "weights": []}),
        }

    if "occlusion" in methods and alpha_atom_mix is not None and alpha_res_mix is not None:
        def _forward_with_mask(mask_atom_idx=None, mask_res_idx=None):
            feat_a = atom_feat_sub.clone()
            feat_r = res_feat_sub.clone()
            if mask_atom_idx is not None and len(mask_atom_idx):
                feat_a[mask_atom_idx] = 0.0
            if mask_res_idx is not None and len(mask_res_idx):
                feat_r[mask_res_idx] = 0.0
            with torch.no_grad():
                logits, _ = model(
                    feat_a, feat_r, None, None,
                    drug_node_to_entity=sub_atom_to_drug,
                    protein_node_to_entity=sub_res_to_prot,
                    drug_node_weight=atom_attn_tensor,
                    protein_node_weight=res_attn_tensor,
                    atom_prior=atom_attn_tensor,
                    res_prior=res_attn_tensor,
                    edge_index=edge_index_t,
                    drug_edge_index=atom_edge_index_t,
                    drug_edge_weight=atom_edge_weight_t,
                    drug_num_nodes=feat_a.size(0),
                    drug_edge_ptr=atom_edge_ptr_t,
                    drug_edge_nodes=atom_edge_nodes_t,
                    drug_edge_psichic=atom_edge_prior_t,
                    prot_edge_index=res_edge_index_t,
                    prot_edge_weight=res_edge_weight_t,
                    prot_num_nodes=feat_r.size(0),
                    prot_edge_ptr=res_edge_ptr_t,
                    prot_edge_nodes=res_edge_nodes_t,
                    prot_edge_psichic=res_edge_prior_t,
                    drug_degree=drug_deg,
                    prot_degree=prot_deg,
                    prot_esm_missing=prot_esm_missing,
                    prot_esm_unreliable=prot_esm_unreliable,
                    drug_knn_edge_index=drug_knn_edge_index,
                    drug_knn_edge_weight=drug_knn_edge_weight,
                    prot_knn_edge_index=prot_knn_edge_index,
                    prot_knn_edge_weight=prot_knn_edge_weight,
                )
            if logits.numel() == 0:
                return 0.0
            return float(logits.view(-1)[0].item())

        # select top nodes within this drug/protein group (subgraph indices)
        atom_mask = (sub_atom_to_drug == drug_idx).detach().cpu().numpy()
        res_mask = (sub_res_to_prot == prot_idx).detach().cpu().numpy()
        atom_scores_nodes = alpha_atom_mix.detach().cpu().numpy()[atom_mask]
        res_scores_nodes = alpha_res_mix.detach().cpu().numpy()[res_mask]
        atom_nodes_idx = np.where(atom_mask)[0]
        res_nodes_idx = np.where(res_mask)[0]
        rng = np.random.default_rng(args.explain_seed + 123)
        ms = [1, 2, 5, 10, 20]
        fidelity = {"top": {}, "random": {}}
        base_logit = logit
        for m in ms:
            if atom_nodes_idx.size > 0:
                m_a = min(m, atom_nodes_idx.size)
                top_a = atom_nodes_idx[np.argsort(-atom_scores_nodes)[:m_a]]
                rand_a = rng.choice(atom_nodes_idx, size=m_a, replace=False)
            else:
                top_a = []
                rand_a = []
            if res_nodes_idx.size > 0:
                m_r = min(m, res_nodes_idx.size)
                top_r = res_nodes_idx[np.argsort(-res_scores_nodes)[:m_r]]
                rand_r = rng.choice(res_nodes_idx, size=m_r, replace=False)
            else:
                top_r = []
                rand_r = []
            logit_top = _forward_with_mask(top_a, top_r)
            logit_rand = _forward_with_mask(rand_a, rand_r)
            fidelity["top"][str(m)] = float(base_logit - logit_top)
            fidelity["random"][str(m)] = float(base_logit - logit_rand)
        out["fidelity"] = fidelity

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--node_level", type=str, default="atomic", choices=["atomic"])
    parser.add_argument("--psichic_attention", type=str, default="")
    parser.add_argument("--protein_feat_mode", type=str, default="concat")
    parser.add_argument("--esm_special_tokens", type=str, default="auto")
    parser.add_argument("--esm_norm", type=str, default="per_protein_zscore")
    parser.add_argument("--esm_strict", action="store_true")
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument("--prune_strategy_tag", type=str, default="train_avg_topk")
    parser.add_argument("--no_self_loop", action="store_true")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--no_coldstart_gate", action="store_true")
    parser.add_argument("--alpha_eps", type=float, default=1e-6)
    parser.add_argument("--prior_eps", type=float, default=1e-6)
    parser.add_argument("--alpha_temp", type=float, default=1.3)
    parser.add_argument("--interaction_head", type=str, default="mlp")
    parser.add_argument("--use_hyperedge_head", action="store_true")
    parser.add_argument("--no_alpha_refine", action="store_true")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=32)
    parser.add_argument("--gat_top_k_sparse", type=int, default=4)
    parser.add_argument("--gat_dense_top_k", type=int, default=32)
    parser.add_argument("--prior_mix_mode", type=str, default="mixture")
    parser.add_argument("--prior_mix_lambda", type=float, default=0.3)
    parser.add_argument("--prior_mix_learnable", action="store_true")
    parser.add_argument("--prior_mix_conditional", action="store_true")
    parser.add_argument("--prior_mix_features", type=str, default="deg_drug,deg_prot,prior_entropy,attn_entropy,esm_missing,esm_unreliable")
    parser.add_argument("--prior_smoothing", type=float, default=0.05)
    parser.add_argument("--pool_topk", type=int, default=16)
    parser.add_argument("--pool_randk", type=int, default=16)
    parser.add_argument("--beta_mix", type=float, default=0.7)
    parser.add_argument("--randk_weight_mode", type=str, default="floor_prior")
    parser.add_argument("--prior_floor", type=float, default=1e-4)
    parser.add_argument("--moe_enable", type=int, nargs="?", const=1, default=1)
    parser.add_argument("--expert_A", type=int, nargs="?", const=1, default=1)
    parser.add_argument("--expert_B", type=int, nargs="?", const=1, default=1)
    parser.add_argument("--expert_C", type=int, nargs="?", const=1, default=1)
    parser.add_argument("--use_knn_graph", type=int, nargs="?", const=1, default=1)
    parser.add_argument("--drug_knn_k", type=int, default=20)
    parser.add_argument("--prot_knn_k", type=int, default=20)
    parser.add_argument("--knn_metric", type=str, default="cosine")
    parser.add_argument("--knn_symmetric", action="store_true")
    parser.add_argument("--knn_weight_temp", type=float, default=0.1)
    parser.add_argument("--knn_setting", type=str, default="inductive")
    parser.add_argument("--cold_deg_th_drug", type=int, default=3)
    parser.add_argument("--cold_deg_th_prot", type=int, default=3)
    parser.add_argument("--mp_gate_mode", type=str, default="node")
    parser.add_argument("--mp_gate_deg_only", action="store_true")
    parser.add_argument("--mp_gate_use_attn_entropy", type=int, default=1)
    parser.add_argument("--mp_gate_use_prior_entropy", type=int, default=1)
    parser.add_argument("--mp_gate_use_esm_missing", type=int, default=1)
    parser.add_argument("--bottleneck_drop", type=float, default=0.3)
    parser.add_argument("--use_bottleneck_gate", action="store_true")
    parser.add_argument("--atom_topk", type=int, default=16)
    parser.add_argument("--atom_randk", type=int, default=0)
    parser.add_argument("--res_topk", type=int, default=16)
    parser.add_argument("--res_randk", type=int, default=0)
    parser.add_argument("--randk_seed", type=int, default=2025)
    parser.add_argument("--explain_methods", type=str, default="attn")
    parser.add_argument("--explain_ig_steps", type=int, default=16)
    parser.add_argument("--explain_topk", type=int, default=20)
    parser.add_argument("--explain_aux_to_cpu", action="store_true")
    parser.add_argument("--explain_use_random_walk", action="store_true")
    parser.add_argument("--explain_walk_length", type=int, default=2)
    parser.add_argument("--explain_num_walks", type=int, default=2)
    parser.add_argument("--explain_max_atoms", type=int, default=16384)
    parser.add_argument("--explain_max_res", type=int, default=8192)
    parser.add_argument("--explain_seed", type=int, default=11)
    parser.add_argument("--knn_topk", type=int, default=10)
    parser.add_argument("--drug_id", type=int, default=None)
    parser.add_argument("--prot_id", type=int, default=None)
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument("--fasta", type=str, default=None)
    parser.add_argument("--pdb", type=str, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--export_dir", type=str, default="explain_out")
    parser.add_argument("--export_json", action="store_true")
    parser.add_argument("--export_rdkit_png", action="store_true")
    parser.add_argument("--export_pdb_bfactor", action="store_true")
    parser.add_argument("--export_html_report", action="store_true")
    parser.add_argument("--run_config", type=str, default="")
    args = parser.parse_args()

    cfg = _load_run_config(args.run_config)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    if args.prior_mix_features:
        args.prior_mix_features = [s.strip() for s in args.prior_mix_features.split(",") if s.strip()]
    else:
        args.prior_mix_features = []

    dataset_dir = os.path.join(os.getcwd(), "data", args.dataset)
    drug_idx, prot_idx, drug_id_list, prot_id_list = _resolve_ids(args, dataset_dir)
    ctx = load_context(args)
    t0 = time.time()
    result = explain_pair(args, ctx, drug_idx, prot_idx, drug_id_list, prot_id_list)
    os.makedirs(args.export_dir, exist_ok=True)
    out_json = os.path.join(args.export_dir, f"drug_{drug_idx}_prot_{prot_idx}_explain.json")
    if args.export_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    atom_png = None
    if args.export_rdkit_png and result.get("scores", {}).get("attn", {}).get("atom_norm") is not None:
        atom_png = os.path.join(args.export_dir, f"drug_{drug_idx}_prot_{prot_idx}_atoms.png")
        render_drug_atom_importance(
            result["pair"]["smiles"],
            result["scores"]["attn"]["atom_norm"],
            atom_png,
            topk=args.explain_topk,
        )

    pdb_out = None
    if args.export_pdb_bfactor and args.pdb and result.get("scores", {}).get("attn", {}).get("res_norm") is not None:
        pdb_out = os.path.join(args.export_dir, f"drug_{drug_idx}_prot_{prot_idx}_residues.pdb")
        residues, res_scores = write_residue_scores_to_pdb(
            args.pdb, pdb_out, result["scores"]["attn"]["res_norm"], chain=None
        )
        csv_out = os.path.join(args.export_dir, f"drug_{drug_idx}_prot_{prot_idx}_residues_topk.csv")
        export_topk_residues_csv(csv_out, residues, res_scores, topk=args.explain_topk)

    if args.export_html_report and args.export_json:
        html_out = os.path.join(args.export_dir, f"drug_{drug_idx}_prot_{prot_idx}_report.html")
        export_html_report(out_json, html_out, atom_png=atom_png, pdb_path=pdb_out)

    print(f"[INFO] explain done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
