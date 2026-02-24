"""Training and evaluation pipeline for HGACN, including batching, incidence construction, scheduling, and metrics."""

import os
import time
import math
import random
import argparse
import gc
import json
import pickle
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import sys
import warnings
from scipy import sparse

from data_preprocess import (
    load_and_construct_hypergraphs,
    load_psichic_attention,
    get_pair_attention,
    align_scores_to_length,
)
from model import HGACN
from loss import CombinedLoss
from graphsaint_sampler import random_walk_subgraph, build_subgraph_edges

warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed DataLoader workers deterministically."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_torch_graph(adj, device):
    """Convert dense or scipy sparse adjacency to torch tensor graph."""
    if isinstance(adj, sparse.spmatrix):
        coo = adj.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
        values = torch.tensor(coo.data, dtype=torch.float32, device=device)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()
    return torch.tensor(adj, dtype=torch.float32, device=device)


def build_group_index(ids, num_groups):
    """Build list-based index for each group id."""
    ids = np.asarray(ids, dtype=np.int64)
    order = np.argsort(ids)
    ids_sorted = ids[order]
    uniq, start, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    groups = [np.empty(0, dtype=np.int64) for _ in range(num_groups)]
    for u, s, c in zip(uniq, start, counts):
        groups[int(u)] = order[s:s + c]
    return groups


def build_group_dict(ids):
    """Build dict-based index mapping group id to sample indices."""
    ids = np.asarray(ids, dtype=np.int64)
    order = np.argsort(ids)
    ids_sorted = ids[order]
    uniq, start, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    return {int(u): order[s:s + c] for u, s, c in zip(uniq, start, counts)}


def build_id_list(mapping):
    """Invert an id->index mapping into index->id list."""
    max_idx = max(mapping.values()) if mapping else -1
    id_list = [None] * (max_idx + 1)
    for k, v in mapping.items():
        if 0 <= v < len(id_list):
            id_list[v] = k
    return id_list


def _get_nodes(mapping, idx):
    """Fetch node list from dict/list mapping with bounds checks."""
    if mapping is None:
        return None
    if hasattr(mapping, "get"):
        return mapping.get(int(idx))
    if 0 <= int(idx) < len(mapping):
        return mapping[int(idx)]
    return None


def build_pair_psichic_ragged(
    edge_pairs,
    drug_nodes_map,
    prot_nodes_map,
    atom_orig_pos,
    residue_orig_pos,
    drug_id_list,
    prot_id_list,
    attn_db,
    device,
):
    """Build ragged psichic-prior tensors for edge pairs."""
    if attn_db is None or drug_id_list is None or prot_id_list is None:
        return None, None, None, None, None, None
    edge_ptr_d = [0]
    edge_nodes_d = []
    edge_ps_d = []
    edge_ptr_p = [0]
    edge_nodes_p = []
    edge_ps_p = []
    cache = {}
    for d, p in edge_pairs:
        d = int(d)
        p = int(p)
        drug_nodes = _get_nodes(drug_nodes_map, d)
        prot_nodes = _get_nodes(prot_nodes_map, p)
        key = (p, d)
        if key in cache:
            atom_scores_raw, res_scores_raw = cache[key]
        else:
            prot_seq = prot_id_list[p] if p < len(prot_id_list) else None
            drug_smiles = drug_id_list[d] if d < len(drug_id_list) else None
            prot_seq = str(prot_seq).strip() if prot_seq is not None else None
            drug_smiles = str(drug_smiles).strip() if drug_smiles is not None else None
            atom_scores_raw, res_scores_raw = get_pair_attention(attn_db, prot_seq, drug_smiles)
            cache[key] = (atom_scores_raw, res_scores_raw)

        if drug_nodes is None or len(drug_nodes) == 0:
            edge_ptr_d.append(len(edge_nodes_d))
        else:
            orig = atom_orig_pos[drug_nodes]
            target_len = int(orig.max()) + 1 if orig.size > 0 else 0
            if target_len > 0 and cache is not None:
                aligned_key = (p, d, 1, target_len)
                atom_scores_full = cache.get(aligned_key)
            else:
                aligned_key = None
                atom_scores_full = None
            if atom_scores_full is None:
                atom_scores_full = align_scores_to_length(atom_scores_raw, target_len, fill=1.0)
                if aligned_key is not None:
                    cache[aligned_key] = atom_scores_full
            ps = atom_scores_full[orig] if target_len > 0 else np.asarray([], dtype=np.float32)
            edge_nodes_d.extend(drug_nodes.tolist())
            edge_ps_d.extend(ps.tolist())
            edge_ptr_d.append(len(edge_nodes_d))

        if prot_nodes is None or len(prot_nodes) == 0:
            edge_ptr_p.append(len(edge_nodes_p))
        else:
            orig = residue_orig_pos[prot_nodes]
            target_len = int(orig.max()) + 1 if orig.size > 0 else 0
            if target_len > 0 and cache is not None:
                aligned_key = (p, d, 0, target_len)
                res_scores_full = cache.get(aligned_key)
            else:
                aligned_key = None
                res_scores_full = None
            if res_scores_full is None:
                res_scores_full = align_scores_to_length(res_scores_raw, target_len, fill=1.0)
                if aligned_key is not None:
                    cache[aligned_key] = res_scores_full
            ps = res_scores_full[orig] if target_len > 0 else np.asarray([], dtype=np.float32)
            edge_nodes_p.extend(prot_nodes.tolist())
            edge_ps_p.extend(ps.tolist())
            edge_ptr_p.append(len(edge_nodes_p))

    edge_ptr_d_t = torch.tensor(edge_ptr_d, dtype=torch.long, device=device)
    edge_nodes_d_t = torch.tensor(edge_nodes_d, dtype=torch.long, device=device)
    edge_ps_d_t = torch.tensor(edge_ps_d, dtype=torch.float32, device=device)
    edge_ptr_p_t = torch.tensor(edge_ptr_p, dtype=torch.long, device=device)
    edge_nodes_p_t = torch.tensor(edge_nodes_p, dtype=torch.long, device=device)
    edge_ps_p_t = torch.tensor(edge_ps_p, dtype=torch.float32, device=device)
    return edge_ptr_d_t, edge_nodes_d_t, edge_ps_d_t, edge_ptr_p_t, edge_nodes_p_t, edge_ps_p_t


def ragged_slice(ptr, nodes, idx):
    """Slice one ragged segment using ptr representation."""
    if ptr is None or nodes is None:
        return np.empty((0,), dtype=np.int64)
    idx = int(idx)
    if idx < 0 or idx + 1 >= ptr.shape[0]:
        return np.empty((0,), dtype=np.int64)
    start = int(ptr[idx])
    end = int(ptr[idx + 1])
    if end <= start:
        return np.empty((0,), dtype=np.int64)
    return nodes[start:end]


def union_ragged(ptr, nodes, ids):
    """Union multiple ragged segments into unique sorted ids."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    pieces = []
    for idx in ids:
        arr = ragged_slice(ptr, nodes, int(idx))
        if arr.size:
            pieces.append(arr)
    if not pieces:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(pieces))


def edge_index_to_numpy(edge_index):
    """Convert edge-index tensor/array to numpy shape [2, E]."""
    if edge_index is None:
        return None
    if torch.is_tensor(edge_index):
        arr = edge_index.detach().cpu().numpy()
    else:
        arr = np.asarray(edge_index)
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 2:
        return arr.astype(np.int64, copy=False)
    if arr.shape[1] == 2:
        return arr.T.astype(np.int64, copy=False)
    return None


def build_knn_neighbor_cache(edge_index, edge_weight=None):
    """Build source->neighbors cache sorted by descending edge weight."""
    edge_np = edge_index_to_numpy(edge_index)
    if edge_np is None or edge_np.size == 0:
        return {}
    src = edge_np[0].astype(np.int64, copy=False)
    dst = edge_np[1].astype(np.int64, copy=False)
    if edge_weight is None:
        w = np.ones(src.shape[0], dtype=np.float32)
    else:
        if torch.is_tensor(edge_weight):
            w = edge_weight.detach().cpu().numpy()
        else:
            w = np.asarray(edge_weight)
        w = w.reshape(-1).astype(np.float32, copy=False)
        if w.shape[0] != src.shape[0]:
            w = np.ones(src.shape[0], dtype=np.float32)

    order = np.argsort(src, kind="mergesort")
    src_sorted = src[order]
    dst_sorted = dst[order]
    w_sorted = w[order]
    uniq, start, counts = np.unique(src_sorted, return_index=True, return_counts=True)
    cache = {}
    for u, s, c in zip(uniq, start, counts):
        neigh = dst_sorted[s:s + c]
        ww = w_sorted[s:s + c]
        if neigh.size == 0:
            continue
        rank = np.lexsort((neigh, -ww.astype(np.float64)))
        neigh = neigh[rank]
        keep = []
        seen = set()
        u_int = int(u)
        for v in neigh.tolist():
            v_int = int(v)
            if v_int == u_int or v_int in seen:
                continue
            seen.add(v_int)
            keep.append(v_int)
        if keep:
            cache[u_int] = np.asarray(keep, dtype=np.int64)
    return cache


def expand_seed_proteins_with_warm_support(
    seed_prots,
    neighbor_cache,
    prot_degree_np=None,
    support_k=0,
    max_add=0,
):
    """
    Add warm support proteins for each seed protein using kNN neighbors.

    Warm is defined by global train degree > 0. Returns (expanded_seed_prots, num_added).
    """
    seed = np.unique(np.asarray(seed_prots, dtype=np.int64).reshape(-1))
    support_k = int(support_k)
    max_add = int(max_add)
    if support_k <= 0 or seed.size == 0 or not neighbor_cache:
        return seed, 0

    degree_arr = None
    if prot_degree_np is not None:
        degree_arr = np.asarray(prot_degree_np).reshape(-1)

    added = []
    for pid in seed.tolist():
        neigh = neighbor_cache.get(int(pid))
        if neigh is None or neigh.size == 0:
            continue
        neigh_sel = neigh
        if degree_arr is not None and degree_arr.size > 0:
            valid = (neigh_sel >= 0) & (neigh_sel < degree_arr.size)
            if not np.any(valid):
                continue
            neigh_sel = neigh_sel[valid]
            warm = degree_arr[neigh_sel] > 0
            if not np.any(warm):
                continue
            neigh_sel = neigh_sel[warm]
        if neigh_sel.size == 0:
            continue
        neigh_sel = neigh_sel[:support_k]
        if neigh_sel.size:
            added.append(neigh_sel)

    if not added:
        return seed, 0
    add_ids = np.unique(np.concatenate(added))
    add_ids = add_ids[~np.isin(add_ids, seed)]
    if add_ids.size == 0:
        return seed, 0
    if max_add > 0 and add_ids.size > max_add:
        add_ids = add_ids[:max_add]
    out = np.unique(np.concatenate([seed, add_ids]))
    return out, int(add_ids.size)


def build_edge_incidence(
    edge_pairs,
    sub_nodes,
    entity_ptr,
    entity_nodes,
    orig_pos,
    attn_db,
    drug_id_list,
    prot_id_list,
    fallback_weight,
    is_atom_side=True,
    alpha_eps=1e-6,
    cache=None,
    debug_step=None,
):
    """Build subgraph edge incidence and prior weights for one side."""
    edge_ptr = [0]
    edge_nodes = []
    edge_prior = []
    counts = np.zeros(edge_pairs.shape[0], dtype=np.int64)
    sub_nodes = np.asarray(sub_nodes, dtype=np.int64)
    entity_cache = {}
    for i, (drug_idx, prot_idx) in enumerate(edge_pairs):
        drug_idx = int(drug_idx)
        prot_idx = int(prot_idx)
        entity_idx = drug_idx if is_atom_side else prot_idx
        cached = entity_cache.get(entity_idx)
        if cached is None:
            kept_nodes = ragged_slice(entity_ptr, entity_nodes, entity_idx)
            if kept_nodes.size == 0:
                entity_cache[entity_idx] = (kept_nodes, None, None, None)
            else:
                orig_idx = orig_pos[kept_nodes]
                if orig_idx.size > 0:
                    valid_mask = orig_idx >= 0
                    if not valid_mask.all():
                        kept_nodes = kept_nodes[valid_mask]
                        orig_idx = orig_idx[valid_mask]
                pos = np.searchsorted(sub_nodes, kept_nodes)
                mask = pos < sub_nodes.size
                if np.any(mask):
                    mask[mask] = sub_nodes[pos[mask]] == kept_nodes[mask]
                sub_idx = pos[mask] if np.any(mask) else None
                entity_cache[entity_idx] = (kept_nodes, orig_idx, mask, sub_idx)
        kept_nodes, orig_idx, mask, sub_idx = entity_cache[entity_idx]
        if kept_nodes.size == 0:
            edge_ptr.append(len(edge_nodes))
            continue
        if mask is None or not np.any(mask):
            edge_ptr.append(len(edge_nodes))
            continue
        prior = None
        if (
            attn_db
            and drug_id_list is not None
            and prot_id_list is not None
            and drug_idx < len(drug_id_list)
            and prot_idx < len(prot_id_list)
        ):
            key = (prot_idx, drug_idx)
            if cache is not None and key in cache:
                atom_scores_raw, res_scores_raw = cache[key]
            else:
                prot_seq = prot_id_list[prot_idx]
                drug_smiles = drug_id_list[drug_idx]
                if prot_seq is None or drug_smiles is None:
                    atom_scores_raw, res_scores_raw = None, None
                else:
                    prot_seq = str(prot_seq).strip()
                    drug_smiles = str(drug_smiles).strip()
                    if debug_step == 0 and cache is not None and not cache.get("_debug_printed", False):
                        print(f"CRITICAL DEBUG: Running Key -> ('{prot_seq[:20]}...', '{drug_smiles[:20]}...')")
                        print(f"CRITICAL DEBUG: Is in DB? -> {(prot_seq, drug_smiles) in attn_db}")
                        cache["_debug_printed"] = True
                    atom_scores_raw, res_scores_raw = get_pair_attention(attn_db, prot_seq, drug_smiles)
                if cache is not None:
                    cache[key] = (atom_scores_raw, res_scores_raw)
            scores_raw = atom_scores_raw if is_atom_side else res_scores_raw
            if scores_raw is not None and orig_idx.size > 0:
                max_idx = int(orig_idx.max())
                target_len = max_idx + 1
                aligned_key = None
                scores_full = None
                if cache is not None:
                    aligned_key = (prot_idx, drug_idx, 1 if is_atom_side else 0, target_len)
                    scores_full = cache.get(aligned_key)
                if scores_full is None:
                    scores_full = align_scores_to_length(scores_raw, target_len, fill=1.0)
                    if aligned_key is not None:
                        cache[aligned_key] = scores_full
                if scores_full.size >= max_idx + 1:
                    prior = scores_full[orig_idx]
        if prior is None or prior.size != kept_nodes.size:
            if fallback_weight is not None and kept_nodes.size > 0:
                prior = fallback_weight[kept_nodes]
            else:
                prior = np.ones((kept_nodes.size,), dtype=np.float32)
        prior_sub = np.asarray(prior, dtype=np.float32)[mask]
        prior_sub = np.clip(prior_sub, alpha_eps, None)
        if not is_atom_side:
            # Sharpen protein-side prior to amplify differences (no extra complexity).
            prior_sub = prior_sub ** 6
        denom = float(prior_sub.sum())
        if not np.isfinite(denom) or denom <= 0:
            prior_sub = np.full_like(prior_sub, 1.0 / max(prior_sub.size, 1))
        else:
            prior_sub = prior_sub / denom
        edge_nodes.extend(sub_idx.tolist())
        edge_prior.extend(prior_sub.tolist())
        counts[i] = prior_sub.size
        edge_ptr.append(len(edge_nodes))
    return (
        np.asarray(edge_ptr, dtype=np.int64),
        np.asarray(edge_nodes, dtype=np.int64),
        np.asarray(edge_prior, dtype=np.float32),
        counts,
    )


def filter_ragged_by_mask(edge_ptr, edge_nodes, edge_prior, edge_mask):
    """Filter ragged edge tensors with edge-level boolean mask."""
    edge_ptr = np.asarray(edge_ptr, dtype=np.int64)
    edge_nodes = np.asarray(edge_nodes, dtype=np.int64)
    edge_prior = np.asarray(edge_prior, dtype=np.float32)
    edge_mask = np.asarray(edge_mask, dtype=bool)
    new_ptr = [0]
    new_nodes = []
    new_prior = []
    for i, keep in enumerate(edge_mask):
        if not keep:
            continue
        start = int(edge_ptr[i])
        end = int(edge_ptr[i + 1])
        if end > start:
            new_nodes.extend(edge_nodes[start:end].tolist())
            new_prior.extend(edge_prior[start:end].tolist())
        new_ptr.append(len(new_nodes))
    return (
        np.asarray(new_ptr, dtype=np.int64),
        np.asarray(new_nodes, dtype=np.int64),
        np.asarray(new_prior, dtype=np.float32),
    )


def apply_episodic_protein_edge_drop(
    res_edge_index_sub,
    res_edge_weight_sub,
    sub_res_to_prot,
    seed_prots,
    drop_prob,
    rng,
):
    """
    Episodic cold-start drill:
    keep nodes/labels untouched, only remove MP edges for sampled proteins in this step.
    """
    stats = {
        "batch_prot": 0,
        "dropped_prot": 0,
        "dropped_present": 0,
        "dropped_zero_after": 0,
    }
    drop_prob = float(drop_prob)
    if drop_prob <= 0:
        return res_edge_index_sub, res_edge_weight_sub, None, stats
    drop_prob = min(max(drop_prob, 0.0), 1.0)
    seed_prots_np = np.asarray(seed_prots, dtype=np.int64).reshape(-1)
    stats["batch_prot"] = int(seed_prots_np.size)
    if seed_prots_np.size == 0:
        return res_edge_index_sub, res_edge_weight_sub, None, stats
    drop_mask = rng.random(seed_prots_np.size) < drop_prob
    dropped_prots_np = seed_prots_np[drop_mask]
    stats["dropped_prot"] = int(dropped_prots_np.size)
    if dropped_prots_np.size == 0:
        return res_edge_index_sub, res_edge_weight_sub, None, stats

    if torch.is_tensor(sub_res_to_prot):
        sub_res_to_prot_np = sub_res_to_prot.detach().cpu().numpy().astype(np.int64, copy=False)
        drop_prots_t = torch.from_numpy(dropped_prots_np).to(device=sub_res_to_prot.device, dtype=torch.long)
    else:
        sub_res_to_prot_np = np.asarray(sub_res_to_prot, dtype=np.int64).reshape(-1)
        drop_prots_t = torch.from_numpy(dropped_prots_np).to(dtype=torch.long)

    num_nodes = int(sub_res_to_prot_np.shape[0])
    drop_node_mask = np.isin(sub_res_to_prot_np, dropped_prots_np)

    deg_before = np.zeros((num_nodes,), dtype=np.int64)
    if isinstance(res_edge_index_sub, np.ndarray) and res_edge_index_sub.size > 0:
        src = res_edge_index_sub[0]
        dst = res_edge_index_sub[1]
        np.add.at(deg_before, src, 1)
        np.add.at(deg_before, dst, 1)
        keep_edge = ~(drop_node_mask[src] | drop_node_mask[dst])
        res_edge_index_sub = res_edge_index_sub[:, keep_edge]
        if res_edge_weight_sub is not None:
            res_edge_weight_sub = res_edge_weight_sub[keep_edge]

    deg_after = np.zeros((num_nodes,), dtype=np.int64)
    if isinstance(res_edge_index_sub, np.ndarray) and res_edge_index_sub.size > 0:
        src = res_edge_index_sub[0]
        dst = res_edge_index_sub[1]
        np.add.at(deg_after, src, 1)
        np.add.at(deg_after, dst, 1)

    dropped_present = 0
    dropped_zero_after = 0
    for pid in dropped_prots_np.tolist():
        idx = np.flatnonzero(sub_res_to_prot_np == pid)
        if idx.size == 0:
            continue
        dropped_present += 1
        if int(deg_after[idx].sum()) == 0:
            dropped_zero_after += 1
    stats["dropped_present"] = int(dropped_present)
    stats["dropped_zero_after"] = int(dropped_zero_after)
    return res_edge_index_sub, res_edge_weight_sub, drop_prots_t, stats


def unpack_vae_params(vae_params):
    """Unpack VAE parameter tuple with robust defaults."""
    if vae_params is None:
        return None, None, None, None, None, None, None, None
    if len(vae_params) == 8:
        return vae_params
    if len(vae_params) == 5:
        drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl = vae_params
        return drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl, attn_kl, None, None
    if len(vae_params) == 7:
        drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl, ent_atom, ent_res = vae_params
        return drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl, attn_kl, ent_atom, ent_res
    if len(vae_params) == 4:
        drug_mu, drug_logvar, prot_mu, prot_logvar = vae_params
        attn_kl = drug_mu.new_tensor(0.0)
        return drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl, attn_kl, None, None
    raise ValueError("Unexpected vae_params length")


def update_ema(ema_state, model, decay):
    """Update exponential moving average weights."""
    if ema_state is None:
        return
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if not torch.is_floating_point(v):
                continue
            if k not in ema_state:
                ema_state[k] = v.detach().clone()
            else:
                ema_state[k].mul_(decay).add_(v, alpha=1.0 - decay)


def swap_ema_weights(model, ema_state):
    """Swap model weights with EMA snapshot for evaluation."""
    if ema_state is None:
        return None
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_state, strict=False)
    return backup


def restore_weights(model, backup_state):
    """Restore model weights after temporary EMA swap."""
    if backup_state is None:
        return
    model.load_state_dict(backup_state, strict=False)


def update_dynamic_params(epoch, args, optimizer, model=None, criterion=None, state=None):
    """Update temperature and regularization schedules by epoch."""
    if state is None:
        state = {}

    # --- 基础参数初始化 (只执行一次) ---
    if "base_lr" not in state:
        state["base_lr"] = float(getattr(args, "lr", 1e-4))
    if "base_alpha_temp" not in state:
        state["base_alpha_temp"] = float(getattr(args, "alpha_temp", 1.0))
    if "base_gate_balance_weight" not in state:
        state["base_gate_balance_weight"] = float(getattr(args, "gate_balance_weight", 0.1))
    if "base_info_nce_weight" not in state:
        state["base_info_nce_weight"] = float(getattr(args, "info_nce_weight", 0.0))
    if "base_prior_smoothing" not in state:
        state["base_prior_smoothing"] = float(getattr(args, "prior_smoothing", 0.0))
    if "base_ema_decay" not in state:
        state["base_ema_decay"] = float(getattr(args, "ema_decay", 0.0))
    if "base_attn_kl_weight" not in state:
        state["base_attn_kl_weight"] = float(getattr(args, "attn_kl_weight", 0.0))
    if "base_attn_kl_w_min" not in state:
        state["base_attn_kl_w_min"] = float(getattr(args, "attn_kl_w_min", 0.0))
    if "base_attn_kl_w_max" not in state:
        state["base_attn_kl_w_max"] = float(getattr(args, "attn_kl_w_max", 0.0))
    if "base_weight_decay" not in state:
        state["base_weight_decay"] = [pg.get("weight_decay", 0.0) for pg in optimizer.param_groups]
    if "cli_flags" not in state:
        cli_flags = set()
        for token in sys.argv[1:]:
            if not token.startswith("--"):
                continue
            cli_flags.add(token.split("=", 1)[0])
        state["cli_flags"] = cli_flags
    # Finetune-ready: keep explicit CLI LR authoritative even after resume.
    if "--lr" in state.get("cli_flags", set()):
        state["base_lr"] = float(getattr(args, "lr", state.get("base_lr", 1e-4)))

    def _apply_weight_decay(wd_value):
        if isinstance(wd_value, (list, tuple, np.ndarray)):
            for pg, wd in zip(optimizer.param_groups, wd_value):
                pg["weight_decay"] = float(wd)
        else:
            for pg in optimizer.param_groups:
                pg["weight_decay"] = float(wd_value)

    def _cli_override(param, fallback):
        flag = f"--{param}"
        if flag in state["cli_flags"]:
            return state.get(f"base_{param}", fallback)
        return fallback

    # --- 阶段定义 ---
    # Phase 1: Warm-up & Exploration (0 - 40)
    # 允许模型在较高的温度下探索，建立初步的 Gate 分配
    if epoch <= 40:
        stage = 1
        current_lr = state["base_lr"]
        # 温度较高，允许注意力分散
        alpha_temp = max(state["base_alpha_temp"], 0.8)
        # 低正则，让 Expert C 有机会启动
        gate_balance_weight = 0.1
        info_nce_weight = 0.1
        expertC_scale = min(1.0, epoch / 20.0)  # 缓慢激活 Expert C

    # Phase 2: Annealing & Shaping (41 - 100)
    # 线性降温，强制模型开始聚焦；增加对比学习权重，拉开特征距离
    elif epoch <= 100:
        stage = 2
        ratio = (epoch - 40) / 60.0  # 0.0 -> 1.0

        # LR: 保持平稳或微降
        current_lr = state["base_lr"] * (1.0 - 0.3 * ratio)

        # Temp: 0.8 -> 0.4 (开始聚焦)
        alpha_temp = 0.8 - (0.8 - 0.4) * ratio

        # Gate: 0.1 -> 1.5 (开始反垄断)
        gate_balance_weight = 0.1 + (1.5 - 0.1) * ratio

        # InfoNCE: 0.1 -> 0.4 (增强判别力)
        info_nce_weight = 0.1 + (0.4 - 0.1) * ratio
        expertC_scale = 1.0

    # Phase 3: Crystallization & SOTA Push (101 - 160+)
    # 模仿 GPCR 最后的成功配置：极低温、高正则、微小 LR
    else:
        stage = 3
        # LR: 大幅降低，精细雕刻
        current_lr = state["base_lr"] * 0.2

        # Temp: 锁定在 0.25 - 0.30 (鹰眼模式)
        alpha_temp = 0.30

        # Gate: 2.5 (强力打压中间层独裁)
        gate_balance_weight = 2.5

        # InfoNCE: 0.5 (最大化流形距离)
        info_nce_weight = 0.5
        expertC_scale = 1.0

    # --- CLI overrides (与旧逻辑一致：显式 CLI 优先) ---
    current_lr = _cli_override("lr", current_lr)
    alpha_temp = _cli_override("alpha_temp", alpha_temp)
    gate_balance_weight = _cli_override("gate_balance_weight", gate_balance_weight)
    info_nce_weight = _cli_override("info_nce_weight", info_nce_weight)
    prior_smoothing = _cli_override("prior_smoothing", state.get("base_prior_smoothing", 0.0))
    ema_decay = _cli_override("ema_decay", state.get("base_ema_decay", 0.0))
    attn_kl_weight = _cli_override("attn_kl_weight", state.get("base_attn_kl_weight", 0.0))
    attn_kl_w_min = _cli_override("attn_kl_w_min", state.get("base_attn_kl_w_min", 0.0))
    attn_kl_w_max = _cli_override("attn_kl_w_max", state.get("base_attn_kl_w_max", 0.0))
    weight_decay_target = _cli_override("weight_decay", state.get("base_weight_decay", 0.0))

    # Respect explicit InfoNCE disable switches.
    if bool(getattr(args, "disable_info_nce", False)) or not bool(getattr(args, "info_nce_enable", 1)):
        info_nce_weight = 0.0

    # --- 应用参数 ---
    # 1. 更新 Optimizer LR
    for pg in optimizer.param_groups:
        pg["lr"] = current_lr
    _apply_weight_decay(weight_decay_target)
    if optimizer.param_groups:
        args.weight_decay = float(optimizer.param_groups[0].get("weight_decay", args.weight_decay))

    # 2. 更新 Args (用于日志打印)
    args.lr = current_lr
    args.alpha_temp = alpha_temp
    args.gate_balance_weight = gate_balance_weight
    args.info_nce_weight = info_nce_weight
    args.prior_smoothing = float(prior_smoothing)
    args.ema_decay = float(ema_decay)
    args.attn_kl_weight = float(attn_kl_weight)
    args.attn_kl_w_min = float(attn_kl_w_min)
    args.attn_kl_w_max = float(attn_kl_w_max)

    # 3. 更新 Model 内部状态
    if model is not None:
        if hasattr(model, "alpha_temp"):
            model.alpha_temp = alpha_temp
        if hasattr(model, "prior_smoothing"):
            model.prior_smoothing = float(prior_smoothing)
        # 你的 model.py 里可能有 _last_expertC_scale
        if hasattr(model, "_last_expertC_scale"):
            model._last_expertC_scale = expertC_scale

    # 4. 更新 Loss 内部状态
    if criterion is not None:
        criterion.info_nce_weight = info_nce_weight
        criterion.gate_balance_weight = gate_balance_weight
        if hasattr(criterion, "attn_kl_weight"):
            criterion.attn_kl_weight = float(attn_kl_weight)
        if hasattr(criterion, "attn_kl_w_min"):
            criterion.attn_kl_w_min = float(attn_kl_w_min)
        if hasattr(criterion, "attn_kl_w_max"):
            criterion.attn_kl_w_max = float(attn_kl_w_max)

    # --- 打印阶段信息 (防止你不知道现在是什么策略) ---
    if epoch % 5 == 0:
        print(f"\n[AUTO-SCHEDULER] Stage {stage}: LR={current_lr:.2e}, "
              f"Temp={alpha_temp:.3f}, GateBal={gate_balance_weight:.2f}, "
              f"InfoNCE={info_nce_weight:.2f}\n")

    state["stage"] = stage
    return {"stage": stage, "expertC_scale": expertC_scale}


def append_diagnostics(path, record):
    """Append one diagnostic record to a JSONL file."""
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write diagnostics: {e}")


def should_block_atomic_cv(eval_mode, node_level, allow_atomic_cv_leakage=False):
    """Guard against leakage-prone atomic-level cross-validation modes."""
    return (
        str(eval_mode).lower() == "cv"
        and str(node_level).lower() == "atomic"
        and not bool(allow_atomic_cv_leakage)
    )


def write_metrics_out(path, payload):
    """Write metrics payload to JSON file."""
    if not path:
        return
    try:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        print(f"[INFO] Wrote metrics to {path}")
    except Exception as e:
        print(f"[WARN] Failed to write metrics_out={path}: {e}")


def accumulate_gate_bins(sum_arr, cnt_arr, gate_values, deg_values):
    """Accumulate gate statistics into degree bins."""
    if gate_values is None or deg_values is None:
        return
    if gate_values.numel() == 0 or deg_values.numel() == 0:
        return
    deg = deg_values.view(-1)
    gate = gate_values.view(-1)
    if deg.numel() != gate.numel():
        return
    bins = torch.zeros_like(deg, dtype=torch.long)
    bins[(deg >= 1) & (deg <= 2)] = 1
    bins[(deg >= 3) & (deg <= 5)] = 2
    bins[(deg >= 6) & (deg <= 10)] = 3
    bins[deg > 10] = 4
    for b in range(5):
        mask = bins == b
        if mask.any():
            sum_arr[b] += gate[mask].sum()
            cnt_arr[b] += mask.sum()


def _append_samples(store, values, max_samples=4096, rng=None):
    """Append sampled values with optional cap and random replacement."""
    if values is None or max_samples <= 0:
        return
    if torch.is_tensor(values):
        arr = values.detach().view(-1).float().cpu().numpy()
    else:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return
    if arr.size > max_samples:
        if rng is None:
            idx = np.random.choice(arr.size, size=max_samples, replace=False)
        else:
            idx = rng.choice(arr.size, size=max_samples, replace=False)
        arr = arr[idx]
    remaining = max_samples - len(store)
    if remaining <= 0:
        return
    if arr.size > remaining:
        arr = arr[:remaining]
    store.extend(arr.tolist())


def _percentiles(values, qs=(0, 10, 50, 90, 100)):
    """Compute requested percentiles with NaN-safe fallback."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return [0.0 for _ in qs]
    return [float(x) for x in np.percentile(arr, qs).tolist()]


def _safe_auc_aupr(labels_np, preds_np):
    """Compute AUC/AUPR safely with exception fallback."""
    if labels_np.size < 2:
        return 0.0, 0.0
    if np.unique(labels_np).size < 2:
        return 0.0, 0.0
    return roc_auc_score(labels_np, preds_np), average_precision_score(labels_np, preds_np)


def _per_protein_mean_center(scores_np, prot_ids_np):
    """Apply per-protein zero-mean centering on scores."""
    scores = np.asarray(scores_np, dtype=np.float32).reshape(-1)
    prot_ids = np.asarray(prot_ids_np, dtype=np.int64).reshape(-1)
    if scores.size == 0 or prot_ids.size == 0 or scores.size != prot_ids.size:
        return scores
    _, inv = np.unique(prot_ids, return_inverse=True)
    num_groups = int(inv.max()) + 1 if inv.size else 0
    if num_groups <= 0:
        return scores
    sums = np.bincount(inv, weights=scores.astype(np.float64), minlength=num_groups)
    counts = np.bincount(inv, minlength=num_groups).astype(np.float64)
    means = sums / np.maximum(counts, 1.0)
    return (scores.astype(np.float64) - means[inv]).astype(np.float32)


def _select_eval_scores(scores_np, prot_ids_np, eval_score_centering="none"):
    """Select score transform for global ranking metrics."""
    mode = str(eval_score_centering or "none").strip().lower()
    scores = np.asarray(scores_np, dtype=np.float32).reshape(-1)
    if mode in ("none", "raw"):
        return scores, "none"
    if mode in ("per_protein_mean", "protein_mean", "center"):
        return _per_protein_mean_center(scores, prot_ids_np), "per_protein_mean"
    raise ValueError(f"Unknown eval_score_centering: {eval_score_centering}")


def _compute_global_metric_variants(labels_np, scores_np, prot_ids_np):
    """Compute both raw and per-protein centered global AUC/AUPR."""
    labels = np.asarray(labels_np).reshape(-1)
    scores = np.asarray(scores_np, dtype=np.float32).reshape(-1)
    prot_ids = np.asarray(prot_ids_np, dtype=np.int64).reshape(-1)
    raw_auc, raw_aupr = _safe_auc_aupr(labels, scores)
    centered_scores = _per_protein_mean_center(scores, prot_ids)
    centered_auc, centered_aupr = _safe_auc_aupr(labels, centered_scores)
    return {
        "raw_auc": float(raw_auc),
        "raw_aupr": float(raw_aupr),
        "centered_auc": float(centered_auc),
        "centered_aupr": float(centered_aupr),
    }


def _attach_global_metrics(stats, metric_variants, selected_mode, selected_auc, selected_aupr):
    """Attach raw/centered global metrics into stats dict."""
    if not isinstance(stats, dict):
        stats = {}
    stats["global_metrics"] = {
        "raw_auc": float(metric_variants.get("raw_auc", 0.0)),
        "raw_aupr": float(metric_variants.get("raw_aupr", 0.0)),
        "centered_auc": float(metric_variants.get("centered_auc", 0.0)),
        "centered_aupr": float(metric_variants.get("centered_aupr", 0.0)),
        "selected_mode": str(selected_mode),
        "selected_auc": float(selected_auc),
        "selected_aupr": float(selected_aupr),
    }
    return stats


def _print_global_metric_variants(stats, prefix="EVAL"):
    """Print raw/centered global metrics if available in stats."""
    if not isinstance(stats, dict):
        return
    gm = stats.get("global_metrics", None)
    if not isinstance(gm, dict):
        return
    raw_auc = float(gm.get("raw_auc", 0.0))
    raw_aupr = float(gm.get("raw_aupr", 0.0))
    cen_auc = float(gm.get("centered_auc", 0.0))
    cen_aupr = float(gm.get("centered_aupr", 0.0))
    sel_mode = str(gm.get("selected_mode", "none"))
    sel_auc = float(gm.get("selected_auc", 0.0))
    sel_aupr = float(gm.get("selected_aupr", 0.0))
    print(
        f"[{prefix}-GLOBAL] raw AUC/AUPR={raw_auc:.5f}/{raw_aupr:.5f}; "
        f"centered AUC/AUPR={cen_auc:.5f}/{cen_aupr:.5f}; "
        f"selected({sel_mode})={sel_auc:.5f}/{sel_aupr:.5f}"
    )


def compute_cold_metrics(edges_np, labels_np, preds_np,
                         drug_degree=None, prot_degree=None,
                         cold_deg_th_drug=2, cold_deg_th_prot=2):
    """Compute metrics on cold-start subsets."""
    if drug_degree is None or prot_degree is None:
        return {}
    if torch.is_tensor(drug_degree):
        drug_deg = drug_degree.detach().cpu().numpy()
    else:
        drug_deg = np.asarray(drug_degree)
    if torch.is_tensor(prot_degree):
        prot_deg = prot_degree.detach().cpu().numpy()
    else:
        prot_deg = np.asarray(prot_degree)
    if torch.is_tensor(edges_np):
        edges_np = edges_np.detach().cpu().numpy()
    if edges_np.size == 0:
        return {}
    d_idx = edges_np[:, 0].astype(np.int64)
    p_idx = edges_np[:, 1].astype(np.int64)
    deg_d = drug_deg[d_idx] if drug_deg.size else np.zeros(d_idx.shape[0], dtype=np.float32)
    deg_p = prot_deg[p_idx] if prot_deg.size else np.zeros(p_idx.shape[0], dtype=np.float32)
    mask_drug = deg_d <= float(cold_deg_th_drug)
    mask_prot = deg_p <= float(cold_deg_th_prot)
    mask_both = mask_drug & mask_prot
    stats = {}
    for name, mask in (("cold_drug", mask_drug), ("cold_prot", mask_prot), ("cold_both", mask_both)):
        if mask.sum() < 2:
            stats[name] = {"n": int(mask.sum()), "auc": 0.0, "aupr": 0.0}
            continue
        auc, aupr = _safe_auc_aupr(labels_np[mask], preds_np[mask])
        stats[name] = {"n": int(mask.sum()), "auc": float(auc), "aupr": float(aupr)}
    return stats


def compute_esm_missing_metrics(edges_np, labels_np, preds_np, prot_missing=None, prot_unreliable=None):
    """Compute metrics on ESM-missing or unreliable subsets."""
    if torch.is_tensor(edges_np):
        edges_np = edges_np.detach().cpu().numpy()
    if edges_np.size == 0:
        return {}
    stats = {}
    p_idx = edges_np[:, 1].astype(np.int64)

    def _compute(mask_src, key):
        if mask_src is None:
            return
        if torch.is_tensor(mask_src):
            mask_src = mask_src.detach().cpu().numpy()
        else:
            mask_src = np.asarray(mask_src)
        mask = mask_src[p_idx] > 0 if mask_src.size else np.zeros(p_idx.shape[0], dtype=bool)
        ratio = float(mask.mean()) if mask.size else 0.0
        if mask.sum() < 2:
            stats[key] = {"n": int(mask.sum()), "ratio": ratio, "auc": 0.0, "aupr": 0.0}
        else:
            auc, aupr = _safe_auc_aupr(labels_np[mask], preds_np[mask])
            stats[key] = {"n": int(mask.sum()), "ratio": ratio, "auc": float(auc), "aupr": float(aupr)}

    _compute(prot_missing, "esm_missing")
    _compute(prot_unreliable, "esm_unreliable")
    return stats


def evaluate_full_graph(model, features_drug_tensor, features_protein_tensor,
                        G_drug_tensor, G_protein_tensor,
                        val_edges_tensor, val_labels_tensor,
                        drug_node_to_entity=None, protein_node_to_entity=None,
                        drug_node_weight=None, protein_node_weight=None,
                        atom_prior=None, res_prior=None,
                        drug_edge_ptr=None, drug_edge_nodes=None, drug_edge_psichic=None,
                        prot_edge_ptr=None, prot_edge_nodes=None, prot_edge_psichic=None,
                        drug_degree=None, prot_degree=None, use_coldstart_gate=None,
                        prot_esm_missing=None, prot_esm_unreliable=None,
                        drug_knn_edge_index=None, drug_knn_edge_weight=None,
                        prot_knn_edge_index=None, prot_knn_edge_weight=None,
                        cold_deg_th_drug=2, cold_deg_th_prot=2,
                        expertC_scale=1.0, wC_cap=None, scaler=None,
                        eval_score_centering="none"):
    """Evaluate model on full graph edges with configurable global score centering."""
    model.eval()
    amp_enabled = (
        scaler is not None
        and scaler.is_enabled()
        and not (G_drug_tensor.is_sparse or G_protein_tensor.is_sparse)
    )
    with torch.no_grad():
        try:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                val_logits, _ = model(
                    features_drug_tensor, features_protein_tensor, G_drug_tensor, G_protein_tensor,
                    drug_node_to_entity=drug_node_to_entity,
                    protein_node_to_entity=protein_node_to_entity,
                    drug_node_weight=drug_node_weight,
                    protein_node_weight=protein_node_weight,
                    atom_prior=atom_prior,
                    res_prior=res_prior,
                    edge_index=val_edges_tensor,
                    drug_edge_ptr=drug_edge_ptr,
                    drug_edge_nodes=drug_edge_nodes,
                    drug_edge_psichic=drug_edge_psichic,
                    prot_edge_ptr=prot_edge_ptr,
                    prot_edge_nodes=prot_edge_nodes,
                    prot_edge_psichic=prot_edge_psichic,
                    drug_degree=drug_degree,
                    prot_degree=prot_degree,
                    use_coldstart_gate=use_coldstart_gate,
                    prot_esm_missing=prot_esm_missing,
                    prot_esm_unreliable=prot_esm_unreliable,
                    drug_knn_edge_index=drug_knn_edge_index,
                    drug_knn_edge_weight=drug_knn_edge_weight,
                    prot_knn_edge_index=prot_knn_edge_index,
                    prot_knn_edge_weight=prot_knn_edge_weight,
                    cold_deg_th_drug=cold_deg_th_drug,
                    cold_deg_th_prot=cold_deg_th_prot,
                    expertC_scale=expertC_scale,
                    wC_cap=wC_cap,
                )
                labels_val = val_labels_tensor
        except RuntimeError as e:
            print(f"Validation forward error: {e}")
            return 0.0, 0.0, {}
        if not torch.isfinite(val_logits).all() or not torch.isfinite(labels_val).all():
            return 0.0, 0.0, {}
        if labels_val.unique().numel() < 2:
            return 0.0, 0.0, {}
        score_val_np = val_logits.detach().cpu().numpy().reshape(-1)
        labels_val_np = labels_val.detach().cpu().numpy()
        edges_np = val_edges_tensor.detach().cpu().numpy()
        n_eff = min(score_val_np.shape[0], labels_val_np.shape[0], edges_np.shape[0])
        if n_eff < 2:
            return 0.0, 0.0, {}
        if score_val_np.shape[0] != n_eff or labels_val_np.shape[0] != n_eff or edges_np.shape[0] != n_eff:
            print("[WARN] Eval size mismatch; truncating scores/labels/edges to aligned length.")
        score_val_np = score_val_np[:n_eff]
        labels_val_np = labels_val_np[:n_eff]
        edges_np = edges_np[:n_eff]
        metric_variants = _compute_global_metric_variants(labels_val_np, score_val_np, edges_np[:, 1])
        score_eval_np, selected_mode = _select_eval_scores(
            score_val_np, edges_np[:, 1], eval_score_centering=eval_score_centering
        )
        auc_val, aupr_val = _safe_auc_aupr(labels_val_np, score_eval_np)
        cold_stats = compute_cold_metrics(
            edges_np, labels_val_np, score_eval_np,
            drug_degree=drug_degree, prot_degree=prot_degree,
            cold_deg_th_drug=cold_deg_th_drug, cold_deg_th_prot=cold_deg_th_prot,
        )
        missing_stats = compute_esm_missing_metrics(
            edges_np, labels_val_np, score_eval_np,
            prot_missing=prot_esm_missing,
            prot_unreliable=prot_esm_unreliable,
        )
        if missing_stats:
            cold_stats.update(missing_stats)
        cold_stats = _attach_global_metrics(
            cold_stats, metric_variants, selected_mode, auc_val, aupr_val
        )
        return auc_val, aupr_val, cold_stats


def evaluate_subgraph_edges(model,
                            features_drug_tensor, features_protein_tensor,
                            atom_indptr, atom_indices, atom_data,
                            res_indptr, res_indices, res_data,
                            edges_np, labels_np,
                            drug_atom_ptr, drug_atom_nodes,
                            prot_res_ptr, prot_res_nodes,
                            atom_orig_pos, residue_orig_pos,
                            atom_to_drug_tensor, residue_to_prot_tensor,
                            atom_attn_tensor, residue_attn_tensor,
                            atom_attn_np, residue_attn_np,
                            attn_db, drug_id_list, prot_id_list,
                            seed_edge_batch_size, walk_length, num_walks,
                            max_atoms_per_step, max_res_per_step,
                            alpha_eps=1e-6,
                            edge_min_incidence_atom=1, edge_min_incidence_res=1,
                            eval_num_samples=1, device=None, rng_seed=0,
                            drug_degree=None, prot_degree=None, use_coldstart_gate=None,
                            prot_esm_missing=None, prot_esm_unreliable=None,
                            drug_knn_edge_index=None, drug_knn_edge_weight=None,
                            prot_knn_edge_index=None, prot_knn_edge_weight=None,
                            cold_deg_th_drug=2, cold_deg_th_prot=2,
                            expertC_scale=1.0, wC_cap=None,
                            eval_score_centering="none",
                            eval_warm_support_k=0,
                            eval_warm_support_max_add=0):
    """Evaluate model on sampled subgraph edges with configurable global score centering."""
    model.eval()
    n_edges = edges_np.shape[0]
    if n_edges == 0:
        return 0.0, 0.0, 0.0, {}
    eval_warm_support_k = max(0, int(eval_warm_support_k))
    eval_warm_support_max_add = max(0, int(eval_warm_support_max_add))
    warm_support_enabled = False
    prot_knn_neighbor_cache = {}
    prot_degree_np = None
    warm_support_added_total = 0
    warm_support_batch_hits = 0
    if eval_warm_support_k > 0:
        prot_knn_neighbor_cache = build_knn_neighbor_cache(prot_knn_edge_index, prot_knn_edge_weight)
        if prot_knn_neighbor_cache:
            warm_support_enabled = True
            if prot_degree is not None:
                if torch.is_tensor(prot_degree):
                    prot_degree_np = prot_degree.detach().cpu().numpy()
                else:
                    prot_degree_np = np.asarray(prot_degree)
        else:
            print("[WARN] eval_warm_support_k > 0 but protein kNN graph is empty; fallback to seed proteins only.")
    preds_samples = []
    for s in range(max(1, int(eval_num_samples))):
        rng = np.random.default_rng(rng_seed + 17 + s * 101)
        preds = np.full(n_edges, np.nan, dtype=np.float32)
        pair_cache = {}
        for start in range(0, n_edges, seed_edge_batch_size):
            end = min(start + seed_edge_batch_size, n_edges)
            batch_edges = edges_np[start:end]
            if batch_edges.size == 0:
                continue
            seed_drugs = np.unique(batch_edges[:, 0])
            seed_prots = np.unique(batch_edges[:, 1])
            if warm_support_enabled:
                seed_prots, n_added = expand_seed_proteins_with_warm_support(
                    seed_prots,
                    prot_knn_neighbor_cache,
                    prot_degree_np=prot_degree_np,
                    support_k=eval_warm_support_k,
                    max_add=eval_warm_support_max_add,
                )
                if n_added > 0:
                    warm_support_batch_hits += 1
                    warm_support_added_total += int(n_added)
            atom_seed_nodes = union_ragged(drug_atom_ptr, drug_atom_nodes, seed_drugs)
            res_seed_nodes = union_ragged(prot_res_ptr, prot_res_nodes, seed_prots)
            if atom_seed_nodes.size == 0 or res_seed_nodes.size == 0:
                continue
            atom_sub_nodes = random_walk_subgraph(
                atom_indptr, atom_indices, atom_seed_nodes,
                walk_length, num_walks, max_atoms_per_step, rng
            )
            res_sub_nodes = random_walk_subgraph(
                res_indptr, res_indices, res_seed_nodes,
                walk_length, num_walks, max_res_per_step, rng
            )
            if atom_sub_nodes.size == 0 or res_sub_nodes.size == 0:
                continue
            atom_edge_index_sub, atom_edge_weight_sub = build_subgraph_edges(
                atom_indptr, atom_indices, atom_data, atom_sub_nodes
            )
            res_edge_index_sub, res_edge_weight_sub = build_subgraph_edges(
                res_indptr, res_indices, res_data, res_sub_nodes
            )

            atom_sub_t = torch.from_numpy(atom_sub_nodes).to(device=device, dtype=torch.long)
            res_sub_t = torch.from_numpy(res_sub_nodes).to(device=device, dtype=torch.long)
            atom_feat_sub = features_drug_tensor[atom_sub_t]
            res_feat_sub = features_protein_tensor[res_sub_t]
            sub_atom_to_drug = atom_to_drug_tensor[atom_sub_t]
            sub_res_to_prot = residue_to_prot_tensor[res_sub_t]
            sub_atom_weight = atom_attn_tensor[atom_sub_t] if atom_attn_tensor is not None else None
            sub_res_weight = residue_attn_tensor[res_sub_t] if residue_attn_tensor is not None else None

            batch_edges_t = torch.from_numpy(batch_edges).to(device=device, dtype=torch.long)
            if drug_id_list is not None:
                num_drugs = len(drug_id_list)
            else:
                num_drugs = int(atom_to_drug_tensor.max().item()) + 1 if atom_to_drug_tensor.numel() else 0
            if prot_id_list is not None:
                num_prots = len(prot_id_list)
            else:
                num_prots = int(residue_to_prot_tensor.max().item()) + 1 if residue_to_prot_tensor.numel() else 0
            atom_counts = torch.bincount(sub_atom_to_drug, minlength=num_drugs)
            res_counts = torch.bincount(sub_res_to_prot, minlength=num_prots)
            inc_mask = (atom_counts[batch_edges_t[:, 0]] >= edge_min_incidence_atom) & (
                res_counts[batch_edges_t[:, 1]] >= edge_min_incidence_res
            )
            if not inc_mask.any():
                continue
            edge_mask_np = inc_mask.detach().cpu().numpy()
            batch_edges_t = batch_edges_t[inc_mask]
            batch_edges_np = batch_edges[edge_mask_np]

            atom_edge_ptr = atom_edge_nodes = atom_edge_prior = None
            res_edge_ptr = res_edge_nodes = res_edge_prior = None
            if batch_edges_np.size > 0:
                atom_edge_ptr, atom_edge_nodes, atom_edge_prior, _ = build_edge_incidence(
                    batch_edges_np,
                    atom_sub_nodes,
                    drug_atom_ptr,
                    drug_atom_nodes,
                    atom_orig_pos,
                    attn_db,
                    drug_id_list,
                    prot_id_list,
                    atom_attn_np,
                    is_atom_side=True,
                    alpha_eps=alpha_eps,
                    cache=pair_cache,
                )
                res_edge_ptr, res_edge_nodes, res_edge_prior, _ = build_edge_incidence(
                    batch_edges_np,
                    res_sub_nodes,
                    prot_res_ptr,
                    prot_res_nodes,
                    residue_orig_pos,
                    attn_db,
                    drug_id_list,
                    prot_id_list,
                    residue_attn_np,
                    is_atom_side=False,
                    alpha_eps=alpha_eps,
                    cache=pair_cache,
                )
            if atom_edge_ptr is not None and atom_edge_ptr.size:
                atom_edge_ptr_t = torch.from_numpy(atom_edge_ptr).to(device=device, dtype=torch.long)
                atom_edge_nodes_t = torch.from_numpy(atom_edge_nodes).to(device=device, dtype=torch.long)
                atom_edge_prior_t = torch.from_numpy(atom_edge_prior).to(device=device, dtype=torch.float32)
            else:
                atom_edge_ptr_t = atom_edge_nodes_t = atom_edge_prior_t = None
            if res_edge_ptr is not None and res_edge_ptr.size:
                res_edge_ptr_t = torch.from_numpy(res_edge_ptr).to(device=device, dtype=torch.long)
                res_edge_nodes_t = torch.from_numpy(res_edge_nodes).to(device=device, dtype=torch.long)
                res_edge_prior_t = torch.from_numpy(res_edge_prior).to(device=device, dtype=torch.float32)
            else:
                res_edge_ptr_t = res_edge_nodes_t = res_edge_prior_t = None

            atom_edge_index_t = torch.from_numpy(atom_edge_index_sub).to(device=device, dtype=torch.long)
            atom_edge_weight_t = torch.from_numpy(atom_edge_weight_sub).to(device=device, dtype=torch.float32)
            res_edge_index_t = torch.from_numpy(res_edge_index_sub).to(device=device, dtype=torch.long)
            res_edge_weight_t = torch.from_numpy(res_edge_weight_sub).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                edge_logits, _ = model(
                    atom_feat_sub, res_feat_sub, None, None,
                    drug_node_to_entity=sub_atom_to_drug,
                    protein_node_to_entity=sub_res_to_prot,
                    drug_node_weight=sub_atom_weight,
                    protein_node_weight=sub_res_weight,
                    atom_prior=sub_atom_weight,
                    res_prior=sub_res_weight,
                    edge_index=batch_edges_t,
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
                    drug_degree=drug_degree,
                    prot_degree=prot_degree,
                    use_coldstart_gate=use_coldstart_gate,
                    prot_esm_missing=prot_esm_missing,
                    prot_esm_unreliable=prot_esm_unreliable,
                    drug_knn_edge_index=drug_knn_edge_index,
                    drug_knn_edge_weight=drug_knn_edge_weight,
                    prot_knn_edge_index=prot_knn_edge_index,
                    prot_knn_edge_weight=prot_knn_edge_weight,
                    cold_deg_th_drug=cold_deg_th_drug,
                    cold_deg_th_prot=cold_deg_th_prot,
                    expertC_scale=expertC_scale,
                    wC_cap=wC_cap,
                )
                pred = edge_logits.detach().cpu().numpy().reshape(-1)
            idx = np.arange(start, end)[edge_mask_np]
            preds[idx] = pred
        preds_samples.append(preds)
    if warm_support_enabled:
        avg_added = warm_support_added_total / max(warm_support_batch_hits, 1)
        print(
            f"[EVAL-WARM-SUPPORT] k={eval_warm_support_k} max_add={eval_warm_support_max_add} "
            f"batches_with_add={warm_support_batch_hits} avg_added={avg_added:.2f}"
        )

    preds_stack = np.stack(preds_samples, axis=0)
    preds_mean = np.nanmean(preds_stack, axis=0)
    mask = np.isfinite(preds_mean)
    coverage = float(mask.mean()) if mask.size else 0.0
    if mask.sum() < 2:
        return 0.0, 0.0, coverage, {}
    labels_eval = labels_np[mask]
    if np.unique(labels_eval).size < 2:
        return 0.0, 0.0, coverage, {}
    edges_eval = edges_np[mask]
    scores_eval = preds_mean[mask]
    metric_variants = _compute_global_metric_variants(labels_eval, scores_eval, edges_eval[:, 1])
    scores_eval_sel, selected_mode = _select_eval_scores(
        scores_eval, edges_eval[:, 1], eval_score_centering=eval_score_centering
    )
    auc_val, aupr_val = _safe_auc_aupr(labels_eval, scores_eval_sel)
    cold_stats = compute_cold_metrics(
        edges_eval, labels_eval, scores_eval_sel,
        drug_degree=drug_degree, prot_degree=prot_degree,
        cold_deg_th_drug=cold_deg_th_drug, cold_deg_th_prot=cold_deg_th_prot,
    )
    missing_stats = compute_esm_missing_metrics(
        edges_eval, labels_eval, scores_eval_sel,
        prot_missing=prot_esm_missing,
        prot_unreliable=prot_esm_unreliable,
    )
    if missing_stats:
        cold_stats.update(missing_stats)
    cold_stats = _attach_global_metrics(
        cold_stats, metric_variants, selected_mode, auc_val, aupr_val
    )
    return auc_val, aupr_val, coverage, cold_stats


def train_one_epoch_saint(epoch, fold, model, optimizer, scheduler, criterion,
                          features_drug_tensor, features_protein_tensor,
                          train_edges, train_labels,
                          atom_indptr, atom_indices, atom_data,
                          res_indptr, res_indices, res_data,
                          drug_atom_ptr, drug_atom_nodes,
                          prot_res_ptr, prot_res_nodes,
                          atom_to_drug_tensor, residue_to_prot_tensor,
                          atom_attn_tensor, residue_attn_tensor,
                          drug_total_count, prot_total_count,
                          atom_attn_np=None, residue_attn_np=None,
                          attn_db=None, drug_id_list=None, prot_id_list=None,
                          atom_orig_pos=None, residue_orig_pos=None,
                          train_losses=None, device=None, seed_edge_batch_size=0,
                          steps_per_epoch=0, walk_length=0, num_walks=0,
                          max_atoms_per_step=0, max_res_per_step=0,
                          alpha_eps=1e-6, edge_min_incidence_atom=1, edge_min_incidence_res=1,
                          scaler=None, log_interval=50, use_reweight=False,
                          drug_degree=None, prot_degree=None, use_coldstart_gate=None,
                          info_nce_on="protein", subpocket_mask_ratio=0.0, subpocket_keep_top=0.2,
                          ema_state=None, ema_decay=0.999, use_ema=False,
                          prot_esm_missing=None, prot_esm_unreliable=None,
                          drug_knn_edge_index=None, drug_knn_edge_weight=None,
                          prot_knn_edge_index=None, prot_knn_edge_weight=None,
                          cold_deg_th_drug=2, cold_deg_th_prot=2,
                          cold_prot_weight=1.0,
                          seed_base=0,
                          expertC_scale=1.0, wC_cap=None, debug_assertions=False,
                          kl_hard_assert=False,
                          cold_start_dropout_p=0.0,
                          cold_start_dropout_assert_ratio=0.1):
    """Train one epoch with GraphSAINT-style subgraph sampling."""
    t = time.time()
    model.train()
    total_loss = 0.0
    valid_steps = 0
    had_valid_batch = False
    edge_valid_total = 0
    atom_inc_total = 0
    res_inc_total = 0
    ent_total = 0.0
    ent_weight_total = 0
    mask_total = 0
    mask_count = 0
    rng = np.random.default_rng(int(seed_base) + int(epoch) * 1009 + int(fold) * 100003 + 11)
    # Sparse matmul does not support FP16 on CUDA; disable AMP in SAINT path.
    amp_enabled = False
    num_edges = train_edges.shape[0]
    b_total = 0
    b_eff_total = 0
    pair_cache = {}
    gate_atom_sum = torch.zeros(5, device=device) if device is not None else torch.zeros(5)
    gate_atom_cnt = torch.zeros(5, device=device) if device is not None else torch.zeros(5)
    gate_res_sum = torch.zeros(5, device=device) if device is not None else torch.zeros(5)
    gate_res_cnt = torch.zeros(5, device=device) if device is not None else torch.zeros(5)
    gate_w_sum = None
    gate_w_count = 0
    t_sampling = 0.0
    t_encoder = 0.0
    t_heads = 0.0
    t_backward = 0.0
    t_ema = 0.0
    cold_edge_sum = 0.0
    cold_edge_count = 0
    cold_drug_only_sum = 0.0
    cold_prot_only_sum = 0.0
    cold_both_sum = 0.0
    wC_warm_sum = 0.0
    wC_warm_count = 0
    wC_cold_sum = 0.0
    wC_cold_count = 0
    deg_d_samples = []
    deg_p_samples = []
    gate_w_samples = [[], [], []]
    mp_alpha_cold_sum = 0.0
    mp_alpha_warm_sum = 0.0
    mp_alpha_cold_count = 0
    mp_alpha_warm_count = 0
    mp_alpha_cold_samples = []
    mp_alpha_warm_samples = []
    cold_drop_seed_prot_total = 0
    cold_drop_selected_total = 0
    cold_drop_present_total = 0
    cold_drop_zero_total = 0
    cold_drop_loss_edge_total = 0
    cold_drop_loss_edge_dropped_total = 0

    for step in range(steps_per_epoch):
        t_sample_start = time.time()
        replace = num_edges < seed_edge_batch_size
        seed_idx = rng.choice(num_edges, size=seed_edge_batch_size, replace=replace)
        seed_edges = train_edges[seed_idx]
        seed_labels = train_labels[seed_idx]
        b = seed_edges.shape[0]

        seed_drugs = np.unique(seed_edges[:, 0])
        seed_prots = np.unique(seed_edges[:, 1])

        atom_seed_nodes = union_ragged(drug_atom_ptr, drug_atom_nodes, seed_drugs)
        res_seed_nodes = union_ragged(prot_res_ptr, prot_res_nodes, seed_prots)

        atom_sub_nodes = random_walk_subgraph(
            atom_indptr, atom_indices, atom_seed_nodes,
            walk_length, num_walks, max_atoms_per_step, rng
        )
        res_sub_nodes = random_walk_subgraph(
            res_indptr, res_indices, res_seed_nodes,
            walk_length, num_walks, max_res_per_step, rng
        )

        if atom_sub_nodes.size == 0 or res_sub_nodes.size == 0:
            b_total += b
            continue

        atom_edge_index_sub, atom_edge_weight_sub = build_subgraph_edges(
            atom_indptr, atom_indices, atom_data, atom_sub_nodes
        )
        res_edge_index_sub, res_edge_weight_sub = build_subgraph_edges(
            res_indptr, res_indices, res_data, res_sub_nodes
        )

        atom_sub_t = torch.from_numpy(atom_sub_nodes).to(device=device, dtype=torch.long)
        res_sub_t = torch.from_numpy(res_sub_nodes).to(device=device, dtype=torch.long)

        atom_feat_sub = features_drug_tensor[atom_sub_t]
        res_feat_sub = features_protein_tensor[res_sub_t]

        sub_atom_to_drug = atom_to_drug_tensor[atom_sub_t]
        sub_res_to_prot = residue_to_prot_tensor[res_sub_t]
        sub_atom_weight = atom_attn_tensor[atom_sub_t] if atom_attn_tensor is not None else None
        sub_res_weight = residue_attn_tensor[res_sub_t] if residue_attn_tensor is not None else None
        dropped_prots_t = None
        cold_drop_stats_step = None
        prot_degree_step = prot_degree
        if float(cold_start_dropout_p) > 0:
            res_edge_index_sub, res_edge_weight_sub, dropped_prots_t, cold_drop_stats_step = apply_episodic_protein_edge_drop(
                res_edge_index_sub,
                res_edge_weight_sub,
                sub_res_to_prot,
                seed_prots,
                cold_start_dropout_p,
                rng,
            )
            if cold_drop_stats_step is not None:
                cold_drop_seed_prot_total += int(cold_drop_stats_step.get("batch_prot", 0))
                cold_drop_selected_total += int(cold_drop_stats_step.get("dropped_prot", 0))
                cold_drop_present_total += int(cold_drop_stats_step.get("dropped_present", 0))
                cold_drop_zero_total += int(cold_drop_stats_step.get("dropped_zero_after", 0))
            if (
                dropped_prots_t is not None
                and dropped_prots_t.numel() > 0
                and prot_degree is not None
            ):
                prot_degree_step = prot_degree.clone()
                prot_degree_step[dropped_prots_t] = 0

        if subpocket_mask_ratio > 0 and sub_res_weight is not None:
            mask = torch.zeros(res_feat_sub.size(0), dtype=torch.bool, device=res_feat_sub.device)
            for prot_id in torch.unique(sub_res_to_prot):
                idx = (sub_res_to_prot == prot_id).nonzero(as_tuple=False).view(-1)
                if idx.numel() < 2:
                    continue
                prior = sub_res_weight[idx]
                keep_top = float(subpocket_keep_top)
                keep_top = min(max(keep_top, 0.0), 1.0)
                k_keep = max(1, int(math.ceil(idx.numel() * keep_top)))
                if k_keep >= idx.numel():
                    continue
                top_idx = torch.topk(prior, k_keep).indices
                keep_mask = torch.zeros(idx.numel(), dtype=torch.bool, device=idx.device)
                keep_mask[top_idx] = True
                low_idx = idx[~keep_mask]
                if low_idx.numel() == 0:
                    continue
                n_mask = max(1, int(math.floor(low_idx.numel() * float(subpocket_mask_ratio))))
                if n_mask <= 0:
                    continue
                perm = torch.randperm(low_idx.numel(), device=idx.device)[:n_mask]
                mask_idx = low_idx[perm]
                mask[mask_idx] = True
            if mask.any():
                res_feat_sub = res_feat_sub.clone()
                res_feat_sub[mask] = 0
                mask_count += int(mask.sum().item())
            mask_total += int(res_feat_sub.size(0))

        seed_edges_t = torch.from_numpy(seed_edges).to(device=device, dtype=torch.long)
        seed_labels_t = torch.from_numpy(seed_labels).to(device=device, dtype=torch.float32)

        drug_ids = torch.unique(sub_atom_to_drug, sorted=True)
        prot_ids = torch.unique(sub_res_to_prot, sorted=True)
        drug_pos = torch.searchsorted(drug_ids, seed_edges_t[:, 0])
        prot_pos = torch.searchsorted(prot_ids, seed_edges_t[:, 1])
        if drug_ids.numel() == 0:
            drug_mask = torch.zeros(seed_edges_t.size(0), dtype=torch.bool, device=seed_edges_t.device)
        else:
            drug_in = drug_pos < drug_ids.numel()
            drug_pos_safe = drug_pos.clamp_max(drug_ids.numel() - 1)
            drug_mask = drug_in & (drug_ids[drug_pos_safe] == seed_edges_t[:, 0])
        if prot_ids.numel() == 0:
            prot_mask = torch.zeros(seed_edges_t.size(0), dtype=torch.bool, device=seed_edges_t.device)
        else:
            prot_in = prot_pos < prot_ids.numel()
            prot_pos_safe = prot_pos.clamp_max(prot_ids.numel() - 1)
            prot_mask = prot_in & (prot_ids[prot_pos_safe] == seed_edges_t[:, 1])
        edge_mask = drug_mask & prot_mask

        num_drugs = int(drug_total_count.numel()) if drug_total_count is not None else int(drug_id_list and len(drug_id_list) or 0)
        num_prots = int(prot_total_count.numel()) if prot_total_count is not None else int(prot_id_list and len(prot_id_list) or 0)
        atom_counts_t = torch.bincount(sub_atom_to_drug, minlength=num_drugs)
        res_counts_t = torch.bincount(sub_res_to_prot, minlength=num_prots)
        inc_mask = (atom_counts_t[seed_edges_t[:, 0]] >= edge_min_incidence_atom) & (
            res_counts_t[seed_edges_t[:, 1]] >= edge_min_incidence_res
        )
        edge_mask = edge_mask & inc_mask
        b_eff = int(edge_mask.sum().item())
        b_total += b
        b_eff_total += b_eff
        edge_valid_total += b_eff
        edge_mask_np = edge_mask.detach().cpu().numpy()
        if b_eff > 0:
            atom_counts_e = atom_counts_t[seed_edges_t[:, 0]]
            res_counts_e = res_counts_t[seed_edges_t[:, 1]]
            atom_inc_total += int(atom_counts_e[edge_mask].sum().item())
            res_inc_total += int(res_counts_e[edge_mask].sum().item())
        log_hit = log_interval and ((step + 1) % log_interval == 0)
        if log_hit:
            ratio = b_eff / max(b, 1)
            atom_counts_e = atom_counts_t[seed_edges_t[:, 0]]
            res_counts_e = res_counts_t[seed_edges_t[:, 1]]
            avg_atom_inc = float(atom_counts_e[edge_mask].float().mean().item()) if b_eff > 0 else 0.0
            avg_res_inc = float(res_counts_e[edge_mask].float().mean().item()) if b_eff > 0 else 0.0
            print(
                f"[SAINT] step {step + 1}/{steps_per_epoch} E_eff/E={b_eff}/{b} ({ratio:.3f}) "
                f"avg_inc(atom/res)={avg_atom_inc:.1f}/{avg_res_inc:.1f}"
            )
            if ratio < 0.90:
                print(f"[WARN] Low E_eff/E={ratio:.3f}.")
        if not edge_mask.any():
            continue

        seed_edges_t = seed_edges_t[edge_mask]
        seed_labels_t = seed_labels_t[edge_mask]
        drug_pos = drug_pos[edge_mask]
        prot_pos = prot_pos[edge_mask]
        seed_edges_np = seed_edges[edge_mask_np]
        if float(cold_start_dropout_p) > 0:
            cold_drop_loss_edge_total += int(seed_edges_t.size(0))
            if dropped_prots_t is not None and dropped_prots_t.numel() > 0 and seed_edges_t.numel() > 0:
                drop_loss_edges = int(torch.isin(seed_edges_t[:, 1], dropped_prots_t).sum().item())
                cold_drop_loss_edge_dropped_total += drop_loss_edges
                if log_hit:
                    print(
                        f"[COLD-DROP] step {step + 1}/{steps_per_epoch} "
                        f"dropped_prot={int(dropped_prots_t.numel())} "
                        f"loss_edges_on_dropped_prot={drop_loss_edges}/{int(seed_edges_t.size(0))}"
                    )

        cold_edge_mask_batch = None
        cold_prot_mask_batch = None
        warm_edge_mask_batch = None
        if drug_degree is not None and prot_degree_step is not None and seed_edges_t.numel():
            deg_d_edge = drug_degree[seed_edges_t[:, 0]]
            deg_p_edge = prot_degree_step[seed_edges_t[:, 1]]
            cold_d_edge = deg_d_edge <= float(cold_deg_th_drug)
            cold_p_edge = deg_p_edge <= float(cold_deg_th_prot)
            cold_prot_mask_batch = cold_p_edge
            cold_edge_mask_batch = cold_d_edge | cold_p_edge
            warm_edge_mask_batch = ~cold_edge_mask_batch
            cold_edge_sum += float(cold_edge_mask_batch.float().sum().item())
            cold_edge_count += int(cold_edge_mask_batch.numel())
            cold_drug_only_sum += float((cold_d_edge & ~cold_p_edge).float().sum().item())
            cold_prot_only_sum += float((cold_p_edge & ~cold_d_edge).float().sum().item())
            cold_both_sum += float((cold_d_edge & cold_p_edge).float().sum().item())
            _append_samples(deg_d_samples, deg_d_edge, max_samples=2048, rng=rng)
            _append_samples(deg_p_samples, deg_p_edge, max_samples=2048, rng=rng)
            if debug_assertions and epoch == 0 and step == 0:
                ratio = float(cold_edge_mask_batch.float().mean().item())
                if ratio > 0.5:
                    print("[WARN] cold_edge_ratio > 0.5 in first batch; check degree source (should be GLOBAL_TRAIN_ONLY).")
                    idx = torch.nonzero(cold_edge_mask_batch, as_tuple=False).view(-1)[:10]
                    for j in idx.tolist():
                        print(
                            f"[WARN] edge_sample drug={int(seed_edges_t[j,0])} prot={int(seed_edges_t[j,1])} "
                            f"deg_d={float(deg_d_edge[j].item()):.1f} deg_p={float(deg_p_edge[j].item()):.1f}"
                        )

        atom_edge_ptr = atom_edge_nodes = atom_edge_prior = None
        res_edge_ptr = res_edge_nodes = res_edge_prior = None
        if seed_edges_np.size > 0:
            atom_edge_ptr, atom_edge_nodes, atom_edge_prior, _ = build_edge_incidence(
                seed_edges_np,
                atom_sub_nodes,
                drug_atom_ptr,
                drug_atom_nodes,
                atom_orig_pos,
                attn_db,
                drug_id_list,
                prot_id_list,
                atom_attn_np,
                is_atom_side=True,
                alpha_eps=alpha_eps,
                cache=pair_cache,
                debug_step=step,
            )
            res_edge_ptr, res_edge_nodes, res_edge_prior, _ = build_edge_incidence(
                seed_edges_np,
                res_sub_nodes,
                prot_res_ptr,
                prot_res_nodes,
                residue_orig_pos,
                attn_db,
                drug_id_list,
                prot_id_list,
                residue_attn_np,
                is_atom_side=False,
                alpha_eps=alpha_eps,
                cache=pair_cache,
                debug_step=step,
            )
        if atom_edge_ptr is not None and atom_edge_ptr.size:
            atom_edge_ptr_t = torch.from_numpy(atom_edge_ptr).to(device=device, dtype=torch.long)
            atom_edge_nodes_t = torch.from_numpy(atom_edge_nodes).to(device=device, dtype=torch.long)
            atom_edge_prior_t = torch.from_numpy(atom_edge_prior).to(device=device, dtype=torch.float32)
        else:
            atom_edge_ptr_t = atom_edge_nodes_t = atom_edge_prior_t = None
        if res_edge_ptr is not None and res_edge_ptr.size:
            res_edge_ptr_t = torch.from_numpy(res_edge_ptr).to(device=device, dtype=torch.long)
            res_edge_nodes_t = torch.from_numpy(res_edge_nodes).to(device=device, dtype=torch.long)
            res_edge_prior_t = torch.from_numpy(res_edge_prior).to(device=device, dtype=torch.float32)
        else:
            res_edge_ptr_t = res_edge_nodes_t = res_edge_prior_t = None

        if use_reweight:
            drug_inv = torch.searchsorted(drug_ids, sub_atom_to_drug)
            prot_inv = torch.searchsorted(prot_ids, sub_res_to_prot)
            drug_counts = torch.bincount(drug_inv, minlength=drug_ids.numel()).float().clamp_min(1.0)
            prot_counts = torch.bincount(prot_inv, minlength=prot_ids.numel()).float().clamp_min(1.0)
            drug_weight = drug_total_count[seed_edges_t[:, 0]] / drug_counts[drug_pos]
            prot_weight = prot_total_count[seed_edges_t[:, 1]] / prot_counts[prot_pos]
            edge_weight = drug_weight * prot_weight
            edge_weight = edge_weight / edge_weight.mean().clamp_min(1e-6)
        else:
            edge_weight = None

        atom_edge_index_t = torch.from_numpy(atom_edge_index_sub).to(device=device, dtype=torch.long)
        atom_edge_weight_t = torch.from_numpy(atom_edge_weight_sub).to(device=device, dtype=torch.float32)
        res_edge_index_t = torch.from_numpy(res_edge_index_sub).to(device=device, dtype=torch.long)
        res_edge_weight_t = torch.from_numpy(res_edge_weight_sub).to(device=device, dtype=torch.float32)

        t_sampling += time.time() - t_sample_start

        optimizer.zero_grad()
        try:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                use_info_nce = bool(criterion.info_nce_weight and criterion.info_nce_weight > 0)
                if use_info_nce:
                    edge_logits, vae_params, pair_repr = model(
                        atom_feat_sub, res_feat_sub, None, None,
                        drug_node_to_entity=sub_atom_to_drug,
                        protein_node_to_entity=sub_res_to_prot,
                        drug_node_weight=sub_atom_weight,
                        protein_node_weight=sub_res_weight,
                        atom_prior=sub_atom_weight,
                        res_prior=sub_res_weight,
                        edge_index=seed_edges_t,
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
                        drug_degree=drug_degree,
                        prot_degree=prot_degree_step,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing,
                        prot_esm_unreliable=prot_esm_unreliable,
                        drug_knn_edge_index=drug_knn_edge_index,
                        drug_knn_edge_weight=drug_knn_edge_weight,
                        prot_knn_edge_index=prot_knn_edge_index,
                        prot_knn_edge_weight=prot_knn_edge_weight,
                        cold_deg_th_drug=cold_deg_th_drug,
                        cold_deg_th_prot=cold_deg_th_prot,
                        expertC_scale=expertC_scale,
                        wC_cap=wC_cap,
                        return_pair_repr=True,
                    )
                else:
                    edge_logits, vae_params = model(
                        atom_feat_sub, res_feat_sub, None, None,
                        drug_node_to_entity=sub_atom_to_drug,
                        protein_node_to_entity=sub_res_to_prot,
                        drug_node_weight=sub_atom_weight,
                        protein_node_weight=sub_res_weight,
                        atom_prior=sub_atom_weight,
                        res_prior=sub_res_weight,
                        edge_index=seed_edges_t,
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
                        drug_degree=drug_degree,
                        prot_degree=prot_degree_step,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing,
                        prot_esm_unreliable=prot_esm_unreliable,
                        drug_knn_edge_index=drug_knn_edge_index,
                        drug_knn_edge_weight=drug_knn_edge_weight,
                        prot_knn_edge_index=prot_knn_edge_index,
                        prot_knn_edge_weight=prot_knn_edge_weight,
                        cold_deg_th_drug=cold_deg_th_drug,
                        cold_deg_th_prot=cold_deg_th_prot,
                        expertC_scale=expertC_scale,
                        wC_cap=wC_cap,
                    )
                profile = getattr(model, "_last_profile", None)
                if profile:
                    t_encoder += float(profile.get("encoder", 0.0))
                    t_heads += float(profile.get("heads", 0.0))
                (drug_mu, drug_logvar, prot_mu, prot_logvar,
                 attn_kl_norm, attn_kl_raw, ent_atom, ent_res) = unpack_vae_params(vae_params)
                gate_atom = getattr(model, "_last_mp_gate_atom", None)
                gate_res = getattr(model, "_last_mp_gate_res", None)
                if gate_atom is not None and drug_degree is not None:
                    deg_atom = drug_degree[sub_atom_to_drug]
                    accumulate_gate_bins(gate_atom_sum, gate_atom_cnt, gate_atom, deg_atom)
                    cold_atom = deg_atom <= float(cold_deg_th_drug)
                    if cold_atom.any():
                        mp_alpha_cold_sum += float(gate_atom[cold_atom].sum().item())
                        mp_alpha_cold_count += int(cold_atom.sum().item())
                        _append_samples(mp_alpha_cold_samples, gate_atom[cold_atom], max_samples=1024, rng=rng)
                    warm_atom = ~cold_atom
                    if warm_atom.any():
                        mp_alpha_warm_sum += float(gate_atom[warm_atom].sum().item())
                        mp_alpha_warm_count += int(warm_atom.sum().item())
                        _append_samples(mp_alpha_warm_samples, gate_atom[warm_atom], max_samples=1024, rng=rng)
                if gate_res is not None and prot_degree_step is not None:
                    deg_res = prot_degree_step[sub_res_to_prot]
                    accumulate_gate_bins(gate_res_sum, gate_res_cnt, gate_res, deg_res)
                    cold_res = deg_res <= float(cold_deg_th_prot)
                    if cold_res.any():
                        mp_alpha_cold_sum += float(gate_res[cold_res].sum().item())
                        mp_alpha_cold_count += int(cold_res.sum().item())
                        _append_samples(mp_alpha_cold_samples, gate_res[cold_res], max_samples=1024, rng=rng)
                    warm_res = ~cold_res
                    if warm_res.any():
                        mp_alpha_warm_sum += float(gate_res[warm_res].sum().item())
                        mp_alpha_warm_count += int(warm_res.sum().item())
                        _append_samples(mp_alpha_warm_samples, gate_res[warm_res], max_samples=1024, rng=rng)
                sample_weight = edge_weight
                if cold_prot_mask_batch is not None and cold_prot_mask_batch.numel() == edge_logits.numel():
                    cpw = float(cold_prot_weight)
                    if cpw != 1.0:
                        cold_scale = torch.ones_like(edge_logits, dtype=edge_logits.dtype, device=edge_logits.device)
                        cold_scale[cold_prot_mask_batch] = cpw
                        if sample_weight is None:
                            sample_weight = cold_scale
                        else:
                            sample_weight = sample_weight.to(device=edge_logits.device, dtype=edge_logits.dtype)
                            sample_weight = sample_weight * cold_scale
                if sample_weight is not None and sample_weight.numel() != edge_logits.numel():
                    raise ValueError("sample_weight must align with edge_logits")
                pair_group = None
                pair_repr_in = None
                if use_info_nce:
                    if info_nce_on == "drug":
                        pair_group = seed_edges_t[:, 0]
                    else:
                        pair_group = seed_edges_t[:, 1]
                    pair_repr_in = pair_repr
                gate_weights = getattr(model, "_last_gate_weights", None)
                if gate_weights is not None and gate_weights.size(0) != edge_logits.numel():
                    gate_weights = None
                distill_logits = None
                distill_mask = None
                if getattr(criterion, "distill_weight", 0.0):
                    distill_logits = getattr(model, "_last_teacher_logits", None)
                    if distill_logits is not None and distill_logits.numel() != edge_logits.numel():
                        distill_logits = None
                    if distill_logits is not None and getattr(criterion, "distill_mode", "warm_only") == "warm_only":
                        if warm_edge_mask_batch is not None and warm_edge_mask_batch.numel() == edge_logits.numel():
                            distill_mask = warm_edge_mask_batch
                if gate_weights is not None:
                    mean_gate = gate_weights.mean(dim=0).detach()
                    gate_w_sum = mean_gate if gate_w_sum is None else gate_w_sum + mean_gate
                    gate_w_count += 1
                    if cold_edge_mask_batch is not None and cold_edge_mask_batch.numel() == gate_weights.size(0):
                        warm_mask = ~cold_edge_mask_batch
                        if warm_mask.any():
                            wC_warm_sum += float(gate_weights[warm_mask, 2].sum().item())
                            wC_warm_count += int(warm_mask.sum().item())
                        if cold_edge_mask_batch.any():
                            wC_cold_sum += float(gate_weights[cold_edge_mask_batch, 2].sum().item())
                            wC_cold_count += int(cold_edge_mask_batch.sum().item())
                        if debug_assertions and epoch == 0 and step == 0 and warm_mask.any():
                            wC_warm_mean = float(gate_weights[warm_mask, 2].mean().item())
                            if wC_warm_mean > 1e-3:
                                msg = (
                                    f"[ASSERT] wC_warm_mean={wC_warm_mean:.6f} > 1e-3; "
                                    "warm edges should suppress ExpertC."
                                )
                                if kl_hard_assert:
                                    raise RuntimeError(msg)
                                print(f"[WARN] {msg}")
                    for i in range(3):
                        _append_samples(gate_w_samples[i], gate_weights[:, i], max_samples=1024, rng=rng)
                aux = getattr(model, "_last_aux", {}) if hasattr(model, "_last_aux") else {}
                prior_conf = None
                if aux:
                    prior_conf = {
                        "atom": aux.get("prior_conf_atom_mean", None),
                        "res": aux.get("prior_conf_res_mean", None),
                    }
                prior_conf_edge = aux.get("prior_conf_edge", None) if isinstance(aux, dict) else None
                if prior_conf is None or all(v is None for v in prior_conf.values()):
                    prior_conf = getattr(model, "_last_prior_conf", None)
                delta_reg = getattr(model, "_last_delta_reg", None)
                kl_stats = aux.get("kl_stats", None) if isinstance(aux, dict) else None
                if kl_stats is None and isinstance(aux, dict):
                    kl_stats = {
                        "kl_clip_count": int(aux.get("kl_clip_count", 0)),
                        "kl_nan_count": int(aux.get("kl_nan_count", 0)),
                        "prior_nan_count": int(aux.get("prior_nan_count", 0)),
                        "renorm_count": int(aux.get("renorm_count", 0)),
                    }
                loss = criterion(edge_logits, seed_labels_t, drug_mu, drug_logvar, prot_mu, prot_logvar,
                                 sample_weight=sample_weight, attn_kl=attn_kl_norm, attn_kl_raw=attn_kl_raw,
                                 pair_repr=pair_repr_in, pair_group=pair_group,
                                 gate_weights=gate_weights, distill_logits=distill_logits,
                                 distill_mask=distill_mask, prior_conf=prior_conf,
                                 prior_conf_edge=prior_conf_edge,
                                 delta_reg=delta_reg, kl_stats=kl_stats)
                if ent_atom is not None and ent_res is not None and b_eff > 0:
                    ent_val = 0.5 * (ent_atom + ent_res)
                    ent_total += float(ent_val.item()) * b_eff
                    ent_weight_total += b_eff
                    if log_hit:
                        print(f"[SAINT] step {step + 1}/{steps_per_epoch} alpha_entropy_raw={float(ent_val.item()):.4f}")
                if log_hit:
                    kl_enabled_step = float(getattr(criterion, "attn_kl_weight", 0.0)) > 0.0
                    if kl_enabled_step:
                        step_attn_kl_norm = float(attn_kl_norm.item())
                        step_attn_kl_raw = float(attn_kl_raw.item())
                    else:
                        step_attn_kl_norm = 0.0
                        step_attn_kl_raw = 0.0
                    print(
                        f"[SAINT] step {step + 1}/{steps_per_epoch} "
                        f"attn_kl_norm={step_attn_kl_norm:.6f} "
                        f"attn_kl_raw={step_attn_kl_raw:.6f}"
                    )
            if torch.isnan(loss) or torch.isinf(loss):
                continue
        except RuntimeError as e:
            print(f"Forward pass error: {e}, skip this batch")
            continue

        t_back_start = time.time()
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        t_backward += time.time() - t_back_start
        if use_ema:
            t_ema_start = time.time()
            update_ema(ema_state, model, ema_decay)
            t_ema += time.time() - t_ema_start
        total_loss += loss.item()
        had_valid_batch = True
        valid_steps += 1
        print(f"Epoch {epoch + 1}, Step {step + 1}/{steps_per_epoch}, Loss: {loss.item():.5f}", end="\r")

    avg_loss = total_loss / max(valid_steps, 1)
    train_losses[fold].append(avg_loss)

    if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()

    if had_valid_batch:
        (bce_loss, weighted_kl_loss, attn_kl_raw, attn_kl_norm,
         kl_ratio, kl_weight_eff, prior_conf_mean, info_nce, info_nce_ratio,
         distill_loss, distill_raw, kd_ratio, kd_conf, gate_balance, gate_entropy,
         total_weighted) = criterion.get_components()
    else:
        bce_loss, weighted_kl_loss, attn_kl_raw, attn_kl_norm, kl_ratio = (0.0, 0.0, 0.0, 0.0, 0.0)
        kl_weight_eff = prior_conf_mean = 0.0
        info_nce = info_nce_ratio = distill_loss = distill_raw = kd_ratio = kd_conf = gate_balance = gate_entropy = 0.0
        total_weighted = 0.0
    epoch_time = time.time() - t
    avg_atom_inc = atom_inc_total / max(edge_valid_total, 1)
    avg_res_inc = res_inc_total / max(edge_valid_total, 1)
    avg_entropy = ent_total / max(ent_weight_total, 1)
    avg_mask_ratio = mask_count / max(mask_total, 1)
    gate_atom_mean = (gate_atom_sum / gate_atom_cnt.clamp_min(1.0)).detach().cpu().numpy()
    gate_res_mean = (gate_res_sum / gate_res_cnt.clamp_min(1.0)).detach().cpu().numpy()
    if gate_w_sum is not None and gate_w_count > 0:
        gate_w_mean = (gate_w_sum / gate_w_count).detach().cpu().numpy()
    else:
        gate_w_mean = np.zeros(3, dtype=np.float32)
    cold_edge_ratio = cold_edge_sum / max(cold_edge_count, 1)
    cold_drug_only_ratio = cold_drug_only_sum / max(cold_edge_count, 1)
    cold_prot_only_ratio = cold_prot_only_sum / max(cold_edge_count, 1)
    cold_both_ratio = cold_both_sum / max(cold_edge_count, 1)
    wC_warm_mean = wC_warm_sum / max(wC_warm_count, 1)
    wC_cold_mean = wC_cold_sum / max(wC_cold_count, 1)
    mp_alpha_cold_mean = mp_alpha_cold_sum / max(mp_alpha_cold_count, 1)
    mp_alpha_warm_mean = mp_alpha_warm_sum / max(mp_alpha_warm_count, 1)
    mp_alpha_cold_p50 = float(np.percentile(mp_alpha_cold_samples, 50)) if mp_alpha_cold_samples else 0.0
    mp_alpha_cold_p90 = float(np.percentile(mp_alpha_cold_samples, 90)) if mp_alpha_cold_samples else 0.0
    mp_alpha_warm_p50 = float(np.percentile(mp_alpha_warm_samples, 50)) if mp_alpha_warm_samples else 0.0
    mp_alpha_warm_p90 = float(np.percentile(mp_alpha_warm_samples, 90)) if mp_alpha_warm_samples else 0.0
    gate_w_p10 = [float(np.percentile(g, 10)) if g else 0.0 for g in gate_w_samples]
    gate_w_p50 = [float(np.percentile(g, 50)) if g else 0.0 for g in gate_w_samples]
    gate_w_p90 = [float(np.percentile(g, 90)) if g else 0.0 for g in gate_w_samples]
    cold_drop_selected_ratio = cold_drop_selected_total / max(cold_drop_seed_prot_total, 1)
    cold_drop_zero_ratio = cold_drop_zero_total / max(cold_drop_present_total, 1)
    cold_drop_loss_ratio = cold_drop_loss_edge_dropped_total / max(cold_drop_loss_edge_total, 1)
    if float(cold_start_dropout_p) > 0:
        gate_expert3_share = float(gate_w_mean[2]) if np.asarray(gate_w_mean).size >= 3 else 0.0
        print(
            f"[COLD-DROP] epoch={epoch + 1} p={float(cold_start_dropout_p):.3f} "
            f"selected={cold_drop_selected_total}/{max(cold_drop_seed_prot_total, 1)} ({cold_drop_selected_ratio:.3f}) "
            f"zero_after={cold_drop_zero_total}/{max(cold_drop_present_total, 1)} ({cold_drop_zero_ratio:.3f}) "
            f"loss_edges_on_dropped_prot={cold_drop_loss_edge_dropped_total}/{max(cold_drop_loss_edge_total, 1)} ({cold_drop_loss_ratio:.3f}) "
            f"gate_expert3_mean={gate_expert3_share:.4f}"
        )
        if debug_assertions:
            target_ratio = min(max(float(cold_start_dropout_assert_ratio), 0.0), 1.0)
            min_ratio = target_ratio * 0.8
            if cold_drop_seed_prot_total > 0 and cold_drop_selected_ratio + 1e-9 < min_ratio:
                msg = (
                    f"[ASSERT] episodic drop ratio too low: got {cold_drop_selected_ratio:.3f}, "
                    f"expect >= {min_ratio:.3f} (target={target_ratio:.3f})."
                )
                if kl_hard_assert:
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}")
            if cold_drop_present_total > 0 and cold_drop_zero_ratio < 0.95:
                msg = (
                    f"[ASSERT] dropped proteins not fully disconnected: zero_after_ratio={cold_drop_zero_ratio:.3f} (<0.95)."
                )
                if kl_hard_assert:
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}")
            if cold_drop_selected_total > 0 and cold_drop_loss_edge_dropped_total <= 0:
                msg = (
                    "[ASSERT] dropped proteins did not appear in BCE edges; check leakage/edge filtering path."
                )
                if kl_hard_assert:
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}")
            if gate_expert3_share <= 0.0:
                msg = (
                    "[ASSERT] gate expert-3 mean weight is zero under cold-start dropout; hard MoE routing not active."
                )
                if kl_hard_assert:
                    raise RuntimeError(msg)
                print(f"[WARN] {msg}")
    deg_d_pcts = _percentiles(deg_d_samples)
    deg_p_pcts = _percentiles(deg_p_samples)
    return (
        avg_loss,
        bce_loss,
        weighted_kl_loss,
        attn_kl_raw,
        attn_kl_norm,
        kl_ratio,
        kl_weight_eff,
        prior_conf_mean,
        epoch_time,
        had_valid_batch,
        b_eff_total,
        b_total,
        avg_atom_inc,
        avg_res_inc,
        avg_entropy,
        avg_mask_ratio,
        info_nce,
        distill_loss,
        kd_ratio,
        kd_conf,
        gate_balance,
        gate_entropy,
        total_weighted,
        gate_atom_mean,
        gate_res_mean,
        gate_w_mean,
        cold_edge_ratio,
        cold_drug_only_ratio,
        cold_prot_only_ratio,
        cold_both_ratio,
        wC_warm_mean,
        wC_cold_mean,
        mp_alpha_cold_mean,
        mp_alpha_warm_mean,
        mp_alpha_cold_p50,
        mp_alpha_cold_p90,
        mp_alpha_warm_p50,
        mp_alpha_warm_p90,
        gate_w_p10,
        gate_w_p50,
        gate_w_p90,
        deg_d_pcts,
        deg_p_pcts,
        t_sampling,
        t_encoder,
        t_heads,
        t_backward,
        t_ema,
    )

def train_one_epoch(epoch, fold, model, optimizer, scheduler, criterion,
                    features_drug_tensor, features_protein_tensor,
                    G_drug_tensor, G_protein_tensor,
                    train_edges_tensor, train_labels_tensor,
                    val_edges_tensor, val_labels_tensor,
                    train_losses, device, batch_size,
                    scaler=None, do_validation=True,
                    drug_node_to_entity=None, protein_node_to_entity=None,
                    drug_node_weight=None, protein_node_weight=None,
                    atom_prior=None, res_prior=None,
                    drug_degree=None, prot_degree=None, use_coldstart_gate=None,
                    num_workers=0, info_nce_on="protein",
                    ema_state=None, ema_decay=0.999, use_ema=False, ema_eval=False,
                    prot_esm_missing=None, prot_esm_unreliable=None,
                    drug_knn_edge_index=None, drug_knn_edge_weight=None,
                    prot_knn_edge_index=None, prot_knn_edge_weight=None,
                    cold_deg_th_drug=2, cold_deg_th_prot=2,
                    cold_prot_weight=1.0,
                    expertC_scale=1.0, wC_cap=None, seed_base=0,
                    eval_score_centering="none"):
    """Train one epoch on full-graph or fallback path."""
    t = time.time()
    model.train()
    total_loss = 0.0
    had_valid_batch = False
    batch_idx = 0
    t_encoder = 0.0
    t_heads = 0.0
    t_backward = 0.0
    t_ema = 0.0
    amp_enabled = (
        scaler is not None
        and scaler.is_enabled()
        and not (G_drug_tensor.is_sparse or G_protein_tensor.is_sparse)
    )
    train_dataset = TensorDataset(train_edges_tensor, train_labels_tensor)
    data_gen = torch.Generator(device="cpu")
    data_gen.manual_seed(int(seed_base) + int(epoch) * 1009 + int(fold) * 100003 + 29)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers) if num_workers is not None else 0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=data_gen,
    )

    for batch_edges, batch_labels in train_loader:
        batch_idx += 1
        optimizer.zero_grad()
        try:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                use_info_nce = bool(criterion.info_nce_weight and criterion.info_nce_weight > 0)
                if use_info_nce:
                    edge_logits, vae_params, pair_repr = model(
                        features_drug_tensor, features_protein_tensor, G_drug_tensor, G_protein_tensor,
                        drug_node_to_entity=drug_node_to_entity,
                        protein_node_to_entity=protein_node_to_entity,
                        drug_node_weight=drug_node_weight,
                        protein_node_weight=protein_node_weight,
                        atom_prior=atom_prior,
                        res_prior=res_prior,
                        edge_index=batch_edges,
                        drug_degree=drug_degree,
                        prot_degree=prot_degree,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing,
                        prot_esm_unreliable=prot_esm_unreliable,
                        drug_knn_edge_index=drug_knn_edge_index,
                        drug_knn_edge_weight=drug_knn_edge_weight,
                        prot_knn_edge_index=prot_knn_edge_index,
                        prot_knn_edge_weight=prot_knn_edge_weight,
                        cold_deg_th_drug=cold_deg_th_drug,
                        cold_deg_th_prot=cold_deg_th_prot,
                        expertC_scale=expertC_scale,
                        wC_cap=wC_cap,
                        return_pair_repr=True,
                    )
                else:
                    edge_logits, vae_params = model(
                        features_drug_tensor, features_protein_tensor, G_drug_tensor, G_protein_tensor,
                        drug_node_to_entity=drug_node_to_entity,
                        protein_node_to_entity=protein_node_to_entity,
                        drug_node_weight=drug_node_weight,
                        protein_node_weight=protein_node_weight,
                        atom_prior=atom_prior,
                        res_prior=res_prior,
                        edge_index=batch_edges,
                        drug_degree=drug_degree,
                        prot_degree=prot_degree,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing,
                        prot_esm_unreliable=prot_esm_unreliable,
                        drug_knn_edge_index=drug_knn_edge_index,
                        drug_knn_edge_weight=drug_knn_edge_weight,
                        prot_knn_edge_index=prot_knn_edge_index,
                        prot_knn_edge_weight=prot_knn_edge_weight,
                        cold_deg_th_drug=cold_deg_th_drug,
                        cold_deg_th_prot=cold_deg_th_prot,
                        expertC_scale=expertC_scale,
                        wC_cap=wC_cap,
                    )
                profile = getattr(model, "_last_profile", None)
                if profile:
                    t_encoder += float(profile.get("encoder", 0.0))
                    t_heads += float(profile.get("heads", 0.0))
                drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl_norm, attn_kl_raw, _, _ = unpack_vae_params(vae_params)
        except RuntimeError as e:
            print(f"Forward pass error: {e}, skip this batch")
            continue

        edge_labels = batch_labels.float()
        cold_edge_mask_batch = None
        cold_prot_mask_batch = None
        warm_edge_mask_batch = None
        if drug_degree is not None and prot_degree is not None and batch_edges.numel():
            deg_d_edge = drug_degree[batch_edges[:, 0]]
            deg_p_edge = prot_degree[batch_edges[:, 1]]
            cold_prot_mask_batch = deg_p_edge <= float(cold_deg_th_prot)
            cold_edge_mask_batch = (deg_d_edge <= float(cold_deg_th_drug)) | (deg_p_edge <= float(cold_deg_th_prot))
            warm_edge_mask_batch = ~cold_edge_mask_batch

        try:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pair_group = None
                pair_repr_in = None
                if use_info_nce:
                    if info_nce_on == "drug":
                        pair_group = batch_edges[:, 0]
                    else:
                        pair_group = batch_edges[:, 1]
                    pair_repr_in = pair_repr
                gate_weights = getattr(model, "_last_gate_weights", None)
                if gate_weights is not None and gate_weights.size(0) != edge_logits.numel():
                    gate_weights = None
                distill_logits = None
                distill_mask = None
                if getattr(criterion, "distill_weight", 0.0):
                    distill_logits = getattr(model, "_last_teacher_logits", None)
                    if distill_logits is not None and distill_logits.numel() != edge_logits.numel():
                        distill_logits = None
                    if distill_logits is not None and getattr(criterion, "distill_mode", "warm_only") == "warm_only":
                        if warm_edge_mask_batch is not None and warm_edge_mask_batch.numel() == edge_logits.numel():
                            distill_mask = warm_edge_mask_batch
                aux = getattr(model, "_last_aux", {}) if hasattr(model, "_last_aux") else {}
                prior_conf = None
                if aux:
                    prior_conf = {
                        "atom": aux.get("prior_conf_atom_mean", None),
                        "res": aux.get("prior_conf_res_mean", None),
                    }
                prior_conf_edge = aux.get("prior_conf_edge", None) if isinstance(aux, dict) else None
                if prior_conf is None or all(v is None for v in prior_conf.values()):
                    prior_conf = getattr(model, "_last_prior_conf", None)
                delta_reg = getattr(model, "_last_delta_reg", None)
                kl_stats = aux.get("kl_stats", None) if isinstance(aux, dict) else None
                if kl_stats is None and isinstance(aux, dict):
                    kl_stats = {
                        "kl_clip_count": int(aux.get("kl_clip_count", 0)),
                        "kl_nan_count": int(aux.get("kl_nan_count", 0)),
                        "prior_nan_count": int(aux.get("prior_nan_count", 0)),
                        "renorm_count": int(aux.get("renorm_count", 0)),
                    }
                sample_weight = None
                if cold_prot_mask_batch is not None and cold_prot_mask_batch.numel() == edge_logits.numel():
                    cpw = float(cold_prot_weight)
                    if cpw != 1.0:
                        sample_weight = torch.ones_like(edge_logits, dtype=edge_logits.dtype, device=edge_logits.device)
                        sample_weight[cold_prot_mask_batch] = cpw
                loss = criterion(
                    edge_logits, edge_labels, drug_mu, drug_logvar, prot_mu, prot_logvar,
                    sample_weight=sample_weight,
                    attn_kl=attn_kl_norm, attn_kl_raw=attn_kl_raw,
                    pair_repr=pair_repr_in, pair_group=pair_group,
                    gate_weights=gate_weights, distill_logits=distill_logits,
                    distill_mask=distill_mask, prior_conf=prior_conf,
                    prior_conf_edge=prior_conf_edge,
                    delta_reg=delta_reg, kl_stats=kl_stats
                )
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: loss is NaN or Inf")
                continue
        except RuntimeError as e:
            print(f"Loss error: {e}, skip this batch")
            continue

        t_back_start = time.time()
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        t_backward += time.time() - t_back_start
        if use_ema:
            t_ema_start = time.time()
            update_ema(ema_state, model, ema_decay)
            t_ema += time.time() - t_ema_start
        total_loss += loss.item()
        had_valid_batch = True
        print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.5f}", end="\r")
        sys.stdout.flush()

    avg_loss = total_loss / max(len(train_loader), 1)
    train_losses[fold].append(avg_loss)

    auc_val = None
    aupr_val = None
    cold_stats = {}
    if do_validation:
        model.eval()
        ema_backup = None
        if use_ema and ema_eval and ema_state is not None:
            ema_backup = swap_ema_weights(model, ema_state)
        with torch.no_grad():
            try:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    val_logits, vae_params = model(
                        features_drug_tensor, features_protein_tensor, G_drug_tensor, G_protein_tensor,
                        drug_node_to_entity=drug_node_to_entity,
                        protein_node_to_entity=protein_node_to_entity,
                        drug_node_weight=drug_node_weight,
                        protein_node_weight=protein_node_weight,
                        atom_prior=atom_prior,
                        res_prior=res_prior,
                        edge_index=val_edges_tensor,
                        drug_degree=drug_degree,
                        prot_degree=prot_degree,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing,
                        prot_esm_unreliable=prot_esm_unreliable,
                        drug_knn_edge_index=drug_knn_edge_index,
                        drug_knn_edge_weight=drug_knn_edge_weight,
                        prot_knn_edge_index=prot_knn_edge_index,
                        prot_knn_edge_weight=prot_knn_edge_weight,
                        cold_deg_th_drug=cold_deg_th_drug,
                        cold_deg_th_prot=cold_deg_th_prot,
                    )
                    _ = unpack_vae_params(vae_params)
                    labels_val = val_labels_tensor
            except RuntimeError as e:
                print(f"Validation forward error: {e}")
                auc_val, aupr_val = 0.0, 0.0
            else:
                if not torch.isfinite(val_logits).all() or not torch.isfinite(labels_val).all():
                    auc_val, aupr_val = 0.0, 0.0
                elif labels_val.unique().numel() < 2:
                    auc_val, aupr_val = 0.0, 0.0
                else:
                    score_val_np = val_logits.detach().cpu().numpy().reshape(-1)
                    labels_val_np = labels_val.detach().cpu().numpy()
                    edges_np = val_edges_tensor.detach().cpu().numpy()
                    n_eff = min(score_val_np.shape[0], labels_val_np.shape[0], edges_np.shape[0])
                    if n_eff < 2:
                        auc_val, aupr_val = 0.0, 0.0
                        cold_stats = {}
                    else:
                        if score_val_np.shape[0] != n_eff or labels_val_np.shape[0] != n_eff or edges_np.shape[0] != n_eff:
                            print("[WARN] Validation size mismatch; truncating scores/labels/edges to aligned length.")
                        score_val_np = score_val_np[:n_eff]
                        labels_val_np = labels_val_np[:n_eff]
                        edges_np = edges_np[:n_eff]
                        metric_variants = _compute_global_metric_variants(labels_val_np, score_val_np, edges_np[:, 1])
                        score_eval_np, selected_mode = _select_eval_scores(
                            score_val_np, edges_np[:, 1], eval_score_centering=eval_score_centering
                        )
                        auc_val, aupr_val = _safe_auc_aupr(labels_val_np, score_eval_np)
                        cold_stats = compute_cold_metrics(
                            edges_np, labels_val_np, score_eval_np,
                            drug_degree=drug_degree, prot_degree=prot_degree,
                            cold_deg_th_drug=cold_deg_th_drug, cold_deg_th_prot=cold_deg_th_prot,
                        )
                        missing_stats = compute_esm_missing_metrics(
                            edges_np, labels_val_np, score_eval_np,
                            prot_missing=prot_esm_missing,
                            prot_unreliable=prot_esm_unreliable,
                        )
                        if missing_stats:
                            cold_stats.update(missing_stats)
                        cold_stats = _attach_global_metrics(
                            cold_stats, metric_variants, selected_mode, auc_val, aupr_val
                        )
            finally:
                if ema_backup is not None:
                    restore_weights(model, ema_backup)

    if had_valid_batch:
        (bce_loss, weighted_kl_loss, attn_kl_raw, attn_kl_norm,
         kl_ratio, kl_weight_eff, prior_conf_mean, info_nce, info_nce_ratio,
         distill_loss, distill_raw, kd_ratio, kd_conf, gate_balance, gate_entropy,
         total_weighted) = criterion.get_components()
    else:
        bce_loss, weighted_kl_loss, attn_kl_raw, attn_kl_norm, kl_ratio = (0.0, 0.0, 0.0, 0.0, 0.0)
        kl_weight_eff = prior_conf_mean = 0.0
        info_nce = info_nce_ratio = distill_loss = distill_raw = kd_ratio = kd_conf = gate_balance = gate_entropy = 0.0
        total_weighted = 0.0
    info_weight = float(getattr(criterion, "info_nce_weight", 0.0))
    gate_bal_weight = float(getattr(criterion, "gate_balance_weight", 0.0))
    total_epoch_loss = float(total_weighted) if torch.is_tensor(total_weighted) else float(total_weighted)
    epoch_time = time.time() - t
    edge_rate = train_edges_tensor.shape[0] / max(epoch_time, 1e-9)
    auc_str = f"{auc_val:.5f}" if auc_val is not None else "N/A"
    aupr_str = f"{aupr_val:.5f}" if aupr_val is not None else "N/A"
    kl_scale = float(getattr(criterion, "_kl_scale_ratio", 1.0))
    kd_w_eff = float(getattr(criterion, "distill_weight", 0.0)) * float(kd_conf)
    aux_last = getattr(model, "_last_aux", {}) if hasattr(model, "_last_aux") else {}
    kl_enabled_log = float(getattr(criterion, "attn_kl_weight", 0.0)) > 0.0
    if kl_enabled_log and aux_last:
        attn_kl_norm_raw = float(aux_last.get("attn_kl_norm_raw", attn_kl_norm))
    else:
        attn_kl_norm_raw = float(attn_kl_norm)
    delta_rms = float(aux_last.get("delta_rms", 0.0)) if aux_last else 0.0
    delta_rms_weighted = float(aux_last.get("delta_rms_weighted", 0.0)) if aux_last else 0.0
    alpha_temp = float(getattr(model, "alpha_temp", 0.0))
    cold_ratio_drug = None
    cold_ratio_prot = None
    if drug_degree is not None and drug_degree.numel():
        cold_ratio_drug = float((drug_degree <= float(cold_deg_th_drug)).float().mean().item())
    if prot_degree is not None and prot_degree.numel():
        cold_ratio_prot = float((prot_degree <= float(cold_deg_th_prot)).float().mean().item())
    esm_missing_ratio = None
    esm_unreliable_ratio = None
    if prot_esm_missing is not None and train_edges_tensor is not None and train_edges_tensor.numel():
        esm_missing_ratio = float((prot_esm_missing[train_edges_tensor[:, 1]] > 0).float().mean().item())
    if prot_esm_unreliable is not None and train_edges_tensor is not None and train_edges_tensor.numel():
        esm_unreliable_ratio = float((prot_esm_unreliable[train_edges_tensor[:, 1]] > 0).float().mean().item())
    breakdown = None
    if had_valid_batch:
        try:
            breakdown = criterion.get_breakdown()
        except Exception:
            breakdown = None
    print(
        f"Epoch: {epoch + 1:04d}, loss: {avg_loss:.5f}, time: {epoch_time:.4f}s, "
        f"speed: {edge_rate:.1f} edges/s, "
        f"auc_val: {auc_str}, aupr_val: {aupr_str}, "
        f"bce_loss: {float(bce_loss):.5f}, kl_term: {float(weighted_kl_loss):.5f}, "
        f"attn_kl_raw: {float(attn_kl_raw):.5f}, "
        f"attn_kl_norm(raw/clip): {attn_kl_norm_raw:.5f}/{float(attn_kl_norm):.5f}, "
        f"kl_ratio: {float(kl_ratio):.3f}, kl_w_eff: {float(kl_weight_eff):.4f}, "
        f"kl_scale: {kl_scale:.3f}, prior_conf: {float(prior_conf_mean):.3f}, "
        f"info_nce: {float(info_nce):.5f}, "
        f"alpha_temp: {alpha_temp:.3f}, gate_bal_w: {gate_bal_weight:.3f}, "
        f"distill: {float(distill_loss):.5f}, kd_ratio: {float(kd_ratio):.3f}, kd_conf: {float(kd_conf):.3f}, "
        f"kd_w_eff: {kd_w_eff:.4f}, "
        f"gate_bal: {float(gate_balance):.5f}, gate_ent: {float(gate_entropy):.5f}, "
        f"delta_rms: {delta_rms:.4f}, delta_rms_w: {delta_rms_weighted:.4f}, "
        f"total_loss: {float(total_epoch_loss):.5f}"
    )
    if cold_ratio_drug is not None or cold_ratio_prot is not None:
        crd = f"{cold_ratio_drug:.3f}" if cold_ratio_drug is not None else "N/A"
        crp = f"{cold_ratio_prot:.3f}" if cold_ratio_prot is not None else "N/A"
        print(f"[COLD-NODE] drug={crd} prot={crp}")
        if (cold_ratio_drug is not None and cold_ratio_drug > 0.35) or (
            cold_ratio_prot is not None and cold_ratio_prot > 0.35
        ):
            print("[WARN] cold_node_ratio > 0.35; consider adjusting cold thresholds or quantiles.")
    if esm_missing_ratio is not None or esm_unreliable_ratio is not None:
        em_str = f"{esm_missing_ratio:.4f}" if esm_missing_ratio is not None else "N/A"
        eu_str = f"{esm_unreliable_ratio:.4f}" if esm_unreliable_ratio is not None else "N/A"
        print(f"[DATA] esm_missing_ratio={em_str} esm_unreliable_ratio={eu_str}")
    if breakdown is not None:
        print(
            f"[LOSS] bce={float(breakdown['bce_raw']):.5f}/{float(breakdown['bce_weighted']):.5f} "
            f"kl={float(breakdown['kl_raw']):.5f}/{float(breakdown['kl_weighted']):.5f} "
            f"kd={float(breakdown['distill_raw']):.5f}/{float(breakdown['distill_weighted']):.5f} "
            f"info={float(breakdown['info_nce_raw']):.5f}/{float(breakdown['info_nce_weighted']):.5f} "
            f"gate={float(breakdown['gate_balance_raw']):.5f}/{float(breakdown['gate_balance_weighted']):.5f} "
            f"ent={float(breakdown['gate_entropy_raw']):.5f}/{float(breakdown['gate_entropy_weighted']):.5f} "
            f"delta={float(breakdown['delta_reg_raw']):.5f}/{float(breakdown['delta_reg_weighted']):.5f} "
            f"total={float(breakdown['total']):.5f}"
        )

    if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    return auc_val, aupr_val, do_validation, {
        "info_nce": float(info_nce),
        "distill": float(distill_loss),
        "gate_balance": float(gate_balance),
        "gate_entropy": float(gate_entropy),
        "epoch_time": float(epoch_time),
        "t_encoder": t_encoder,
        "t_heads": t_heads,
        "t_backward": t_backward,
        "t_ema": t_ema,
        "cold_stats": cold_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("--seed_list", type=str, default="",
                        help="Comma-separated seeds for multi-run (optional).")
    parser.add_argument("--seeds", type=str, default="",
                        help="Alias for --seed_list.")
    parser.add_argument("--metrics_out", type=str, default="",
                        help=argparse.SUPPRESS)
    parser.add_argument("--child_run", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--resume", type=str, default=None, help="载入模型权重路径")
    parser.add_argument("--start_epoch", type=int, default=1, help="起始轮次")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dim.")
    parser.add_argument("--out_dim", type=int, default=128, help="Latent dim.")
    parser.add_argument("--num_fold", type=int, default=5, help="Number of folds.")
    parser.add_argument("--eval_mode", type=str, default="fixed", choices=["fixed", "cv"],
                        help="Evaluation protocol: fixed val/test or CV on train.")
    parser.add_argument("--allow_atomic_cv_leakage", action="store_true",
                        help="Allow eval_mode=cv in atomic mode (unsafe: graph leakage risk).")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size (0 = use mode default).")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Validation interval (epochs).")
    parser.add_argument("--seed_edge_batch_size", type=int, default=4096,
                        help="Seed DTI edges per step for subgraph training.")
    parser.add_argument("--max_atoms_per_step", type=int, default=512,
                        help="Max atom nodes per step (subgraph budget).")
    parser.add_argument("--max_residues_per_step", type=int, default=1024,
                        help="Max residue nodes per step (subgraph budget).")
    parser.add_argument("--rw_walk_length", type=int, default=2,
                        help="Random walk length for subgraph sampling.")
    parser.add_argument("--rw_num_walks", type=int, default=4,
                        help="Random walks per seed node.")
    parser.add_argument("--saint_steps", type=int, default=0,
                        help="Steps per epoch for subgraph training (0 = auto).")
    parser.add_argument("--saint_min_steps", type=int, default=0,
                        help="Fallback GraphSAINT steps when --saint_steps=0 (0 = auto natural steps).")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log interval (steps) for subgraph training.")
    parser.add_argument("--saint_reweight", type=int, nargs="?", const=1, default=1,
                        help="Enable DTI edge reweighting for subgraph training (0/1).")
    parser.add_argument("--gat_top_k_sparse", type=int, default=0,
                        help="Override GAT top-k for sparse neighbors (0 = use mode default).")
    parser.add_argument("--gat_dense_top_k", type=int, default=0,
                        help="Override GAT top-k for dense neighbors (0 = use mode default).")
    parser.add_argument("--interaction_head", type=str, default="mlp", choices=["dot", "bilinear", "mlp"],
                        help="Edge interaction head (mlp kept as backward-compatible alias to dot).")
    parser.add_argument("--use_hyperedge_head", action="store_true",
                        help="Enable hyperedge head (default: atomic True).")
    parser.add_argument("--no_hyperedge_head", action="store_true",
                        help="Disable hyperedge head.")
    parser.add_argument("--alpha_refine", action="store_true",
                        help="Enable attention refinement (default: atomic True).")
    parser.add_argument("--no_alpha_refine", action="store_true",
                        help="Disable attention refinement.")
    parser.add_argument("--alpha_eps", type=float, default=1e-6,
                        help="Epsilon for attention normalization.")
    parser.add_argument("--prior_eps", type=float, default=1e-4,
                        help="Epsilon for prior probability clamp in KL.")
    parser.add_argument("--alpha_temp", type=float, default=1.3,
                        help="Softmax temperature for attention refinement.")
    parser.add_argument("--edge_min_incidence_atom", type=int, default=1,
                        help="Min atom incidence per DTI edge.")
    parser.add_argument("--edge_min_incidence_res", type=int, default=1,
                        help="Min residue incidence per DTI edge.")
    parser.add_argument("--eval_num_samples", type=int, default=5,
                        help="Number of subgraph samples to average at eval.")
    parser.add_argument("--eval_coverage_min", type=float, default=0.90,
                        help="Minimum subgraph eval coverage required for model selection.")
    parser.add_argument("--eval_score_centering", type=str, default="none",
                        choices=["none", "per_protein_mean"],
                        help="Global eval score transform before AUC/AUPR.")
    parser.add_argument("--eval_warm_support_k", type=int, default=0,
                        help="Eval-only: per seed protein, add up to k warm kNN neighbors in subgraph eval (0 disables).")
    parser.add_argument("--eval_warm_support_max_add", type=int, default=256,
                        help="Eval-only: cap added warm proteins per eval batch when --eval_warm_support_k > 0.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers for non-GraphSAINT training.")
    parser.add_argument("--dataset", type=str, default="drugbank", help="Dataset name.")
    parser.add_argument("--kl_weight", type=float, default=0.00, help="KL loss weight.")
    parser.add_argument("--attn_kl_weight", type=float, default=0.02,
                        help="Attention KL weight (legacy base; use schedule args).")
    parser.add_argument("--attn_kl_weight_schedule", type=str, default="sigmoid",
                        choices=["sigmoid", "cosine", "linear"],
                        help="Schedule for attention KL weight.")
    parser.add_argument("--kl_stage1_epochs", type=int, default=5,
                        help="Epochs to keep KL at min weight.")
    parser.add_argument("--attn_kl_w_min", type=float, default=0.005,
                        help="Min KL weight for warm stage.")
    parser.add_argument("--attn_kl_w_max", type=float, default=0.05,
                        help="Max KL weight after ramp.")
    parser.add_argument("--attn_kl_ramp_epochs", type=int, default=10,
                        help="Ramp epochs from w_min to w_max.")
    parser.add_argument("--attn_kl_max_ratio", type=float, default=0.2,
                        help="Max ratio of (attn_kl_term / bce_loss).")
    parser.add_argument("--attn_kl_clip", type=float, default=2.0,
                        help="Clip attn_kl_norm to this value (no abort).")
    parser.add_argument("--attn_kl_warmup_epochs", type=int, default=0,
                        help="[Deprecated] Warmup epochs for attn_kl_weight.")
    parser.add_argument("--prior_conf_ref", type=float, default=0.2,
                        help="Reference prior confidence for KL scaling.")
    parser.add_argument("--prior_smoothing", type=float, default=0.05,
                        help="Prior smoothing factor for attention (0 disables).")
    parser.add_argument("--prior_mix_mode", type=str, default="mixture",
                        choices=["additive", "mixture"],
                        help="Prior mixing mode for attention refine.")
    parser.add_argument("--prior_mix_lambda", type=float, default=0.3,
                        help="Mixing weight for prior when prior_mix_mode=mixture.")
    parser.add_argument("--prior_mix_learnable", action="store_true",
                        help="Learn a global prior_mix_lambda when not conditional.")
    parser.add_argument("--prior_mix_conditional", action="store_true",
                        help="Condition prior_mix_lambda on edge features.")
    parser.add_argument("--prior_mix_features", type=str, default="deg_drug,deg_prot,prior_entropy,attn_entropy,esm_missing,esm_unreliable",
                        help="Comma-separated feature names for conditional prior mix.")
    parser.add_argument("--moe_enable", type=int, nargs="?", const=1, default=1,
                        help="Enable head-level MoE (0/1).")
    parser.add_argument("--expert_A", type=int, nargs="?", const=1, default=1,
                        help="Enable Expert A (prior refine).")
    parser.add_argument("--expert_B", type=int, nargs="?", const=1, default=1,
                        help="Enable Expert B (robust learned).")
    parser.add_argument("--expert_C", type=int, nargs="?", const=1, default=1,
                        help="Enable Expert C (cold retrieval).")
    parser.add_argument("--expertC_warmup_epochs", type=int, default=2,
                        help="Warmup epochs for Expert C gate (0->1).")
    parser.add_argument("--wC_cap", type=float, default=0.15,
                        help="Cap for Expert C gate weight during training/eval.")
    parser.add_argument("--cold_zero_route_mode", type=str, default="hard",
                        choices=["hard", "soft", "off"],
                        help="Routing policy on degree==0 cold edges: hard one-hot C, soft C-lower-bound, or off.")
    parser.add_argument("--cold_zero_route_min_wc", type=float, default=1.0,
                        help="When --cold_zero_route_mode=soft, enforce Expert C weight >= this value before renorm.")
    parser.add_argument("--pool_topk", type=int, default=16,
                        help="TopK for head-level pooling.")
    parser.add_argument("--pool_randk", type=int, default=16,
                        help="RandK for head-level pooling.")
    parser.add_argument("--topk", type=int, default=None,
                        help="Alias for --pool_topk.")
    parser.add_argument("--randk", type=int, default=None,
                        help="Alias for --pool_randk.")
    parser.add_argument("--beta_mix", type=float, default=0.7,
                        help="Mixing weight for topK vs randK pooling.")
    parser.add_argument("--no_bottleneck_gate", action="store_true",
                        help="Disable bottleneck gating after fusion.")
    parser.add_argument("--bottleneck_drop", type=float, default=0.3,
                        help="Dropout used in bottleneck gate.")
    parser.add_argument("--mp_gate_mode", type=str, default="node",
                        choices=["none", "node", "edge"],
                        help="Message passing gate mode (node/edge/none).")
    parser.add_argument("--mp_gate_deg_only", action="store_true",
                        help="Use degree-only features for MP gate.")
    parser.add_argument("--mp_gate_init_bias", type=float, default=0.0,
                        help="Initial bias for MP gate logits.")
    parser.add_argument("--mp_gate_use_attn_entropy", type=int, nargs="?", const=1, default=1,
                        help="Use attention entropy in MP gate features (0/1).")
    parser.add_argument("--mp_gate_use_prior_entropy", type=int, nargs="?", const=1, default=1,
                        help="Use prior entropy in MP gate features (0/1).")
    parser.add_argument("--mp_gate_use_esm_missing", type=int, nargs="?", const=1, default=1,
                        help="Use ESM missing flag in MP gate features (0/1).")
    parser.add_argument("--mp_gate_use_prior_conf", type=int, nargs="?", const=1, default=1,
                        help="Use prior confidence in MP gate features (0/1).")
    parser.add_argument("--mp_gate_cold_scale", type=float, default=0.3,
                        help="Scale MP gate alpha on cold nodes (0 disables MP on cold).")
    parser.add_argument("--info_nce_weight", type=float, default=0.0,
                        help="Weight for lightweight InfoNCE loss (0 disables).")
    parser.add_argument("--info_nce_temp", type=float, default=0.05,
                        help="Temperature for InfoNCE.")
    parser.add_argument("--info_nce_neg_k", type=int, default=64,
                        help="Number of negatives per anchor for InfoNCE.")
    parser.add_argument("--info_nce_on", type=str, default="protein",
                        choices=["protein", "drug"],
                        help="Group definition for InfoNCE positives.")
    parser.add_argument("--disable_info_nce", action="store_true",
                        help="Force disable InfoNCE (overrides weight).")
    parser.add_argument("--info_nce_enable", type=int, nargs="?", const=1, default=0,
                        help="Enable InfoNCE loss (0/1).")
    parser.add_argument("--info_nce_max_ratio", type=float, default=0.05,
                        help="Max InfoNCE/BCE ratio (weighted).")
    parser.add_argument("--gate_balance_weight", type=float, default=0.01,
                        help="MoE gate load-balancing regularization weight.")
    parser.add_argument("--gate_entropy_weight", type=float, default=0.001,
                        help="MoE gate entropy regularization weight.")
    parser.add_argument("--delta_reg_weight", type=float, default=0.001,
                        help="L2 regularization weight for attention refine logits (delta).")
    parser.add_argument("--distill_enable", type=int, nargs="?", const=1, default=1,
                        help="Enable in-model distillation (0/1).")
    parser.add_argument("--distill_start_epoch", type=int, default=5,
                        help="Start epoch for distillation (>=).")
    parser.add_argument("--distill_mode", type=str, default="warm_only",
                        choices=["warm_only", "all"],
                        help="Distillation mode (warm_only/all).")
    parser.add_argument("--distill_T", type=float, default=3.0,
                        help="Distillation temperature.")
    parser.add_argument("--distill_weight", type=float, default=0.05,
                        help="Distillation loss weight.")
    parser.add_argument("--distill_max_ratio", type=float, default=None,
                        help="Alias for --kd_max_ratio.")
    parser.add_argument("--kd_max_ratio", type=float, default=0.15,
                        help="Max KD/BCE ratio (applied to weighted KD).")
    parser.add_argument("--kd_warmup_epochs", type=int, default=3,
                        help="Warmup epochs for KD weight.")
    parser.add_argument("--distill_warmup_epochs", type=int, default=3,
                        help="Warmup epochs for KD weight (alias, overrides kd_warmup_epochs).")
    parser.add_argument("--vc_enable", type=int, nargs="?", const=1, default=0,
                        help="Enable VC (default off).")
    parser.add_argument("--dg_enable", type=int, nargs="?", const=1, default=0,
                        help="Enable domain generalization (default off).")
    parser.add_argument("--subpocket_mask_ratio", type=float, default=0.0,
                        help="Mask ratio for subpocket masking on residues (0 disables).")
    parser.add_argument("--subpocket_keep_top", type=float, default=0.2,
                        help="Keep top prior fraction when masking residues.")
    parser.add_argument("--protein_feat_mode", type=str, default="concat",
                        choices=["onehot", "esm2", "concat"],
                        help="Protein feature mode for residues.")
    parser.add_argument("--esm_special_tokens", type=str, default="auto",
                        choices=["auto", "bos_only", "eos_only", "none", "bos_eos", "bos", "eos"],
                        help="How to handle ESM special tokens when aligning embeddings.")
    parser.add_argument("--esm_norm", type=str, default="per_protein_zscore",
                        choices=["none", "per_protein_zscore", "per_dim_global"],
                        help="ESM normalization mode.")
    parser.add_argument("--esm_fallback", type=str, default="onehot_only",
                        choices=["onehot_only", "zeros", "uniform_noise", "mean"],
                        help="Fallback strategy when ESM is missing or mismatched.")
    parser.add_argument("--atom_topk", type=int, default=0,
                        help="Override atom top-k incidence (0 = use env/default).")
    parser.add_argument("--res_topk", type=int, default=0,
                        help="Override residue top-k incidence (0 = use env/default).")
    parser.add_argument("--atom_randk", type=int, default=0,
                        help="Random supplement count per drug hyperedge.")
    parser.add_argument("--res_randk", type=int, default=0,
                        help="Random supplement count per protein hyperedge.")
    parser.add_argument("--randk_seed", type=int, default=None,
                        help="Seed for rand-k sampling (default: seed).")
    parser.add_argument("--randk_weight_mode", type=str, default="floor_prior",
                        choices=["prior", "uniform", "floor_prior"],
                        help="Weighting for rand-k incidence nodes.")
    parser.add_argument("--prior_floor", type=float, default=1e-4,
                        help="Floor for prior weights in rand-k sampling.")
    parser.add_argument("--esm_strict", action="store_true",
                        help="Strict ESM alignment (raise on mismatch).")
    parser.add_argument("--no_physchem_feat", action="store_true",
                        help="Disable residue physchem features in concat mode.")
    parser.add_argument("--reuse_cache", action="store_true",
                        help="Reuse cached preprocessing even if meta mismatches.")
    parser.add_argument("--use_knn_graph", type=int, nargs="?", const=1, default=1,
                        help="Enable kNN graph for cold-start (0/1).")
    parser.add_argument("--knn_setting", type=str, default="inductive",
                        choices=["inductive", "transductive"],
                        help="kNN setting (inductive/transductive).")
    parser.add_argument("--cold_deg_th_drug", type=int, default=3,
                        help="Cold threshold for drug degree.")
    parser.add_argument("--cold_deg_th_prot", type=int, default=3,
                        help="Cold threshold for protein degree.")
    parser.add_argument("--cold_prot_weight", type=float, default=1.0,
                        help="Weight multiplier for cold protein edges loss")
    parser.add_argument("--cold_start_dropout_p", type=float, default=0.0,
                        help="Episodic protein edge-drop probability in message passing graph.")
    parser.add_argument("--cold_start_dropout_assert_ratio", type=float, default=0.1,
                        help="Expected dropped-protein ratio target for epoch-level assertions.")
    parser.add_argument("--cold_mode", type=str, default="th",
                        choices=["th", "quantile"],
                        help="Cold selection mode: threshold or quantile.")
    parser.add_argument("--cold_q", type=float, default=0.1,
                        help="Quantile for cold_mode=quantile.")
    parser.add_argument("--cold_quantile_drug", type=float, default=None,
                        help="Drug quantile for cold_mode=quantile (overrides cold_q).")
    parser.add_argument("--cold_quantile_prot", type=float, default=None,
                        help="Protein quantile for cold_mode=quantile (overrides cold_q).")
    parser.add_argument("--cold_q_drug", dest="cold_quantile_drug", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--cold_q_prot", dest="cold_quantile_prot", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--use_global_degree_for_cold", type=int, nargs="?", const=1, default=1,
                        help="Use precomputed global train degrees for cold gating (0/1).")
    parser.add_argument("--drug_knn_k", type=int, default=20,
                        help="kNN k for drugs.")
    parser.add_argument("--prot_knn_k", type=int, default=20,
                        help="kNN k for proteins.")
    parser.add_argument("--knn_metric", type=str, default="cosine",
                        choices=["cosine"],
                        help="kNN metric.")
    parser.add_argument("--knn_symmetric", type=int, nargs="?", const=1, default=1,
                        help="Make kNN edges symmetric (0/1).")
    parser.add_argument("--knn_weight_temp", type=float, default=0.1,
                        help="Temperature for kNN edge weights.")
    parser.add_argument("--epoch_time_budget_sec", type=int, default=800,
                        help="Epoch time budget in seconds (warn if exceeded).")
    parser.add_argument("--debug_assertions", type=int, nargs="?", const=1, default=1,
                        help="Enable debug assertions/warnings in early epochs (0/1).")
    parser.add_argument("--kl_hard_assert", type=int, nargs="?", const=1, default=0,
                        help="Hard assert on KL diagnostic failures (0/1).")
    parser.add_argument("--prune_strategy_tag", type=str, default="train_avg_topk",
                        help="Tag recorded in preprocessing meta for pruning strategy.")
    parser.add_argument("--node_level", type=str, default="atomic", choices=["atomic"],
                        help="Use atomic-level (atom/residue) nodes.")
    parser.add_argument("--psichic_attention", type=str, default="",
                        help="Path to PSICHIC attention pickle for atomic mode.")
    parser.add_argument("--no_self_loop", action="store_true",
                        help="Disable self-loop in graph normalization.")
    parser.add_argument("--no_residual", action="store_true",
                        help="Disable initial residual connections.")
    parser.add_argument("--no_coldstart_gate", action="store_true",
                        help="Disable cold-start feature-only gating.")
    parser.add_argument("--allow_zero_features", action="store_true",
                        help="Allow zero feature rows without raising an error.")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅进行评估，不执行训练循环")
    # Explain-only flags (no training overhead)
    parser.add_argument("--explain_enable", action="store_true",
                        help="Enable explain-mode outputs (inference only).")
    parser.add_argument("--explain_methods", type=str, default="attn",
                        help="Explain methods: attn,grad,ig,occlusion (comma-separated).")
    parser.add_argument("--explain_ig_steps", type=int, default=16,
                        help="Integrated gradients steps (explain only).")
    parser.add_argument("--explain_topk", type=int, default=20,
                        help="Top-k atoms/residues to export (explain only).")
    parser.add_argument("--explain_num_samples", type=int, default=64,
                        help="Number of samples for explain summaries (eval only).")
    parser.add_argument("--explain_out_dir", type=str, default="runs/explain",
                        help="Output directory for explain artifacts.")
    parser.add_argument("--explain_aux_to_cpu", action="store_true",
                        help="Detach aux tensors to CPU for explain (inference only).")
    parser.add_argument("--export_rdkit_png", action="store_true",
                        help="Export RDKit atom-level PNG in explain mode.")
    parser.add_argument("--export_pdb_bfactor", action="store_true",
                        help="Export PDB B-factor for residue scores in explain mode.")
    parser.add_argument("--export_html_report", action="store_true",
                        help="Export HTML explain report (explain only).")
    parser.add_argument("--use_ema", type=int, nargs="?", const=1, default=1,
                        help="Enable EMA of model weights (0/1).")
    parser.add_argument("--ema_decay", type=float, default=0.9995,
                        help="EMA decay.")
    parser.add_argument("--ema_eval", type=int, nargs="?", const=1, default=1,
                        help="Use EMA weights for evaluation (0/1).")
    parser.add_argument("--lr_scheduler", type=str, default="plateau",
                        choices=["none", "plateau", "cosine"],
                        help="LR scheduler type.")
    parser.add_argument("--plateau_patience", type=int, default=5,
                        help="ReduceLROnPlateau patience.")
    parser.add_argument("--plateau_factor", type=float, default=0.5,
                        help="ReduceLROnPlateau factor.")
    parser.add_argument("--plateau_min_lr", type=float, default=1e-6,
                        help="ReduceLROnPlateau minimum lr.")
    args = parser.parse_args()

    seed_str = args.seeds if args.seeds else args.seed_list
    if seed_str and not args.child_run:
        seeds = [int(s.strip()) for s in str(seed_str).split(",") if s.strip()]
        if not seeds:
            seeds = [int(args.seed)]
        base_args = []
        skip_next = False
        for arg in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg in ("--seed", "--seed_list", "--seeds", "--metrics_out", "--child_run"):
                if arg in ("--seed", "--seed_list", "--seeds", "--metrics_out"):
                    skip_next = True
                continue
            base_args.append(arg)
        script_path = os.path.abspath(__file__)
        metrics_all = []
        for seed in seeds:
            metrics_path = f"metrics_seed{seed}.json"
            cmd = [sys.executable, script_path] + base_args + [
                "--seed", str(seed),
                "--child_run",
                "--metrics_out", metrics_path,
            ]
            print(f"[INFO] Multi-seed run: seed={seed}")
            subprocess.run(cmd, check=True)
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics_all.append(json.load(f))
                except Exception as e:
                    print(f"[WARN] Failed to read metrics file {metrics_path}: {e}")
        if metrics_all:
            auc_vals = [float(m.get("mean_auc", 0.0)) for m in metrics_all]
            aupr_vals = [float(m.get("mean_aupr", 0.0)) for m in metrics_all]
            test_auc_vals = [float(m.get("test_auc", 0.0)) for m in metrics_all if m.get("test_auc") is not None]
            test_aupr_vals = [float(m.get("test_aupr", 0.0)) for m in metrics_all if m.get("test_aupr") is not None]
            print(
                f"[SEEDS] mean_auc={np.mean(auc_vals):.5f} +/- {np.std(auc_vals):.5f}, "
                f"mean_aupr={np.mean(aupr_vals):.5f} +/- {np.std(aupr_vals):.5f}"
            )
            if test_auc_vals and test_aupr_vals:
                print(
                    f"[SEEDS-TEST] auc={np.mean(test_auc_vals):.5f} +/- {np.std(test_auc_vals):.5f}, "
                    f"aupr={np.mean(test_aupr_vals):.5f} +/- {np.std(test_aupr_vals):.5f}"
                )
            summary_payload = {
                "seeds": seeds,
                "runs": metrics_all,
                "mean_auc": float(np.mean(auc_vals)),
                "std_auc": float(np.std(auc_vals)),
                "mean_aupr": float(np.mean(aupr_vals)),
                "std_aupr": float(np.std(aupr_vals)),
                "mean_test_auc": float(np.mean(test_auc_vals)) if test_auc_vals else None,
                "std_test_auc": float(np.std(test_auc_vals)) if test_auc_vals else None,
                "mean_test_aupr": float(np.mean(test_aupr_vals)) if test_aupr_vals else None,
                "std_test_aupr": float(np.std(test_aupr_vals)) if test_aupr_vals else None,
            }
            write_metrics_out("metrics_seeds_summary.json", summary_payload)
        else:
            print("[WARN] No metrics collected from multi-seed runs.")
        sys.exit(0)

    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Device: {device}")

    if args.topk is not None:
        args.pool_topk = int(args.topk)
    if args.randk is not None:
        args.pool_randk = int(args.randk)
    if args.distill_max_ratio is not None:
        args.kd_max_ratio = float(args.distill_max_ratio)
    if args.distill_warmup_epochs is not None:
        args.kd_warmup_epochs = int(args.distill_warmup_epochs)

    set_seed(args.seed)

    data_root = os.path.join(os.getcwd(), "data")
    is_atomic = True
    use_self_loop = not args.no_self_loop
    use_residual = not args.no_residual
    use_coldstart_gate = not args.no_coldstart_gate
    use_physchem_feat = not args.no_physchem_feat
    use_bottleneck_gate = not args.no_bottleneck_gate
    default_batch_size = 8
    gat_top_k_sparse = 4
    gat_dense_top_k = 32
    if int(args.hidden) <= 0 or int(args.out_dim) <= 0:
        raise ValueError(f"hidden/out_dim must be positive, got hidden={args.hidden}, out_dim={args.out_dim}")
    if args.gat_top_k_sparse and args.gat_top_k_sparse > 0:
        gat_top_k_sparse = int(args.gat_top_k_sparse)
    if args.gat_dense_top_k and args.gat_dense_top_k > 0:
        gat_dense_top_k = int(args.gat_dense_top_k)
    if args.no_hyperedge_head:
        use_hyperedge_head = False
    elif args.use_hyperedge_head:
        use_hyperedge_head = True
    else:
        use_hyperedge_head = is_atomic
    if args.no_alpha_refine:
        alpha_refine = False
    elif args.alpha_refine:
        alpha_refine = True
    else:
        alpha_refine = is_atomic
    args.alpha_eps = max(float(args.alpha_eps), 1e-8)
    args.prior_eps = float(args.prior_eps)
    if (not np.isfinite(args.prior_eps)) or args.prior_eps <= 0.0:
        raise ValueError(f"--prior_eps must be finite and > 0, got {args.prior_eps}")
    args.alpha_temp = max(float(args.alpha_temp), 1e-3)
    args.eval_coverage_min = max(0.0, min(1.0, float(args.eval_coverage_min)))
    args.eval_score_centering = str(args.eval_score_centering).strip().lower()
    args.eval_warm_support_k = max(0, int(args.eval_warm_support_k))
    args.eval_warm_support_max_add = max(0, int(args.eval_warm_support_max_add))
    args.cold_zero_route_mode = str(args.cold_zero_route_mode or "hard").strip().lower()
    args.cold_zero_route_min_wc = min(max(float(args.cold_zero_route_min_wc), 0.0), 1.0)
    if args.randk_seed is None:
        args.randk_seed = int(args.seed)
    cold_mode_cli = any(str(a).startswith("--cold_mode") for a in sys.argv[1:])
    cold_q_cli = any(str(a).startswith("--cold_q") for a in sys.argv[1:])
    cold_q_drug_cli = any(str(a).startswith("--cold_quantile_drug") or str(a).startswith("--cold_q_drug") for a in sys.argv[1:])
    cold_q_prot_cli = any(str(a).startswith("--cold_quantile_prot") or str(a).startswith("--cold_q_prot") for a in sys.argv[1:])
    if (not cold_mode_cli) and ("gpcr" in str(args.dataset).lower()):
        args.cold_mode = "quantile"
        if not (cold_q_cli or cold_q_drug_cli or cold_q_prot_cli):
            args.cold_quantile_drug = 0.05
            args.cold_quantile_prot = 0.0
        print(
            f"[INFO] gpcr default: cold_mode={args.cold_mode}, "
            f"cold_quantile_drug={args.cold_quantile_drug}, cold_quantile_prot={args.cold_quantile_prot}"
        )
    if args.disable_info_nce:
        args.info_nce_weight = 0.0
        print("[INFO] InfoNCE disabled by --disable_info_nce.")
    if not bool(args.info_nce_enable):
        if args.info_nce_weight and args.info_nce_weight > 0:
            print("[INFO] InfoNCE disabled by --info_nce_enable=0.")
        args.info_nce_weight = 0.0
    if float(args.attn_kl_weight) <= 0.0 and int(args.kl_stage1_epochs) > 0:
        print(
            "[INFO] attn_kl_weight<=0: force kl_stage1_epochs=0 "
            "(disable stage-1 delta freeze)."
        )
        args.kl_stage1_epochs = 0
    cli_flags = {
        str(token).split("=", 1)[0]
        for token in sys.argv[1:]
        if str(token).startswith("--")
    }
    pinned_sched = [
        flag for flag in ("--alpha_temp", "--gate_balance_weight", "--info_nce_weight")
        if flag in cli_flags
    ]
    if pinned_sched:
        print(
            "[INFO] Auto-scheduler notice: explicit CLI flags keep these fixed each epoch: "
            + ", ".join(pinned_sched)
        )
    prior_mix_features = [s.strip() for s in str(args.prior_mix_features).split(",") if s.strip()]
    use_knn_graph = bool(args.use_knn_graph)
    knn_setting = args.knn_setting
    knn_symmetric = bool(args.knn_symmetric)
    use_global_degree_for_cold = bool(args.use_global_degree_for_cold)
    debug_assertions = bool(args.debug_assertions)
    kl_hard_assert = bool(args.kl_hard_assert)
    use_ema = bool(args.use_ema)
    ema_eval = bool(args.ema_eval)
    moe_enable = bool(args.moe_enable)
    expert_A = bool(args.expert_A)
    expert_B = bool(args.expert_B)
    expert_C = bool(args.expert_C)
    distill_enable = bool(args.distill_enable)
    batch_size = args.batch_size if args.batch_size and args.batch_size > 0 else default_batch_size
    val_interval = max(1, int(args.val_interval))
    use_graphsaint = True
    seed_edge_batch_size = max(1, int(args.seed_edge_batch_size))
    max_atoms_per_step = max(1, int(args.max_atoms_per_step))
    max_res_per_step = max(1, int(args.max_residues_per_step))
    walk_length = max(1, int(args.rw_walk_length))
    num_walks = max(1, int(args.rw_num_walks))
    log_interval = max(1, int(args.log_interval))
    print("\n" + "=" * 72)
    print("[CONFIGURATION CHECK]")
    print(
        f"hidden={args.hidden} | seed_edge_batch_size={seed_edge_batch_size} | "
        f"batch_size={batch_size} | node_level={args.node_level}"
    )
    print("=" * 72)
    edge_min_incidence_atom = max(1, int(args.edge_min_incidence_atom))
    edge_min_incidence_res = max(1, int(args.edge_min_incidence_res))
    eval_num_samples = max(1, int(args.eval_num_samples))
    use_reweight = bool(args.saint_reweight)
    if use_graphsaint and not use_reweight:
        print("[WARN] GraphSAINT reweight is disabled; sampling may be biased.")
    atom_topk_env = os.environ.get("HGACN_ATOM_TOPK", "").strip()
    res_topk_env = os.environ.get("HGACN_RES_TOPK", "").strip()
    atom_keep_env = os.environ.get("HGACN_ATOM_KEEP", "").strip()
    res_keep_env = os.environ.get("HGACN_RES_KEEP", "").strip()
    prune_fallback_env = os.environ.get("HGACN_PRUNE_FALLBACK", "").strip()
    try:
        atom_keep_ref = int(atom_keep_env) if atom_keep_env else 0
    except Exception:
        atom_keep_ref = 0
    try:
        res_keep_ref = int(res_keep_env) if res_keep_env else 0
    except Exception:
        res_keep_ref = 0
    if atom_topk_env or res_topk_env:
        print(f"[INFO] Atomic top-k: atom={atom_topk_env or 'unset'}, residue={res_topk_env or 'unset'}")
    if atom_keep_env or res_keep_env:
        print(
            f"[INFO] Atomic keep: atom={atom_keep_env or 'unset'}, residue={res_keep_env or 'unset'}, "
            f"fallback={prune_fallback_env or 'unset'}"
        )
    print(
        f"[INFO] GraphSAINT: seed_edge_batch_size={seed_edge_batch_size}, "
        f"saint_steps={args.saint_steps}, walk_length={walk_length}, num_walks={num_walks}, "
        f"max_atoms_per_step={max_atoms_per_step}, max_residues_per_step={max_res_per_step}"
    )
    if args.eval_warm_support_k > 0:
        print(
            f"[INFO] eval_warm_support enabled: k={args.eval_warm_support_k}, "
            f"max_add={args.eval_warm_support_max_add}"
        )
    if args.cold_zero_route_mode != "hard" or abs(args.cold_zero_route_min_wc - 1.0) > 1e-12:
        print(
            f"[INFO] cold_zero_route: mode={args.cold_zero_route_mode}, "
            f"min_wc={args.cold_zero_route_min_wc:.3f}"
        )
    run_config = {
        "args": vars(args),
        "env": {
            "HGACN_ATOM_TOPK": atom_topk_env,
            "HGACN_RES_TOPK": res_topk_env,
            "HGACN_ATOM_KEEP": atom_keep_env,
            "HGACN_RES_KEEP": res_keep_env,
            "HGACN_PRUNE_FALLBACK": prune_fallback_env,
            "HGACN_PROT_ESM2": os.environ.get("HGACN_PROT_ESM2", "").strip(),
        },
    }
    try:
        with open("run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write run_config.json: {e}")
    time_start = time.time()
    best_global_auc = 0
    best_model_state = None
    scaler = torch.cuda.amp.GradScaler(enabled=args.cuda)

    try:
        data = load_and_construct_hypergraphs(
            args.dataset, data_root, node_level=args.node_level,
            psichic_attention_path=args.psichic_attention, add_self_loop=use_self_loop,
            protein_feat_mode=args.protein_feat_mode,
            esm_special_tokens=args.esm_special_tokens,
            esm_norm=args.esm_norm,
            esm_strict=args.esm_strict,
            esm_fallback=args.esm_fallback,
            use_physchem_feat=use_physchem_feat,
            reuse_cache=args.reuse_cache,
            prune_strategy_tag=args.prune_strategy_tag,
            atom_topk=args.atom_topk,
            res_topk=args.res_topk,
            atom_randk=args.atom_randk,
            res_randk=args.res_randk,
            randk_seed=args.randk_seed,
            randk_weight_mode=args.randk_weight_mode,
            prior_floor=args.prior_floor,
            use_knn_graph=use_knn_graph,
            knn_setting=knn_setting,
            cold_deg_th_drug=args.cold_deg_th_drug,
            cold_deg_th_prot=args.cold_deg_th_prot,
            cold_mode=args.cold_mode,
            cold_q=args.cold_q,
            cold_q_drug=args.cold_quantile_drug,
            cold_q_prot=args.cold_quantile_prot,
            drug_knn_k=args.drug_knn_k,
            prot_knn_k=args.prot_knn_k,
            knn_metric=args.knn_metric,
            knn_symmetric=knn_symmetric,
            knn_weight_temp=args.knn_weight_temp,
        )
        (train_edges, val_edges, test_edges,
         num_drugs, num_prots,
         H, H_T, G_drug, G_protein,
         features_drug, features_protein,
         atom_to_drug, residue_to_prot,
         atom_attn, residue_attn,
         atom_orig_pos, residue_orig_pos,
         drug_atom_ptr, drug_atom_nodes,
         prot_res_ptr, prot_res_nodes,
         prot_esm_missing, prot_esm_unreliable,
         drug_knn_edge_index, drug_knn_edge_weight,
         prot_knn_edge_index, prot_knn_edge_weight) = data
        expected_hyper_cols = int(train_edges.shape[0])
        actual_hyper_cols = int(H.shape[1])
        print(
            f"[CHECK] hyper_cols={actual_hyper_cols}, expected={expected_hyper_cols}, "
            f"supervised_train_edges={int(train_edges.shape[0])}"
        )
        if actual_hyper_cols != expected_hyper_cols:
            raise RuntimeError(
                f"Atomic hypergraph columns mismatch: got {actual_hyper_cols}, expected {expected_hyper_cols}."
            )
        print(f"Loaded dataset: train={len(train_edges)}, val={len(val_edges)}, test={len(test_edges)}")
    except Exception as e:
        print(f"Dataset load failed: {e}")
        sys.exit(1)

    dataset_dir = os.path.join(data_root, args.dataset)
    deg_dir = os.path.join(dataset_dir, "processed_atomic")
    cfg_path = os.path.join(deg_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    drug_desc_missing_ratio = float(cfg.get("drug_desc_missing_ratio", 0.0)) if cfg else 0.0
    drug_desc_missing_count = int(cfg.get("drug_desc_missing_count", 0)) if cfg else 0
    drug_desc_impute = str(cfg.get("drug_desc_impute", "")) if cfg else ""
    drug_missing_flag_enabled = bool(cfg.get("drug_desc_missing_flag", False)) if cfg else False
    drug_feat_missing_count = int(cfg.get("drug_feat_missing_count", 0)) if cfg else 0
    drug_feat_missing_ratio = float(cfg.get("drug_feat_missing_ratio", 0.0)) if cfg else 0.0
    drug_feat_missing_impute = str(cfg.get("drug_feat_missing_impute", "")) if cfg else ""
    drug_feat_missing_flag = bool(cfg.get("drug_feat_missing_flag", False)) if cfg else False
    esm_mismatch_count = int(cfg.get("esm_mismatch_count", 0)) if cfg else 0
    esm_mismatch_path = str(cfg.get("esm_mismatch_path", "")) if cfg else ""
    esm_mismatch_global_fb = int(cfg.get("esm_mismatch_global_fallback", 0)) if cfg else 0
    prot_knn_disabled_reason = str(cfg.get("prot_knn_disabled_reason", "")) if cfg else ""
    deg_train_drug_path = os.path.join(deg_dir, "deg_train_drug.npy")
    deg_train_prot_path = os.path.join(deg_dir, "deg_train_prot.npy")
    deg_meta_path = os.path.join(deg_dir, "deg_train_meta.json")
    if use_global_degree_for_cold:
        if not (os.path.exists(deg_train_drug_path) and os.path.exists(deg_train_prot_path)):
            raise RuntimeError(
                "Global train degree cache missing. Re-run preprocessing to generate deg_train_drug.npy/deg_train_prot.npy "
                "or disable with --use_global_degree_for_cold=0."
            )
        drug_deg = np.load(deg_train_drug_path).astype(np.float32)
        prot_deg = np.load(deg_train_prot_path).astype(np.float32)
        if os.path.exists(deg_meta_path):
            try:
                with open(deg_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if int(meta.get("train_edges_count", len(train_edges))) != int(len(train_edges)):
                    print("[WARN] deg_train_meta train_edges_count mismatch with current train edges.")
            except Exception as e:
                print(f"[WARN] Failed to read deg_train_meta.json: {e}")
        print("[INFO] deg_source=GLOBAL_TRAIN_ONLY")
    else:
        drug_deg_path = os.path.join(deg_dir, "drug_deg.npy")
        prot_deg_path = os.path.join(deg_dir, "prot_deg.npy")
        if os.path.exists(drug_deg_path) and os.path.exists(prot_deg_path):
            drug_deg = np.load(drug_deg_path).astype(np.float32)
            prot_deg = np.load(prot_deg_path).astype(np.float32)
            print("[WARN] Using cached degree files (not enforced global train-only).")
        else:
            print("[WARN] Degree files missing; fallback to train edge counts.")
            drug_deg = np.bincount(train_edges[:, 0], minlength=num_drugs).astype(np.float32) if len(train_edges) else np.zeros(num_drugs, dtype=np.float32)
            prot_deg = np.bincount(train_edges[:, 1], minlength=num_prots).astype(np.float32) if len(train_edges) else np.zeros(num_prots, dtype=np.float32)
    drug_zero = int((drug_deg == 0).sum())
    prot_zero = int((prot_deg == 0).sum())
    print(f"degree=0 nodes: drug={drug_zero}/{num_drugs}, prot={prot_zero}/{num_prots}")

    # Cold threshold strategy
    cold_mode = (args.cold_mode or "th").strip().lower()
    cold_q_drug = None
    cold_q_prot = None
    if cold_mode == "quantile":
        cold_q_drug = args.cold_quantile_drug if args.cold_quantile_drug is not None else args.cold_q
        cold_q_prot = args.cold_quantile_prot if args.cold_quantile_prot is not None else args.cold_q
        cold_q_drug = max(0.0, min(1.0, float(cold_q_drug)))
        cold_q_prot = max(0.0, min(1.0, float(cold_q_prot)))
        cold_deg_th_drug_eff = float(np.quantile(drug_deg, cold_q_drug)) if drug_deg.size else 0.0
        cold_deg_th_prot_eff = float(np.quantile(prot_deg, cold_q_prot)) if prot_deg.size else 0.0
    else:
        cold_deg_th_drug_eff = float(args.cold_deg_th_drug)
        cold_deg_th_prot_eff = float(args.cold_deg_th_prot)
        if drug_deg.size:
            p10 = float(np.quantile(drug_deg, 0.1))
            if cold_deg_th_drug_eff > max(2.0, p10):
                print(
                    f"[WARN] cold_deg_th_drug={cold_deg_th_drug_eff:.2f} exceeds p10={p10:.2f}; "
                    f"cold_edge_ratio may be high. Consider --cold_mode quantile or --cold_deg_th_drug 2."
                )
    print(
        f"[INFO] cold_mode={cold_mode}, cold_th_drug={cold_deg_th_drug_eff:.3f}, "
        f"cold_th_prot={cold_deg_th_prot_eff:.3f}"
    )
    cold_node_ratio_drug = float((drug_deg <= cold_deg_th_drug_eff).mean()) if drug_deg.size else 0.0
    cold_node_ratio_prot = float((prot_deg <= cold_deg_th_prot_eff).mean()) if prot_deg.size else 0.0
    if drug_deg.size and cold_node_ratio_drug > 0.25:
        if cold_mode == "quantile":
            new_q = max(float(cold_q_drug or 0.0) * 0.5, 0.0)
            cold_q_drug = new_q
            cold_deg_th_drug_eff = float(np.quantile(drug_deg, cold_q_drug)) if drug_deg.size else 0.0
            print(
                f"[WARN] cold_node_ratio_drug={cold_node_ratio_drug:.3f} > 0.25; "
                f"shrink cold_q_drug -> {cold_q_drug:.3f}, cold_th_drug -> {cold_deg_th_drug_eff:.3f}"
            )
        else:
            p10 = float(np.quantile(drug_deg, 0.1))
            cold_deg_th_drug_eff = min(cold_deg_th_drug_eff, p10)
            print(
                f"[WARN] cold_node_ratio_drug={cold_node_ratio_drug:.3f} > 0.25; "
                f"clamp cold_th_drug -> {cold_deg_th_drug_eff:.3f} (p10={p10:.3f})"
            )
        cold_node_ratio_drug = float((drug_deg <= cold_deg_th_drug_eff).mean()) if drug_deg.size else 0.0
    cold_node_count_drug = int((drug_deg <= cold_deg_th_drug_eff).sum()) if drug_deg.size else 0
    cold_node_count_prot = int((prot_deg <= cold_deg_th_prot_eff).sum()) if prot_deg.size else 0
    cold_q_drug_str = f"{cold_q_drug:.3f}" if cold_q_drug is not None else "N/A"
    cold_q_prot_str = f"{cold_q_prot:.3f}" if cold_q_prot is not None else "N/A"
    print(
        f"[INFO] cold_node_ratio drug={cold_node_ratio_drug:.4f} prot={cold_node_ratio_prot:.4f} "
        f"cold_node_count drug={cold_node_count_drug} prot={cold_node_count_prot} "
        f"cold_th_drug_actual={cold_deg_th_drug_eff:.3f} cold_q_drug_actual={cold_q_drug_str} "
        f"cold_th_prot_actual={cold_deg_th_prot_eff:.3f} cold_q_prot_actual={cold_q_prot_str}"
    )
    if cold_node_ratio_drug > 0.35 or cold_node_ratio_prot > 0.35:
        print(
            f"[WARN] cold_node_ratio high: drug={cold_node_ratio_drug:.3f} prot={cold_node_ratio_prot:.3f}. "
            "Consider --cold_mode quantile or higher degree thresholds."
        )
    if drug_missing_flag_enabled:
        print(
            f"[INFO] drug_desc_missing count={drug_desc_missing_count} "
            f"ratio={drug_desc_missing_ratio:.4f} impute={drug_desc_impute} missing_flag=on"
        )
    if drug_feat_missing_count or drug_feat_missing_flag:
        print(
            f"[INFO] drug_features_missing count={drug_feat_missing_count} "
            f"ratio={drug_feat_missing_ratio:.4f} impute={drug_feat_missing_impute or 'none'} "
            f"missing_flag={'on' if drug_feat_missing_flag else 'off'}"
        )
    if esm_mismatch_count:
        print(
            f"[INFO] esm_mismatch count={esm_mismatch_count} "
            f"global_fallback={esm_mismatch_global_fb} path={esm_mismatch_path or 'N/A'}"
        )
    if prot_knn_disabled_reason:
        prot_cold_node_count = int((prot_deg <= cold_deg_th_prot_eff).sum()) if prot_deg.size else 0
        print(
            f"[INFO] prot_knn_disabled_reason={prot_knn_disabled_reason} "
            f"prot_cold_node_count={prot_cold_node_count}"
        )
    drug_pcts = _percentiles(drug_deg)
    prot_pcts = _percentiles(prot_deg)
    print(
        f"[INFO] global_degree_percentiles drug(p0/p10/p50/p90/p100)={np.round(drug_pcts, 3).tolist()} "
        f"prot(p0/p10/p50/p90/p100)={np.round(prot_pcts, 3).tolist()}"
    )
    if drug_deg.shape[0] != num_drugs or prot_deg.shape[0] != num_prots:
        raise RuntimeError(
            f"Global degree size mismatch: drug_deg={drug_deg.shape[0]} num_drugs={num_drugs}; "
            f"prot_deg={prot_deg.shape[0]} num_prots={num_prots}."
        )

    def _zero_ratio(feat):
        if feat is None or feat.size == 0:
            return 0.0
        norms = np.linalg.norm(feat, axis=1)
        return float((norms == 0).mean())

    zero_drug_ratio = _zero_ratio(features_drug)
    zero_prot_ratio = _zero_ratio(features_protein)
    print(f"zero feature ratio: drug={zero_drug_ratio:.4f}, prot={zero_prot_ratio:.4f}")
    if not args.allow_zero_features and (zero_drug_ratio > 0.005 or zero_prot_ratio > 0.005):
        extra = ""
        try:
            with open(os.path.join(dataset_dir, "drug_to_idx.pkl"), "rb") as f:
                drug_to_idx = pickle.load(f)
            with open(os.path.join(dataset_dir, "protein_to_idx.pkl"), "rb") as f:
                prot_to_idx = pickle.load(f)
            drug_id_list = build_id_list(drug_to_idx)
            prot_id_list = build_id_list(prot_to_idx)
            if features_drug is not None and features_drug.size:
                zero_idx = np.where(np.linalg.norm(features_drug, axis=1) == 0)[0][:10]
                extra += f"\n  drug_zero_examples={[(drug_id_list[i] if i < len(drug_id_list) else None) for i in zero_idx]}"
            if features_protein is not None and features_protein.size:
                zero_idx = np.where(np.linalg.norm(features_protein, axis=1) == 0)[0][:10]
                extra += f"\n  prot_zero_examples={[(prot_id_list[i] if i < len(prot_id_list) else None) for i in zero_idx]}"
        except Exception:
            pass
        raise RuntimeError(
            f"Too many zero feature rows (drug={zero_drug_ratio:.4f}, prot={zero_prot_ratio:.4f})."
            f" Use --allow_zero_features to override." + extra
        )

    print(
        f"self_loop={'on' if use_self_loop else 'off'}, "
        f"residual={'on' if use_residual else 'off'}, "
        f"coldstart_gate={'on' if use_coldstart_gate else 'off'}"
    )
    print(
        f"protein_feat_mode={args.protein_feat_mode}, esm_norm={args.esm_norm}, "
        f"esm_special_tokens={args.esm_special_tokens}, esm_fallback={args.esm_fallback}, "
        f"physchem={'on' if use_physchem_feat else 'off'}, alpha_temp={args.alpha_temp:.3f}"
    )

    attn_db = None

    drug_id_list = None
    prot_id_list = None
    if args.psichic_attention:
        attn_db = load_psichic_attention(args.psichic_attention)
        dataset_dir = os.path.join(data_root, args.dataset)
        try:
            with open(os.path.join(dataset_dir, "drug_to_idx.pkl"), "rb") as f:
                drug_to_idx = pickle.load(f)
            with open(os.path.join(dataset_dir, "protein_to_idx.pkl"), "rb") as f:
                prot_to_idx = pickle.load(f)
            drug_id_list = build_id_list(drug_to_idx)
            prot_id_list = build_id_list(prot_to_idx)
        except Exception as e:
            print(f"[WARN] Failed to load drug/protein id list: {e}")
            drug_id_list = None
            prot_id_list = None

    val_edges_tensor_full = torch.tensor(val_edges[:, :2], dtype=torch.long, device=device)
    val_labels_tensor_full = torch.tensor(val_edges[:, 2], dtype=torch.float32, device=device)
    test_edges_tensor_full = torch.tensor(test_edges[:, :2], dtype=torch.long, device=device)
    test_labels_tensor_full = torch.tensor(test_edges[:, 2], dtype=torch.float32, device=device)

    atom_indptr = None
    atom_indices = None
    atom_data = None
    res_indptr = None
    res_indices = None
    res_data = None
    if use_graphsaint:
        if not sparse.isspmatrix_csr(G_drug):
            G_drug = G_drug.tocsr()
        if not sparse.isspmatrix_csr(G_protein):
            G_protein = G_protein.tocsr()
        atom_indptr = G_drug.indptr.astype(np.int64, copy=False)
        atom_indices = G_drug.indices.astype(np.int64, copy=False)
        atom_data = G_drug.data.astype(np.float32, copy=False)
        res_indptr = G_protein.indptr.astype(np.int64, copy=False)
        res_indices = G_protein.indices.astype(np.int64, copy=False)
        res_data = G_protein.data.astype(np.float32, copy=False)

    val_edges_np = val_edges
    test_edges_np = test_edges
    # Free unused large arrays early to reduce RAM pressure.
    del H, H_T
    gc.collect()

    # Move large arrays to device, then free CPU copies to reduce RAM pressure.
    features_drug_tensor = torch.from_numpy(features_drug).to(device=device, dtype=torch.float32)
    features_protein_tensor = torch.from_numpy(features_protein).to(device=device, dtype=torch.float32)
    G_drug_tensor = to_torch_graph(G_drug, device)
    G_protein_tensor = to_torch_graph(G_protein, device)
    drug_deg_tensor = torch.from_numpy(drug_deg).to(device=device, dtype=torch.float32)
    prot_deg_tensor = torch.from_numpy(prot_deg).to(device=device, dtype=torch.float32)
    prot_esm_missing_tensor = (
        torch.from_numpy(prot_esm_missing).to(device=device, dtype=torch.float32)
        if prot_esm_missing is not None
        else None
    )
    prot_esm_unreliable_tensor = (
        torch.from_numpy(prot_esm_unreliable).to(device=device, dtype=torch.float32)
        if prot_esm_unreliable is not None
        else None
    )
    esm_missing_node_ratio = None
    esm_missing_edge_ratio = None
    esm_unrel_node_ratio = None
    esm_unrel_edge_ratio = None
    if prot_esm_missing is not None and len(train_edges):
        esm_missing_node_ratio = float(np.mean(prot_esm_missing))
        esm_missing_edge_ratio = float((prot_esm_missing[train_edges[:, 1].astype(np.int64)] > 0).mean())
        print(
            f"[INFO] esm_missing node_ratio={esm_missing_node_ratio:.4f} "
            f"edge_ratio(train)={esm_missing_edge_ratio:.4f}"
        )
    if prot_esm_unreliable is not None and len(train_edges):
        esm_unrel_node_ratio = float(np.mean(prot_esm_unreliable))
        esm_unrel_edge_ratio = float((prot_esm_unreliable[train_edges[:, 1].astype(np.int64)] > 0).mean())
        print(
            f"[INFO] esm_unreliable node_ratio={esm_unrel_node_ratio:.4f} "
            f"edge_ratio(train)={esm_unrel_edge_ratio:.4f}"
        )
    drug_knn_edge_index_t = (
        torch.from_numpy(drug_knn_edge_index).to(device=device, dtype=torch.long)
        if drug_knn_edge_index is not None
        else None
    )
    drug_knn_edge_weight_t = (
        torch.from_numpy(drug_knn_edge_weight).to(device=device, dtype=torch.float32)
        if drug_knn_edge_weight is not None
        else None
    )
    prot_knn_edge_index_t = (
        torch.from_numpy(prot_knn_edge_index).to(device=device, dtype=torch.long)
        if prot_knn_edge_index is not None
        else None
    )
    prot_knn_edge_weight_t = (
        torch.from_numpy(prot_knn_edge_weight).to(device=device, dtype=torch.float32)
        if prot_knn_edge_weight is not None
        else None
    )
    del features_drug, features_protein, G_drug, G_protein
    gc.collect()

    atom_to_drug_tensor = None
    residue_to_prot_tensor = None
    atom_attn_tensor = None
    residue_attn_tensor = None
    atom_attn_np = None
    residue_attn_np = None
    drug_total_count = None
    prot_total_count = None
    atom_attn_np = atom_attn.astype(np.float32, copy=False) if atom_attn is not None else None
    residue_attn_np = residue_attn.astype(np.float32, copy=False) if residue_attn is not None else None
    atom_to_drug_tensor = torch.from_numpy(atom_to_drug).to(device=device, dtype=torch.long)
    residue_to_prot_tensor = torch.from_numpy(residue_to_prot).to(device=device, dtype=torch.long)
    atom_attn_tensor = torch.from_numpy(atom_attn).to(device=device, dtype=torch.float32)
    residue_attn_tensor = torch.from_numpy(residue_attn).to(device=device, dtype=torch.float32)
    drug_total_count = torch.bincount(atom_to_drug_tensor, minlength=num_drugs).float().clamp_min(1.0)
    prot_total_count = torch.bincount(residue_to_prot_tensor, minlength=num_prots).float().clamp_min(1.0)
    del atom_to_drug, residue_to_prot, atom_attn, residue_attn
    gc.collect()

    train_edge_pairs = train_edges[:, :2]
    train_edge_labels = train_edges[:, 2]
    unique_labels, label_counts = np.unique(train_edge_labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("Training edges only contain one label.")

    eval_mode = args.eval_mode
    if eval_mode == "fixed" and args.num_fold != 1:
        print("[WARN] eval_mode=fixed forces num_fold=1; ignoring num_fold>1.")
        args.num_fold = 1
    if should_block_atomic_cv(
        eval_mode=eval_mode,
        node_level=args.node_level,
        allow_atomic_cv_leakage=args.allow_atomic_cv_leakage,
    ):
        raise RuntimeError(
            "Blocked unsafe setup: eval_mode=cv with node_level=atomic can leak fold information "
            "because graphs are currently built from full train.csv before splitting. "
            "Use --eval_mode fixed (recommended for paper reporting), or pass "
            "--allow_atomic_cv_leakage to keep legacy behavior for debugging only."
        )
    if eval_mode == "cv" and args.allow_atomic_cv_leakage:
        print(
            "[WARN] eval_mode=cv with atomic nodes is leakage-prone; "
            "--allow_atomic_cv_leakage is set, continuing in legacy mode."
        )

    max_supported_folds = int(label_counts.min())
    num_folds = min(args.num_fold, max_supported_folds)
    if num_folds < 1:
        raise ValueError("num_fold must be >= 1.")

    train_losses = [[] for _ in range(num_folds)]
    auc_list, aupr_list = [], []
    fold_metrics = []

    skf = None
    if num_folds > 1:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
    drug_feat_dim = features_drug_tensor.shape[1]
    prot_feat_dim = features_protein_tensor.shape[1]
    print(f"[INFO] Feature dims: drug={drug_feat_dim}, protein={prot_feat_dim}")

    if num_folds > 1:
        fold_splits = skf.split(train_edge_pairs, train_edge_labels)
    else:
        fold_splits = [(np.arange(train_edge_pairs.shape[0]), np.array([], dtype=np.int64))]

    for fold_idx, (train_index, val_index) in enumerate(fold_splits, start=1):
        print(f"\n--- Fold {fold_idx}/{num_folds} ---")
        fold_train_edges = train_edges[train_index]
        fold_val_edges = train_edges[val_index]
        fold_neg = int((fold_train_edges[:, 2] == 0).sum()) if fold_train_edges.size else 0
        print(
            f"[CHECK] fold_supervised_edges={int(fold_train_edges.shape[0])}, fold_neg_edges={fold_neg}"
        )

        if eval_mode == "cv":
            if fold_val_edges.size > 0:
                val_edges_tensor = torch.tensor(
                    fold_val_edges[:, :2], dtype=torch.long, device=device
                )
                val_labels_tensor = torch.tensor(
                    fold_val_edges[:, 2], dtype=torch.float32, device=device
                )
            else:
                val_edges_tensor = None
                val_labels_tensor = None
            val_edges_np_eval = fold_val_edges
        else:
            val_edges_tensor = val_edges_tensor_full
            val_labels_tensor = val_labels_tensor_full
            val_edges_np_eval = val_edges_np
        val_labels_np_eval = val_edges_np_eval[:, 2] if val_edges_np_eval.size else np.empty((0,), dtype=np.float32)
        train_edges_tensor = None
        train_labels_tensor = None
        num_pos = float(fold_train_edges[:, 2].sum())
        num_neg = float(len(fold_train_edges) - num_pos)

        model = HGACN(
            drug_feat_dim=drug_feat_dim,
            prot_feat_dim=prot_feat_dim,
            hidden_dim=args.hidden,
            out_dim=args.out_dim,
            gat_top_k_sparse=gat_top_k_sparse,
            gat_dense_top_k=gat_dense_top_k,
            interaction=args.interaction_head,
            use_hyperedge_head=use_hyperedge_head,
            alpha_refine=alpha_refine,
            alpha_eps=args.alpha_eps,
            prior_eps=args.prior_eps,
            alpha_temp=args.alpha_temp,
            use_residual=use_residual,
            use_coldstart_gate=use_coldstart_gate,
            prior_smoothing=args.prior_smoothing,
            use_bottleneck_gate=use_bottleneck_gate,
            bottleneck_drop=args.bottleneck_drop,
            moe_enable=moe_enable,
            expert_A=expert_A,
            expert_B=expert_B,
            expert_C=expert_C,
            pool_topk=args.pool_topk,
            pool_randk=args.pool_randk,
            beta_mix=args.beta_mix,
            randk_weight_mode=args.randk_weight_mode,
            prior_floor=args.prior_floor,
            prior_mix_mode=args.prior_mix_mode,
            prior_mix_lambda=args.prior_mix_lambda,
            prior_mix_learnable=args.prior_mix_learnable,
            prior_mix_conditional=args.prior_mix_conditional,
            prior_mix_features=prior_mix_features,
            mp_gate_mode=args.mp_gate_mode,
            mp_gate_deg_only=args.mp_gate_deg_only,
            mp_gate_init_bias=args.mp_gate_init_bias,
            mp_gate_use_attn_entropy=bool(args.mp_gate_use_attn_entropy),
            mp_gate_use_prior_entropy=bool(args.mp_gate_use_prior_entropy),
            mp_gate_use_esm_missing=bool(args.mp_gate_use_esm_missing),
            mp_gate_use_prior_conf=bool(args.mp_gate_use_prior_conf),
            mp_gate_cold_scale=float(args.mp_gate_cold_scale),
            use_knn_graph=use_knn_graph,
            cold_deg_th_drug=cold_deg_th_drug_eff,
            cold_deg_th_prot=cold_deg_th_prot_eff,
            attn_kl_clip=args.attn_kl_clip,
            kl_stage1_epochs=args.kl_stage1_epochs,
            cold_zero_route_mode=args.cold_zero_route_mode,
            cold_zero_route_min_wc=args.cold_zero_route_min_wc,
            use_drug_missing_flag=drug_missing_flag_enabled,
        ).to(device)

        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=device) if num_pos > 0 else None

        criterion = CombinedLoss(
            kl_weight=args.kl_weight,
            attn_kl_weight=args.attn_kl_weight,
            kl_stage1_epochs=args.kl_stage1_epochs,
            attn_kl_w_min=args.attn_kl_w_min,
            attn_kl_w_max=args.attn_kl_w_max,
            attn_kl_ramp_epochs=args.attn_kl_ramp_epochs,
            attn_kl_schedule=args.attn_kl_weight_schedule,
            attn_kl_max_ratio=args.attn_kl_max_ratio,
            kl_hard_assert=kl_hard_assert,
            prior_conf_ref=args.prior_conf_ref,
            info_nce_weight=args.info_nce_weight,
            info_nce_temp=args.info_nce_temp,
            info_nce_neg_k=args.info_nce_neg_k,
            info_nce_max_ratio=args.info_nce_max_ratio,
            gate_balance_weight=args.gate_balance_weight,
            gate_entropy_weight=args.gate_entropy_weight,
            delta_reg_weight=args.delta_reg_weight,
            distill_weight=args.distill_weight if distill_enable else 0.0,
            distill_T=args.distill_T,
            distill_mode=args.distill_mode,
            kd_max_ratio=args.kd_max_ratio,
            num_drugs=num_drugs,
            num_proteins=num_prots,
            pos_weight=pos_weight,
            debug_assertions=bool(args.debug_assertions),
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(args.plateau_factor),
                patience=int(args.plateau_patience),
                min_lr=float(args.plateau_min_lr),
            )
        else:
            scheduler = None
        if args.resume:
            print(f"==> 正在从 {args.resume} 恢复训练...")
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[WARN] Missing keys when loading checkpoint: {len(missing_keys)}")
            if unexpected_keys:
                print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected_keys)}")
            # Hard override CLI hyperparams after loading weights.
            if hasattr(model, "alpha_temp"):
                model.alpha_temp = float(args.alpha_temp)
            if hasattr(model, "prior_smoothing"):
                model.prior_smoothing = float(args.prior_smoothing)
            print("==> 权重载入成功！")
        ema_state = None
        if use_ema:
            ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def run_truth_radar(epoch_idx):
            if eval_mode != "fixed":
                return None, None
            if test_labels_tensor_full is None or test_labels_tensor_full.numel() <= 0:
                return None, None
            ema_backup = None
            if use_ema and ema_eval and ema_state is not None:
                ema_backup = swap_ema_weights(model, ema_state)
            try:
                radar_stats = {}
                if use_hyperedge_head:
                    test_auc_epoch, test_aupr_epoch, test_coverage_epoch, radar_stats = evaluate_subgraph_edges(
                        model,
                        features_drug_tensor, features_protein_tensor,
                        atom_indptr, atom_indices, atom_data,
                        res_indptr, res_indices, res_data,
                        test_edges_np[:, :2], test_edges_np[:, 2],
                        drug_atom_ptr, drug_atom_nodes,
                        prot_res_ptr, prot_res_nodes,
                        atom_orig_pos, residue_orig_pos,
                        atom_to_drug_tensor, residue_to_prot_tensor,
                        atom_attn_tensor, residue_attn_tensor,
                        atom_attn_np, residue_attn_np,
                        attn_db, drug_id_list, prot_id_list,
                        seed_edge_batch_size, walk_length, num_walks,
                        max_atoms_per_step, max_res_per_step,
                        alpha_eps=args.alpha_eps,
                        edge_min_incidence_atom=edge_min_incidence_atom,
                        edge_min_incidence_res=edge_min_incidence_res,
                        eval_num_samples=eval_num_samples,
                        device=device,
                        rng_seed=args.seed + int(epoch_idx) * 17 + fold_idx * 101,
                        drug_degree=drug_deg_tensor,
                        prot_degree=prot_deg_tensor,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing_tensor,
                        prot_esm_unreliable=prot_esm_unreliable_tensor,
                        drug_knn_edge_index=drug_knn_edge_index_t,
                        drug_knn_edge_weight=drug_knn_edge_weight_t,
                        prot_knn_edge_index=prot_knn_edge_index_t,
                        prot_knn_edge_weight=prot_knn_edge_weight_t,
                        cold_deg_th_drug=cold_deg_th_drug_eff,
                        cold_deg_th_prot=cold_deg_th_prot_eff,
                        expertC_scale=1.0,
                        wC_cap=args.wC_cap,
                        eval_score_centering=args.eval_score_centering,
                        eval_warm_support_k=args.eval_warm_support_k,
                        eval_warm_support_max_add=args.eval_warm_support_max_add,
                    )
                    if (
                        test_coverage_epoch is not None
                        and test_coverage_epoch < float(args.eval_coverage_min)
                    ):
                        print(f"[WARN] Truth radar coverage low: {test_coverage_epoch:.3f}")
                else:
                    test_auc_epoch, test_aupr_epoch, radar_stats = evaluate_full_graph(
                        model,
                        features_drug_tensor, features_protein_tensor,
                        G_drug_tensor, G_protein_tensor,
                        test_edges_tensor_full, test_labels_tensor_full,
                        drug_node_to_entity=atom_to_drug_tensor,
                        protein_node_to_entity=residue_to_prot_tensor,
                        drug_node_weight=atom_attn_tensor,
                        protein_node_weight=residue_attn_tensor,
                        atom_prior=atom_attn_tensor,
                        res_prior=residue_attn_tensor,
                        drug_degree=drug_deg_tensor,
                        prot_degree=prot_deg_tensor,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing_tensor,
                        prot_esm_unreliable=prot_esm_unreliable_tensor,
                        drug_knn_edge_index=drug_knn_edge_index_t,
                        drug_knn_edge_weight=drug_knn_edge_weight_t,
                        prot_knn_edge_index=prot_knn_edge_index_t,
                        prot_knn_edge_weight=prot_knn_edge_weight_t,
                        cold_deg_th_drug=cold_deg_th_drug_eff,
                        cold_deg_th_prot=cold_deg_th_prot_eff,
                        expertC_scale=1.0,
                        wC_cap=args.wC_cap,
                        scaler=scaler,
                        eval_score_centering=args.eval_score_centering,
                    )
                if test_auc_epoch is not None and test_aupr_epoch is not None:
                    print(
                        f"| [TRUTH RADAR] Epoch {int(epoch_idx):03d} | "
                        f"TEST AUC: {float(test_auc_epoch):.5f} | TEST AUPR: {float(test_aupr_epoch):.5f} |"
                    )
                    _print_global_metric_variants(radar_stats, prefix="TRUTH-RADAR")
                return test_auc_epoch, test_aupr_epoch
            finally:
                if ema_backup is not None:
                    restore_weights(model, ema_backup)

        if args.eval_only:
            # Ensure eval uses CLI-provided hyperparams (avoid dynamic scheduler overrides).
            model.alpha_temp = float(args.alpha_temp)
            model.prior_smoothing = float(args.prior_smoothing)
            criterion.info_nce_weight = float(args.info_nce_weight)
            expertC_scale = 1.0
            wC_cap = args.wC_cap
            do_validation = (
                val_edges_tensor is not None
                and val_labels_tensor is not None
                and val_labels_tensor.numel() > 0
            )
            auc_val = None
            aupr_val = None
            test_auc = None
            test_aupr = None
            val_cold_stats = {}
            test_cold_stats = {}
            ema_backup = None
            if use_ema and ema_eval and ema_state is not None:
                ema_backup = swap_ema_weights(model, ema_state)
            if do_validation:
                if use_hyperedge_head:
                    auc_val, aupr_val, coverage, cold_stats = evaluate_subgraph_edges(
                        model,
                        features_drug_tensor, features_protein_tensor,
                        atom_indptr, atom_indices, atom_data,
                        res_indptr, res_indices, res_data,
                        val_edges_np_eval[:, :2], val_labels_np_eval,
                        drug_atom_ptr, drug_atom_nodes,
                        prot_res_ptr, prot_res_nodes,
                        atom_orig_pos, residue_orig_pos,
                        atom_to_drug_tensor, residue_to_prot_tensor,
                        atom_attn_tensor, residue_attn_tensor,
                        atom_attn_np, residue_attn_np,
                        attn_db, drug_id_list, prot_id_list,
                        seed_edge_batch_size, walk_length, num_walks,
                        max_atoms_per_step, max_res_per_step,
                        alpha_eps=args.alpha_eps,
                        edge_min_incidence_atom=edge_min_incidence_atom,
                        edge_min_incidence_res=edge_min_incidence_res,
                        eval_num_samples=eval_num_samples,
                        device=device,
                        rng_seed=args.seed + fold_idx * 101,
                        drug_degree=drug_deg_tensor,
                        prot_degree=prot_deg_tensor,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing_tensor,
                        prot_esm_unreliable=prot_esm_unreliable_tensor,
                        drug_knn_edge_index=drug_knn_edge_index_t,
                        drug_knn_edge_weight=drug_knn_edge_weight_t,
                        prot_knn_edge_index=prot_knn_edge_index_t,
                        prot_knn_edge_weight=prot_knn_edge_weight_t,
                        cold_deg_th_drug=cold_deg_th_drug_eff,
                        cold_deg_th_prot=cold_deg_th_prot_eff,
                        expertC_scale=expertC_scale,
                        wC_cap=wC_cap,
                        eval_score_centering=args.eval_score_centering,
                        eval_warm_support_k=args.eval_warm_support_k,
                        eval_warm_support_max_add=args.eval_warm_support_max_add,
                    )
                    if coverage < float(args.eval_coverage_min):
                        print(f"[WARN] Eval coverage low: {coverage:.3f}")
                    print(f"[INFO] val_eval_samples={eval_num_samples} (GraphSAINT)")
                else:
                    auc_val, aupr_val, cold_stats = evaluate_full_graph(
                        model,
                        features_drug_tensor, features_protein_tensor,
                        G_drug_tensor, G_protein_tensor,
                        val_edges_tensor, val_labels_tensor,
                        drug_node_to_entity=atom_to_drug_tensor,
                        protein_node_to_entity=residue_to_prot_tensor,
                        drug_node_weight=atom_attn_tensor,
                        protein_node_weight=residue_attn_tensor,
                        atom_prior=atom_attn_tensor,
                        res_prior=residue_attn_tensor,
                        drug_degree=drug_deg_tensor,
                        prot_degree=prot_deg_tensor,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing_tensor,
                        prot_esm_unreliable=prot_esm_unreliable_tensor,
                        drug_knn_edge_index=drug_knn_edge_index_t,
                        drug_knn_edge_weight=drug_knn_edge_weight_t,
                        prot_knn_edge_index=prot_knn_edge_index_t,
                        prot_knn_edge_weight=prot_knn_edge_weight_t,
                        cold_deg_th_drug=cold_deg_th_drug_eff,
                        cold_deg_th_prot=cold_deg_th_prot_eff,
                        expertC_scale=expertC_scale,
                        wC_cap=wC_cap,
                        scaler=scaler,
                        eval_score_centering=args.eval_score_centering,
                    )
                    print("[INFO] val_eval_mode=full")
                val_cold_stats = cold_stats if isinstance(cold_stats, dict) else {}
                print(f"[EVAL-ONLY] Val AUC: {auc_val:.5f}, AUPR: {aupr_val:.5f}")
                _print_global_metric_variants(val_cold_stats, prefix="EVAL-ONLY-VAL")
                if cold_stats:
                    cd = cold_stats.get("cold_drug", {})
                    cp = cold_stats.get("cold_prot", {})
                    cb = cold_stats.get("cold_both", {})
                    print(
                        f"[EVAL-ONLY-COLD] drug AUC/AUPR={cd.get('auc', 0.0):.4f}/{cd.get('aupr', 0.0):.4f} n={cd.get('n', 0)}; "
                        f"prot AUC/AUPR={cp.get('auc', 0.0):.4f}/{cp.get('aupr', 0.0):.4f} n={cp.get('n', 0)}; "
                        f"both AUC/AUPR={cb.get('auc', 0.0):.4f}/{cb.get('aupr', 0.0):.4f} n={cb.get('n', 0)}"
                    )
            else:
                print("[EVAL-ONLY] No validation edges available.")

            if eval_mode == "fixed" and test_labels_tensor_full.numel() > 0:
                if use_hyperedge_head:
                    test_auc, test_aupr, coverage, cold_stats = evaluate_subgraph_edges(
                        model,
                        features_drug_tensor, features_protein_tensor,
                        atom_indptr, atom_indices, atom_data,
                        res_indptr, res_indices, res_data,
                        test_edges_np[:, :2], test_edges_np[:, 2],
                        drug_atom_ptr, drug_atom_nodes,
                        prot_res_ptr, prot_res_nodes,
                        atom_orig_pos, residue_orig_pos,
                        atom_to_drug_tensor, residue_to_prot_tensor,
                        atom_attn_tensor, residue_attn_tensor,
                        atom_attn_np, residue_attn_np,
                        attn_db, drug_id_list, prot_id_list,
                        seed_edge_batch_size, walk_length, num_walks,
                        max_atoms_per_step, max_res_per_step,
                        alpha_eps=args.alpha_eps,
                        edge_min_incidence_atom=edge_min_incidence_atom,
                        edge_min_incidence_res=edge_min_incidence_res,
                        eval_num_samples=eval_num_samples,
                        device=device,
                        rng_seed=args.seed + 999,
                        drug_degree=drug_deg_tensor,
                        prot_degree=prot_deg_tensor,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing_tensor,
                        prot_esm_unreliable=prot_esm_unreliable_tensor,
                        drug_knn_edge_index=drug_knn_edge_index_t,
                        drug_knn_edge_weight=drug_knn_edge_weight_t,
                        prot_knn_edge_index=prot_knn_edge_index_t,
                        prot_knn_edge_weight=prot_knn_edge_weight_t,
                        cold_deg_th_drug=cold_deg_th_drug_eff,
                        cold_deg_th_prot=cold_deg_th_prot_eff,
                        expertC_scale=expertC_scale,
                        wC_cap=wC_cap,
                        eval_score_centering=args.eval_score_centering,
                        eval_warm_support_k=args.eval_warm_support_k,
                        eval_warm_support_max_add=args.eval_warm_support_max_add,
                    )
                    if coverage < float(args.eval_coverage_min):
                        print(f"[WARN] Test coverage low: {coverage:.3f}")
                else:
                    test_auc, test_aupr, cold_stats = evaluate_full_graph(
                        model,
                        features_drug_tensor, features_protein_tensor,
                        G_drug_tensor, G_protein_tensor,
                        test_edges_tensor_full, test_labels_tensor_full,
                        drug_node_to_entity=atom_to_drug_tensor,
                        protein_node_to_entity=residue_to_prot_tensor,
                        drug_node_weight=atom_attn_tensor,
                        protein_node_weight=residue_attn_tensor,
                        atom_prior=atom_attn_tensor,
                        res_prior=residue_attn_tensor,
                        drug_degree=drug_deg_tensor,
                        prot_degree=prot_deg_tensor,
                        use_coldstart_gate=use_coldstart_gate,
                        prot_esm_missing=prot_esm_missing_tensor,
                        prot_esm_unreliable=prot_esm_unreliable_tensor,
                        drug_knn_edge_index=drug_knn_edge_index_t,
                        drug_knn_edge_weight=drug_knn_edge_weight_t,
                        prot_knn_edge_index=prot_knn_edge_index_t,
                        prot_knn_edge_weight=prot_knn_edge_weight_t,
                        cold_deg_th_drug=cold_deg_th_drug_eff,
                        cold_deg_th_prot=cold_deg_th_prot_eff,
                        expertC_scale=expertC_scale,
                        wC_cap=wC_cap,
                        scaler=scaler,
                        eval_score_centering=args.eval_score_centering,
                    )
                test_cold_stats = cold_stats if isinstance(cold_stats, dict) else {}
                print(f"[EVAL-ONLY] Test AUC: {test_auc:.5f}, AUPR: {test_aupr:.5f}")
                _print_global_metric_variants(test_cold_stats, prefix="EVAL-ONLY-TEST")
                if cold_stats:
                    cd = cold_stats.get("cold_drug", {})
                    cp = cold_stats.get("cold_prot", {})
                    cb = cold_stats.get("cold_both", {})
                    print(
                        f"[EVAL-ONLY-COLD-TEST] drug AUC/AUPR={cd.get('auc', 0.0):.4f}/{cd.get('aupr', 0.0):.4f} n={cd.get('n', 0)}; "
                        f"prot AUC/AUPR={cp.get('auc', 0.0):.4f}/{cp.get('aupr', 0.0):.4f} n={cp.get('n', 0)}; "
                        f"both AUC/AUPR={cb.get('auc', 0.0):.4f}/{cb.get('aupr', 0.0):.4f} n={cb.get('n', 0)}"
                    )

            if ema_backup is not None:
                restore_weights(model, ema_backup)
            write_metrics_out(
                args.metrics_out,
                {
                    "mode": "eval_only",
                    "dataset": str(args.dataset),
                    "seed": int(args.seed),
                    "fold": int(fold_idx),
                    "eval_mode": str(eval_mode),
                    "node_level": str(args.node_level),
                    "val_auc": float(auc_val) if auc_val is not None else None,
                    "val_aupr": float(aupr_val) if aupr_val is not None else None,
                    "val_cold_stats": val_cold_stats,
                    "test_auc": float(test_auc) if test_auc is not None else None,
                    "test_aupr": float(test_aupr) if test_aupr is not None else None,
                    "test_cold_stats": test_cold_stats,
                    "mean_auc": float(auc_val) if auc_val is not None else 0.0,
                    "mean_aupr": float(aupr_val) if aupr_val is not None else 0.0,
                },
            )
            sys.exit(0)

        early_stopping_patience = 20
        best_val_metric = -float("inf")
        patience_counter = 0
        best_auc, best_aupr = 0, 0
        high_ent_atom_epochs = 0
        high_ent_res_epochs = 0
        cold_ablation_note_printed = False
        dynamic_state = {}
        last_epoch_metrics = {}

        for epoch in range(args.start_epoch, args.epochs + 1):
            dyn = update_dynamic_params(
                epoch, args, optimizer, model=model, criterion=criterion, state=dynamic_state
            )
            if use_graphsaint and not use_reweight:
                print("[WARN] GraphSAINT reweight is disabled; sampling may be biased.")
            if hasattr(criterion, "set_epoch"):
                criterion.set_epoch(epoch)
            if hasattr(model, "set_epoch"):
                model.set_epoch(epoch)
            criterion.distill_T = args.distill_T
            criterion.distill_mode = args.distill_mode
            criterion.gate_balance_weight = args.gate_balance_weight
            criterion.gate_entropy_weight = args.gate_entropy_weight
            criterion.kd_max_ratio = args.kd_max_ratio
            distill_warmup_epochs = int(args.distill_warmup_epochs) if args.distill_warmup_epochs is not None else int(args.kd_warmup_epochs)
            distill_start_epoch = max(0, int(args.distill_start_epoch))
            if distill_enable and epoch >= distill_start_epoch:
                if distill_warmup_epochs > 0:
                    kd_scale = min(1.0, float(epoch - distill_start_epoch + 1) / float(distill_warmup_epochs))
                else:
                    kd_scale = 1.0
                criterion.distill_weight = args.distill_weight * kd_scale
            else:
                criterion.distill_weight = 0.0
            expertC_scale = dyn.get("expertC_scale", 1.0)
            wC_cap = args.wC_cap
            do_validation = (
                val_edges_tensor is not None
                and val_labels_tensor is not None
                and val_labels_tensor.numel() > 0
            )
            if use_graphsaint:
                # Priority 1: explicit fixed steps from CLI.
                if args.saint_steps > 0:
                    steps_per_epoch = args.saint_steps
                    print(f"[INFO] GraphSAINT Steps (--saint_steps): {steps_per_epoch}")
                # Priority 2: explicit fallback steps from CLI.
                elif args.saint_min_steps > 0:
                    steps_per_epoch = int(args.saint_min_steps)
                    natural_steps = max(1, int(np.ceil(fold_train_edges.shape[0] / seed_edge_batch_size)))
                    print(
                        f"[INFO] GraphSAINT Steps (--saint_min_steps): {steps_per_epoch} "
                        f"(natural={natural_steps})"
                    )
                else:
                    # Priority 3: natural steps based on data size.
                    steps_per_epoch = max(1, int(np.ceil(fold_train_edges.shape[0] / seed_edge_batch_size)))
                    print(f"[INFO] GraphSAINT Steps (natural): {steps_per_epoch}")
                (avg_loss, bce_loss, weighted_kl_loss, attn_kl_raw, attn_kl_norm, kl_ratio,
                 kl_weight_eff, prior_conf_mean, epoch_time, had_valid_batch,
                 b_eff_total, b_total, avg_atom_inc, avg_res_inc, avg_entropy, avg_mask_ratio, info_nce,
                 distill_loss, kd_ratio, kd_conf, gate_balance, gate_entropy, total_weighted,
                 gate_atom_mean, gate_res_mean, gate_w_mean,
                 cold_edge_ratio, cold_drug_only_ratio, cold_prot_only_ratio, cold_both_ratio,
                 wC_warm_mean, wC_cold_mean,
                 mp_alpha_cold_mean, mp_alpha_warm_mean,
                 mp_alpha_cold_p50, mp_alpha_cold_p90, mp_alpha_warm_p50, mp_alpha_warm_p90,
                 gate_w_p10, gate_w_p50, gate_w_p90,
                 deg_d_pcts, deg_p_pcts,
                 t_sampling, t_encoder, t_heads, t_backward, t_ema) = train_one_epoch_saint(
                    epoch, fold_idx - 1, model, optimizer, scheduler, criterion,
                    features_drug_tensor, features_protein_tensor,
                    fold_train_edges[:, :2], fold_train_edges[:, 2],
                    atom_indptr, atom_indices, atom_data,
                    res_indptr, res_indices, res_data,
                    drug_atom_ptr, drug_atom_nodes,
                    prot_res_ptr, prot_res_nodes,
                    atom_to_drug_tensor, residue_to_prot_tensor,
                    atom_attn_tensor, residue_attn_tensor,
                    atom_attn_np=atom_attn_np, residue_attn_np=residue_attn_np,
                    drug_total_count=drug_total_count, prot_total_count=prot_total_count,
                    attn_db=attn_db,
                    drug_id_list=drug_id_list,
                    prot_id_list=prot_id_list,
                    atom_orig_pos=atom_orig_pos,
                    residue_orig_pos=residue_orig_pos,
                    train_losses=train_losses,
                    device=device,
                    seed_edge_batch_size=seed_edge_batch_size,
                    steps_per_epoch=steps_per_epoch,
                    walk_length=walk_length,
                    num_walks=num_walks,
                    max_atoms_per_step=max_atoms_per_step,
                    max_res_per_step=max_res_per_step,
                    alpha_eps=args.alpha_eps,
                    edge_min_incidence_atom=edge_min_incidence_atom,
                    edge_min_incidence_res=edge_min_incidence_res,
                    scaler=scaler, log_interval=log_interval, use_reweight=use_reweight,
                    drug_degree=drug_deg_tensor, prot_degree=prot_deg_tensor, use_coldstart_gate=use_coldstart_gate,
                    info_nce_on=args.info_nce_on,
                    subpocket_mask_ratio=args.subpocket_mask_ratio,
                    subpocket_keep_top=args.subpocket_keep_top,
                    ema_state=ema_state, ema_decay=args.ema_decay, use_ema=use_ema,
                    prot_esm_missing=prot_esm_missing_tensor,
                    prot_esm_unreliable=prot_esm_unreliable_tensor,
                    drug_knn_edge_index=drug_knn_edge_index_t,
                    drug_knn_edge_weight=drug_knn_edge_weight_t,
                    prot_knn_edge_index=prot_knn_edge_index_t,
                    prot_knn_edge_weight=prot_knn_edge_weight_t,
                    cold_deg_th_drug=cold_deg_th_drug_eff,
                    cold_deg_th_prot=cold_deg_th_prot_eff,
                    cold_prot_weight=args.cold_prot_weight,
                    seed_base=args.seed,
                    expertC_scale=expertC_scale,
                    wC_cap=args.wC_cap,
                    debug_assertions=debug_assertions,
                    kl_hard_assert=kl_hard_assert,
                    cold_start_dropout_p=args.cold_start_dropout_p,
                    cold_start_dropout_assert_ratio=args.cold_start_dropout_assert_ratio,
                )
                auc_val = None
                aupr_val = None
                cold_stats = {}
                coverage = None
                val_metric_eligible = False
                if do_validation:
                    ema_backup = None
                    if use_ema and ema_eval and ema_state is not None:
                        ema_backup = swap_ema_weights(model, ema_state)
                    if use_hyperedge_head:
                        auc_val, aupr_val, coverage, cold_stats = evaluate_subgraph_edges(
                            model,
                            features_drug_tensor, features_protein_tensor,
                            atom_indptr, atom_indices, atom_data,
                            res_indptr, res_indices, res_data,
                            val_edges_np_eval[:, :2], val_labels_np_eval,
                            drug_atom_ptr, drug_atom_nodes,
                            prot_res_ptr, prot_res_nodes,
                            atom_orig_pos, residue_orig_pos,
                            atom_to_drug_tensor, residue_to_prot_tensor,
                            atom_attn_tensor, residue_attn_tensor,
                            atom_attn_np, residue_attn_np,
                            attn_db, drug_id_list, prot_id_list,
                            seed_edge_batch_size, walk_length, num_walks,
                            max_atoms_per_step, max_res_per_step,
                            alpha_eps=args.alpha_eps,
                            edge_min_incidence_atom=edge_min_incidence_atom,
                            edge_min_incidence_res=edge_min_incidence_res,
                            eval_num_samples=eval_num_samples,
                            device=device,
                            rng_seed=args.seed + epoch * 13 + fold_idx * 101,
                            drug_degree=drug_deg_tensor,
                            prot_degree=prot_deg_tensor,
                            use_coldstart_gate=use_coldstart_gate,
                            prot_esm_missing=prot_esm_missing_tensor,
                            prot_esm_unreliable=prot_esm_unreliable_tensor,
                            drug_knn_edge_index=drug_knn_edge_index_t,
                            drug_knn_edge_weight=drug_knn_edge_weight_t,
                            prot_knn_edge_index=prot_knn_edge_index_t,
                            prot_knn_edge_weight=prot_knn_edge_weight_t,
                            cold_deg_th_drug=cold_deg_th_drug_eff,
                            cold_deg_th_prot=cold_deg_th_prot_eff,
                            expertC_scale=1.0,
                            wC_cap=args.wC_cap,
                            eval_score_centering=args.eval_score_centering,
                            eval_warm_support_k=args.eval_warm_support_k,
                            eval_warm_support_max_add=args.eval_warm_support_max_add,
                        )
                        if coverage < float(args.eval_coverage_min):
                            print(
                                f"[WARN] Eval coverage low: {coverage:.3f} < min={float(args.eval_coverage_min):.3f}; "
                                "skip model-selection update this epoch."
                            )
                        else:
                            val_metric_eligible = auc_val is not None
                        print(f"[INFO] val_eval_samples={eval_num_samples} (GraphSAINT)")
                    else:
                        auc_val, aupr_val, cold_stats = evaluate_full_graph(
                            model,
                            features_drug_tensor, features_protein_tensor,
                            G_drug_tensor, G_protein_tensor,
                            val_edges_tensor, val_labels_tensor,
                            drug_node_to_entity=atom_to_drug_tensor,
                            protein_node_to_entity=residue_to_prot_tensor,
                            drug_node_weight=atom_attn_tensor,
                            protein_node_weight=residue_attn_tensor,
                            atom_prior=atom_attn_tensor,
                            res_prior=residue_attn_tensor,
                            drug_degree=drug_deg_tensor,
                            prot_degree=prot_deg_tensor,
                            use_coldstart_gate=use_coldstart_gate,
                            prot_esm_missing=prot_esm_missing_tensor,
                            prot_esm_unreliable=prot_esm_unreliable_tensor,
                            drug_knn_edge_index=drug_knn_edge_index_t,
                            drug_knn_edge_weight=drug_knn_edge_weight_t,
                            prot_knn_edge_index=prot_knn_edge_index_t,
                            prot_knn_edge_weight=prot_knn_edge_weight_t,
                            cold_deg_th_drug=cold_deg_th_drug_eff,
                            cold_deg_th_prot=cold_deg_th_prot_eff,
                            scaler=scaler,
                            eval_score_centering=args.eval_score_centering,
                        )
                        val_metric_eligible = auc_val is not None
                        print("[INFO] val_eval_mode=full")
                    if ema_backup is not None:
                        restore_weights(model, ema_backup)
                    run_truth_radar(epoch)
                    _print_global_metric_variants(cold_stats, prefix="VAL")
                    if val_metric_eligible and auc_val is not None and auc_val > best_val_metric:
                        best_val_metric = auc_val
                        patience_counter = 0
                        best_auc, best_aupr = auc_val, aupr_val
                        torch.save(model.state_dict(), f"best_model_fold{fold_idx}.pth")
                        if auc_val > best_global_auc:
                            best_global_auc = auc_val
                            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    elif val_metric_eligible and auc_val is not None:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            break
                    elif coverage is not None and coverage < float(args.eval_coverage_min):
                        print(
                            f"[INFO] Skip best-model/patience update due low coverage "
                            f"({coverage:.3f} < {float(args.eval_coverage_min):.3f})."
                        )
                else:
                    run_truth_radar(epoch)
                last_epoch_metrics = {
                    "mode": "saint",
                    "epoch": int(epoch),
                    "gate_atom_mean": gate_atom_mean.tolist() if hasattr(gate_atom_mean, "tolist") else [],
                    "gate_res_mean": gate_res_mean.tolist() if hasattr(gate_res_mean, "tolist") else [],
                    "gate_w_mean": gate_w_mean.tolist() if hasattr(gate_w_mean, "tolist") else [],
                    "cold_edge_ratio": float(cold_edge_ratio),
                    "cold_drug_only_ratio": float(cold_drug_only_ratio),
                    "cold_prot_only_ratio": float(cold_prot_only_ratio),
                    "cold_both_ratio": float(cold_both_ratio),
                    "wC_warm_mean": float(wC_warm_mean),
                    "wC_cold_mean": float(wC_cold_mean),
                    "val_auc": float(auc_val) if auc_val is not None else None,
                    "val_aupr": float(aupr_val) if aupr_val is not None else None,
                    "val_coverage": float(coverage) if coverage is not None else None,
                    "val_metric_eligible": bool(val_metric_eligible),
                    "val_cold_stats": cold_stats if isinstance(cold_stats, dict) else {},
                }
                if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metric_eligible and auc_val is not None:
                        scheduler.step(auc_val)
                info_weight = float(getattr(criterion, "info_nce_weight", 0.0))
                distill_weight = float(getattr(criterion, "distill_weight", 0.0))
                gate_bal_weight = float(getattr(criterion, "gate_balance_weight", 0.0))
                gate_ent_weight = float(getattr(criterion, "gate_entropy_weight", 0.0))
                alpha_temp = float(getattr(model, "alpha_temp", getattr(args, "alpha_temp", 0.0)))
                kl_scale = float(getattr(criterion, "_kl_scale_ratio", 1.0))
                kd_w_eff = distill_weight * float(kd_conf)
                total_epoch_loss = float(total_weighted) if torch.is_tensor(total_weighted) else float(total_weighted)
                breakdown = None
                if had_valid_batch:
                    try:
                        breakdown = criterion.get_breakdown()
                    except Exception:
                        breakdown = None
                auc_str = f"{auc_val:.5f}" if auc_val is not None else "N/A"
                aupr_str = f"{aupr_val:.5f}" if aupr_val is not None else "N/A"
                ratio = b_eff_total / max(b_total, 1)
                edge_rate = b_total / max(epoch_time, 1e-9)
                edge_eff_rate = b_eff_total / max(epoch_time, 1e-9)
                aux = getattr(model, "_last_aux", {}) if hasattr(model, "_last_aux") else {}
                kl_enabled_log = float(getattr(criterion, "attn_kl_weight", 0.0)) > 0.0
                if kl_enabled_log and aux:
                    attn_kl_norm_raw = float(aux.get("attn_kl_norm_raw", attn_kl_norm))
                else:
                    attn_kl_norm_raw = float(attn_kl_norm)
                delta_rms = float(aux.get("delta_rms", 0.0)) if aux else 0.0
                delta_rms_weighted = float(aux.get("delta_rms_weighted", 0.0)) if aux else 0.0
                gate_w_str = "/".join([f"{x:.2f}" for x in gate_w_mean.tolist()]) if hasattr(gate_w_mean, "tolist") else str(gate_w_mean)
                profile_str = (
                    f"profile(samp/enc/head/bwd/ema)={t_sampling:.1f}/{t_encoder:.1f}/{t_heads:.1f}/"
                    f"{t_backward:.1f}/{t_ema:.1f}"
                )
                if not use_reweight:
                    print("[WARN] GraphSAINT reweight is disabled; sampling may be biased.")
                print(
                    f"Epoch: {epoch + 1:04d}, loss: {avg_loss:.5f}, time: {epoch_time:.4f}s, "
                    f"speed: {edge_rate:.1f}/{edge_eff_rate:.1f} edges/s, "
                    f"auc_val: {auc_str}, aupr_val: {aupr_str}, "
                    f"bce_loss: {float(bce_loss):.5f}, kl_term: {float(weighted_kl_loss):.5f}, "
                    f"attn_kl_raw: {float(attn_kl_raw):.5f}, "
                    f"attn_kl_norm(raw/clip): {attn_kl_norm_raw:.5f}/{float(attn_kl_norm):.5f}, "
                    f"kl_ratio: {float(kl_ratio):.3f}, kl_w_eff: {float(kl_weight_eff):.4f}, "
                    f"kl_scale: {kl_scale:.3f}, prior_conf: {float(prior_conf_mean):.3f}, "
                    f"info_nce: {float(info_nce):.5f}, "
                    f"alpha_temp: {alpha_temp:.3f}, gate_bal_w: {gate_bal_weight:.3f}, "
                    f"distill: {float(distill_loss):.5f}, kd_ratio: {float(kd_ratio):.3f}, kd_conf: {float(kd_conf):.3f}, "
                    f"kd_w_eff: {kd_w_eff:.4f}, "
                    f"gate_bal: {float(gate_balance):.5f}, gate_ent: {float(gate_entropy):.5f}, "
                    f"delta_rms: {delta_rms:.4f}, delta_rms_w: {delta_rms_weighted:.4f}, "
                    f"subpocket_mask: {avg_mask_ratio:.3f}, total_loss: {float(total_epoch_loss):.5f}, "
                    f"B_eff/B={b_eff_total}/{b_total} ({ratio:.3f}), "
                    f"inc(atom/res)={avg_atom_inc:.1f}/{avg_res_inc:.1f}, "
                    f"H_raw={avg_entropy:.4f}, gate_w={gate_w_str}, "
                    f"cold_node_ratio={cold_node_ratio_drug:.3f}/{cold_node_ratio_prot:.3f}, "
                    f"{profile_str}"
                )
                if breakdown is not None:
                    print(
                        f"[LOSS] bce={float(breakdown['bce_raw']):.5f}/{float(breakdown['bce_weighted']):.5f} "
                        f"kl={float(breakdown['kl_raw']):.5f}/{float(breakdown['kl_weighted']):.5f} "
                        f"kd={float(breakdown['distill_raw']):.5f}/{float(breakdown['distill_weighted']):.5f} "
                        f"info={float(breakdown['info_nce_raw']):.5f}/{float(breakdown['info_nce_weighted']):.5f} "
                        f"gate={float(breakdown['gate_balance_raw']):.5f}/{float(breakdown['gate_balance_weighted']):.5f} "
                        f"ent={float(breakdown['gate_entropy_raw']):.5f}/{float(breakdown['gate_entropy_weighted']):.5f} "
                        f"delta={float(breakdown['delta_reg_raw']):.5f}/{float(breakdown['delta_reg_weighted']):.5f} "
                        f"total={float(breakdown['total']):.5f}"
                    )
                if drug_missing_flag_enabled or esm_mismatch_count or esm_missing_node_ratio is not None or esm_unrel_node_ratio is not None:
                    esm_missing_str = f"{esm_missing_node_ratio:.4f}" if esm_missing_node_ratio is not None else "N/A"
                    esm_unrel_str = f"{esm_unrel_node_ratio:.4f}" if esm_unrel_node_ratio is not None else "N/A"
                    print(
                        f"[DATA] drug_desc_missing_ratio={drug_desc_missing_ratio:.4f} "
                        f"drug_feat_missing_ratio={drug_feat_missing_ratio:.4f} "
                        f"esm_mismatch_count={esm_mismatch_count} "
                        f"esm_missing_ratio={esm_missing_str} "
                        f"esm_unreliable_ratio={esm_unrel_str}"
                    )
                print(
                    f"[COLD-RATIO] edge={cold_edge_ratio:.4f} drug_only={cold_drug_only_ratio:.4f} "
                    f"prot_only={cold_prot_only_ratio:.4f} both={cold_both_ratio:.4f}"
                )
                print(
                    f"[DEG-BATCH] drug_pcts(p0/p10/p50/p90/p100)={np.round(deg_d_pcts, 3).tolist()} "
                    f"prot_pcts(p0/p10/p50/p90/p100)={np.round(deg_p_pcts, 3).tolist()}"
                )
                gate_w_warm_mean = aux.get("gate_w_warm_mean", [0.0, 0.0, 0.0]) if aux else [0.0, 0.0, 0.0]
                gate_w_cold_mean = aux.get("gate_w_cold_mean", [0.0, 0.0, 0.0]) if aux else [0.0, 0.0, 0.0]
                print(
                    f"[GATE] warm_mean={np.round(gate_w_warm_mean, 3).tolist()} "
                    f"cold_mean={np.round(gate_w_cold_mean, 3).tolist()} "
                    f"wC_warm_mean={wC_warm_mean:.4f} wC_cold_mean={wC_cold_mean:.4f} "
                    f"wC_mean={float(aux.get('wC_mean', 0.0)):.4f} wC_scale={float(aux.get('wC_global_scale', 1.0)):.3f} "
                    f"gate_w_p50={np.round(gate_w_p50, 3).tolist()} gate_w_p90={np.round(gate_w_p90, 3).tolist()}"
                )
                print(
                    f"[MPGATE] alpha_cold_mean={mp_alpha_cold_mean:.4f} alpha_warm_mean={mp_alpha_warm_mean:.4f} "
                    f"cold_p50/p90={mp_alpha_cold_p50:.3f}/{mp_alpha_cold_p90:.3f} "
                    f"warm_p50/p90={mp_alpha_warm_p50:.3f}/{mp_alpha_warm_p90:.3f}"
                )
                if cold_edge_ratio > 0.35:
                    print("[WARN] cold_edge_ratio > 0.35; check degree source (should be GLOBAL_TRAIN_ONLY).")
                if wC_warm_mean > 1e-3:
                    msg = (
                        f"[ASSERT] wC_warm_mean={wC_warm_mean:.6f} > 1e-3; "
                        "warm edges should suppress ExpertC."
                    )
                    if kl_hard_assert:
                        raise RuntimeError(msg)
                    print(f"[WARN] {msg}")
                if aux:
                    ent_atom_norm = float(aux.get("attn_entropy_atom_norm_mean", 0.0))
                    ent_res_norm = float(aux.get("attn_entropy_res_norm_mean", 0.0))
                    ent_atom_raw = float(aux.get("attn_entropy_atom_raw_mean", 0.0))
                    ent_res_raw = float(aux.get("attn_entropy_res_raw_mean", 0.0))
                    logk_atom_mean = float(aux.get("logK_atom_mean", 0.0))
                    logk_res_mean = float(aux.get("logK_res_mean", 0.0))
                    print(
                        f"[DIAG] H_norm(atom/res)={ent_atom_norm:.4f}/{ent_res_norm:.4f} "
                        f"H_raw(atom/res)={ent_atom_raw:.4f}/{ent_res_raw:.4f} "
                        f"logK(atom/res)={logk_atom_mean:.3f}/{logk_res_mean:.3f}"
                    )
                    logits_atom_pre_mean = float(aux.get("logits_atom_pre_mean", 0.0))
                    logits_atom_pre_std = float(aux.get("logits_atom_pre_std", 0.0))
                    logits_atom_post_mean = float(aux.get("logits_atom_post_mean", 0.0))
                    logits_atom_post_std = float(aux.get("logits_atom_post_std", 0.0))
                    logits_res_pre_mean = float(aux.get("logits_res_pre_mean", 0.0))
                    logits_res_pre_std = float(aux.get("logits_res_pre_std", 0.0))
                    logits_res_post_mean = float(aux.get("logits_res_post_mean", 0.0))
                    logits_res_post_std = float(aux.get("logits_res_post_std", 0.0))
                    print(
                        f"[DIAG] logits_pre(atom/res)={logits_atom_pre_mean:.4f}±{logits_atom_pre_std:.4f}/"
                        f"{logits_res_pre_mean:.4f}±{logits_res_pre_std:.4f} "
                        f"logits_post(atom/res)={logits_atom_post_mean:.4f}±{logits_atom_post_std:.4f}/"
                        f"{logits_res_post_mean:.4f}±{logits_res_post_std:.4f}"
                    )
                    print(
                        f"[DIAG] prior_ent_norm(atom/res)={aux.get('prior_entropy_atom_norm_mean', 0.0):.4f}/"
                        f"{aux.get('prior_entropy_res_norm_mean', 0.0):.4f} "
                        f"prior_ent_raw(atom/res)={aux.get('prior_entropy_atom_raw_mean', 0.0):.4f}/"
                        f"{aux.get('prior_entropy_res_raw_mean', 0.0):.4f}"
                    )
                    print(
                        f"[DIAG] prior_conf(atom/res)={aux.get('prior_conf_atom_mean', 0.0):.4f}/"
                        f"{aux.get('prior_conf_res_mean', 0.0):.4f} "
                        f"K_atom(mean/p50/p90/max)={aux.get('group_size_atom_mean', 0.0):.1f}/"
                        f"{aux.get('group_size_atom_p50', 0.0):.1f}/"
                        f"{aux.get('group_size_atom_p90', 0.0):.1f}/"
                        f"{aux.get('group_size_atom_max', 0.0):.1f} "
                        f"K_res(mean/p50/p90/max)={aux.get('group_size_res_mean', 0.0):.1f}/"
                        f"{aux.get('group_size_res_p50', 0.0):.1f}/"
                        f"{aux.get('group_size_res_p90', 0.0):.1f}/"
                        f"{aux.get('group_size_res_max', 0.0):.1f} "
                        f"wC_warm={aux.get('wC_warm_mean', 0.0):.4f} cold_edge_ratio={aux.get('cold_edge_ratio', 0.0):.4f}"
                    )
                    print(
                        f"[DIAG] prior_q(atom/res)={aux.get('prior_atom_min', 0.0):.3e}/"
                        f"{aux.get('prior_atom_p50', 0.0):.3e}/"
                        f"{aux.get('prior_atom_p90', 0.0):.3e} "
                        f"{aux.get('prior_res_min', 0.0):.3e}/"
                        f"{aux.get('prior_res_p50', 0.0):.3e}/"
                        f"{aux.get('prior_res_p90', 0.0):.3e} "
                        f"alpha_q(atom/res)={aux.get('alpha_atom_min', 0.0):.3e}/"
                        f"{aux.get('alpha_atom_p50', 0.0):.3e}/"
                        f"{aux.get('alpha_atom_p90', 0.0):.3e} "
                        f"{aux.get('alpha_res_min', 0.0):.3e}/"
                        f"{aux.get('alpha_res_p50', 0.0):.3e}/"
                        f"{aux.get('alpha_res_p90', 0.0):.3e} "
                        f"prior_eps={float(args.prior_eps):.1e}"
                    )
                    print(
                        f"[DIAG] KL_max(atom/res): gid={aux.get('kl_atom_max_group_id', -1)}/"
                        f"{aux.get('kl_res_max_group_id', -1)} "
                        f"K={aux.get('kl_atom_max_group_k', 0.0):.0f}/"
                        f"{aux.get('kl_res_max_group_k', 0.0):.0f} "
                        f"prior_conf={aux.get('kl_atom_max_group_prior_conf', 0.0):.3f}/"
                        f"{aux.get('kl_res_max_group_prior_conf', 0.0):.3f} "
                        f"deg={aux.get('kl_atom_max_group_deg', 0.0):.1f}/"
                        f"{aux.get('kl_res_max_group_deg', 0.0):.1f} "
                        f"kl_g={aux.get('kl_atom_max_group_kl', 0.0):.4f}/"
                        f"{aux.get('kl_res_max_group_kl', 0.0):.4f}"
                    )
                    print(
                        f"[DIAG] prior_max_p50/p90(atom)={aux.get('prior_max_prob_atom_p50', 0.0):.3f}/"
                        f"{aux.get('prior_max_prob_atom_p90', 0.0):.3f} "
                        f"prior_max_p50/p90(res)={aux.get('prior_max_prob_res_p50', 0.0):.3f}/"
                        f"{aux.get('prior_max_prob_res_p90', 0.0):.3f} "
                        f"logits_pre(atom/res)={aux.get('logits_atom_pre_mean', 0.0):.3f}/"
                        f"{aux.get('logits_res_pre_mean', 0.0):.3f} "
                        f"logits_post(atom/res)={aux.get('logits_atom_post_mean', 0.0):.3f}/"
                        f"{aux.get('logits_res_post_mean', 0.0):.3f}"
                    )
                    if (
                        aux.get("prior_entropy_atom_norm_mean", 0.0) > 0.98
                        or aux.get("prior_entropy_res_norm_mean", 0.0) > 0.98
                    ):
                        print(
                            "[WARN] prior is near-uniform; KL will be down-weighted. "
                            f"prior_max_p50/p90(atom)={aux.get('prior_max_prob_atom_p50', 0.0):.3f}/"
                            f"{aux.get('prior_max_prob_atom_p90', 0.0):.3f} "
                            f"prior_max_p50/p90(res)={aux.get('prior_max_prob_res_p50', 0.0):.3f}/"
                            f"{aux.get('prior_max_prob_res_p90', 0.0):.3f}"
                        )
                    if debug_assertions:
                        for key in (
                            "attn_entropy_atom_norm_mean",
                            "attn_entropy_res_norm_mean",
                            "prior_entropy_atom_norm_mean",
                            "prior_entropy_res_norm_mean",
                        ):
                            val = float(aux.get(key, 0.0))
                            if val < -1e-3 or val > 1.0 + 1e-3:
                                print(f"[WARN] {key} out of [0,1]: {val:.4f}")
                        if logk_atom_mean > 0 and ent_atom_raw > logk_atom_mean + 1e-3:
                            print(f"[WARN] atom entropy_raw {ent_atom_raw:.4f} > logK {logk_atom_mean:.4f}")
                        if logk_res_mean > 0 and ent_res_raw > logk_res_mean + 1e-3:
                            print(f"[WARN] res entropy_raw {ent_res_raw:.4f} > logK {logk_res_mean:.4f}")
                        if (ent_atom_norm > 0.97 or ent_res_norm > 0.97) and float(attn_kl_norm) > 0.5:
                            msg = (
                                "entropy_norm near 1.0 but KL_norm high; "
                                "check KL normalization/grouping."
                            )
                            if kl_hard_assert:
                                raise RuntimeError(msg)
                            print(f"[WARN] {msg}")
                        atom_ref = max(int(args.pool_topk), int(args.pool_randk), int(atom_keep_ref), 1)
                        res_ref = max(int(args.pool_topk), int(args.pool_randk), int(res_keep_ref), 1)
                        if float(aux.get("group_size_atom_max", 0.0)) > atom_ref * 4:
                            print(
                                f"[WARN] atom group_size_max={aux.get('group_size_atom_max', 0.0):.1f} "
                                f"> {atom_ref * 4}; check grouping."
                            )
                        if float(aux.get("group_size_res_max", 0.0)) > res_ref * 4:
                            print(
                                f"[WARN] res group_size_max={aux.get('group_size_res_max', 0.0):.1f} "
                                f"> {res_ref * 4}; check grouping."
                            )
                    if ent_atom_norm > 0.97:
                        high_ent_atom_epochs += 1
                    else:
                        high_ent_atom_epochs = 0
                    if ent_res_norm > 0.97:
                        high_ent_res_epochs += 1
                    else:
                        high_ent_res_epochs = 0
                    if high_ent_atom_epochs >= 3 or high_ent_res_epochs >= 3:
                        print(
                            "[WARN] attention entropy_norm >0.97 for >=3 epochs; "
                            "attention may be too uniform."
                        )
                print(
                    f"[MPGATE] atom_bins={np.round(gate_atom_mean, 4).tolist()} "
                    f"res_bins={np.round(gate_res_mean, 4).tolist()}"
                )
                if cold_stats:
                    cd = cold_stats.get("cold_drug", {})
                    cp = cold_stats.get("cold_prot", {})
                    cb = cold_stats.get("cold_both", {})
                    print(
                        f"[COLD] drug AUC/AUPR={cd.get('auc', 0.0):.4f}/{cd.get('aupr', 0.0):.4f} n={cd.get('n', 0)}; "
                        f"prot AUC/AUPR={cp.get('auc', 0.0):.4f}/{cp.get('aupr', 0.0):.4f} n={cp.get('n', 0)}; "
                        f"both AUC/AUPR={cb.get('auc', 0.0):.4f}/{cb.get('aupr', 0.0):.4f} n={cb.get('n', 0)}"
                    )
                    em = cold_stats.get("esm_missing", {})
                    if em:
                        print(
                            f"[ESM-MISS] AUC/AUPR={em.get('auc', 0.0):.4f}/{em.get('aupr', 0.0):.4f} "
                            f"n={em.get('n', 0)} ratio={em.get('ratio', 0.0):.4f}"
                        )
                    eu = cold_stats.get("esm_unreliable", {})
                    if eu:
                        print(
                            f"[ESM-UNREL] AUC/AUPR={eu.get('auc', 0.0):.4f}/{eu.get('aupr', 0.0):.4f} "
                            f"n={eu.get('n', 0)} ratio={eu.get('ratio', 0.0):.4f}"
                        )
                    if not cold_ablation_note_printed:
                        print(
                            "[INFO] cold-protein ablation: set --cold_deg_th_prot 0 "
                            "(or --cold_mode quantile with --cold_q 0) to disable cold-prot split in metrics."
                        )
                        cold_ablation_note_printed = True
                if t_heads / max(epoch_time, 1e-9) > 0.25:
                    print(f"[WARN] heads forward time ratio high: {t_heads / max(epoch_time, 1e-9):.2f}")
                if epoch_time > float(args.epoch_time_budget_sec):
                    msg = (
                        f"epoch_time {epoch_time:.2f}s exceeds budget {args.epoch_time_budget_sec}s; "
                        f"profile(samp/enc/head/bwd/ema)={t_sampling:.1f}/{t_encoder:.1f}/{t_heads:.1f}/"
                        f"{t_backward:.1f}/{t_ema:.1f}"
                    )
                    print(f"[WARN] {msg}")
            else:
                auc_val, aupr_val, did_validate, extra_metrics = train_one_epoch(
                    epoch, fold_idx - 1, model, optimizer, scheduler, criterion,
                    features_drug_tensor, features_protein_tensor,
                    G_drug_tensor, G_protein_tensor,
                    train_edges_tensor, train_labels_tensor,
                    val_edges_tensor, val_labels_tensor,
                    train_losses, device, batch_size,
                    scaler=scaler, do_validation=do_validation,
                    drug_node_to_entity=atom_to_drug_tensor,
                    protein_node_to_entity=residue_to_prot_tensor,
                    drug_node_weight=atom_attn_tensor,
                    protein_node_weight=residue_attn_tensor,
                    atom_prior=atom_attn_tensor,
                    res_prior=residue_attn_tensor,
                    drug_degree=drug_deg_tensor,
                    prot_degree=prot_deg_tensor,
                    use_coldstart_gate=use_coldstart_gate,
                    num_workers=args.num_workers,
                    info_nce_on=args.info_nce_on,
                    ema_state=ema_state, ema_decay=args.ema_decay, use_ema=use_ema, ema_eval=ema_eval,
                    prot_esm_missing=prot_esm_missing_tensor,
                    prot_esm_unreliable=prot_esm_unreliable_tensor,
                    drug_knn_edge_index=drug_knn_edge_index_t,
                    drug_knn_edge_weight=drug_knn_edge_weight_t,
                    prot_knn_edge_index=prot_knn_edge_index_t,
                    prot_knn_edge_weight=prot_knn_edge_weight_t,
                    cold_deg_th_drug=cold_deg_th_drug_eff,
                    cold_deg_th_prot=cold_deg_th_prot_eff,
                    cold_prot_weight=args.cold_prot_weight,
                    expertC_scale=expertC_scale,
                    wC_cap=args.wC_cap,
                    seed_base=args.seed,
                    eval_score_centering=args.eval_score_centering,
                )
                run_truth_radar(epoch)
                if did_validate:
                    if auc_val > best_val_metric:
                        best_val_metric = auc_val
                        patience_counter = 0
                        best_auc, best_aupr = auc_val, aupr_val
                        torch.save(model.state_dict(), f"best_model_fold{fold_idx}.pth")
                        if auc_val > best_global_auc:
                            best_global_auc = auc_val
                            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            break
                    if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if auc_val is not None:
                            scheduler.step(auc_val)
                if extra_metrics:
                    last_epoch_metrics = dict(extra_metrics)
                    last_epoch_metrics["mode"] = "full"
                    last_epoch_metrics["epoch"] = int(epoch)
                    cold_stats = extra_metrics.get("cold_stats", {})
                    _print_global_metric_variants(cold_stats, prefix="VAL")
                    if cold_stats:
                        cd = cold_stats.get("cold_drug", {})
                        cp = cold_stats.get("cold_prot", {})
                        cb = cold_stats.get("cold_both", {})
                        print(
                            f"[COLD] drug AUC/AUPR={cd.get('auc', 0.0):.4f}/{cd.get('aupr', 0.0):.4f} n={cd.get('n', 0)}; "
                            f"prot AUC/AUPR={cp.get('auc', 0.0):.4f}/{cp.get('aupr', 0.0):.4f} n={cp.get('n', 0)}; "
                            f"both AUC/AUPR={cb.get('auc', 0.0):.4f}/{cb.get('aupr', 0.0):.4f} n={cb.get('n', 0)}"
                        )
                        em = cold_stats.get("esm_missing", {})
                        if em:
                            print(
                                f"[ESM-MISS] AUC/AUPR={em.get('auc', 0.0):.4f}/{em.get('aupr', 0.0):.4f} "
                                f"n={em.get('n', 0)} ratio={em.get('ratio', 0.0):.4f}"
                            )
                        eu = cold_stats.get("esm_unreliable", {})
                        if eu:
                            print(
                                f"[ESM-UNREL] AUC/AUPR={eu.get('auc', 0.0):.4f}/{eu.get('aupr', 0.0):.4f} "
                                f"n={eu.get('n', 0)} ratio={eu.get('ratio', 0.0):.4f}"
                            )
                    epoch_time = float(extra_metrics.get("epoch_time", 0.0))
                    t_heads = float(extra_metrics.get("t_heads", 0.0))
                    if epoch_time > float(args.epoch_time_budget_sec):
                        print(
                            f"[WARN] epoch_time {epoch_time:.2f}s exceeds budget {args.epoch_time_budget_sec}s; "
                            f"profile(enc/head/bwd/ema)={extra_metrics.get('t_encoder', 0.0):.1f}/"
                            f"{t_heads:.1f}/{extra_metrics.get('t_backward', 0.0):.1f}/"
                            f"{extra_metrics.get('t_ema', 0.0):.1f}"
                        )

        auc_list.append(best_auc)
        aupr_list.append(best_aupr)
        fold_metrics.append(
            {
                "fold": int(fold_idx),
                "best_auc": float(best_auc),
                "best_aupr": float(best_aupr),
                "supervised_edges": int(fold_train_edges.shape[0]),
            }
        )

    mean_auc = float(np.mean(auc_list)) if auc_list else 0.0
    std_auc = float(np.std(auc_list)) if auc_list else 0.0
    mean_aupr = float(np.mean(aupr_list)) if aupr_list else 0.0
    std_aupr = float(np.std(aupr_list)) if aupr_list else 0.0
    print("\n" + "=" * 50)
    print("Final results")
    print("=" * 50)
    print(f"Mean AUC:  {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"Mean AUPR: {mean_aupr:.5f} +/- {std_aupr:.5f}")
    print(f"Total time: {time.time() - time_start:.5f}s")
    test_auc = None
    test_aupr = None
    test_cold_stats = {}
    if eval_mode == "fixed" and best_model_state is not None and test_labels_tensor_full.numel() > 0:
        model.load_state_dict(best_model_state)
        if use_hyperedge_head:
            test_auc, test_aupr, coverage, cold_stats = evaluate_subgraph_edges(
                model,
                features_drug_tensor, features_protein_tensor,
                atom_indptr, atom_indices, atom_data,
                res_indptr, res_indices, res_data,
                test_edges_np[:, :2], test_edges_np[:, 2],
                drug_atom_ptr, drug_atom_nodes,
                prot_res_ptr, prot_res_nodes,
                atom_orig_pos, residue_orig_pos,
                atom_to_drug_tensor, residue_to_prot_tensor,
                atom_attn_tensor, residue_attn_tensor,
                atom_attn_np, residue_attn_np,
                attn_db, drug_id_list, prot_id_list,
                seed_edge_batch_size, walk_length, num_walks,
                max_atoms_per_step, max_res_per_step,
                alpha_eps=args.alpha_eps,
                edge_min_incidence_atom=edge_min_incidence_atom,
                edge_min_incidence_res=edge_min_incidence_res,
                eval_num_samples=eval_num_samples,
                device=device,
                rng_seed=args.seed + 999,
                drug_degree=drug_deg_tensor,
                prot_degree=prot_deg_tensor,
                use_coldstart_gate=use_coldstart_gate,
                prot_esm_missing=prot_esm_missing_tensor,
                prot_esm_unreliable=prot_esm_unreliable_tensor,
                drug_knn_edge_index=drug_knn_edge_index_t,
                drug_knn_edge_weight=drug_knn_edge_weight_t,
                prot_knn_edge_index=prot_knn_edge_index_t,
                prot_knn_edge_weight=prot_knn_edge_weight_t,
                cold_deg_th_drug=cold_deg_th_drug_eff,
                cold_deg_th_prot=cold_deg_th_prot_eff,
                expertC_scale=1.0,
                wC_cap=args.wC_cap,
                eval_score_centering=args.eval_score_centering,
                eval_warm_support_k=args.eval_warm_support_k,
                eval_warm_support_max_add=args.eval_warm_support_max_add,
            )
            if coverage < float(args.eval_coverage_min):
                print(f"[WARN] Test coverage low: {coverage:.3f}")
        else:
            test_auc, test_aupr, cold_stats = evaluate_full_graph(
                model,
                features_drug_tensor, features_protein_tensor,
                G_drug_tensor, G_protein_tensor,
                test_edges_tensor_full, test_labels_tensor_full,
                drug_node_to_entity=atom_to_drug_tensor,
                protein_node_to_entity=residue_to_prot_tensor,
                drug_node_weight=atom_attn_tensor,
                protein_node_weight=residue_attn_tensor,
                atom_prior=atom_attn_tensor,
                res_prior=residue_attn_tensor,
                drug_degree=drug_deg_tensor,
                prot_degree=prot_deg_tensor,
                use_coldstart_gate=use_coldstart_gate,
                prot_esm_missing=prot_esm_missing_tensor,
                prot_esm_unreliable=prot_esm_unreliable_tensor,
                drug_knn_edge_index=drug_knn_edge_index_t,
                drug_knn_edge_weight=drug_knn_edge_weight_t,
                prot_knn_edge_index=prot_knn_edge_index_t,
                prot_knn_edge_weight=prot_knn_edge_weight_t,
                cold_deg_th_drug=cold_deg_th_drug_eff,
                cold_deg_th_prot=cold_deg_th_prot_eff,
                expertC_scale=1.0,
                wC_cap=args.wC_cap,
                scaler=scaler,
                eval_score_centering=args.eval_score_centering,
            )
        test_cold_stats = cold_stats if isinstance(cold_stats, dict) else {}
        print(f"Test AUC:  {test_auc:.5f}")
        print(f"Test AUPR: {test_aupr:.5f}")
        _print_global_metric_variants(test_cold_stats, prefix="TEST")
        if cold_stats:
            cd = cold_stats.get("cold_drug", {})
            cp = cold_stats.get("cold_prot", {})
            cb = cold_stats.get("cold_both", {})
            print(
                f"[COLD-TEST] drug AUC/AUPR={cd.get('auc', 0.0):.4f}/{cd.get('aupr', 0.0):.4f} n={cd.get('n', 0)}; "
                f"prot AUC/AUPR={cp.get('auc', 0.0):.4f}/{cp.get('aupr', 0.0):.4f} n={cp.get('n', 0)}; "
                f"both AUC/AUPR={cb.get('auc', 0.0):.4f}/{cb.get('aupr', 0.0):.4f} n={cb.get('n', 0)}"
            )
            em = cold_stats.get("esm_missing", {})
            if em:
                print(
                    f"[ESM-MISS-TEST] AUC/AUPR={em.get('auc', 0.0):.4f}/{em.get('aupr', 0.0):.4f} "
                    f"n={em.get('n', 0)} ratio={em.get('ratio', 0.0):.4f}"
                )
            eu = cold_stats.get("esm_unreliable", {})
            if eu:
                print(
                    f"[ESM-UNREL-TEST] AUC/AUPR={eu.get('auc', 0.0):.4f}/{eu.get('aupr', 0.0):.4f} "
                    f"n={eu.get('n', 0)} ratio={eu.get('ratio', 0.0):.4f}"
                )

    write_metrics_out(
        args.metrics_out,
        {
            "mode": "train",
            "dataset": str(args.dataset),
            "seed": int(args.seed),
            "eval_mode": str(eval_mode),
            "node_level": str(args.node_level),
            "num_folds": int(num_folds),
            "auc_list": [float(x) for x in auc_list],
            "aupr_list": [float(x) for x in aupr_list],
            "fold_metrics": fold_metrics,
            "mean_auc": float(mean_auc),
            "std_auc": float(std_auc),
            "mean_aupr": float(mean_aupr),
            "std_aupr": float(std_aupr),
            "test_auc": float(test_auc) if test_auc is not None else None,
            "test_aupr": float(test_aupr) if test_aupr is not None else None,
            "test_cold_stats": test_cold_stats,
            "last_epoch_metrics": last_epoch_metrics,
        },
    )

    loss_records = []
    for fold_idx, losses in enumerate(train_losses, start=1):
        for epoch_idx, loss_val in enumerate(losses, start=1):
            loss_records.append({"fold": fold_idx, "epoch": epoch_idx, "loss": loss_val})
    if loss_records:
        pd.DataFrame(loss_records).to_csv("train_loss_curve.csv", index=False)

    fig, axes = plt.subplots(num_folds, 1, figsize=(10, 5 * num_folds))
    for i in range(num_folds):
        ax = axes[i] if num_folds > 1 else axes
        ax.plot(train_losses[i], label="Training Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title(f"Fold {i + 1} Training Loss")
    plt.tight_layout()
    plt.savefig("train_loss.png")
    plt.show()

    final_model_path = "final_model.pth"
    if best_model_state is not None:
        torch.save(best_model_state, final_model_path)
        print(f"Best model saved to {final_model_path}")
    else:
        print("No best model saved.")
