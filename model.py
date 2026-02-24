"""Model definitions for HGACN, including grouped utilities, hypergraph layers, and the full HGACN network."""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def group_sum(x, group, num_groups):
    """Sum values by group index."""
    if x.dim() == 1:
        out = x.new_zeros((num_groups,))
        out.index_add_(0, group, x)
        return out
    out = x.new_zeros((num_groups, x.size(1)))
    out.index_add_(0, group, x)
    return out


def normalize_by_group(prior, group, num_groups, eps=1e-12):
    """Normalize values independently inside each group."""
    denom = group_sum(prior, group, num_groups)
    return prior / denom[group].clamp_min(eps)


def group_softmax(logits, group, num_groups):
    """Compute softmax within each group."""
    if logits.numel() == 0:
        return logits
    max_per_group = logits.new_full((num_groups,), -float("inf"))
    if hasattr(max_per_group, "scatter_reduce_"):
        max_per_group.scatter_reduce_(0, group, logits, reduce="amax", include_self=True)
    else:
        uniq = torch.unique(group)
        for g in uniq.tolist():
            mask = group == g
            if mask.any():
                max_per_group[g] = logits[mask].max()
    exp = torch.exp(logits - max_per_group[group])
    denom = group_sum(exp, group, num_groups)
    return exp / denom[group].clamp_min(1e-12)


def group_max(x, group, num_groups):
    """Compute maximum value per group."""
    if x.numel() == 0:
        return x.new_full((num_groups,), -float("inf"))
    max_per_group = x.new_full((num_groups,), -float("inf"))
    if hasattr(max_per_group, "scatter_reduce_"):
        max_per_group.scatter_reduce_(0, group, x, reduce="amax", include_self=True)
    else:
        uniq = torch.unique(group)
        for g in uniq.tolist():
            mask = group == g
            if mask.any():
                max_per_group[g] = x[mask].max()
    return max_per_group


def _group_counts(group, num_groups):
    """Count number of elements in each group."""
    if group.numel() == 0:
        return group.new_zeros((num_groups,), dtype=torch.float32)
    ones = torch.ones_like(group, dtype=torch.float32)
    return group_sum(ones, group, num_groups)


def _sanitize_group_probs(p, group, num_groups, eps=1e-6):
    """Repair invalid probabilities and renormalize per group."""
    if p.numel() == 0:
        return p, 0, 0
    p = p.clone()
    nan_mask = ~torch.isfinite(p)
    nan_count = int(nan_mask.sum().item())
    if nan_mask.any():
        p[nan_mask] = 0.0
    p = p.clamp_min(float(eps))
    denom = group_sum(p, group, num_groups)
    counts = _group_counts(group, num_groups)
    valid_group = counts > 0
    invalid = (~torch.isfinite(denom)) | (denom <= float(eps))
    invalid = invalid & valid_group
    renorm_count = int(invalid.sum().item())
    denom_safe = denom.clone()
    denom_safe[invalid] = 1.0
    p = p / denom_safe[group].clamp_min(float(eps))
    if invalid.any():
        uniform = (1.0 / counts.clamp_min(1.0)).to(dtype=p.dtype, device=p.device)
        mask = invalid[group]
        p[mask] = uniform[group][mask]
    return p, nan_count, renorm_count


def _delta_reg_stats(delta, inv, num_groups, prior_conf_group=None, eps=1e-12):
    """Compute delta regularization statistics by group."""
    if delta is None or delta.numel() == 0 or inv is None or num_groups <= 0:
        zero = delta.new_tensor(0.0) if torch.is_tensor(delta) else torch.tensor(0.0)
        return zero, zero, zero
    delta = delta.view(-1)
    inv = inv.view(-1)
    counts = _group_counts(inv, num_groups).clamp_min(1.0)
    mean = group_sum(delta, inv, num_groups) / counts
    delta_centered = delta - mean[inv]
    delta_sq = delta_centered.pow(2)
    delta_rms = torch.sqrt(delta_sq.mean().clamp_min(eps))
    if prior_conf_group is not None and prior_conf_group.numel() == num_groups:
        weights = prior_conf_group[inv].clamp(0.0, 1.0)
        delta_sq_w = delta_sq * weights
        delta_rms_w = torch.sqrt(delta_sq_w.mean().clamp_min(eps))
        delta_reg = delta_sq_w.mean()
    else:
        delta_rms_w = delta_rms
        delta_reg = delta_sq.mean()
    return delta_reg, delta_rms, delta_rms_w


def smooth_by_group(p, group, num_groups, smooth=0.0):
    """Blend group distribution with uniform smoothing prior."""
    if smooth is None or smooth <= 0.0 or p.numel() == 0:
        return p
    counts = _group_counts(group, num_groups).clamp_min(1.0)
    uniform = (1.0 / counts).to(dtype=p.dtype, device=p.device)
    return (1.0 - smooth) * p + smooth * uniform[group]


def group_entropy(p, group, num_groups, eps=1e-12, normalize=True):
    """Compute (optionally normalized) entropy per group."""
    if p.numel() == 0:
        return p.new_zeros((num_groups,))
    log_p = torch.log(p.clamp_min(eps))
    ent = group_sum(-p * log_p, group, num_groups)
    if not normalize:
        return ent
    counts = _group_counts(group, num_groups).clamp_min(2.0)
    logk = torch.log(counts)
    return ent / logk


def _group_kl_stats(alpha, prior, group, num_groups, alpha_eps=1e-6, prior_eps=1e-6, return_details=False):
    """Compute grouped KL statistics with optional diagnostic details."""
    if alpha.numel() == 0:
        zero = alpha.new_tensor(0.0)
        if return_details:
            empty = alpha.new_zeros((num_groups,))
            return zero, zero, zero, {
                "kl_group": empty,
                "counts": empty,
                "alpha": alpha,
                "prior": prior,
                "logk": empty,
                "alpha_nan_count": 0,
                "prior_nan_count": 0,
                "renorm_count": 0,
            }
        return zero, zero, zero
    alpha, alpha_nan, alpha_renorm = _sanitize_group_probs(alpha, group, num_groups, eps=float(alpha_eps))
    prior, prior_nan, prior_renorm = _sanitize_group_probs(prior, group, num_groups, eps=float(prior_eps))
    log_alpha = torch.log(alpha.clamp_min(float(alpha_eps)))
    log_prior = torch.log(prior.clamp_min(float(prior_eps)))
    kl_group = group_sum(alpha * (log_alpha - log_prior), group, num_groups)
    counts = _group_counts(group, num_groups)
    logk = torch.log(counts.clamp_min(2.0))
    valid = counts > 0
    denom = max(int(valid.sum().item()), 1)
    kl_raw = kl_group[valid].sum() / denom
    kl_norm = (kl_group[valid] / logk[valid]).sum() / denom
    ent_group = group_sum(-alpha * log_alpha, group, num_groups)
    ent = ent_group[valid].sum() / denom
    if return_details:
        return kl_raw, kl_norm, ent, {
            "kl_group": kl_group,
            "counts": counts,
            "alpha": alpha,
            "prior": prior,
            "logk": logk,
            "alpha_nan_count": int(alpha_nan),
            "prior_nan_count": int(prior_nan),
            "renorm_count": int(alpha_renorm + prior_renorm),
        }
    return kl_raw, kl_norm, ent


def _finite_mean_var(x):
    """Return finite-value mean/variance/count for 1D diagnostics."""
    if x is None:
        return 0.0, 0.0, 0
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().to(dtype=torch.float32).view(-1)
    if x.numel() == 0:
        return 0.0, 0.0, 0
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return 0.0, 0.0, 0
    return float(x.mean().item()), float(x.var(unbiased=False).item()), int(x.numel())


def _kl_entropy_sanity(
    alpha_entropy,
    prior_entropy,
    group_entropy,
    alpha_kl,
    prior_kl,
    group_kl,
    num_groups,
    softmax_dim_entropy="group(0)",
    softmax_dim_kl="group(0)",
):
    """Compare entropy-side and KL-side distributions for minimal sanity checks."""
    num_groups = int(num_groups) if num_groups is not None else 0
    ea_mean, ea_var, _ = _finite_mean_var(alpha_entropy)
    ep_mean, ep_var, _ = _finite_mean_var(prior_entropy)
    ka_mean, ka_var, _ = _finite_mean_var(alpha_kl)
    kp_mean, kp_var, _ = _finite_mean_var(prior_kl)

    same_softmax_dim = str(softmax_dim_entropy) == str(softmax_dim_kl)
    same_group = (
        group_entropy is not None
        and group_kl is not None
        and group_entropy.shape == group_kl.shape
        and torch.equal(group_entropy.view(-1), group_kl.view(-1))
    )

    mask_entropy = None
    if group_entropy is not None and num_groups > 0 and group_entropy.numel() > 0:
        mask_entropy = _group_counts(group_entropy.view(-1), num_groups) > 0
    mask_kl = None
    if group_kl is not None and num_groups > 0 and group_kl.numel() > 0:
        mask_kl = _group_counts(group_kl.view(-1), num_groups) > 0
    same_mask = (
        mask_entropy is not None
        and mask_kl is not None
        and mask_entropy.shape == mask_kl.shape
        and torch.equal(mask_entropy, mask_kl)
    )
    entropy_valid_groups = int(mask_entropy.sum().item()) if mask_entropy is not None else 0
    kl_valid_groups = int(mask_kl.sum().item()) if mask_kl is not None else 0
    mismatch = not (same_softmax_dim and same_group and same_mask)
    return {
        "entropy_alpha_mean": ea_mean,
        "entropy_alpha_var": ea_var,
        "entropy_prior_mean": ep_mean,
        "entropy_prior_var": ep_var,
        "kl_alpha_mean": ka_mean,
        "kl_alpha_var": ka_var,
        "kl_prior_mean": kp_mean,
        "kl_prior_var": kp_var,
        "softmax_dim_entropy": str(softmax_dim_entropy),
        "softmax_dim_kl": str(softmax_dim_kl),
        "same_softmax_dim": bool(same_softmax_dim),
        "same_group": bool(same_group),
        "same_mask": bool(same_mask),
        "entropy_valid_groups": int(entropy_valid_groups),
        "kl_valid_groups": int(kl_valid_groups),
        "mismatch": bool(mismatch),
    }


def _edge_ids_from_ptr(edge_ptr):
    """Expand ragged ptr array into per-node edge ids."""
    num_edges = int(edge_ptr.numel() - 1)
    if num_edges <= 0:
        return edge_ptr.new_empty((0,), dtype=torch.long), 0
    counts = (edge_ptr[1:] - edge_ptr[:-1]).clamp_min(0)
    edge_ids = torch.repeat_interleave(torch.arange(num_edges, device=edge_ptr.device), counts)
    return edge_ids, num_edges


def _ragged_edge_aggregate(x, edge_ptr, edge_nodes, edge_weight):
    """Aggregate node features to ragged-edge features."""
    edge_ids, num_edges = _edge_ids_from_ptr(edge_ptr)
    if edge_nodes is not None and edge_ids.numel() != edge_nodes.numel():
        if edge_ids.numel() < edge_nodes.numel():
            edge_nodes = edge_nodes[:edge_ids.numel()]
            if edge_weight is not None and edge_weight.numel() >= edge_ids.numel():
                edge_weight = edge_weight[:edge_ids.numel()]
        else:
            edge_ids = edge_ids[:edge_nodes.numel()]
    if edge_ids.numel() == 0 or edge_nodes.numel() == 0:
        return x.new_zeros((num_edges, x.size(1))), edge_ids, num_edges
    if edge_weight is None or edge_weight.numel() == 0:
        w = torch.ones(edge_nodes.size(0), device=x.device, dtype=x.dtype)
    else:
        w = edge_weight.to(device=x.device, dtype=x.dtype).view(-1)
    x_edge = x.new_zeros((num_edges, x.size(1)))
    x_edge.index_add_(0, edge_ids, x[edge_nodes] * w.view(-1, 1))
    return x_edge, edge_ids, num_edges


def _ragged_edge_broadcast(x_edge, edge_ids, edge_nodes, edge_weight, num_nodes):
    """Broadcast ragged-edge features back to node space."""
    if edge_ids.numel() == 0 or edge_nodes.numel() == 0:
        return x_edge.new_zeros((num_nodes, x_edge.size(1)))
    if edge_ids.numel() != edge_nodes.numel():
        if edge_ids.numel() < edge_nodes.numel():
            edge_nodes = edge_nodes[:edge_ids.numel()]
            if edge_weight is not None and edge_weight.numel() >= edge_ids.numel():
                edge_weight = edge_weight[:edge_ids.numel()]
        else:
            edge_ids = edge_ids[:edge_nodes.numel()]
    if edge_weight is None or edge_weight.numel() == 0:
        w = torch.ones(edge_nodes.size(0), device=x_edge.device, dtype=x_edge.dtype)
    else:
        w = edge_weight.to(device=x_edge.device, dtype=x_edge.dtype).view(-1)
    x_node = x_edge.new_zeros((num_nodes, x_edge.size(1)))
    x_node.index_add_(0, edge_nodes, x_edge[edge_ids] * w.view(-1, 1))
    return x_node


class HGNN_conv(nn.Module):
    """
    Hypergraph convolution (two-step with incidence H):
      Step A: X_edge = H^T * X_node
      Step B: X_node = H * X_edge
    Falls back to adjacency multiply when H is square.
    """

    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, H):
        """Apply linear projection and hypergraph propagation."""
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        if H is None:
            return x, None
        if H.is_sparse:
            if H.size(0) == H.size(1):
                # adjacency fallback
                return torch.sparse.mm(H, x), None
            x_edge = torch.sparse.mm(H.transpose(0, 1), x)
            x_node = torch.sparse.mm(H, x_edge)
            return x_node, x_edge
        if H.size(0) == H.size(1):
            return H.matmul(x), None
        x_edge = H.transpose(0, 1).matmul(x)
        x_node = H.matmul(x_edge)
        return x_node, x_edge


class HGNN(nn.Module):
    """Two-layer hypergraph network."""

    def __init__(self, in_ch, n_hid, n_class, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, H):
        """Run stacked hypergraph convolution layers."""
        x, _ = self.hgc1(x, H)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x, edge = self.hgc2(x, H)
        return x, edge


class HGATConv(nn.Module):
    """
    Hybrid attention:
    - Dense adjacency: full dense GAT
    - Sparse adjacency: GAT on top-k for most nodes, dense GAT on top-degree nodes
    """

    def __init__(self, in_features, out_features, heads=4, alpha=0.2, dropout=0.3,
                 top_k_sparse=16, dense_top_k=6000):
        super(HGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.head_dim = out_features // heads
        self.alpha = alpha
        self.dropout = dropout
        self.top_k_sparse = top_k_sparse
        self.dense_top_k = dense_top_k

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _dense_gat(self, h, adj_dense):
        """Dense GAT fallback path when sparse ops are unavailable."""
        h = h.view(h.size(0), self.heads, self.head_dim)
        h = h.permute(1, 0, 2)  # (heads, N, head_dim)
        attn_scores = []
        for head in range(self.heads):
            h_head = h[head]
            a_input = torch.cat([
                h_head.unsqueeze(1).expand(-1, h_head.size(0), -1),
                h_head.unsqueeze(0).expand(h_head.size(0), -1, -1)
            ], dim=2)
            e_head = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            zero_vec = -9e15 * torch.ones_like(e_head)
            attn = torch.where(adj_dense > 0, e_head, zero_vec)
            attn = F.softmax(attn, dim=1)
            attn_scores.append(attn)
        h_prime = torch.stack([torch.matmul(attn_scores[i], h[i]) for i in range(self.heads)])
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(h.size(1), -1)
        return F.elu(h_prime)

    def forward(self, x, adj):
        """Compute attention-weighted message passing on hypergraph inputs."""
        h_all = torch.mm(x, self.W)
        N = h_all.size(0)

        if not adj.is_sparse:
            return self._dense_gat(h_all, adj)

        coo = adj.coalesce()
        row, col = coo.indices()
        data = coo.values()

        deg = torch.bincount(row, minlength=N)
        dense_k = min(self.dense_top_k, N)
        dense_nodes = torch.topk(deg, dense_k, largest=True, sorted=False).indices
        dense_mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        dense_mask[dense_nodes] = True
        node_to_dense = -torch.ones(N, dtype=torch.long, device=x.device)
        node_to_dense[dense_nodes] = torch.arange(dense_nodes.numel(), device=x.device)

        dense_edge_mask = dense_mask[row] & dense_mask[col]
        if dense_edge_mask.any():
            sub_row = node_to_dense[row[dense_edge_mask]]
            sub_col = node_to_dense[col[dense_edge_mask]]
            adj_dense = torch.zeros(
                (dense_nodes.numel(), dense_nodes.numel()),
                device=x.device, dtype=x.dtype
            )
            adj_dense[sub_row, sub_col] = 1.0
            h_dense = h_all[dense_nodes]
            out_dense = self._dense_gat(h_dense, adj_dense)
        else:
            out_dense = None

        h_all_heads = h_all.view(N, self.heads, self.head_dim)
        h_src_h = h_all_heads[row]
        h_dst_h = h_all_heads[col]
        cat_h = torch.cat([h_src_h, h_dst_h], dim=2)
        e = self.leakyrelu(torch.matmul(cat_h, self.a).squeeze(2))

        counts = torch.bincount(row, minlength=N)
        offsets = torch.cumsum(torch.cat([row.new_zeros(1), counts[:-1]]), dim=0)

        out_heads = []
        for head in range(self.heads):
            out_h = torch.zeros((N, self.head_dim), device=x.device, dtype=x.dtype)
            e_head = e[:, head]
            for node in range(N):
                c = counts[node].item()
                if c == 0:
                    continue
                start = offsets[node].item()
                end = start + c
                e_seg = e_head[start:end]
                idx_seg = torch.arange(start, end, device=x.device)
                if self.top_k_sparse is not None and c > self.top_k_sparse:
                    vals, top_idx = torch.topk(e_seg, self.top_k_sparse)
                    e_seg = vals
                    idx_seg = idx_seg[top_idx]
                alpha = F.softmax(e_seg, dim=0)
                msg = torch.sum(alpha.unsqueeze(1) * h_dst_h[idx_seg, head, :], dim=0)
                out_h[node] = msg
            out_heads.append(out_h)

        out_sparse = torch.cat(out_heads, dim=1)
        if out_dense is not None:
            out_sparse[dense_nodes] = out_dense
        return F.elu(out_sparse)


class DotProductHead(nn.Module):
    """
    Cosine interaction head for DTI logits.

    - Aligns drug/protein embeddings to a shared dimension when needed.
    - Applies L2 normalization on both sides.
    - Uses learnable global logit scale (no bias).
    """

    def __init__(self, drug_dim, prot_dim, shared_dim=None, mode="dot", use_bias=True):
        super().__init__()
        drug_dim = int(drug_dim)
        prot_dim = int(prot_dim)
        if shared_dim is None:
            shared_dim = max(drug_dim, prot_dim)
        self.shared_dim = int(shared_dim)
        self.mode = str(mode or "dot").strip().lower()
        if self.mode not in ("dot", "bilinear"):
            raise ValueError(f"Unknown DotProductHead mode: {self.mode}")

        self.drug_align = (
            nn.Identity() if drug_dim == self.shared_dim else nn.Linear(drug_dim, self.shared_dim, bias=False)
        )
        self.prot_align = (
            nn.Identity() if prot_dim == self.shared_dim else nn.Linear(prot_dim, self.shared_dim, bias=False)
        )
        # Keep legacy placeholders for checkpoint compatibility.
        self.register_parameter("diag", None)
        self.register_parameter("bias", None)
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, drug_repr, prot_repr):
        if drug_repr.dim() != 2 or prot_repr.dim() != 2:
            raise ValueError("DotProductHead expects 2D tensors [batch, dim].")
        if drug_repr.size(0) != prot_repr.size(0):
            raise ValueError("Drug/protein batch sizes must match.")
        if drug_repr.numel() == 0:
            return drug_repr.new_zeros((drug_repr.size(0),))

        h_d = self.drug_align(drug_repr)
        h_p = self.prot_align(prot_repr)
        if h_d.size(-1) != h_p.size(-1):
            raise ValueError("Aligned drug/protein dimensions must match.")

        h_d = F.normalize(h_d, p=2, dim=-1, eps=1e-8)
        h_p = F.normalize(h_p, p=2, dim=-1, eps=1e-8)
        logits = torch.sum(h_d * h_p, dim=-1) * self.logit_scale
        return logits.view(-1)


class HGACN(nn.Module):
    """
    Hybrid hypergraph network:
    - HGNN + HGAT for each graph
    - Optional pooling to map atom/residue nodes to drug/protein nodes
    """

    def __init__(self, drug_feat_dim, prot_feat_dim, hidden_dim=256, out_dim=128,
                 gat_top_k_sparse=16, gat_dense_top_k=6000, interaction="dot",
                 use_hyperedge_head=True, alpha_refine=True, alpha_eps=1e-6, prior_eps=1e-4, alpha_temp=1.3,
                 use_residual=True, use_coldstart_gate=True, prior_smoothing=0.05,
                 use_bottleneck_gate=True, bottleneck_drop=0.3,
                 moe_enable=True, expert_A=True, expert_B=True, expert_C=True,
                 pool_topk=16, pool_randk=16, beta_mix=0.7,
                 randk_weight_mode="uniform", prior_floor=1e-4,
                 prior_mix_mode="mixture", prior_mix_lambda=0.3,
                 prior_mix_learnable=False, prior_mix_conditional=False,
                 prior_mix_features=None,
                 mp_gate_mode="none", mp_gate_deg_only=False, mp_gate_init_bias=0.0,
                 mp_gate_use_attn_entropy=False, mp_gate_use_prior_entropy=False,
                 mp_gate_use_esm_missing=False, mp_gate_use_prior_conf=False,
                 mp_gate_cold_scale=0.3,
                 use_knn_graph=False, cold_deg_th_drug=3, cold_deg_th_prot=3,
                 attn_kl_clip=2.0,
                 kl_stage1_epochs=0,
                 cold_zero_route_mode="hard", cold_zero_route_min_wc=1.0,
                 use_drug_missing_flag=True):
        super(HGACN, self).__init__()
        interaction = str(interaction or "dot").strip().lower()
        if interaction == "mlp":
            # Keep CLI backward compatibility while forcing metric-space head.
            interaction = "dot"
        if interaction not in ("dot", "bilinear"):
            raise ValueError(f"Unknown interaction head: {interaction}")
        self.interaction = interaction
        self.use_hyperedge_head = use_hyperedge_head
        self.alpha_refine = alpha_refine
        self.alpha_eps = float(alpha_eps)
        self.prior_eps = float(prior_eps) if prior_eps is not None else self.alpha_eps
        self.alpha_temp = float(alpha_temp)
        self.prior_smoothing = max(0.0, min(float(prior_smoothing), 1.0))
        self.use_residual = bool(use_residual)
        self.use_coldstart_gate = bool(use_coldstart_gate)
        self.use_bottleneck_gate = bool(use_bottleneck_gate)
        self.moe_enable = bool(moe_enable)
        self.expert_A = bool(expert_A)
        self.expert_B = bool(expert_B)
        self.expert_C = bool(expert_C)
        self.pool_topk = max(int(pool_topk or 0), 1)
        self.pool_randk = max(int(pool_randk or 0), 0)
        self.beta_mix = float(beta_mix)
        self.randk_weight_mode = (randk_weight_mode or "uniform").strip().lower()
        self.prior_floor = float(prior_floor)
        self.prior_mix_mode = (prior_mix_mode or "mixture").strip().lower()
        self.prior_mix_lambda = float(prior_mix_lambda)
        self.prior_mix_learnable = bool(prior_mix_learnable)
        self.prior_mix_conditional = bool(prior_mix_conditional)
        if prior_mix_features is None:
            prior_mix_list = ["deg_drug", "deg_prot", "prior_entropy", "attn_entropy", "esm_missing", "esm_unreliable"]
        elif isinstance(prior_mix_features, (list, tuple, set)):
            prior_mix_list = [str(s).strip() for s in prior_mix_features if str(s).strip()]
        else:
            prior_mix_list = [s.strip() for s in str(prior_mix_features).split(",") if s.strip()]
        self.prior_mix_features = set(prior_mix_list)
        self.mp_gate_mode = (mp_gate_mode or "none").strip().lower()
        self.mp_gate_deg_only = bool(mp_gate_deg_only)
        self.mp_gate_init_bias = float(mp_gate_init_bias)
        self.mp_gate_use_attn_entropy = bool(mp_gate_use_attn_entropy)
        self.mp_gate_use_prior_entropy = bool(mp_gate_use_prior_entropy)
        self.mp_gate_use_esm_missing = bool(mp_gate_use_esm_missing)
        self.mp_gate_use_prior_conf = bool(mp_gate_use_prior_conf)
        self.mp_gate_cold_scale = float(mp_gate_cold_scale)
        self.kl_stage1_epochs = max(int(kl_stage1_epochs or 0), 0)
        self._current_epoch = None
        self._freeze_delta = False
        self.use_knn_graph = bool(use_knn_graph)
        self.cold_deg_th_drug = int(cold_deg_th_drug or 0)
        self.cold_deg_th_prot = int(cold_deg_th_prot or 0)
        self.cold_zero_route_mode = str(cold_zero_route_mode or "hard").strip().lower()
        if self.cold_zero_route_mode not in ("hard", "soft", "off"):
            raise ValueError(
                f"Unknown cold_zero_route_mode={cold_zero_route_mode}; expected one of: hard, soft, off."
            )
        self.cold_zero_route_min_wc = float(cold_zero_route_min_wc)
        if not np.isfinite(self.cold_zero_route_min_wc):
            raise ValueError(
                f"cold_zero_route_min_wc must be finite, got {cold_zero_route_min_wc}."
            )
        self.cold_zero_route_min_wc = min(max(self.cold_zero_route_min_wc, 0.0), 1.0)
        self.attn_kl_clip = float(attn_kl_clip) if attn_kl_clip is not None else 2.0
        self._kl_warn_count = 0
        self._sanity_kl_print_count = 0
        self._sanity_kl_print_limit = 1
        self.use_drug_missing_flag = bool(use_drug_missing_flag)
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.drug_gcn = HGNN(hidden_dim, hidden_dim, hidden_dim)
        self.prot_gcn = HGNN(hidden_dim, hidden_dim, hidden_dim)

        self.drug_gat = HGATConv(
            hidden_dim, hidden_dim,
            top_k_sparse=gat_top_k_sparse,
            dense_top_k=gat_dense_top_k
        )
        self.prot_gat = HGATConv(
            hidden_dim, hidden_dim,
            top_k_sparse=gat_top_k_sparse,
            dense_top_k=gat_dense_top_k
        )

        self.drug_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.prot_fusion = nn.Linear(2 * hidden_dim, hidden_dim)

        # Hyperedge fusion for broadcast uses hidden_dim; head uses out_dim projection.
        self.edge_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.edge_fusion_out = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2),
        )

        self.drug_projection = nn.Linear(hidden_dim, out_dim)
        self.prot_projection = nn.Linear(hidden_dim, out_dim)
        attn_hidden = max(hidden_dim // 2, 1)
        self.atom_attn_refiner = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1),
        )
        self.res_attn_refiner = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.ReLU(),
            nn.Linear(attn_hidden, 1),
        )
        self.attn_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.drug_bn = nn.BatchNorm1d(hidden_dim)
        self.prot_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=float(bottleneck_drop))
        self.bottleneck_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.bottleneck_norm = nn.LayerNorm(hidden_dim)
        self.bottleneck_gate = nn.Linear(hidden_dim, hidden_dim)
        self.drug_feat_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        self.prot_feat_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        # Inductive cold-start: encode sequence-only protein features to match graph feature space.
        self.prot_seq_align = nn.Sequential(
            nn.Linear(prot_feat_dim, out_dim),
            nn.ReLU(),
        )
        self.inductive_knn_k = 3
        self.inductive_knn_mix = 0.5
        self.gate_drug = nn.Linear(1, 1)
        self.gate_prot = nn.Linear(1, 1)
        gate_in_dim = 2 * out_dim + 1
        self.atom_gate = nn.Sequential(
            nn.Linear(gate_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.res_gate = nn.Sequential(
            nn.Linear(gate_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        refine_in_dim = 2 * out_dim
        self.atom_refine = nn.Sequential(
            nn.Linear(refine_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.res_refine = nn.Sequential(
            nn.Linear(refine_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.edge_head = DotProductHead(
            drug_dim=out_dim,
            prot_dim=out_dim,
            shared_dim=out_dim,
            mode=("bilinear" if self.interaction == "bilinear" else "dot"),
            use_bias=True,
        )
        self.moe_gate = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        if self.prior_mix_learnable:
            init_lam = max(min(self.prior_mix_lambda, 1.0 - 1e-6), 1e-6)
            init_logit = math.log(init_lam / (1.0 - init_lam))
            self.prior_mix_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.prior_mix_logit = None
        if self.prior_mix_conditional:
            self.prior_mix_mlp_drug = nn.Linear(4, 1)
            self.prior_mix_mlp_prot = nn.Linear(4, 1)
        else:
            self.prior_mix_mlp_drug = None
            self.prior_mix_mlp_prot = None
        self.mp_gate_atom = None
        self.mp_gate_res = None
        if self.mp_gate_mode != "none":
            mp_in_dim = 1
            if not self.mp_gate_deg_only:
                if self.mp_gate_use_prior_entropy:
                    mp_in_dim += 1
                if self.mp_gate_use_prior_conf:
                    mp_in_dim += 1
                if self.mp_gate_use_attn_entropy:
                    mp_in_dim += 1
                if self.mp_gate_use_esm_missing:
                    mp_in_dim += 1
            self.mp_gate_atom = nn.Linear(mp_in_dim, 1)
            self.mp_gate_res = nn.Linear(mp_in_dim, 1)
            nn.init.zeros_(self.mp_gate_atom.weight)
            nn.init.zeros_(self.mp_gate_res.weight)
            if self.mp_gate_atom.bias is not None:
                self.mp_gate_atom.bias.data.fill_(self.mp_gate_init_bias)
            if self.mp_gate_res.bias is not None:
                self.mp_gate_res.bias.data.fill_(self.mp_gate_init_bias)
        if self.use_drug_missing_flag:
            self.drug_missing_bias = nn.Parameter(torch.zeros(drug_feat_dim))
        else:
            self.register_parameter("drug_missing_bias", None)

    def set_epoch(self, epoch):
        """Store current epoch for schedule-dependent behavior."""
        self._current_epoch = int(epoch) if epoch is not None else None

    def _pool_by_unique_index(self, x, index, weight=None, gate_net=None, partner=None):
        """Pool node features by unique index with optional gates."""
        if index is None:
            return x, None
        if index.numel() == 0:
            return x.new_zeros((0, x.size(1))), index.new_zeros((0,))
        uniq_ids, inv = torch.unique(index, sorted=True, return_inverse=True)
        out = x.new_zeros((uniq_ids.numel(), x.size(1)))
        if weight is None:
            out.index_add_(0, inv, x)
            counts = x.new_zeros((uniq_ids.numel(),))
            counts.index_add_(0, inv, torch.ones_like(inv, dtype=x.dtype))
            out = out / counts.clamp_min(1.0).unsqueeze(1)
            return out, uniq_ids
        w = weight.view(-1)
        if gate_net is not None:
            if partner is None:
                partner = x.new_zeros((x.size(0), x.size(1)))
            gate_in = torch.cat([x, partner, w.view(-1, 1)], dim=1)
            gate = torch.sigmoid(gate_net(gate_in)).view(-1)
            w = w * gate
        out.index_add_(0, inv, x * w.view(-1, 1))
        denom = x.new_zeros((uniq_ids.numel(),))
        denom.index_add_(0, inv, w)
        out = out / denom.clamp_min(1e-6).unsqueeze(1)
        return out, uniq_ids

    def _pool_by_index_with_alpha(self, x, index, alpha):
        """Pool node features using externally provided alpha weights."""
        if index is None:
            return x, None
        if index.numel() == 0:
            return x.new_zeros((0, x.size(1))), index.new_zeros((0,))
        uniq_ids, inv = torch.unique(index, sorted=True, return_inverse=True)
        out = x.new_zeros((uniq_ids.numel(), x.size(1)))
        out.index_add_(0, inv, x * alpha.view(-1, 1))
        return out, uniq_ids, inv

    def _pool_by_index_max(self, x, index):
        """Max-pool node features by grouped index (keep feature_dim unchanged)."""
        if index is None:
            return x, None, None
        if index.numel() == 0:
            empty = index.new_zeros((0,))
            return x.new_zeros((0, x.size(1))), empty, empty
        uniq_ids, inv = torch.unique(index, sorted=True, return_inverse=True)
        out = x.new_full((uniq_ids.numel(), x.size(1)), -float("inf"))
        if hasattr(out, "scatter_reduce_"):
            scatter_idx = inv.view(-1, 1).expand(-1, x.size(1))
            out.scatter_reduce_(0, scatter_idx, x, reduce="amax", include_self=True)
        else:
            for gid in range(int(uniq_ids.numel())):
                mask = inv == gid
                if mask.any():
                    out[gid] = x[mask].max(dim=0).values
        return out, uniq_ids, inv

    def _pool_by_unique_index_max(self, x, index):
        """Max-pool wrapper that matches _pool_by_unique_index return signature."""
        out, uniq_ids, _ = self._pool_by_index_max(x, index)
        return out, uniq_ids

    def _pool_topk_randk(self, x, index, scores, topk, randk, beta_mix=0.7,
                         rand_weight_mode="uniform", prior_floor=1e-4, seed=0):
        """Hybrid top-k and random-k pooling for robust selection."""
        if index is None:
            return x, None, None
        if index.numel() == 0:
            return x.new_zeros((0, x.size(1))), index.new_zeros((0,)), None
        uniq_ids, inv = torch.unique(index, sorted=True, return_inverse=True)
        num_groups = int(uniq_ids.numel())
        order = torch.argsort(inv)
        inv_sorted = inv[order]
        counts = torch.bincount(inv_sorted, minlength=num_groups)
        offsets = torch.cumsum(torch.cat([inv_sorted.new_zeros(1), counts[:-1]]), dim=0)
        out = x.new_zeros((num_groups, x.size(1)))
        for g in range(num_groups):
            c = int(counts[g].item())
            if c <= 0:
                continue
            start = int(offsets[g].item())
            end = start + c
            idx = order[start:end]
            sc = scores[idx]
            k = min(int(topk), c)
            if k > 0:
                top_loc = torch.topk(sc, k=k, largest=True, sorted=False).indices
                top_idx = idx[top_loc]
                w_top = torch.softmax(sc[top_loc], dim=0)
                agg_top = (x[top_idx] * w_top.view(-1, 1)).sum(dim=0)
            else:
                agg_top = x.new_zeros((x.size(1),))
                top_loc = None
            rand_idx = None
            if randk and c > k:
                mask = torch.ones((c,), device=idx.device, dtype=torch.bool)
                if top_loc is not None:
                    mask[top_loc] = False
                remain = idx[mask]
                rk = min(int(randk), int(remain.numel()))
                if rk > 0:
                    gen = torch.Generator(device=idx.device)
                    gen.manual_seed(int(seed) + int(uniq_ids[g].item()) * 1315423911)
                    perm = torch.randperm(remain.numel(), generator=gen, device=idx.device)[:rk]
                    rand_idx = remain[perm]
            if rand_idx is not None and rand_idx.numel() > 0:
                if rand_weight_mode == "uniform":
                    w_rand = torch.ones((rand_idx.numel(),), device=idx.device, dtype=x.dtype)
                else:
                    sc_rand = scores[rand_idx]
                    w_rand = torch.clamp(sc_rand, min=float(prior_floor))
                w_rand = w_rand / w_rand.sum().clamp_min(1e-12)
                agg_rand = (x[rand_idx] * w_rand.view(-1, 1)).sum(dim=0)
                out[g] = beta_mix * agg_top + (1.0 - beta_mix) * agg_rand
            else:
                out[g] = agg_top
        return out, uniq_ids, inv

    def _knn_aggregate(self, x, edge_index, edge_weight, num_nodes):
        """Aggregate KNN features for query nodes from neighbor nodes.

        Expected edge direction: edge_index[0]=query_id, edge_index[1]=neighbor_id.
        """
        if edge_index is None or edge_weight is None:
            return x.new_zeros((num_nodes, x.size(1)))
        if edge_index.numel() == 0 or edge_weight.numel() == 0:
            return x.new_zeros((num_nodes, x.size(1)))
        query = edge_index[0]
        neigh = edge_index[1]
        out = x.new_zeros((num_nodes, x.size(1)))
        w = edge_weight.view(-1, 1).to(dtype=x.dtype, device=x.device)
        out.index_add_(0, query, x[neigh] * w)
        return out

    def _inductive_knn_feature_borrow(
        self,
        gnn_repr,
        seq_repr,
        group_ids,
        degree,
        warm_allow_mask=None,
        topk=3,
        mix=0.5,
    ):
        """
        Strict inductive protein feature borrowing for degree==0 proteins.

        - Neighbors are searched ONLY among warm proteins (degree > 0) in current batch.
        - Optional warm_allow_mask can further restrict warm candidates (e.g., train-only).
        - Similarity uses cosine over raw sequence features via torch.cdist.
        - Cold features are fused by residual mix:
            final_cold = mix * seq_encoded + (1 - mix) * borrowed_graph
        """
        stats = {
            "n_total": 0,
            "n_warm": 0,
            "n_warm_degree": 0,
            "n_cold": 0,
            "k_eff": 0,
            "fallback_seq_only": 0,
            "warm_mask_invalid": 0,
        }
        if (
            gnn_repr is None
            or seq_repr is None
            or group_ids is None
            or degree is None
        ):
            return gnn_repr, stats
        if gnn_repr.numel() == 0 or seq_repr.numel() == 0 or group_ids.numel() == 0:
            return gnn_repr, stats
        if gnn_repr.size(0) != seq_repr.size(0) or gnn_repr.size(0) != group_ids.numel():
            return gnn_repr, stats

        if not torch.is_tensor(degree):
            degree = torch.as_tensor(degree, device=gnn_repr.device, dtype=gnn_repr.dtype)
        else:
            degree = degree.to(device=gnn_repr.device)

        deg_group = degree[group_ids]
        cold_mask = deg_group <= 0
        warm_degree_mask = ~cold_mask
        if warm_allow_mask is not None:
            if not torch.is_tensor(warm_allow_mask):
                warm_allow_mask = torch.as_tensor(warm_allow_mask, device=gnn_repr.device)
            else:
                warm_allow_mask = warm_allow_mask.to(device=gnn_repr.device)
            if warm_allow_mask.dtype != torch.bool:
                warm_allow_mask = warm_allow_mask > 0
            if warm_allow_mask.numel() > int(group_ids.max().item()):
                warm_mask = warm_degree_mask & warm_allow_mask[group_ids]
            else:
                warm_mask = warm_degree_mask.new_zeros(warm_degree_mask.size(0), dtype=torch.bool)
                stats["warm_mask_invalid"] = 1
        else:
            warm_mask = warm_degree_mask
        stats["n_total"] = int(group_ids.numel())
        stats["n_warm"] = int(warm_mask.sum().item())
        stats["n_warm_degree"] = int(warm_degree_mask.sum().item())
        stats["n_cold"] = int(cold_mask.sum().item())
        if not cold_mask.any():
            return gnn_repr, stats

        seq_clean = torch.nan_to_num(seq_repr, nan=0.0, posinf=0.0, neginf=0.0)
        seq_encoded = self.prot_seq_align(seq_clean)
        out = gnn_repr.clone()
        if warm_mask.any():
            k_eff = min(int(max(topk, 1)), int(warm_mask.sum().item()))
            stats["k_eff"] = int(k_eff)
            # Cosine similarity via cdist on l2-normalized vectors:
            # cosine(a,b) = 1 - 0.5 * ||a-b||^2
            seq_knn = seq_clean.to(dtype=torch.float32)
            cold_seq = F.normalize(seq_knn[cold_mask], p=2, dim=1, eps=1e-8)
            warm_seq = F.normalize(seq_knn[warm_mask], p=2, dim=1, eps=1e-8)
            dist = torch.cdist(cold_seq, warm_seq, p=2)
            sim = 1.0 - 0.5 * dist.pow(2)
            knn_idx = torch.topk(sim, k=k_eff, dim=1, largest=True).indices
            warm_gnn_feats = gnn_repr[warm_mask]
            borrowed_graph_feat = warm_gnn_feats[knn_idx].mean(dim=1)
            mix = float(mix)
            mix = min(max(mix, 0.0), 1.0)
            out[cold_mask] = mix * seq_encoded[cold_mask] + (1.0 - mix) * borrowed_graph_feat
        else:
            # No warm proteins in this batch: fallback to sequence-only feature.
            out[cold_mask] = seq_encoded[cold_mask]
            stats["fallback_seq_only"] = 1
        return out, stats

    def _edge_logits(self, edge_drug, edge_prot):
        """Compute edge logits from drug/protein edge representations."""
        return self.edge_head(edge_drug, edge_prot)

    def _pair_pool(self, node_repr, edge_ptr, edge_nodes, edge_psichic, partner_summary, gate_net):
        """Pool per-edge node sets using learned gates and context."""
        if edge_ptr is None or edge_nodes is None or edge_psichic is None:
            return None, node_repr.new_tensor(0.0)
        eps = 1e-6
        num_edges = int(edge_ptr.numel() - 1)
        out = node_repr.new_zeros((num_edges, node_repr.size(1)))
        kl_total = node_repr.new_tensor(0.0)
        for i in range(num_edges):
            start = int(edge_ptr[i].item())
            end = int(edge_ptr[i + 1].item())
            if start >= end:
                continue
            idx = edge_nodes[start:end]
            ps = edge_psichic[start:end]
            nr = node_repr[idx]
            partner = partner_summary[i].unsqueeze(0).expand(nr.size(0), -1)
            gate_in = torch.cat([nr, partner, ps.view(-1, 1)], dim=1)
            gate = torch.sigmoid(gate_net(gate_in)).view(-1)
            logit = torch.log(ps.clamp_min(eps)) + torch.log(gate.clamp_min(eps))
            p = torch.softmax(logit, dim=0)
            q = torch.softmax(ps, dim=0)
            kl_total = kl_total + torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)))
            out[i] = torch.sum(nr * p.unsqueeze(1), dim=0)
        kl_mean = kl_total / max(num_edges, 1)
        return out, kl_mean

    def _pair_pool_prior(self, node_repr, edge_ptr, edge_nodes, edge_prior):
        """Pool per-edge node sets using prior weights only."""
        if edge_ptr is None or edge_nodes is None or edge_prior is None:
            return None
        # Use a tiny eps to avoid flattening.
        eps = 1e-12
        num_edges = int(edge_ptr.numel() - 1)
        out = node_repr.new_zeros((num_edges, node_repr.size(1)))
        for i in range(num_edges):
            start = int(edge_ptr[i].item())
            end = int(edge_ptr[i + 1].item())
            if start >= end:
                continue
            idx = edge_nodes[start:end]
            prior = edge_prior[start:end]
            # Power-boost prior to sharpen separation.
            prior = torch.pow(prior, 4.0)
            denom = prior.sum().clamp_min(eps)
            w = prior / denom
            out[i] = torch.sum(node_repr[idx] * w.unsqueeze(1), dim=0)
        return out

    def _pair_pool_refine(self, node_repr, edge_ptr, edge_nodes, edge_prior, edge_ctx, refine_net):
        """Refine edge priors with context before weighted pooling."""
        if edge_ptr is None or edge_nodes is None or edge_prior is None:
            zero = node_repr.new_tensor(0.0)
            return None, zero, zero, zero
        eps = 1e-12
        temp = self.alpha_temp
        num_edges = int(edge_ptr.numel() - 1)
        out = node_repr.new_zeros((num_edges, node_repr.size(1)))
        kl_total = node_repr.new_tensor(0.0)
        kl_norm_total = node_repr.new_tensor(0.0)
        ent_total = node_repr.new_tensor(0.0)
        for i in range(num_edges):
            start = int(edge_ptr[i].item())
            end = int(edge_ptr[i + 1].item())
            if start >= end:
                continue
            idx = edge_nodes[start:end]
            prior = edge_prior[start:end]
            # Power-boost prior to sharpen separation.
            prior = torch.pow(prior, 4.0)
            denom = prior.sum().clamp_min(eps)
            prior = prior / denom
            ctx = edge_ctx[i].unsqueeze(0).expand(idx.numel(), -1)
            z = node_repr[idx]
            delta = refine_net(torch.cat([z, ctx], dim=1)).view(-1)
            if getattr(self, "_freeze_delta", False):
                delta = delta.detach() * 0.0
            logit = torch.log(prior.clamp_min(eps)) / temp + delta
            alpha = torch.softmax(logit, dim=0)
            out[i] = torch.sum(z * alpha.unsqueeze(1), dim=0)
            kl_edge = torch.sum(alpha * (torch.log(alpha + eps) - torch.log(prior + eps)))
            k_eff = max(int(idx.numel()), 2)
            logk = alpha.new_tensor(math.log(k_eff))
            kl_total = kl_total + kl_edge
            kl_norm_total = kl_norm_total + kl_edge / logk
            ent_total = ent_total + torch.sum(-alpha * torch.log(alpha + eps))
        denom = max(num_edges, 1)
        kl_mean = kl_total / denom
        kl_norm = kl_norm_total / denom
        ent_mean = ent_total / max(num_edges, 1)
        return out, kl_mean, kl_norm, ent_mean

    def _degree_gate(self, deg, gate_layer, device):
        """Map node degree to gate bias signal."""
        if deg is None:
            return None
        deg_t = torch.as_tensor(deg, device=device, dtype=torch.float32)
        if deg_t.numel() == 0:
            return None
        max_deg = deg_t.max()
        denom = torch.log1p(max_deg + 1.0)
        if denom.item() <= 0:
            deg_norm = torch.zeros_like(deg_t)
        else:
            deg_norm = torch.log1p(deg_t) / denom
        gate = torch.sigmoid(gate_layer(deg_norm.view(-1, 1))).view(-1)
        return gate

    def _apply_bottleneck(self, x):
        """Apply optional variational bottleneck transform."""
        if not self.use_bottleneck_gate:
            return x
        z = self.bottleneck_norm(self.bottleneck_mlp(x))
        g = torch.sigmoid(self.bottleneck_gate(z))
        return x * g

    def forward(self, drug_feat, prot_feat, G_drug, G_protein,
                drug_node_to_entity=None, protein_node_to_entity=None,
                drug_node_weight=None, protein_node_weight=None,
                atom_prior=None, res_prior=None,
                edge_index=None,
                drug_edge_index=None, drug_edge_weight=None, drug_num_nodes=None,
                prot_edge_index=None, prot_edge_weight=None, prot_num_nodes=None,
                drug_edge_ptr=None, drug_edge_nodes=None, drug_edge_psichic=None,
                prot_edge_ptr=None, prot_edge_nodes=None, prot_edge_psichic=None,
                drug_degree=None, prot_degree=None, prot_warm_mask=None, use_coldstart_gate=None,
                prot_esm_missing=None, prot_esm_unreliable=None,
                drug_knn_edge_index=None, drug_knn_edge_weight=None,
                prot_knn_edge_index=None, prot_knn_edge_weight=None,
                cold_deg_th_drug=None, cold_deg_th_prot=None,
                expertC_scale=None, wC_cap=None,
                return_pair_repr=False, return_aux=False, explain_cfg=None):
        """Execute full HGACN forward path and return prediction package."""
        if G_drug is None:
            if drug_edge_index is None or drug_num_nodes is None:
                raise ValueError("drug_edge_index/drug_num_nodes required when G_drug is None")
            if drug_edge_weight is None:
                drug_edge_weight = torch.ones(drug_edge_index.size(1), device=drug_feat.device, dtype=drug_feat.dtype)
            G_drug = torch.sparse_coo_tensor(
                drug_edge_index, drug_edge_weight,
                size=(drug_num_nodes, drug_num_nodes),
                device=drug_feat.device, dtype=drug_feat.dtype
            ).coalesce()
        if G_protein is None:
            if prot_edge_index is None or prot_num_nodes is None:
                raise ValueError("prot_edge_index/prot_num_nodes required when G_protein is None")
            if prot_edge_weight is None:
                prot_edge_weight = torch.ones(prot_edge_index.size(1), device=prot_feat.device, dtype=prot_feat.dtype)
            G_protein = torch.sparse_coo_tensor(
                prot_edge_index, prot_edge_weight,
                size=(prot_num_nodes, prot_num_nodes),
                device=prot_feat.device, dtype=prot_feat.dtype
            ).coalesce()
        use_coldstart_gate = self.use_coldstart_gate if use_coldstart_gate is None else bool(use_coldstart_gate)
        cold_drug_th = self.cold_deg_th_drug if cold_deg_th_drug is None else int(cold_deg_th_drug)
        cold_prot_th = self.cold_deg_th_prot if cold_deg_th_prot is None else int(cold_deg_th_prot)
        freeze_delta = (
            self._current_epoch is not None
            and self.kl_stage1_epochs > 0
            and int(self._current_epoch) < int(self.kl_stage1_epochs)
        )
        self._freeze_delta = bool(freeze_delta)
        if expertC_scale is None:
            expertC_scale = 1.0
        else:
            expertC_scale = float(expertC_scale)
        expertC_scale = max(0.0, min(1.0, expertC_scale))
        if wC_cap is not None:
            wC_cap = float(wC_cap)
            if wC_cap < 0:
                wC_cap = 0.0
        self._last_expertC_scale = expertC_scale
        if prot_esm_missing is not None and not torch.is_tensor(prot_esm_missing):
            prot_esm_missing = torch.as_tensor(prot_esm_missing, device=prot_feat.device, dtype=torch.float32)
        if prot_esm_unreliable is not None and not torch.is_tensor(prot_esm_unreliable):
            prot_esm_unreliable = torch.as_tensor(prot_esm_unreliable, device=prot_feat.device, dtype=torch.float32)
        if prot_warm_mask is not None and not torch.is_tensor(prot_warm_mask):
            prot_warm_mask = torch.as_tensor(prot_warm_mask, device=prot_feat.device)
        if prot_warm_mask is not None:
            prot_warm_mask = prot_warm_mask.to(device=prot_feat.device)
            if prot_warm_mask.dtype != torch.bool:
                prot_warm_mask = prot_warm_mask > 0
        esm_flag = prot_esm_unreliable if prot_esm_unreliable is not None else prot_esm_missing

        t_start = time.time()
        t_head_start = None
        t_gate = 0.0
        t_mpgate = 0.0
        t_knn = 0.0
        # Reset per-forward aux to avoid stale values when MoE is off.
        self._last_gate_weights = None
        self._last_teacher_logits = None
        self._last_cold_edge_mask = None
        self._last_aux = {}
        self._last_prior_conf = None
        self._last_mp_gate_atom = None
        self._last_mp_gate_res = None
        self._last_mp_gate_atom_cold = None
        self._last_mp_gate_res_cold = None
        self._last_delta_reg = None
        self._last_delta_rms = None
        self._last_delta_rms_weighted = None

        explain_cfg = explain_cfg or {}
        aux_to_cpu = bool(explain_cfg.get("aux_to_cpu", True))
        keep_grad = bool(explain_cfg.get("keep_grad", False))
        need_node_repr = bool(explain_cfg.get("need_node_repr", False))
        retain_node_grad = bool(explain_cfg.get("retain_node_grad", False))

        def _record_profile():
            if t_head_start is None:
                return
            self._last_profile = {
                "encoder": t_head_start - t_start,
                "heads": time.time() - t_head_start,
                "gate": float(t_gate),
                "mp_gate": float(t_mpgate),
                "knn": float(t_knn),
            }

        def _maybe_detach(val):
            if val is None:
                return None
            if torch.is_tensor(val):
                if keep_grad and not aux_to_cpu:
                    return val
                out = val.detach()
                if aux_to_cpu:
                    out = out.cpu()
                return out
            if isinstance(val, dict):
                return {k: _maybe_detach(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [ _maybe_detach(v) for v in val ]
            return val

        def _pack(edge_logits, vae_params, pair_repr=None, aux=None):
            _record_profile()
            if return_aux:
                if aux is None:
                    aux = {}
                return edge_logits, _maybe_detach(aux)
            if return_pair_repr:
                return edge_logits, vae_params, pair_repr
            return edge_logits, vae_params
        if self.use_drug_missing_flag and self.drug_missing_bias is not None and drug_feat is not None and drug_feat.numel():
            miss_flag = drug_feat[:, -1].to(dtype=drug_feat.dtype).view(-1, 1)
            drug_feat = drug_feat + miss_flag * self.drug_missing_bias.view(1, -1)
        res_drug = self.drug_proj(drug_feat)
        res_prot = self.prot_proj(prot_feat)

        drug_gcn, _ = self.drug_gcn(res_drug, G_drug)
        prot_gcn, _ = self.prot_gcn(res_prot, G_protein)

        drug_gat = self.drug_gat(res_drug, G_drug)
        prot_gat = self.prot_gat(res_prot, G_protein)
        if self.use_residual:
            drug_gcn = drug_gcn + res_drug
            drug_gat = drug_gat + res_drug
            prot_gcn = prot_gcn + res_prot
            prot_gat = prot_gat + res_prot

        if self.mp_gate_mode != "none" and drug_node_to_entity is not None and protein_node_to_entity is not None:
            t0_mpgate = time.time()
            temp = max(self.alpha_temp, 1e-3)
            drug_inv = None
            prot_inv = None
            if drug_node_to_entity.numel() > 0:
                _, drug_inv = torch.unique(drug_node_to_entity, sorted=True, return_inverse=True)
            if protein_node_to_entity.numel() > 0:
                _, prot_inv = torch.unique(protein_node_to_entity, sorted=True, return_inverse=True)
            # Degree features
            if drug_degree is not None:
                deg_drug_node = torch.log1p(drug_degree[drug_node_to_entity].to(device=drug_feat.device, dtype=drug_feat.dtype))
            else:
                deg_drug_node = drug_gcn.new_zeros((drug_gcn.size(0),))
            if prot_degree is not None:
                deg_prot_node = torch.log1p(prot_degree[protein_node_to_entity].to(device=prot_feat.device, dtype=prot_feat.dtype))
            else:
                deg_prot_node = prot_gcn.new_zeros((prot_gcn.size(0),))
            cold_drug_node = None
            cold_prot_node = None
            if drug_degree is not None:
                cold_drug_node = (drug_degree[drug_node_to_entity] <= float(cold_drug_th))
            if prot_degree is not None:
                cold_prot_node = (prot_degree[protein_node_to_entity] <= float(cold_prot_th))
            # Prior entropy features
            prior_ent_atom = None
            prior_ent_res = None
            prior_conf_atom = None
            prior_conf_res = None
            if self.mp_gate_use_prior_entropy and atom_prior is not None and drug_inv is not None:
                p_atom_gate = normalize_by_group(atom_prior + self.alpha_eps, drug_inv, int(drug_inv.max().item()) + 1, eps=self.alpha_eps)
                prior_ent_atom = group_entropy(p_atom_gate, drug_inv, int(drug_inv.max().item()) + 1)
            if self.mp_gate_use_prior_entropy and res_prior is not None and prot_inv is not None:
                p_res_gate = normalize_by_group(res_prior + self.alpha_eps, prot_inv, int(prot_inv.max().item()) + 1, eps=self.alpha_eps)
                prior_ent_res = group_entropy(p_res_gate, prot_inv, int(prot_inv.max().item()) + 1)
            if self.mp_gate_use_prior_conf:
                if prior_ent_atom is not None:
                    prior_conf_atom = (1.0 - prior_ent_atom).clamp(0.0, 1.0)
                if prior_ent_res is not None:
                    prior_conf_res = (1.0 - prior_ent_res).clamp(0.0, 1.0)
            # Attn entropy features (cheap pass from raw projections)
            attn_ent_atom = None
            attn_ent_res = None
            if self.mp_gate_use_attn_entropy and drug_inv is not None:
                s_atom_gate = self.atom_attn_refiner(res_drug).squeeze(-1)
                alpha_atom_gate = group_softmax(s_atom_gate / temp, drug_inv, int(drug_inv.max().item()) + 1)
                attn_ent_atom = group_entropy(alpha_atom_gate, drug_inv, int(drug_inv.max().item()) + 1)
            if self.mp_gate_use_attn_entropy and prot_inv is not None:
                s_res_gate = self.res_attn_refiner(res_prot).squeeze(-1)
                alpha_res_gate = group_softmax(s_res_gate / temp, prot_inv, int(prot_inv.max().item()) + 1)
                attn_ent_res = group_entropy(alpha_res_gate, prot_inv, int(prot_inv.max().item()) + 1)
            # ESM unreliable (residue side)
            if esm_flag is not None:
                esm_missing_node = esm_flag[protein_node_to_entity].to(device=prot_feat.device, dtype=prot_feat.dtype)
            else:
                esm_missing_node = prot_gcn.new_zeros((prot_gcn.size(0),))

            def _build_gate_feat(deg_node, inv, prior_ent, prior_conf, attn_ent, esm_missing, is_prot=False):
                feats = [deg_node.view(-1, 1)]
                if not self.mp_gate_deg_only:
                    if self.mp_gate_use_prior_entropy:
                        if prior_ent is not None and inv is not None:
                            feats.append(prior_ent[inv].view(-1, 1))
                        else:
                            feats.append(deg_node.new_zeros((deg_node.size(0), 1)))
                    if self.mp_gate_use_prior_conf:
                        if prior_conf is not None and inv is not None:
                            feats.append(prior_conf[inv].view(-1, 1))
                        else:
                            feats.append(deg_node.new_zeros((deg_node.size(0), 1)))
                    if self.mp_gate_use_attn_entropy:
                        if attn_ent is not None and inv is not None:
                            feats.append(attn_ent[inv].view(-1, 1))
                        else:
                            feats.append(deg_node.new_zeros((deg_node.size(0), 1)))
                    if self.mp_gate_use_esm_missing:
                        if is_prot and esm_missing is not None:
                            feats.append(esm_missing.view(-1, 1))
                        else:
                            feats.append(deg_node.new_zeros((deg_node.size(0), 1)))
                return torch.cat(feats, dim=1)

            atom_feat = _build_gate_feat(
                deg_drug_node, drug_inv, prior_ent_atom, prior_conf_atom, attn_ent_atom, None, is_prot=False
            )
            res_feat = _build_gate_feat(
                deg_prot_node, prot_inv, prior_ent_res, prior_conf_res, attn_ent_res, esm_missing_node, is_prot=True
            )
            alpha_atom = torch.sigmoid(self.mp_gate_atom(atom_feat)).view(-1)
            alpha_res = torch.sigmoid(self.mp_gate_res(res_feat)).view(-1)
            if cold_drug_node is not None and self.mp_gate_cold_scale < 1.0:
                scale = torch.where(
                    cold_drug_node,
                    alpha_atom.new_tensor(self.mp_gate_cold_scale),
                    alpha_atom.new_tensor(1.0),
                )
                alpha_atom = alpha_atom * scale
            if cold_prot_node is not None and self.mp_gate_cold_scale < 1.0:
                scale = torch.where(
                    cold_prot_node,
                    alpha_res.new_tensor(self.mp_gate_cold_scale),
                    alpha_res.new_tensor(1.0),
                )
                alpha_res = alpha_res * scale
            self._last_mp_gate_atom = alpha_atom.detach()
            self._last_mp_gate_res = alpha_res.detach()
            self._last_mp_gate_atom_cold = cold_drug_node.detach() if cold_drug_node is not None else None
            self._last_mp_gate_res_cold = cold_prot_node.detach() if cold_prot_node is not None else None
            drug_gcn = res_drug + alpha_atom.view(-1, 1) * (drug_gcn - res_drug)
            drug_gat = res_drug + alpha_atom.view(-1, 1) * (drug_gat - res_drug)
            prot_gcn = res_prot + alpha_res.view(-1, 1) * (prot_gcn - res_prot)
            prot_gat = res_prot + alpha_res.view(-1, 1) * (prot_gat - res_prot)
            t_mpgate += time.time() - t0_mpgate

        edge_fused = None
        edge_fused_hidden = None
        if (
            drug_edge_ptr is not None
            and drug_edge_nodes is not None
            and prot_edge_ptr is not None
            and prot_edge_nodes is not None
        ):
            edge_atom, edge_ids, num_edges = _ragged_edge_aggregate(
                drug_gcn, drug_edge_ptr, drug_edge_nodes, drug_edge_psichic
            )
            edge_res, _, num_edges_res = _ragged_edge_aggregate(
                prot_gcn, prot_edge_ptr, prot_edge_nodes, prot_edge_psichic
            )
            if num_edges > 0 and num_edges_res == num_edges:
                edge_fused_hidden = self.edge_fusion(torch.cat([edge_atom, edge_res], dim=1))
                edge_fused = self.edge_fusion_out(edge_fused_hidden)
                drug_hyper = _ragged_edge_broadcast(
                    edge_fused_hidden, edge_ids, drug_edge_nodes, drug_edge_psichic, drug_gcn.size(0)
                )
                prot_hyper = _ragged_edge_broadcast(
                    edge_fused_hidden, edge_ids, prot_edge_nodes, prot_edge_psichic, prot_gcn.size(0)
                )
                drug_gcn = drug_gcn + drug_hyper
                prot_gcn = prot_gcn + prot_hyper

        drug_fused = torch.cat([drug_gcn, drug_gat], dim=1)
        drug_fused = F.relu(self.drug_fusion(drug_fused))
        drug_fused = self.drug_bn(drug_fused)
        drug_fused = self.dropout(drug_fused)
        drug_fused = self._apply_bottleneck(drug_fused)

        prot_fused = torch.cat([prot_gcn, prot_gat], dim=1)
        prot_fused = F.relu(self.prot_fusion(prot_fused))
        prot_fused = self.prot_bn(prot_fused)
        prot_fused = self.dropout(prot_fused)
        prot_fused = self._apply_bottleneck(prot_fused)

        drug_node_repr = self.drug_projection(drug_fused)
        prot_node_repr = self.prot_projection(prot_fused)
        drug_feat_node = self.drug_projection(res_drug)
        prot_feat_node = self.prot_projection(res_prot)
        t_head_start = time.time()
        if edge_index is not None:
            if (
                self.moe_enable
                and atom_prior is not None
                and res_prior is not None
                and drug_node_to_entity is not None
                and protein_node_to_entity is not None
            ):
                temp = max(self.alpha_temp, 1e-3)
                drug_ids, drug_inv = torch.unique(drug_node_to_entity, sorted=True, return_inverse=True)
                prot_ids, prot_inv = torch.unique(protein_node_to_entity, sorted=True, return_inverse=True)
                if drug_ids.numel() == 0 or prot_ids.numel() == 0:
                    edge_logits = drug_node_repr.new_zeros((0,))
                    zero_drug = torch.zeros((drug_ids.numel(), drug_node_repr.size(1)), device=drug_node_repr.device)
                    zero_prot = torch.zeros((prot_ids.numel(), prot_node_repr.size(1)), device=prot_node_repr.device)
                    zero = edge_logits.new_tensor(0.0)
                    self._last_aux = {}
                    return _pack(
                        edge_logits,
                        (zero_drug, zero_drug, zero_prot, zero_prot, zero, zero, None, None),
                        None,
                    )

                p_atom = normalize_by_group(atom_prior + self.alpha_eps, drug_inv, drug_ids.numel(), eps=self.alpha_eps)
                p_res = normalize_by_group(res_prior + self.alpha_eps, prot_inv, prot_ids.numel(), eps=self.alpha_eps)
                p_atom = p_atom.clamp_min(self.prior_eps)
                p_res = p_res.clamp_min(self.prior_eps)
                p_atom = normalize_by_group(p_atom, drug_inv, drug_ids.numel(), eps=self.prior_eps)
                p_res = normalize_by_group(p_res, prot_inv, prot_ids.numel(), eps=self.prior_eps)
                if p_res is not None:
                    prior_power = 4.0
                    p_res = torch.pow(p_res, prior_power)
                    p_res = normalize_by_group(p_res, prot_inv, prot_ids.numel(), eps=1e-12)
                if self.prior_smoothing > 0:
                    p_atom = smooth_by_group(p_atom, drug_inv, drug_ids.numel(), smooth=self.prior_smoothing)
                    p_res = smooth_by_group(p_res, prot_inv, prot_ids.numel(), smooth=self.prior_smoothing)

                s_atom = self.atom_attn_refiner(drug_fused).squeeze(-1)
                s_res = self.res_attn_refiner(prot_fused).squeeze(-1)
                s_atom_ref = s_atom
                s_res_ref = s_res
                if self._freeze_delta:
                    s_atom_ref = s_atom.detach() * 0.0
                    s_res_ref = s_res.detach() * 0.0
                # logit stats before/after temperature (for diagnostics)
                if self.prior_mix_mode == "additive":
                    log_p_atom_dbg = torch.log(p_atom.clamp_min(self.prior_eps))
                    log_p_res_dbg = torch.log(p_res.clamp_min(self.prior_eps))
                    logits_atom_pre_dbg = log_p_atom_dbg + self.attn_logit_scale * s_atom_ref
                    logits_res_pre_dbg = log_p_res_dbg + self.attn_logit_scale * s_res_ref
                else:
                    logits_atom_pre_dbg = self.attn_logit_scale * s_atom_ref
                    logits_res_pre_dbg = self.attn_logit_scale * s_res_ref
                logits_atom_post_dbg = logits_atom_pre_dbg / temp
                logits_res_post_dbg = logits_res_pre_dbg / temp
                alpha_learn_atom = group_softmax(s_atom_ref / temp, drug_inv, drug_ids.numel())
                alpha_learn_res = group_softmax(s_res_ref / temp, prot_inv, prot_ids.numel())

                if self.prior_mix_mode == "additive":
                    log_p_atom = torch.log(p_atom.clamp_min(self.prior_eps))
                    log_p_res = torch.log(p_res.clamp_min(self.prior_eps))
                    logits_atom = log_p_atom + self.attn_logit_scale * s_atom_ref
                    logits_res = log_p_res + self.attn_logit_scale * s_res_ref
                    alpha_atom = group_softmax(logits_atom / temp, drug_inv, drug_ids.numel())
                    alpha_res = group_softmax(logits_res / temp, prot_inv, prot_ids.numel())
                else:
                    if self.prior_mix_conditional and self.prior_mix_mlp_drug is not None:
                        deg_d = drug_degree[drug_ids].to(device=drug_feat.device, dtype=drug_feat.dtype) if drug_degree is not None else drug_ids.new_zeros((drug_ids.numel(),), dtype=drug_feat.dtype)
                        deg_p = prot_degree[prot_ids].to(device=prot_feat.device, dtype=prot_feat.dtype) if prot_degree is not None else prot_ids.new_zeros((prot_ids.numel(),), dtype=prot_feat.dtype)
                        prior_ent_atom = group_entropy(p_atom, drug_inv, drug_ids.numel())
                        prior_ent_res = group_entropy(p_res, prot_inv, prot_ids.numel())
                        attn_ent_atom = group_entropy(alpha_learn_atom, drug_inv, drug_ids.numel())
                        attn_ent_res = group_entropy(alpha_learn_res, prot_inv, prot_ids.numel())
                        esm_missing_d = deg_d.new_zeros((deg_d.numel(),))
                        if esm_flag is not None:
                            esm_missing_p = esm_flag[prot_ids].to(device=prot_feat.device, dtype=prot_feat.dtype)
                        else:
                            esm_missing_p = deg_p.new_zeros((deg_p.numel(),))
                        if "deg_drug" not in self.prior_mix_features:
                            deg_d = deg_d * 0
                        if "deg_prot" not in self.prior_mix_features:
                            deg_p = deg_p * 0
                        if "prior_entropy" not in self.prior_mix_features:
                            prior_ent_atom = prior_ent_atom * 0
                            prior_ent_res = prior_ent_res * 0
                        if "attn_entropy" not in self.prior_mix_features:
                            attn_ent_atom = attn_ent_atom * 0
                            attn_ent_res = attn_ent_res * 0
                        if ("esm_missing" not in self.prior_mix_features) and ("esm_unreliable" not in self.prior_mix_features):
                            esm_missing_p = esm_missing_p * 0
                        feat_d = torch.stack([deg_d, prior_ent_atom, attn_ent_atom, esm_missing_d], dim=1)
                        feat_p = torch.stack([deg_p, prior_ent_res, attn_ent_res, esm_missing_p], dim=1)
                        lam_d = torch.sigmoid(self.prior_mix_mlp_drug(feat_d)).view(-1)
                        lam_p = torch.sigmoid(self.prior_mix_mlp_prot(feat_p)).view(-1)
                    else:
                        if self.prior_mix_learnable:
                            lam = torch.sigmoid(self.prior_mix_logit)
                        else:
                            lam = torch.tensor(self.prior_mix_lambda, device=drug_feat.device, dtype=drug_feat.dtype)
                        lam_d = lam.expand(drug_ids.numel())
                        lam_p = lam.expand(prot_ids.numel())
                    alpha_atom = (1.0 - lam_d[drug_inv]) * alpha_learn_atom + lam_d[drug_inv] * p_atom
                    alpha_res = (1.0 - lam_p[prot_inv]) * alpha_learn_res + lam_p[prot_inv] * p_res
                    alpha_atom = normalize_by_group(alpha_atom, drug_inv, drug_ids.numel(), eps=self.alpha_eps)
                    alpha_res = normalize_by_group(alpha_res, prot_inv, prot_ids.numel(), eps=self.alpha_eps)

                # Expert A: prior refine
                drug_repr_A, _, _ = self._pool_by_index_with_alpha(
                    drug_node_repr, drug_node_to_entity, alpha_atom
                )
                prot_repr_A, _, _ = self._pool_by_index_max(
                    prot_node_repr, protein_node_to_entity
                )
                # Expert B: topK + randK (learned scores)
                drug_repr_B, _, _ = self._pool_topk_randk(
                    drug_node_repr, drug_node_to_entity, s_atom,
                    self.pool_topk, self.pool_randk,
                    beta_mix=self.beta_mix,
                    rand_weight_mode=self.randk_weight_mode,
                    prior_floor=self.prior_floor,
                    seed=0,
                )
                prot_repr_B, _, _ = self._pool_topk_randk(
                    prot_node_repr, protein_node_to_entity, s_res,
                    self.pool_topk, self.pool_randk,
                    beta_mix=self.beta_mix,
                    rand_weight_mode=self.randk_weight_mode,
                    prior_floor=self.prior_floor,
                    seed=1,
                )

                # Expert C: cold retrieval (kNN)
                if drug_degree is not None:
                    cold_drug_mask = drug_degree[drug_ids] <= float(cold_drug_th)
                else:
                    cold_drug_mask = drug_ids.new_zeros((drug_ids.numel(),), dtype=torch.bool)
                if prot_degree is not None:
                    cold_prot_mask = prot_degree[prot_ids] <= float(cold_prot_th)
                else:
                    cold_prot_mask = prot_ids.new_zeros((prot_ids.numel(),), dtype=torch.bool)

                drug_repr_C = drug_repr_B
                prot_repr_C = prot_repr_B
                if self.use_knn_graph and drug_knn_edge_index is not None and drug_knn_edge_weight is not None:
                    t0_knn = time.time()
                    num_drugs_total = int(drug_degree.numel()) if drug_degree is not None else int(drug_ids.max().item()) + 1
                    drug_full = drug_repr_B.new_zeros((num_drugs_total, drug_repr_B.size(1)))
                    drug_full[drug_ids] = drug_repr_B
                    drug_knn_full = self._knn_aggregate(
                        drug_full, drug_knn_edge_index, drug_knn_edge_weight, num_drugs_total
                    )
                    drug_knn_repr = drug_knn_full[drug_ids]
                    drug_repr_C = drug_repr_B.clone()
                    drug_repr_C[cold_drug_mask] = drug_knn_repr[cold_drug_mask]
                    t_knn += time.time() - t0_knn
                if self.use_knn_graph and prot_knn_edge_index is not None and prot_knn_edge_weight is not None:
                    t0_knn = time.time()
                    num_prot_total = int(prot_degree.numel()) if prot_degree is not None else int(prot_ids.max().item()) + 1
                    prot_full = prot_repr_B.new_zeros((num_prot_total, prot_repr_B.size(1)))
                    prot_full[prot_ids] = prot_repr_B
                    prot_knn_full = self._knn_aggregate(
                        prot_full, prot_knn_edge_index, prot_knn_edge_weight, num_prot_total
                    )
                    prot_knn_repr = prot_knn_full[prot_ids]
                    prot_repr_C = prot_repr_B.clone()
                    prot_repr_C[cold_prot_mask] = prot_knn_repr[cold_prot_mask]
                    t_knn += time.time() - t0_knn

                # Cold-start feature-only gate (applied to all experts)
                if use_coldstart_gate and drug_degree is not None and prot_degree is not None:
                    gate_d = self._degree_gate(drug_degree, self.gate_drug, device=drug_feat.device)
                    gate_p = self._degree_gate(prot_degree, self.gate_prot, device=prot_feat.device)
                    if gate_d is not None and gate_p is not None:
                        gate_d = gate_d[drug_ids]
                        gate_p = gate_p[prot_ids]
                        drug_feat_repr_c, _, _ = self._pool_by_index_with_alpha(
                            drug_feat_node, drug_node_to_entity, alpha_atom
                        )
                        prot_feat_repr_c, _, _ = self._pool_by_index_max(
                            prot_feat_node, protein_node_to_entity
                        )
                        drug_feat_repr_c = self.drug_feat_mlp(drug_feat_repr_c)
                        prot_feat_repr_c = self.prot_feat_mlp(prot_feat_repr_c)
                        drug_repr_A = gate_d.view(-1, 1) * drug_repr_A + (1.0 - gate_d).view(-1, 1) * drug_feat_repr_c
                        prot_repr_A = gate_p.view(-1, 1) * prot_repr_A + (1.0 - gate_p).view(-1, 1) * prot_feat_repr_c
                        drug_repr_B = gate_d.view(-1, 1) * drug_repr_B + (1.0 - gate_d).view(-1, 1) * drug_feat_repr_c
                        prot_repr_B = gate_p.view(-1, 1) * prot_repr_B + (1.0 - gate_p).view(-1, 1) * prot_feat_repr_c
                        drug_repr_C = gate_d.view(-1, 1) * drug_repr_C + (1.0 - gate_d).view(-1, 1) * drug_feat_repr_c
                        prot_repr_C = gate_p.view(-1, 1) * prot_repr_C + (1.0 - gate_p).view(-1, 1) * prot_feat_repr_c
                # Cross-modal InfoNCE key must use raw protein feature space (no graph message passing).
                prot_raw_repr, _, _ = self._pool_by_index_max(
                    prot_feat_node, protein_node_to_entity
                )
                prot_seq_repr_raw, _, _ = self._pool_by_index_max(
                    prot_feat, protein_node_to_entity
                )
                # Strict inductive borrowing: degree==0 proteins only retrieve from warm(degree>0) proteins.
                borrow_stats_moe = {}
                prot_repr_A, borrow_stats_moe["A"] = self._inductive_knn_feature_borrow(
                    prot_repr_A,
                    prot_seq_repr_raw,
                    prot_ids,
                    prot_degree,
                    warm_allow_mask=prot_warm_mask,
                    topk=self.inductive_knn_k,
                    mix=self.inductive_knn_mix,
                )
                prot_repr_B, borrow_stats_moe["B"] = self._inductive_knn_feature_borrow(
                    prot_repr_B,
                    prot_seq_repr_raw,
                    prot_ids,
                    prot_degree,
                    warm_allow_mask=prot_warm_mask,
                    topk=self.inductive_knn_k,
                    mix=self.inductive_knn_mix,
                )
                prot_repr_C, borrow_stats_moe["C"] = self._inductive_knn_feature_borrow(
                    prot_repr_C,
                    prot_seq_repr_raw,
                    prot_ids,
                    prot_degree,
                    warm_allow_mask=prot_warm_mask,
                    topk=self.inductive_knn_k,
                    mix=self.inductive_knn_mix,
                )

                edge_drug_ids = edge_index[:, 0]
                edge_prot_ids = edge_index[:, 1]
                drug_pos = torch.searchsorted(drug_ids, edge_drug_ids)
                prot_pos = torch.searchsorted(prot_ids, edge_prot_ids)
                drug_mask = (drug_pos < drug_ids.numel()) & (drug_ids[drug_pos] == edge_drug_ids)
                prot_mask = (prot_pos < prot_ids.numel()) & (prot_ids[prot_pos] == edge_prot_ids)
                edge_mask = drug_mask & prot_mask
                pair_repr = None
                if edge_mask.any():
                    drug_pos_m = drug_pos[edge_mask]
                    prot_pos_m = prot_pos[edge_mask]
                    edge_drug_A = drug_repr_A[drug_pos_m]
                    edge_prot_A = prot_repr_A[prot_pos_m]
                    edge_drug_B = drug_repr_B[drug_pos_m]
                    edge_prot_B = prot_repr_B[prot_pos_m]
                    edge_drug_C = drug_repr_C[drug_pos_m]
                    edge_prot_C = prot_repr_C[prot_pos_m]
                    edge_prot_raw = prot_raw_repr[prot_pos_m]
                    pair_repr = torch.cat([edge_drug_B, edge_prot_raw], dim=1)

                    def _edge_logit(edge_drug, edge_prot):
                        return self._edge_logits(edge_drug, edge_prot)

                    logit_A = _edge_logit(edge_drug_A, edge_prot_A) if self.expert_A else edge_drug_A.new_zeros((edge_drug_A.size(0),))
                    logit_B = _edge_logit(edge_drug_B, edge_prot_B) if self.expert_B else edge_drug_B.new_zeros((edge_drug_B.size(0),))
                    logit_C = _edge_logit(edge_drug_C, edge_prot_C) if self.expert_C else edge_drug_C.new_zeros((edge_drug_C.size(0),))

                    # gate
                    if drug_degree is not None:
                        deg_d = torch.log1p(drug_degree[edge_drug_ids[edge_mask]].to(device=drug_feat.device, dtype=drug_feat.dtype))
                    else:
                        deg_d = edge_drug_A.new_zeros((edge_drug_A.size(0),))
                    if prot_degree is not None:
                        deg_p = torch.log1p(prot_degree[edge_prot_ids[edge_mask]].to(device=prot_feat.device, dtype=prot_feat.dtype))
                    else:
                        deg_p = edge_prot_A.new_zeros((edge_prot_A.size(0),))
                    t0_gate = time.time()
                    prior_ent_atom = group_entropy(p_atom, drug_inv, drug_ids.numel()).clamp(0.0, 1.0)
                    prior_ent_res = group_entropy(p_res, prot_inv, prot_ids.numel()).clamp(0.0, 1.0)
                    attn_ent_atom = group_entropy(alpha_learn_atom, drug_inv, drug_ids.numel()).clamp(0.0, 1.0)
                    attn_ent_res = group_entropy(alpha_learn_res, prot_inv, prot_ids.numel()).clamp(0.0, 1.0)
                    prior_ent_d = prior_ent_atom[drug_pos_m]
                    prior_ent_p = prior_ent_res[prot_pos_m]
                    prior_conf_d = (1.0 - prior_ent_d).clamp(0.0, 1.0)
                    prior_conf_p = (1.0 - prior_ent_p).clamp(0.0, 1.0)
                    prior_conf_edge = 0.5 * (prior_conf_d + prior_conf_p)
                    attn_ent_d = attn_ent_atom[drug_pos_m]
                    attn_ent_p = attn_ent_res[prot_pos_m]
                    if esm_flag is not None:
                        esm_miss = esm_flag[edge_prot_ids[edge_mask]].to(device=prot_feat.device, dtype=prot_feat.dtype)
                    else:
                        esm_miss = edge_prot_A.new_zeros((edge_prot_A.size(0),))
                    # Use prior confidence in gate features to down-weight uniform priors.
                    gate_prior_d = prior_conf_d
                    gate_prior_p = prior_conf_p
                    gate_feat = torch.stack([deg_d, deg_p, gate_prior_d, gate_prior_p, attn_ent_d, attn_ent_p, esm_miss], dim=1)
                    self._last_gate_inputs = gate_feat.detach()
                    gate_logits = self.moe_gate(gate_feat)
                    gate_w = torch.softmax(gate_logits, dim=1)
                    # Disable experts (out-of-place to keep autograd happy)
                    expert_mask = gate_w.new_tensor([
                        1.0 if self.expert_A else 0.0,
                        1.0 if self.expert_B else 0.0,
                        1.0 if self.expert_C else 0.0,
                    ])
                    gate_w = gate_w * expert_mask.view(1, -1)
                    # Cold-only Expert C (out-of-place)
                    if drug_degree is not None and prot_degree is not None:
                        edge_deg_drug = drug_degree[edge_drug_ids[edge_mask]]
                        edge_deg_prot = prot_degree[edge_prot_ids[edge_mask]]
                        cold_edge = (edge_deg_drug <= float(cold_drug_th)) | (
                            edge_deg_prot <= float(cold_prot_th)
                        )
                        cold_edge_zero = (edge_deg_drug <= 0) | (edge_deg_prot <= 0)
                    else:
                        cold_edge = torch.zeros_like(gate_w[:, 0], dtype=torch.bool)
                        cold_edge_zero = torch.zeros_like(gate_w[:, 0], dtype=torch.bool)
                    warm_mask = ~cold_edge
                    if gate_w.numel():
                        cold_mask = cold_edge.to(dtype=gate_w.dtype).view(-1, 1)
                        ones_mask = torch.ones_like(cold_mask)
                        allow_mask = torch.cat([ones_mask, ones_mask, cold_mask], dim=1)
                        gate_w = gate_w * allow_mask
                        if expertC_scale is not None and expertC_scale < 1.0:
                            scale_vec = gate_w.new_tensor([1.0, 1.0, expertC_scale]).view(1, 3)
                            gate_w = gate_w * scale_vec
                        if wC_cap is not None and wC_cap > 0:
                            cap_vec = gate_w.new_tensor([1.0, 1.0, float(wC_cap)]).view(1, 3)
                            gate_w = torch.minimum(gate_w, cap_vec)
                    gate_w = gate_w / gate_w.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    if warm_mask.any():
                        warm_mask_col = warm_mask.view(-1, 1)
                        warm_scale = gate_w.new_tensor([1.0, 1.0, 0.0]).view(1, 3)
                        gate_w_warm_only = gate_w * warm_scale
                        gate_w_warm_only = gate_w_warm_only / gate_w_warm_only.sum(
                            dim=1, keepdim=True
                        ).clamp_min(1e-12)
                        gate_w = torch.where(warm_mask_col, gate_w_warm_only, gate_w)
                    wC_mean = float(gate_w[:, 2].mean().item()) if gate_w.numel() else 0.0
                    wC_global_scale = 1.0
                    if wC_mean > 0.2:
                        wC_global_scale = 0.2 / max(wC_mean, 1e-12)
                        c_scale_vec = gate_w.new_tensor([1.0, 1.0, wC_global_scale]).view(1, 3)
                        gate_w = gate_w * c_scale_vec
                        gate_w = gate_w / gate_w.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    cold_zero_edge_count = 0
                    cold_zero_edge_ratio = 0.0
                    hard_zero_edge_count = 0
                    hard_zero_edge_ratio = 0.0
                    if gate_w.numel() and cold_edge_zero.any():
                        cold_zero_edge_count = int(cold_edge_zero.sum().item())
                        cold_zero_edge_ratio = float(cold_edge_zero.float().mean().item())
                        if self.cold_zero_route_mode == "hard":
                            if self.expert_C:
                                cold_routing = gate_w.new_tensor([0.0, 0.0, 1.0]).view(1, 3).expand_as(gate_w)
                                gate_w = torch.where(cold_edge_zero.view(-1, 1), cold_routing, gate_w)
                                hard_zero_edge_count = cold_zero_edge_count
                                hard_zero_edge_ratio = cold_zero_edge_ratio
                        elif self.cold_zero_route_mode == "soft":
                            min_wc = self.cold_zero_route_min_wc if self.expert_C else 0.0
                            if min_wc > 0.0:
                                wc = gate_w[:, 2]
                                min_wc_col = wc.new_full(wc.size(), float(min_wc))
                                wc = torch.where(cold_edge_zero, torch.maximum(wc, min_wc_col), wc)
                                gate_w = torch.stack([gate_w[:, 0], gate_w[:, 1], wc], dim=1)
                                gate_w = gate_w / gate_w.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    wC_mean = float(gate_w[:, 2].mean().item()) if gate_w.numel() else 0.0
                    self._last_gate_weights = gate_w.detach()
                    t_gate += time.time() - t0_gate

                    edge_logits = gate_w[:, 0] * logit_A + gate_w[:, 1] * logit_B + gate_w[:, 2] * logit_C
                    wC_warm = float(gate_w[warm_mask, 2].mean().item()) if warm_mask.any() else 0.0
                    wC_cold = float(gate_w[cold_edge, 2].mean().item()) if cold_edge.any() else 0.0
                    if gate_w is not None:
                        gate_w_warm = gate_w[warm_mask].mean(dim=0) if warm_mask.any() else gate_w.new_zeros((3,))
                        gate_w_cold = gate_w[cold_edge].mean(dim=0) if cold_edge.any() else gate_w.new_zeros((3,))
                    else:
                        gate_w_warm = None
                        gate_w_cold = None
                    cold_ratio = float(cold_edge.float().mean().item()) if gate_w.numel() else 0.0
                    cold_drug_only = float((cold_edge & ~cold_prot_mask[prot_pos_m]).float().mean().item()) if cold_edge.any() else 0.0
                    cold_prot_only = float((cold_edge & ~cold_drug_mask[drug_pos_m]).float().mean().item()) if cold_edge.any() else 0.0
                    cold_both = float((cold_edge & cold_drug_mask[drug_pos_m] & cold_prot_mask[prot_pos_m]).float().mean().item()) if cold_edge.any() else 0.0
                    self._last_teacher_logits = logit_A.detach()
                    self._last_cold_edge_mask = cold_edge.detach()
                else:
                    edge_logits = drug_repr_A.new_zeros((0,))
                    gate_w = None
                    wC_warm = 0.0
                    wC_cold = 0.0
                    cold_zero_edge_count = 0
                    cold_zero_edge_ratio = 0.0
                    hard_zero_edge_count = 0
                    hard_zero_edge_ratio = 0.0
                    cold_edge_zero = torch.zeros_like(edge_logits, dtype=torch.bool)
                    cold_ratio = 0.0
                    cold_drug_only = 0.0
                    cold_prot_only = 0.0
                    cold_both = 0.0
                    gate_w_warm = None
                    gate_w_cold = None
                    self._last_teacher_logits = None
                    self._last_cold_edge_mask = None

                # KL stats
                if self.alpha_refine:
                    stage1_kl = bool(self._freeze_delta)
                    if self.prior_mix_mode == "mixture":
                        if stage1_kl:
                            alpha_for_kl_atom = alpha_atom
                            alpha_for_kl_res = alpha_res
                        else:
                            alpha_for_kl_atom = alpha_learn_atom
                            alpha_for_kl_res = alpha_learn_res
                    else:
                        alpha_for_kl_atom = alpha_atom
                        alpha_for_kl_res = alpha_res
                    kl_atom_raw, kl_atom_norm, ent_atom, kl_atom_detail = _group_kl_stats(
                        alpha_for_kl_atom, p_atom, drug_inv, drug_ids.numel(),
                        alpha_eps=self.alpha_eps, prior_eps=self.prior_eps, return_details=True
                    )
                    kl_res_raw, kl_res_norm, ent_res, kl_res_detail = _group_kl_stats(
                        alpha_for_kl_res, p_res, prot_inv, prot_ids.numel(),
                        alpha_eps=self.alpha_eps, prior_eps=self.prior_eps, return_details=True
                    )
                    attn_kl_raw = kl_atom_raw + kl_res_raw
                    attn_kl_norm = kl_atom_norm + kl_res_norm
                else:
                    attn_kl_raw = drug_repr_A.new_tensor(0.0)
                    attn_kl_norm = drug_repr_A.new_tensor(0.0)
                    ent_atom = drug_repr_A.new_tensor(0.0)
                    ent_res = drug_repr_A.new_tensor(0.0)
                    kl_atom_detail = None
                    kl_res_detail = None

                kl_clip_count = 0
                kl_nan_count = 0
                prior_nan_count = 0
                renorm_count = 0
                attn_kl_norm_raw = attn_kl_norm
                if kl_atom_detail is not None:
                    prior_nan_count += int(kl_atom_detail.get("prior_nan_count", 0))
                    renorm_count += int(kl_atom_detail.get("renorm_count", 0))
                if kl_res_detail is not None:
                    prior_nan_count += int(kl_res_detail.get("prior_nan_count", 0))
                    renorm_count += int(kl_res_detail.get("renorm_count", 0))
                if not torch.isfinite(attn_kl_norm).all():
                    kl_nan_count = 1
                    attn_kl_norm = drug_repr_A.new_tensor(0.0)
                    attn_kl_raw = drug_repr_A.new_tensor(0.0)
                else:
                    if self.attn_kl_clip is not None and float(self.attn_kl_clip) > 0:
                        if float(attn_kl_norm.detach().item()) > float(self.attn_kl_clip):
                            kl_clip_count = 1
                            attn_kl_norm = drug_repr_A.new_tensor(float(self.attn_kl_clip))
                            if self._kl_warn_count < 5:
                                print("[WARN] attn_kl_norm clipped to attn_kl_clip.")
                                self._kl_warn_count += 1

                # Bind entropy diagnostics to the exact distributions used by KL (sanitized when available).
                alpha_for_ent_atom = (
                    kl_atom_detail.get("alpha") if isinstance(kl_atom_detail, dict) else None
                )
                alpha_for_ent_res = (
                    kl_res_detail.get("alpha") if isinstance(kl_res_detail, dict) else None
                )
                prior_for_ent_atom = (
                    kl_atom_detail.get("prior") if isinstance(kl_atom_detail, dict) else None
                )
                prior_for_ent_res = (
                    kl_res_detail.get("prior") if isinstance(kl_res_detail, dict) else None
                )
                if alpha_for_ent_atom is None:
                    alpha_for_ent_atom = alpha_for_kl_atom if self.alpha_refine else alpha_atom
                if alpha_for_ent_res is None:
                    alpha_for_ent_res = alpha_for_kl_res if self.alpha_refine else alpha_res
                if prior_for_ent_atom is None:
                    prior_for_ent_atom = p_atom
                if prior_for_ent_res is None:
                    prior_for_ent_res = p_res

                kl_alpha_atom = (
                    kl_atom_detail.get("alpha") if isinstance(kl_atom_detail, dict) else alpha_for_ent_atom
                )
                kl_prior_atom = (
                    kl_atom_detail.get("prior") if isinstance(kl_atom_detail, dict) else prior_for_ent_atom
                )
                kl_alpha_res = (
                    kl_res_detail.get("alpha") if isinstance(kl_res_detail, dict) else alpha_for_ent_res
                )
                kl_prior_res = (
                    kl_res_detail.get("prior") if isinstance(kl_res_detail, dict) else prior_for_ent_res
                )
                sanity_atom = _kl_entropy_sanity(
                    alpha_for_ent_atom, prior_for_ent_atom, drug_inv,
                    kl_alpha_atom, kl_prior_atom, drug_inv,
                    drug_ids.numel(),
                    softmax_dim_entropy="group(0)",
                    softmax_dim_kl="group(0)",
                )
                sanity_res = _kl_entropy_sanity(
                    alpha_for_ent_res, prior_for_ent_res, prot_inv,
                    kl_alpha_res, kl_prior_res, prot_inv,
                    prot_ids.numel(),
                    softmax_dim_entropy="group(0)",
                    softmax_dim_kl="group(0)",
                )
                if (
                    self._sanity_kl_print_count < self._sanity_kl_print_limit
                    or sanity_atom["mismatch"]
                    or sanity_res["mismatch"]
                ):
                    print(
                        f"[SANITY-KL][atom] ent(mean/var) alpha={sanity_atom['entropy_alpha_mean']:.6f}/{sanity_atom['entropy_alpha_var']:.6f} "
                        f"prior={sanity_atom['entropy_prior_mean']:.6f}/{sanity_atom['entropy_prior_var']:.6f}; "
                        f"kl(mean/var) alpha={sanity_atom['kl_alpha_mean']:.6f}/{sanity_atom['kl_alpha_var']:.6f} "
                        f"prior={sanity_atom['kl_prior_mean']:.6f}/{sanity_atom['kl_prior_var']:.6f}; "
                        f"same_softmax_dim={sanity_atom['same_softmax_dim']} same_mask={sanity_atom['same_mask']} "
                        f"same_group={sanity_atom['same_group']} valid_groups(ent/kl)="
                        f"{sanity_atom['entropy_valid_groups']}/{sanity_atom['kl_valid_groups']}"
                    )
                    print(
                        f"[SANITY-KL][res ] ent(mean/var) alpha={sanity_res['entropy_alpha_mean']:.6f}/{sanity_res['entropy_alpha_var']:.6f} "
                        f"prior={sanity_res['entropy_prior_mean']:.6f}/{sanity_res['entropy_prior_var']:.6f}; "
                        f"kl(mean/var) alpha={sanity_res['kl_alpha_mean']:.6f}/{sanity_res['kl_alpha_var']:.6f} "
                        f"prior={sanity_res['kl_prior_mean']:.6f}/{sanity_res['kl_prior_var']:.6f}; "
                        f"same_softmax_dim={sanity_res['same_softmax_dim']} same_mask={sanity_res['same_mask']} "
                        f"same_group={sanity_res['same_group']} valid_groups(ent/kl)="
                        f"{sanity_res['entropy_valid_groups']}/{sanity_res['kl_valid_groups']}"
                    )
                    self._sanity_kl_print_count += 1

                attn_ent_atom_group = group_entropy(alpha_for_ent_atom, drug_inv, drug_ids.numel())
                attn_ent_res_group = group_entropy(alpha_for_ent_res, prot_inv, prot_ids.numel())
                prior_ent_atom_group = group_entropy(prior_for_ent_atom, drug_inv, drug_ids.numel())
                prior_ent_res_group = group_entropy(prior_for_ent_res, prot_inv, prot_ids.numel())
                attn_ent_atom_raw_group = group_entropy(alpha_for_ent_atom, drug_inv, drug_ids.numel(), normalize=False)
                attn_ent_res_raw_group = group_entropy(alpha_for_ent_res, prot_inv, prot_ids.numel(), normalize=False)
                prior_ent_atom_raw_group = group_entropy(prior_for_ent_atom, drug_inv, drug_ids.numel(), normalize=False)
                prior_ent_res_raw_group = group_entropy(prior_for_ent_res, prot_inv, prot_ids.numel(), normalize=False)
                prior_conf_atom_group = (1.0 - prior_ent_atom_group).clamp(0.0, 1.0)
                prior_conf_res_group = (1.0 - prior_ent_res_group).clamp(0.0, 1.0)
                prior_conf_atom_mean = float(prior_conf_atom_group.mean().item()) if drug_ids.numel() else 0.0
                prior_conf_res_mean = float(prior_conf_res_group.mean().item()) if prot_ids.numel() else 0.0
                if drug_ids.numel() and prot_ids.numel():
                    prior_conf_mean = 0.5 * (prior_conf_atom_mean + prior_conf_res_mean)
                elif drug_ids.numel():
                    prior_conf_mean = prior_conf_atom_mean
                elif prot_ids.numel():
                    prior_conf_mean = prior_conf_res_mean
                else:
                    prior_conf_mean = None
                if prior_conf_mean is not None:
                    self._last_prior_conf = edge_logits.new_tensor(prior_conf_mean)
                delta_reg_atom, delta_rms_atom, delta_rms_w_atom = _delta_reg_stats(
                    s_atom_ref, drug_inv, drug_ids.numel(), prior_conf_group=prior_conf_atom_group
                )
                delta_reg_res, delta_rms_res, delta_rms_w_res = _delta_reg_stats(
                    s_res_ref, prot_inv, prot_ids.numel(), prior_conf_group=prior_conf_res_group
                )
                self._last_delta_reg = 0.5 * (delta_reg_atom + delta_reg_res)
                delta_rms = 0.5 * (delta_rms_atom + delta_rms_res)
                delta_rms_weighted = 0.5 * (delta_rms_w_atom + delta_rms_w_res)
                self._last_delta_rms = delta_rms
                self._last_delta_rms_weighted = delta_rms_weighted
                drug_counts = _group_counts(drug_inv, drug_ids.numel())
                prot_counts = _group_counts(prot_inv, prot_ids.numel())
                prior_max_atom = group_max(prior_for_ent_atom, drug_inv, drug_ids.numel()).clamp_min(0.0)
                prior_max_res = group_max(prior_for_ent_res, prot_inv, prot_ids.numel()).clamp_min(0.0)
                def _stat_vals(t):
                    if t is None or t.numel() == 0:
                        return 0.0, 0.0, 0.0, 0.0, 0.0
                    t = t.to(dtype=torch.float32)
                    t = t[torch.isfinite(t)]
                    if t.numel() == 0:
                        return 0.0, 0.0, 0.0, 0.0, 0.0
                    return (
                        float(t.mean().item()),
                        float(torch.quantile(t, 0.5).item()),
                        float(torch.quantile(t, 0.9).item()),
                        float(t.min().item()),
                        float(t.max().item()),
                    )
                k_atom_mean, k_atom_p50, k_atom_p90, k_atom_min, k_atom_max = _stat_vals(drug_counts)
                k_res_mean, k_res_p50, k_res_p90, k_res_min, k_res_max = _stat_vals(prot_counts)
                logk_atom = torch.log(drug_counts.clamp_min(2.0))
                logk_res = torch.log(prot_counts.clamp_min(2.0))
                logk_atom_mean, logk_atom_p50, logk_atom_p90, logk_atom_min, logk_atom_max = _stat_vals(logk_atom)
                logk_res_mean, logk_res_p50, logk_res_p90, logk_res_min, logk_res_max = _stat_vals(logk_res)
                prior_max_atom_mean, prior_max_atom_p50, prior_max_atom_p90, prior_max_atom_min, prior_max_atom_max = _stat_vals(prior_max_atom)
                prior_max_res_mean, prior_max_res_p50, prior_max_res_p90, prior_max_res_min, prior_max_res_max = _stat_vals(prior_max_res)
                def _q_vals(t):
                    if t is None or t.numel() == 0:
                        return 0.0, 0.0, 0.0
                    t = t.to(dtype=torch.float32)
                    t = t[torch.isfinite(t)]
                    if t.numel() == 0:
                        return 0.0, 0.0, 0.0
                    q = torch.quantile(t, torch.tensor([0.0, 0.5, 0.9], device=t.device))
                    return float(q[0].item()), float(q[1].item()), float(q[2].item())
                alpha_atom_min, alpha_atom_p50, alpha_atom_p90 = _q_vals(
                    kl_atom_detail.get("alpha") if kl_atom_detail is not None else None
                )
                alpha_res_min, alpha_res_p50, alpha_res_p90 = _q_vals(
                    kl_res_detail.get("alpha") if kl_res_detail is not None else None
                )
                prior_atom_min, prior_atom_p50, prior_atom_p90 = _q_vals(
                    kl_atom_detail.get("prior") if kl_atom_detail is not None else None
                )
                prior_res_min, prior_res_p50, prior_res_p90 = _q_vals(
                    kl_res_detail.get("prior") if kl_res_detail is not None else None
                )
                prior_conf_atom_min, prior_conf_atom_p50, prior_conf_atom_p90 = _q_vals(
                    prior_conf_atom_group if prior_conf_atom_group is not None else None
                )
                prior_conf_res_min, prior_conf_res_p50, prior_conf_res_p90 = _q_vals(
                    prior_conf_res_group if prior_conf_res_group is not None else None
                )
                kl_atom_max_idx = None
                kl_res_max_idx = None
                kl_atom_max_val = 0.0
                kl_res_max_val = 0.0
                kl_atom_max_k = 0.0
                kl_res_max_k = 0.0
                if kl_atom_detail is not None and kl_atom_detail.get("kl_group") is not None:
                    klg = kl_atom_detail["kl_group"]
                    if klg.numel() > 0:
                        kl_atom_max_idx = int(torch.argmax(klg).item())
                        kl_atom_max_val = float(klg[kl_atom_max_idx].item())
                        if kl_atom_detail.get("counts") is not None:
                            kl_atom_max_k = float(kl_atom_detail["counts"][kl_atom_max_idx].item())
                if kl_res_detail is not None and kl_res_detail.get("kl_group") is not None:
                    klg = kl_res_detail["kl_group"]
                    if klg.numel() > 0:
                        kl_res_max_idx = int(torch.argmax(klg).item())
                        kl_res_max_val = float(klg[kl_res_max_idx].item())
                        if kl_res_detail.get("counts") is not None:
                            kl_res_max_k = float(kl_res_detail["counts"][kl_res_max_idx].item())
                kl_atom_max_group_id = int(drug_ids[kl_atom_max_idx].item()) if (kl_atom_max_idx is not None and drug_ids.numel()) else -1
                kl_res_max_group_id = int(prot_ids[kl_res_max_idx].item()) if (kl_res_max_idx is not None and prot_ids.numel()) else -1
                kl_atom_max_prior_conf = float(prior_conf_atom_group[kl_atom_max_idx].item()) if (kl_atom_max_idx is not None and prior_conf_atom_group.numel()) else 0.0
                kl_res_max_prior_conf = float(prior_conf_res_group[kl_res_max_idx].item()) if (kl_res_max_idx is not None and prior_conf_res_group.numel()) else 0.0
                kl_atom_max_deg = float(drug_degree[drug_ids[kl_atom_max_idx]].item()) if (kl_atom_max_idx is not None and drug_degree is not None and drug_ids.numel()) else 0.0
                kl_res_max_deg = float(prot_degree[prot_ids[kl_res_max_idx]].item()) if (kl_res_max_idx is not None and prot_degree is not None and prot_ids.numel()) else 0.0

                def _split_stats(alpha, cold_mask):
                    if alpha is None or cold_mask is None:
                        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    alpha = alpha.view(-1)
                    cold_mask = cold_mask.view(-1)
                    if alpha.numel() == 0 or cold_mask.numel() == 0 or alpha.numel() != cold_mask.numel():
                        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    warm_mask = ~cold_mask
                    def _qs(x):
                        if x.numel() == 0:
                            return 0.0, 0.0, 0.0
                        return (
                            float(x.mean().item()),
                            float(torch.quantile(x, 0.5).item()),
                            float(torch.quantile(x, 0.9).item()),
                        )
                    warm = alpha[warm_mask]
                    cold = alpha[cold_mask]
                    w_mean, w_p50, w_p90 = _qs(warm)
                    c_mean, c_p50, c_p90 = _qs(cold)
                    return w_mean, w_p50, w_p90, c_mean, c_p50, c_p90

                mp_atom_w_mean, mp_atom_w_p50, mp_atom_w_p90, mp_atom_c_mean, mp_atom_c_p50, mp_atom_c_p90 = _split_stats(
                    self._last_mp_gate_atom, self._last_mp_gate_atom_cold
                )
                mp_res_w_mean, mp_res_w_p50, mp_res_w_p90, mp_res_c_mean, mp_res_c_p50, mp_res_c_p90 = _split_stats(
                    self._last_mp_gate_res, self._last_mp_gate_res_cold
                )
                def _mean_std(x):
                    if x is None or x.numel() == 0:
                        return 0.0, 0.0
                    return float(x.mean().item()), float(x.std().item())
                logits_atom_pre_mean, logits_atom_pre_std = _mean_std(logits_atom_pre_dbg)
                logits_atom_post_mean, logits_atom_post_std = _mean_std(logits_atom_post_dbg)
                logits_res_pre_mean, logits_res_pre_std = _mean_std(logits_res_pre_dbg)
                logits_res_post_mean, logits_res_post_std = _mean_std(logits_res_post_dbg)

                aux = {
                    "attn_entropy_atom_norm_mean": float(attn_ent_atom_group.mean().item()) if drug_ids.numel() else 0.0,
                    "attn_entropy_res_norm_mean": float(attn_ent_res_group.mean().item()) if prot_ids.numel() else 0.0,
                    "prior_entropy_atom_norm_mean": float(prior_ent_atom_group.mean().item()) if drug_ids.numel() else 0.0,
                    "prior_entropy_res_norm_mean": float(prior_ent_res_group.mean().item()) if prot_ids.numel() else 0.0,
                    "attn_entropy_atom_std": float(attn_ent_atom_group.std().item()) if attn_ent_atom_group.numel() > 1 else 0.0,
                    "attn_entropy_res_std": float(attn_ent_res_group.std().item()) if attn_ent_res_group.numel() > 1 else 0.0,
                    "prior_entropy_atom_std": float(prior_ent_atom_group.std().item()) if prior_ent_atom_group.numel() > 1 else 0.0,
                    "prior_entropy_res_std": float(prior_ent_res_group.std().item()) if prior_ent_res_group.numel() > 1 else 0.0,
                    "attn_entropy_atom_raw_mean": float(attn_ent_atom_raw_group.mean().item()) if drug_ids.numel() else 0.0,
                    "attn_entropy_res_raw_mean": float(attn_ent_res_raw_group.mean().item()) if prot_ids.numel() else 0.0,
                    "prior_entropy_atom_raw_mean": float(prior_ent_atom_raw_group.mean().item()) if drug_ids.numel() else 0.0,
                    "prior_entropy_res_raw_mean": float(prior_ent_res_raw_group.mean().item()) if prot_ids.numel() else 0.0,
                    "attn_entropy_atom_raw_std": float(attn_ent_atom_raw_group.std().item()) if attn_ent_atom_raw_group.numel() > 1 else 0.0,
                    "attn_entropy_res_raw_std": float(attn_ent_res_raw_group.std().item()) if attn_ent_res_raw_group.numel() > 1 else 0.0,
                    "prior_entropy_atom_raw_std": float(prior_ent_atom_raw_group.std().item()) if prior_ent_atom_raw_group.numel() > 1 else 0.0,
                    "prior_entropy_res_raw_std": float(prior_ent_res_raw_group.std().item()) if prior_ent_res_raw_group.numel() > 1 else 0.0,
                    "prior_conf_atom_mean": prior_conf_atom_mean,
                    "prior_conf_res_mean": prior_conf_res_mean,
                    "prior_conf_mean": float(prior_conf_mean) if prior_conf_mean is not None else 0.0,
                    "prior_conf_edge_mean": float(prior_conf_edge.mean().item()) if "prior_conf_edge" in locals() else 0.0,
                    "prior_conf_edge": prior_conf_edge.detach() if "prior_conf_edge" in locals() else None,
                    "prior_conf_atom_p50": prior_conf_atom_p50,
                    "prior_conf_atom_p90": prior_conf_atom_p90,
                    "prior_conf_res_p50": prior_conf_res_p50,
                    "prior_conf_res_p90": prior_conf_res_p90,
                    "attn_kl_norm_raw": float(attn_kl_norm_raw.item()) if torch.is_tensor(attn_kl_norm_raw) else float(attn_kl_norm_raw),
                    "kl_clip_count": int(kl_clip_count),
                    "kl_nan_count": int(kl_nan_count),
                    "prior_nan_count": int(prior_nan_count),
                    "renorm_count": int(renorm_count),
                    "kl_stats": {
                        "kl_clip_count": int(kl_clip_count),
                        "kl_nan_count": int(kl_nan_count),
                        "prior_nan_count": int(prior_nan_count),
                        "renorm_count": int(renorm_count),
                    },
                    "sanity_kl_atom": sanity_atom,
                    "sanity_kl_res": sanity_res,
                    "wC_mean": float(wC_mean) if "wC_mean" in locals() else 0.0,
                    "wC_global_scale": float(wC_global_scale) if "wC_global_scale" in locals() else 1.0,
                    "delta_rms": float(delta_rms.item()) if torch.is_tensor(delta_rms) else float(delta_rms),
                    "delta_rms_weighted": float(delta_rms_weighted.item()) if torch.is_tensor(delta_rms_weighted) else float(delta_rms_weighted),
                    "group_size_atom_mean": k_atom_mean,
                    "group_size_atom_p50": k_atom_p50,
                    "group_size_atom_p90": k_atom_p90,
                    "group_size_atom_min": k_atom_min,
                    "group_size_atom_max": k_atom_max,
                    "group_size_res_mean": k_res_mean,
                    "group_size_res_p50": k_res_p50,
                    "group_size_res_p90": k_res_p90,
                    "group_size_res_min": k_res_min,
                    "group_size_res_max": k_res_max,
                    "logK_atom_mean": logk_atom_mean,
                    "logK_atom_p50": logk_atom_p50,
                    "logK_atom_p90": logk_atom_p90,
                    "logK_atom_min": logk_atom_min,
                    "logK_atom_max": logk_atom_max,
                    "logK_res_mean": logk_res_mean,
                    "logK_res_p50": logk_res_p50,
                    "logK_res_p90": logk_res_p90,
                    "logK_res_min": logk_res_min,
                    "logK_res_max": logk_res_max,
                    "prior_max_prob_atom_mean": prior_max_atom_mean,
                    "prior_max_prob_atom_p50": prior_max_atom_p50,
                    "prior_max_prob_atom_p90": prior_max_atom_p90,
                    "prior_max_prob_atom_min": prior_max_atom_min,
                    "prior_max_prob_atom_max": prior_max_atom_max,
                    "prior_max_prob_res_mean": prior_max_res_mean,
                    "prior_max_prob_res_p50": prior_max_res_p50,
                    "prior_max_prob_res_p90": prior_max_res_p90,
                    "prior_max_prob_res_min": prior_max_res_min,
                    "prior_max_prob_res_max": prior_max_res_max,
                    "alpha_atom_min": alpha_atom_min,
                    "alpha_atom_p50": alpha_atom_p50,
                    "alpha_atom_p90": alpha_atom_p90,
                    "alpha_res_min": alpha_res_min,
                    "alpha_res_p50": alpha_res_p50,
                    "alpha_res_p90": alpha_res_p90,
                    "prior_atom_min": prior_atom_min,
                    "prior_atom_p50": prior_atom_p50,
                    "prior_atom_p90": prior_atom_p90,
                    "prior_res_min": prior_res_min,
                    "prior_res_p50": prior_res_p50,
                    "prior_res_p90": prior_res_p90,
                    "kl_atom_max_group_idx": kl_atom_max_idx if kl_atom_max_idx is not None else -1,
                    "kl_atom_max_group_id": kl_atom_max_group_id,
                    "kl_atom_max_group_kl": kl_atom_max_val,
                    "kl_atom_max_group_k": kl_atom_max_k,
                    "kl_atom_max_group_prior_conf": kl_atom_max_prior_conf,
                    "kl_atom_max_group_deg": kl_atom_max_deg,
                    "kl_res_max_group_idx": kl_res_max_idx if kl_res_max_idx is not None else -1,
                    "kl_res_max_group_id": kl_res_max_group_id,
                    "kl_res_max_group_kl": kl_res_max_val,
                    "kl_res_max_group_k": kl_res_max_k,
                    "kl_res_max_group_prior_conf": kl_res_max_prior_conf,
                    "kl_res_max_group_deg": kl_res_max_deg,
                    "logits_atom_pre_mean": logits_atom_pre_mean,
                    "logits_atom_pre_std": logits_atom_pre_std,
                    "logits_atom_post_mean": logits_atom_post_mean,
                    "logits_atom_post_std": logits_atom_post_std,
                    "logits_res_pre_mean": logits_res_pre_mean,
                    "logits_res_pre_std": logits_res_pre_std,
                    "logits_res_post_mean": logits_res_post_mean,
                    "logits_res_post_std": logits_res_post_std,
                    "gate_w_mean": gate_w.mean(dim=0).detach().cpu().tolist() if gate_w is not None else [0.0, 0.0, 0.0],
                    "gate_w_warm_mean": gate_w_warm.detach().cpu().tolist() if gate_w_warm is not None else [0.0, 0.0, 0.0],
                    "gate_w_cold_mean": gate_w_cold.detach().cpu().tolist() if gate_w_cold is not None else [0.0, 0.0, 0.0],
                    "mp_alpha_atom_warm_mean": mp_atom_w_mean,
                    "mp_alpha_atom_warm_p50": mp_atom_w_p50,
                    "mp_alpha_atom_warm_p90": mp_atom_w_p90,
                    "mp_alpha_atom_cold_mean": mp_atom_c_mean,
                    "mp_alpha_atom_cold_p50": mp_atom_c_p50,
                    "mp_alpha_atom_cold_p90": mp_atom_c_p90,
                    "mp_alpha_res_warm_mean": mp_res_w_mean,
                    "mp_alpha_res_warm_p50": mp_res_w_p50,
                    "mp_alpha_res_warm_p90": mp_res_w_p90,
                    "mp_alpha_res_cold_mean": mp_res_c_mean,
                    "mp_alpha_res_cold_p50": mp_res_c_p50,
                    "mp_alpha_res_cold_p90": mp_res_c_p90,
                    "wC_warm_mean": wC_warm,
                    "wC_cold_mean": wC_cold,
                    "cold_zero_route_mode": str(self.cold_zero_route_mode),
                    "cold_zero_route_min_wc": float(self.cold_zero_route_min_wc),
                    "cold_zero_edge_ratio": float(cold_zero_edge_ratio),
                    "cold_zero_edge_count": int(cold_zero_edge_count),
                    "hard_zero_edge_ratio": float(hard_zero_edge_ratio),
                    "hard_zero_edge_count": int(hard_zero_edge_count),
                    "gate_w_hard_zero_mean": gate_w[cold_edge_zero].mean(dim=0).detach().cpu().tolist()
                    if (gate_w is not None and cold_edge_zero.any()) else [0.0, 0.0, 0.0],
                    "cold_edge_ratio": cold_ratio,
                    "cold_by_drug_only": cold_drug_only,
                    "cold_by_prot_only": cold_prot_only,
                    "cold_by_both": cold_both,
                    "expertC_scale": float(expertC_scale),
                    "wC_cap": float(wC_cap) if wC_cap is not None else None,
                    "inductive_borrow": borrow_stats_moe if "borrow_stats_moe" in locals() else None,
                }
                aux["kl_stats"] = {
                    "kl_clip_count": int(kl_clip_count),
                    "kl_nan_count": int(kl_nan_count),
                    "prior_nan_count": int(prior_nan_count),
                    "renorm_count": int(renorm_count),
                }
                if return_aux:
                    aux.update({
                        "edge_index": edge_index,
                        "edge_mask": edge_mask,
                        "edge_index_filtered": edge_index[edge_mask] if (edge_mask is not None and edge_mask.any()) else edge_index.new_zeros((0, 2)),
                        "atom_attention": alpha_atom,
                        "residue_attention": alpha_res,
                        "atom_attention_learned": alpha_learn_atom,
                        "residue_attention_learned": alpha_learn_res,
                        "prior_attention_atom": p_atom,
                        "prior_attention_res": p_res,
                        "attn_entropy_atom_group": group_entropy(alpha_for_ent_atom, drug_inv, drug_ids.numel()),
                        "attn_entropy_res_group": group_entropy(alpha_for_ent_res, prot_inv, prot_ids.numel()),
                        "prior_entropy_atom_group": group_entropy(prior_for_ent_atom, drug_inv, drug_ids.numel()),
                        "prior_entropy_res_group": group_entropy(prior_for_ent_res, prot_inv, prot_ids.numel()),
                        "drug_ids": drug_ids,
                        "prot_ids": prot_ids,
                        "drug_inv": drug_inv,
                        "prot_inv": prot_inv,
                        "atom_to_drug": drug_node_to_entity,
                        "residue_to_protein": protein_node_to_entity,
                        "mp_alpha_atom": self._last_mp_gate_atom,
                        "mp_alpha_res": self._last_mp_gate_res,
                        "expert_logits": torch.stack([logit_A, logit_B, logit_C], dim=1) if gate_w is not None else None,
                        "gate_weights": gate_w,
                        "gate_inputs": gate_feat.detach() if gate_w is not None else None,
                        "final_logits": edge_logits,
                        "pair_repr": pair_repr,
                        "cold_edge_mask": self._last_cold_edge_mask,
                        "is_cold_drug": cold_drug_mask if drug_degree is not None else None,
                        "is_cold_prot": cold_prot_mask if prot_degree is not None else None,
                    })
                    if drug_degree is not None and prot_degree is not None and edge_mask is not None and edge_mask.any():
                        aux["is_cold_drug_edge"] = (drug_degree[edge_drug_ids[edge_mask]] <= float(cold_drug_th))
                        aux["is_cold_prot_edge"] = (prot_degree[edge_prot_ids[edge_mask]] <= float(cold_prot_th))
                        aux["is_cold_pair"] = aux["is_cold_drug_edge"] | aux["is_cold_prot_edge"]

                    if explain_cfg:
                        aux["atom_orig_pos"] = explain_cfg.get("atom_orig_pos")
                        aux["residue_orig_pos"] = explain_cfg.get("residue_orig_pos")
                        aux["drug_atom_ptr"] = explain_cfg.get("drug_atom_ptr")
                        aux["prot_res_ptr"] = explain_cfg.get("prot_res_ptr")
                        aux["drug_id_list"] = explain_cfg.get("drug_id_list")
                        aux["prot_id_list"] = explain_cfg.get("prot_id_list")

                    if need_node_repr:
                        if retain_node_grad:
                            drug_node_repr.retain_grad()
                            prot_node_repr.retain_grad()
                        aux["drug_node_repr"] = drug_node_repr
                        aux["prot_node_repr"] = prot_node_repr

                    if self.use_knn_graph and explain_cfg.get("knn_topk") is not None:
                        k_top = int(explain_cfg.get("knn_topk", 10))
                        def _knn_neighbors(edge_index_knn, edge_weight_knn, node_ids):
                            out = {}
                            if edge_index_knn is None or edge_weight_knn is None:
                                return out
                            src = edge_index_knn[0]
                            dst = edge_index_knn[1]
                            w = edge_weight_knn
                            for nid in node_ids.tolist():
                                mask = (src == nid)
                                if not mask.any():
                                    out[int(nid)] = {"ids": [], "weights": []}
                                    continue
                                idx = mask.nonzero(as_tuple=False).view(-1)
                                if k_top > 0 and idx.numel() > k_top:
                                    vals, order = torch.topk(w[idx], k_top)
                                    sel = idx[order]
                                else:
                                    sel = idx
                                    vals = w[sel]
                                out[int(nid)] = {
                                    "ids": dst[sel].detach().cpu().tolist(),
                                    "weights": vals.detach().cpu().tolist(),
                                }
                            return out
                        aux["drug_knn_neighbors"] = _knn_neighbors(
                            drug_knn_edge_index, drug_knn_edge_weight, drug_ids
                        )
                        aux["prot_knn_neighbors"] = _knn_neighbors(
                            prot_knn_edge_index, prot_knn_edge_weight, prot_ids
                        )

                self._last_aux = aux
                return _pack(
                    edge_logits,
                    (drug_repr_A, drug_repr_A, prot_repr_A, prot_repr_A, attn_kl_norm, attn_kl_raw, ent_atom, ent_res),
                    pair_repr,
                    aux=aux,
                )
            if (
                atom_prior is not None
                and res_prior is not None
                and drug_node_to_entity is not None
                and protein_node_to_entity is not None
            ):
                drug_repr_c, drug_ids, drug_inv = self._pool_by_index_with_alpha(
                    drug_node_repr, drug_node_to_entity,
                    torch.ones(drug_node_repr.size(0), device=drug_node_repr.device, dtype=drug_node_repr.dtype),
                )
                prot_repr_c, prot_ids, prot_inv = self._pool_by_index_max(
                    prot_node_repr, protein_node_to_entity
                )
                if drug_ids.numel() == 0 or prot_ids.numel() == 0:
                    edge_logits = drug_repr_c.new_zeros((0,))
                    zero_drug = torch.zeros_like(drug_repr_c)
                    zero_prot = torch.zeros_like(prot_repr_c)
                    zero = edge_logits.new_tensor(0.0)
                    return _pack(
                        edge_logits,
                        (zero_drug, zero_drug, zero_prot, zero_prot, zero, zero, None, None),
                        None,
                    )
                p_atom = normalize_by_group(atom_prior + self.alpha_eps, drug_inv, drug_ids.numel(), eps=self.alpha_eps)
                p_res = normalize_by_group(res_prior + self.alpha_eps, prot_inv, prot_ids.numel(), eps=self.alpha_eps)
                p_atom = p_atom.clamp_min(self.prior_eps)
                p_res = p_res.clamp_min(self.prior_eps)
                p_atom = normalize_by_group(p_atom, drug_inv, drug_ids.numel(), eps=self.prior_eps)
                p_res = normalize_by_group(p_res, prot_inv, prot_ids.numel(), eps=self.prior_eps)
                if p_res is not None:
                    prior_power = 4.0
                    p_res = torch.pow(p_res, prior_power)
                    p_res = normalize_by_group(p_res, prot_inv, prot_ids.numel(), eps=1e-12)
                if self.prior_smoothing > 0:
                    p_atom = smooth_by_group(p_atom, drug_inv, drug_ids.numel(), smooth=self.prior_smoothing)
                    p_res = smooth_by_group(p_res, prot_inv, prot_ids.numel(), smooth=self.prior_smoothing)
                prior_ent_atom_group = group_entropy(p_atom, drug_inv, drug_ids.numel())
                prior_ent_res_group = group_entropy(p_res, prot_inv, prot_ids.numel())
                prior_conf_atom_group = (1.0 - prior_ent_atom_group).clamp(0.0, 1.0)
                prior_conf_res_group = (1.0 - prior_ent_res_group).clamp(0.0, 1.0)
                prior_conf_atom_mean = float(prior_conf_atom_group.mean().item()) if drug_ids.numel() else 0.0
                prior_conf_res_mean = float(prior_conf_res_group.mean().item()) if prot_ids.numel() else 0.0
                if drug_ids.numel() and prot_ids.numel():
                    prior_conf_mean = 0.5 * (prior_conf_atom_mean + prior_conf_res_mean)
                elif drug_ids.numel():
                    prior_conf_mean = prior_conf_atom_mean
                elif prot_ids.numel():
                    prior_conf_mean = prior_conf_res_mean
                else:
                    prior_conf_mean = None
                if prior_conf_mean is not None:
                    self._last_prior_conf = drug_node_repr.new_tensor(prior_conf_mean)
                if self.alpha_refine:
                    s_atom = self.atom_attn_refiner(drug_fused).squeeze(-1)
                    s_res = self.res_attn_refiner(prot_fused).squeeze(-1)
                    s_atom_ref = s_atom
                    s_res_ref = s_res
                    if self._freeze_delta:
                        s_atom_ref = s_atom.detach() * 0.0
                        s_res_ref = s_res.detach() * 0.0
                    delta_reg_atom, delta_rms_atom, delta_rms_w_atom = _delta_reg_stats(
                        s_atom_ref, drug_inv, drug_ids.numel(), prior_conf_group=prior_conf_atom_group
                    )
                    delta_reg_res, delta_rms_res, delta_rms_w_res = _delta_reg_stats(
                        s_res_ref, prot_inv, prot_ids.numel(), prior_conf_group=prior_conf_res_group
                    )
                    self._last_delta_reg = 0.5 * (delta_reg_atom + delta_reg_res)
                    self._last_delta_rms = 0.5 * (delta_rms_atom + delta_rms_res)
                    self._last_delta_rms_weighted = 0.5 * (delta_rms_w_atom + delta_rms_w_res)
                    temp = max(self.alpha_temp, 1e-3)
                    log_p_atom = torch.log(p_atom)
                    log_p_res = torch.log(p_res)
                    logits_atom = log_p_atom + self.attn_logit_scale * s_atom_ref
                    logits_res = log_p_res + self.attn_logit_scale * s_res_ref
                    alpha_atom = group_softmax(logits_atom / temp, drug_inv, drug_ids.numel())
                    alpha_res = group_softmax(logits_res / temp, prot_inv, prot_ids.numel())
                else:
                    alpha_atom = p_atom
                    alpha_res = p_res
                    self._last_delta_reg = drug_node_repr.new_tensor(0.0)
                    self._last_delta_rms = drug_node_repr.new_tensor(0.0)
                    self._last_delta_rms_weighted = drug_node_repr.new_tensor(0.0)
                drug_repr_c, drug_ids, _ = self._pool_by_index_with_alpha(
                    drug_node_repr, drug_node_to_entity, alpha_atom
                )
                prot_repr_c, prot_ids, _ = self._pool_by_index_max(
                    prot_node_repr, protein_node_to_entity
                )

                if use_coldstart_gate and drug_degree is not None and prot_degree is not None:
                    drug_feat_repr_c, _, _ = self._pool_by_index_with_alpha(
                        drug_feat_node, drug_node_to_entity, alpha_atom
                    )
                    prot_feat_repr_c, _, _ = self._pool_by_index_max(
                        prot_feat_node, protein_node_to_entity
                    )
                    drug_feat_repr_c = self.drug_feat_mlp(drug_feat_repr_c)
                    prot_feat_repr_c = self.prot_feat_mlp(prot_feat_repr_c)
                    gate_d = self._degree_gate(drug_degree, self.gate_drug, device=drug_feat.device)
                    gate_p = self._degree_gate(prot_degree, self.gate_prot, device=prot_feat.device)
                    if gate_d is not None and gate_p is not None:
                        gate_d = gate_d[drug_ids]
                        gate_p = gate_p[prot_ids]
                        drug_repr_c = gate_d.view(-1, 1) * drug_repr_c + (1.0 - gate_d).view(-1, 1) * drug_feat_repr_c
                        prot_repr_c = gate_p.view(-1, 1) * prot_repr_c + (1.0 - gate_p).view(-1, 1) * prot_feat_repr_c

                pair_repr = None
                prot_raw_repr, _, _ = self._pool_by_index_max(
                    prot_feat_node, protein_node_to_entity
                )
                prot_seq_repr_raw, _, _ = self._pool_by_index_max(
                    prot_feat, protein_node_to_entity
                )
                prot_repr_c, borrow_stats_refine = self._inductive_knn_feature_borrow(
                    prot_repr_c,
                    prot_seq_repr_raw,
                    prot_ids,
                    prot_degree,
                    warm_allow_mask=prot_warm_mask,
                    topk=self.inductive_knn_k,
                    mix=self.inductive_knn_mix,
                )
                edge_drug_ids = edge_index[:, 0]
                edge_prot_ids = edge_index[:, 1]
                drug_pos = torch.searchsorted(drug_ids, edge_drug_ids)
                prot_pos = torch.searchsorted(prot_ids, edge_prot_ids)
                drug_mask = (drug_pos < drug_ids.numel()) & (drug_ids[drug_pos] == edge_drug_ids)
                prot_mask = (prot_pos < prot_ids.numel()) & (prot_ids[prot_pos] == edge_prot_ids)
                edge_mask = drug_mask & prot_mask
                if edge_mask.any():
                    edge_drug = drug_repr_c[drug_pos[edge_mask]]
                    edge_prot = prot_repr_c[prot_pos[edge_mask]]
                    edge_prot_raw = prot_raw_repr[prot_pos[edge_mask]]
                    pair_repr = torch.cat([edge_drug, edge_prot_raw], dim=1)
                    edge_logits = self._edge_logits(edge_drug, edge_prot)
                else:
                    edge_logits = drug_repr_c.new_zeros((0,))

                if self.alpha_refine:
                    kl_atom_raw, kl_atom_norm, ent_atom, kl_atom_detail = _group_kl_stats(
                        alpha_atom, p_atom, drug_inv, drug_ids.numel(),
                        alpha_eps=self.alpha_eps, prior_eps=self.prior_eps, return_details=True
                    )
                    kl_res_raw, kl_res_norm, ent_res, kl_res_detail = _group_kl_stats(
                        alpha_res, p_res, prot_inv, prot_ids.numel(),
                        alpha_eps=self.alpha_eps, prior_eps=self.prior_eps, return_details=True
                    )
                    attn_kl_raw = kl_atom_raw + kl_res_raw
                    attn_kl_norm = kl_atom_norm + kl_res_norm
                else:
                    attn_kl_raw = drug_repr_c.new_tensor(0.0)
                    attn_kl_norm = drug_repr_c.new_tensor(0.0)
                    ent_atom = drug_repr_c.new_tensor(0.0)
                    ent_res = drug_repr_c.new_tensor(0.0)
                    kl_atom_detail = None
                    kl_res_detail = None

                kl_clip_count = 0
                kl_nan_count = 0
                prior_nan_count = 0
                renorm_count = 0
                attn_kl_norm_raw = attn_kl_norm
                if kl_atom_detail is not None:
                    prior_nan_count += int(kl_atom_detail.get("prior_nan_count", 0))
                    renorm_count += int(kl_atom_detail.get("renorm_count", 0))
                if kl_res_detail is not None:
                    prior_nan_count += int(kl_res_detail.get("prior_nan_count", 0))
                    renorm_count += int(kl_res_detail.get("renorm_count", 0))
                if not torch.isfinite(attn_kl_norm).all():
                    kl_nan_count = 1
                    attn_kl_norm = drug_repr_c.new_tensor(0.0)
                    attn_kl_raw = drug_repr_c.new_tensor(0.0)
                else:
                    if self.attn_kl_clip is not None and float(self.attn_kl_clip) > 0:
                        if float(attn_kl_norm.detach().item()) > float(self.attn_kl_clip):
                            kl_clip_count = 1
                            attn_kl_norm = drug_repr_c.new_tensor(float(self.attn_kl_clip))
                            if self._kl_warn_count < 5:
                                print("[WARN] attn_kl_norm clipped to attn_kl_clip.")
                                self._kl_warn_count += 1

                alpha_for_ent_atom = (
                    kl_atom_detail.get("alpha") if isinstance(kl_atom_detail, dict) else None
                )
                alpha_for_ent_res = (
                    kl_res_detail.get("alpha") if isinstance(kl_res_detail, dict) else None
                )
                prior_for_ent_atom = (
                    kl_atom_detail.get("prior") if isinstance(kl_atom_detail, dict) else None
                )
                prior_for_ent_res = (
                    kl_res_detail.get("prior") if isinstance(kl_res_detail, dict) else None
                )
                if alpha_for_ent_atom is None:
                    alpha_for_ent_atom = alpha_atom
                if alpha_for_ent_res is None:
                    alpha_for_ent_res = alpha_res
                if prior_for_ent_atom is None:
                    prior_for_ent_atom = p_atom
                if prior_for_ent_res is None:
                    prior_for_ent_res = p_res
                sanity_atom = _kl_entropy_sanity(
                    alpha_for_ent_atom, prior_for_ent_atom, drug_inv,
                    alpha_for_ent_atom, prior_for_ent_atom, drug_inv,
                    drug_ids.numel(),
                    softmax_dim_entropy="group(0)",
                    softmax_dim_kl="group(0)",
                )
                sanity_res = _kl_entropy_sanity(
                    alpha_for_ent_res, prior_for_ent_res, prot_inv,
                    alpha_for_ent_res, prior_for_ent_res, prot_inv,
                    prot_ids.numel(),
                    softmax_dim_entropy="group(0)",
                    softmax_dim_kl="group(0)",
                )
                if (
                    self._sanity_kl_print_count < self._sanity_kl_print_limit
                    or sanity_atom["mismatch"]
                    or sanity_res["mismatch"]
                ):
                    print(
                        f"[SANITY-KL][atom] ent(mean/var) alpha={sanity_atom['entropy_alpha_mean']:.6f}/{sanity_atom['entropy_alpha_var']:.6f} "
                        f"prior={sanity_atom['entropy_prior_mean']:.6f}/{sanity_atom['entropy_prior_var']:.6f}; "
                        f"kl(mean/var) alpha={sanity_atom['kl_alpha_mean']:.6f}/{sanity_atom['kl_alpha_var']:.6f} "
                        f"prior={sanity_atom['kl_prior_mean']:.6f}/{sanity_atom['kl_prior_var']:.6f}; "
                        f"same_softmax_dim={sanity_atom['same_softmax_dim']} same_mask={sanity_atom['same_mask']} "
                        f"same_group={sanity_atom['same_group']} valid_groups(ent/kl)="
                        f"{sanity_atom['entropy_valid_groups']}/{sanity_atom['kl_valid_groups']}"
                    )
                    print(
                        f"[SANITY-KL][res ] ent(mean/var) alpha={sanity_res['entropy_alpha_mean']:.6f}/{sanity_res['entropy_alpha_var']:.6f} "
                        f"prior={sanity_res['entropy_prior_mean']:.6f}/{sanity_res['entropy_prior_var']:.6f}; "
                        f"kl(mean/var) alpha={sanity_res['kl_alpha_mean']:.6f}/{sanity_res['kl_alpha_var']:.6f} "
                        f"prior={sanity_res['kl_prior_mean']:.6f}/{sanity_res['kl_prior_var']:.6f}; "
                        f"same_softmax_dim={sanity_res['same_softmax_dim']} same_mask={sanity_res['same_mask']} "
                        f"same_group={sanity_res['same_group']} valid_groups(ent/kl)="
                        f"{sanity_res['entropy_valid_groups']}/{sanity_res['kl_valid_groups']}"
                    )
                    self._sanity_kl_print_count += 1

                self._last_aux = {
                    "attn_kl_norm_raw": float(attn_kl_norm_raw.item()) if torch.is_tensor(attn_kl_norm_raw) else float(attn_kl_norm_raw),
                    "kl_stats": {
                        "kl_clip_count": int(kl_clip_count),
                        "kl_nan_count": int(kl_nan_count),
                        "prior_nan_count": int(prior_nan_count),
                        "renorm_count": int(renorm_count),
                    },
                    "sanity_kl_atom": sanity_atom,
                    "sanity_kl_res": sanity_res,
                    "delta_rms": float(self._last_delta_rms.item()) if torch.is_tensor(self._last_delta_rms) else 0.0,
                    "delta_rms_weighted": float(self._last_delta_rms_weighted.item()) if torch.is_tensor(self._last_delta_rms_weighted) else 0.0,
                    "inductive_borrow": borrow_stats_refine if "borrow_stats_refine" in locals() else None,
                }

                zero_drug = torch.zeros_like(drug_repr_c)
                zero_prot = torch.zeros_like(prot_repr_c)
                return _pack(
                    edge_logits,
                    (zero_drug, zero_drug, zero_prot, zero_prot, attn_kl_norm, attn_kl_raw, ent_atom, ent_res),
                    pair_repr,
                )

            use_hyper = (
                self.use_hyperedge_head
                and drug_edge_ptr is not None
                and drug_edge_nodes is not None
                and drug_edge_psichic is not None
                and prot_edge_ptr is not None
                and prot_edge_nodes is not None
                and prot_edge_psichic is not None
            )
            if use_hyper:
                atom_base = self._pair_pool_prior(
                    drug_node_repr, drug_edge_ptr, drug_edge_nodes, drug_edge_psichic
                )
                res_base = self._pair_pool_prior(
                    prot_node_repr, prot_edge_ptr, prot_edge_nodes, prot_edge_psichic
                )
                if self.alpha_refine:
                    edge_drug, kl_d_raw, kl_d_norm, ent_d = self._pair_pool_refine(
                        drug_node_repr, drug_edge_ptr, drug_edge_nodes, drug_edge_psichic,
                        res_base, self.atom_refine
                    )
                    edge_prot, kl_p_raw, kl_p_norm, ent_p = self._pair_pool_refine(
                        prot_node_repr, prot_edge_ptr, prot_edge_nodes, prot_edge_psichic,
                        atom_base, self.res_refine
                    )
                    attn_kl_raw = 0.5 * (kl_d_raw + kl_p_raw)
                    attn_kl_norm = 0.5 * (kl_d_norm + kl_p_norm)
                else:
                    edge_drug = atom_base
                    edge_prot = res_base
                    ent_d = drug_node_repr.new_tensor(0.0)
                    ent_p = drug_node_repr.new_tensor(0.0)
                    attn_kl_raw = drug_node_repr.new_tensor(0.0)
                    attn_kl_norm = drug_node_repr.new_tensor(0.0)
                attn_kl_norm_raw = attn_kl_norm
                kl_clip_count = 0
                kl_nan_count = 0
                if not torch.isfinite(attn_kl_norm).all():
                    kl_nan_count = 1
                    attn_kl_norm = drug_node_repr.new_tensor(0.0)
                    attn_kl_raw = drug_node_repr.new_tensor(0.0)
                elif self.attn_kl_clip is not None and float(self.attn_kl_clip) > 0:
                    if float(attn_kl_norm.detach().item()) > float(self.attn_kl_clip):
                        kl_clip_count = 1
                        attn_kl_norm = drug_node_repr.new_tensor(float(self.attn_kl_clip))
                        if self._kl_warn_count < 5:
                            print("[WARN] attn_kl_norm clipped to attn_kl_clip.")
                            self._kl_warn_count += 1
                if use_coldstart_gate and drug_degree is not None and prot_degree is not None:
                    edge_drug_feat = self._pair_pool_prior(
                        drug_feat_node, drug_edge_ptr, drug_edge_nodes, drug_edge_psichic
                    )
                    edge_prot_feat = self._pair_pool_prior(
                        prot_feat_node, prot_edge_ptr, prot_edge_nodes, prot_edge_psichic
                    )
                    if edge_drug_feat is not None and edge_prot_feat is not None:
                        edge_drug_feat = self.drug_feat_mlp(edge_drug_feat)
                        edge_prot_feat = self.prot_feat_mlp(edge_prot_feat)
                        gate_d = self._degree_gate(drug_degree, self.gate_drug, device=drug_feat.device)
                        gate_p = self._degree_gate(prot_degree, self.gate_prot, device=prot_feat.device)
                        if gate_d is not None and gate_p is not None:
                            gate_d_e = gate_d[edge_index[:, 0]]
                            gate_p_e = gate_p[edge_index[:, 1]]
                            edge_drug = gate_d_e.view(-1, 1) * edge_drug + (1.0 - gate_d_e).view(-1, 1) * edge_drug_feat
                            edge_prot = gate_p_e.view(-1, 1) * edge_prot + (1.0 - gate_p_e).view(-1, 1) * edge_prot_feat
                if edge_fused is None:
                    edge_fused = edge_drug.new_zeros((edge_drug.size(0), edge_drug.size(1)))
                edge_prot_raw = self._pair_pool_prior(
                    prot_feat_node, prot_edge_ptr, prot_edge_nodes, prot_edge_psichic
                )
                if edge_prot_raw is None:
                    edge_prot_raw = edge_prot
                pair_repr = torch.cat([edge_drug, edge_prot_raw], dim=1)
                edge_logits = self._edge_logits(edge_drug, edge_prot)
                self._last_aux = {
                    "attn_kl_norm_raw": float(attn_kl_norm_raw.item()) if torch.is_tensor(attn_kl_norm_raw) else float(attn_kl_norm_raw),
                    "kl_stats": {
                        "kl_clip_count": int(kl_clip_count),
                        "kl_nan_count": int(kl_nan_count),
                        "prior_nan_count": 0,
                        "renorm_count": 0,
                    },
                    "delta_rms": float(self._last_delta_rms.item()) if torch.is_tensor(self._last_delta_rms) else 0.0,
                    "delta_rms_weighted": float(self._last_delta_rms_weighted.item()) if torch.is_tensor(self._last_delta_rms_weighted) else 0.0,
                }
                zero = edge_logits.new_tensor(0.0)
                return _pack(
                    edge_logits,
                    (zero, zero, zero, zero, attn_kl_norm, attn_kl_raw, ent_d, ent_p),
                    pair_repr,
                )

            drug_repr_c, drug_ids = self._pool_by_unique_index(
                drug_node_repr, drug_node_to_entity, drug_node_weight, gate_net=self.atom_gate
            )
            prot_repr_c, prot_ids = self._pool_by_unique_index_max(
                prot_node_repr, protein_node_to_entity
            )
            if use_coldstart_gate and drug_degree is not None and prot_degree is not None:
                drug_feat_repr_c, _ = self._pool_by_unique_index(
                    drug_feat_node, drug_node_to_entity, drug_node_weight, gate_net=self.atom_gate
                )
                prot_feat_repr_c, _ = self._pool_by_unique_index_max(
                    prot_feat_node, protein_node_to_entity
                )
                drug_feat_repr_c = self.drug_feat_mlp(drug_feat_repr_c)
                prot_feat_repr_c = self.prot_feat_mlp(prot_feat_repr_c)
                gate_d = self._degree_gate(drug_degree, self.gate_drug, device=drug_feat.device)
                gate_p = self._degree_gate(prot_degree, self.gate_prot, device=prot_feat.device)
                if gate_d is not None and gate_p is not None:
                    if drug_ids is not None:
                        gate_d = gate_d[drug_ids]
                    if prot_ids is not None:
                        gate_p = gate_p[prot_ids]
                    drug_repr_c = gate_d.view(-1, 1) * drug_repr_c + (1.0 - gate_d).view(-1, 1) * drug_feat_repr_c
                    prot_repr_c = gate_p.view(-1, 1) * prot_repr_c + (1.0 - gate_p).view(-1, 1) * prot_feat_repr_c
            pair_repr = None
            prot_raw_repr_c, _ = self._pool_by_unique_index_max(
                prot_feat_node, protein_node_to_entity
            )
            prot_seq_repr_raw_c, _ = self._pool_by_unique_index_max(
                prot_feat, protein_node_to_entity
            )
            prot_repr_c, _ = self._inductive_knn_feature_borrow(
                prot_repr_c,
                prot_seq_repr_raw_c,
                prot_ids,
                prot_degree,
                warm_allow_mask=prot_warm_mask,
                topk=self.inductive_knn_k,
                mix=self.inductive_knn_mix,
            )
            edge_drug_ids = edge_index[:, 0]
            edge_prot_ids = edge_index[:, 1]
            if drug_ids is None:
                drug_pos = edge_drug_ids
                drug_mask = torch.ones(edge_drug_ids.size(0), dtype=torch.bool, device=edge_drug_ids.device)
            else:
                drug_pos = torch.searchsorted(drug_ids, edge_drug_ids)
                drug_mask = (drug_pos < drug_ids.numel()) & (drug_ids[drug_pos] == edge_drug_ids)
            if prot_ids is None:
                prot_pos = edge_prot_ids
                prot_mask = torch.ones(edge_prot_ids.size(0), dtype=torch.bool, device=edge_prot_ids.device)
            else:
                prot_pos = torch.searchsorted(prot_ids, edge_prot_ids)
                prot_mask = (prot_pos < prot_ids.numel()) & (prot_ids[prot_pos] == edge_prot_ids)
            edge_mask = drug_mask & prot_mask
            if edge_mask.any():
                drug_pos = drug_pos[edge_mask]
                prot_pos = prot_pos[edge_mask]
                edge_drug = drug_repr_c[drug_pos]
                edge_prot = prot_repr_c[prot_pos]
                edge_prot_raw = prot_raw_repr_c[prot_pos]
                pair_repr = torch.cat([edge_drug, edge_prot_raw], dim=1)
                edge_logits = self._edge_logits(edge_drug, edge_prot)
            else:
                edge_logits = drug_repr_c.new_zeros((0,))
            zero_drug = torch.zeros_like(drug_repr_c)
            zero_prot = torch.zeros_like(prot_repr_c)
            zero = edge_logits.new_tensor(0.0)
            return _pack(
                edge_logits,
                (zero_drug, zero_drug, zero_prot, zero_prot, zero, zero, None, None),
                pair_repr,
            )

        drug_repr_c, drug_ids = self._pool_by_unique_index(
            drug_node_repr, drug_node_to_entity, drug_node_weight, gate_net=self.atom_gate
        )
        prot_repr_c, prot_ids = self._pool_by_unique_index_max(
            prot_node_repr, protein_node_to_entity
        )

        reconstruction = torch.mm(drug_repr_c, prot_repr_c.t())
        zero_drug = torch.zeros_like(drug_repr_c)
        zero_prot = torch.zeros_like(prot_repr_c)
        zero = drug_repr_c.new_tensor(0.0)
        return _pack(
            reconstruction,
            (zero_drug, zero_drug, zero_prot, zero_prot, zero, zero, None, None),
            None,
        )
