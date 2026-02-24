import numpy as np
import torch


def minmax_norm(x, eps=1e-12):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x_min = float(x.min().item()) if x.numel() else 0.0
        x_max = float(x.max().item()) if x.numel() else 1.0
        denom = max(x_max - x_min, eps)
        return (x - x_min) / denom
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    x_min = float(x.min())
    x_max = float(x.max())
    denom = max(x_max - x_min, eps)
    return (x - x_min) / denom


def combine_attention(alpha_prior, alpha_learned, gate_w=None):
    if gate_w is None:
        return alpha_prior
    wA = float(gate_w[0])
    wB = float(gate_w[1])
    wC = float(gate_w[2]) if len(gate_w) > 2 else 0.0
    return wA * alpha_prior + (wB + wC) * alpha_learned


def grad_score(node_repr, node_grad, mode="grad_x_input"):
    if node_repr is None or node_grad is None:
        return None
    if mode == "grad":
        score = node_grad.abs().sum(dim=1)
    else:
        score = (node_repr * node_grad).abs().sum(dim=1)
    return score


def select_edge_index(edge_index, drug_id, prot_id):
    if edge_index is None:
        return None
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    if edge_index.size == 0:
        return None
    mask = (edge_index[:, 0] == drug_id) & (edge_index[:, 1] == prot_id)
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size else None
