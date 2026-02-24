import os
import argparse
import numpy as np
import torch

from data_preprocess import load_and_construct_hypergraphs
from model import HGACN
from loss import CombinedLoss


def describe_array(name, array, max_values=5):
    if isinstance(array, torch.Tensor):
        array_info = {
            "shape": tuple(array.shape),
            "dtype": str(array.dtype),
            "device": str(array.device),
            "nan": torch.isnan(array).any().item(),
        }
        sample = array.reshape(-1)[:max_values].detach().cpu().numpy()
    else:
        array_info = {
            "shape": tuple(array.shape),
            "dtype": str(array.dtype),
        }
        sample = array.reshape(-1)[:max_values]
    print(f"[{name}] -> {array_info}, sample={sample}")


def main():
    parser = argparse.ArgumentParser(description="调试管线：输出各模块输入/输出摘要")
    parser.add_argument("--dataset", type=str, default="drugbank", help="数据集名称，对应 data/<dataset>")
    parser.add_argument("--data_root", type=str, default=None, help="数据根目录，默认使用当前工作目录的 data 子目录")
    parser.add_argument("--samples", type=int, default=4, help="示例边数量")
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(os.getcwd(), "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading dataset '{args.dataset}' from {data_root}")

    (train_edges, val_edges, test_edges,
     num_drugs, num_prots,
     H, H_T, G_drug, G_protein,
     features_drug, features_protein) = load_and_construct_hypergraphs(args.dataset, data_root)

    print("\n=== 数据预处理阶段 ===")
    describe_array("features_drug", features_drug)
    describe_array("features_protein", features_protein)
    describe_array("G_drug", G_drug)
    describe_array("G_protein", G_protein)
    print(f"train_edges shape={train_edges.shape}, val_edges shape={val_edges.shape}, test_edges shape={test_edges.shape}")

    features_drug_tensor = torch.tensor(features_drug, dtype=torch.float32, device=device)
    features_protein_tensor = torch.tensor(features_protein, dtype=torch.float32, device=device)
    G_drug_tensor = torch.tensor(G_drug, dtype=torch.float32, device=device)
    G_protein_tensor = torch.tensor(G_protein, dtype=torch.float32, device=device)

    describe_array("features_drug_tensor", features_drug_tensor)
    describe_array("features_protein_tensor", features_protein_tensor)
    describe_array("G_drug_tensor", G_drug_tensor)
    describe_array("G_protein_tensor", G_protein_tensor)

    sample_edges = train_edges[:args.samples]
    if len(sample_edges) == 0:
        raise RuntimeError("训练集中无可用边，无法调试")
    sample_pairs = torch.tensor(sample_edges[:, :2], dtype=torch.long, device=device)
    sample_labels = torch.tensor(sample_edges[:, 2], dtype=torch.float32, device=device)

    print("\n=== 模型阶段 ===")
    model = HGACN(
        drug_feat_dim=features_drug_tensor.shape[1],
        prot_feat_dim=features_protein_tensor.shape[1],
        hidden_dim=256,
        out_dim=128
    ).to(device)
    model.eval()

    with torch.no_grad():
        reconstruction, vae_params = model(
            features_drug_tensor, features_protein_tensor, G_drug_tensor, G_protein_tensor
        )
    describe_array("reconstruction_logits", reconstruction)
    for idx, tensor in enumerate(vae_params):
        describe_array(f"vae_param[{idx}]", tensor)

    edge_logits = reconstruction[sample_pairs[:, 0], sample_pairs[:, 1]]
    describe_array("sample_edge_logits", edge_logits)
    describe_array("sample_edge_labels", sample_labels)

    print("\n=== 损失模块阶段 ===")
    num_pos = sample_labels.sum().item()
    num_neg = len(sample_labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], device=device) if num_pos > 0 else None

    criterion = CombinedLoss(
        kl_weight=0.0,
        num_drugs=num_drugs,
        num_proteins=num_prots,
        pos_weight=pos_weight
    ).to(device)

    if len(vae_params) == 8:
        drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl_norm, attn_kl_raw, _, _ = vae_params
    elif len(vae_params) == 7:
        drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl_norm, _, _ = vae_params
        attn_kl_raw = attn_kl_norm
    elif len(vae_params) == 5:
        drug_mu, drug_logvar, prot_mu, prot_logvar, attn_kl_norm = vae_params
        attn_kl_raw = attn_kl_norm
    elif len(vae_params) == 4:
        drug_mu, drug_logvar, prot_mu, prot_logvar = vae_params
        attn_kl_norm = edge_logits.new_tensor(0.0)
        attn_kl_raw = attn_kl_norm
    else:
        raise ValueError("Unexpected vae_params length in debug pipeline")

    loss = criterion(
        edge_logits, sample_labels, drug_mu, drug_logvar, prot_mu, prot_logvar,
        attn_kl=attn_kl_norm, attn_kl_raw=attn_kl_raw
    )
    bce, kl_term, kl_raw, kl_norm, kl_ratio = criterion.get_components()
    print(
        f"Loss total={loss.item():.6f}, "
        f"BCE={float(bce):.6f}, KL_term={float(kl_term):.6f}, "
        f"KL_raw={float(kl_raw):.6f}, KL_norm={float(kl_norm):.6f}, "
        f"KL_ratio={float(kl_ratio):.3f}"
    )


if __name__ == "__main__":
    main()
