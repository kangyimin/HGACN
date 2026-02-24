import numpy as np
from scipy import sparse
import os

def inspect_sparse_matrix(name, M, show_rows=5, show_cols=5):
    print(f"\n===== {name} =====")
    print(f"Shape: {M.shape}")
    print(f"Type : {type(M)}")
    print(f"Non-zero count: {M.nnz}")

    # 稀疏度
    total = M.shape[0] * M.shape[1]
    print(f"Sparsity: {1 - M.nnz / total:.6f}")

    # 打印前几行/列
    M_csr = M.tocsr()
    print("\n[前几行的非零元素索引与值]")

    for r in range(min(show_rows, M.shape[0])):
        row_data = M_csr[r]
        nz_cols = row_data.indices
        nz_vals = row_data.data
        print(f"Row {r}: index={nz_cols[:show_cols]}, values={nz_vals[:show_cols]}")


def main():
    dataset_name = "DTI-main-data\split-random-hyper"
    data_root = r"C:\Users\29119\PycharmProjects\PythonProject\HGACN\HGACN\data"
    processed = os.path.join(data_root, dataset_name, "processed")

    print("加载 processed 数据...\n")

    # 加载矩阵
    H = sparse.load_npz(os.path.join(processed, "H.npz"))
    H_T = sparse.load_npz(os.path.join(processed, "H_T.npz"))
    G_drug = sparse.load_npz(os.path.join(processed, "G_drug.npz"))
    G_protein = sparse.load_npz(os.path.join(processed, "G_protein.npz"))

    # 加载特征
    features_drug = np.load(os.path.join(processed, "features_drug.npy"))
    features_protein = np.load(os.path.join(processed, "features_protein.npy"))

    print("===== 数据基本信息 =====")
    print(f"Drug count: {features_drug.shape[0]}")
    print(f"Protein count: {features_protein.shape[0]}")
    print(f"Drug feat dim: {features_drug.shape[1]}")
    print(f"Protein feat dim: {features_protein.shape[1]}")

    # 检查 H（drug × hyperedge）
    inspect_sparse_matrix("H (drug × hyperedge)", H)

    # 检查 H_T（protein × hyperedge）
    inspect_sparse_matrix("H_T (protein × hyperedge)", H_T)

    # 检查 drug 图
    inspect_sparse_matrix("G_drug (drug × drug)", G_drug)

    # 检查 protein 图
    inspect_sparse_matrix("G_protein (protein × protein)", G_protein)

    # 重点：查看前几条超边连接了哪些 drug/protein
    print("\n===== 查看前 5 条超边的连接情况（drug/protein） =====")
    H_csr = H.tocsr()
    H_T_csr = H_T.tocsr()

    for e in range(5):
        drugs = H_csr[:, e].nonzero()[0]
        prots = H_T_csr[:, e].nonzero()[0]
        print(f"超边 {e}:  drugs = {drugs},  proteins = {prots}")


if __name__ == "__main__":
    main()
