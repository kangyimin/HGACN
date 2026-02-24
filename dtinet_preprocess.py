import os
import random
import numpy as np
import torch
from hypergraph_utils import _generate_G_from_H, generate_G_from_H  # noqa: F401

random.seed(3)
torch.manual_seed(3)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "DTInet")


def pre_processed_DTInet(dataset_dir: str = DATASET_DIR):
    """Generate the positive interaction list from the drug-protein matrix."""
    i_m = np.genfromtxt(os.path.join(dataset_dir, "mat_drug_protein_remove_homo.txt"), dtype=np.int32)
    edge = []
    for i in range(len(i_m)):
        for j in range(len(i_m[0])):
            if i_m[i][j] == 1:
                edge.append([i, j])
    print("positive edges:", len(edge))

    out_path = os.path.join(dataset_dir, "drug_target_interaction_remove_homo.txt")
    with open(out_path, "w") as f0:
        for pair in edge:
            s = str(pair).replace("[", " ").replace("]", " ")
            s = s.replace("'", " ").replace(",", "") + "\n"
            f0.write(s)
    print("written:", out_path)


def load_data_DTInet(
    dataset_train: str = "DTInet_train_0.1_0",
    dataset_test: str = "DTInet_test_0.1_0",
    dataset_dir: str = DATASET_DIR,
):
    edge_train = np.genfromtxt(os.path.join(dataset_dir, f"{dataset_train}.txt"), dtype=np.int32)
    edge_all = np.genfromtxt(os.path.join(dataset_dir, "DTInet_all.txt"), dtype=np.int32)
    edge_test = np.genfromtxt(os.path.join(dataset_dir, f"{dataset_test}.txt"), dtype=np.int32)

    i_m = np.genfromtxt(os.path.join(dataset_dir, "mat_drug_protein.txt"), dtype=np.int32)

    H_T = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)
    H_T_all = np.zeros((len(i_m), len(i_m[0])), dtype=np.int32)

    for i in edge_train:
        H_T[i[0]][i[1]] = 1

    for i in edge_all:
        H_T_all[i[0]][i[1]] = 1

    test = np.zeros(len(edge_test))
    for i in range(len(test)):
        if i <= len(edge_test) // 2:
            test[i] = 1

    H_T = torch.Tensor(H_T)
    H = H_T.t()
    H_T_all = torch.Tensor(H_T_all)
    H_all = H_T_all.t()

    drug_feat = torch.eye(708)
    protein_feat = torch.eye(1512)

    drugDisease = torch.Tensor(np.genfromtxt(os.path.join(dataset_dir, "mat_drug_disease.txt"), dtype=np.int32))
    proteinDisease = torch.Tensor(np.genfromtxt(os.path.join(dataset_dir, "mat_protein_disease.txt"), dtype=np.int32))

    print("DTInet H shape:", H.size())  # 1512, 708

    return drugDisease, proteinDisease, drug_feat, protein_feat, H, H_T, edge_test, test


def generate_data_2(dataset_str: str = "drug_target_interaction_remove_homo", dataset_dir: str = DATASET_DIR):
    """Create 10 train/test splits with 10% held out and negative sampling."""
    edge = np.genfromtxt(os.path.join(dataset_dir, f"{dataset_str}.txt"), dtype=np.int32)
    print("loaded edges:", edge.shape)

    data = torch.utils.data.DataLoader(edge, shuffle=True)
    edge_shuffled = []
    for i in data:
        edge_shuffled.append(i[0].tolist())

    test_ration = [0.1]
    for d in test_ration:
        for a in range(10):
            edge_test = edge_shuffled[a * int(len(edge_shuffled) * d) : (a + 1) * int(len(edge_shuffled) * d)]
            edge_train = edge_shuffled[: a * int(len(edge_shuffled) * d)] + edge_shuffled[
                (a + 1) * int(len(edge_shuffled) * d) :
            ]

            test_zeros = []
            while len(test_zeros) < len(edge_test):
                x1 = random.sample(range(0, 708), 1)[0]
                y1 = random.sample(range(0, 1512), 1)[0]
                if [x1, y1] not in edge_shuffled and [x1, y1] not in test_zeros:
                    test_zeros.append([x1, y1])

            edge_test = edge_test + test_zeros

            train_path = os.path.join(
                dataset_dir, f"DTInet_train_{d}_{a}_remove_homo.txt",
            )
            test_path = os.path.join(
                dataset_dir, f"DTInet_test_{d}_{a}_remove_homo.txt",
            )

            with open(train_path, "w") as f0:
                for item in edge_train:
                    s = str(item).replace("[", " ").replace("]", " ")
                    s = s.replace("'", " ").replace(",", "") + "\n"
                    f0.write(s)

            with open(test_path, "w") as f1:
                for item in edge_test:
                    s = str(item).replace("[", " ").replace("]", " ")
                    s = s.replace("'", " ").replace(",", "") + "\n"
                    f1.write(s)

            print(f"saved split d={d}, fold={a} -> {train_path}, {test_path}")


if __name__ == "__main__":
    pre_processed_DTInet()
    generate_data_2()
