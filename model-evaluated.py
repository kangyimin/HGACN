import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from model import HGACN
from data_preprocess import load_and_construct_hypergraphs

# 设置随机种子
torch.manual_seed(11)
np.random.seed(11)

# 加载药物-靶标相互作用矩阵
dataset_dir = 'deepDTnet'
drug_protein_matrix = np.genfromtxt(os.path.join(dataset_dir, 'drugProtein.txt'), dtype=np.int32)
drug_num, protein_num = drug_protein_matrix.shape

hidden_units = 512

# 修改后的模型初始化，参数与HGACN类定义保持一致
model = HGACN(
    num_drugs=732,  # 药物数量
    num_prots=1915,  # 蛋白质数量
    feat_dim=200,  # 特征维度（应与训练时一致）
    hidden_dim=hidden_units,
    out_dim=hidden_units
)

# 加载模型权重
try:
    model.load_state_dict(torch.load('final_model_cold.pth'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

model.eval()

if torch.cuda.is_available():
    model.cuda()
    print("Using CUDA acceleration")

# 评估多个测试集
auc_list, aupr_list = [], []
test_indices = range(5, 10)  # 测试集5-9

for i in test_indices:
    test_file = f"DTnet_test_0.1_{i}"
    print(f"\nProcessing test set: {test_file}")

    # 加载数据（无需训练集）
    try:
        _, _, drug_feat, prot_feat, H, H_T, edge_test, _, G_drug, G_protein = load_and_construct_hypergraphs(
            dataset_train=None,
            dataset_test=test_file
        )
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        continue

    # 确保索引在有效范围内
    test_pairs = [(d, p) for (d, p) in edge_test if d < drug_num and p < protein_num]
    if not test_pairs:
        print(f"No valid test pairs in {test_file}")
        continue

    true_labels = [drug_protein_matrix[d][p] for (d, p) in test_pairs]

    # 转移到GPU
    if torch.cuda.is_available():
        drug_feat = drug_feat.cuda()
        prot_feat = prot_feat.cuda()
        H = H.cuda()
        H_T = H_T.cuda()
        G_drug = G_drug.cuda()
        G_protein = G_protein.cuda()

    # 预测
    with torch.no_grad():
        try:
            reconstruction, _ = model(G_drug, G_protein)
            pred_scores = torch.sigmoid(reconstruction).cpu().numpy()
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            continue

    # 提取预测分数
    pred_labels = [pred_scores[drug_idx][protein_idx] for (drug_idx, protein_idx) in test_pairs]

    # 计算指标
    try:
        auc = roc_auc_score(true_labels, pred_labels)
        aupr = average_precision_score(true_labels, pred_labels)
        auc_list.append(auc)
        aupr_list.append(aupr)
        print(f"Test Set {i}: AUC = {auc:.4f}, AUPR = {aupr:.4f}")
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        continue

# 输出最终结果
if auc_list and aupr_list:
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
    aupr_mean, aupr_std = np.mean(aupr_list), np.std(aupr_list)

    print("\nFinal Evaluation Results:")
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"AUPR: {aupr_mean:.4f} ± {aupr_std:.4f}")
else:
    print("\nNo valid evaluation results were obtained")