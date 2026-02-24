#hypergraph_utils.py
import numpy as np
import scipy
import torch
def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    欧式距离矩阵计算
    输入：一个形状为 [N, D] 的特征矩阵，N 是节点数，D 是每个节点的特征维度；
    输出：一个 [N, N] 的距离矩阵 dist_mat[i][j] 表示节点 i 和 j 的欧式距离。
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(normal_col=False, *F_list):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    多模态特征拼接
    输入：多个特征矩阵；
    功能：
    支持把高维度特征 reshape 成二维；
    可以按列归一化；
    横向拼接所有特征；
    输出：合并后的特征矩阵 [N, F1+F2+...]
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    拼接超边结构
    处理不同顶点空间的超边拼接
    把多个超图结构 H 进行横向拼接；
    兼容 H 是稀疏矩阵或列表的情况；
    例子：你有基于药物和蛋白质的两个 H，可以拼在一起作为一个联合图。
    """
    H = None
    for h in H_list:
        if h is None or h == []:
            continue
        # 转换为稠密矩阵
        if isinstance(h, list):
            h = [hi.toarray() if isinstance(hi, scipy.sparse.csr_matrix) else hi for hi in h]
        else:
            h = h.toarray() if isinstance(h, scipy.sparse.csr_matrix) else h

        # 首个子超图初始化
        if H is None:
            H = h
        else:
            # 水平拼接超边
            if type(h) != list:
                H = np.hstack((H, h))
            else:
                H = [np.hstack((H[i], h[i])) for i in range(len(H))]
    return H
# def hyperedge_concat(*H_list):
#     """
#     Concatenate hyperedge group in H_list
#     :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
#     :return: Fused hypergraph incidence matrix
#     """
#     H = None
#     for h in H_list:
#         if h is not None and h != []:
#             # for the first H appended to fused hypergraph incidence matrix
#             if H is None:
#                 H = h
#             else:
#                 if type(h) != list:
#                     H = np.hstack((H, h))
#                 else:
#                     tmp = []
#                     for a, b in zip(H, h):
#                         tmp.append(np.hstack((a, b)))
#                     H = tmp
#     return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    从 H 生成邻接矩阵 G
    输入：超图结构 H（可以是单个或多个）；
    输出：图神经网络输入所需的邻接矩阵 G；
    可选 variable_weight=True 时，会根据超边内容设置权重 W。
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    修正版的超图邻接矩阵计算
    Args:
        H: 超图关联矩阵 [n_nodes, n_hyperedges]
        variable_weight: 是否使用可变的超边权重
    Returns:
        G: 超图邻接矩阵 [n_nodes, n_nodes]
    """
    H = np.array(H, dtype=np.float32)
    n_nodes, n_hyperedges = H.shape

    # ------------------ 节点度计算 ------------------
    DV = np.maximum(H.sum(axis=1), 1e-8)  # 形状 (n_nodes,)

    # ------------------ 超边度计算 ------------------
    if variable_weight:
        W = np.maximum(H.mean(axis=0), 1e-8)  # 形状 (n_hyperedges,)
    else:
        W = np.ones(n_hyperedges)  # 形状 (n_hyperedges,)

    # 关键修正：超边度应该是H的列和（即每个超边包含的节点权重和）
    DE = np.maximum(np.sum(H, axis=0), 1e-8)  # 形状 (n_hyperedges,)

    # ------------------ 矩阵归一化 ------------------
    DV2 = np.diag(1.0 / np.sqrt(DV))  # 形状 (n_nodes, n_nodes)
    invDE = np.diag(1.0 / DE)  # 形状 (n_hyperedges, n_hyperedges)

    # ------------------ 邻接矩阵计算 ------------------
    if variable_weight:
        W_mat = np.diag(W)  # 形状 (n_hyperedges, n_hyperedges)
        # G = D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)
        G = DV2 @ H @ W_mat @ invDE @ H.T @ DV2
    else:
        # W为单位矩阵时的简化计算
        G = DV2 @ H @ invDE @ H.T @ DV2

    return torch.FloatTensor(G)


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    通过距离矩阵构造 H
    使用距离矩阵 + KNN，生成稀疏超图结构矩阵 H；
    每个节点构造一个以自己为中心的超边，连接它最近的 k 个节点；
    is_probH=True 表示构造的是“概率型超边”（权重随距离衰减），否则是硬连接（1/0）；
    用指数函数计算连接强度（如果开启 is_probH）。
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    从原始特征构造多尺度超图结构
    综合版本，支持多尺度（多层K邻近）超图结构;
    会先计算距离矩阵，再调用前面的函数构造 H；
    如果设置了多个 K_neigs，并开启 split_diff_scale=True，则会返回多个子超图 H1, H2,...；
    默认会把它们拼成一个大 H。
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H
