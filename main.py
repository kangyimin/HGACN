# model.py
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from layers import *

class HGACN(nn.Module):
   #这是整个模型的主类，继承自 torch.nn.Module，它在 __init__ 中构建模型组件，
   # 在 forward 中定义数据流向。
    def __init__(self, num_in_node, num_in_edge, num_hidden1, num_out, num_out1=64):
        super(HGACN, self).__init__()

        # Hypergraph Convolutional Layers
        #超图卷积部分：分别对节点（蛋白）和超边（药物）进行两层 HGNN 卷积处理
        self.hgcn_node1 = HGNN2(num_in_node, num_hidden1, num_hidden1, num_in_node, num_in_node)
        self.hgcn_hyperedge1 = HGNN2(num_in_edge, num_hidden1, num_hidden1, num_in_edge, num_in_edge)

        # Hypergraph Attention Layers
        # 注意力层（HGAT）部分：使用多头注意力机制进一步编码节点/超边之间的依赖关系。
        self.hgat_node1 = HGAT(num_in_node, num_hidden1)
        self.hgat_hyperedge1 = HGAT(num_in_edge,num_hidden1)

        # Other components
        # 编码器部分（变分自编码器 VAE 使用）：对原始的结构连接矩阵 H、H_T 进行编码。
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, 0.3)
        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, 0.3)

        #输出映射与融合：对编码器输出的嵌入进行线性变换，使其与 GCN/HGAT 结果融合维度一致。
        self.map_node_output = nn.Linear(num_hidden1, num_hidden1)
        self.map_hyperedge_output = nn.Linear(num_hidden1, num_hidden1)

        # Use HGNN_conv from layers.py instead of nn.Conv2d
        #
        self.conv_after_hgat = HGNN_conv1(num_hidden1, num_hidden1)


        self.decoder2 = decoder2(act=lambda x: x)
        #解码器：用于输出药物-蛋白交互预测矩阵（重构关系）。
        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        #变分采样模块（重参数化技巧）：用于构造隐变量 z ~ N(μ, σ²)，即 VAE 的关键部分。

    def sample_latent(self, z_node, z_hyperedge):#采样函数
        # Return the latent normal sample z ~ N(mu, sigma^2)
        #该函数实现了 VAE 中的采样技巧（Reparameterization Trick）：
        #对节点和超边分别计算：
        #均值 μ = encoder(x)
        #标准差 σ = exp(logσ)
        #噪声采样 ε ~ N(0, 1)
        #得到 z = μ + σ * ε
        #训练时返回采样值；测试时返回均值作为稳定输出。
        self.z_node_mean = self._enc_mu_node(z_node)  # mu
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)  # sigma
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).float()
        self.z_node_std_ = z_node_std_
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(self.z_node_std_, requires_grad=True))

        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)  # sigma
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).float()
        self.z_edge_std_ = z_edge_std_
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(self.z_edge_std_, requires_grad=True))

        if self.training:
            return self.z_node_, self.z_hyperedge_  # Reparameterization trick
        else:
            return self.z_node_mean, self.z_edge_mean

    def forward(self, G1, G2, drug_vec, protein_vec, H, H_T):#前向传播
        #输入说明：
        #G1：药物之间的超图邻接矩阵；
        #G2：蛋白之间的超图邻接矩阵；
        #drug_vec：药物初始特征；
        #protein_vec：蛋白初始特征；
        #H, H_T：药物-蛋白连接矩阵及其转置。

        # Branch 1: Directly to Convolutional Layer 纯 HGNN 分支
        # 对药物、蛋白分别使用 HGNN 得到 GCN-style 表征。
        drug_feature_gcn = self.hgcn_hyperedge1(drug_vec, G1)
        protein_feature_gcn = self.hgcn_node1(protein_vec, G2)

        # Branch 2: HGAT + Convolution HGAT + 卷积融合
        #使用注意力层学习邻居重要性，再加一个简单的线性层做融合（HGNN_conv1），与 GCN 分支的输出相加。
        drug_feature_hgat = self.hgat_hyperedge1(drug_vec, G1)
        protein_feature_hgat = self.hgat_node1(protein_vec, G2)
        drug_feature_conv = self.conv_after_hgat(drug_feature_hgat)
        protein_feature_conv = self.conv_after_hgat(protein_feature_hgat)
        drug_feature_combined = drug_feature_gcn + drug_feature_conv
        protein_feature_combined = protein_feature_gcn + protein_feature_conv

        # Key Embedding
        z_protein_encoder = self.node_encoders1(H)  # 代表目标蛋白编码
        z_drug_encoder = self.hyperedge_encoders1(H_T)  # 代表药物编码
        z_protein_encoder_mapped = self.map_node_output(z_protein_encoder)
        z_drug_encoder_mapped = self.map_hyperedge_output(z_drug_encoder)
        #映射 & 融合表示：融合结构表示和注意力/卷积特征，作为最终隐变量输入。
        z_node_s=z_protein_encoder_mapped + protein_feature_combined
        z_hyperedge_s=z_drug_encoder_mapped + drug_feature_combined

        #重参数采样：输出两个隐变量：节点 z_node 和超边 z_hyperedge 的 VAE 表示。
        self.z_protein_s, self.z_drug_s = self.sample_latent(z_node_s,z_hyperedge_s)
        # Fusion Layer
        z_node = self.z_protein_s
        z_hyperedge =  self.z_drug_s

        # Reconstruction Layer 解码器的预测结果（用于最终任务，如 DTI 预测）
        reconstruction = self.decoder2(z_node, z_hyperedge)
        # print("reconstruction :", reconstruction.shape)
        # Recover 基于均值的重构结果（可以用于评估重构损失如 KL divergence）
        recover = self.z_node_mean.mm(self.z_edge_mean.t())

        return reconstruction, recover
