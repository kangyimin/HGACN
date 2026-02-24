import torch
from torch.nn.modules.module import Module

"""
    VAE 的 KL 散度模块
    用于计算隐变量分布 q(z|x) ~ N(mean, exp(log_var)) 与标准正态 N(0, I) 的 KL 散度
"""


class KLDivLoss(Module):
    """
    Args:
        norm_factor (int, optional): 归一化因子（一般是节点数，例如药物数/蛋白数）。
        clamp_value (float, optional): 对 log_var 的 clamp 范围，增强数值稳定性，默认 10。
    """

    def __init__(self, norm_factor: int = None, clamp_value: float = 10.0):
        super(KLDivLoss, self).__init__()
        self.norm_factor = float(norm_factor) if norm_factor is not None else None
        self.clamp_value = clamp_value

    def forward(self, log_var: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_var (torch.Tensor): log(sigma^2)，形状 [batch_size, latent_dim]
            mean (torch.Tensor): mu，形状 [batch_size, latent_dim]

        Returns:
            torch.Tensor: KL 散度标量
        """
        if log_var.shape != mean.shape:
            raise ValueError(f"log_var shape {log_var.shape} must match mean shape {mean.shape}")

        # 限制数值范围，避免 exp 爆炸
        clamped_log_var = log_var.clamp(min=-self.clamp_value, max=self.clamp_value)

        # KL = -0.5 * Σ (1 + log_var - mean^2 - exp(log_var))
        kl_div = -0.5 * torch.sum(1 + clamped_log_var - mean.pow(2) - torch.exp(clamped_log_var), dim=-1)

        # 先对 batch 求均值
        kl_loss = torch.mean(kl_div)

        # 再按节点数归一化（如果提供了 norm_factor）
        if self.norm_factor is not None:
            kl_loss = kl_loss / self.norm_factor

        return kl_loss


