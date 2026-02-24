"""Loss functions for HGACN: base BCE objective with KL, attention-KL, contrastive, and auxiliary regularization terms."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    """
    KL divergence for Gaussian latent variables:
    q(z|x) ~ N(mean, exp(log_var)) vs N(0, I)
    """

    def __init__(self, norm_factor: int = None, clamp_value: float = 10.0):
        super().__init__()
        self.norm_factor = float(norm_factor) if norm_factor is not None else None
        self.clamp_value = clamp_value

    def forward(self, log_var: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between Gaussian posterior and unit Gaussian prior."""
        if log_var.shape != mean.shape:
            raise ValueError(f"log_var shape {log_var.shape} must match mean shape {mean.shape}")

        clamped_log_var = log_var.clamp(min=-self.clamp_value, max=self.clamp_value)
        kl_div = -0.5 * torch.sum(
            1 + clamped_log_var - mean.pow(2) - torch.exp(clamped_log_var), dim=-1
        )
        kl_loss = torch.mean(kl_div)
        if self.norm_factor is not None:
            kl_loss = kl_loss / self.norm_factor
        return kl_loss


class CombinedLoss(nn.Module):
    """
    BCEWithLogitsLoss + KL(drug) + KL(protein) + Attn-KL(learned || psichic)
    """

    def __init__(
        self,
        kl_weight=0.1,
        attn_kl_weight=0.02,
        attn_kl_max_ratio=0.2,
        kl_hard_assert=False,
        prior_conf_ref=0.2,
        kl_stage1_epochs=5,
        attn_kl_w_min=0.005,
        attn_kl_w_max=0.05,
        attn_kl_ramp_epochs=10,
        attn_kl_schedule="sigmoid",
        info_nce_weight=0.0,
        info_nce_temp=0.05,
        info_nce_neg_k=64,
        info_nce_max_ratio=0.05,
        gate_balance_weight=0.0,
        gate_entropy_weight=0.0,
        delta_reg_weight=0.0,
        distill_weight=0.0,
        distill_T=1.0,
        distill_mode="warm_only",
        kd_max_ratio=0.2,
        num_drugs=None,
        num_proteins=None,
        pos_weight=None,
        debug_assertions=False,
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        self.kl_weight = kl_weight
        self.attn_kl_weight = attn_kl_weight
        self.attn_kl_max_ratio = attn_kl_max_ratio
        self.kl_hard_assert = bool(kl_hard_assert)
        self.prior_conf_ref = float(prior_conf_ref) if prior_conf_ref is not None else 0.0
        self.kl_stage1_epochs = max(int(kl_stage1_epochs or 0), 0)
        self.attn_kl_w_min = float(attn_kl_w_min)
        self.attn_kl_w_max = float(attn_kl_w_max)
        self.attn_kl_ramp_epochs = max(int(attn_kl_ramp_epochs or 0), 0)
        self.attn_kl_schedule = str(attn_kl_schedule or "sigmoid").strip().lower()
        self._current_epoch = None
        self.info_nce_weight = info_nce_weight
        self.info_nce_temp = info_nce_temp
        self.info_nce_neg_k = info_nce_neg_k
        self.info_nce_max_ratio = info_nce_max_ratio
        self.gate_balance_weight = gate_balance_weight
        self.gate_entropy_weight = gate_entropy_weight
        self.delta_reg_weight = float(delta_reg_weight)
        self.distill_weight = distill_weight
        self.distill_T = distill_T
        self.distill_mode = distill_mode
        self.kd_max_ratio = kd_max_ratio
        self.num_drugs = num_drugs
        self.num_proteins = num_proteins
        self.debug_assertions = bool(debug_assertions)
        self._bce = None
        self._kl_total = None
        self._attn_kl_raw = None
        self._attn_kl_norm = None
        self._attn_kl_scaled = None
        self._kl_scale_ratio = None
        self._kl_weight_eff = None
        self._prior_conf_mean = None
        self._kl_weight_base = None
        self._kl_clip_count = 0
        self._kl_nan_count = 0
        self._prior_nan_count = 0
        self._renorm_count = 0
        self._kl_ratio = None
        self._reg_total = None
        self._info_nce = None
        self._info_nce_ratio = None
        self._distill_raw = None
        self._distill = None
        self._kd_ratio = None
        self._gate_balance = None
        self._gate_entropy = None
        self._delta_reg = None
        self._delta_weight_eff = None
        self._breakdown = None
        self._total = None

    def set_epoch(self, epoch):
        """Store current epoch for dynamic loss scheduling."""
        self._current_epoch = int(epoch) if epoch is not None else None

    def attn_kl_weight_schedule(self, epoch):
        """Return epoch-dependent attention-KL weight."""
        if epoch is None:
            return float(self.attn_kl_weight)
        stage1 = max(int(self.kl_stage1_epochs), 0)
        w_min = float(self.attn_kl_w_min)
        w_max = float(self.attn_kl_w_max)
        if epoch < stage1:
            return w_min
        ramp = max(int(self.attn_kl_ramp_epochs), 1)
        t = (float(epoch - stage1 + 1) / float(ramp))
        t = max(0.0, min(1.0, t))
        sched = self.attn_kl_schedule
        if sched == "cosine":
            alpha = 0.5 * (1.0 - math.cos(math.pi * t))
        elif sched == "linear":
            alpha = t
        else:
            # sigmoid
            alpha = 1.0 / (1.0 + math.exp(-12.0 * (t - 0.5)))
        return w_min + (w_max - w_min) * float(alpha)

    def _is_attn_kl_enabled(self):
        """Hard gate for attention-KL terms."""
        return float(self.attn_kl_weight) > 0.0

    def _set_attn_kl_disabled(self, ref_tensor):
        """Populate zero-valued KL diagnostics when attention-KL is disabled."""
        zero = ref_tensor.new_tensor(0.0)
        self._attn_kl_norm = zero
        self._attn_kl_raw = zero
        self._attn_kl_scaled = zero
        self._kl_scale_ratio = zero
        self._kl_weight_base = zero
        self._kl_weight_eff = zero
        self._kl_ratio = zero
        self._prior_conf_mean = zero
        self._kl_clip_count = 0
        self._kl_nan_count = 0
        self._prior_nan_count = 0
        self._renorm_count = 0

    def forward(
        self,
        preds,
        labels,
        mean_d,
        log_var_d,
        mean_p,
        log_var_p,
        sample_weight=None,
        attn_kl=None,
        attn_kl_raw=None,
        edge_mask=None,
        pair_repr=None,
        pair_group=None,
        gate_weights=None,
        distill_logits=None,
        distill_mask=None,
        prior_conf=None,
        prior_conf_edge=None,
        delta_reg=None,
        kl_stats=None,
    ):
        """Compute total training loss and cache detailed components."""
        if not (torch.isfinite(preds).all() and torch.isfinite(labels).all()):
            raise ValueError("preds or labels contain NaN/Inf")
        if preds.shape != labels.shape:
            raise ValueError(f"preds shape {preds.shape} must match labels shape {labels.shape}")
        kl_enabled = self._is_attn_kl_enabled()

        if edge_mask is not None:
            edge_mask = edge_mask.to(device=preds.device).view(-1)
            preds = preds.view(-1)[edge_mask]
            labels = labels.view(-1)[edge_mask]
            if sample_weight is not None:
                sample_weight = sample_weight.view(-1)[edge_mask]
            if pair_repr is not None:
                pair_repr = pair_repr[edge_mask]
            if pair_group is not None:
                pair_group = pair_group.view(-1)[edge_mask]
            if preds.numel() == 0:
                self._bce = preds.new_tensor(0.0)
                self._kl_total = preds.new_tensor(0.0)
                if kl_enabled:
                    self._attn_kl_norm = preds.new_tensor(0.0) if attn_kl is None else attn_kl
                    self._attn_kl_raw = preds.new_tensor(0.0) if attn_kl_raw is None else attn_kl_raw
                    self._kl_scale_ratio = preds.new_tensor(1.0)
                    self._kl_clip_count = int(kl_stats.get("kl_clip_count", 0)) if isinstance(kl_stats, dict) else 0
                    self._kl_nan_count = int(kl_stats.get("kl_nan_count", 0)) if isinstance(kl_stats, dict) else 0
                    self._prior_nan_count = int(kl_stats.get("prior_nan_count", 0)) if isinstance(kl_stats, dict) else 0
                    self._renorm_count = int(kl_stats.get("renorm_count", 0)) if isinstance(kl_stats, dict) else 0
                    if not torch.is_tensor(self._attn_kl_norm):
                        self._attn_kl_norm = preds.new_tensor(float(self._attn_kl_norm))
                    if not torch.is_tensor(self._attn_kl_raw):
                        self._attn_kl_raw = preds.new_tensor(float(self._attn_kl_raw))
                    base_w = self.attn_kl_weight_schedule(self._current_epoch)
                    self._kl_weight_base = preds.new_tensor(float(base_w))
                    self._prior_conf_mean = None
                    if prior_conf is not None:
                        if isinstance(prior_conf, dict):
                            vals = [v for v in prior_conf.values() if v is not None]
                            if vals:
                                prior_conf = vals
                            else:
                                prior_conf = None
                        if prior_conf is not None:
                            if isinstance(prior_conf, (list, tuple)):
                                vals = []
                                for v in prior_conf:
                                    if v is None:
                                        continue
                                    if torch.is_tensor(v):
                                        vals.append(v.detach().view(-1))
                                    else:
                                        vals.append(preds.new_tensor(float(v)).view(-1))
                                if vals:
                                    conf = torch.cat(vals, dim=0)
                                else:
                                    conf = None
                            elif torch.is_tensor(prior_conf):
                                conf = prior_conf.detach().view(-1)
                            else:
                                conf = preds.new_tensor(float(prior_conf)).view(-1)
                            if conf is not None and conf.numel() > 0:
                                nan_mask = ~torch.isfinite(conf)
                                if nan_mask.any():
                                    self._prior_nan_count += int(nan_mask.sum().item())
                                    conf = conf[~nan_mask]
                                if conf.numel() > 0:
                                    conf = conf.clamp(0.0, 1.0)
                                    self._prior_conf_mean = conf.mean()
                    if self._prior_conf_mean is not None:
                        self._kl_weight_eff = self._kl_weight_base * self._prior_conf_mean
                    else:
                        self._kl_weight_eff = self._kl_weight_base
                    self._attn_kl_scaled = self._kl_weight_eff * self._attn_kl_norm
                else:
                    self._set_attn_kl_disabled(preds)
                self._reg_total = self.kl_weight * self._kl_total + self._attn_kl_scaled
                self._kl_ratio = preds.new_tensor(0.0) if kl_enabled else self._kl_ratio
                self._info_nce = preds.new_tensor(0.0)
                self._distill = preds.new_tensor(0.0)
                self._gate_balance = preds.new_tensor(0.0)
                self._gate_entropy = preds.new_tensor(0.0)
                self._delta_reg = preds.new_tensor(0.0)
                return self._bce + self._reg_total

        if sample_weight is None:
            self._bce = self.bce_loss(preds, labels)
        else:
            preds = preds.view(-1)
            labels = labels.to(device=preds.device, dtype=preds.dtype).view(-1)
            weight = sample_weight.to(device=preds.device, dtype=preds.dtype).view(-1)
            if weight.numel() != preds.numel():
                raise ValueError("sample_weight must align with preds")
            if not torch.isfinite(weight).all():
                raise ValueError("sample_weight contains NaN/Inf")
            loss_vec = F.binary_cross_entropy_with_logits(
                preds, labels, pos_weight=self.bce_loss.pos_weight, reduction="none"
            )
            weight_sum = weight.sum().clamp_min(1e-6)
            self._bce = (loss_vec * weight).sum() / weight_sum

        self._kl_total = preds.new_tensor(0.0)
        if kl_enabled:
            self._attn_kl_norm = preds.new_tensor(0.0) if attn_kl is None else attn_kl
            self._attn_kl_raw = preds.new_tensor(0.0) if attn_kl_raw is None else attn_kl_raw
            self._kl_scale_ratio = preds.new_tensor(1.0)
            self._kl_clip_count = int(kl_stats.get("kl_clip_count", 0)) if isinstance(kl_stats, dict) else 0
            self._kl_nan_count = int(kl_stats.get("kl_nan_count", 0)) if isinstance(kl_stats, dict) else 0
            self._prior_nan_count = int(kl_stats.get("prior_nan_count", 0)) if isinstance(kl_stats, dict) else 0
            self._renorm_count = int(kl_stats.get("renorm_count", 0)) if isinstance(kl_stats, dict) else 0
            if not torch.is_tensor(self._attn_kl_norm):
                self._attn_kl_norm = preds.new_tensor(float(self._attn_kl_norm))
            if not torch.is_tensor(self._attn_kl_raw):
                self._attn_kl_raw = preds.new_tensor(float(self._attn_kl_raw))
            self._prior_conf_mean = None
            if not torch.isfinite(self._attn_kl_norm).all():
                print("[WARN] kl_nan set_to_zero")
                self._attn_kl_norm = preds.new_tensor(0.0)
                self._attn_kl_raw = preds.new_tensor(0.0)
                self._kl_scale_ratio = preds.new_tensor(0.0)
                self._kl_nan_count += 1
            base_w = self.attn_kl_weight_schedule(self._current_epoch)
            self._kl_weight_base = preds.new_tensor(float(base_w))
            if prior_conf is not None:
                if isinstance(prior_conf, dict):
                    vals = [v for v in prior_conf.values() if v is not None]
                    if vals:
                        prior_conf = vals
                    else:
                        prior_conf = None
                if prior_conf is not None:
                    if isinstance(prior_conf, (list, tuple)):
                        vals = []
                        for v in prior_conf:
                            if v is None:
                                continue
                            if torch.is_tensor(v):
                                vals.append(v.detach().view(-1))
                            else:
                                vals.append(preds.new_tensor(float(v)).view(-1))
                        if vals:
                            conf = torch.cat(vals, dim=0)
                        else:
                            conf = None
                    elif torch.is_tensor(prior_conf):
                        conf = prior_conf.detach().view(-1)
                    else:
                        conf = preds.new_tensor(float(prior_conf)).view(-1)
                    if conf is not None and conf.numel() > 0:
                        nan_mask = ~torch.isfinite(conf)
                        if nan_mask.any():
                            self._prior_nan_count += int(nan_mask.sum().item())
                            conf = conf[~nan_mask]
                        if conf.numel() > 0:
                            conf = conf.clamp(0.0, 1.0)
                            self._prior_conf_mean = conf.mean()
            if self._prior_conf_mean is not None:
                self._kl_weight_eff = self._kl_weight_base * self._prior_conf_mean
            else:
                self._kl_weight_eff = self._kl_weight_base
            self._attn_kl_scaled = self._kl_weight_eff * self._attn_kl_norm
            if self.attn_kl_max_ratio is not None and self.attn_kl_max_ratio > 0:
                limit = self._bce.detach() * float(self.attn_kl_max_ratio)
                if torch.isfinite(self._attn_kl_scaled).all() and torch.isfinite(limit).all():
                    denom = self._attn_kl_scaled.detach().clamp_min(1e-12)
                    self._kl_scale_ratio = (limit / denom).clamp(0.0, 1.0)
                    self._attn_kl_scaled = self._attn_kl_scaled * self._kl_scale_ratio
            self._kl_ratio = self._attn_kl_scaled / (self._bce + 1e-12)
            self._reg_total = self.kl_weight * self._kl_total + self._attn_kl_scaled
        else:
            self._set_attn_kl_disabled(preds)
            self._reg_total = self.kl_weight * self._kl_total + self._attn_kl_scaled
        self._info_nce = preds.new_tensor(0.0)
        if self.info_nce_weight and pair_repr is not None and pair_group is not None:
            self._info_nce = self._info_nce_loss(pair_repr, pair_group)
            if self.info_nce_max_ratio is not None and self.info_nce_max_ratio > 0:
                limit = self._bce.detach() * float(self.info_nce_max_ratio) / max(float(self.info_nce_weight), 1e-12)
                self._info_nce = torch.minimum(self._info_nce, limit)
            self._info_nce_ratio = (self.info_nce_weight * self._info_nce) / (self._bce + 1e-12)
        self._distill = preds.new_tensor(0.0)
        self._kd_ratio = preds.new_tensor(0.0)
        self._kd_conf_mean = preds.new_tensor(0.0)
        if self.distill_weight and distill_logits is not None:
            t = max(float(self.distill_T), 1e-3)
            # Two-class KL with log_softmax / softmax (binary case).
            zeros = torch.zeros_like(preds)
            student_logits2 = torch.stack([zeros, preds], dim=1) / t
            teacher_logits2 = torch.stack([zeros, distill_logits], dim=1) / t
            log_p_s = F.log_softmax(student_logits2, dim=1)
            p_t = F.softmax(teacher_logits2, dim=1).detach()
            kd_vec = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1)
            ent = -(p_t * torch.log(p_t.clamp_min(1e-12))).sum(dim=1)
            conf = 1.0 - ent / float(math.log(2.0))
            if prior_conf_edge is not None:
                if torch.is_tensor(prior_conf_edge):
                    pconf = prior_conf_edge.to(device=conf.device, dtype=conf.dtype).view(-1)
                else:
                    pconf = conf.new_tensor(float(prior_conf_edge)).view(-1)
                if pconf.numel() == conf.numel():
                    pconf = pconf.clamp(0.0, 1.0)
                    conf = conf * pconf
            kd_vec = kd_vec * conf
            if distill_mask is not None:
                mask = distill_mask.view(-1).to(device=kd_vec.device, dtype=kd_vec.dtype)
                kd_vec = kd_vec * mask
                conf = conf * mask
                denom = mask.sum().clamp_min(1.0)
                kd = kd_vec.sum() / denom
                self._kd_conf_mean = conf.sum() / denom
            else:
                kd = kd_vec.mean()
                self._kd_conf_mean = conf.mean()
            kd = kd * (t * t)
            self._distill_raw = kd
            if self.kd_max_ratio is not None and self.kd_max_ratio > 0 and self.distill_weight > 0:
                limit = self._bce.detach() * float(self.kd_max_ratio) / max(float(self.distill_weight), 1e-12)
                kd = torch.minimum(kd, limit)
            self._distill = kd
            self._kd_ratio = (self.distill_weight * self._distill) / (self._bce + 1e-12)
        self._gate_balance = preds.new_tensor(0.0)
        self._gate_entropy = preds.new_tensor(0.0)
        self._delta_weight_eff = preds.new_tensor(float(self.delta_reg_weight))
        if delta_reg is None:
            self._delta_reg = preds.new_tensor(0.0)
        else:
            if torch.is_tensor(delta_reg):
                self._delta_reg = delta_reg.to(device=preds.device, dtype=preds.dtype)
            else:
                self._delta_reg = preds.new_tensor(float(delta_reg))
        if kl_enabled and self.attn_kl_w_max and self.attn_kl_w_max > 0 and self._kl_weight_base is not None:
            scale = float(self._kl_weight_base.detach().item()) / float(self.attn_kl_w_max)
            scale = max(0.0, min(1.0, scale))
            self._delta_weight_eff = preds.new_tensor(float(self.delta_reg_weight) * scale)
        if gate_weights is not None and gate_weights.numel() > 0:
            gate = gate_weights
            mean_gate = gate.mean(dim=0)
            target = gate.new_full(mean_gate.shape, 1.0 / mean_gate.numel())
            self._gate_balance = ((mean_gate - target) ** 2).sum()
            ent = -(gate * torch.log(gate.clamp_min(1e-12))).sum(dim=1).mean()
            self._gate_entropy = -ent
        total = (
            self._bce
            + self._reg_total
            + self.info_nce_weight * self._info_nce
            + self.distill_weight * self._distill
            + self.gate_balance_weight * self._gate_balance
            + self.gate_entropy_weight * self._gate_entropy
            + self._delta_weight_eff * self._delta_reg
        )
        self._total = total
        self._breakdown = {
            "bce_raw": self._bce.detach(),
            "bce_weighted": self._bce.detach(),
            "kl_raw": self._attn_kl_norm.detach(),
            "kl_weighted": self._attn_kl_scaled.detach(),
            "kl_weight_eff": self._kl_weight_eff.detach() if self._kl_weight_eff is not None else self._bce.new_tensor(0.0),
            "kl_weight_base": self._kl_weight_base.detach() if self._kl_weight_base is not None else self._bce.new_tensor(0.0),
            "kl_scale_ratio": self._kl_scale_ratio.detach() if self._kl_scale_ratio is not None else self._bce.new_tensor(1.0),
            "kl_ratio": self._kl_ratio.detach() if self._kl_ratio is not None else self._bce.new_tensor(0.0),
            "attn_kl_raw": self._attn_kl_raw.detach() if self._attn_kl_raw is not None else self._bce.new_tensor(0.0),
            "kl_clip_count": self._bce.new_tensor(float(self._kl_clip_count)),
            "kl_nan_count": self._bce.new_tensor(float(self._kl_nan_count)),
            "prior_nan_count": self._bce.new_tensor(float(self._prior_nan_count)),
            "renorm_count": self._bce.new_tensor(float(self._renorm_count)),
            "info_nce_raw": self._info_nce.detach(),
            "info_nce_weighted": (self.info_nce_weight * self._info_nce).detach(),
            "distill_raw": self._distill_raw.detach() if self._distill_raw is not None else self._bce.new_tensor(0.0),
            "distill_weighted": (self.distill_weight * self._distill).detach(),
            "kd_ratio": self._kd_ratio.detach() if self._kd_ratio is not None else self._bce.new_tensor(0.0),
            "kd_conf": self._kd_conf_mean.detach() if self._kd_conf_mean is not None else self._bce.new_tensor(0.0),
            "gate_balance_raw": self._gate_balance.detach(),
            "gate_balance_weighted": (self.gate_balance_weight * self._gate_balance).detach(),
            "gate_entropy_raw": self._gate_entropy.detach(),
            "gate_entropy_weighted": (self.gate_entropy_weight * self._gate_entropy).detach(),
            "delta_reg_raw": self._delta_reg.detach(),
            "delta_reg_weighted": (self._delta_weight_eff * self._delta_reg).detach(),
            "total": total.detach(),
        }
        if self.debug_assertions:
            sum_weighted = (
                self._breakdown["bce_weighted"]
                + self._breakdown["kl_weighted"]
                + self._breakdown["info_nce_weighted"]
                + self._breakdown["distill_weighted"]
                + self._breakdown["gate_balance_weighted"]
                + self._breakdown["gate_entropy_weighted"]
                + self._breakdown["delta_reg_weighted"]
            )
            if not torch.allclose(total.detach(), sum_weighted, atol=1e-6):
                raise AssertionError("[LOSS] total_loss mismatch with weighted sum.")
        return total

    def _info_nce_loss(self, pair_repr, pair_group):
        """
        Asymmetric cross-modal InfoNCE:
        query = drug_graph embedding, key = prot_raw embedding.
        pair_repr is expected as concat([query, key], dim=1).
        """
        if pair_repr is None:
            return self._bce.new_tensor(0.0) if self._bce is not None else torch.tensor(0.0)
        if pair_repr.numel() == 0:
            return pair_repr.new_tensor(0.0)
        if pair_repr.dim() != 2 or pair_repr.size(1) < 2:
            return pair_repr.new_tensor(0.0)
        feat_dim = int(pair_repr.size(1))
        if feat_dim % 2 != 0:
            raise ValueError(f"pair_repr dim must be even for cross-modal InfoNCE, got {feat_dim}")
        half = feat_dim // 2
        q = pair_repr[:, :half]
        k = pair_repr[:, half:]
        n = int(q.size(0))
        if n < 2:
            return pair_repr.new_tensor(0.0)
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        temp = max(float(self.info_nce_temp), 1e-6)
        logits = torch.matmul(q, k.t()) / temp
        labels = torch.arange(n, device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    def get_components(self):
        """Return scalar loss components for logging."""
        if self._bce is None or self._attn_kl_scaled is None:
            raise ValueError("Call forward() before get_components()")
        info_nce = self._info_nce if self._info_nce is not None else self._bce.new_tensor(0.0)
        info_nce_ratio = self._info_nce_ratio if self._info_nce_ratio is not None else self._bce.new_tensor(0.0)
        distill = self._distill if self._distill is not None else self._bce.new_tensor(0.0)
        distill_raw = self._distill_raw if self._distill_raw is not None else self._bce.new_tensor(0.0)
        kd_ratio = self._kd_ratio if self._kd_ratio is not None else self._bce.new_tensor(0.0)
        kd_conf = self._kd_conf_mean if self._kd_conf_mean is not None else self._bce.new_tensor(0.0)
        gate_balance = self._gate_balance if self._gate_balance is not None else self._bce.new_tensor(0.0)
        gate_entropy = self._gate_entropy if self._gate_entropy is not None else self._bce.new_tensor(0.0)
        return (
            self._bce,
            self._attn_kl_scaled,
            self._attn_kl_raw,
            self._attn_kl_norm,
            self._kl_ratio,
            self._kl_weight_eff if self._kl_weight_eff is not None else self._bce.new_tensor(0.0),
            self._prior_conf_mean if self._prior_conf_mean is not None else self._bce.new_tensor(0.0),
            info_nce,
            info_nce_ratio,
            distill,
            distill_raw,
            kd_ratio,
            kd_conf,
            gate_balance,
            gate_entropy,
            self._total if self._total is not None else self._bce.new_tensor(0.0),
        )

    def get_breakdown(self):
        """Return detailed diagnostic breakdown collected during forward."""
        if self._breakdown is None:
            raise ValueError("Call forward() before get_breakdown().")
        return self._breakdown
