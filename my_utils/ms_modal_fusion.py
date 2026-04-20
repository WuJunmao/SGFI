# 文件: my_utils/ms_modal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class StageModalFusion(nn.Module):
    """
    在某一层特征上做模态融合 + 门控（更稳版本）：
      feat       : (B, C_feat, H_feat, W_feat)
      modal_full : (B, C_modal, H_img,  W_img) —— 原图分辨率模态特征（如 Noiseprint++）

    流程：
      1) modal_full 下采样到 (H_feat, W_feat)
      2) 对 modal 做 per-sample 标准化（按空间维度）
      3) 1x1 conv 投影到 C_feat => m
      4) 对 feat/m 做 GroupNorm 后 concat 生成 gate
      5) out = feat + gate * m
    """

    def __init__(
        self,
        feat_channels: int,
        modal_channels: int,
        reduction: int = 4,
        normalize_modal: bool = True,
        modal_drop: float = 0.10,   # 建议 0.05~0.2
        eps: float = 1e-6,
        clamp_std: float = 5.0,
        gn_groups: int = 32,
    ):
        super().__init__()
        self.normalize_modal = normalize_modal
        self.modal_drop = modal_drop
        self.eps = eps
        self.clamp_std = clamp_std

        # 1x1：modal -> C_feat
        self.modal_proj = nn.Conv2d(modal_channels, feat_channels, kernel_size=1, bias=False)

        # 更稳：用 GroupNorm 做门控输入归一化（比 BN 对 batch size 更鲁棒）
        g = min(gn_groups, feat_channels)
        while feat_channels % g != 0 and g > 1:
            g -= 1
        self.feat_gn = nn.GroupNorm(g, feat_channels)
        self.modal_gn = nn.GroupNorm(g, feat_channels)

        mid = max(feat_channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(feat_channels * 2, mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(mid, feat_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.modal_dropout2d = nn.Dropout2d(p=modal_drop) if modal_drop and modal_drop > 0 else nn.Identity()

        self._init_stable()

    def _init_stable(self):
        # ✅ 关键：让“刚开始≈不注入模态”，训练自己学
        # 这样就算你没在 SparseViT 外面加 alpha，也不容易掉点
        nn.init.zeros_(self.modal_proj.weight)

        # gate 最后一层置零 => 输出初期≈sigmoid(0)=0.5，但因为 m 初期≈0，所以 out≈feat
        last = self.gate[-2]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def _norm_modal(self, m: torch.Tensor) -> torch.Tensor:
        # m: (B, C, H, W)，按空间维度做标准化（每张图每通道）
        mean = m.mean(dim=(2, 3), keepdim=True)
        std = m.std(dim=(2, 3), keepdim=True)
        m = (m - mean) / (std + self.eps)
        if self.clamp_std is not None:
            m = m.clamp(-self.clamp_std, self.clamp_std)
        return m

    def forward(self, feat: torch.Tensor, modal_full: torch.Tensor) -> torch.Tensor:
        """
        feat:       (B, C_feat, H_feat, W_feat)
        modal_full: (B, C_modal, H_img,  W_img)
        """
        B, C, H, W = feat.shape

        # 1) 下采样模态到当前 stage 尺度
        m = F.interpolate(modal_full, size=(H, W), mode="bilinear", align_corners=False)

        # 2) 模态标准化（强烈建议对 np++ 这类先验做）
        if self.normalize_modal:
            m = self._norm_modal(m)

        # 3) 投影到与 feat 相同通道数
        m = self.modal_proj(m)  # (B, C_feat, H, W)

        # 4) 模态 dropout（防过拟合先验）
        m = self.modal_dropout2d(m)

        # 5) gate（对输入做 GN，更稳）
        feat_n = self.feat_gn(feat)
        m_n = self.modal_gn(m)
        g_in = torch.cat([feat_n, m_n], dim=1)
        g = self.gate(g_in)  # (B, C_feat, H, W)

        # 6) 融合
        out = feat + g * m
        return out