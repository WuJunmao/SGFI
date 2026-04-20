# 文件: SparseViT_Mul.py
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from SparseViT import SparseViT
from decoderhead import Multiple
from my_utils.modal_extract import ModalitiesExtractor


class SparseViT_Mul(nn.Module):
    def __init__(self,
                 depth=[5, 8, 20, 7],
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 img_size=512,
                 s_blocks3=[8, 4, 2, 1],
                 s_blocks4=[2, 1],
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pretrained_path=None,
                 # Noiseprint++ 权重路径（记得改成你自己的）
                 noiseprint_weights_path: str = "/home/lab301-3090/wujun/SparseViT-main/checkpoint/np++.pth",
                 ):
        super(SparseViT_Mul, self).__init__()
        self.img_size = img_size

        # 1) 主干：已经内置 4 个 stage 的模态融合
        self.encoder_net = SparseViT(
            layers=depth,
            embed_dim=embed_dim,
            img_size=img_size,
            s_blocks3=s_blocks3,
            s_blocks4=s_blocks4,
            head_dim=head_dim,
            drop_path_rate=0.2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pretrained_path=pretrained_path,
        )

        # 2) Multiple 解码头（保持不变）
        self.lmu = Multiple(embed_dim=embed_dim[-1])  # 512

        # 3) 损失函数
        self.BCE_loss = nn.BCEWithLogitsLoss()

        # 4) 先初始化主干 + 解码头（不要动噪声模型的权重）
        self.apply(self._init_weights)

        # 5) 手工模态特征提取器（Noiseprint++ / Bayar / SRM）
        #    放在 self.apply(self._init_weights) 之后，避免把 np++ 的预训练权重重新初始化
        self.modal_extractor = ModalitiesExtractor(
            modals=['noiseprint', 'bayar', 'srm'],
            noiseprint_path=noiseprint_weights_path,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image, mask, *args, **kwargs):
        """
        image: (B,3,H,W)
        mask : (B,1,H,W)
        返回: predict_loss, prob
        """
        # ===== 1) 提取 full-res 模态特征 =====
        # out: [noiseprint, bayar, srm]，每个 (B,3,H,W)
        mod_list = self.modal_extractor(image)
        modal_full = torch.cat(mod_list, dim=1)  # (B,9,H,W)

        # ===== 2) 主干 + 4-stage 模态融合 =====
        feats = self.encoder_net(image, modal_full=modal_full)  # dict，共 6 个特征

        feature_list = []
        for _, v in feats.items():
            feature_list.append(v)

        # ===== 3) Multiple 解码头 =====
        logits = self.lmu(feature_list)  # (B,1,H_dec,W_dec)

        # 上采样到原图大小做监督
        logits_up = F.interpolate(
            logits,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        gt = mask.float()
        predict_loss = self.BCE_loss(logits_up, gt)
        prob = torch.sigmoid(logits_up)

        return predict_loss, prob