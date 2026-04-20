from SparseViT import SparseViT
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_
from decoderhead import Multiple


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
                 # 新增：辅助分支开关和权重
                 use_aux_stage3: bool = True,
                 aux_stage3_weight: float = 0.2,
                 ):
        super(SparseViT_Mul, self).__init__()
        self.img_size = img_size
        self.use_aux_stage3 = use_aux_stage3
        self.aux_stage3_weight = aux_stage3_weight
        self.embed_dim = embed_dim

        # ====== 编码器 ======
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

        # ====== 解码头（原来的 Multiple） ======
        self.lmu = Multiple(embed_dim=512)

        # ====== 主损失 ======
        self.BCE_loss = nn.BCEWithLogitsLoss()

        # ====== Stage3 辅助 head：用在 outputs["third"] 上 ======
        if self.use_aux_stage3:
            in_ch_stage3 = self.embed_dim[2]  # 对应 third 的通道数（默认 320）
            self.aux_head_s3 = nn.Conv2d(in_ch_stage3, 1, kernel_size=1)

        self.apply(self._init_weights)

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
        返回：
          predict_loss: 主损失 + λ * 辅助损失（如果开启）
          image:        主分支 sigmoid 后概率图（B×1×H×W）
        """
        # encoder_net 返回的是一个 dict：
        # { "third1":..., "third2":..., "third3":..., "third":..., "last1":..., "last":... }
        feats = self.encoder_net(image)

        # ====== 主分支：保持你原来的逻辑 ======
        feature_list = []
        for k, v in feats.items():
            feature_list.append(v)

        logits_main = self.lmu(feature_list)  # B×1×h×w（低分辨率）
        logits_main = F.interpolate(
            logits_main,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        loss_main = self.BCE_loss(logits_main, mask)

        # ====== 辅助分支：在 outputs["third"] 上加一个 1×1 conv 后上采样 ======
        if self.use_aux_stage3:
            if "third" not in feats:
                raise KeyError(
                    "SparseViT 输出字典中没有 key='third'，"
                    "请确认 SparseViT.forward_features 是否与当前代码一致。"
                )
            feat_s3 = feats["third"]  # B×C3×H/16×W/16

            logits_aux = self.aux_head_s3(feat_s3)  # B×1×H/16×W/16
            logits_aux = F.interpolate(
                logits_aux,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )

            loss_aux = self.BCE_loss(logits_aux, mask)

            predict_loss = loss_main + self.aux_stage3_weight * loss_aux
        else:
            predict_loss = loss_main

        # ====== 主分支 sigmoid 概率输出（用于 F1 / 可视化） ======
        image = torch.sigmoid(logits_main)
        return predict_loss, image