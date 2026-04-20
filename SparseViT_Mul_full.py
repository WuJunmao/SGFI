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
                 noiseprint_weights_path: str = "/home/lab301-3090/wujun/SparseViT-main/checkpoint/np++.pth",
                 use_aux_stage3: bool = True,
                 aux_stage3_weight: float = 0.2,
                 ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_aux_stage3 = use_aux_stage3
        self.aux_stage3_weight = aux_stage3_weight

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

        self.lmu = Multiple(embed_dim=embed_dim[-1])
        self.BCE_loss = nn.BCEWithLogitsLoss()

        if self.use_aux_stage3:
            self.aux_head_s3 = nn.Conv2d(embed_dim[2], 1, kernel_size=1)

        self.apply(self._init_weights)

        self.modal_extractor = ModalitiesExtractor(
            modals=['noiseprint'],
            noiseprint_path=noiseprint_weights_path,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            pass

    def forward(self, image, mask, *args, **kwargs):
        mod_list = self.modal_extractor(image)
        modal_full = mod_list[0]  # (B,1,H,W)

        feats = self.encoder_net(image, modal_full=modal_full)

        ordered_keys = ["third1", "third2", "third3", "third", "last1", "last"]
        feature_list = [feats[k] for k in ordered_keys if k in feats]

        logits_main = self.lmu(feature_list)
        logits_main = F.interpolate(
            logits_main,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        gt = mask.float()
        loss_main = self.BCE_loss(logits_main, gt)

        if self.use_aux_stage3:
            if "third" not in feats:
                raise KeyError("SparseViT outputs missing key 'third'")

            feat_s3 = feats["third"]
            logits_aux = self.aux_head_s3(feat_s3)
            logits_aux = F.interpolate(
                logits_aux,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )

            loss_aux = self.BCE_loss(logits_aux, gt)
            predict_loss = loss_main + self.aux_stage3_weight * loss_aux
        else:
            predict_loss = loss_main

        prob_main = torch.sigmoid(logits_main)
        return predict_loss, prob_main