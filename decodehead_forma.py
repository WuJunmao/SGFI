import os
import torch
from torch import nn
from my_utils.DnCNN_noiseprint import make_net
import torch.nn.functional as F
import torch
import torch.nn as nn


class SparseFormaHeadH16(nn.Module):
    """
    参考尺度：H/16
    输入：来自 SparseViT.forward(x) 的字典
        - H/16: third1, third2, third3, third  (C=320)
        - H/32: last1, last                   (C=512)
    上采样：仅 H/32 -> PixelShuffle(2) 到 H/16
    融合：与 ForMA/SegFormer 基线一致（拼接→1x1融合→预测）
    输出：(B, num_classes, H/16, W/16)
    """
    def __init__(self,
                 num_classes=1,
                 in_ch_16=320,   # H/16 通道
                 in_ch_32=512,   # H/32 通道
                 embedding_dim=96,   # 工作通道
                 dropout_ratio=0.1):
        super().__init__()

        # H/16 分支：MLP → (B, E, H/16, W/16)
        self.lin_t1 = MLP(input_dim=in_ch_16, embed_dim=embedding_dim)
        self.lin_t2 = MLP(input_dim=in_ch_16, embed_dim=embedding_dim)
        self.lin_t3 = MLP(input_dim=in_ch_16, embed_dim=embedding_dim)
        self.lin_t  = MLP(input_dim=in_ch_16, embed_dim=embedding_dim)

        # H/32 分支：MLP 到 E*4，再 PixelShuffle(2) → (B, E, H/16, W/16)
        self.lin_l1 = MLP(input_dim=in_ch_32, embed_dim=embedding_dim * 4)
        self.lin_l  = MLP(input_dim=in_ch_32, embed_dim=embedding_dim * 4)
        self.ps2 = nn.PixelShuffle(2)

        # 模态分支与 ForMA 对齐：3→16
        self.conv_modals = nn.Conv2d(3, 16, 1)

        # 与 ForMA 基线一致：拼接后 1×1 降到 E，再预测
        self.linear_fuse = ConvModule(
            c1=embedding_dim * 6 + 16,   # 6 路主干 + 16 通道模态
            c2=embedding_dim,
            k=1,
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.pred = nn.Conv2d(embedding_dim, num_classes, 1)

    def _mlp_to_feat(self, x, linear):
        # x: (B, C, H, W)
        n, c, h, w = x.shape
        x = linear(x)  # (B, H*W, E) 由 MLP 内部完成 flatten+proj
        x = x.permute(0, 2, 1).reshape(n, -1, h, w)  # (B, E, H, W)
        return x

    def _mlp_ps_to_feat(self, x, linear, ps):
        # x: (B, C, h, w) 这里是 H/32
        n, c, h, w = x.shape
        x = linear(x)  # (B, h*w, E*4)
        x = x.permute(0, 2, 1).reshape(n, -1, h, w)  # (B, E*4, h, w)
        x = ps(x)  # PixelShuffle(2) -> (B, E, 2h, 2w) == H/16
        return x

    def forward(self, feats, x_modals):
        # 仅首次打印 shape，确认分辨率
        if not hasattr(self, "_printed"):
            print("[decode] feats shapes:", {k: list(v.shape) for k, v in feats.items()})
            self._printed = True

        # 参照尺寸：third 的空间分辨率（H/16）
        ref_h, ref_w = feats["third"].shape[-2:]

        def align(x):
            # 统一到 (ref_h, ref_w)
            if x.shape[-2:] != (ref_h, ref_w):
                x = F.interpolate(x, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            return x

        # --- 4 路 H/16 ---
        t1 = align(self._mlp_to_feat(feats["third1"], self.lin_t1))
        t2 = align(self._mlp_to_feat(feats["third2"], self.lin_t2))
        t3 = align(self._mlp_to_feat(feats["third3"], self.lin_t3))
        t = align(self._mlp_to_feat(feats["third"], self.lin_t))

        # --- 2 路 H/32 -> PixelShuffle(2) 到 H/16，再统一一次 ---
        l1 = align(self._mlp_ps_to_feat(feats["last1"], self.lin_l1, self.ps2))
        l = align(self._mlp_ps_to_feat(feats["last"], self.lin_l, self.ps2))

        # --- 模态：对齐到 H/16，再 1x1 ---
        x_modals = align(x_modals)
        x_modals = self.conv_modals(x_modals)

        # ForMA 融合：拼接 -> 1x1 -> Dropout -> 预测
        fused = torch.cat([t1, t2, t3, t, l1, l, x_modals], dim=1)  # (B, 6E+16, H/16, W/16)
        fused = self.linear_fuse(fused)  # (B, E, H/16, W/16)
        fused = self.dropout(fused)
        out = self.pred(fused)  # (B, num_classes, H/16, W/16)
        return out






class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class EarlyConv(nn.Module):

    def __init__(self, depth=3, in_channels=3, out_channels=None):
        super().__init__()
        self.depth = depth
        channels = [in_channels]
        if out_channels is None:
            out_channels = in_channels
        channels.extend([24*2**i for i in range(depth)])
        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], 3, 1, 1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU()
                )
                for i in range(depth)
            ]
        )
        self.final = nn.Conv2d(channels[-1], out_channels, 1, 1, 0)##stride变成4了,pad为1了，为了让输出的图大小变为原来4分之一

    def forward(self, x):
        x = self.convs(x)
        x = self.final(x)
        return x
class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]

        f2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]

        f3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]

        q = torch.tensor([[4.], [12.], [2.]]).unsqueeze(-1).unsqueeze(-1)
        filters = torch.tensor([[f1, f1, f1], [f2, f2, f2], [f3, f3, f3]], dtype=torch.float) / q
        self.register_buffer('filters', filters)
        self.truc = nn.Hardtanh(-2, 2)

    def forward(self, x):
        x = F.conv2d(x, self.filters, padding='same', stride=1)
        x = self.truc(x)
        return x
class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super().__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)

    def bayarConstraint(self):
        self.kernel.data = self.kernel.data.div(self.kernel.data.sum(-1, keepdims=True))
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

class ModalMixer(nn.Module):

    def __init__(self, modals=['noiseprint', 'bayar', 'srm'], in_channels=[3, 3, 3], out_channels=3):
        super().__init__()

        w = len(modals)
        assert len(modals) == len(in_channels)

        c_h = sum(in_channels)

        self.blocks = nn.ModuleList(
            [
                EarlyConv() for _ in range(w)
            ]
        )
        self.dropout = nn.Dropout(0.33)
        self.mixer = EarlyConv(in_channels=c_h, out_channels=out_channels)

    def forward(self, x):
        m = []
        for m_i, blk in enumerate(self.blocks):
            m.append(blk(x[m_i]))

        x = torch.cat(m, dim=1)
        x = self.dropout(x)
        x = self.mixer(x)
        return x

class ModalitiesExtractor(nn.Module):
    def __init__(self,
                 modals: list = ('noiseprint', 'bayar','srm'),
                 noiseprint_path: str = None):
        super().__init__()
        self.mod_extract = []
        print(modals)
        if 'noiseprint' in modals:
            num_levels = 17
            out_channel = 1
            self.noiseprint = make_net(3, kernels=[3, ] * num_levels,
                                  features=[64, ] * (num_levels - 1) + [out_channel],
                                  bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                                  acts=['relu', ] * (num_levels - 1) + ['linear', ],
                                  dilats=[1, ] * num_levels,
                                  bn_momentum=0.1, padding=1)

            if noiseprint_path:
                np_weights = noiseprint_path
                assert os.path.isfile(np_weights)
                dat = torch.load(np_weights, map_location=torch.device('cpu'))
                print(f'Noiseprint++ weights: {np_weights}')
                self.noiseprint.load_state_dict(dat)

            self.noiseprint.eval()
            for param in self.noiseprint.parameters():
                param.requires_grad = False
            self.mod_extract.append(self.noiseprint)
        if 'bayar' in modals:
            self.bayar = BayarConv2d(3, 3, padding=2)
            self.mod_extract.append(self.bayar)
        if 'srm' in modals:
            self.srm = SRMFilter()
            self.mod_extract.append(self.srm)

    def set_train(self):
        if hasattr(self, 'bayar'):
            self.bayar.train()

    def set_val(self):
        if hasattr(self, 'bayar'):
            self.bayar.eval()

    def multi_output(self, x) -> list:
        out = []
        for modal in self.mod_extract:
            y = modal(x)
            if y.size()[-3] == 1:
                # y = tile(y, (3, 1, 1))
                y = y.repeat((1,3, 1, 1))
            out.append(y)

        return out

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=2, in_channels=[96, 192, 384, 768], embedding_dim=96, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels  # b0
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = [128, 256, 640,1024]#b3

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim*64)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim*16)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim*4)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.conv_modals = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4+16,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs,x_modals):
        c1, c2, c3, c4 = inputs
        x_modals=self.conv_modals(x_modals)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        ps4 = nn.PixelShuffle(8)
        ps3 = nn.PixelShuffle(4)
        ps2 = nn.PixelShuffle(2)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = ps4(_c4)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = ps3(_c3)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = ps2(_c2)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1,x_modals], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x