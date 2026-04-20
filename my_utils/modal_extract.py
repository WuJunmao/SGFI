"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import torch
import torch.nn as nn
from .DnCNN_noiseprint import make_net
import logging
import os
import torch.nn.functional as F


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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)
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


class ModalitiesExtractor(nn.Module):
    def __init__(self,
                 modals: list = ('noiseprint',),
                 noiseprint_path: str = None):
        super().__init__()
        self.mod_extract = nn.ModuleList()

        if 'noiseprint' in modals:
            num_levels = 17
            out_channel = 1
            self.noiseprint = make_net(
                3,
                kernels=[3] * num_levels,
                features=[64] * (num_levels - 1) + [out_channel],
                bns=[False] + [True] * (num_levels - 2) + [False],
                acts=['relu'] * (num_levels - 1) + ['linear'],
                dilats=[1] * num_levels,
                bn_momentum=0.1,
                padding=1
            )

            if noiseprint_path:
                np_weights = noiseprint_path
                assert os.path.isfile(np_weights), f"Noiseprint++ 权重文件不存在: {np_weights}"
                dat = torch.load(np_weights, map_location=torch.device('cpu'))
                logging.info(f'Noiseprint++ weights: {np_weights}')
                print(f"[Noiseprint] 加载权重文件: {np_weights}")

                if isinstance(dat, dict):
                    if 'state_dict' in dat:
                        dat = dat['state_dict']
                    elif 'model' in dat:
                        dat = dat['model']
                    elif 'net' in dat:
                        dat = dat['net']

                missing, unexpected = self.noiseprint.load_state_dict(dat, strict=False)
                print(f"[Noiseprint] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

            self.noiseprint.eval()
            for p in self.noiseprint.parameters():
                p.requires_grad = False
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

    def forward(self, x) -> list:
        out = []
        for mod in self.mod_extract:
            y = mod(x)
            out.append(y)
        return out