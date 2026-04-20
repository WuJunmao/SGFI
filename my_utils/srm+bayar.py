import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 配置区：把参数写在这里
# =========================
INPUT_PATH = r"/home/lab301-3090/wujun/datasets/CocoGlide/fake/glide_inpainting_val2017_49269_up.png"     # 单张图片路径 或 图片目录
OUTDIR = r"./srm_bayar_out"                 # 输出目录
DO_SRM = True                          # 是否导出 SRM 特征
DO_BAYAR = True                        # 是否导出 Bayar 特征
DEVICE = "cuda"                        # "cuda" 或 "cpu"
BAYAR_CKPT = None                      # Bayar 的权重路径（没有就 None；强烈建议有训练好的权重）
SAVE_PER_CHANNEL = False               # 是否把每个通道单独保存为灰度图
ROBUST_Q = 0.01                        # Bayar 可视化用分位数归一化强度（0.01~0.05 常用）


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
        self.register_buffer("filters", filters)
        self.truc = nn.Hardtanh(-2, 2)

    def forward(self, x):
        x = F.conv2d(x, self.filters, padding=2, stride=1)  # 保证同尺寸
        x = self.truc(x)
        return x


class BayarConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = torch.ones(self.in_channels, self.out_channels, 1) * -1.0
        self.kernel = nn.Parameter(
            torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
            requires_grad=True
        )

    def bayarConstraint(self):
        self.kernel.data = self.kernel.data.div(self.kernel.data.sum(-1, keepdim=True))
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat(
            (self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]),
            dim=2
        )
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)  # 保证同尺寸
        return x


def list_images(inp):
    if os.path.isdir(inp):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(inp, e)))
        files.sort()
        return files
    return [inp]


def to_tensor_rgb(pil_img):
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x


def save_rgb_tensor01(x_chw, out_path):
    x = x_chw.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def save_gray_tensor01(x_hw, out_path):
    x = x_hw.detach().cpu().clamp(0, 1)
    arr = (x.numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def robust_norm_01(y_bchw, q=0.01, eps=1e-6):
    b, c, h, w = y_bchw.shape
    y2 = y_bchw.view(b, c, -1)
    lo = torch.quantile(y2, q, dim=2, keepdim=True)
    hi = torch.quantile(y2, 1.0 - q, dim=2, keepdim=True)
    y01 = (y2 - lo) / (hi - lo + eps)
    return y01.view(b, c, h, w).clamp(0, 1)


@torch.no_grad()
def main():
    dev = torch.device("cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    os.makedirs(OUTDIR, exist_ok=True)

    srm = SRMFilter().to(dev).eval()
    bayar = BayarConv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2).to(dev).eval()

    if BAYAR_CKPT is not None:
        sd = torch.load(BAYAR_CKPT, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = bayar.load_state_dict(sd, strict=False)
        print(f"[Bayar] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    img_files = list_images(INPUT_PATH)
    if len(img_files) == 0:
        raise FileNotFoundError(f"No images found in: {INPUT_PATH}")

    for p in img_files:
        base = os.path.splitext(os.path.basename(p))[0]
        img = Image.open(p)
        x = to_tensor_rgb(img).to(dev)

        if DO_SRM:
            y = srm(x)                 # [1,3,H,W], 值域约 [-2,2]
            y01 = (y + 2.0) / 4.0       # 映射到 [0,1]
            out_rgb = os.path.join(OUTDIR, f"{base}_srm_rgb.png")
            save_rgb_tensor01(y01[0], out_rgb)

            if SAVE_PER_CHANNEL:
                for ci in range(y01.shape[1]):
                    out_c = os.path.join(OUTDIR, f"{base}_srm_c{ci}.png")
                    save_gray_tensor01(y01[0, ci], out_c)

        if DO_BAYAR:
            y = bayar(x)               # [1,3,H,W], 值域不固定
            y01 = robust_norm_01(y, q=ROBUST_Q)
            out_rgb = os.path.join(OUTDIR, f"{base}_bayar_rgb.png")
            save_rgb_tensor01(y01[0], out_rgb)

            if SAVE_PER_CHANNEL:
                for ci in range(y01.shape[1]):
                    out_c = os.path.join(OUTDIR, f"{base}_bayar_c{ci}.png")
                    save_gray_tensor01(y01[0, ci], out_c)

        print(f"[OK] {p} -> {OUTDIR}")


if __name__ == "__main__":
    main()