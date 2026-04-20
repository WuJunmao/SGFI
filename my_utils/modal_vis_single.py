# -*- coding: utf-8 -*-
"""
单张图片 → 提取 Noiseprint / Bayar / SRM 三个模态并保存可视化结果

运行方式：
    python my_utils/modal_vis_single.py
"""

import os
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from my_utils.modal_extract import ModalitiesExtractor


def _minmax_norm(arr: np.ndarray):
    """把数组归一化到 [0,1]，方便转成 0~255 图像"""
    arr = arr.astype(np.float32)
    vmin = arr.min()
    vmax = arr.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - vmin) / (vmax - vmin)


def main():
    # ===== 自己改这里 =====
    IMG_PATH = r"/home/lab301-3090/wujun/datasets/columbia-tp+true/4cam_splc/0jpg/canong3_canonxt_sub_11.jpg"
    OUT_DIR = r"/home/lab301-3090/wujun/SparseViT-main/modal_vis_debug"
    # 如果没有 Noiseprint++ 权重，设成 "" 或 None
    NOISEPRINT_CKPT = r"/home/lab301-3090/wujun/SparseViT-main/checkpoint/np++.pth"
    # =====================

    os.makedirs(OUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(IMG_PATH))[0]

    # 1. 读原图并保存
    img = Image.open(IMG_PATH).convert("RGB")
    img.save(os.path.join(OUT_DIR, f"{base_name}_original.png"))

    # 2. 转 tensor: [1,3,H,W]
    to_tensor = ToTensor()
    x = to_tensor(img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    # 3. 构建模态提取器
    ckpt = NOISEPRINT_CKPT if (NOISEPRINT_CKPT and os.path.isfile(NOISEPRINT_CKPT)) else None
    modal_ext = ModalitiesExtractor(
        modals=['noiseprint', 'bayar', 'srm'],
        noiseprint_path=ckpt
    )
    modal_ext.to(device)
    modal_ext.eval()

    # 4. 前向提取三种模态
    with torch.no_grad():
        out_list = modal_ext(x)   # [noiseprint, bayar, srm]

    # ================= Noiseprint++ =================
    noise = out_list[0]            # [1,3,H',W']，第 0 个通道为主
    noise_ch0 = noise[0, 0].cpu().numpy()  # [H',W']

    # 和原作者一样做裁剪 + 下采样（可选，不想裁剪就用 noise_ch0）
    noise_crop = noise_ch0[16:-16:4, 16:-16:4]
    noise_norm = _minmax_norm(noise_crop)
    noise_img = (noise_norm * 255.0).astype(np.uint8)
    Image.fromarray(noise_img, mode="L").save(
        os.path.join(OUT_DIR, f"{base_name}_noiseprint.png")
    )

    # ================= Bayar =================
    bayar = out_list[1]             # [1,3,H,W]
    bayar_np = bayar[0].cpu().numpy()          # [3,H,W]
    bayar_np = np.transpose(bayar_np, (1, 2, 0))  # [H,W,3]
    bayar_norm = _minmax_norm(bayar_np)
    bayar_img = (bayar_norm * 255.0).astype(np.uint8)
    Image.fromarray(bayar_img).save(
        os.path.join(OUT_DIR, f"{base_name}_bayar.png")
    )

    # ================= SRM =================
    srm = out_list[2]               # [1,3,H,W]
    srm_np = srm[0].cpu().numpy()             # [3,H,W]
    srm_np = np.transpose(srm_np, (1, 2, 0))      # [H,W,3]
    srm_norm = _minmax_norm(srm_np)
    srm_img = (srm_norm * 255.0).astype(np.uint8)
    Image.fromarray(srm_img).save(
        os.path.join(OUT_DIR, f"{base_name}_srm.png")
    )

    print("DONE:", OUT_DIR)


if __name__ == "__main__":
    main()