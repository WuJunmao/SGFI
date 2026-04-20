import os
import numpy as np
from PIL import Image

import torch

# =========================
# 参数（直接写死在这里）
# =========================
IMG_PATH = r"/home/lab301-3090/wujun/datasets/CocoGlide/fake/glide_inpainting_val2017_49269_up.png"
NP_WEIGHTS_PATH = r"//home/lab301-3090/wujun/SparseViT-main/checkpoint/np++.pth"
OUT_DIR = r"./np_out"
DEVICE = "cuda"  # "cuda" or "cpu"
CENTER_INPUT = False  # True: 输入减0.5（可选尝试，不同实现预处理可能不同）

# 可视化方式：minmax 或 percentile(更稳)
VIS_MODE = "percentile"  # "minmax" or "percentile"
P_LOW, P_HIGH = 1.0, 99.0  # VIS_MODE="percentile" 时使用


# 1) 把这里改成你工程里 ModalitiesExtractor 的真实导入路径
#    例如：from my_utils.modal_extract import ModalitiesExtractor
from my_utils.modal_extract import ModalitiesExtractor


def load_rgb_tensor(img_path: str, center: bool) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]
    if center:
        arr = arr - 0.5
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def to_uint8_vis(feat_hw: np.ndarray) -> np.ndarray:
    f = feat_hw.astype(np.float32)

    if VIS_MODE == "percentile":
        lo = np.percentile(f, P_LOW)
        hi = np.percentile(f, P_HIGH)
    else:
        lo = float(f.min())
        hi = float(f.max())

    if abs(hi - lo) < 1e-12:
        return np.zeros_like(f, dtype=np.uint8)

    vis = (255.0 * (f - lo) / (hi - lo)).clip(0, 255).astype(np.uint8)
    return vis


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")

    extractor = ModalitiesExtractor(modals=("noiseprint",), noiseprint_path=NP_WEIGHTS_PATH).to(device)
    extractor.eval()

    x = load_rgb_tensor(IMG_PATH, CENTER_INPUT).to(device)

    with torch.no_grad():
        outs = extractor(x)
        np_map = outs[0].detach().cpu().float().numpy()  # [1,1,H,W]

    base = os.path.splitext(os.path.basename(IMG_PATH))[0]
    out_npy = os.path.join(OUT_DIR, f"{base}_noiseprint.npy")
    out_png = os.path.join(OUT_DIR, f"{base}_noiseprint.png")

    np.save(out_npy, np_map.astype(np.float32))

    feat_hw = np_map[0, 0]  # [H,W]
    vis = to_uint8_vis(feat_hw)
    Image.fromarray(vis, mode="L").save(out_png)

    print(f"[OK] npy  -> {out_npy}")
    print(f"[OK] png  -> {out_png}")
    print(f"[INFO] feature stats: min={feat_hw.min():.6f}, max={feat_hw.max():.6f}, mean={feat_hw.mean():.6f}, std={feat_hw.std():.6f}")
    print(f"[INFO] shape: {np_map.shape}")


if __name__ == "__main__":
    main()