import os
import json
from typing import Dict, Any, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import my_utils.datasets
import my_utils.sparsevit_transforms
import SparseViT_Mul


# -----------------------------
# basic utils
# -----------------------------
def _align_predict_to_mask(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
    if pred.shape[-2:] != mask.shape[-2:]:
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    return pred


def _binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    #兼容 0/255 和 0/1
    return (mask > 0.5).float()


def _to_prob(pred: torch.Tensor) -> torch.Tensor:
    # 若像 logits（出现<0 或 >1），就 sigmoid；否则当作概率
    mn = float(pred.min().detach().cpu())
    mx = float(pred.max().detach().cpu())
    if mn < 0.0 or mx > 1.0:
        pred = torch.sigmoid(pred)
    return pred.clamp_(0.0, 1.0)


# -----------------------------
# per-image confusion -> metrics
# -----------------------------
def _confusion_per_image(pred_prob: torch.Tensor, mask_bin: torch.Tensor, thr: float) -> Tuple[torch.Tensor, ...]:
    # pred_prob/mask_bin: [B,1,H,W] or [B,H,W]
    if pred_prob.dim() == 3:
        pred_prob = pred_prob.unsqueeze(1)
    if mask_bin.dim() == 3:
        mask_bin = mask_bin.unsqueeze(1)

    pred_bin = (pred_prob > thr).float()
    tp = torch.sum(pred_bin * mask_bin, dim=(1, 2, 3))
    tn = torch.sum((1 - pred_bin) * (1 - mask_bin), dim=(1, 2, 3))
    fp = torch.sum(pred_bin * (1 - mask_bin), dim=(1, 2, 3))
    fn = torch.sum((1 - pred_bin) * mask_bin, dim=(1, 2, 3))
    return tp, tn, fp, fn


def _f1_from_conf(tp, fp, fn, eps=1e-8) -> torch.Tensor:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def _iou_from_conf(tp, fp, fn, eps=1e-8) -> torch.Tensor:
    return tp / (tp + fp + fn + eps)


def _acc_from_conf(tp, tn, fp, fn, eps=1e-8) -> torch.Tensor:
    return (tp + tn) / (tp + tn + fp + fn + eps)


# -----------------------------
# per-image AUC (trapz) like IMDLBenCo PixelAUC
# -----------------------------
def _auc_trapz_1d(y_true_1d: torch.Tensor, y_score_1d: torch.Tensor, safe: bool = True) -> float:
    # sort by score desc
    idx = torch.argsort(y_score_1d, descending=True)
    y = y_true_1d[idx]

    n_pos = torch.sum(y).item()
    n_neg = y.numel() - n_pos

    # IMDLBenCo PixelAUC 原本只处理 n_pos==0；这里建议也处理 n_neg==0，避免 NaN
    if safe and (n_pos <= 0 or n_neg <= 0):
        return 0.0

    tps = torch.cumsum(y, dim=0)
    fps = torch.cumsum(1 - y, dim=0)

    tpr = tps / n_pos
    fpr = fps / n_neg

    auc = torch.trapz(tpr, fpr).item()
    if safe and (np.isnan(auc) or np.isinf(auc)):
        return 0.0
    return float(auc)


def _pixel_auc_per_image(mask_1hw: torch.Tensor, pred_1hw: torch.Tensor, safe: bool = True) -> float:
    # mask/pred: [H,W] or [1,H,W]
    m = mask_1hw.reshape(-1).float()
    p = pred_1hw.reshape(-1).float()

    # PixelAUC 原实现：若全 0，直接返回 0.0
    if torch.sum(m) == 0:
        return 0.0

    return _auc_trapz_1d(m, p, safe=safe)


# -----------------------------
# dataloader / ckpt listing
# -----------------------------
def build_dataloader(json_path: str, transform, batch_size=2, num_workers=4) -> torch.utils.data.DataLoader:
    dataset_test = my_utils.datasets.json_dataset(path=json_path, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    print(f"  samples={len(dataset_test)} ({type(dataset_test).__name__})")
    return loader


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    exts = (".pth", ".pt", ".ckpt", ".pth.tar")
    files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(exts)]

    def _epoch_key(path: str) -> int:
        b = os.path.basename(path)
        n, _ = os.path.splitext(b)
        for token in ["checkpoint-", "checkpoint_", "ckpt-", "ckpt_", "epoch-", "epoch_"]:
            if token in n:
                try:
                    return int(n.split(token)[-1])
                except Exception:
                    pass
        return 10**9

    files.sort(key=_epoch_key)
    return files


def build_model(pretrained_path: str) -> torch.nn.Module:
    return SparseViT_Mul.SparseViT_Mul(pretrained_path=pretrained_path)


def load_ckpt(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    msg = model.load_state_dict(sd, strict=False)
    return msg


# -----------------------------
# eval one dataset: per-image then mean
# -----------------------------
@torch.no_grad()
def eval_one_dataset_pixel_metrics(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thr: float = 0.5,
    safe_auc: bool = True,
) -> Dict[str, float]:
    model.eval()

    f1_list: List[float] = []
    iou_list: List[float] = []
    acc_list: List[float] = []
    auc_list: List[float] = []

    img_pos = 0
    img_neg = 0

    for images, masks in data_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        masks = _binarize_mask(masks)

        # model output: (_, pred)
        _, pred = model(images, masks)

        pred = _align_predict_to_mask(pred, masks)
        pred = _to_prob(pred)

        # image pos/neg by mask
        if masks.dim() == 3:
            masks_b1hw = masks.unsqueeze(1)
        else:
            masks_b1hw = masks
        per_img_has_pos = (torch.sum(masks_b1hw, dim=(1, 2, 3)) > 0)
        img_pos += int(per_img_has_pos.sum().item())
        img_neg += int((~per_img_has_pos).sum().item())

        tp, tn, fp, fn = _confusion_per_image(pred, masks, thr=thr)
        f1 = _f1_from_conf(tp, fp, fn)
        iou = _iou_from_conf(tp, fp, fn)
        acc = _acc_from_conf(tp, tn, fp, fn)

        f1_list.extend(f1.detach().cpu().tolist())
        iou_list.extend(iou.detach().cpu().tolist())
        acc_list.extend(acc.detach().cpu().tolist())

        # AUC per image
        b = pred.shape[0]
        for i in range(b):
            # pred/mask maybe [1,H,W] or [H,W]
            pm = pred[i, 0]
            mk = masks_b1hw[i, 0]
            auc_list.append(_pixel_auc_per_image(mk, pm, safe=safe_auc))

    out = {
        "img_pos": float(img_pos),
        "img_neg": float(img_neg),
        "pixel_f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "pixel_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "pixel_acc": float(np.mean(acc_list)) if acc_list else 0.0,
        "pixel_auc": float(np.mean(auc_list)) if auc_list else 0.0,
    }
    return out


# -----------------------------
# main
# -----------------------------
def main():
    CHECKPOINT_DIR = "/home/lab301-3090/wujun/SparseViT-main/output_dir"
    PRETRAINED_PATH = "checkpoint/train/pretrain/uniformer_base_ls_in1k.pth"
    TEST_DATASETS_JSON = "/home/lab301-3090/wujun/benco/test_datasets.json"

    DATASET_ORDER = ["Columbia", "coverage", "NIST16", "cocoglide", "realistic_tampering"]

    TEST_BATCH_SIZE = 10
    NUM_WORKERS = 4
    THR = 0.5
    SAFE_AUC = True  # True: 避免全正/全负导致 NaN；False: 更接近原始实现但可能 NaN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    with open(TEST_DATASETS_JSON, "r", encoding="utf-8") as f:
        ds_map: Dict[str, str] = json.load(f)

    for k in DATASET_ORDER:
        if k not in ds_map:
            raise KeyError(f"test_datasets.json 缺少 key: {k}, keys={list(ds_map.keys())}")

    transform = my_utils.sparsevit_transforms.get_albu_transforms("test")

    print("\n== build dataloaders ==")
    loaders: Dict[str, Any] = {}
    for name in DATASET_ORDER:
        print(f"[{name}] {ds_map[name]}")
        loaders[name] = build_dataloader(ds_map[name], transform, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS)

    ckpts = list_checkpoints(CHECKPOINT_DIR)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in: {CHECKPOINT_DIR}")
    print(f"\nfound {len(ckpts)} checkpoints in {CHECKPOINT_DIR}")

    # results[ckpt][ds] = metrics
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for i, ckpt_path in enumerate(ckpts):
        ckpt_name = os.path.basename(ckpt_path)
        print("\n" + "=" * 100)
        print(f"[{i+1}/{len(ckpts)}] ckpt = {ckpt_name}")
        print("=" * 100)

        model = build_model(pretrained_path=PRETRAINED_PATH)
        msg = load_ckpt(model, ckpt_path)
        print(f"load_state_dict(strict=False): missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

        model.to(device)
        model.eval()

        results[ckpt_name] = {}

        # per-dataset eval
        for ds_name in DATASET_ORDER:
            print(f"\n-- dataset: {ds_name}")
            m = eval_one_dataset_pixel_metrics(
                model=model,
                data_loader=loaders[ds_name],
                device=device,
                thr=THR,
                safe_auc=SAFE_AUC,
            )
            results[ckpt_name][ds_name] = m

            print(f"img_pos={int(m['img_pos'])} img_neg={int(m['img_neg'])}")
            print(
                f"pixel_f1={m['pixel_f1']:.4f}  "
                f"pixel_iou={m['pixel_iou']:.4f}  "
                f"pixel_acc={m['pixel_acc']:.4f}  "
                f"pixel_auc={m['pixel_auc']:.4f}"
            )

        # AVG over datasets (simple mean across 6 datasets)
        avg_f1 = float(np.mean([results[ckpt_name][ds]["pixel_f1"] for ds in DATASET_ORDER]))
        avg_iou = float(np.mean([results[ckpt_name][ds]["pixel_iou"] for ds in DATASET_ORDER]))
        avg_acc = float(np.mean([results[ckpt_name][ds]["pixel_acc"] for ds in DATASET_ORDER]))
        avg_auc = float(np.mean([results[ckpt_name][ds]["pixel_auc"] for ds in DATASET_ORDER]))

        results[ckpt_name]["AVG"] = {
            "pixel_f1": avg_f1,
            "pixel_iou": avg_iou,
            "pixel_acc": avg_acc,
            "pixel_auc": avg_auc,
        }

        print("\n== AVG over 6 datasets ==")
        print(
            f"AVG_pixel_f1={avg_f1:.4f}  "
            f"AVG_pixel_iou={avg_iou:.4f}  "
            f"AVG_pixel_acc={avg_acc:.4f}  "
            f"AVG_pixel_auc={avg_auc:.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ranking by AVG pixel_f1
    print("\n" + "=" * 110)
    print("Ranking by AVG(pixel_f1) over 6 datasets")
    print("=" * 110)

    ranking = [(ckpt, results[ckpt]["AVG"]["pixel_f1"]) for ckpt in results.keys()]
    ranking.sort(key=lambda x: x[1], reverse=True)

    print(f"{'ckpt':<28} | {'AVG_F1':>8} {'AVG_IoU':>8} {'AVG_Acc':>8} {'AVG_AUC':>8}")
    print("-" * 70)
    for ckpt, _ in ranking:
        a = results[ckpt]["AVG"]
        print(f"{ckpt:<28} | {a['pixel_f1']:>8.4f} {a['pixel_iou']:>8.4f} {a['pixel_acc']:>8.4f} {a['pixel_auc']:>8.4f}")

    best_ckpt = ranking[0][0]
    print("\nBEST =", best_ckpt)
    print("BEST_AVG =", results[best_ckpt]["AVG"])


if __name__ == "__main__":
    main()