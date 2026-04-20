from typing import Iterable
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import my_utils.datasets
import my_utils.sparsevit_transforms
import my_utils.evaluation as evaluation
import my_utils.misc as misc
import SparseViT_Mul

# ========= 单批评估（复用你现有口径，不改主评估逻辑） =========
def _eval_step_once(model, images, masks, threshold):
    import torch.nn.functional as F
    from my_utils import evaluation

    model.eval()
    with torch.no_grad():
        _, pred = model(images, masks)

        # 尺寸对齐（与你已有一致）
        if pred.dim() == 3: pred = pred.unsqueeze(1)
        if masks.dim() == 3: masks = masks.unsqueeze(1)
        Hm, Wm = masks.shape[-2:]
        if pred.shape[-2:] != (Hm, Wm):
            pred = F.interpolate(pred, size=(Hm, Wm), mode='bilinear', align_corners=False)

        # 概率域判断：若像 logits（有负数或>1），则 sigmoid；否则按概率使用
        if float(pred.min()) < -0.01 or float(pred.max()) > 1.01:
            pred = torch.sigmoid(pred)

        # mask 口径：若有255，按 >127 二值；否则 >0.5
        masks_eval = (masks>127).float() if masks.max()>1 else (masks>0.5).float()

        TP, TN, FP, FN = evaluation.cal_confusion_matrix(pred, masks_eval, threshold=float(threshold))
        f1 = evaluation.cal_F1(TP, TN, FP, FN).mean()
        return float(f1)



# ========= 诊断辅助：统计一批图像的通道均值/标准差 =========
def _diag_print_image_stats(data_loader, device):
    images, masks = next(iter(data_loader))
    x = images.to(device).float()
    ch_mean = x.mean(dim=(0,2,3)).detach().cpu().numpy().tolist()
    ch_std  = x.std(dim=(0,2,3)).detach().cpu().numpy().tolist()
    print("\n=== Normalize 体检（首个batch） ===")
    print("image channel mean:", [round(v,4) for v in ch_mean])
    print("image channel std :", [round(v,4) for v in ch_std])
    print("提示：若均值/方差看起来像 0~1 原始值（而非标准化后≈0/≈1），说明测试可能未 Normalize。")

# ========= 诊断辅助：统计一批 mask 的值域/口径 =========
def _diag_print_mask_stats(data_loader, device):
    images, masks = next(iter(data_loader))
    m = masks.to(device)
    u = torch.unique(m)
    print("\n=== Mask 体检（首个batch） ===")
    print("mask min/max:", float(m.min()), float(m.max()))
    print("mask unique（前10）:", u[:10].detach().cpu().tolist())
    print("pos ratio (>127):", float((m>127).float().mean()))
    print("pos ratio (>0.5):", float((m>0.5).float().mean()))
    print("提示：若存在 255，评估时应按 (>127) 作为正类口径。")

# ===============================
# 工具函数：尺寸对齐
# ===============================
def _align_predict_to_mask(predict: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    若 predict 和 masks 尺寸不同，把 predict 上采样到 masks 的 HxW
    predict: [B, 1, Hp, Wp] 或 [B, Hp, Wp]
    masks  : [B, 1, Hm, Wm] 或 [B, Hm, Wm]
    返回   : 与 masks 空间尺寸一致的 predict (保持通道维)
    """
    if predict.dim() == 3:
        predict = predict.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)      # [B,H,W] -> [B,1,H,W]

    ph, pw = predict.shape[-2], predict.shape[-1]
    mh, mw = masks.shape[-2], masks.shape[-1]
    if (ph, pw) != (mh, mw):
        predict = F.interpolate(predict, size=(mh, mw), mode='bilinear', align_corners=False)
    return predict


# ===============================
# 与训练一致（逐样本F1再平均）
# ===============================
def training_consistent_evaluation(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    threshold: float = 0.5,
):
    """
    与训练时 test_one_epoch 完全一致的评估方式：
    - 对每个样本独立计算 F1
    - 对全体样本的 F1 做平均
    """
    model.eval()

    all_f1_scores = []     # 每个样本的 F1
    all_predictions = []   # 累积概率用于 AUC
    all_targets = []       # 累积 GT 用于 AUC

    print(f"=== 与训练一致的评估方法 (阈值={threshold}) ===")
    print("评估级别: 样本级别 (先计算每个样本的F1，再求平均)")

    with torch.no_grad():
        for data_iter_step, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)

            # 前向
            _, predict = model(images, masks)
            # 尺寸对齐
            predict = _align_predict_to_mask(predict, masks)

            # 与训练一致：用 cal_confusion_matrix + cal_F1
            TP, TN, FP, FN = evaluation.cal_confusion_matrix(predict, masks, threshold=threshold)
            batch_f1 = evaluation.cal_F1(TP, TN, FP, FN)  # [B]
            all_f1_scores.extend(batch_f1.detach().cpu().numpy())

            # AUC 累积
            all_predictions.append(predict.detach().cpu())
            all_targets.append(masks.detach().cpu())

            if data_iter_step % 50 == 0:
                print(f"批次 {data_iter_step}: 本批平均F1 = {float(batch_f1.mean()):.4f}")

    # 汇总
    final_f1 = float(np.mean(all_f1_scores))
    f1_std = float(np.std(all_f1_scores))

    all_predictions = torch.cat(all_predictions).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    dataset_auc = roc_auc_score(all_targets, all_predictions)

    print("\n📊 **评估结果 (与训练一致)**")
    print(f"   总样本数: {len(all_f1_scores)}")
    print(f"   总像素数: {len(all_predictions):,}")
    print(f"   正样本比例: {all_targets.mean():.4f}")
    print(f"🎯 **平均F1分数**: {final_f1:.4f} ± {f1_std:.4f}")
    print(f"📈 **AUC分数**: {dataset_auc:.4f}")

    # F1 分布
    f1_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"\n📋 F1分数分布:")
    for i in range(len(f1_bins) - 1):
        low, high = f1_bins[i], f1_bins[i + 1]
        cnt = sum(1 for f1 in all_f1_scores if low <= f1 < high)
        pct = cnt / len(all_f1_scores) * 100 if all_f1_scores else 0
        print(f"   F1 [{low:.1f}-{high:.1f}): {cnt} 样本 ({pct:.1f}%)")

    return {
        'f1': final_f1,
        'f1_std': f1_std,
        'auc': float(dataset_auc),
        'all_f1_scores': all_f1_scores,
        'all_predictions': all_predictions,
        'all_targets': all_targets,
        'num_samples': len(all_f1_scores),
        'threshold': threshold
    }


# ===============================
# 图像级别评估（整图二值后计算）
# ===============================
def image_level_evaluation(model, data_loader, device, threshold=0.5):
    model.eval()
    image_f1_scores = []

    print(f"\n=== 图像级别评估 (阈值={threshold}) ===")

    with torch.no_grad():
        for data_iter_step, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            _, predict = model(images, masks)
            predict = _align_predict_to_mask(predict, masks)

            binary_pred = (predict > threshold).float()

            # 逐图计算
            B = images.shape[0]
            for i in range(B):
                pred_flat = binary_pred[i].flatten()
                mask_flat = masks[i].flatten()

                TP = ((pred_flat == 1) & (mask_flat == 1)).sum().item()
                FP = ((pred_flat == 1) & (mask_flat == 0)).sum().item()
                FN = ((pred_flat == 0) & (mask_flat == 1)).sum().item()

                precision = TP / (TP + FP + 1e-8)
                recall = TP / (TP + FN + 1e-8)
                denom = (precision + recall)
                f1 = 2 * precision * recall / denom if denom > 0 else 0.0
                image_f1_scores.append(f1)

            if data_iter_step % 50 == 0:
                print(f"批次 {data_iter_step}: 本批平均F1 = {np.mean(image_f1_scores[-B:]):.4f}")

    final_f1 = float(np.mean(image_f1_scores)) if image_f1_scores else 0.0
    f1_std = float(np.std(image_f1_scores)) if image_f1_scores else 0.0

    print(f"\n🎯 **图像级别F1分数**: {final_f1:.4f} ± {f1_std:.4f}")

    return {
        'image_f1': final_f1,
        'image_f1_std': f1_std,
        'all_image_f1_scores': image_f1_scores
    }


# ===============================
# 预测调试
# ===============================
def debug_predictions(model, data_loader, device, num_samples=3):
    model.eval()
    print(f"\n=== 预测结果调试 (查看{num_samples}个样本) ===")

    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            if i >= 1:  # 只看第一个 batch
                break

            images, masks = images.to(device), masks.to(device)
            _, predictions = model(images, masks)
            predictions = _align_predict_to_mask(predictions, masks)

            print(f"输入图像范围: [{images.min().item():.3f}, {images.max().item():.3f}]")
            print(f"预测值范围: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
            print(f"真实Mask中1的比例: {masks.float().mean().item():.3f}")
            print(f"预测值中>0.5的比例: {(predictions > 0.5).float().mean().item():.3f}")

            for j in range(min(num_samples, images.shape[0])):
                pred_j = predictions[j]
                mask_j = masks[j]

                TP = ((pred_j > 0.5) & (mask_j == 1)).sum().item()
                FP = ((pred_j > 0.5) & (mask_j == 0)).sum().item()
                FN = ((pred_j <= 0.5) & (mask_j == 1)).sum().item()

                precision = TP / (TP + FP + 1e-8)
                recall = TP / (TP + FN + 1e-8)
                denom = precision + recall
                f1 = 2 * precision * recall / denom if denom > 0 else 0.0

                print(f"样本 {j}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
                print(f"       预测>0.5: {(pred_j > 0.5).float().mean().item():.3f}, 真实1: {mask_j.float().mean().item():.3f}")


# ===============================
# 数据预处理检查（安全处理 Tensor/Numpy）
# ===============================
def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def check_data_processing(dataset):
    print("\n=== 数据预处理检查 ===")
    if len(dataset) == 0:
        print("数据集为空。")
        return
    sample = dataset[0]
    img, msk = sample[0], sample[1]

    img_np = _as_numpy(img)
    msk_np = _as_numpy(msk)

    # 兼容 [C,H,W] 或 [H,W] 形状
    img_shape = img_np.shape
    msk_shape = msk_np.shape

    print(f"图像形状: {img_shape}")
    print(f"Mask形状: {msk_shape}")
    print(f"图像值范围: [{img_np.min():.3f}, {img_np.max():.3f}]")
    print(f"Mask值范围: [{msk_np.min():.3f}, {msk_np.max():.3f}]")

    # 正样本比例
    # (mask 可能是 {0,1} 或 {0,255}，这里统一 /255 后再看均值也可；直接均值也能看比例趋势)
    if msk_np.max() > 1.0:
        pos_ratio = (msk_np > 127).mean()
    else:
        pos_ratio = msk_np.mean()
    print(f"Mask中正样本比例(粗略): {float(pos_ratio):.4f}")


# ===============================
# 数据集顺序一致性检查
# ===============================
def check_dataset_consistency(dataset, num_checks=5):
    print(f"\n=== 数据集顺序一致性检查 (检查{num_checks}个位置) ===")
    test_indices = [0, 100, 200, 300, 400]

    consistency_results = []
    for idx in test_indices[:num_checks]:
        if idx >= len(dataset):
            continue
        sample1 = dataset[idx]
        sample2 = dataset[idx]

        img_same = torch.allclose(sample1[0], sample2[0]) if isinstance(sample1[0], torch.Tensor) else np.allclose(_as_numpy(sample1[0]), _as_numpy(sample2[0]))
        mask_same = torch.allclose(sample1[1], sample2[1]) if isinstance(sample1[1], torch.Tensor) else np.allclose(_as_numpy(sample1[1]), _as_numpy(sample2[1]))

        ok = bool(img_same and mask_same)
        consistency_results.append((idx, ok))
        print(f"  索引 {idx}: {'✅ 一致' if ok else '❌ 不一致'}")

    all_consistent = all(flag for _, flag in consistency_results) if consistency_results else True
    print(f"总体一致性: {'✅ 通过' if all_consistent else '❌ 失败'}")
    return all_consistent


# ===============================
# 阈值扫描
# ===============================
def sweep_thresholds(model, data_loader, device, thresholds=None):
    """
    对多个阈值计算像素级 F1，找到最优阈值（与训练一致的计算口径）
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)  # 0.10, 0.15, ..., 0.90

    model.eval()
    results = []

    print("\n=== 阈值扫描开始 ===")
    with torch.no_grad():
        for t in thresholds:
            all_f1 = []
            for images, masks in data_loader:
                images, masks = images.to(device), masks.to(device)
                _, predict = model(images, masks)
                predict = _align_predict_to_mask(predict, masks)

                TP, TN, FP, FN = evaluation.cal_confusion_matrix(predict, masks, threshold=float(t))
                f1_batch = evaluation.cal_F1(TP, TN, FP, FN)
                all_f1.extend(f1_batch.detach().cpu().numpy())

            mean_f1 = float(np.mean(all_f1)) if len(all_f1) else 0.0
            results.append((float(t), mean_f1))
            print(f"阈值 {t:.2f} → 平均 F1 = {mean_f1:.4f}")

    best_t, best_f1 = max(results, key=lambda x: x[1]) if results else (0.5, 0.0)
    print("=== 阈值扫描完成 ===")
    print(f"最佳阈值: {best_t:.2f} → 最优平均 F1 = {best_f1:.4f}")

    return results, best_t, best_f1


# ===============================
# 主流程
# ===============================
def main():
    MODEL_PATH = '/home/lab301-3090/wujun/SparseViT-main/output_dir/checkpoint-162.pth'
    TEST_DATA_PATH = '/home/lab301-3090/wujun/SparseViT-main/data___json/Columbia_dataset.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("🚀 加载模型...")
    model = SparseViT_Mul.SparseViT_Mul(pretrained_path='checkpoint/train/pretrain/uniformer_base_ls_in1k.pth')
    model.to(device)

    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("✅ 从checkpoint的'model'键加载模型参数")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 直接加载checkpoint参数")

    # 模型设置信息
    verify_model_setup(model)

    # ====== 加载测试数据 ======
    print("\n📊 加载测试数据...")
    # ⚠️ 若你希望严格复现论文“测试不做resize”，请在 my_utils.sparsevit_transforms.get_albu_transforms('test')
    # 内只保留 Normalize + ToTensor，不要 Resize / Crop。
    test_transform = my_utils.sparsevit_transforms.get_albu_transforms('test')

    try:
        dataset_test = my_utils.datasets.json_dataset(
            path=TEST_DATA_PATH,
            transform=test_transform,
        )
        print("✅ 使用 json_dataset 加载测试数据")
    except:
        print("⚠️ JSON格式不支持，尝试使用 mani_dataset")
        dataset_test = my_utils.datasets.mani_dataset(
            path=os.path.dirname(TEST_DATA_PATH),
            transform=test_transform,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=False,   # 测试保持顺序
        num_workers=4
    )

    print(f"测试数据集: {len(dataset_test)} 个样本")
    print(f"数据集类型: {type(dataset_test).__name__}")
    # ====== 诊断区：不改主评估逻辑，只做打印与对比 ======
    # 1. 是否做了 Normalize（首个batch的通道均值/方差）
    _diag_print_image_stats(data_loader_test, device)

    # 2. Mask 是否 0/255 口径
    _diag_print_mask_stats(data_loader_test, device)

    # 3. RGB/BGR A-B 测试：同一批数据对比通道翻转前后 F1（少量批次）
    print("\n=== RGB/BGR A-B 测试（前3个batch）===")
    max_batches = 3
    f1_rgb_list, f1_bgr_list = [], []
    with torch.no_grad():
        for bi, (images, masks) in enumerate(data_loader_test):
            if bi >= max_batches:
                break
            images = images.to(device).float()
            masks = masks.to(device)
            # 当前喂法（假设是 RGB）
            f1_rgb = _eval_step_once(model, images, masks, threshold=0.5)
            # 通道交换（BGR <-> RGB）
            images_bgr = images[:, [2, 1, 0], :, :]
            f1_bgr = _eval_step_once(model, images_bgr, masks, threshold=0.5)
            f1_rgb_list.append(f1_rgb)
            f1_bgr_list.append(f1_bgr)
            ch_mean = images.mean(dim=(0, 2, 3)).detach().cpu().numpy().tolist()
            print(
                f"[batch {bi}] ch-mean={[round(v, 4) for v in ch_mean]}  F1_RGB={f1_rgb:.4f}  F1_BGRswap={f1_bgr:.4f}")

    if f1_bgr_list and f1_rgb_list:
        import numpy as np
        print("A/B 汇总：RGB均值F1=%.4f,  BGR-swap均值F1=%.4f"
              % (np.mean(f1_rgb_list), np.mean(f1_bgr_list)))
        print("提示：若 BGR-swap 明显更高，说明当前通道顺序/Normalize 可能有偏差。")
    # ====== 诊断区结束 ======

    # 基本检查
    is_consistent = check_dataset_consistency(dataset_test)
    if not is_consistent:
        print("🚨 警告: 数据集采样不是顺序的！可能存在随机采样问题（检查 __getitem__/transform）")

    check_data_processing(dataset_test)

    # 预测范围调试
    debug_predictions(model, data_loader_test, device)

    # ====== 1) 与训练一致评估 (阈值=0.5) ======
    print("\n" + "=" * 60)
    print("1. 与训练一致的评估 (阈值=0.5)")
    print("=" * 60)
    results = training_consistent_evaluation(model, data_loader_test, device, threshold=0.5)

    # ====== 2) 图像级别评估 (阈值=0.5) ======
    print("\n" + "=" * 60)
    print("2. 图像级别评估 (阈值=0.5)")
    print("=" * 60)
    image_results = image_level_evaluation(model, data_loader_test, device, threshold=0.5)

    # ====== 3) 与论文结果对比 ======
    print("\n" + "=" * 60)
    print("3. 与论文结果对比")
    print("=" * 60)
    paper_results = {
        'COVERAGE': 0.513,
        'Columbia': 0.959,
        'CASIAv1': 0.827,
        'NIST16': 0.384,
        'DEF-12k': 0.197
    }
    your_pixel_f1 = results['f1']
    your_image_f1 = image_results['image_f1']
    paper_f1 = paper_results.get('CASIAv1', 0.827)

    print(f"您的像素级别F1 (阈值0.5): {your_pixel_f1:.4f}")
    print(f"您的图像级别F1 (阈值0.5): {your_image_f1:.4f}")
    print(f"论文F1 (CASIAv1): {paper_f1:.4f}")
    print(f"像素级别差异: {your_pixel_f1 - paper_f1:+.4f}")
    print(f"图像级别差异: {your_image_f1 - paper_f1:+.4f}")

    # ====== 4) 阈值扫描 ======
    print("\n" + "=" * 60)
    print("4. 阈值扫描 (寻找最优阈值)")
    print("=" * 60)
    _, best_t, best_f1 = sweep_thresholds(model, data_loader_test, device)

    # ====== 5) 最终结果总结 ======
    print("\n" + "=" * 60)
    print("📈 最终结果总结")
    print("=" * 60)
    print(f"[阈值=0.50] 像素级 F1: {results['f1']:.4f} | 图像级 F1: {image_results['image_f1']:.4f} | AUC: {results['auc']:.4f}")
    print(f"[最佳阈值={best_t:.2f}] 最优像素级 F1: {best_f1:.4f}")
    print(f"正样本比例(像素级): {results['all_targets'].mean():.4f}")
    print(f"评估样本数: {results['num_samples']}")
    print(f"数据集顺序一致性: {'✅ 通过' if is_consistent else '❌ 失败'}")


# ===============================
# 额外：模型结构/可训练参数打印
# ===============================
def verify_model_setup(model):
    print("\n=== 模型设置验证 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    training_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.BatchNorm2d):
            training_layers.append((name, type(module).__name__))
    if training_layers:
        print("注意: 模型包含训练时特有的层:")
        for name, layer_type in training_layers:
            print(f"  {name}: {layer_type}")
    else:
        print("模型不包含Dropout或BatchNorm层")


if __name__ == '__main__':
    main()

