import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ======= 这些 import 按你工程实际目录改 =======
# dataset.py 里定义的 json_dataset
from my_utils.datasets import json_dataset          # TODO: 改成你实际的路径，比如 just "from dataset import json_dataset"

# 你改好的带 Stage3 辅助分支的 SparseViT_Mul
from SparseViT_Mul import SparseViT_Mul    # TODO: 改成你 SparseViT_Mul 的实际文件路径
# ============================================


def compute_f1(pred_prob, mask, thresh=0.5, eps=1e-6):
    """
    pred_prob: B×1×H×W, [0,1]
    mask:      B×1×H×W, {0,1}
    """
    pred_bin = (pred_prob > thresh).float()

    tp = (pred_bin * mask).sum()
    fp = (pred_bin * (1 - mask)).sum()
    fn = ((1 - pred_bin) * mask).sum()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return f1.item(), precision.item(), recall.item()


def debug_overfit_small_json(
    json_path,
    img_size=512,
    device='cuda:0',
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ====== 构建数据集 ======
    # 这里我们只用 json_dataset，不用 balanced_dataset，
    # transform=None，表示不过随机 copy-move / inpainting 等增强，
    # 只用 base_dataset 里的 resize_transform（Normalize + Resize + ToTensorV2）。
    dataset = json_dataset(
        path=json_path,
        output_size=img_size,
        transform=None,          # 过拟合测试建议先关掉随机增强
        if_return_shape=False,
    )

    print(f"small_overfit.json 里共有 {len(dataset)} 张图，将全部一起做过拟合测试。")

    # 一次性把全部样本打成一个 batch，方便 overfit
    dataloader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # ====== 构建模型 ======
    model = SparseViT_Mul(
        img_size=img_size,
        pretrained_path=None,      # 这里只是结构/训练逻辑测试，可以先不加载预训练
        use_aux_stage3=True,       # 开启 Stage3 辅助分支
        aux_stage3_weight=0.1,     # 辅助损失权重，可以后面再调
    ).to(device)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    max_epoch = 80

    for epoch in range(max_epoch):
        for batch in dataloader:
            # base_dataset.__getitem__ 返回的是一个 list:
            # [tp_img, gt_img] (+ 可选 name/shape/type)
            images = batch[0].to(device)  # B×3×H×W
            masks  = batch[1].to(device)  # B×1×H×W

            optimizer.zero_grad()
            loss, pred_prob = model(images, masks)  # pred_prob 是 sigmoid 后概率
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            f1, p, r = compute_f1(pred_prob, masks, thresh=0.5)

        print(f"[Epoch {epoch:03d}] "
              f"loss={loss.item():.4f}, F1={f1:.4f}, P={p:.4f}, R={r:.4f}")

    print("small_overfit.json 真实样本过拟合测试完成。")


if __name__ == "__main__":
    json_path = "/home/lab301-3090/wujun/SparseViT-main/small_overfit.json"
    debug_overfit_small_json(json_path, img_size=512)