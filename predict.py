import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
import cv2
import os
from SparseViT_Mul import SparseViT_Mul  # 导入你的模型类


class ImagePredictor:
    def __init__(self, model_path, img_size=512, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.model_path = model_path

        # 初始化模型
        self.model = SparseViT_Mul(
            depth=[5, 8, 20, 7],
            embed_dim=[64, 128, 320, 512],
            head_dim=64,
            img_size=img_size,
            s_blocks3=[8, 4, 2, 1],
            s_blocks4=[2, 1],
            mlp_ratio=4,
            qkv_bias=True
        )

        # 加载训练好的权重
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """加载模型权重"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)

            if 'model_state_dict' in checkpoint:
                # 从完整检查点加载
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                # 从模型保存的检查点加载
                self.model.load_state_dict(checkpoint['model'])
            else:
                # 直接加载模型权重
                self.model.load_state_dict(checkpoint)

            print(f"成功加载模型权重: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

    def preprocess_image(self, image_path):
        """预处理图像"""
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)

        # 应用预处理
        input_tensor = self.transform(image).unsqueeze(0)  # 增加batch维度
        return input_tensor, original_size

    def predict(self, image_path, threshold=0.5):
        """进行预测"""
        # 预处理
        input_tensor, original_size = self.preprocess_image(image_path)
        input_tensor = input_tensor.to(self.device)

        # 创建虚拟的mask（推理时不需要真实mask）
        dummy_mask = torch.zeros(1, 1, self.img_size, self.img_size).to(self.device)

        # 推理
        with torch.no_grad():
            _, prediction = self.model(input_tensor, dummy_mask)

        # 后处理
        prediction = prediction.squeeze(0).squeeze(0)  # 移除batch和channel维度
        prediction = prediction.cpu().numpy()

        # 调整到原始图像大小
        prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_LINEAR)

        # 二值化
        binary_mask = (prediction > threshold).astype(np.uint8) * 255

        return prediction, binary_mask

    def save_results(self, image_path, prediction, binary_mask, output_dir):
        """保存预测结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 获取文件名
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # 保存概率图
        prob_map = (prediction * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_prob.png"), prob_map)

        # 保存二值掩码
        cv2.imwrite(os.path.join(output_dir, f"{filename}_mask.png"), binary_mask)

        # 保存叠加图
        original_image = cv2.imread(image_path)
        overlay = original_image.copy()
        overlay[binary_mask > 0] = [0, 255, 0]  # 绿色标注预测区域
        cv2.addWeighted(overlay, 0.3, original_image, 0.7, 0, overlay)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_overlay.png"), overlay)

        print(f"结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='使用SparseViT模型进行图像预测')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径 (.pth)')
    parser.add_argument('--image_path', type=str, required=True, help='要预测的图像路径')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda 或 cpu')

    args = parser.parse_args()

    # 创建预测器
    predictor = ImagePredictor(
        model_path=args.model_path,
        img_size=512,
        device=args.device
    )

    # 进行预测
    print(f"正在预测图像: {args.image_path}")
    prediction, binary_mask = predictor.predict(args.image_path, args.threshold)

    # 保存结果
    predictor.save_results(args.image_path, prediction, binary_mask, args.output_dir)

    print("预测完成!")


if __name__ == "__main__":
    import sys

    # 如果没有命令行参数，使用默认值
    if len(sys.argv) == 1:
        # 直接指定参数值
        model_path = 'checkpoint_train/checkpoint-156.pth'
        image_path = 'images/Tp/sample_1.jpg'
        output_dir = './predictions'
        threshold = 0.5
        device = 'cuda'

        # 创建预测器
        predictor = ImagePredictor(
            model_path=model_path,
            img_size=512,
            device=device
        )

        # 进行预测
        print(f"正在预测图像: {image_path}")
        prediction, binary_mask = predictor.predict(image_path, threshold)

        # 保存结果
        predictor.save_results(image_path, prediction, binary_mask, output_dir)

        print("预测完成!")
    else:
        # 使用命令行参数
        main()