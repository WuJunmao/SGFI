import os
import json
from pathlib import Path


def match_casia_files():
    # 定义路径
    image_base_path = r"/home/lab301-3090/wujun/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp"
    au_base_path = r"/home/lab301-3090/wujun/datasets/CASIA/CASIA 1.0 dataset/Au"
    gt_base_path = r"/home/lab301-3090/wujun/datasets/CASIA/CASIA 1.0 groundtruth"

    # 支持的图像扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 存储匹配结果
    matched_pairs = []
    missing_gt_files = []

    # 首先处理Tp目录（有真值文件的图像）
    print("处理Tp目录（有真值文件的图像）...")
    for folder in ['CM', 'Sp']:
        image_folder = os.path.join(image_base_path, folder)
        gt_folder = os.path.join(gt_base_path, folder)

        print(f"正在处理 {folder} 文件夹...")

        # 检查文件夹是否存在
        if not os.path.exists(image_folder):
            print(f"警告: 图像文件夹不存在: {image_folder}")
            continue
        if not os.path.exists(gt_folder):
            print(f"警告: 真值文件夹不存在: {gt_folder}")
            continue

        # 获取图像文件夹中的所有文件
        image_files = [f for f in os.listdir(image_folder)
                       if os.path.isfile(os.path.join(image_folder, f)) and
                       Path(f).suffix.lower() in image_extensions]

        print(f"在 {folder} 中找到 {len(image_files)} 个图像文件")

        # 获取真值文件夹中的所有文件
        gt_files = [f for f in os.listdir(gt_folder)
                    if os.path.isfile(os.path.join(gt_folder, f)) and
                    f.endswith('_gt.png')]

        print(f"在 {folder} 中找到 {len(gt_files)} 个真值文件")

        # 创建真值文件的查找字典（去掉_gt.png后缀作为键）
        gt_dict = {}
        for gt_file in gt_files:
            key = gt_file.replace('_gt.png', '')
            gt_dict[key] = gt_file

        # 匹配文件
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)

            # 提取文件名（不含扩展名）
            image_name_without_ext = Path(image_file).stem

            # 检查对应的真值文件是否存在
            if image_name_without_ext in gt_dict:
                gt_filename = gt_dict[image_name_without_ext]
                gt_path = os.path.join(gt_folder, gt_filename)

                matched_pairs.append([
                    image_path,
                    gt_path
                ])
            else:
                missing_gt_files.append(image_file)
                print(f"✗ 未找到匹配的真值文件: {image_file}")

    # 然后处理Au目录（没有真值文件的图像）
    print("\n处理Au目录（没有真值文件的图像）...")
    if os.path.exists(au_base_path):
        au_files = [f for f in os.listdir(au_base_path)
                    if os.path.isfile(os.path.join(au_base_path, f)) and
                    Path(f).suffix.lower() in image_extensions]

        print(f"在Au目录中找到 {len(au_files)} 个图像文件")

        for au_file in au_files:
            au_path = os.path.join(au_base_path, au_file)
            matched_pairs.append([
                au_path,
                "Negative"
            ])
    else:
        print(f"警告: Au目录不存在: {au_base_path}")

    return matched_pairs, missing_gt_files


def save_to_json(matched_pairs, output_file="casia1.0.json"):
    """将匹配结果保存为JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_pairs, f, indent=2, ensure_ascii=False)

    print(f"\n匹配结果已保存到: {output_file}")
    print(f"总共匹配到 {len(matched_pairs)} 对文件")


def analyze_results(matched_pairs, missing_gt_files):
    """分析匹配结果"""
    if not matched_pairs:
        print("没有找到任何匹配的文件对")
        return

    # 统计各类文件数量
    tp_with_gt_count = sum(1 for pair in matched_pairs if pair[1] != "Negative")
    au_negative_count = sum(1 for pair in matched_pairs if pair[1] == "Negative")

    cm_count = sum(1 for pair in matched_pairs if "CM" in pair[0] and pair[1] != "Negative")
    sp_count = sum(1 for pair in matched_pairs if "Sp" in pair[0] and pair[1] != "Negative")

    print(f"\n=== 匹配结果统计 ===")
    print(f"Tp目录（有真值）匹配数量: {tp_with_gt_count}")
    print(f"  - CM 文件夹: {cm_count}")
    print(f"  - Sp 文件夹: {sp_count}")
    print(f"Au目录（无真值）数量: {au_negative_count}")
    print(f"总文件数量: {len(matched_pairs)}")
    print(f"缺少真值文件的Tp图像数量: {len(missing_gt_files)}")

    if missing_gt_files:
        print(f"\n=== 前10个缺少真值文件的图像 ===")
        for i, file in enumerate(missing_gt_files[:10]):
            print(f"{i + 1}. {file}")


def debug_folder_structure():
    """调试文件夹结构"""
    image_base_path = r"C:\Users\alice\Downloads\CASIA 1.0 dataset\\Modified Tp\Tp"
    au_base_path = r"C:\Users\alice\Downloads\CASIA 1.0 dataset\Au"
    gt_base_path = r"C:\Users\alice\Downloads\casia1groundtruth-master\CASIA 1.0 groundtruth"

    print("=== 文件夹结构调试信息 ===")

    # 检查Tp目录
    for folder in ['CM', 'Sp']:
        image_folder = os.path.join(image_base_path, folder)
        gt_folder = os.path.join(gt_base_path, folder)

        print(f"\n{folder} 文件夹:")
        print(f"图像文件夹: {image_folder} - {'存在' if os.path.exists(image_folder) else '不存在'}")
        print(f"真值文件夹: {gt_folder} - {'存在' if os.path.exists(gt_folder) else '不存在'}")

        if os.path.exists(image_folder):
            image_files = os.listdir(image_folder)
            print(f"图像文件数量: {len(image_files)}")

        if os.path.exists(gt_folder):
            gt_files = os.listdir(gt_folder)
            print(f"真值文件数量: {len(gt_files)}")

    # 检查Au目录
    print(f"\nAu目录:")
    print(f"Au文件夹: {au_base_path} - {'存在' if os.path.exists(au_base_path) else '不存在'}")
    if os.path.exists(au_base_path):
        au_files = os.listdir(au_base_path)
        print(f"Au文件数量: {len(au_files)}")
        if au_files:
            print(f"前5个Au文件: {au_files[:5]}")


if __name__ == "__main__":
    print("开始匹配CASIA数据集文件...")

    # 先调试文件夹结构
    debug_folder_structure()

    print("\n" + "=" * 50)
    print("开始文件匹配...")
    print("=" * 50)

    # 执行文件匹配
    matched_pairs, missing_gt_files = match_casia_files()

    # 分析结果
    analyze_results(matched_pairs, missing_gt_files)

    # 保存到JSON文件
    if matched_pairs:
        save_to_json(matched_pairs)

        # 显示匹配结果示例
        print(f"\n=== 匹配结果示例 ===")

        # 显示有真值文件的示例
        tp_examples = [pair for pair in matched_pairs if pair[1] != "Negative"][:3]
        if tp_examples:
            print("有真值文件的示例:")
            for i, pair in enumerate(tp_examples):
                print(f"{i + 1}. 图像: {os.path.basename(pair[0])}")
                print(f"   真值: {os.path.basename(pair[1])}")
                print(f"   图像路径: {pair[0]}")
                print(f"   真值路径: {pair[1]}\n")

        # 显示无真值文件的示例
        au_examples = [pair for pair in matched_pairs if pair[1] == "Negative"][:3]
        if au_examples:
            print("无真值文件的示例:")
            for i, pair in enumerate(au_examples):
                print(f"{i + 1}. 图像: {os.path.basename(pair[0])}")
                print(f"   真值: {pair[1]}")
                print(f"   图像路径: {pair[0]}\n")
    else:
        print("没有找到任何匹配的文件对，请检查路径和文件命名是否正确。")