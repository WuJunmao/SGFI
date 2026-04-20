"""
严格以TXT文件为基准生成COCO数据集JSON文件
确保每个TXT文件的每一行都对应处理
"""

import json
from pathlib import Path

# 配置路径
dataset_base = Path("/home/lab301-3090/wujun/datasets/tempcoco/tampCOCO")
output_dir = Path("/home/lab301-3090/wujun/SparseViT-main/data___json")

txt_files = {
    "bcm_COCO": "/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/bcm_COCO_train_list.txt",
    "bcmc_COCO": "/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/bcmc_COCO_train_list.txt",
    "cm_COCO": "/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/cm_COCO_train_list.txt",
    "sp_COCO": "/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/sp_COCO_train_list.txt"
}

def process_dataset_strict(txt_file_path, dataset_name):
    """严格处理单个数据集，确保与TXT文件完全对应"""

    txt_file = Path(txt_file_path)
    if not txt_file.exists():
        print(f"❌ TXT file not found: {txt_file_path}")
        return None, 0, 0, 0

    # 读取TXT文件所有行
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    total_lines = len(lines)
    valid_entries = []
    missing_entries = []

    print(f"\n🔍 处理 {dataset_name}: {total_lines} 行")

    # 严格按TXT文件顺序处理每一行
    for line_num, line in enumerate(lines, 1):
        if not line.strip():  # 跳过空行
            continue

        parts = line.strip().split(',')
        if len(parts) < 2:
            print(f"⚠️  第{line_num}行格式错误: {line}")
            missing_entries.append((line_num, "格式错误", line))
            continue

        image_name = parts[0].strip()
        mask_name = parts[1].strip()

        # 严格按TXT文件中的路径构建
        image_path = dataset_base / image_name
        mask_path = dataset_base / mask_name

        # 检查文件是否存在
        image_exists = image_path.exists()
        mask_exists = mask_path.exists()

        if image_exists and mask_exists:
            valid_entries.append([str(image_path), str(mask_path)])
        else:
            missing_info = []
            if not image_exists:
                missing_info.append(f"图像不存在: {image_name}")
            if not mask_exists:
                missing_info.append(f"掩码不存在: {mask_name}")
            missing_entries.append((line_num, " | ".join(missing_info), line))

    # 输出详细报告
    print(f"✅ {dataset_name} 处理完成:")
    print(f"   TXT文件总行数: {total_lines}")
    print(f"   有效条目: {len(valid_entries)}")
    print(f"   缺失条目: {len(missing_entries)}")

    if missing_entries:
        print(f"   ⚠️ 缺失详情 (前10个):")
        for i, (line_num, reason, content) in enumerate(missing_entries[:10]):
            print(f"     第{line_num}行: {reason}")

    return valid_entries, total_lines, len(valid_entries), len(missing_entries)

def main():
    """主函数 - 严格处理所有数据集"""

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    # 严格处理每个数据集
    for dataset_name, txt_path in txt_files.items():
        print(f"\n{'='*60}")

        valid_entries, total_lines, valid_count, missing_count = process_dataset_strict(txt_path, dataset_name)

        if valid_entries is not None:
            # 保存JSON文件
            output_file = output_dir / f"{dataset_name}.json"
            with open(output_file, 'w') as f:
                json.dump(valid_entries, f, indent=2)

            print(f"💾 JSON文件已保存: {output_file}")

            # 验证保存的JSON与TXT对应关系
            with open(output_file, 'r') as f:
                saved_data = json.load(f)

            if len(saved_data) == valid_count:
                print(f"✅ 验证通过: JSON条目数({len(saved_data)}) = 有效条目数({valid_count})")
            else:
                print(f"❌ 验证失败: JSON条目数({len(saved_data)}) ≠ 有效条目数({valid_count})")

            summary[dataset_name] = {
                'txt_lines': total_lines,
                'valid_entries': valid_count,
                'missing_entries': missing_count,
                'output_file': str(output_file)
            }

    # 最终总结报告
    print(f"\n{'='*60}")
    print("🎯 严格处理完成 - 最终总结:")
    print(f"{'数据集':<12} {'TXT行数':<8} {'有效条目':<8} {'缺失条目':<8} {'完成度':<8}")
    print("-" * 50)

    for dataset_name, stats in summary.items():
        completion_rate = (stats['valid_entries'] / stats['txt_lines']) * 100
        print(f"{dataset_name:<12} {stats['txt_lines']:<8} {stats['valid_entries']:<8} {stats['missing_entries']:<8} {completion_rate:.1f}%")

    # 保存总结报告
    summary_file = output_dir / "processing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("COCO数据集处理总结报告\n")
        f.write("=" * 50 + "\n")
        for dataset_name, stats in summary.items():
            completion_rate = (stats['valid_entries'] / stats['txt_lines']) * 100
            f.write(f"{dataset_name}: {stats['valid_entries']}/{stats['txt_lines']} ({completion_rate:.1f}%)\n")

    print(f"\n📊 详细总结已保存: {summary_file}")

if __name__ == "__main__":
    main()