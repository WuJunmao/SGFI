import os
import json
from pathlib import Path


def create_imd_json():
    """
    处理IMD2020数据集，生成JSON文件
    """
    # 配置路径
    dataset_base_path = Path("/home/lab301-3090/wujun/datasets/IMD2020")

    # 列表文件路径
    imd_list_path = Path("/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/IMD_train_list.txt")

    # 输出路径
    output_json_path = Path("/home/lab301-3090/wujun/SparseViT-main/data___json") / "IMD_dataset.json"

    print("=" * 80)
    print("处理IMD2020数据集")
    print("=" * 80)

    # 确保输出目录存在
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 检查路径
    print("检查路径是否存在...")
    paths_to_check = {
        "数据集根目录": dataset_base_path,
        "列表文件": imd_list_path,
        "输出目录": output_json_path.parent
    }

    for name, path in paths_to_check.items():
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path}")
            return

    all_entries = []
    missing_files = []
    txt_entries_count = 0

    # 读取和处理TXT文件
    print("\n" + "=" * 80)
    print("处理IMD2020列表文件")
    print("=" * 80)

    if imd_list_path.exists():
        with open(imd_list_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                txt_entries_count += 1
                parts = line.split(',')

                if len(parts) >= 4:
                    # 格式: 路径1,掩码路径,原始图像路径,篡改图像路径
                    mask_path = parts[1].strip()  # 第二项：掩码路径
                    tamp_path = parts[3].strip()  # 第四项：篡改图像路径

                    # 构建完整路径
                    full_tamp_path = dataset_base_path / tamp_path
                    full_mask_path = dataset_base_path / mask_path

                    # 检查文件是否存在
                    tamp_exists = full_tamp_path.exists()
                    mask_exists = full_mask_path.exists()

                    if tamp_exists and mask_exists:
                        # 两个文件都存在
                        all_entries.append([
                            str(full_tamp_path),
                            str(full_mask_path)
                        ])
                        print(f"✅ 第{line_num}行: {tamp_path} + {mask_path}")
                    elif tamp_exists and not mask_exists:
                        # 只有篡改图像存在
                        all_entries.append([
                            str(full_tamp_path),
                            ""
                        ])
                        missing_files.append(("掩码文件", line_num, mask_path))
                        print(f"⚠️  第{line_num}行: {tamp_path} (找到图像但未找到掩码 {mask_path})")
                    elif not tamp_exists and mask_exists:
                        # 只有掩码存在
                        missing_files.append(("篡改图像", line_num, tamp_path))
                        print(f"❌ 第{line_num}行: 找到掩码但未找到图像 {tamp_path}")
                    else:
                        # 两个文件都不存在
                        missing_files.append(("篡改图像", line_num, tamp_path))
                        missing_files.append(("掩码文件", line_num, mask_path))
                        print(f"❌ 第{line_num}行: 两个文件都不存在 {tamp_path}, {mask_path}")
                else:
                    print(f"⚠️  第{line_num}行格式异常: {line}")

    # 检查多余的文件
    print("\n" + "=" * 80)
    print("检查多余的文件")
    print("=" * 80)

    # 获取TXT文件中提到的所有文件
    txt_files_mentioned = set()
    if imd_list_path.exists():
        with open(imd_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        # 添加掩码路径和篡改图像路径
                        txt_files_mentioned.add(parts[1].strip())  # 掩码
                        txt_files_mentioned.add(parts[3].strip())  # 篡改图像

    # 获取实际数据集中的所有图像文件
    all_actual_files = set()
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    if dataset_base_path.exists():
        for file_path in dataset_base_path.rglob("*.*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # 转换为相对于数据集根目录的路径
                relative_path = str(file_path.relative_to(dataset_base_path))
                all_actual_files.add(relative_path)

    # 找出在数据集中但不在TXT中的文件
    extra_files = all_actual_files - txt_files_mentioned

    if extra_files:
        print(f"⚠️  多余的文件 ({len(extra_files)} 个):")
        for file_path in sorted(extra_files)[:20]:
            print(f"  {file_path}")
        if len(extra_files) > 20:
            print(f"  ... 还有 {len(extra_files) - 20} 个")
    else:
        print("✅ 没有多余的文件")

    # 保存JSON文件
    print("\n" + "=" * 80)
    print("保存JSON文件")
    print("=" * 80)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON文件已保存: {output_json_path}")

    # 生成详细报告
    print("\n" + "=" * 80)
    print("详细报告")
    print("=" * 80)

    print(f"📊 TXT文件统计:")
    print(f"  TXT文件总行数: {txt_entries_count}")

    print(f"\n📊 实际处理统计:")
    print(f"  成功处理的条目: {len(all_entries)}")

    # 统计有掩码和无掩码的篡改图像
    with_mask_count = len([entry for entry in all_entries if entry[1] != ""])
    without_mask_count = len([entry for entry in all_entries if entry[1] == ""])

    print(f"  篡改图像(有掩码): {with_mask_count}")
    print(f"  篡改图像(无掩码): {without_mask_count}")

    print(f"\n📊 缺失统计:")
    missing_images = len([f for f in missing_files if f[0] == "篡改图像"])
    missing_masks = len([f for f in missing_files if f[0] == "掩码文件"])

    print(f"  缺失的篡改图像: {missing_images}")
    print(f"  缺失的掩码文件: {missing_masks}")
    print(f"  总缺失文件: {len(missing_files)}")

    # 详细缺失文件列表
    if missing_files:
        print(f"\n❌ 详细的缺失文件列表:")
        for file_type, line_num, file_path in missing_files[:20]:  # 只显示前20个
            print(f"  第{line_num}行 {file_type}: {file_path}")
        if len(missing_files) > 20:
            print(f"  ... 还有 {len(missing_files) - 20} 个")

    # 最终总结
    print("\n" + "=" * 80)
    print("最终总结")
    print("=" * 80)

    if not missing_files and not extra_files:
        print("🎉 🎉 🎉 完美！所有文件都一一对应！ 🎉 🎉 🎉")
        print("✅ 所有TXT文件中的条目都在数据集中找到")
        print("✅ 数据集中没有多余的文件")
    else:
        if missing_files:
            print("⚠️  存在问题：")
            if missing_images > 0:
                print(f"  - 缺失 {missing_images} 个篡改图像")
            if missing_masks > 0:
                print(f"  - 缺失 {missing_masks} 个掩码文件")

        if extra_files:
            print(f"  - 有 {len(extra_files)} 个多余的文件在数据集中但不在TXT中")

        print(f"\n💡 建议:")
        if missing_files:
            print("  - 请检查缺失的文件是否在正确的位置")
        if extra_files:
            print("  - 可以考虑将这些多余文件添加到TXT文件中")


def verify_imd_structure():
    """
    验证IMD2020数据集结构
    """
    print("\n" + "=" * 80)
    print("验证IMD2020数据集结构")
    print("=" * 80)

    dataset_base_path = Path("/home/lab301-3090/wujun/datasets/IMD2020")

    if dataset_base_path.exists():
        print("数据集目录结构:")
        # 统计一级子目录
        subdirs = [d for d in dataset_base_path.iterdir() if d.is_dir()]
        print(f"  一级子目录数量: {len(subdirs)}")

        # 显示前10个子目录
        for i, subdir in enumerate(subdirs[:10]):
            file_count = len(list(subdir.glob("*.*")))
            print(f"  📁 {subdir.name}: {file_count} 个文件")

        if len(subdirs) > 10:
            print(f"  ... 还有 {len(subdirs) - 10} 个子目录")

        # 检查文件类型分布
        print(f"\n文件类型统计:")
        extensions = {}
        total_files = 0

        for file_path in dataset_base_path.rglob("*.*"):
            if file_path.is_file():
                total_files += 1
                ext = file_path.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1

        print(f"  总文件数: {total_files}")
        for ext, count in sorted(extensions.items()):
            if ext:  # 排除无后缀的文件
                print(f"  {ext}: {count} 个文件")
    else:
        print("❌ 数据集目录不存在")


if __name__ == "__main__":
    # 先验证数据集结构
    verify_imd_structure()

    # 处理数据集
    create_imd_json()