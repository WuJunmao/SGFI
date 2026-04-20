import os
import json
from pathlib import Path


def create_fr_separate_jsons():
    """
    分别生成真实图像和篡改图像的JSON文件
    """
    # 配置路径
    dataset_base_path = Path("/home/lab301-3090/wujun/datasets/fealt reality/FantasticReality_v1/dataset")

    # 列表文件路径
    auth_list_path = Path("/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/FR_auth_train_list.txt")
    tamp_list_path = Path("/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/FR_train_list.txt")

    # 输出路径
    output_dir = Path("/home/lab301-3090/wujun/SparseViT-main/data___json")
    auth_output_path = output_dir / "FR_authentic.json"
    tamp_output_path = output_dir / "FR_tampered.json"

    print("=" * 80)
    print("分别生成FR数据集JSON文件")
    print("=" * 80)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    def find_corresponding_mask(image_filename):
        """根据图像文件名查找对应的掩码文件（图像名 + '_mask'）"""
        image_stem = Path(image_filename).stem

        # 掩码文件名格式：图像名 + '_mask'
        mask_stem = f"{image_stem}_mask"

        # 在mask目录中查找对应的文件
        mask_dir = dataset_base_path / "mask"
        if not mask_dir.exists():
            return None

        # 查找所有可能的掩码文件（不关心后缀名）
        for mask_file in mask_dir.glob("*"):
            if mask_file.is_file():
                if mask_file.stem == mask_stem:
                    return mask_file

        return None

    # 1. 处理真实图像（生成单独的JSON文件）
    print("\n" + "=" * 80)
    print("1. 处理真实图像列表")
    print("=" * 80)

    authentic_entries = []
    authentic_missing_files = []

    if auth_list_path.exists():
        with open(auth_list_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 真实图像是单文件名格式
                image_filename = line

                # 在多个位置查找真实图像文件
                possible_image_paths = [
                    dataset_base_path / image_filename,
                    dataset_base_path / "ColorRealImages" / image_filename,
                    dataset_base_path / "images" / image_filename,
                    dataset_base_path / "authentic" / image_filename,
                    dataset_base_path / "Au" / image_filename
                ]

                found_image_path = None
                for possible_path in possible_image_paths:
                    if possible_path.exists():
                        found_image_path = possible_path
                        break

                if found_image_path:
                    authentic_entries.append([
                        str(found_image_path),
                        "Negative"
                    ])
                    print(f"✅ 真实图像 第{line_num}行: {image_filename}")
                else:
                    authentic_missing_files.append(("真实图像", line_num, image_filename))
                    print(f"❌ 真实图像 第{line_num}行: 找不到 {image_filename}")

        # 保存真实图像JSON文件
        with open(auth_output_path, 'w', encoding='utf-8') as f:
            json.dump(authentic_entries, f, indent=2, ensure_ascii=False)
        print(f"✅ 真实图像JSON文件已保存: {auth_output_path}")
    else:
        print("❌ 真实图像列表文件不存在")

    # 2. 处理篡改图像（生成单独的JSON文件）
    print("\n" + "=" * 80)
    print("2. 处理篡改图像列表")
    print("=" * 80)

    tampered_entries = []
    tampered_missing_files = []

    if tamp_list_path.exists():
        with open(tamp_list_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 篡改图像是单文件名格式
                image_filename = line

                # 在多个位置查找篡改图像文件
                possible_image_paths = [
                    dataset_base_path / image_filename,
                    dataset_base_path / "ColorFakeImages" / image_filename,
                    dataset_base_path / "images" / image_filename,
                    dataset_base_path / "tampered" / image_filename,
                    dataset_base_path / "Tp" / image_filename
                ]

                found_image_path = None
                for possible_path in possible_image_paths:
                    if possible_path.exists():
                        found_image_path = possible_path
                        break

                if found_image_path:
                    # 查找对应的掩码文件（图像名 + '_mask'）
                    found_mask_path = find_corresponding_mask(image_filename)

                    if found_mask_path:
                        tampered_entries.append([
                            str(found_image_path),
                            str(found_mask_path)
                        ])
                        print(f"✅ 篡改图像 第{line_num}行: {image_filename} + {found_mask_path.name}")
                    else:
                        tampered_entries.append([
                            str(found_image_path),
                            ""
                        ])
                        tampered_missing_files.append(("篡改图像掩码", line_num, f"{Path(image_filename).stem}_mask.*"))
                        print(f"⚠️  篡改图像 第{line_num}行: {image_filename} (找到图像但未找到掩码)")
                else:
                    tampered_missing_files.append(("篡改图像", line_num, image_filename))
                    print(f"❌ 篡改图像 第{line_num}行: 找不到 {image_filename}")

        # 保存篡改图像JSON文件
        with open(tamp_output_path, 'w', encoding='utf-8') as f:
            json.dump(tampered_entries, f, indent=2, ensure_ascii=False)
        print(f"✅ 篡改图像JSON文件已保存: {tamp_output_path}")
    else:
        print("❌ 篡改图像列表文件不存在")

    # 3. 生成详细报告
    print("\n" + "=" * 80)
    print("3. 详细报告")
    print("=" * 80)

    print(f"📊 统计信息:")
    print(f"  真实图像JSON: {auth_output_path.name}")
    print(f"    - 条目数: {len(authentic_entries)}")
    print(f"  篡改图像JSON: {tamp_output_path.name}")
    print(f"    - 条目数: {len(tampered_entries)}")
    print(f"    - 有掩码: {len([entry for entry in tampered_entries if entry[1] != ''])}")
    print(f"    - 无掩码: {len([entry for entry in tampered_entries if entry[1] == ''])}")

    # 显示缺失文件报告
    all_missing_files = authentic_missing_files + tampered_missing_files
    if all_missing_files:
        print(f"\n❌ 缺失的文件 ({len(all_missing_files)} 个):")
        if authentic_missing_files:
            print(f"  真实图像缺失 ({len(authentic_missing_files)} 个):")
            for file_type, line_num, file_path in authentic_missing_files:
                print(f"    第{line_num}行 {file_type}: {file_path}")
        if tampered_missing_files:
            print(f"  篡改图像缺失 ({len(tampered_missing_files)} 个):")
            for file_type, line_num, file_path in tampered_missing_files:
                print(f"    第{line_num}行 {file_type}: {file_path}")
    else:
        print(f"\n🎉 所有文件都成功处理！")

    # 显示JSON文件示例
    print(f"\n📄 真实图像JSON文件格式示例:")
    print("[")
    for i, entry in enumerate(authentic_entries[:2]):
        print(f'  ["{entry[0]}", "{entry[1]}"],')
    if len(authentic_entries) > 2:
        print("  ...")
    print("]")

    print(f"\n📄 篡改图像JSON文件格式示例:")
    print("[")
    for i, entry in enumerate(tampered_entries[:2]):
        print(f'  ["{entry[0]}", "{entry[1]}"],')
    if len(tampered_entries) > 2:
        print("  ...")
    print("]")


def verify_naming_pattern():
    """
    验证命名模式
    """
    print("\n" + "=" * 80)
    print("验证命名模式")
    print("=" * 80)

    dataset_base_path = Path("/home/lab301-3090/wujun/datasets/fealt reality/FantasticReality_v1/dataset")
    mask_dir = dataset_base_path / "mask"

    # 读取篡改图像列表中的几个文件名
    tamp_list_path = Path("/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/FR_train_list.txt")

    if tamp_list_path.exists():
        with open(tamp_list_path, 'r', encoding='utf-8') as f:
            sample_filenames = []
            for i, line in enumerate(f):
                if i >= 5:  # 只取前5个
                    break
                line = line.strip()
                if line:
                    sample_filenames.append(line)

        print("篡改图像文件名示例:")
        for filename in sample_filenames:
            image_stem = Path(filename).stem
            expected_mask_pattern = f"{image_stem}_mask.*"
            print(f"  图像: {filename}")
            print(f"  期望掩码: {expected_mask_pattern}")

            # 检查是否存在对应的掩码
            if mask_dir.exists():
                found_masks = list(mask_dir.glob(f"{image_stem}_mask.*"))
                if found_masks:
                    print(f"  ✅ 找到掩码: {found_masks[0].name}")
                else:
                    print(f"  ❌ 未找到掩码")
            print()


if __name__ == "__main__":
    # 先验证命名模式
    verify_naming_pattern()

    # 分别处理数据集生成两个JSON文件
    create_fr_separate_jsons()