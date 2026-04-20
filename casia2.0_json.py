import os
import json
from pathlib import Path
from PIL import Image


def create_casia_separate_jsons():
    """
    重新整理CASIA 2.0数据集，分别生成真实图像和篡改图像的JSON文件
    """
    # 配置路径
    dataset_base_path = Path("/home/lab301-3090/wujun/datasets/CASIA")

    # 列表文件路径
    auth_list_path = Path("/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/CASIA_v2_auth_train_list.txt")
    tamp_list_path = Path("/home/lab301-3090/wujun/CAT-Net-main/Splicing/data/CASIA_v2_train_list.txt")

    # 输出路径
    output_dir = Path("/home/lab301-3090/wujun/SparseViT-main/data___json")
    auth_output_path = output_dir / "CASIA_authentic.json"
    tamp_output_path = output_dir / "CASIA_tampered.json"
    reorganized_jpg_path = dataset_base_path / "reorganized_jpg"

    print("=" * 80)
    print("重新整理CASIA 2.0数据集并分别生成JSON文件")
    print("=" * 80)

    # 创建重新组织的目录
    reorganized_jpg_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查路径
    print("检查路径是否存在...")
    paths_to_check = {
        "数据集根目录": dataset_base_path,
        "真实图像列表": auth_list_path,
        "篡改图像列表": tamp_list_path,
        "掩码目录": dataset_base_path / "CASIA 2 Groundtruth",
        "Au目录": dataset_base_path / "CASIA 2.0" / "Au",
        "Tp目录": dataset_base_path / "CASIA 2.0" / "Tp",
        "jpg目录": dataset_base_path / "CASIA 2.0" / "jpg"
    }

    for name, path in paths_to_check.items():
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path}")

    authentic_entries = []
    tampered_entries = []
    authentic_missing_files = []
    tampered_missing_files = []
    converted_files = []

    def convert_to_jpg(input_path, output_path):
        """将图像转换为JPEG格式"""
        try:
            img = Image.open(input_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output_path, 'JPEG', quality=100, subsampling=0)
            return True
        except Exception as e:
            print(f"转换失败: {input_path}, 错误: {e}")
            return False

    def find_file(filename, search_dirs):
        """在多个目录中查找文件"""
        for search_dir in search_dirs:
            possible_path = search_dir / filename
            if possible_path.exists():
                return possible_path
        return None

    # 搜索目录配置
    jpg_search_dirs = [
        dataset_base_path / "CASIA 2.0" / "jpg",
        dataset_base_path / "CASIA 2.0" / "Au",
        dataset_base_path / "CASIA 2.0" / "Tp",
        reorganized_jpg_path
    ]

    mask_search_dirs = [
        dataset_base_path / "CASIA 2 Groundtruth"
    ]

    # 1. 处理真实图像（生成单独的JSON文件）
    print("\n" + "=" * 80)
    print("处理真实图像...")
    print("=" * 80)

    if auth_list_path.exists():
        with open(auth_list_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) >= 3:
                    original_path = parts[0].strip()
                    jpg_path = parts[2].strip()
                    jpg_filename = Path(jpg_path).name

                    # 查找JPEG文件
                    found_jpg = find_file(jpg_filename, jpg_search_dirs)

                    if not found_jpg:
                        # 如果找不到，尝试从原始文件转换
                        original_filename = Path(original_path).name
                        original_stem = Path(original_path).stem

                        # 查找原始文件
                        original_search_dirs = [
                            dataset_base_path / "CASIA 2.0" / "Au",
                            dataset_base_path / "CASIA 2.0" / "jpg"
                        ]

                        found_original = find_file(original_filename, original_search_dirs)

                        if found_original:
                            # 转换到重新组织的目录
                            new_jpg_path = reorganized_jpg_path / f"{original_stem}.jpg"
                            if convert_to_jpg(found_original, new_jpg_path):
                                found_jpg = new_jpg_path
                                converted_files.append(f"真实图像: {original_filename} -> {new_jpg_path.name}")
                                print(f"🔄 转换: {original_filename} -> {new_jpg_path.name}")

                    if found_jpg:
                        authentic_entries.append([
                            str(found_jpg),
                            "Negative"
                        ])
                        print(f"✅ 真实图像 第{line_num}行: {found_jpg.name}")
                    else:
                        authentic_missing_files.append(("真实图像", line_num, jpg_path))
                        print(f"❌ 真实图像 第{line_num}行: 找不到 {jpg_path}")

        # 保存真实图像JSON文件
        with open(auth_output_path, 'w', encoding='utf-8') as f:
            json.dump(authentic_entries, f, indent=2, ensure_ascii=False)
        print(f"✅ 真实图像JSON文件已保存: {auth_output_path}")
    else:
        print("❌ 真实图像列表文件不存在")

    # 2. 处理篡改图像（生成单独的JSON文件）
    print("\n" + "=" * 80)
    print("处理篡改图像...")
    print("=" * 80)

    if tamp_list_path.exists():
        with open(tamp_list_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) >= 3:
                    original_path = parts[0].strip()
                    mask_path = parts[1].strip()
                    jpg_path = parts[2].strip()

                    jpg_filename = Path(jpg_path).name
                    mask_filename = Path(mask_path).name if mask_path != "None" else None

                    # 查找JPEG文件
                    found_jpg = find_file(jpg_filename, jpg_search_dirs)

                    if not found_jpg:
                        # 如果找不到，尝试从原始文件转换
                        original_filename = Path(original_path).name
                        original_stem = Path(original_path).stem

                        # 查找原始文件
                        original_search_dirs = [
                            dataset_base_path / "CASIA 2.0" / "Tp",
                            dataset_base_path / "CASIA 2.0" / "jpg"
                        ]

                        found_original = find_file(original_filename, original_search_dirs)

                        if found_original:
                            # 转换到重新组织的目录
                            new_jpg_path = reorganized_jpg_path / f"{original_stem}.jpg"
                            if convert_to_jpg(found_original, new_jpg_path):
                                found_jpg = new_jpg_path
                                converted_files.append(f"篡改图像: {original_filename} -> {new_jpg_path.name}")
                                print(f"🔄 转换: {original_filename} -> {new_jpg_path.name}")

                    # 查找掩码文件
                    found_mask = None
                    if mask_path != "None" and mask_filename:
                        found_mask = find_file(mask_filename, mask_search_dirs)

                    if found_jpg:
                        if mask_path == "None" or found_mask:
                            mask_entry = str(found_mask) if found_mask else ""
                            tampered_entries.append([
                                str(found_jpg),
                                mask_entry
                            ])
                            print(f"✅ 篡改图像 第{line_num}行: {found_jpg.name}")
                        else:
                            tampered_missing_files.append(("篡改图像掩码", line_num, mask_path))
                            print(f"❌ 篡改图像 第{line_num}行: 找不到掩码 {mask_path}")
                    else:
                        tampered_missing_files.append(("篡改图像", line_num, jpg_path))
                        print(f"❌ 篡改图像 第{line_num}行: 找不到 {jpg_path}")

        # 保存篡改图像JSON文件
        with open(tamp_output_path, 'w', encoding='utf-8') as f:
            json.dump(tampered_entries, f, indent=2, ensure_ascii=False)
        print(f"✅ 篡改图像JSON文件已保存: {tamp_output_path}")
    else:
        print("❌ 篡改图像列表文件不存在")

    # 3. 生成详细报告
    print("\n" + "=" * 80)
    print("处理报告")
    print("=" * 80)

    print(f"📊 统计信息:")
    print(f"  真实图像JSON: {auth_output_path.name}")
    print(f"    - 条目数: {len(authentic_entries)}")
    print(f"  篡改图像JSON: {tamp_output_path.name}")
    print(f"    - 条目数: {len(tampered_entries)}")
    print(f"    - 有掩码: {len([entry for entry in tampered_entries if entry[1] != ''])}")
    print(f"    - 无掩码: {len([entry for entry in tampered_entries if entry[1] == ''])}")

    if converted_files:
        print(f"\n🔄 转换的文件 ({len(converted_files)} 个):")
        for conversion in converted_files[:10]:
            print(f"  {conversion}")
        if len(converted_files) > 10:
            print(f"  ... 还有 {len(converted_files) - 10} 个")

    # 显示缺失文件报告
    all_missing_files = authentic_missing_files + tampered_missing_files
    if all_missing_files:
        print(f"\n❌ 缺失的文件 ({len(all_missing_files)} 个):")
        if authentic_missing_files:
            print(f"  真实图像缺失 ({len(authentic_missing_files)} 个):")
            for file_type, line_num, file_path in authentic_missing_files[:10]:
                print(f"    第{line_num}行 {file_type}: {file_path}")
            if len(authentic_missing_files) > 10:
                print(f"    ... 还有 {len(authentic_missing_files) - 10} 个")

        if tampered_missing_files:
            print(f"  篡改图像缺失 ({len(tampered_missing_files)} 个):")
            for file_type, line_num, file_path in tampered_missing_files[:10]:
                print(f"    第{line_num}行 {file_type}: {file_path}")
            if len(tampered_missing_files) > 10:
                print(f"    ... 还有 {len(tampered_missing_files) - 10} 个")
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


if __name__ == "__main__":
    create_casia_separate_jsons()