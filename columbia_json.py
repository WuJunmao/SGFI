import json
from pathlib import Path


def create_columbia_json():
    # 基础路径
    base_path = Path("/home/lab301-3090/wujun/datasets/columbia-tp+true")

    # 列表文件路径
    data_dir = Path("/home/lab301-3090/wujun/SparseViT-main")
    columbia_list_path = data_dir / "Columbia_list.txt"
    columbia_auth_list_path = data_dir / "Columbia_auth_list.txt"

    # 输出JSON文件路径
    output_json_path = data_dir / "Columbia_dataset.json"

    all_entries = []

    # 处理篡改图像
    spliced_image_dir = base_path
    spliced_mask_dir = base_path

    # 处理真实图像
    auth_image_dir = base_path

    # 读取篡改图像列表
    with open(columbia_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                jpg_name = parts[2].strip()
                mask_name = parts[1].strip()

                # 直接构建路径
                jpg_path = spliced_image_dir / jpg_name
                mask_path = spliced_mask_dir / mask_name if mask_name != "None" else ""

                all_entries.append([str(jpg_path), str(mask_path)])

    # 读取真实图像列表
    with open(columbia_auth_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                jpg_name = parts[2].strip()

                # 直接构建路径
                jpg_path = auth_image_dir / jpg_name

                all_entries.append([str(jpg_path), "Negative"])

    # 保存JSON
    with open(output_json_path, 'w') as f:
        json.dump(all_entries, f, indent=2)

    print(f"JSON文件已保存: {output_json_path}")
    print(f"总计条目: {len(all_entries)}")


if __name__ == "__main__":
    create_columbia_json()