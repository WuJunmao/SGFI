import numpy as np
import os
import csv
import json

if __name__ == '__main__':

    root_path = r'/home/lab301-3090/wujun/datasets/NIST'
    mani_path = os.path.join(root_path, 'reference/manipulation', 'NC2016-manipulation-ref.csv')

    rem_cnt, spli_cnt, copy_cnt = 0, 0, 0
    rem_list, spli_list, copy_list = [], [], []

    with open(mani_path, 'r') as file:
        data = csv.reader(file)
        for i, line in enumerate(data):
            if i == 0:
                print(line)
                rem_index = line[0].split('|').index('IsManipulationTypeRemoval')
                splice_index = line[0].split('|').index('IsManipulationTypeSplice')
                copy_index = line[0].split('|').index('IsManipulationTypeCopyClone')
                probe_index = line[0].split('|').index('ProbeFileName')
                mask_index = line[0].split('|').index('ProbeMaskFileName')
                basefile_index = line[0].split('|').index('BaseFileName')

            # process removal data
            if line[0].split('|')[rem_index] == 'Y':
                rem_cnt += 1
                probe_file = str(line[0].split('|')[probe_index])
                mask_file = str(line[0].split('|')[mask_index].replace('manipulation', 'removal'))
                # 构建完整路径 - 保持Linux路径格式
                probe_full_path = os.path.join(root_path, probe_file)
                mask_full_path = os.path.join(root_path, mask_file)
                rem_list.append([probe_full_path, mask_full_path])

            # process splicing data
            if line[0].split('|')[splice_index] == 'Y':
                spli_cnt += 1
                probe_file = str(line[0].split('|')[probe_index])
                mask_file = str(line[0].split('|')[mask_index])
                # 构建完整路径 - 保持Linux路径格式
                probe_full_path = os.path.join(root_path, probe_file)
                mask_full_path = os.path.join(root_path, mask_file)
                spli_list.append([probe_full_path, mask_full_path])

            # process copy data
            if line[0].split('|')[copy_index] == 'Y':
                copy_cnt += 1
                probe_file = str(line[0].split('|')[probe_index])
                mask_file = str(line[0].split('|')[mask_index])
                # 构建完整路径 - 保持Linux路径格式
                probe_full_path = os.path.join(root_path, probe_file)
                mask_full_path = os.path.join(root_path, mask_file)
                copy_list.append([probe_full_path, mask_full_path])

    print(f'removal data count is :{rem_cnt}')
    print(f'splicing data count is :{spli_cnt}')
    print(f'copy data count is :{copy_cnt}')

    # 合并所有数据
    total_list = rem_list + spli_list + copy_list
    print(f'total data count is : {len(total_list)}')

    # 打乱数据
    import random

    random.seed(42)
    random.shuffle(total_list)

    # 保存为JSON格式
    output_json_path = os.path.join(root_path, 'NIST16_dataset.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(total_list, json_file, indent=2)

    print(f'JSON文件已保存到: {output_json_path}')

    # 可选：同时按类别保存单独的JSON文件
    output_removal_json = os.path.join(root_path, 'NIST16_removal.json')
    output_splice_json = os.path.join(root_path, 'NIST16_splice.json')
    output_copy_json = os.path.join(root_path, 'NIST16_copy.json')

    with open(output_removal_json, 'w') as f:
        json.dump(rem_list, f, indent=2)
    with open(output_splice_json, 'w') as f:
        json.dump(spli_list, f, indent=2)
    with open(output_copy_json, 'w') as f:
        json.dump(copy_list, f, indent=2)

    print('按类别保存的JSON文件也已生成')

    # 验证生成的JSON格式
    print("\n前3个样本的格式验证:")
    for i, sample in enumerate(total_list[:3]):
        print(f"样本 {i + 1}: {sample}")