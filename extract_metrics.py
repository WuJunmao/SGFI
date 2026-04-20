#!/usr/bin/env python3
import os
import json
import glob
from collections import defaultdict

def extract_metrics_from_log(log_file_path):
    """Extract metrics from a log.txt file"""
    try:
        with open(log_file_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return None

            # Handle multiple JSON objects in the file
            lines = content.split('\n')
            metrics_list = []

            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        metrics = json.loads(line)
                        metrics_list.append(metrics)
                    except json.JSONDecodeError:
                        continue

            # Return the first valid metrics if we found any
            return metrics_list[0] if metrics_list else None
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return None

def get_all_eval_directories():
    """Get all evaluation directories based on the task description"""
    base_path = '/home/lab301-3090/wujun/benco'

    # List of all evaluation directories as per the task
    eval_dirs = [
        # cat_net
        'eval_dir_catnet',
        'eval_dir_catnetfanzhuan',

        # pscc
        'eval_dir_pscc',
        'eval_dir_psccfanzhuan',

        # mvss
        'eval_dir_mvss',
        'eval_dir_mvssfanzhuan',

        # objectformer (obj)
        'eval_dir_obj',
        'eval_dir_objfanzhuan',

        # trufor
        'eval_dir_trufor',
        'eval_dir_truforfanzhuan',

        # sparsevit
        'eval_dir_sparsevit',
        'eval_dir_sparsevitfanzhuan',

        # sparsevit_full
        'eval_dir_sparsevit_mulfull',
        'eval_dir_sparsevit_mulfull_fanzhuan',

        # Additional sparsevit directories mentioned
        'eval_dir_sparsevit_aux0.1',
        'eval_dir_sparsevit_aux0.2',
        'eval_dir_sparsevit_aux0.3',
        'eval_dir_sparsevit_shougong',
        'eval_dir_sparsevit_mulfull_np+srm+bayar'
    ]

    return [os.path.join(base_path, d) for d in eval_dirs if os.path.exists(os.path.join(base_path, d))]

def extract_all_metrics():
    """Extract metrics from all evaluation directories"""
    eval_dirs = get_all_eval_directories()
    results = defaultdict(lambda: defaultdict(list))

    for eval_dir in eval_dirs:
        model_name = os.path.basename(eval_dir)

        # Find all log.txt files, excluding archive directory
        log_files = glob.glob(os.path.join(eval_dir, '[!a]*/log.txt'))  # [!a]* excludes archive
        log_files += glob.glob(os.path.join(eval_dir, '[!a]*/[!a]*/log.txt'))  # Also check nested directories

        for log_file in log_files:
            # Get dataset name from path
            parts = log_file.split('/')
            dataset_name = None

            # Find the dataset directory name (should be before 'log.txt')
            for i, part in enumerate(parts):
                if part == 'log.txt' and i > 0:
                    dataset_name = parts[i-1]
                    break

            if not dataset_name or dataset_name == 'archive':
                continue

            metrics = extract_metrics_from_log(log_file)
            if metrics:
                # Store metrics with model and dataset info
                results[model_name][dataset_name].append(metrics)

    return results

def write_final_report(results):
    """Write the aggregated results to final.txt"""
    output_path = '/home/lab301-3090/wujun/SparseViT-main/final.txt'

    with open(output_path, 'w') as f:
        f.write("Model Evaluation Metrics Summary\n")
        f.write("=" * 50 + "\n\n")

        # Group models by type for better organization
        model_groups = {
            'CatNet': ['eval_dir_catnet', 'eval_dir_catnetfanzhuan'],
            'PSCC': ['eval_dir_pscc', 'eval_dir_psccfanzhuan'],
            'MVSS': ['eval_dir_mvss', 'eval_dir_mvssfanzhuan'],
            'ObjectFormer': ['eval_dir_obj', 'eval_dir_objfanzhuan'],
            'TruFor': ['eval_dir_trufor', 'eval_dir_truforfanzhuan'],
            'SparseViT': ['eval_dir_sparsevit', 'eval_dir_sparsevitfanzhuan'],
            'SparseViT Auxiliary': ['eval_dir_sparsevit_aux0.1', 'eval_dir_sparsevit_aux0.2', 'eval_dir_sparsevit_aux0.3'],
            'SparseViT Full': ['eval_dir_sparsevit_mulfull', 'eval_dir_sparsevit_mulfull_fanzhuan'],
            'SparseViT Other': ['eval_dir_sparsevit_shougong', 'eval_dir_sparsevit_mulfull_np+srm+bayar']
        }

        for group_name, model_names in model_groups.items():
            f.write(f"\n{group_name} Models\n")
            f.write("-" * 30 + "\n")

            for model_name in model_names:
                if model_name in results:
                    f.write(f"\nModel: {model_name}\n")
                    f.write("Dataset".ljust(20) + "F1".rjust(12) + "AUC".rjust(12) +
                           "IOU".rjust(12) + "Accuracy".rjust(12) + "\n")
                    f.write("-" * 70 + "\n")

                    for dataset_name, metrics_list in results[model_name].items():
                        # Take the first metrics entry (assuming there's only one per dataset)
                        if metrics_list:
                            metrics = metrics_list[0]
                            f1 = metrics.get('test_pixel-level F1', 'N/A')
                            auc = metrics.get('test_pixel-level AUC', 'N/A')
                            iou = metrics.get('test_pixel-level IOU', 'N/A')
                            acc = metrics.get('test_pixel-level Accuracy', 'N/A')

                            # Format values safely
                            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
                            auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else str(auc)
                            iou_str = f"{iou:.4f}" if isinstance(iou, (int, float)) else str(iou)
                            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)

                            f.write(f"{dataset_name.ljust(20)} {f1_str.rjust(12)} {auc_str.rjust(12)} {iou_str.rjust(12)} {acc_str.rjust(12)}\n")

                    f.write("\n")

if __name__ == "__main__":
    print("Extracting metrics from evaluation directories...")
    results = extract_all_metrics()
    print(f"Found metrics for {len(results)} models")

    print("Writing final report...")
    write_final_report(results)
    print("Done! Results saved to final.txt")
