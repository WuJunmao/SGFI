#!/usr/bin/env python3
import os
import json
import glob
from collections import defaultdict

def extract_gaussian_noise_results():
    # Find all log.txt files in the robust_dir
    log_files = glob.glob('/home/lab301-3090/wujun/benco/robust_dir/**/log.txt', recursive=True)

    # Organize results by model and test type (only Gaussian Noise)
    results = defaultdict(lambda: defaultdict(dict))

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)

            # Extract model name and test type from path
            parts = log_file.split('/')
            model_name = parts[6]  # robust_dir/model_name/test_type/log.txt
            test_type = parts[7] if len(parts) > 7 else "None"

            # Only process Gaussian Noise tests
            if "GaussNoise" in test_type:
                # Store the results
                results[model_name][test_type]['IOU'] = data.get('test_pixel-level IOU', 'N/A')
                results[model_name][test_type]['epoch'] = data.get('epoch', 'N/A')

        except Exception as e:
            print(f"Error processing {log_file}: {e}")

    # Generate the report
    report_lines = []
    report_lines.append("=== Gaussian Noise Robustness Test Results ===\n")

    # Sort models alphabetically
    sorted_models = sorted(results.keys())

    for model_name in sorted_models:
        report_lines.append(f"\n=== Model: {model_name} ===")
        test_types = sorted(results[model_name].keys())

        for test_type in test_types:
            test_data = results[model_name][test_type]
            report_lines.append(f"Test: {test_type}")
            report_lines.append(f"  IOU: {test_data['IOU']}")
            report_lines.append(f"  Epoch: {test_data['epoch']}")
            report_lines.append("")

    # Save to robust111.txt
    with open('robust111.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Successfully extracted Gaussian Noise results from {len([f for f in log_files if 'GaussNoise' in f])} log files")
    print(f"Report saved to robust111.txt")
    return len([f for f in log_files if 'GaussNoise' in f])

if __name__ == "__main__":
    extract_gaussian_noise_results()
