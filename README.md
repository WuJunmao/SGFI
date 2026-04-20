# SGFI: Stage-wise Gated Forensic Prior Injection for Image Manipulation Localization

This repository contains the official implementation of the paper:

> **Stage-wise Gated Forensic Prior Injection with Training-only Deep Supervision for Image Manipulation Localization**  
> Huihuang Zhao, Jun Wu  
> *Submitted to Image and Vision Computing*

## Overview

Image Manipulation Localization (IML) aims to predict pixel-level masks of manipulated regions. This work introduces **SGFI**, a lightweight extension framework built on SparseViT, with two key designs:

1. **Stage-wise Gated Residual Fusion** – Injects noise-sensitive Noiseprint++ prior into each encoder stage through spatially adaptive gates, preserving weak forensic traces throughout the network.
2. **Training-only Deep Supervision** – Attaches an auxiliary supervision head to Stage 3 during training to provide direct constraints on intermediate representations. The auxiliary branch is removed during inference, incurring no additional deployment cost.

Additionally, a conservative polarity-aware heuristic is applied during evaluation to mitigate instability caused by mask inversion in threshold-based metrics.

## Framework Overview

![Framework](figs/Fig3.png)

## Results

Under the CAT-Net training protocol, SGFI achieves state-of-the-art performance on five public benchmarks:

| Method | COVERAGE | Columbia | NIST16 | COCO-Glide | Realistic | **AVG F1** |
|--------|----------|----------|--------|------------|-----------|-----------|
| SparseViT (baseline) | 0.502 | 0.922 | 0.376 | 0.361 | 0.205 | 0.473 |
| **SGFI (Ours)** | 0.548 | 0.946 | 0.383 | 0.407 | 0.228 | **0.502** |

- **F1 gain**: +0.029
- **IoU gain**: +0.027

For detailed quantitative and qualitative results, please refer to the paper.

## Datasets

### Training Data (following CAT-Net protocol)
- [CASIA v2.0](https://www.kaggle.com/datasets/casia)
- [IMD2020](https://www.kaggle.com/datasets/imbikramsaha/imd2020)
- MS COCO (used for splicing set construction)

### Testing Benchmarks
- [COVERAGE](https://github.com/wenbihan/coverage)
- [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/)
- [NIST16](https://www.nist.gov/itl/iad/mig/nist-16-deepfake-media-forensics-challenge)
- COCO-Glide
- [Realistic Tampering](https://github.com/ymnis/Realistic-Tampering-Dataset)

> **Note**: CASIA v1.0 is excluded from testing because the training set includes CASIA v2.0, and the two datasets share similar content patterns.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- opencv-python
- scikit-learn
- matplotlib
- tqdm

## Installation

```bash
git clone https://github.com/WuJunmao/SGFI.git
cd SGFI
pip install -r requirements.txt
