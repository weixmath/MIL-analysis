# MIL-Based Deep Learning for TEM Micrograph Analysis

This repository contains a Two-Stage Multiple Instance Learning (MIL) framework designed for the classification and analysis of Transmission Electron Microscopy (TEM) images (specifically CuO catalyst states).

The pipeline features:
1.  **Stage 1:** Self-supervised/Supervised backbone training on image patches.
2.  **Stage 2:** MIL Aggregator training (AttentionMIL or TransMIL) using frozen backbones.
3.  **Validation:** Integrated "Label Permutation Test" to detect data leakage.
4.  **Visualization:** "Nature-journal" style plotting (ROC, Confusion Matrices) and Attention Heatmaps with Plasma overlay.

## 🛠️ Requirements

* Python 3.8+
* PyTorch (CUDA supported)
* Torchvision
* Scikit-learn
* Matplotlib & Seaborn
* OpenCV
* Pillow

Install dependencies:
```bash
pip install -r requirements.txt
