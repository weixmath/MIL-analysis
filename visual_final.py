# -*- coding: utf-8 -*-
# file: visualize_attention_simple.py
"""
AttentionMIL模型可视化工具 (增强版)
=====================================
功能:
    1. Attention热力图叠加可视化
    2. Top-k高注意力patches保存
    3. 分类结果保存与Nature期刊级别可视化
       - ROC曲线
       - 混淆矩阵
       - Precision-Recall曲线
       - 类别指标条形图
       - 分类报告
"""
import os
import re
import json
import math
import argparse
import datetime
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
from scipy import ndimage
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# 可视化和评估相关导入
import matplotlib
matplotlib.use('Agg')  # 非GUI后端，适合服务器
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

import warnings
warnings.filterwarnings("ignore")

# ====================== Nature期刊风格配置 ======================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

# Nature期刊风格颜色
NATURE_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'quinary': '#3B1F2B',
    'success': '#4CAF50',
    'neutral': '#6C757D',
    'background': '#FAFAFA',
}
NATURE_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', 
                  '#9B59B6', '#1ABC9C', '#E74C3C', '#3498DB', '#2C3E50']

# ====================== 基础配置 ======================
PATCH_DATA_DIR = './patches_by_image'
INPUT_SIZE = 224
ALLOWED_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
GROUPS = {
    1: ['15-1', '30-1', '45-1', '100'],
    2: ['15-2', '30-2', '45-2', '100'],
    3: ['15-3', '30-3', '45-3', '100'],
    4: ['15-1', '30-1', '45-1', '15-2', '30-2', '45-2', '15-3', '30-3', '45-3', '100'],
    # 以下为不包含100类别的版本
    5: ['15-1', '30-1', '45-1'],                                                          # 对应 group 1
    6: ['15-2', '30-2', '45-2'],                                                          # 对应 group 2
    7: ['15-3', '30-3', '45-3'],                                                          # 对应 group 3
    8: ['15-1', '30-1', '45-1', '15-2', '30-2', '45-2', '15-3', '30-3', '45-3'],          # 对应 group 4
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ====================== 自定义collate函数 ======================
def custom_collate_fn(batch):
    """自定义collate函数，处理包含路径的batch"""
    batch = [item for item in batch if item[0] is not None]

    if len(batch) == 0:
        return None, None, None, None

    patches_list = []
    labels_list = []
    bag_names_list = []
    patch_paths_list = []

    for patches, label, bag_name, patch_paths in batch:
        patches_list.append(patches)
        labels_list.append(label)
        bag_names_list.append(bag_name)
        patch_paths_str = [str(p) for p in patch_paths]
        patch_paths_list.append(patch_paths_str)

    return patches_list, labels_list, bag_names_list, patch_paths_list


# ====================== 数据集定义 ======================
class VisualizationDataset(Dataset):
    """用于可视化的数据集，返回patch路径信息"""

    def __init__(self, root_dir, transform=None, class_names=None, exclude_classes=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.exclude_classes = set(exclude_classes) if exclude_classes else set()
        self.bags = []
        self.labels = []
        self.bag_names = []
        self.class_dirs = []

        print(f"正在从 {root_dir} 加载数据...")
        if self.exclude_classes:
            print(f"排除的类别: {self.exclude_classes}")
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_to_idx:
                # 跳过被排除的类别
                if class_dir.name in self.exclude_classes:
                    print(f"  跳过排除的类别: {class_dir.name}")
                    continue
                label = self.class_to_idx[class_dir.name]
                for bag_dir in class_dir.iterdir():
                    if bag_dir.is_dir():
                        patches = [p for p in bag_dir.glob('*') if p.suffix.lower() in ALLOWED_EXTS]
                        if patches:
                            self.bags.append(patches)
                            self.labels.append(label)
                            self.bag_names.append(f"{class_dir.name}_{bag_dir.name}")
                            self.class_dirs.append(class_dir.name)
        print(f"加载完成，共找到 {len(self.bags)} 个bags。")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        patch_paths = self.bags[idx]
        label = self.labels[idx]
        bag_name = self.bag_names[idx]

        patches = []
        valid_paths = []
        for patch_path in patch_paths:
            try:
                patch_img = Image.open(patch_path).convert('RGB')
                if self.transform:
                    patch_img = self.transform(patch_img)
                patches.append(patch_img)
                valid_paths.append(patch_path)
            except Exception as e:
                print(f"警告: 加载或转换图像失败 {patch_path}: {e}")
                continue

        if not patches:
            return None, None, None, None

        patches_tensor = torch.stack(patches)
        return patches_tensor, label, bag_name, valid_paths


# ====================== 模型定义 ======================
class AttentionMIL(nn.Module):
    def __init__(self, num_classes, backbone='convnext_tiny'):
        super().__init__()
        self.stage1_model = getattr(models, backbone)(weights=None)

        if 'convnext' in backbone:
            feat_dim = self.stage1_model.classifier[2].in_features
            self.stage1_model.classifier[2] = nn.Linear(feat_dim, num_classes)
        elif 'resnet' in backbone or 'resnext' in backbone:
            feat_dim = self.stage1_model.fc.in_features
            self.stage1_model.fc = nn.Linear(feat_dim, num_classes)
        else:
            try:
                feat_dim = self.stage1_model.classifier[-1].in_features
                self.stage1_model.classifier[-1] = nn.Linear(feat_dim, num_classes)
            except (AttributeError, IndexError):
                raise ValueError(f"无法自动确定 '{backbone}' 的分类头，请手动修改代码。")

        if 'convnext' in backbone:
            self.feature_extractor = nn.Sequential(*list(self.stage1_model.children())[:-1])
        elif 'resnet' in backbone or 'resnext' in backbone:
            self.feature_extractor = nn.Sequential(*list(self.stage1_model.children())[:-1])
        else:
            self.feature_extractor = self.stage1_model.features
        self.feat_dim = feat_dim

        self.attention_net = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward_with_attention(self, x):
        """返回预测结果和attention权重"""
        x = x.cuda()
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x)
            feats = feats.view(feats.size(0), -1)

        A_unnormalized = self.attention_net(feats)
        A = F.softmax(A_unnormalized, dim=0)
        M = torch.sum(A * feats, dim=0)
        logits = self.classifier(M.unsqueeze(0))

        return logits, A.squeeze().cpu().numpy()


# ====================== 热力图可视化函数 ======================
def create_smooth_attention_heatmap(attention_weights, patch_paths, patch_size=256,
                                    smooth_factor=2.5, interpolation_factor=6):
    """创建平滑的attention热力图"""
    coordinates = []
    max_x, max_y = 0, 0

    # 解析坐标
    for path in patch_paths:
        filename = Path(path).stem
        if filename.startswith('patch_'):
            parts = filename.split('_')
            if len(parts) == 3:
                try:
                    y, x = int(parts[1]), int(parts[2])
                    coordinates.append((x, y))
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                except ValueError:
                    coordinates.append((0, 0))
            else:
                coordinates.append((0, 0))
        else:
            coordinates.append((0, 0))

    # 计算原图尺寸
    original_width = max_x + patch_size
    original_height = max_y + patch_size

    # 创建网格坐标
    grid_width = original_width // patch_size
    grid_height = original_height // patch_size

    # 创建基础热力图
    base_heatmap = np.zeros((grid_height, grid_width))

    for i, (x, y) in enumerate(coordinates):
        grid_x = x // patch_size
        grid_y = y // patch_size
        if 0 <= grid_y < grid_height and 0 <= grid_x < grid_width:
            base_heatmap[grid_y, grid_x] = attention_weights[i]

    # 高分辨率插值
    high_res_height = grid_height * interpolation_factor
    high_res_width = grid_width * interpolation_factor

    # 使用双三次插值进行上采样
    smooth_heatmap = cv2.resize(base_heatmap, (high_res_width, high_res_height),
                                interpolation=cv2.INTER_CUBIC)

    # 应用高斯平滑
    smooth_heatmap = ndimage.gaussian_filter(smooth_heatmap, sigma=smooth_factor)

    # 调整到原图尺寸
    final_heatmap = cv2.resize(smooth_heatmap, (original_width, original_height),
                               interpolation=cv2.INTER_CUBIC)

    return final_heatmap


def create_plasma_overlay(image_path, heatmap, alpha=0.6):
    """创建plasma风格的热力图叠加，使用更高透明度"""
    try:
        # 读取原始图像
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            print(f"无法读取图像: {image_path}")
            return None

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # 调整热力图大小以匹配原始图像
        if original_img.shape[:2] != heatmap.shape:
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # 归一化热力图到0-1范围
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # 伽马校正增强对比度
        heatmap_norm = np.power(heatmap_norm, 1.0 / 1.3)

        # 应用plasma颜色映射
        heatmap_colored = plt.cm.plasma(heatmap_norm)[:, :, :3]  # 去掉alpha通道
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # 创建mask，只在有attention的区域显示热力图
        mask = heatmap_norm > 0.01  # 阈值可调

        # 叠加图像 - 使用更高的透明度
        overlay = original_img.copy().astype(np.float32)
        heatmap_colored = heatmap_colored.astype(np.float32)

        # 使用mask进行选择性叠加
        for c in range(3):
            overlay[:, :, c] = np.where(mask,
                                        (1 - alpha) * overlay[:, :, c] + alpha * heatmap_colored[:, :, c],
                                        overlay[:, :, c])

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return None


def save_top_attention_patches_with_scores(patch_paths, attention_weights, output_dir, bag_name, top_k=20):
    """保存top-k高/低attention权重的patch，文件名包含attention系数
    
    返回:
        tuple: (top_high_info, top_low_info)
    """
    # 创建子目录
    top_high_dir = ensure_dir(os.path.join(output_dir, 'top_high'))
    top_low_dir = ensure_dir(os.path.join(output_dir, 'top_low'))
    
    # Top-k highest attention
    top_high_indices = np.argsort(attention_weights)[-top_k:][::-1]
    top_high_info = []
    
    for rank, idx in enumerate(top_high_indices):
        patch_path = patch_paths[idx]
        attention_score = attention_weights[idx]

        try:
            patch_img = Image.open(patch_path).convert('RGB')
            original_name = Path(patch_path).stem

            output_filename = f"high_rank{rank + 1:02d}_att{attention_score:.6f}_{original_name}.png"
            output_path = os.path.join(top_high_dir, output_filename)
            patch_img.save(output_path)

            top_high_info.append({
                'rank': rank + 1,
                'attention': float(attention_score),
                'original_name': original_name,
                'saved_path': output_filename
            })

        except Exception as e:
            print(f"保存patch时出错 {patch_path}: {e}")

    # Top-k lowest attention
    top_low_indices = np.argsort(attention_weights)[:top_k]  # 从小到大排序，取前top_k
    top_low_info = []
    
    for rank, idx in enumerate(top_low_indices):
        patch_path = patch_paths[idx]
        attention_score = attention_weights[idx]

        try:
            patch_img = Image.open(patch_path).convert('RGB')
            original_name = Path(patch_path).stem

            output_filename = f"low_rank{rank + 1:02d}_att{attention_score:.6f}_{original_name}.png"
            output_path = os.path.join(top_low_dir, output_filename)
            patch_img.save(output_path)

            top_low_info.append({
                'rank': rank + 1,
                'attention': float(attention_score),
                'original_name': original_name,
                'saved_path': output_filename
            })

        except Exception as e:
            print(f"保存patch时出错 {patch_path}: {e}")

    return top_high_info, top_low_info


def find_original_image(bag_name, original_images_dir):
    """查找对应的原始图像"""
    parts = bag_name.split('_', 1)
    if len(parts) == 2:
        class_name, image_name = parts

        for ext in ['.tif', '.tiff', '.bmp', '.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(original_images_dir, class_name, f"{image_name}{ext}")
            if os.path.exists(potential_path):
                return potential_path

    return None


# ====================== Nature期刊风格可视化函数 ======================

def plot_roc_curves(y_true, y_probs, class_names, output_dir, run_name):
    """绘制多类别ROC曲线 (Nature期刊风格)"""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_probs = np.array(y_probs)
    
    # 处理二分类情况
    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    
    colors = cycle(NATURE_PALETTE[:n_classes])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')
    
    ax.plot(fpr["macro"], tpr["macro"], color='#1a1a1a', lw=2.5, linestyle='-.',
            label=f'Macro-avg (AUC={roc_auc["macro"]:.3f})')
    
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontweight='medium')
    ax.set_ylabel('True Positive Rate', fontweight='medium')
    ax.set_title('ROC Curves', fontweight='bold', pad=10)
    ax.legend(loc='lower right', frameon=True, fontsize=7 if n_classes > 5 else 8)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(os.path.join(output_dir, f'{run_name}_roc_curves.{fmt}'), format=fmt, dpi=300)
    plt.close()
    print("✓ ROC曲线已保存")
    return roc_auc


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, run_name, normalize=True):
    """绘制混淆矩阵 (Nature期刊风格)"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = sns.light_palette(NATURE_COLORS['primary'], as_cmap=True)
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, aspect='auto')
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    
    thresh = (cm_display.max() + cm_display.min()) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f'{cm_display[i, j]:.1%}' if normalize else f'{cm[i, j]}'
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                   color="white" if cm_display[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted Label', fontweight='medium')
    ax.set_ylabel('True Label', fontweight='medium')
    ax.set_title(title, fontweight='bold', pad=10)
    
    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(os.path.join(output_dir, f'{run_name}_confusion_matrix.{fmt}'), format=fmt, dpi=300)
    plt.close()
    print("✓ 混淆矩阵已保存")
    return cm


def plot_precision_recall_curves(y_true, y_probs, class_names, output_dir, run_name):
    """绘制Precision-Recall曲线 (Nature期刊风格)"""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_probs = np.array(y_probs)
    
    # 处理二分类情况
    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
    
    precision = dict()
    recall = dict()
    ap = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    
    fig, ax = plt.subplots(figsize=(5.5, 5))
    
    colors = cycle(NATURE_PALETTE[:n_classes])
    for i, color in zip(range(n_classes), colors):
        ax.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_names[i]} (AP={ap[i]:.3f})')
    
    mean_ap = np.mean(list(ap.values()))
    ax.axhline(y=mean_ap, color='#1a1a1a', linestyle='-.', lw=1.5,
               label=f'Mean AP={mean_ap:.3f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontweight='medium')
    ax.set_ylabel('Precision', fontweight='medium')
    ax.set_title('Precision-Recall Curves', fontweight='bold', pad=10)
    ax.legend(loc='lower left', fontsize=7 if n_classes > 5 else 8)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(os.path.join(output_dir, f'{run_name}_pr_curves.{fmt}'), format=fmt, dpi=300)
    plt.close()
    print("✓ Precision-Recall曲线已保存")
    return ap


def plot_class_metrics_bar(y_true, y_pred, class_names, output_dir, run_name):
    """绘制每个类别的性能指标条形图 (Nature期刊风格)"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(7, len(class_names)*0.8), 4.5))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color=NATURE_COLORS['primary'], alpha=0.9)
    bars2 = ax.bar(x, recall, width, label='Recall', color=NATURE_COLORS['secondary'], alpha=0.9)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color=NATURE_COLORS['tertiary'], alpha=0.9)
    
    ax.set_ylabel('Score', fontweight='medium')
    ax.set_xlabel('Class', fontweight='medium')
    ax.set_title('Per-class Classification Metrics', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(os.path.join(output_dir, f'{run_name}_class_metrics.{fmt}'), format=fmt, dpi=300)
    plt.close()
    print("✓ 类别指标条形图已保存")
    return precision, recall, f1


def generate_classification_report_file(y_true, y_pred, class_names, output_dir, run_name):
    """生成分类报告"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    
    report_path = os.path.join(output_dir, f'{run_name}_classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("       AttentionMIL Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✓ 分类报告已保存到: {report_path}")
    return report


def save_classification_results(results, output_dir, run_name):
    """保存分类结果到JSON"""
    results_path = os.path.join(output_dir, f'{run_name}_classification_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 分类结果已保存到: {results_path}")
    return results_path


def create_bag_summary_figure(bag_name, class_names, probabilities, attention_scores, 
                              patch_paths, output_dir, true_label=None, pred_label=None,
                              heatmap=None, overlay_image=None, top_k=6):
    """生成包含概率、热力图与Top patch缩略图的综合图像 (Nature期刊风格)"""
    probs = np.array(probabilities, dtype=float)
    order = np.argsort(probs)[::-1]

    display_true = int(true_label) if true_label is not None else None
    display_pred = int(pred_label) if pred_label is not None else None

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, 1])

    ax_prob = fig.add_subplot(gs[:, 0])
    ordered_probs = probs[order]
    ordered_labels = [class_names[i] for i in order]

    bar_colors = []
    for idx in order:
        if display_true is not None and display_pred is not None and idx == display_true == display_pred:
            bar_colors.append(NATURE_COLORS['success'])
        elif display_true is not None and idx == display_true:
            bar_colors.append(NATURE_COLORS['secondary'])
        elif display_pred is not None and idx == display_pred:
            bar_colors.append(NATURE_COLORS['tertiary'])
        else:
            bar_colors.append(NATURE_COLORS['neutral'])

    ax_prob.barh(range(len(order)), ordered_probs, color=bar_colors)
    ax_prob.set_yticks(range(len(order)))
    ax_prob.set_yticklabels(ordered_labels, fontsize=8)
    ax_prob.invert_yaxis()
    ax_prob.set_xlabel('Probability')
    ax_prob.set_xlim(0, 1)
    ax_prob.grid(axis='x', linestyle='--', alpha=0.3)
    ax_prob.set_title('Bag-level Probabilities', fontweight='bold')

    ax_heat = fig.add_subplot(gs[0, 1:])
    if overlay_image is not None:
        ax_heat.imshow(overlay_image)
        ax_heat.set_title('Attention Overlay', fontweight='bold')
        ax_heat.axis('off')
    elif heatmap is not None:
        hm = heatmap.astype(float)
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        ax_heat.imshow(hm, cmap='plasma')
        ax_heat.set_title('Attention Heatmap', fontweight='bold')
        ax_heat.axis('off')
    else:
        ax_heat.axis('off')
        ax_heat.text(0.5, 0.5, '无热力图', ha='center', va='center')

    top_show = min(top_k, len(patch_paths))
    if top_show > 0:
        cols = min(3, top_show)
        rows = math.ceil(top_show / cols)
        subgs = gs[1, 1:].subgridspec(rows, cols)
        sorted_indices = np.argsort(attention_scores)[-top_show:][::-1]
        for idx, patch_idx in enumerate(sorted_indices):
            ax_patch = fig.add_subplot(subgs[idx])
            try:
                img = Image.open(patch_paths[patch_idx]).convert('RGB')
                ax_patch.imshow(img)
            except Exception:
                ax_patch.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
            att_val = float(attention_scores[patch_idx])
            ax_patch.set_title(f'#{idx + 1} | {att_val:.3f}', fontsize=7)
            ax_patch.axis('off')
    else:
        ax_empty = fig.add_subplot(gs[1, 1:])
        ax_empty.axis('off')
        ax_empty.text(0.5, 0.5, '无可用Patch', ha='center', va='center')

    true_txt = class_names[display_true] if display_true is not None else 'N/A'
    pred_txt = class_names[display_pred] if display_pred is not None else 'N/A'
    fig.suptitle(f"{bag_name} | True: {true_txt} | Pred: {pred_txt}", fontweight='bold')

    fig.tight_layout()
    summary_path = os.path.join(output_dir, 'bag_summary.png')
    fig.savefig(summary_path, dpi=300)
    plt.close(fig)
    return summary_path


# ====================== 主函数 ======================
def visualize_attention_maps(model_path, data_dir, output_dir, group, backbone, original_images_dir=None,
                             batch_size=1, top_k=20, max_bags=None, alpha=0.6, run_name=None,
                             exclude_classes=None):
    """可视化attention maps - 增强版本（包含Nature期刊级别分类可视化）
    
    Args:
        exclude_classes: 要排除的类别列表（如 ['100']），这些类别的数据不会被加载和统计，
                        但模型仍使用原来的类别数加载
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    classes_for_group = GROUPS[group]
    classes_this_run = sorted(classes_for_group)
    num_classes = len(classes_this_run)

    print(f"分类组 {group}: {classes_this_run}")

    # 创建输出目录
    ensure_dir(output_dir)
    
    # 生成运行名称
    if run_name is None:
        run_name = f"attmil_vis_{backbone}_group{group}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = AttentionMIL(num_classes=num_classes, backbone=backbone)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("模型加载完成")

    # 准备数据变换
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集（排除指定类别）
    dataset = VisualizationDataset(
        root_dir=data_dir,
        transform=transform,
        class_names=classes_this_run,
        exclude_classes=exclude_classes
    )
    
    # 计算实际参与评估的类别（用于可视化）
    if exclude_classes:
        eval_class_names = [c for c in classes_this_run if c not in exclude_classes]
        print(f"实际评估的类别: {eval_class_names}")
        # 创建从原始标签索引到新评估标签索引的映射
        old_to_new_label = {}
        new_idx = 0
        for old_idx, cls_name in enumerate(classes_this_run):
            if cls_name not in exclude_classes:
                old_to_new_label[old_idx] = new_idx
                new_idx += 1
        print(f"标签映射: {old_to_new_label}")
    else:
        eval_class_names = classes_this_run
        old_to_new_label = {i: i for i in range(num_classes)}
    
    num_eval_classes = len(eval_class_names)

    if max_bags:
        dataset.bags = dataset.bags[:max_bags]
        dataset.labels = dataset.labels[:max_bags]
        dataset.bag_names = dataset.bag_names[:max_bags]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=custom_collate_fn)

    print(f"开始处理 {len(dataset)} 个bags...")

    # 收集所有推理结果
    all_labels = []
    all_preds = []
    all_probs = []
    all_bag_names = []
    all_attention_weights = []
    all_patch_paths = []
    
    results_summary = []

    for batch_data in tqdm(dataloader, desc="处理bags"):
        if batch_data[0] is None:
            continue

        bags, labels, bag_names, patch_paths_list = batch_data

        for i in range(len(bags)):
            bag = bags[i]
            label = labels[i]
            bag_name = bag_names[i]
            patch_paths = patch_paths_list[i]

            print(f"\n处理 bag: {bag_name} (真实标签: {classes_this_run[label]})")

            # 为每个bag创建专门的目录
            bag_output_dir = ensure_dir(os.path.join(output_dir, 'attention_maps', bag_name))

            # 获取预测结果和attention权重
            with torch.no_grad():
                logits, attention_weights = model.forward_with_attention(bag)
                predicted_class = torch.argmax(logits, dim=1).item()
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                confidence = probs.max()

            print(f"预测类别: {classes_this_run[predicted_class]} (置信度: {confidence:.4f})")
            print(f"Attention权重范围: [{attention_weights.min():.6f}, {attention_weights.max():.6f}]")
            print(f"Patch数量: {len(attention_weights)}")

            # 收集用于整体评估的数据（使用映射后的标签）
            mapped_label = old_to_new_label[label]
            mapped_pred = old_to_new_label.get(predicted_class, -1)  # 如果预测到被排除的类，标记为-1
            
            # 提取评估类别的概率（排除被排除类别的概率）
            if exclude_classes:
                eval_probs = np.array([probs[i] for i in range(len(classes_this_run)) 
                                       if classes_this_run[i] not in exclude_classes])
                # 重新归一化
                eval_probs = eval_probs / (eval_probs.sum() + 1e-8)
            else:
                eval_probs = probs
            
            all_labels.append(mapped_label)
            all_preds.append(mapped_pred if mapped_pred >= 0 else 0)  # 如果预测到排除类，默认取第一个
            all_probs.append(eval_probs)
            all_bag_names.append(bag_name)
            all_attention_weights.append(attention_weights)
            all_patch_paths.append(patch_paths)

            # 保存top-k attention patches（高和低，带attention系数）
            top_high_info, top_low_info = save_top_attention_patches_with_scores(
                patch_paths, attention_weights, bag_output_dir, bag_name, top_k
            )

            # 创建热力图和叠加图
            smooth_heatmap = create_smooth_attention_heatmap(
                attention_weights, patch_paths, smooth_factor=2.5, interpolation_factor=6
            )
            
            overlay_img = None
            
            # 如果提供了原始图像目录，创建plasma叠加图
            if original_images_dir:
                original_image_path = find_original_image(bag_name, original_images_dir)
                if original_image_path:
                    print(f"找到原始图像: {original_image_path}")

                    # 创建plasma叠加图
                    overlay_img = create_plasma_overlay(original_image_path, smooth_heatmap, alpha=alpha)

                    if overlay_img is not None:
                        # 保存叠加图到同一目录
                        overlay_output_path = os.path.join(
                            bag_output_dir,
                            f"{bag_name}_plasma_alpha{alpha:.1f}_overlay.png"
                        )

                        # 直接保存叠加图像
                        overlay_pil = Image.fromarray(overlay_img)
                        overlay_pil.save(overlay_output_path, dpi=(300, 300))

                        print(f"保存plasma叠加图: {overlay_output_path}")
                else:
                    print(f"未找到原始图像: {bag_name}")

            # 生成bag综合摘要图
            summary_path = create_bag_summary_figure(
                bag_name=bag_name,
                class_names=classes_this_run,
                probabilities=probs,
                attention_scores=attention_weights,
                patch_paths=patch_paths,
                output_dir=bag_output_dir,
                true_label=label,
                pred_label=predicted_class,
                heatmap=smooth_heatmap,
                overlay_image=overlay_img,
                top_k=min(6, top_k)
            )

            # 保存详细信息到文本文件
            info_file_path = os.path.join(bag_output_dir, f"{bag_name}_info.txt")
            with open(info_file_path, 'w', encoding='utf-8') as f:
                f.write(f"样本信息: {bag_name}\n")
                f.write(f"真实标签: {classes_this_run[label]}\n")
                f.write(f"预测标签: {classes_this_run[predicted_class]}\n")
                f.write(f"预测置信度: {confidence:.4f}\n")
                f.write(f"预测正确: {'✓' if label == predicted_class else '✗'}\n\n")
                
                f.write(f"各类别概率:\n")
                for j, cls_name in enumerate(classes_this_run):
                    f.write(f"• {cls_name}: {probs[j]:.4f}\n")
                f.write("\n")

                f.write(f"Attention统计:\n")
                f.write(f"• 最大值: {attention_weights.max():.6f}\n")
                f.write(f"• 最小值: {attention_weights.min():.6f}\n")
                f.write(f"• 均值: {attention_weights.mean():.6f}\n")
                f.write(f"• 标准差: {attention_weights.std():.6f}\n")
                f.write(f"• Patch数量: {len(attention_weights)}\n\n")

                f.write(f"热力图参数:\n")
                f.write(f"• 平滑因子: 2.5\n")
                f.write(f"• 插值倍数: 6x\n")
                f.write(f"• 透明度: {alpha}\n")
                f.write(f"• 颜色映射: plasma\n\n")

                f.write(f"Top-{top_k} Highest Attention Patches:\n")
                for patch in top_high_info:
                    f.write(
                        f"• Rank {patch['rank']:2d}: {patch['attention']:.6f} - {patch['saved_path']}\n")
                
                f.write(f"\nTop-{top_k} Lowest Attention Patches:\n")
                for patch in top_low_info:
                    f.write(
                        f"• Rank {patch['rank']:2d}: {patch['attention']:.6f} - {patch['saved_path']}\n")

            # 记录结果
            results_summary.append({
                'bag_name': bag_name,
                'true_label': classes_this_run[label],
                'predicted_label': classes_this_run[predicted_class],
                'confidence': float(confidence),
                'probabilities': {cls_name: float(probs[j]) for j, cls_name in enumerate(classes_this_run)},
                'correct': label == predicted_class,
                'max_attention': float(attention_weights.max()),
                'min_attention': float(attention_weights.min()),
                'mean_attention': float(attention_weights.mean()),
                'std_attention': float(attention_weights.std()),
                'num_patches': len(attention_weights),
                'output_dir': bag_output_dir
            })

    # ====================== Nature期刊级别可视化 ======================
    print("\n" + "=" * 60)
    print(" 生成Nature期刊级别可视化...")
    print("=" * 60)
    
    # 构建详细分类结果
    detailed_results = {
        'run_name': run_name,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'class_names': eval_class_names,  # 使用评估类别
        'excluded_classes': exclude_classes if exclude_classes else [],
        'num_samples': len(all_labels),
        'samples': results_summary
    }
    
    # 计算整体准确率
    overall_acc = accuracy_score(all_labels, all_preds)
    detailed_results['overall_accuracy'] = float(overall_acc)
    
    # 保存分类结果JSON
    save_classification_results(detailed_results, output_dir, run_name)
    
    # 生成ROC曲线（使用评估类别）
    try:
        roc_auc = plot_roc_curves(all_labels, all_probs, eval_class_names, output_dir, run_name)
        mean_auc = np.mean([roc_auc[i] for i in range(num_eval_classes)])
    except Exception as e:
        print(f"警告: ROC曲线生成失败: {e}")
        roc_auc = {}
        mean_auc = 0.0
    
    # 生成混淆矩阵（使用评估类别）
    try:
        cm = plot_confusion_matrix(all_labels, all_preds, eval_class_names, output_dir, run_name)
    except Exception as e:
        print(f"警告: 混淆矩阵生成失败: {e}")
        cm = None
    
    # 生成Precision-Recall曲线（使用评估类别）
    try:
        ap = plot_precision_recall_curves(all_labels, all_probs, eval_class_names, output_dir, run_name)
    except Exception as e:
        print(f"警告: Precision-Recall曲线生成失败: {e}")
        ap = {}
    
    # 生成类别指标条形图（使用评估类别）
    try:
        prec, rec, f1 = plot_class_metrics_bar(all_labels, all_preds, eval_class_names, output_dir, run_name)
    except Exception as e:
        print(f"警告: 类别指标条形图生成失败: {e}")
        prec, rec, f1 = None, None, None
    
    # 生成分类报告（使用评估类别）
    report = generate_classification_report_file(all_labels, all_preds, eval_class_names, output_dir, run_name)
    
    # 计算汇总统计
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 打印统计信息
    total_bags = len(results_summary)
    correct_predictions = sum(1 for r in results_summary if r['correct'])
    accuracy = correct_predictions / total_bags if total_bags > 0 else 0

    print(f"\n" + "=" * 60)
    print(f"🎉 可视化完成!")
    print(f"=" * 60)
    print(f"📊 总处理bags: {total_bags}")
    print(f"🎯 预测准确率: {accuracy:.4f} ({correct_predictions}/{total_bags})")
    print(f"📁 结果保存在: {output_dir}")
    print(f"\n生成的文件:")
    print(f"   • 📄 {run_name}_classification_results.json")
    print(f"   • 📊 {run_name}_roc_curves.[pdf/png/svg]")
    print(f"   • 📊 {run_name}_confusion_matrix.[pdf/png/svg]")
    print(f"   • 📊 {run_name}_pr_curves.[pdf/png/svg]")
    print(f"   • 📊 {run_name}_class_metrics.[pdf/png/svg]")
    print(f"   • 📝 {run_name}_classification_report.txt")
    print(f"   • 📁 attention_maps/ (每个样本的plasma叠加图和top-{top_k} patches)")
    print(f"\n" + "=" * 60)
    print(f" 最终性能汇总")
    print(f"=" * 60)
    print(f"   • Overall Accuracy: {overall_acc:.4f}")
    print(f"   • Macro F1-Score:   {macro_f1:.4f}")
    print(f"   • Mean AUC:         {mean_auc:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AttentionMIL可视化 - 增强版（含Nature期刊级别分类可视化）")
    parser.add_argument('--model_path', type=str, required=True,
                        help="训练好的模型权重文件路径")
    parser.add_argument('--data_dir', type=str, default='./patches_by_image/val',
                        help="patch数据目录")
    parser.add_argument('--original_images_dir', type=str, default=None,
                        help="原始图像目录（用于生成热力图叠加）")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="输出目录")
    parser.add_argument('--group', type=int, required=True, choices=[1, 2, 3, 4],
                        help="分类组ID")
    parser.add_argument('--backbone', type=str, default='convnext_tiny',
                        help="backbone模型名称")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="批处理大小")
    parser.add_argument('--top_k', type=int, default=20,
                        help="保存top-k高attention权重的patch")
    parser.add_argument('--max_bags', type=int, default=None,
                        help="最大处理的bag数量（用于调试）")
    parser.add_argument('--alpha', type=float, default=0.6,
                        help="热力图透明度 (0-1)，越大越不透明")
    parser.add_argument('--run_name', type=str, default=None,
                        help="运行名称（用于命名输出文件）")
    parser.add_argument('--exclude_classes', type=str, nargs='+', default=None,
                        help="要排除的类别列表，如 --exclude_classes 100，模型仍用原类别数加载但这些类别不参与统计")

    args = parser.parse_args()

    visualize_attention_maps(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        group=args.group,
        backbone=args.backbone,
        original_images_dir=args.original_images_dir,
        batch_size=args.batch_size,
        top_k=args.top_k,
        max_bags=args.max_bags,
        alpha=args.alpha,
        run_name=args.run_name,
        exclude_classes=args.exclude_classes
    )


if __name__ == '__main__':
    main()
