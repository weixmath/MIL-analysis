# file: train_final_stage2_v4_fixed.py
# =====================================================================
# Stage 2 MIL 训练脚本 - 支持标签随机置换实验 (Label Permutation Test)
# 使用 --shuffle-labels 参数启用标签置换，验证模型是否学到真实信号
# 重要：shuffle-seed 需要与 Stage 1 保持一致，确保两阶段标签置换一致
# =====================================================================
import os
import re
import json
import argparse
import datetime
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 可视化和评估相关导入
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

# 设置Nature期刊风格的matplotlib样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
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
})

# ====================== 基础配置 (与v4相同) ======================
PATCH_DATA_DIR = './patches_by_image'
INPUT_SIZE = 224
ALLOWED_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
NUM_WORKERS = min(8, os.cpu_count() or 0)
PIN_MEMORY = torch.cuda.is_available()
GROUPS = {
    1: ['15-1', '30-1', '45-1', '100'],
    2: ['15-2', '30-2', '45-2', '100'],
    3: ['15-3', '30-3', '45-3', '100'],
    4: ['15-1', '30-1', '45-1', '15-2', '30-2', '45-2', '15-3', '30-3', '45-3', '100'],
}


# ====================== 工具函数 (与v4相同) ======================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def init_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


# ====================== 数据集定义 (与v4相同) ======================
class MILDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_names=None, rank=0):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.bags = []
        self.labels = []

        if rank == 0: print(f"正在从 {root_dir} 加载数据...")
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_to_idx:
                label = self.class_to_idx[class_dir.name]
                for bag_dir in class_dir.iterdir():
                    if bag_dir.is_dir():
                        patches = [p for p in bag_dir.glob('*') if p.suffix.lower() in ALLOWED_EXTS]
                        if patches:
                            self.bags.append(patches)
                            self.labels.append(label)
        if rank == 0: print(f"加载完成，共找到 {len(self.bags)} 个bags。")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        patch_paths = self.bags[idx]
        label = self.labels[idx]

        patches = []
        for patch_path in patch_paths:
            try:
                patch_img = Image.open(patch_path).convert('RGB')
                if self.transform:
                    patch_img = self.transform(patch_img)
                patches.append(patch_img)
            except Exception as e:
                print(f"警告: 加载或转换图像失败 {patch_path}: {e}")
                continue

        if not patches: return None, None

        patches_tensor = torch.stack(patches)
        return patches_tensor, label


class PatchDatasetForVoting(Dataset):
    def __init__(self, root_dir, transform=None, class_names=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_to_idx:
                label = self.class_to_idx[class_dir.name]
                for bag_dir in class_dir.iterdir():
                    if bag_dir.is_dir():
                        bag_id = bag_dir.name
                        for patch_path in bag_dir.glob('*'):
                            if patch_path.suffix.lower() in ALLOWED_EXTS:
                                samples.append((patch_path, label, bag_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_path, label, bag_id = self.samples[idx]
        patch_img = Image.open(patch_path).convert('RGB')
        if self.transform:
            patch_img = self.transform(patch_img)
        return patch_img, label, bag_id


def mil_collate_fn(batch):
    batch = [(data, label) for data, label in batch if data is not None]
    if not batch: return None, None
    data, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return data, labels


# ====================== 标签随机置换实验 (Label Permutation Test) ======================
def shuffle_bag_labels_mil(dataset: MILDataset, seed: int = 42) -> Dict[str, int]:
    """
    对 MIL 训练集进行 bag-level 的标签随机置换。
    
    核心思想：
    - 每个 bag 是一个完整的 micrograph/image 目录
    - 打乱是在 bag 级别进行的，即把所有 bag 的标签做一个随机 permutation
    - 使用 bag 目录名作为标识符，确保两阶段使用相同 seed 时一致
    
    【关键】为了确保 Stage 1 和 Stage 2 使用相同 seed 时结果一致：
    - 使用 bag 目录名进行稳定排序后再打乱
    - 这样两阶段的 bag 顺序一致，打乱结果也一致
    
    Args:
        dataset: MILDataset 对象
        seed: 随机种子，确保可复现性，必须与 Stage 1 一致
    
    Returns:
        bag_to_shuffled: bag目录名到打乱后标签的映射（用于日志记录）
    """
    # 1. 提取所有 bag 的目录名、原始标签和索引
    bag_info = []  # [(bag_name, original_label, dataset_idx), ...]
    
    for idx, bag_paths in enumerate(dataset.bags):
        # bag_paths 是一个 Path 列表，取第一个 patch 的父目录名作为 bag 标识
        bag_dir_name = bag_paths[0].parent.name
        bag_info.append((bag_dir_name, dataset.labels[idx], idx))
    
    # 2. 按 bag 目录名稳定排序（与 Stage 1 保持一致的排序方式）
    bag_info_sorted = sorted(bag_info, key=lambda x: x[0])
    
    # 3. 提取排序后的原始标签并执行随机置换
    original_labels_sorted = [info[1] for info in bag_info_sorted]
    rng = random.Random(seed)
    shuffled_labels = original_labels_sorted.copy()
    rng.shuffle(shuffled_labels)
    
    # 4. 构建映射：bag 名称 -> 打乱后标签
    bag_to_shuffled = {}
    for (bag_name, _, _), new_label in zip(bag_info_sorted, shuffled_labels):
        bag_to_shuffled[bag_name] = new_label
    
    # 5. 更新 dataset 中的标签（使用 bag_to_shuffled 映射）
    for idx, bag_paths in enumerate(dataset.bags):
        bag_dir_name = bag_paths[0].parent.name
        dataset.labels[idx] = bag_to_shuffled[bag_dir_name]
    
    return bag_to_shuffled


def log_label_shuffle_info_stage2(bag_to_shuffled: Dict[str, int], 
                                   dataset: MILDataset, 
                                   output_dir: str):
    """
    记录 Stage 2 标签打乱的详细信息到文件中。
    """
    log_path = os.path.join(output_dir, 'label_shuffle_log_stage2.txt')
    
    # 反向构建，获取原始标签信息
    # 由于打乱后我们无法直接知道原始标签，需要从目录结构推断
    class_names = dataset.class_names
    
    # 从 bags 路径推断原始标签
    bag_to_original = {}
    for bag_paths in dataset.bags:
        bag_dir = bag_paths[0].parent
        bag_name = bag_dir.name
        class_name = bag_dir.parent.name  # 父目录是类别名
        if class_name in dataset.class_to_idx:
            bag_to_original[bag_name] = dataset.class_to_idx[class_name]
    
    # 计算变化数量
    changed_count = 0
    total_bags = len(bag_to_shuffled)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Stage 2 标签随机置换实验日志 (Label Permutation Test Log)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"总 Bag 数量: {total_bags}\n")
        f.write(f"类别列表: {class_names}\n\n")
        
        f.write("详细映射 (Bag Name -> [Original Label] -> [Shuffled Label]):\n")
        f.write("-" * 70 + "\n")
        
        for bag_name, shuffled_label in sorted(bag_to_shuffled.items()):
            original_label = bag_to_original.get(bag_name, -1)
            original_name = class_names[original_label] if 0 <= original_label < len(class_names) else "Unknown"
            shuffled_name = class_names[shuffled_label] if 0 <= shuffled_label < len(class_names) else "Unknown"
            
            if original_label != shuffled_label:
                changed_count += 1
                f.write(f"[CHANGED] {bag_name}: {original_name} -> {shuffled_name}\n")
            else:
                f.write(f"[SAME]    {bag_name}: {original_name} -> {shuffled_name}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"标签变化统计: {changed_count}/{total_bags} bags 的标签被改变 "
                f"({100*changed_count/total_bags:.1f}%)\n")
        f.write("=" * 70 + "\n")
    
    return changed_count, total_bags


# ====================== 模型定义 (与v4相同) ======================
class TransMIL(nn.Module):
    def __init__(self, num_classes, backbone='convnext_tiny', pretrained_backbone_path=None, rank=0,
                 transformer_layers=2, transformer_heads=4, dropout_rate=0.3):
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

        if rank == 0: print(f"正在从 '{pretrained_backbone_path}' 加载自定义预训练Backbone...")
        state_dict = torch.load(pretrained_backbone_path, map_location='cpu')
        self.stage1_model.load_state_dict(state_dict)
        if rank == 0: print("成功加载自定义Backbone权重！")

        if 'convnext' in backbone:
            self.feature_extractor = nn.Sequential(*list(self.stage1_model.children())[:-1])
        elif 'resnet' in backbone or 'resnext' in backbone:
            self.feature_extractor = nn.Sequential(*list(self.stage1_model.children())[:-1])
        else:
            self.feature_extractor = self.stage1_model.features
        self.feat_dim = feat_dim
        for param in self.feature_extractor.parameters(): param.requires_grad = False

        if rank == 0: print("构建聚合器: Transformer")
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feat_dim, nhead=transformer_heads, dropout=dropout_rate,
                                                   dim_feedforward=self.feat_dim * 4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.feat_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 101, self.feat_dim))
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        logits_list = []
        for bag in x:
            bag = bag.cuda()
            self.feature_extractor.eval()
            with torch.no_grad():
                feats = self.feature_extractor(bag)
                feats = feats.view(feats.size(0), -1)

            cls_token = self.class_token.expand(1, -1, -1)
            feats = torch.cat((cls_token, feats.unsqueeze(0)), dim=1)

            max_len = self.pos_embedding.shape[1]
            if feats.shape[1] > max_len: feats = feats[:, :max_len, :]

            feats += self.pos_embedding[:, :feats.shape[1], :]

            out = self.transformer_encoder(feats)
            logits = self.classifier(out[:, 0, :])
            logits_list.append(logits)

        return torch.cat(logits_list, dim=0)


class AttentionMIL(nn.Module):
    def __init__(self, num_classes, backbone='convnext_tiny', pretrained_backbone_path=None, rank=0):
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

        if rank == 0: print(f"正在从 '{pretrained_backbone_path}' 加载自定义预训练Backbone...")
        state_dict = torch.load(pretrained_backbone_path, map_location='cpu')
        self.stage1_model.load_state_dict(state_dict)
        if rank == 0: print("成功加载自定义Backbone权重！")

        if 'convnext' in backbone:
            self.feature_extractor = nn.Sequential(*list(self.stage1_model.children())[:-1])
        elif 'resnet' in backbone or 'resnext' in backbone:
            self.feature_extractor = nn.Sequential(*list(self.stage1_model.children())[:-1])
        else:
            self.feature_extractor = self.stage1_model.features
        self.feat_dim = feat_dim
        for param in self.feature_extractor.parameters(): param.requires_grad = False

        if rank == 0: print("构建聚合器: 经典 Attention-MIL")
        self.attention_net = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        logits_list = []
        for bag in x:
            bag = bag.cuda()
            self.feature_extractor.eval()
            with torch.no_grad():
                feats = self.feature_extractor(bag)
                feats = feats.view(feats.size(0), -1)

            A_unnormalized = self.attention_net(feats)
            A = F.softmax(A_unnormalized, dim=0)
            M = torch.sum(A * feats, dim=0)
            logits = self.classifier(M.unsqueeze(0))
            logits_list.append(logits)

        return torch.cat(logits_list, dim=0)


# ====================== 训练与验证函数 (与v4相同) ======================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch, rank):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    loader.sampler.set_epoch(epoch)

    pbar = tqdm(loader, disable=(rank != 0), desc=f"Epoch {epoch + 1} Training")
    for bags, labels in pbar:
        if bags is None or labels is None: continue
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(bags)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

        if rank == 0:
            pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, loader, criterion, device, rank):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, disable=(rank != 0), desc="Validating")
    with torch.no_grad():
        for bags, labels in pbar:
            if bags is None or labels is None: continue
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                logits = model(bags)
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            if rank == 0:
                pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    if total_samples == 0: return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def validate_majority_voting(model, loader, device, rank):
    model.eval()
    model.to(device)
    bag_predictions = defaultdict(list)
    bag_labels = {}

    pbar = tqdm(loader, disable=(rank != 0), desc="基线验证 (多数投票法)")
    with torch.no_grad():
        for patches, labels, bag_ids in pbar:
            patches = patches.to(device)
            outputs = model(patches)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for i in range(len(bag_ids)):
                bag_id = bag_ids[i]
                bag_predictions[bag_id].append(preds[i])
                if bag_id not in bag_labels:
                    bag_labels[bag_id] = labels[i].item()

    if not bag_predictions: return 0.0

    correct_bags, total_bags = 0, 0
    for bag_id, preds in bag_predictions.items():
        final_pred = Counter(preds).most_common(1)[0][0]
        if final_pred == bag_labels[bag_id]:
            correct_bags += 1
        total_bags += 1

    return correct_bags / total_bags if total_bags > 0 else 0.0


def validate_with_results(model, loader, criterion, device, rank, class_names):
    """验证函数，同时收集详细的分类结果用于可视化"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, disable=(rank != 0), desc="Final Validation with Results")
    with torch.no_grad():
        for bags, labels in pbar:
            if bags is None or labels is None: continue
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                logits = model(bags)
                loss = criterion(logits, labels)
            
            probs = F.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
            if rank == 0:
                pbar.set_postfix(loss=total_loss / total_samples, acc=total_correct / total_samples)

    if total_samples == 0: 
        return 0.0, 0.0, {}, [], [], []
    
    # 构建分类结果字典
    results = {
        'labels': [int(l) for l in all_labels],
        'predictions': [int(p) for p in all_preds],
        'probabilities': [p.tolist() for p in all_probs],
        'class_names': class_names,
        'accuracy': total_correct / total_samples,
        'loss': total_loss / total_samples
    }
    
    return total_loss / total_samples, total_correct / total_samples, results, all_labels, all_preds, all_probs


# ====================== Nature期刊风格可视化函数 ======================
# 定义Nature期刊风格的颜色调色板
NATURE_COLORS = {
    'primary': '#2E86AB',      # 深蓝 - 主色调
    'secondary': '#A23B72',    # 玫红 - 次色调
    'tertiary': '#F18F01',     # 橙色
    'quaternary': '#C73E1D',   # 红色
    'quinary': '#3B1F2B',      # 深棕
    'success': '#4CAF50',      # 绿色
    'neutral': '#6C757D',      # 灰色
}

NATURE_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9B59B6', '#1ABC9C', '#E74C3C']


def save_classification_results(results: Dict, output_dir: str, run_name: str):
    """保存分类结果到JSON文件"""
    results_path = os.path.join(output_dir, f'{run_name}_classification_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 分类结果已保存到: {results_path}")
    return results_path


def plot_roc_curves(y_true, y_probs, class_names, output_dir, run_name):
    """
    绘制多类别ROC曲线 (Nature期刊风格)
    使用One-vs-Rest策略计算每个类别的ROC曲线
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_probs = np.array(y_probs)
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 计算宏平均ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(5.5, 5))
    
    # 绘制对角线 (随机分类器)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
    
    # 绘制每个类别的ROC曲线
    colors = cycle(NATURE_PALETTE[:n_classes])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # 绘制宏平均ROC曲线
    ax.plot(fpr["macro"], tpr["macro"], color='#1a1a1a', lw=2.5, linestyle='-.',
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    # 图形设置
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontweight='medium')
    ax.set_ylabel('True Positive Rate', fontweight='medium')
    ax.set_title('ROC Curves for Multi-class Classification', fontweight='bold', pad=10)
    
    # 图例设置
    legend = ax.legend(loc='lower right', frameon=True, fancybox=False, 
                       edgecolor='#cccccc', framealpha=0.95)
    legend.get_frame().set_linewidth(0.5)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(output_dir, f'{run_name}_roc_curves.pdf')
    plt.savefig(save_path, format='pdf', dpi=300)
    save_path_png = os.path.join(output_dir, f'{run_name}_roc_curves.png')
    plt.savefig(save_path_png, format='png', dpi=300)
    plt.close()
    
    print(f"✓ ROC曲线已保存到: {save_path}")
    return roc_auc


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, run_name, normalize=True):
    """
    绘制混淆矩阵 (Nature期刊风格)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 使用自定义颜色映射 (从白色到深蓝)
    cmap = sns.light_palette(NATURE_COLORS['primary'], as_cmap=True)
    
    # 绘制热力图
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, aspect='auto')
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    if normalize:
        cbar.set_label('Proportion', fontsize=9)
    else:
        cbar.set_label('Count', fontsize=9)
    
    # 设置刻度和标签
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # 添加文本注释
    thresh = (cm_display.max() + cm_display.min()) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f'{cm_display[i, j]:.1%}'
            else:
                text = f'{cm[i, j]}'
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                   color="white" if cm_display[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted Label', fontweight='medium')
    ax.set_ylabel('True Label', fontweight='medium')
    ax.set_title(title, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(output_dir, f'{run_name}_confusion_matrix.pdf')
    plt.savefig(save_path, format='pdf', dpi=300)
    save_path_png = os.path.join(output_dir, f'{run_name}_confusion_matrix.png')
    plt.savefig(save_path_png, format='png', dpi=300)
    plt.close()
    
    print(f"✓ 混淆矩阵已保存到: {save_path}")
    return cm


def plot_precision_recall_curves(y_true, y_probs, class_names, output_dir, run_name):
    """
    绘制Precision-Recall曲线 (Nature期刊风格)
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_probs = np.array(y_probs)
    
    # 计算每个类别的P-R曲线
    precision = dict()
    recall = dict()
    ap = dict()  # Average Precision
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(5.5, 5))
    
    # 绘制每个类别的P-R曲线
    colors = cycle(NATURE_PALETTE[:n_classes])
    for i, color in zip(range(n_classes), colors):
        ax.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_names[i]} (AP = {ap[i]:.3f})')
    
    # 计算并绘制平均AP
    mean_ap = np.mean(list(ap.values()))
    ax.axhline(y=mean_ap, color='#1a1a1a', linestyle='-.', lw=1.5,
               label=f'Mean AP = {mean_ap:.3f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontweight='medium')
    ax.set_ylabel('Precision', fontweight='medium')
    ax.set_title('Precision-Recall Curves', fontweight='bold', pad=10)
    
    legend = ax.legend(loc='lower left', frameon=True, fancybox=False,
                       edgecolor='#cccccc', framealpha=0.95)
    legend.get_frame().set_linewidth(0.5)
    
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{run_name}_pr_curves.pdf')
    plt.savefig(save_path, format='pdf', dpi=300)
    save_path_png = os.path.join(output_dir, f'{run_name}_pr_curves.png')
    plt.savefig(save_path_png, format='png', dpi=300)
    plt.close()
    
    print(f"✓ Precision-Recall曲线已保存到: {save_path}")
    return ap


def plot_class_metrics_bar(y_true, y_pred, class_names, output_dir, run_name):
    """
    绘制每个类别的性能指标条形图 (Precision, Recall, F1-Score)
    Nature期刊风格
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color=NATURE_COLORS['primary'], alpha=0.9)
    bars2 = ax.bar(x, recall, width, label='Recall', color=NATURE_COLORS['secondary'], alpha=0.9)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color=NATURE_COLORS['tertiary'], alpha=0.9)
    
    # 添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    ax.set_ylabel('Score', fontweight='medium')
    ax.set_xlabel('Class', fontweight='medium')
    ax.set_title('Per-class Classification Metrics', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False,
                       edgecolor='#cccccc', framealpha=0.95, ncol=3)
    legend.get_frame().set_linewidth(0.5)
    
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{run_name}_class_metrics.pdf')
    plt.savefig(save_path, format='pdf', dpi=300)
    save_path_png = os.path.join(output_dir, f'{run_name}_class_metrics.png')
    plt.savefig(save_path_png, format='png', dpi=300)
    plt.close()
    
    print(f"✓ 类别性能指标图已保存到: {save_path}")
    return precision, recall, f1


def plot_summary_figure(y_true, y_pred, y_probs, class_names, output_dir, run_name, baseline_acc=None):
    """
    创建综合性的汇总图 (Nature期刊风格 2x2布局)
    包含: ROC曲线、混淆矩阵、P-R曲线、性能对比
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_probs = np.array(y_probs)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    
    # === 1. ROC曲线 (左上) ===
    ax1 = axes[0, 0]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 宏平均
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    colors = cycle(NATURE_PALETTE[:n_classes])
    for i, color in zip(range(n_classes), colors):
        ax1.plot(fpr[i], tpr[i], color=color, lw=1.8,
                label=f'{class_names[i]} ({roc_auc[i]:.3f})')
    
    ax1.set_xlim([-0.02, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel('False Positive Rate', fontsize=10)
    ax1.set_ylabel('True Positive Rate', fontsize=10)
    ax1.set_title('a) ROC Curves', fontweight='bold', loc='left', fontsize=11)
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # === 2. 混淆矩阵 (右上) ===
    ax2 = axes[0, 1]
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cmap = sns.light_palette(NATURE_COLORS['primary'], as_cmap=True)
    im = ax2.imshow(cm_norm, interpolation='nearest', cmap=cmap, aspect='auto')
    
    cbar = fig.colorbar(im, ax=ax2, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    
    ax2.set_xticks(np.arange(n_classes))
    ax2.set_yticks(np.arange(n_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(class_names, fontsize=9)
    
    thresh = (cm_norm.max() + cm_norm.min()) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, f'{cm_norm[i, j]:.1%}', ha="center", va="center",
                    fontsize=8, color="white" if cm_norm[i, j] > thresh else "black")
    
    ax2.set_xlabel('Predicted', fontsize=10)
    ax2.set_ylabel('True', fontsize=10)
    ax2.set_title('b) Confusion Matrix', fontweight='bold', loc='left', fontsize=11)
    
    # === 3. Precision-Recall曲线 (左下) ===
    ax3 = axes[1, 0]
    
    precision = dict()
    recall = dict()
    ap = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    
    colors = cycle(NATURE_PALETTE[:n_classes])
    for i, color in zip(range(n_classes), colors):
        ax3.plot(recall[i], precision[i], color=color, lw=1.8,
                label=f'{class_names[i]} ({ap[i]:.3f})')
    
    mean_ap = np.mean(list(ap.values()))
    ax3.axhline(y=mean_ap, color='#1a1a1a', linestyle='-.', lw=1.2, alpha=0.7)
    
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Recall', fontsize=10)
    ax3.set_ylabel('Precision', fontsize=10)
    ax3.set_title('c) Precision-Recall Curves', fontweight='bold', loc='left', fontsize=11)
    ax3.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # === 4. 性能对比/汇总统计 (右下) ===
    ax4 = axes[1, 1]
    
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 绘制条形图
    x = np.arange(n_classes)
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, rec, width, label='Recall', color=NATURE_COLORS['primary'], alpha=0.85)
    bars2 = ax4.bar(x + width/2, prec, width, label='Precision', color=NATURE_COLORS['secondary'], alpha=0.85)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_ylim(0, 1.15)
    ax4.set_title('d) Per-class Performance', fontweight='bold', loc='left', fontsize=11)
    ax4.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax4.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 添加整体指标文本框
    textstr = f'Overall Accuracy: {overall_acc:.3f}\nMacro F1-Score: {macro_f1:.3f}\nMean AUC: {np.mean(list(roc_auc.values())):.3f}'
    if baseline_acc is not None:
        textstr += f'\nBaseline (Voting): {baseline_acc:.3f}'
    
    props = dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.9)
    ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(output_dir, f'{run_name}_summary.pdf')
    plt.savefig(save_path, format='pdf', dpi=300)
    save_path_png = os.path.join(output_dir, f'{run_name}_summary.png')
    plt.savefig(save_path_png, format='png', dpi=300)
    plt.close()
    
    print(f"✓ 综合汇总图已保存到: {save_path}")
    
    return {
        'accuracy': overall_acc,
        'macro_f1': macro_f1,
        'mean_auc': np.mean(list(roc_auc.values())),
        'per_class_auc': roc_auc,
        'per_class_ap': ap
    }


def generate_classification_report_file(y_true, y_pred, class_names, output_dir, run_name):
    """生成详细的分类报告文本文件"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    report_path = os.path.join(output_dir, f'{run_name}_classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("       Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✓ 分类报告已保存到: {report_path}")
    return report


def run_visualization(model, loader, criterion, device, rank, class_names, output_dir, run_name, baseline_acc=None):
    """
    执行完整的验证和可视化流程
    """
    print("\n" + "=" * 50)
    print(" 开始生成分类结果可视化")
    print("=" * 50)
    
    # 1. 进行验证并收集结果
    val_loss, val_acc, results, y_true, y_pred, y_probs = validate_with_results(
        model, loader, criterion, device, rank, class_names
    )
    
    if not results:
        print("警告: 没有获取到有效的验证结果！")
        return None
    
    # 只在主进程进行可视化
    if rank != 0:
        return None
    
    print(f"\n验证准确率: {val_acc:.4f}")
    print(f"验证损失: {val_loss:.4f}")
    
    # 2. 保存分类结果
    results['baseline_accuracy'] = baseline_acc
    save_classification_results(results, output_dir, run_name)
    
    # 3. 生成各种可视化图
    print("\n正在生成可视化图...")
    
    # ROC曲线
    roc_auc = plot_roc_curves(y_true, y_probs, class_names, output_dir, run_name)
    
    # 混淆矩阵
    cm = plot_confusion_matrix(y_true, y_pred, class_names, output_dir, run_name)
    
    # Precision-Recall曲线
    ap = plot_precision_recall_curves(y_true, y_probs, class_names, output_dir, run_name)
    
    # 类别指标条形图
    prec, rec, f1 = plot_class_metrics_bar(y_true, y_pred, class_names, output_dir, run_name)
    
    # 综合汇总图
    summary_stats = plot_summary_figure(y_true, y_pred, y_probs, class_names, output_dir, run_name, baseline_acc)
    
    # 分类报告
    report = generate_classification_report_file(y_true, y_pred, class_names, output_dir, run_name)
    
    print("\n" + "=" * 50)
    print(" 可视化生成完成！")
    print("=" * 50)
    print(f"所有结果已保存到: {output_dir}")
    print(f"  - 分类结果 (JSON)")
    print(f"  - ROC曲线 (PDF/PNG)")
    print(f"  - 混淆矩阵 (PDF/PNG)")
    print(f"  - Precision-Recall曲线 (PDF/PNG)")
    print(f"  - 类别指标条形图 (PDF/PNG)")
    print(f"  - 综合汇总图 (PDF/PNG)")
    print(f"  - 分类报告 (TXT)")
    
    return summary_stats


# ============================== Main 执行函数 (v4 修复版) ==============================
def main():
    parser = argparse.ArgumentParser(description="V4 Fixed: DDP Train MIL with correct label mapping.")
    parser.add_argument('--aggregator', type=str, default='attentionmil', choices=['transmil', 'attentionmil'],
                        help="选择要使用的聚合器")
    parser.add_argument('--group', type=int, required=True, choices=[1, 2, 3, 4], help='Group ID to select classes')
    parser.add_argument('--backbone', type=str, default='convnext_tiny', help="必须与Stage 1训练时使用的backbone一致")
    parser.add_argument('--pretrained_backbone_path', type=str, required=True,
                        help="Stage 1训练好的backbone权重文件路径 (.pth)")
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help="每个GPU的Batch size (bags的数量)")
    parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=15, help='Patience for Early Stopping (in epochs)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate in Transformer (仅用于transmil)')
    parser.add_argument('--transformer-layers', type=int, default=2,
                        help='Number of layers in Transformer Encoder (仅用于transmil)')
    parser.add_argument('--transformer-heads', type=int, default=4,
                        help='Number of heads in Transformer Encoder (仅用于transmil)')
    # --- 标签随机置换实验参数 ---
    parser.add_argument('--shuffle-labels', action='store_true',
                        help='启用标签随机置换实验：随机打乱训练集的bag-level标签，验证模型是否学到真实信号')
    parser.add_argument('--shuffle-seed', type=int, default=42,
                        help='标签置换的随机种子（默认42），【重要】必须与Stage 1使用相同的seed，确保两阶段标签置换一致')
    args = parser.parse_args()

    local_rank = init_distributed()
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # --- [FIX] ---
    # 1. 从 GROUPS 获取原始的类别名称列表
    classes_for_group = GROUPS[args.group]
    # 2. 模拟 ImageFolder 的行为，按字母顺序排序类别名称。
    #    这确保了 Stage 2 使用的 class_to_idx 映射与 Stage 1 完全一致。
    classes_this_run = sorted(classes_for_group)
    # --- [END FIX] ---

    num_classes = len(classes_this_run)
    # 如果启用标签置换，在 run_name 中加入 SHUFFLED 标识
    shuffle_tag = f"_SHUFFLED_seed{args.shuffle_seed}" if args.shuffle_labels else ""
    run_name = f"stage2-v4fix-{args.aggregator}_group{args.group}_{args.backbone}{shuffle_tag}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = ensure_dir(os.path.join('runs_stage2', run_name))
    best_model_path = os.path.join(output_dir, 'best_model_stage2.pth')

    if rank == 0:
        class_to_idx_correct = {name: i for i, name in enumerate(classes_this_run)}
        print("-" * 50)
        print(f"运行名称: {run_name}")
        print(f"聚合器: {args.aggregator.upper()}")
        print(f"分类组: {args.group}")
        print(f"使用的类别及其索引 (按字母顺序): {class_to_idx_correct}")
        print(f"Backbone: {args.backbone}")
        print("-" * 50)
        
        if args.shuffle_labels:
            print("\n" + "!" * 70)
            print("!!!  警告：Stage 2 标签随机置换实验模式已启用  !!!")
            print("!!!  WARNING: STAGE 2 LABEL PERMUTATION TEST MODE ENABLED  !!!")
            print("!!!  训练集标签将被随机打乱，验证集标签保持不变  !!!")
            print(f"!!!  Shuffle Seed: {args.shuffle_seed} (必须与 Stage 1 一致)  !!!")
            print("!" * 70 + "\n")

    val_transforms = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_transforms = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.aggregator == 'transmil':
        model = TransMIL(num_classes=num_classes, backbone=args.backbone,
                         pretrained_backbone_path=args.pretrained_backbone_path, rank=rank,
                         transformer_layers=args.transformer_layers, transformer_heads=args.transformer_heads,
                         dropout_rate=args.dropout)
    else:  # attentionmil
        model = AttentionMIL(num_classes=num_classes, backbone=args.backbone,
                             pretrained_backbone_path=args.pretrained_backbone_path, rank=rank)

    # 1. 运行基线验证
    # --- [FIX] ---
    # 使用排序后的 `classes_this_run` 列表来初始化数据集，以确保标签索引正确
    val_patch_ds = PatchDatasetForVoting(root_dir=os.path.join(PATCH_DATA_DIR, 'val'), transform=val_transforms,
                                         class_names=classes_this_run)
    # --- [END FIX] ---
    val_patch_loader = DataLoader(val_patch_ds, batch_size=args.batch_size * 8, shuffle=False, num_workers=NUM_WORKERS)
    baseline_acc = validate_majority_voting(model.stage1_model, val_patch_loader, device, rank)
    dist.barrier()
    if rank == 0:
        print("-" * 50)
        print(f"基线验证完成: 多数投票法在验证集上的准确率为: {baseline_acc:.4f}")
        print("-" * 50)

    # 2. 准备 MIL 训练
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # --- [FIX] ---
    # 同样，使用排序后的 `classes_this_run` 列表来初始化训练和验证集
    train_ds = MILDataset(root_dir=os.path.join(PATCH_DATA_DIR, 'train'), transform=train_transforms,
                          class_names=classes_this_run, rank=rank)
    val_ds = MILDataset(root_dir=os.path.join(PATCH_DATA_DIR, 'val'), transform=val_transforms,
                        class_names=classes_this_run, rank=rank)
    # --- [END FIX] ---

    # ============== 标签随机置换实验 ==============
    # 关键点：
    # 1. 只打乱训练集的标签，验证集保持真实标签
    # 2. 打乱单位是 bag（原始图像），与 Stage 1 一致
    # 3. 使用相同的 seed，确保两阶段标签置换一致
    # 4. 其余所有设置（模型、优化器、数据增强等）保持完全一致
    if args.shuffle_labels:
        if rank == 0:
            print("正在对训练集进行 bag-level 标签随机置换...")
        
        bag_to_shuffled = shuffle_bag_labels_mil(train_ds, seed=args.shuffle_seed)
        
        if rank == 0:
            changed_count, total_bags = log_label_shuffle_info_stage2(
                bag_to_shuffled, train_ds, output_dir
            )
            print(f"标签置换完成：{changed_count}/{total_bags} 个 bags 的标签被改变 ({100*changed_count/total_bags:.1f}%)")
            print(f"详细日志已保存到: {os.path.join(output_dir, 'label_shuffle_log_stage2.txt')}\n")
    # ============================================

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, collate_fn=mil_collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_val_acc_at_best_loss = 0.0

    if rank == 0:
        print("\n" + "=" * 20 + f" 开始 {args.aggregator.upper()} 训练 " + "=" * 20)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, rank)
        val_loss, val_acc = validate(model, val_loader, criterion, device, rank)
        scheduler.step()
        dist.barrier()

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{args.epochs} | LR: {current_lr:.1e} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                torch.save(model.module.state_dict(), best_model_path)
                best_val_loss = val_loss
                best_val_acc_at_best_loss = val_acc
                epochs_no_improve = 0
                print(f"🎉 新的最佳验证损失: {best_val_loss:.4f} (Acc: {best_val_acc_at_best_loss:.4f})！模型已保存。")
            else:
                epochs_no_improve += 1
                print(f"验证损失未改善. 已持续 {epochs_no_improve}/{args.patience} epochs.")

            if epochs_no_improve >= args.patience:
                print(f"\n验证损失已连续 {args.patience} epochs 未改善, 触发早停！")
                # 在触发早停时，再进行一次同步，确保所有进程都将退出
                dist.barrier()
                break

                # 无论是否早停，都需要一个 barrier 来同步所有进程
        # 如果 rank 0 早停了，其他进程需要收到信号并一起退出
        # 如果 rank 0 没有早停，其他进程也需要同步后才能进入下一轮
        stop_signal = torch.tensor([1 if epochs_no_improve >= args.patience else 0], device=device)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            if rank != 0: print(f"Rank {rank} 收到早停信号, 准备退出...")
            break

    # 训练结束后，在销毁进程组之前进行可视化
    # 先同步所有进程
    dist.barrier()
    
    # 加载最佳模型并生成可视化（仅在主进程进行）
    if rank == 0:
        print("\n" + "=" * 20 + " 训练结束 " + "=" * 20)
        print(f"多数投票基线准确率: {baseline_acc:.4f}")
        print(f"最佳验证准确率 (在最低验证损失时): {best_val_acc_at_best_loss:.4f}")
        improvement = best_val_acc_at_best_loss - baseline_acc
        print(f"性能提升: {improvement:+.4f}")
        
        # 如果是标签置换实验，给出结果解读
        if args.shuffle_labels:
            random_baseline = 1.0 / num_classes
            print("\n" + "-" * 70)
            print("【Stage 2 标签随机置换实验结果分析】")
            print(f"随机基线 (Random Baseline): {random_baseline:.4f} ({num_classes}分类)")
            print(f"最佳验证准确率: {best_val_acc_at_best_loss:.4f}")
            
            if best_val_acc_at_best_loss <= random_baseline + 0.10:  # 比随机高不超过10%
                print("\n✅ 结论：模型性能接近随机水平")
                print("   这说明当前 MIL pipeline 的性能主要来自真实的可学习信号，")
                print("   没有发现明显的数据泄漏或伪相关问题。")
            elif best_val_acc_at_best_loss <= random_baseline + 0.25:  # 比随机高10%-25%
                print("\n⚠️  结论：模型性能略高于随机水平")
                print("   建议进一步检查是否存在轻微的数据泄漏或训练/验证集的分布差异。")
            else:  # 比随机高超过25%
                print("\n❌ 警告：模型在打乱标签后仍能学习")
                print("   这强烈暗示可能存在数据泄漏或伪相关！")
                print("   建议检查：")
                print("   1. 训练集和验证集之间是否有数据重叠")
                print("   2. patch 切分是否导致同一图像的 patches 分布在不同集合中")
                print("   3. MIL aggregator 是否存在过拟合到 bag 结构的问题")
            print("-" * 70)
        else:
            if improvement > 0.01:
                print("🚀 模型成功超越了多数投票基线！")
            else:
                print("🤔 模型未能显著超越基线，建议继续调整超参数或检查数据。")
        
        # 尝试加载最佳模型进行可视化
        if os.path.exists(best_model_path):
            print(f"\n正在加载最佳模型进行可视化: {best_model_path}")
            
            # 创建新的模型实例（不使用DDP包装）
            if args.aggregator == 'transmil':
                vis_model = TransMIL(num_classes=num_classes, backbone=args.backbone,
                                     pretrained_backbone_path=args.pretrained_backbone_path, rank=rank,
                                     transformer_layers=args.transformer_layers, transformer_heads=args.transformer_heads,
                                     dropout_rate=args.dropout)
            else:
                vis_model = AttentionMIL(num_classes=num_classes, backbone=args.backbone,
                                         pretrained_backbone_path=args.pretrained_backbone_path, rank=rank)
            
            # 加载最佳权重
            vis_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
            vis_model.to(device)
            vis_model.eval()
            
            # 创建不使用分布式采样器的验证加载器用于可视化
            val_ds_vis = MILDataset(root_dir=os.path.join(PATCH_DATA_DIR, 'val'), transform=val_transforms,
                                    class_names=classes_this_run, rank=rank)
            val_loader_vis = DataLoader(val_ds_vis, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=mil_collate_fn)
            
            # 执行可视化
            summary_stats = run_visualization(
                model=vis_model,
                loader=val_loader_vis,
                criterion=criterion,
                device=device,
                rank=rank,
                class_names=classes_this_run,
                output_dir=output_dir,
                run_name=run_name,
                baseline_acc=baseline_acc
            )
            
            if summary_stats:
                print("\n" + "=" * 50)
                print(" 最终性能汇总")
                print("=" * 50)
                print(f"  • Overall Accuracy: {summary_stats['accuracy']:.4f}")
                print(f"  • Macro F1-Score:   {summary_stats['macro_f1']:.4f}")
                print(f"  • Mean AUC:         {summary_stats['mean_auc']:.4f}")
                print(f"  • Baseline (MV):    {baseline_acc:.4f}")
                print("=" * 50)
        else:
            print(f"\n警告: 未找到最佳模型文件 {best_model_path}，跳过可视化。")
    
    # 最后销毁进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

