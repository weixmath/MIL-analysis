# file: train_final.py
# =====================================================================
# 训练脚本 - 支持标签随机置换实验 (Label Permutation Test)
# 使用 --shuffle-labels 参数启用标签置换，验证模型是否学到真实信号
# =====================================================================
import os, argparse, datetime
from typing import List, Dict, Tuple
from PIL import Image
from collections import defaultdict
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np

# --- [DDP] 引入分布式训练相关的库 ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def set_seed(seed: int):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 注意：DDP训练中，每个进程会有不同的数据增强，这是正常的
    # cudnn determinism 可能影响性能，这里不强制设置


def shuffle_bag_labels(dataset: Dataset, seed: int = 42) -> Dict[str, int]:
    """
    对训练集进行 bag-level 的标签随机置换。
    
    核心思想：
    - 在 MIL 场景中，每个 bag 是一个完整的 micrograph/image
    - 同一个 bag 下的所有 patches 共享同一个标签
    - 打乱是在 bag 级别进行的，即把所有 bag 的标签做一个随机 permutation
    
    【关键】为了确保 Stage 1 和 Stage 2 使用相同 seed 时结果一致：
    - 使用 bag 目录名（basename）进行稳定排序
    - 这样两阶段的 bag 顺序一致，打乱结果也一致
    
    Args:
        dataset: ImagePatchesDataset 对象
        seed: 随机种子，确保可复现性
    
    Returns:
        original_to_shuffled: 原始图像目录到打乱后标签的映射（用于日志记录）
    """
    # 1. 按原始图片（bag）分组
    image_to_label = {}  # 原始图像目录 -> 原始标签
    image_to_patches = defaultdict(list)  # 原始图像目录 -> patch索引列表
    
    for idx, (path, label_idx) in enumerate(dataset.samples):
        original_image_dir = os.path.dirname(path)
        if original_image_dir not in image_to_label:
            image_to_label[original_image_dir] = label_idx
        image_to_patches[original_image_dir].append(idx)
    
    # 2. 获取所有唯一的 bag，并按 basename 稳定排序（确保两阶段顺序一致）
    all_images = sorted(image_to_label.keys(), key=lambda x: os.path.basename(x))
    original_labels = [image_to_label[img] for img in all_images]
    
    # 3. 执行随机置换
    rng = random.Random(seed)
    shuffled_labels = original_labels.copy()
    rng.shuffle(shuffled_labels)
    
    # 4. 构建新的标签映射
    original_to_shuffled = {}
    for img, new_label in zip(all_images, shuffled_labels):
        original_to_shuffled[img] = new_label
    
    # 5. 更新 dataset 中所有 patches 的标签
    for image_dir, patch_indices in image_to_patches.items():
        new_label = original_to_shuffled[image_dir]
        for idx in patch_indices:
            path, _ = dataset.samples[idx]
            dataset.samples[idx] = (path, new_label)
    
    return original_to_shuffled


def log_label_shuffle_info(original_to_shuffled: Dict[str, int], 
                           dataset: Dataset, 
                           output_dir: str):
    """
    记录标签打乱的详细信息到文件中，便于事后分析。
    """
    log_path = os.path.join(output_dir, 'label_shuffle_log.txt')
    
    # 统计打乱前后的标签分布变化
    class_names = dataset.classes
    original_counts = defaultdict(int)
    shuffled_counts = defaultdict(int)
    
    image_to_original = {}
    for path, _ in dataset.samples:
        image_dir = os.path.dirname(path)
        # 从路径推断原始标签
        for class_name in class_names:
            if os.sep + class_name + os.sep in path or path.startswith(os.path.join(dataset.root_dir, class_name)):
                image_to_original[image_dir] = dataset.class_to_idx[class_name]
                break
    
    # 计算变化数量
    changed_count = 0
    total_bags = len(original_to_shuffled)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("标签随机置换实验日志 (Label Permutation Test Log)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"总 Bag 数量: {total_bags}\n")
        f.write(f"类别列表: {class_names}\n\n")
        
        f.write("详细映射 (Image Directory -> [Original Label] -> [Shuffled Label]):\n")
        f.write("-" * 70 + "\n")
        
        for image_dir, shuffled_label in sorted(original_to_shuffled.items()):
            original_label = image_to_original.get(image_dir, -1)
            original_name = class_names[original_label] if original_label >= 0 else "Unknown"
            shuffled_name = class_names[shuffled_label]
            
            if original_label != shuffled_label:
                changed_count += 1
                f.write(f"[CHANGED] {os.path.basename(image_dir)}: {original_name} -> {shuffled_name}\n")
            else:
                f.write(f"[SAME]    {os.path.basename(image_dir)}: {original_name} -> {shuffled_name}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"标签变化统计: {changed_count}/{total_bags} bags 的标签被改变 "
                f"({100*changed_count/total_bags:.1f}%)\n")
        f.write("=" * 70 + "\n")
    
    return changed_count, total_bags

# ====================== 基础配置 ======================
PATCH_DATA_DIR = './patches_by_image'
INPUT_SIZE = 224
NUM_WORKERS = min(8, os.cpu_count() or 0)
PIN_MEMORY = torch.cuda.is_available()

GROUPS = {
    1: ['15-1', '30-1', '45-1', '100'],
    2: ['15-2', '30-2', '45-2', '100'],
    3: ['15-3', '30-3', '45-3', '100'],
    4: ['15-1', '30-1', '45-1', '15-2', '30-2', '45-2', '15-3', '30-3', '45-3', '100'],
}


# ============================== 工具函数 ==============================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True);
    return p


def init_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


# ============================== 自定义数据集 (无变化) ==============================
class ImagePatchesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        if not self.classes: raise FileNotFoundError(f"在 '{root_dir}' 目录下没有找到任何类别文件夹。")
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root_dir, class_name)
            for image_folder_name in sorted(os.listdir(class_dir)):
                image_folder_path = os.path.join(class_dir, image_folder_name)
                if os.path.isdir(image_folder_path):
                    for patch_filename in sorted(os.listdir(image_folder_path)):
                        if patch_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                            patch_path = os.path.join(image_folder_path, patch_filename)
                            self.samples.append((patch_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None: sample = self.transform(sample)
        return sample, target


# ============================== 训练/验证循环 (无变化) ==============================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch, rank):
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    loss_sum_tensor = torch.tensor(0.0, device=device)
    correct_tensor = torch.tensor(0.0, device=device)
    total_tensor = torch.tensor(0.0, device=device)
    iterator = tqdm(loader, desc=f"Training (Epoch {epoch + 1})", leave=False) if rank == 0 else loader

    for images, labels in iterator:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(images)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum_tensor += loss.item() * images.size(0)
        correct_tensor += (out.argmax(1) == labels).sum()
        total_tensor += images.size(0)

    dist.all_reduce(loss_sum_tensor)
    dist.all_reduce(correct_tensor)
    dist.all_reduce(total_tensor)
    if total_tensor.item() == 0: return 0.0, 0.0
    return loss_sum_tensor.item() / total_tensor.item(), correct_tensor.item() / total_tensor.item()


@torch.no_grad()
def validate(model, loader, criterion, device, rank, num_classes):
    model.eval()
    loss_sum_tensor = torch.tensor(0.0, device=device)
    correct_tensor = torch.tensor(0.0, device=device)
    total_tensor = torch.tensor(0.0, device=device)
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    iterator = tqdm(loader, desc="Validating (Patches)", leave=False) if rank == 0 else loader

    for images, labels in iterator:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(images)
            loss = criterion(out, labels)
        preds = out.argmax(1)
        loss_sum_tensor += loss.item() * images.size(0)
        correct_tensor += (preds == labels).sum()
        total_tensor += images.size(0)
        for c in range(num_classes):
            class_mask = (labels == c)
            class_total[c] += class_mask.sum()
            class_correct[c] += (preds[class_mask] == labels[class_mask]).sum()

    dist.all_reduce(loss_sum_tensor)
    dist.all_reduce(correct_tensor)
    dist.all_reduce(total_tensor)
    dist.all_reduce(class_correct)
    dist.all_reduce(class_total)
    per_class_acc = class_correct / class_total.clamp(min=1)
    if total_tensor.item() == 0: return 0.0, 0.0, [0.0] * num_classes
    return loss_sum_tensor.item() / total_tensor.item(), correct_tensor.item() / total_tensor.item(), per_class_acc.cpu().tolist()


# ============================== <<< MODIFIED: 聚合评估函数重构 >>> ==============================
@torch.no_grad()
def validate_with_aggregation(model, val_dataset, device):
    """
    在每个 epoch 结束后，使用多数投票策略在验证集上进行聚合评估。
    只在 rank 0 进程上运行。
    """
    model.eval()  # 确保模型处于评估模式

    # 1. 按原始图片分组 Patch
    image_groups = defaultdict(list)
    for path, label_idx in val_dataset.samples:
        original_image_dir = os.path.dirname(path)
        image_groups[original_image_dir].append((path, label_idx))

    # 2. 遍历每个图片分组进行投票评估
    voting_correct, total_images = 0, 0
    transform = val_dataset.transform

    # 使用 tqdm 显示进度条，leave=False 表示完成后从屏幕上消失
    iterator = tqdm(image_groups.items(), desc="Validating (Image Voting)", leave=False)
    for image_dir, patch_infos in iterator:
        patch_paths = [info[0] for info in patch_infos]
        true_label_idx = patch_infos[0][1]  # 同一图片的所有patch标签相同

        patch_tensors = [transform(Image.open(p).convert('RGB')) for p in patch_paths]
        patches_batch = torch.stack(patch_tensors).to(device)

        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(patches_batch)

        # 执行投票
        pred_labels = logits.argmax(dim=1)
        voted_pred = torch.mode(pred_labels).values.item()
        if voted_pred == true_label_idx:
            voting_correct += 1

        total_images += 1

    # 3. 返回聚合后的准确率
    voting_acc = voting_correct / total_images if total_images > 0 else 0
    return voting_acc


# ============================== Main 执行函数 ==============================
def main():
    parser = argparse.ArgumentParser(description='DDP Image Classification Training')
    parser.add_argument('--group', type=int, required=True, choices=[1, 2, 3, 4], help='Group ID to select classes')
    parser.add_argument('--backbone', type=str, default='convnext_tiny', help='Model backbone from torchvision.models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor')
    # --- 标签随机置换实验参数 ---
    parser.add_argument('--shuffle-labels', action='store_true',
                        help='启用标签随机置换实验：随机打乱训练集的bag-level标签，验证模型是否学到真实信号')
    parser.add_argument('--shuffle-seed', type=int, default=42,
                        help='标签置换的随机种子（默认42），用于确保实验可复现')
    args = parser.parse_args()

    local_rank = init_distributed()
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    group_classes = GROUPS[args.group]
    # 如果启用标签置换，在 run_name 中加入 SHUFFLED 标识
    shuffle_tag = f"_SHUFFLED_seed{args.shuffle_seed}" if args.shuffle_labels else ""
    run_name = f"group{args.group}_{args.backbone}{shuffle_tag}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = ensure_dir(os.path.join('runs', run_name))
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    if rank == 0:
        print(f"Starting run: {run_name}")
        print(f"Classes for this run: {group_classes}")
        if args.shuffle_labels:
            print("\n" + "!" * 70)
            print("!!!  警告：标签随机置换实验模式已启用  !!!")
            print("!!!  WARNING: LABEL PERMUTATION TEST MODE ENABLED  !!!")
            print("!!!  训练集标签将被随机打乱，验证集标签保持不变  !!!")
            print(f"!!!  Shuffle Seed: {args.shuffle_seed}  !!!")
            print("!" * 70 + "\n")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_ds = ImagePatchesDataset(root_dir=os.path.join(PATCH_DATA_DIR, 'train'), transform=data_transforms['train'])
    val_ds = ImagePatchesDataset(root_dir=os.path.join(PATCH_DATA_DIR, 'val'), transform=data_transforms['val'])
    num_classes = len(train_ds.classes)

    # ============== 标签随机置换实验 ==============
    # 关键点：
    # 1. 只打乱训练集的标签，验证集保持真实标签
    # 2. 打乱单位是 bag（原始图像），不是 patch
    # 3. 其余所有设置（模型、优化器、数据增强等）保持完全一致
    if args.shuffle_labels:
        if rank == 0:
            print("正在对训练集进行 bag-level 标签随机置换...")
        
        original_to_shuffled = shuffle_bag_labels(train_ds, seed=args.shuffle_seed)
        
        if rank == 0:
            changed_count, total_bags = log_label_shuffle_info(
                original_to_shuffled, train_ds, output_dir
            )
            print(f"标签置换完成：{changed_count}/{total_bags} 个 bags 的标签被改变 ({100*changed_count/total_bags:.1f}%)")
            print(f"详细日志已保存到: {os.path.join(output_dir, 'label_shuffle_log.txt')}\n")
    # ============================================

    class_counts = torch.zeros(num_classes)
    for _, label_idx in train_ds.samples:
        class_counts[label_idx] += 1

    if rank == 0 and any(c == 0 for c in class_counts):
        print(f"!!! WARNING: One or more classes have 0 samples in the training set. Counts: {class_counts.tolist()}")

    class_weights = 1. / class_counts.clamp(min=1.0)
    sample_weights = torch.tensor([class_weights[label] for _, label in train_ds.samples])
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY, shuffle=False)

    model = getattr(models, args.backbone)(weights='DEFAULT')
    if 'convnext' in args.backbone:
        feat_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(feat_dim, num_classes)
    elif 'resnet' in args.backbone or 'resnext' in args.backbone:
        feat_dim = model.fc.in_features
        model.fc = nn.Linear(feat_dim, num_classes)
    else:
        feat_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(feat_dim, num_classes)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    weights_tensor = class_weights.to(device)
    dist.broadcast(weights_tensor, src=0)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()

    # <<< MODIFIED: 变量名从 best_val_acc 改为 best_agg_acc 以反映其新含义 >>>
    best_agg_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, rank)
        val_loss, val_patch_acc, per_class_acc = validate(model, val_loader, criterion, device, rank, num_classes)

        # --- <<< MODIFIED: 在主进程上执行聚合验证并保存最佳模型 >>> ---
        if rank == 0:
            # 1. 打印常规的 Patch 级别统计信息
            print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Patch Acc: {val_patch_acc:.4f}")

            acc_str = " -> Per-class Val Acc: "
            for i, acc in enumerate(per_class_acc):
                acc_str += f"[{train_ds.classes[i]}: {acc:.3f}] "
            print(acc_str)

            # 2. 运行新的聚合验证
            agg_val_acc = validate_with_aggregation(model.module, val_ds, device)
            print(f" -> Aggregated Val Acc (Image Voting): {agg_val_acc:.4f}")

            # 3. 根据聚合准确率保存最佳模型
            if agg_val_acc > best_agg_acc:
                best_agg_acc = agg_val_acc
                torch.save(model.module.state_dict(), best_model_path)
                print(f"** New best model saved to {best_model_path} with Aggregated Val Acc: {best_agg_acc:.4f} **\n")
            else:
                print("")  # 打印一个空行以分隔 epoch

    dist.barrier()
    if rank == 0:
        print("=" * 70)
        print("Training finished.")
        print(f"Best model saved at: {best_model_path}")
        print(f"Achieved best aggregated validation accuracy (voting): {best_agg_acc:.4f}")
        
        # 如果是标签置换实验，给出结果解读
        if args.shuffle_labels:
            random_baseline = 1.0 / num_classes
            print("\n" + "-" * 70)
            print("【标签随机置换实验结果分析】")
            print(f"随机基线 (Random Baseline): {random_baseline:.4f} ({num_classes}分类)")
            print(f"最佳验证准确率: {best_agg_acc:.4f}")
            
            if best_agg_acc <= random_baseline + 0.10:  # 比随机高不超过10%
                print("\n✅ 结论：模型性能接近随机水平")
                print("   这说明当前 pipeline 的性能主要来自真实的可学习信号，")
                print("   没有发现明显的数据泄漏或伪相关问题。")
            elif best_agg_acc <= random_baseline + 0.25:  # 比随机高10%-25%
                print("\n⚠️  结论：模型性能略高于随机水平")
                print("   建议进一步检查是否存在轻微的数据泄漏或训练/验证集的分布差异。")
            else:  # 比随机高超过25%
                print("\n❌ 警告：模型在打乱标签后仍能学习")
                print("   这强烈暗示可能存在数据泄漏或伪相关！")
                print("   建议检查：")
                print("   1. 训练集和验证集之间是否有数据重叠")
                print("   2. patch 切分是否导致同一图像的 patches 分布在不同集合中")
                print("   3. 是否存在 batch normalization 等造成的信息泄漏")
            print("-" * 70)
        
        print("=" * 70)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
