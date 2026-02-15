import os
import shutil
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import argparse
import sys

# ==============================================================================
# 1. 配置参数 (Configuration)
# ==============================================================================
# 允许通过命令行覆盖这些默认值： --data-dir --out-dir --val-split --patch-size --stride
RAW_DATA_DIR = './data_CU'  # 顶层目录，包含 10 个类别文件夹：30-1, 30-2, 45-1, ...
PATCH_OUTPUT_DIR = './patches_by_image'  # 所有处理后数据的根目录 (建议改名以区分)
PATCH_SIZE = 256  # 切片大小
STRIDE = 256  # 步长
VALIDATION_SPLIT = 0.2  # 验证集所占的比例 (20%)
RANDOM_SEED = 42  # 设置随机种子以保证每次划分结果一致

# --- 其他设置 ---
IMAGE_EXTS = ('.tif', '.tiff', '.bmp', '.jpg')  # 允许的图像后缀
Image.MAX_IMAGE_PIXELS = None  # 允许处理超大图像


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data_CU into patches organized by image')
    parser.add_argument('--data-dir', type=str, default=RAW_DATA_DIR, help='root folder e.g. ./data_CU')
    parser.add_argument('--out-dir', type=str, default=PATCH_OUTPUT_DIR, help='output patches root')
    parser.add_argument('--val-split', type=float, default=VALIDATION_SPLIT)
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    return parser.parse_args()


# ==============================================================================
# 2. 工具函数 (Utils)
# ==============================================================================

def _is_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTS)


def _collect_images(class_dir: str):
    """
    收集指定类别目录下所有的图像路径。
    目录结构示例：
        data_CU/
          └── 30-1/
              ├── xxx.tif
              └── yyy.tif
    """
    image_paths = []
    if not os.path.isdir(class_dir):
        return image_paths

    for f in os.listdir(class_dir):
        if _is_image(f):
            image_paths.append(os.path.join(class_dir, f))
    return image_paths


# ==============================================================================
# 3. 功能函数 (Functions)
# ==============================================================================

def split_whole_images(raw_dir: str, val_split: float, seed: int):
    """
    在“整图”级别上划分数据集，防止数据泄露。
    - 适配目录结构：data_CU/类别/图像
    - 返回一个字典，包含每个类别下用于训练和验证的完整图像路径列表。
    """
    print("步骤 1: 正在按整图级别划分训练集和验证集...")
    random.seed(seed)

    image_splits = {
        'train': defaultdict(list),
        'val': defaultdict(list)
    }

    class_names = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    class_names.sort()

    if not class_names:
        raise RuntimeError(f"在 {raw_dir} 下未找到任何类别文件夹！")

    for class_name in class_names:
        class_path = os.path.join(raw_dir, class_name)

        all_image_paths = _collect_images(class_path)

        if not all_image_paths:
            print(f"警告：类别 '{class_name}' 未找到任何图像，跳过该类别。")
            continue

        random.shuffle(all_image_paths)
        split_point = int(len(all_image_paths) * (1 - val_split))
        train_paths = all_image_paths[:split_point]
        val_paths = all_image_paths[split_point:]

        image_splits['train'][class_name].extend(train_paths)
        image_splits['val'][class_name].extend(val_paths)

        print(f" - 类别 '{class_name}': {len(train_paths)} 张图用于训练, {len(val_paths)} 张图用于验证。")

    print("整图划分完成！\n")
    return image_splits


def create_patches_from_split(image_dict, output_base_dir, patch_size, stride):
    """
    根据给定的图像文件列表，生成切片并保存到以原图命名的独立文件夹中。
    """
    print(f"步骤 2: 正在为 '{os.path.basename(output_base_dir)}' 集生成切片...")
    os.makedirs(output_base_dir, exist_ok=True)

    total_patches = 0

    for class_name, image_paths in image_dict.items():
        output_class_path = os.path.join(output_base_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        if not image_paths:
            continue

        print(f" 处理类别: {class_name}")
        for img_path in tqdm(image_paths, desc=f" - {class_name}"):
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    img_basename = os.path.splitext(os.path.basename(img_path))[0]

                    # 为每张图片的 patch 创建一个专用文件夹
                    output_patch_dir = os.path.join(output_class_path, img_basename)
                    os.makedirs(output_patch_dir, exist_ok=True)

                    # 对图像进行滑窗切片
                    for i in range(0, img_h - patch_size + 1, stride):
                        for j in range(0, img_w - patch_size + 1, stride):
                            box = (j, i, j + patch_size, i + patch_size)
                            patch = img.crop(box)

                            # 简化 patch 文件名并保存到新目录
                            patch_filename = f"patch_{i}_{j}.png"
                            patch.save(os.path.join(output_patch_dir, patch_filename))
                            total_patches += 1

            except Exception as e:
                print(f"处理文件 {img_path} 时出错: {e}")

    print(f"为 '{os.path.basename(output_base_dir)}' 集共生成 {total_patches} 个切片。\n")


# ==============================================================================
# 4. 主执行逻辑 (Main Execution)
# ==============================================================================
if __name__ == '__main__':
    args = parse_args()

    RAW_DATA_DIR = os.path.abspath(args.data_dir)
    PATCH_OUTPUT_DIR = os.path.abspath(args.out_dir)

    print('运行目录 (cwd):', os.getcwd())
    print('数据根目录   :', RAW_DATA_DIR)
    print('输出根目录   :', PATCH_OUTPUT_DIR)

    if not os.path.exists(RAW_DATA_DIR):
        print(f"错误：数据根目录不存在: {RAW_DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(PATCH_OUTPUT_DIR):
        print(f"发现旧目录 '{PATCH_OUTPUT_DIR}'，正在删除...")
        shutil.rmtree(PATCH_OUTPUT_DIR)
        print("旧目录已删除。")

    image_splits = split_whole_images(RAW_DATA_DIR, args.val_split, args.seed)

    create_patches_from_split(
        image_splits['train'],
        os.path.join(PATCH_OUTPUT_DIR, 'train'),
        args.patch_size,
        args.stride
    )

    create_patches_from_split(
        image_splits['val'],
        os.path.join(PATCH_OUTPUT_DIR, 'val'),
        args.patch_size,
        args.stride
    )

    print("所有处理完成！")
