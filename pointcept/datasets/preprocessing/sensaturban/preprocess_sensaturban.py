import os
import argparse
import glob
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from tqdm import tqdm

try:
    from plyfile import PlyData
except ImportError:
    raise ImportError("Please install plyfile: pip install plyfile")

def process_scene(scene_path, out_root, grid_size=0.1, chunk_size=(50, 50), chunk_stride=(25, 25)):
    scene_path = Path(scene_path)
    scene_name = scene_path.stem.lower()
    
    # -------------------------------------------------------
    # 1. 智能判断输出路径
    # -------------------------------------------------------
    # 如果源文件在 'train' 文件夹，输出到 'processed/train'
    # 如果源文件在 'test' 文件夹，且是 Block 2/8 (验证集)，输出到 'processed/val'
    parent_folder = scene_path.parent.name
    
    # 定义你选取的验证集 Block 名
    val_block_names = ["birmingham_block_1", "birmingham_block_6", "cambridge_block_12", "cambridge_block_6"]

    scene_name_lower = scene_path.stem.lower()

    # 只要是在官方 train 文件夹下的，都可以读到标签
    if parent_folder == "train":
        if any(bn in scene_name_lower for bn in val_block_names):
            save_dir = out_root / "val"  # 真正有标签的验证集
        else:
            save_dir = out_root / "train" # 训练集
    elif parent_folder == "test":
        save_dir = out_root / "test"     # 纯推理用的测试集 (全 255)

    # -------------------------------------------------------
    # 2. 读取 PLY
    # -------------------------------------------------------
    try:
        with open(str(scene_path), 'rb') as f:
            plydata = PlyData.read(f)
        
        vertex = plydata['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
        
        if 'red' in vertex:
            colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.uint8)
        else:
            colors = np.zeros_like(points, dtype=np.uint8)
        
        # 标签处理
        if 'class' in vertex:
            labels = np.array(vertex['class']).astype(np.int16)
        elif 'label' in vertex:
            labels = np.array(vertex['label']).astype(np.int16)
        else:
            labels = np.full(points.shape[0], 255, dtype=np.int16)
            
    except Exception as e:
        return f"❌ Error reading {scene_name}: {e}"

    # -------------------------------------------------------
    # 3. 坐标归一化 & Grid Sampling
    # -------------------------------------------------------
    points -= points.min(axis=0) # 归一化

    if grid_size > 0:
        scaled_coord = points / grid_size
        grid_coord = np.floor(scaled_coord).astype(int)
        _, indices = np.unique(grid_coord, axis=0, return_index=True)
        points = points[indices]
        colors = colors[indices]
        labels = labels[indices]

    # -------------------------------------------------------
    # 4. 切块
    # -------------------------------------------------------
    x_max, y_max = points.max(axis=0)[:2]
    
    chunk_idx = 0
    saved_count = 0
    
    x_range = np.arange(0, x_max, chunk_stride[0])
    y_range = np.arange(0, y_max, chunk_stride[1])

    for x in x_range:
        for y in y_range:
            mask = (
                (points[:, 0] >= x) & (points[:, 0] < x + chunk_size[0]) &
                (points[:, 1] >= y) & (points[:, 1] < y + chunk_size[1])
            )
            
            if np.sum(mask) < 1000:
                continue
            
            # 存到对应的 train 或 val 文件夹下
            chunk_name = f"{scene_name}_{chunk_idx}"
            chunk_path = save_dir / chunk_name
            chunk_path.mkdir(parents=True, exist_ok=True)
            
            np.save(chunk_path / "coord.npy", points[mask])
            np.save(chunk_path / "color.npy", colors[mask])
            np.save(chunk_path / "segment.npy", labels[mask])
            
            chunk_idx += 1
            saved_count += 1

    return f"✅ {scene_name} -> {save_dir.name}: {saved_count} chunks"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--grid_size", default=0.1, type=float)
    args = parser.parse_args()

    # 递归扫描 ply 目录下的所有文件
    ply_files = sorted(list(args.dataset_root.glob("**/*.ply")))
    
    if not ply_files:
        print(f"❌ No .ply files found in {args.dataset_root}")
        exit(1)

    print(f"🚀 Processing {len(ply_files)} scenes...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        results = list(tqdm(pool.map(
            process_scene,
            ply_files,
            repeat(args.output_root),
            repeat(args.grid_size)
        ), total=len(ply_files)))
        
    for res in results:
        print(res)