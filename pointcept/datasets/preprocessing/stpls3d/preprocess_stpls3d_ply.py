"""
STPLS3D Preprocessing Script (Raw PLY to NPY Chunks)
Location: pointcept/datasets/preprocessing/stpls3d/preprocess_stpls3d_ply.py
Author: Bole's Assistant
"""

import os
import argparse
import glob
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from tqdm import tqdm

# 依赖检查
try:
    from plyfile import PlyData
except ImportError:
    raise ImportError("请先安装依赖库: pip install plyfile")

def process_scene(scene_path, out_root, grid_size=0.1, chunk_size=(50, 50), chunk_stride=(25, 25)):
    """
    核心处理函数：读取PLY -> 体素下采样 -> 滑动窗口切块 -> 保存NPY
    """
    scene_path = Path(scene_path)
    # 获取文件名作为场景名 (去除后缀)
    scene_name = scene_path.stem
    
    # ------------------------------------------------------------------
    # 1. 读取 PLY 文件
    # ------------------------------------------------------------------
    try:
        with open(str(scene_path), 'rb') as f:
            plydata = PlyData.read(f)
        
        vertex = plydata['vertex']
        
        # 提取坐标 (x, y, z)
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
        
        # 提取颜色 (red, green, blue) - 处理可能缺失的情况
        if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
            colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.uint8)
        else:
            # 如果没有颜色，填充全0或全127
            colors = np.zeros_like(points, dtype=np.uint8)
        
        # 提取标签 (class 或 label) - 关键步骤！
        if 'class' in vertex:
            labels = np.array(vertex['class']).astype(np.int16)
        elif 'label' in vertex:
            labels = np.array(vertex['label']).astype(np.int16)
        else:
            # 如果完全没有标签（比如测试集），填充 255 (ignore_index)
            # print(f"⚠️ Warning: {scene_name} has no label field. Filling with 255.")
            labels = np.full(points.shape[0], 255, dtype=np.int16)
            
    except Exception as e:
        return f"❌ Error reading {scene_name}: {e}"

    # ------------------------------------------------------------------
    # 2. Grid Sampling (体素下采样)
    # ------------------------------------------------------------------
    # 这一步对于 5090 显存至关重要，哪怕是 24GB/32GB 显存，如果不下采样，几千万个点也会爆
    if grid_size is not None and grid_size > 0:
        scaled_coord = points / grid_size
        grid_coord = np.floor(scaled_coord).astype(int)
        
        # 获取体素去重后的索引
        _, indices = np.unique(grid_coord, axis=0, return_index=True)
        
        points = points[indices]
        colors = colors[indices]
        labels = labels[indices]

    # ------------------------------------------------------------------
    # 3. Sliding Window Chunking (滑动窗口切块)
    # ------------------------------------------------------------------
    # 归一化坐标以便计算切块位置
    coord_min = points.min(axis=0)
    coord_rel = points - coord_min
    
    x_max, y_max = coord_rel.max(axis=0)[:2]
    
    # 定义输出文件夹名称规范：grid0.10_chunk50x50_stride25x25
    folder_name = f"grid{grid_size:.2f}_chunk{chunk_size[0]}x{chunk_size[1]}_stride{chunk_stride[0]}x{chunk_stride[1]}"
    save_dir_root = out_root / folder_name
    
    chunk_idx = 0
    saved_count = 0
    
    # 生成滑动窗口坐标
    x_range = np.arange(0, x_max, chunk_stride[0])
    y_range = np.arange(0, y_max, chunk_stride[1])

    for x in x_range:
        for y in y_range:
            # 筛选当前窗口内的点
            mask = (
                (coord_rel[:, 0] >= x) & (coord_rel[:, 0] < x + chunk_size[0]) &
                (coord_rel[:, 1] >= y) & (coord_rel[:, 1] < y + chunk_size[1])
            )
            
            # 过滤掉极小的碎片块 (少于1000个点通常无法训练)
            if np.sum(mask) < 1000:
                continue
            
            # 提取数据
            chunk_points = points[mask]
            chunk_colors = colors[mask]
            chunk_labels = labels[mask]
            
            # 保存路径：SceneName_ChunkID
            chunk_name = f"{scene_name}_{chunk_idx}"
            chunk_save_path = save_dir_root / chunk_name
            chunk_save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存为 Pointcept 标准格式
            np.save(chunk_save_path / "coord.npy", chunk_points)
            np.save(chunk_save_path / "color.npy", chunk_colors)
            np.save(chunk_save_path / "segment.npy", chunk_labels)
            
            chunk_idx += 1
            saved_count += 1

    return f"✅ {scene_name}: Generated {saved_count} chunks"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to raw .ply folder")
    parser.add_argument("--output_root", type=str, required=True, help="Path to save processed .npy")
    parser.add_argument("--grid_size", type=float, default=0.1, help="Voxel size for downsampling")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size in meters")
    parser.add_argument("--chunk_stride", type=int, default=25, help="Stride size in meters")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_root)
    output_path = Path(args.output_root)
    
    # 搜索 .ply 文件
    ply_files = sorted(list(dataset_path.glob("*.ply")))
    
    if not ply_files:
        print(f"❌ Error: No .ply files found in {dataset_path}")
        # 尝试递归搜索
        print("   Trying recursive search...")
        ply_files = sorted(list(dataset_path.glob("**/*.ply")))
        if not ply_files:
            exit(1)

    print(f"🚀 Found {len(ply_files)} PLY files. Starting preprocessing...")
    print(f"⚙️  Config: Grid={args.grid_size}m, Chunk={args.chunk_size}m, Stride={args.chunk_stride}m")

    # 并行处理
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        results = list(tqdm(
            pool.map(
                process_scene,
                ply_files,
                repeat(output_path),
                repeat(args.grid_size),
                repeat((args.chunk_size, args.chunk_size)),
                repeat((args.chunk_stride, args.chunk_stride)),
            ),
            total=len(ply_files),
            desc="Preprocessing"
        ))

    print("\nProcessing Report:")
    for res in results:
        print(res)