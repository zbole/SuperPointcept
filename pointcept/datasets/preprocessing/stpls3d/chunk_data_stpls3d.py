"""
STPLS3D Preprocessing Script (Robust 20-Class Version)
Author: Bole's Assistant (Based on Pointcept)

Usage:
    python pointcept/datasets/preprocessing/stpls3d/chunk_data_stpls3d.py \
        --dataset_root data/stpls3d \
        --output_root data/stpls3d/processed \
        --num_workers 16
"""

import os
import argparse
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


def chunking_scene(
    scene_path,
    out_root,
    grid_size=0.1,
    chunk_size=(50, 50),
    chunk_stride=(25, 25),
    chunk_min_points=1000,
):
    """
    Process a single scene: Load PLY -> Grid Sample -> Chunk -> Save NPY
    """
    scene_path = Path(scene_path)
    scene_name = scene_path.stem
    # STPLS3D specific: Determine split based on parent folder name if needed
    # e.g., RealWorldData -> realworld, Synthetic_v3 -> synthetic
    
    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    try:
        plydata = PlyData.read(str(scene_path))
        vertex = plydata['vertex']
        
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.uint8)
        
        # Robust label loading
        if 'class' in vertex:
            labels = np.array(vertex['class']).astype(np.int16)
        elif 'label' in vertex:
            labels = np.array(vertex['label']).astype(np.int16)
        else:
            # Fallback for unlabelled data (e.g. test set)
            labels = np.full(points.shape[0], 255, dtype=np.int16)
            
    except Exception as e:
        return f"❌ Error reading {scene_name}: {e}"

    # ------------------------------------------------------------------
    # 2. Grid Sampling
    # ------------------------------------------------------------------
    if grid_size is not None and grid_size > 0:
        scaled_coord = points / grid_size
        grid_coord = np.floor(scaled_coord).astype(int)
        _, indices = np.unique(grid_coord, axis=0, return_index=True)
        points = points[indices]
        colors = colors[indices]
        labels = labels[indices]

    # ------------------------------------------------------------------
    # 3. Sliding Window Chunking
    # ------------------------------------------------------------------
    coord_min = points.min(axis=0)
    coord_rel = points - coord_min
    x_max, y_max = coord_rel.max(axis=0)[:2]
    
    # Define output folder name
    # e.g. train_grid0.1_chunk50x50_stride25x25
    if grid_size:
        param_name = f"grid{grid_size:.2f}_chunk{chunk_size[0]}x{chunk_size[1]}_stride{chunk_stride[0]}x{chunk_stride[1]}"
    else:
        param_name = f"chunk{chunk_size[0]}x{chunk_size[1]}_stride{chunk_stride[0]}x{chunk_stride[1]}"

    # Determine split folder (train/val/test) - STPLS3D logic
    # RealWorld usually goes to one folder, Synthetic to another.
    # Here we simplify: Put everything under output_root/param_name/scene_name_chunks
    
    save_dir_root = Path(out_root) / param_name
    
    chunk_idx = 0
    saved_count = 0
    
    x_range = np.arange(0, x_max, chunk_stride[0])
    y_range = np.arange(0, y_max, chunk_stride[1])

    for x in x_range:
        for y in y_range:
            mask = (
                (coord_rel[:, 0] >= x) & (coord_rel[:, 0] < x + chunk_size[0]) &
                (coord_rel[:, 1] >= y) & (coord_rel[:, 1] < y + chunk_size[1])
            )
            
            if np.sum(mask) < chunk_min_points:
                continue
            
            # Save chunk
            chunk_name = f"{scene_name}_{chunk_idx}"
            chunk_save_path = save_dir_root / chunk_name
            chunk_save_path.mkdir(parents=True, exist_ok=True)
            
            np.save(chunk_save_path / "coord.npy", points[mask])
            np.save(chunk_save_path / "color.npy", colors[mask])
            np.save(chunk_save_path / "segment.npy", labels[mask])
            
            chunk_idx += 1
            saved_count += 1

    return f"✅ {scene_name}: {saved_count} chunks created."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=Path, help="Path to STPLS3D raw folders")
    parser.add_argument("--output_root", required=True, type=Path, help="Output folder")
    parser.add_argument("--grid_size", default=0.1, type=float)
    parser.add_argument("--chunk_size", default=50, type=int)
    parser.add_argument("--chunk_stride", default=25, type=int)
    parser.add_argument("--num_workers", default=mp.cpu_count(), type=int)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Auto-Discovery of PLY files in standard STPLS3D folders
    # ------------------------------------------------------------------
    target_folders = ["RealWorldData", "Synthetic_v3"] # Add others if needed
    ply_files = []
    
    print(f"🔍 Scanning for PLY files in {args.dataset_root}...")
    
    # 1. Search in specific subfolders first
    for folder in target_folders:
        search_path = args.dataset_root / folder
        if search_path.exists():
            found = list(search_path.glob("*.ply"))
            print(f"  - Found {len(found)} scenes in {folder}")
            ply_files.extend(found)
            
    # 2. Fallback: If no subfolders found, search root recursively
    if not ply_files:
        print("  - Target folders not found, searching recursively...")
        ply_files = list(args.dataset_root.glob("**/*.ply"))
    
    if not ply_files:
        print("❌ Error: No .ply files found! Check your dataset_root.")
        exit(1)
        
    print(f"🚀 Starting processing for {len(ply_files)} scenes...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        results = list(tqdm(
            pool.map(
                chunking_scene,
                ply_files,
                repeat(args.output_root),
                repeat(args.grid_size),
                repeat((args.chunk_size, args.chunk_size)),
                repeat((args.chunk_stride, args.chunk_stride)),
            ),
            total=len(ply_files),
            desc="Preprocessing"
        ))
        
    print("\nProcessing complete.")