"""
Manual Validation Splitter for STPLS3D (USC_points)
Author: Bole's Assistant
"""

import os
import shutil
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm

def manual_split(processed_root, val_ratio=0.15):
    root = Path(processed_root)
    train_dir = root / "train"
    val_dir = root / "val"
    
    # 确保 val 文件夹存在
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 找到所有的 chunks
    # 注意：只找 USC_points 开头的（真实场景），合成数据全部留给训练
    print(f"🔍 Scanning {train_dir} for USC_points...")
    all_chunks = list(train_dir.iterdir())
    
    # 筛选出 RealWorld 数据 (USC_points)
    real_world_chunks = [p for p in all_chunks if "USC_points" in p.name]
    
    # 筛选出合成数据 (用于统计)
    synthetic_chunks = [p for p in all_chunks if "USC_points" not in p.name]
    
    print(f"📊 Statistics:")
    print(f"  - Real World (USC): {len(real_world_chunks)} chunks")
    print(f"  - Synthetic:        {len(synthetic_chunks)} chunks")
    print(f"  - Total Train:      {len(all_chunks)} chunks")
    
    if len(real_world_chunks) == 0:
        print("❌ No 'USC_points' found! Please check chunk names in train folder.")
        # 如果名字不是 USC_points，尝试打印前几个看看是什么
        if len(all_chunks) > 0:
            print(f"   Example chunk name: {all_chunks[0].name}")
        return

    # 2. 按索引排序 (保证空间连续性)
    # 文件名通常是 USC_points_0, USC_points_1 ...
    # 我们提取最后的数字进行排序
    try:
        real_world_chunks.sort(key=lambda x: int(x.name.split('_')[-1]))
    except Exception as e:
        print(f"⚠️ Sorting failed, using default order. Error: {e}")
    
    # 3. 计算切分点 (后 15% 做验证)
    split_idx = int(len(real_world_chunks) * (1 - val_ratio))
    val_chunks = real_world_chunks[split_idx:]
    
    print(f"✂️  Moving last {len(val_chunks)} chunks ({val_ratio*100}%) to 'val'...")
    
    # 4. 执行移动
    for chunk_path in tqdm(val_chunks, desc="Moving to Val"):
        target_path = val_dir / chunk_path.name
        shutil.move(str(chunk_path), str(target_path))
        
    print("\n✅ Split Complete!")
    print(f"  - Final Train: {len(list(train_dir.iterdir()))}")
    print(f"  - Final Val:   {len(list(val_dir.iterdir()))}")

if __name__ == "__main__":
    # 🚨 你的数据路径
    TARGET_DIR = "data/stpls3d/processed/grid0.10_chunk50x50_stride25x25"
    manual_split(TARGET_DIR)