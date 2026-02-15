import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm
import random

# 🚨 你的数据路径 (生成的那个文件夹)
DATA_DIR = Path("data/stpls3d/processed/grid0.10_chunk50x50_stride25x25")

def split_data():
    if not DATA_DIR.exists():
        print(f"❌ 找不到路径: {DATA_DIR}")
        return

    # 创建 train / val 文件夹
    (DATA_DIR / "train").mkdir(exist_ok=True)
    (DATA_DIR / "val").mkdir(exist_ok=True)

    # 获取所有 .npy 文件夹 (排除 train/val 自身)
    # 注意：preprocess_stpls3d_ply 生成的是文件夹，每个文件夹里有 coord.npy 等
    chunks = [f for f in DATA_DIR.iterdir() if f.is_dir() and f.name not in ["train", "val"]]

    if not chunks:
        print("❌ 当前目录下没有数据块！可能已经分好了？")
        return

    print(f"📦 找到 {len(chunks)} 个数据块，开始划分...")

    # 简单策略：按场景名划分 (RealWorld 1-15 Train, 16-20 Val)
    # 如果文件名是 "RealWorldData_Scene_01_chunk0" 这种格式
    # 或者如果是 USC_points 这种，就按 9:1 随机分
    
    # 你的文件名可能是: OCCC_points_0, RealWorldData_Scene_01_0 ...
    
    moves = {"train": 0, "val": 0}

    for chunk in tqdm(chunks):
        name = chunk.name
        target = "train" # 默认

        # 简单的验证集逻辑：如果文件名里包含 'Scene_16' 到 'Scene_20' -> Val
        # 或者如果是 USC / OCCC 数据，随机抽 10% 做 Val
        if "Scene_16" in name or "Scene_17" in name or "Scene_18" in name or "Scene_19" in name or "Scene_20" in name:
            target = "val"
        elif "USC" in name or "OCCC" in name or "WMSC" in name:
             # 对于这种单独的大场景，简单的 hash 取模来划分验证集
             # 比如 chunk_id % 10 == 0 的放入 val (10%)
             try:
                 chunk_id = int(name.split('_')[-1])
                 if chunk_id % 10 == 0:
                     target = "val"
             except:
                 pass
        
        # 移动文件夹
        shutil.move(str(chunk), str(DATA_DIR / target / name))
        moves[target] += 1

    print(f"✅ 完成！\n  Train: {moves['train']}\n  Val: {moves['val']}")

if __name__ == "__main__":
    split_data()