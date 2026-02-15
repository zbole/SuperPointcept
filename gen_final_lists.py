import os
from pathlib import Path

# 🚨 指向你刚刚生成数据的目录
DATA_ROOT = Path("data/OpenDataLab___SensatUrban/data/processed")

def make_list(split):
    target_dir = DATA_ROOT / split
    if not target_dir.exists():
        print(f"⚠️  跳过 {split}: 文件夹不存在")
        return

    # 扫描文件夹名
    names = sorted([p.name for p in target_dir.iterdir() if p.is_dir()])
    
    # 写入 txt (放在 split 文件夹内部，例如 processed/train/train.txt)
    save_path = target_dir / f"{split}.txt"
    with open(save_path, "w") as f:
        for n in names:
            f.write(n + "\n")
            
    print(f"✅ 生成 {split}.txt: 包含 {len(names)} 个样本")

if __name__ == "__main__":
    make_list("train")
    make_list("val")
    make_list("test")
