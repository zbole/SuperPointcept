import os
from pathlib import Path

# 指向 processed 根目录
DATA_ROOT = Path("/workspace/Pointcept/data/OpenDataLab___SensatUrban/data/processed")

def make_list(split):
    # 数据实际所在的子文件夹
    split_dir = DATA_ROOT / split
    
    if not split_dir.exists():
        print(f"❌ 找不到: {split_dir}")
        return

    # 扫描子文件夹里的 chunk 名
    names = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    
    # 🚨 关键修改：生成的 txt 放在 processed 根目录
    save_path = DATA_ROOT / f"{split}.txt"
    
    with open(save_path, "w") as f:
        for n in names:
            # 🚨 关键修改：写入 "split/chunk_name" 格式 (例如 train/cambridge_block_0)
            f.write(f"{split}/{n}\n")
            
    print(f"✅ 生成 {save_path} (包含 {len(names)} 个样本)")

if __name__ == "__main__":
    make_list("train")
    make_list("val")
    make_list("test")
