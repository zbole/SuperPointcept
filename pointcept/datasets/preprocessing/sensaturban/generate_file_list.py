import os
from pathlib import Path

# 🚨 你的数据根目录
DATA_ROOT = Path("../../../../../../data/datasets/OpenDataLab___SensatUrban/data/processed_10d")

def generate_list(split):
    target_dir = DATA_ROOT / split
    # 增加了一个 .resolve() 打印，这样万一找不到，你能立刻看到 Python 到底去哪找了
    if not target_dir.exists():
        print(f"❌ {split} 目录不存在! Python 试图寻找的路径是: {target_dir.resolve()}")
        return

    # 1. 获取所有子文件夹的名字 (即样本名)
    # 过滤掉 .DS_Store 或其他非文件夹文件
    sample_names = sorted([p.name for p in target_dir.iterdir() if p.is_dir()])
    
    if len(sample_names) == 0:
        print(f"⚠️ {split} 里面是空的！请检查切块是否成功")
        return

    # 2. 写入 {split}.txt (例如 train.txt)
    txt_path = target_dir / f"{split}.txt"
    with open(txt_path, "w") as f:
        for name in sample_names:
            f.write(name + "\n")
            
    print(f"✅ 已生成 {txt_path} (包含 {len(sample_names)} 个样本)")

if __name__ == "__main__":
    generate_list("train")
    generate_list("val")