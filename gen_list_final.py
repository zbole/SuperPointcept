import os
from pathlib import Path

# 🚨 使用绝对路径，一个字符都不许差
DATA_ROOT = Path("/workspace/Pointcept/data/OpenDataLab___SensatUrban/data/processed")

def make_list(split):
    target_dir = DATA_ROOT / split
    
    # 调试信息：确认 Python 到底看到了什么
    print(f"Checking: {target_dir}")
    if not target_dir.exists():
        print(f"❌ 错误: 找不到路径 {target_dir}")
        return
    if not target_dir.is_dir():
        print(f"❌ 错误: {target_dir} 居然不是文件夹？")
        return

    # 扫描文件名
    names = sorted([p.name for p in target_dir.iterdir() if p.is_dir()])
    
    # 写入 txt (放在 train 文件夹内部)
    save_path = target_dir / f"{split}.txt"
    with open(save_path, "w") as f:
        for n in names:
            f.write(n + "\n")
            
    print(f"✅ 成功生成: {save_path} (包含 {len(names)} 个样本)")

if __name__ == "__main__":
    make_list("train")
    make_list("val")
    # 如果有 test 也可以加上
    # make_list("test")
