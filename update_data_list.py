import os
from pathlib import Path

# 🚨 1. 这里修改为你刚才扫描时确认的、真实的绝对路径
# (根据你上一条命令的反馈，你的数据似乎在这里)
DATA_ROOT = Path("data/OpenDataLab__SensatUrban/data/processed")

def update_list(split):
    # 拼接路径: data/.../processed/train
    target_dir = DATA_ROOT / split
    
    # 双重检查路径是否存在
    if not target_dir.exists():
        # 尝试回退到默认路径看看 (兼容性检查)
        fallback_path = Path("data/sensaturban/processed") / split
        if fallback_path.exists():
            print(f"⚠️ 警告: 原路径 {target_dir} 不存在，切换到 {fallback_path}")
            target_dir = fallback_path
        else:
            print(f"❌ 错误: 找不到 {split} 文件夹！请检查路径配置。")
            print(f"   尝试寻找: {target_dir}")
            return

    print(f"🔍 正在扫描 {target_dir} ...")

    # 2. 获取当前硬盘上真正存在的文件夹名
    # 过滤掉非文件夹项
    real_folders = sorted([p.name for p in target_dir.iterdir() if p.is_dir()])
    
    if not real_folders:
        print(f"⚠️ {split} 目录是空的！")
        return

    # 3. 覆盖写入新的 txt 文件
    # txt 文件通常放在 split 文件夹内部，或者 processed 根目录
    # Pointcept DefaultDataset 默认会在 split 文件夹里找 (例如 processed/train/train.txt)
    txt_path = target_dir / f"{split}.txt"
    
    with open(txt_path, "w") as f:
        for name in real_folders:
            f.write(name + "\n")
            
    print(f"✅ 刷新成功: {txt_path}")
    print(f"   - 实际收录: {len(real_folders)} 个样本")

if __name__ == "__main__":
    # 刷新训练集
    update_list("train")
    # 刷新验证集
    update_list("val")