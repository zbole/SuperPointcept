import laspy
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 🚨 使用绝对路径，防止相对路径混乱
# 原始数据路径 (你的原始 .laz 位置)
RAW_DIR = Path("/workspace/Pointcept/data/bole___SegmentedForests/SegmentedForests/pointclouds")

# 🚨 输出路径 (处理后的 .npy 存放位置)
# 脚本会自动创建: processed/train, processed/val, processed/train.txt 等
OUTPUT_DIR = Path("/workspace/Pointcept/data/segmentedforests/processed")

# ✅ 标签映射策略
# 0->0, 1->1, 2->2, 3->3, 4->255(忽略)
LABEL_MAPPING = {
    0: 0,   # Ground
    1: 1,   # Low Veg
    2: 2,   # Stem (Tree Trunk)
    3: 3,   # Foliage (Leaves)
    4: 255, # Noise / Rare -> Ignore
}
# ===========================================

def process_plot(laz_path, split, output_root):
    scene_name = laz_path.stem # e.g., plot_01
    
    # 1. 读取 LAZ
    try:
        las = laspy.read(laz_path)
    except Exception as e:
        print(f"❌ 读取失败 {laz_path}: {e}")
        return False

    # 2. 提取坐标 (归一化到重心，保留 float32 精度)
    coord = np.vstack((las.x, las.y, las.z)).T
    coord -= np.min(coord, axis=0)
    
    # 3. 提取颜色 (如果有则归一化，没有则全黑)
    if hasattr(las, "red"):
        color = np.vstack((las.red, las.green, las.blue)).T
        # 16bit -> 8bit
        if np.max(color) > 255:
            color = (color / 256).astype(np.uint8)
        else:
            color = color.astype(np.uint8)
    else:
        color = np.zeros_like(coord, dtype=np.uint8)

    # 4. 提取标签 (从自定义字段 Class 读取)
    if 'Class' in list(las.point_format.dimension_names):
        raw_label = np.array(las['Class'])
    else:
        print(f"⚠️ {scene_name} 没有 'Class' 字段，跳过。")
        return False

    # 5. 应用映射
    segment = np.full_like(raw_label, 255, dtype=np.int16)
    for k, v in LABEL_MAPPING.items():
        segment[raw_label == k] = v

    # 6. 保存 (保持原名 plot_xx)
    save_dir = output_root / split / scene_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / "coord.npy", coord.astype(np.float32))
    np.save(save_dir / "color.npy", color)
    np.save(save_dir / "segment.npy", segment)
    
    return True

if __name__ == "__main__":
    # 🌲 官方推荐的划分标准
    splits = {
        "train": [f"plot_{i:02d}" for i in range(1, 10)],  # 01-09
        "val":   [f"plot_{i:02d}" for i in range(10, 13)], # 10-12
        "test":  [f"plot_{i:02d}" for i in range(13, 15)]  # 13-14
    }

    print(f"🚀 开始预处理...")
    print(f"📂 输入: {RAW_DIR}")
    print(f"📂 输出: {OUTPUT_DIR}")

    # 第一步：处理数据
    for split, plots in splits.items():
        print(f"\nProcessing {split} set...")
        for p_name in tqdm(plots):
            laz_file = RAW_DIR / f"{p_name}.laz"
            if laz_file.exists():
                success = process_plot(laz_file, split, OUTPUT_DIR)
                if not success:
                    print(f"⚠️ 处理失败: {p_name}")
            else:
                print(f"⚠️ 文件不存在: {laz_file}")

    # 第二步：生成名单 (train.txt, val.txt, test.txt)
    print("\n📝 正在生成索引文件 (File Lists)...")
    for split, plots in splits.items():
        # 检查该 split 下实际生成了哪些文件 (防止文件不存在导致名单虚假)
        valid_plots = []
        target_dir = OUTPUT_DIR / split
        
        if target_dir.exists():
            for p_name in plots:
                # 确认文件夹真的生成了
                if (target_dir / p_name).exists():
                    valid_plots.append(p_name)
        
        # 写入 txt
        list_file = OUTPUT_DIR / f"{split}.txt"
        with open(list_file, "w") as f:
            for p_name in valid_plots:
                # 写入格式: "split/plot_name" (例如 train/plot_01)
                f.write(f"{split}/{p_name}\n")
        
        print(f"✅ 生成 {list_file}: 包含 {len(valid_plots)} 个样本")

    print("\n🎉 全部完成！Ready for training!")