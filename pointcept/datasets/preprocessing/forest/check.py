import laspy
import numpy as np
from pathlib import Path

# 🚨 你的数据路径
LAZ_FILE = Path("../../../../data/bole___SegmentedForests/SegmentedForests/pointclouds/plot_01.laz")

def inspect_real_labels():
    if not LAZ_FILE.exists():
        print(f"❌ 找不到文件: {LAZ_FILE}")
        return

    print(f"🔍 正在读取: {LAZ_FILE.name} ...")
    las = laspy.read(LAZ_FILE)
    
    # 1. 检查自定义字段 'Class'
    if 'Class' in list(las.point_format.dimension_names):
        print("\n✅ 找到自定义字段 'Class'！正在统计...")
        
        # 读取自定义维度
        real_labels = np.array(las['Class'])
        unique_labels, counts = np.unique(real_labels, return_counts=True)
        
        print(f"\n📊 真实标签分布 (True Label Distribution):")
        print("-" * 40)
        print(f"{'Class ID':<10} | {'Count':<15} | {'Percentage':<10}")
        print("-" * 40)
        
        total = len(real_labels)
        for label, count in zip(unique_labels, counts):
            print(f"{label:<10} | {count:<15} | {count/total*100:.2f}%")
    else:
        print("❌ 没找到 'Class' 字段，请再次检查 dimension_names")

    # 2. 顺便看看 'Split' 是什么
    if 'Split' in list(las.point_format.dimension_names):
        split_val = np.unique(np.array(las['Split']))
        print(f"\nℹ️ Split 字段包含的值: {split_val}")

if __name__ == "__main__":
    inspect_real_labels()