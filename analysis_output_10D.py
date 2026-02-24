import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体和样式，方便直接放入论文 (Paper-ready plots)
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
sns.set_theme(style="whitegrid")

# SensatUrban 13 Classes
CLASS_NAMES = [
    "Ground", "Vegetation", "Building", "Wall", "Bridge", "Parking", 
    "Rail", "TrafficRoad", "StreetFurniture", "Car", "Footpath", "Bike", "Water"
]

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ... 保持 CLASS_NAMES 不变 ...

# 👇 [修改点 1] 设定你的 train 文件夹的根目录
# 注意：如果你之前用绝对路径没问题，强烈建议改成绝对路径，例如 "/data/datasets/OpenDataLab___SensatUrban/data/processed_10d/train"
TRAIN_DIR = Path("../Pointcept/data/OpenDataLab___SensatUrban/data/processed_10d/train")

# 👇 [修改点 2] 自动获取该目录下所有的子文件夹路径
# p.is_dir() 确保我们只拿文件夹，不拿 train.txt 这种文件
SAMPLE_CHUNKS = [str(p) for p in TRAIN_DIR.iterdir() if p.is_dir()]
# 👇 [修改点 1] 在获取所有的 block 后，随机打乱并只取前 100 个！
import random

SAMPLE_CHUNKS = [str(p) for p in TRAIN_DIR.iterdir() if p.is_dir()]
random.seed(42)
random.shuffle(SAMPLE_CHUNKS)
SAMPLE_CHUNKS = SAMPLE_CHUNKS[:100]  # 👈 限制为 100 个 Block，防止内存爆炸！

print(f"🔥 自动找到了 {len(SAMPLE_CHUNKS)} 个 Block 将进行抽样分析！")

def load_and_compute_features(chunk_paths, max_points_per_chunk=20000): # 稍微调小一点，2万完全够了
    all_dfs = []
    
    for path in chunk_paths:
        p = Path(path)
        if not p.exists():
            continue
            
        print(f"Loading data from {p.name}...")
        segment = np.load(p / "segment.npy").reshape(-1)
        extra_feat = np.load(p / "extra_feat.npy") 
        
        # 强制特征值降序排列
        eigenvalues = np.sort(extra_feat[:, :3], axis=1)[:, ::-1]
        l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        density = extra_feat[:, 3]
        
        eps = 1e-8
        l1_safe = l1 + eps
        linearity = (l1 - l2) / l1_safe
        planarity = (l2 - l3) / l1_safe
        scattering = l3 / l1_safe
        
        df = pd.DataFrame({
            "Class_ID": segment,
            "Lambda_1": l1,
            "Lambda_2": l2,
            "Lambda_3": l3,
            "Density": density,
            "Linearity": linearity,
            "Planarity": planarity,
            "Scattering": scattering
        })
        
        # 过滤掉 ignore_index
        df = df[df["Class_ID"] < len(CLASS_NAMES)]
        df["Class_Name"] = df["Class_ID"].apply(lambda x: CLASS_NAMES[x])
        
        # 🚨 [修复点 2] 修复采样 Bug：加入 min() 保护！
        if len(df) > max_points_per_chunk:
            is_rare = df["Class_Name"].isin(["Bike", "StreetFurniture", "Car"])
            df_rare = df[is_rare]
            df_common = df[~is_rare]
            
            # 确保要采样的数量不会超过 common 点的总数
            sample_size = min(len(df_common), max_points_per_chunk)
            
            df_common_sampled = df_common.sample(n=sample_size, random_state=42)
            df = pd.concat([df_rare, df_common_sampled])
            
        all_dfs.append(df)
        
    if not all_dfs:
        raise ValueError("没有成功加载任何数据！")
        
    return pd.concat(all_dfs, ignore_index=True)

def plot_feature_distributions(df, feature_name, save_dir="./analysis_output"):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    
    # 我们把 Bike 单独标红，其他颜色调淡，以突出对比
    palette = {name: "red" if name == "Bike" else "lightgray" for name in CLASS_NAMES}
    
    # 采用 Violin Plot 展示分布规律
    sns.violinplot(
        data=df, 
        x="Class_Name", 
        y=feature_name, 
        palette=palette,
        scale="width",
        inner="quartile" # 显示四分位数
    )
    
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribution of {feature_name} across SensatUrban Classes", fontweight='bold')
    plt.ylabel(feature_name)
    plt.xlabel("")
    plt.tight_layout()
    
    # 截断极值 (Outliers) 以获得更好的可视化效果
    lower = df[feature_name].quantile(0.01)
    upper = df[feature_name].quantile(0.99)
    plt.ylim(lower, upper)
    
    save_path = os.path.join(save_dir, f"{feature_name}_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 1. 抽取数据
    df_combined = load_and_compute_features(SAMPLE_CHUNKS)
    print(f"Total points analyzed: {len(df_combined)}")
    
    # 2. 如果点数太多导致画图极慢，可以随机降采样 (Random Subsampling)
    if len(df_combined) > 500000:
        print("Downsampling data for faster plotting...")
        df_combined = df_combined.sample(n=500000, random_state=42)
        
    # 3. 绘制我们关心的所有特征
    features_to_plot = ["Linearity", "Planarity", "Scattering", "Density", "Lambda_1"]
    
    for feat in features_to_plot:
        plot_feature_distributions(df_combined, feat)
        
    # 4. 打印均值统计表 (针对 Bike 的特殊分析)
    print("\n=== Mean Values per Class ===")
    mean_stats = df_combined.groupby("Class_Name")[features_to_plot].mean()
    print(mean_stats)
    
    # 重点查看 Bike 和易混淆类别 (Car, StreetFurniture) 的差异
    print("\n🔍 重点差异对比 (Bike vs StreetFurniture vs Car):")
    focus_classes = ["Bike", "StreetFurniture", "Car", "Vegetation"]
    focus_stats = mean_stats.loc[focus_classes]
    print(focus_stats)