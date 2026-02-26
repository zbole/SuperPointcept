import os
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from plyfile import PlyData

# ========================================================
# 硬编码参数设置 (Hardcoded Parameters)
# ========================================================
RAW_DATA_ROOT = Path("../../../../../../data/datasets/sensaturban/raw/")
OUT_DATA_ROOT = Path("../../../../../../data/datasets/sensaturban/processed_10d")

GRID_SIZE = 0.1         # 下采样格点大小
NUM_WORKERS = 2         # 进程并行数 (建议设为 CPU 核心数)
KNN_K = 16              # 计算几何特征的近邻数
CHUNK_SIZE = (50, 50)   # 切块大小 (meters)
CHUNK_STRIDE = (25, 25) # 切块步长 (meters)

# 验证集选取的 Block 名
VAL_BLOCKS = ["birmingham_block_1", "birmingham_block_6", "cambridge_block_12", "cambridge_block_6"]

# ========================================================
# 核心计算模块
# ========================================================

def extract_geometric_features(points, k=16):
    """使用 Open3D 高速提取几何特征值"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    n_points = len(points)
    eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
    
    points_np = np.asarray(pcd.points)
    for i in range(n_points):
        # 1. KNN 搜索
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points_np[i], k)
        neighbors = points_np[idx]
        
        if len(neighbors) >= 3:
            # 2. 计算局部协方差并求特征值
            cov = np.cov(neighbors.T)
            vals, _ = np.linalg.eigh(cov) # 默认从小到大: lambda_1, lambda_2, lambda_3
            
            # 3. 归一化 (让特征值表示概率：线性、平面、散射)
            sum_vals = np.sum(vals) + 1e-6
            eigenvalues[i] = vals / sum_vals # 结果在 [0, 1] 之间
            
    return eigenvalues

def process_scene(scene_path):
    scene_path = Path(scene_path)
    scene_name = scene_path.stem.lower()
    
    # 判断归属：train / val / test
    parent_folder = scene_path.parent.name # 'train' 或 'test'
    if parent_folder == "train":
        save_dir = OUT_DATA_ROOT / ("val" if any(bn in scene_name for bn in VAL_BLOCKS) else "train")
    else:
        save_dir = OUT_DATA_ROOT / "test"

    # 1. 读取 PLY 数据
    try:
        with open(str(scene_path), 'rb') as f:
            plydata = PlyData.read(f)
        vertex = plydata['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.uint8) if 'red' in vertex else np.zeros_like(points, dtype=np.uint8)
        labels = np.array(vertex['class' if 'class' in vertex else 'label']).astype(np.int16) if ('class' in vertex or 'label' in vertex) else np.full(points.shape[0], 255, dtype=np.int16)
    except Exception as e:
        return f"❌ Error {scene_name}: {e}"

    # 2. 坐标平移 & Grid Sampling
    points -= points.min(axis=0)
    
    # 提取 Grid Density (特征第 10 维)
    scaled_coord = points / GRID_SIZE
    grid_coord = np.floor(scaled_coord).astype(int)
    _, indices, counts = np.unique(grid_coord, axis=0, return_index=True, return_counts=True)
    
    points = points[indices]
    colors = colors[indices]
    labels = labels[indices]
    grid_density = counts.astype(np.float32).reshape(-1, 1) # 局部密度计数

    # 3. 计算 3 维特征值 (特征第 7, 8, 9 维)
    # 在下采样后的点上算，速度极快
    eigen_feats = extract_geometric_features(points, k=KNN_K)
    
    # 拼接成 4 维额外特征 (Eigenvalues + Density)
    extra_feat = np.concatenate([eigen_feats, grid_density], axis=1).astype(np.float32)

    # 4. 切块保存 (Chunking)
    x_max, y_max = points.max(axis=0)[:2]
    chunk_idx = 0
    
    for x in np.arange(0, x_max, CHUNK_STRIDE[0]):
        for y in np.arange(0, y_max, CHUNK_STRIDE[1]):
            mask = (points[:, 0] >= x) & (points[:, 0] < x + CHUNK_SIZE[0]) & \
                   (points[:, 1] >= y) & (points[:, 1] < y + CHUNK_SIZE[1])
            
            if np.sum(mask) < 1000: continue # 过滤点数太少的块
            
            chunk_path = save_dir / f"{scene_name}_{chunk_idx}"
            chunk_path.mkdir(parents=True, exist_ok=True)
            
            np.save(chunk_path / "coord.npy", points[mask])
            np.save(chunk_path / "color.npy", colors[mask])
            np.save(chunk_path / "segment.npy", labels[mask])
            np.save(chunk_path / "extra_feat.npy", extra_feat[mask]) # 关键：存储 4 维几何先验
            
            chunk_idx += 1

    return f"✅ {scene_name} finished. Chunks: {chunk_idx}"

# ========================================================
# 主程序入口
# ========================================================
if __name__ == "__main__":
    # 扫描所有 ply 文件
    ply_files = sorted(list(RAW_DATA_ROOT.glob("**/*.ply")))
    print(f"🚀 Total scenes: {len(ply_files)} | Workers: {NUM_WORKERS}")
    
    # 多进程执行
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        results = list(tqdm(pool.map(process_scene, ply_files), total=len(ply_files)))
        
    for res in results:
        print(res)