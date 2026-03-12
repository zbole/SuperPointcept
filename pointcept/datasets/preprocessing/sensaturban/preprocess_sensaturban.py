import os
import sys
from pathlib import Path

# 优化多线程，防止与 DataLoaders 冲突
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from scipy.spatial import cKDTree
from torchvision import transforms
from PIL import Image
import timm
import warnings

warnings.filterwarnings("ignore")

# ========================================================
# 1. 路径与全局配置
# ========================================================
dataset_root = Path("../../../../../datasets/sensaturban").resolve()
RAW_DATA_ROOT = dataset_root / "raw"
# 最终特征维度：3(Eigen) + 1(Density) + 768(DINO) = 772 维
OUT_DATA_ROOT = dataset_root / "processed_772d" 
OUT_DATA_ROOT.mkdir(parents=True, exist_ok=True)

# DINOv3 权重路径
WEIGHT_PATH = "/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/hf_cache/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================================
# 2. 模型加载 (针对 GH200 优化的 DINOv3)
# ========================================================
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    print(f"🚀 Loading DINOv3 ViT-B/16 on {DEVICE}...")
    # 开启 dynamic_img_size 以适配不同比例的 BEV 图像
    model = timm.create_model('vit_base_patch16_224', pretrained=False, dynamic_img_size=True).to(DEVICE)
    state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
    if "model" in state_dict: state_dict = state_dict["model"]
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Weights injected. Missing keys: {len(msg.missing_keys)}")
    return model.eval()

model = load_model()

# ========================================================
# 3. 核心特征计算模块
# ========================================================

def extract_geometric_features_vectorized(points, k=16):
    """
    向量化提取局部几何特征：Linearity, Planarity, Scattering
    """
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k, workers=-1)
    neighbors = points[idx] # [N, K, 3]
    
    centered = neighbors - np.mean(neighbors, axis=1, keepdims=True)
    # 批量计算 3x3 协方差矩阵
    cov = np.matmul(centered.transpose(0, 2, 1), centered) / (k - 1)
    
    # 特征值求解 (λ1 >= λ2 >= λ3)
    evals = np.linalg.eigvalsh(cov)
    evals = np.flip(evals, axis=1) # 降序
    evals = np.maximum(evals, 1e-8)
    
    sum_vals = np.sum(evals, axis=1, keepdims=True)
    l1, l2, l3 = evals[:, 0:1], evals[:, 1:2], evals[:, 2:3]
    
    linearity = (l1 - l2) / sum_vals
    planarity = (l2 - l3) / sum_vals
    scattering = l3 / sum_vals
    
    return np.concatenate([linearity, planarity, scattering], axis=1).astype(np.float32)

def extract_dino_semantics_bev(points, colors, grid_size=0.1):
    """
    BEV 投影并通过 DINOv3 提取 768 维语义先验
    """
    try:
        x_min, y_min = np.min(points[:, :2], axis=0)
        x_max, y_max = np.max(points[:, :2], axis=0)
        dx, dy = max(x_max - x_min, 0.1), max(y_max - y_min, 0.1)
        
        # 针对 GH200 内存充足的情况，适当提高最大分辨率上限
        MAX_RES = 1600 
        curr_grid = grid_size
        if max(dx, dy) / curr_grid > MAX_RES:
            curr_grid = max(dx, dy) / MAX_RES

        width, height = int(np.ceil(dx/curr_grid))+1, int(np.ceil(dy/curr_grid))+1
        
        # 向量化 Top-Z 投影逻辑
        # 1. 按照 Z 轴升序排序，利用覆盖效果实现 Top-Z
        sort_idx = np.argsort(points[:, 2])
        pts_sorted = points[sort_idx]
        colors_sorted = colors[sort_idx]
        
        u = np.clip(np.floor((pts_sorted[:, 0] - x_min) / curr_grid).astype(int), 0, width - 1)
        v = np.clip(np.floor((pts_sorted[:, 1] - y_min) / curr_grid).astype(int), 0, height - 1)
        
        bev_img = np.zeros((height, width, 3), dtype=np.uint8)
        bev_img[v, u] = colors_sorted 

        # DINO 推理流程
        img_t = preprocess(Image.fromarray(bev_img)).unsqueeze(0).to(DEVICE)
        
        # Padding 使尺寸能被 16 (Patch Size) 整除
        pad_h = (16 - img_t.shape[2] % 16) % 16
        pad_w = (16 - img_t.shape[3] % 16) % 16
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h))
        pad_height, pad_width = img_t.shape[2], img_t.shape[3]

        with torch.no_grad():
            # 使用 BF16 (GH200 原生支持极佳) 进一步优化性能
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feat = model.forward_features(img_t) # [1, SeqLen, 768]
                
            # 剔除第一个 CLS Token
            patch_tokens = feat[:, 1:, :] 
            Hidden_Dim = patch_tokens.shape[-1]
            
            grid_H, grid_W = pad_height // 16, pad_width // 16
            feat_2d = patch_tokens.reshape(1, grid_H, grid_W, Hidden_Dim).permute(0, 3, 1, 2)

            # 双线性插值回到 BEV 原始分辨率
            upsampled = F.interpolate(feat_2d.float(), size=(pad_height, pad_width), mode='bilinear', align_corners=False)
            # 截取掉 Padding 部分
            feat_map = upsampled[0, :, :height, :width].permute(1, 2, 0).cpu().numpy() # [H, W, 768]

        # 映射回原始点顺序
        u_orig = np.clip(np.floor((points[:, 0] - x_min) / curr_grid).astype(int), 0, width - 1)
        v_orig = np.clip(np.floor((points[:, 1] - y_min) / curr_grid).astype(int), 0, height - 1)
        
        point_feats = feat_map[v_orig, u_orig]
        return point_feats.astype(np.float16)

    except Exception as e:
        print(f"⚠️ DINO Extraction Error: {e}")
        return np.zeros((len(points), 768), dtype=np.float16)

# ========================================================
# 4. 场景处理与分块保存流程
# ========================================================

def process_scene(scene_path):
    scene_path = Path(scene_path)
    scene_name = scene_path.stem.lower()
    
    # 验证集分配
    if "train" in str(scene_path):
        VAL_LIST = ["birmingham_block_1", "birmingham_block_6", "cambridge_block_12", "cambridge_block_6"]
        split = "val" if any(b in scene_name for b in VAL_LIST) else "train"
    else:
        split = "test"
    
    save_dir = OUT_DATA_ROOT / split
    save_dir.mkdir(parents=True, exist_ok=True)

    # 读取原始 PLY
    try:
        with open(str(scene_path), 'rb') as f:
            v = PlyData.read(f)['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
        sem = v['class'].astype(np.int16) if 'class' in v else np.full(len(pts), 255)
    except Exception as e: return f"❌ Load Error: {e}"

    # 1. 0.1m 体素降采样 (PTV3 标准预处理)
    pts -= pts.min(axis=0)
    grid_coord = np.floor(pts / 0.1).astype(int)
    _, idx, counts = np.unique(grid_coord, axis=0, return_index=True, return_counts=True)
    pts, rgb, sem = pts[idx], rgb[idx], sem[idx]
    
    # 2. 计算 4D 几何特征
    density = (counts.astype(np.float32).reshape(-1, 1) / counts.max()).astype(np.float32)
    eigen = extract_geometric_features_vectorized(pts)
    geo_prior = np.concatenate([eigen, density], axis=1).astype(np.float32)
    
    # 3. 提取 768D DINO 语义特征
    dino_prior = extract_dino_semantics_bev(pts, rgb)
    
    # 4. 空间切块 (Chunking) - 50m 大小, 25m 步长
    x_max, y_max = pts.max(axis=0)[:2]
    chunk_count = 0
    
    for x in np.arange(0, x_max, 25):
        for y in np.arange(0, y_max, 25):
            mask = (pts[:, 0] >= x) & (pts[:, 0] < x + 50) & (pts[:, 1] >= y) & (pts[:, 1] < y + 50)
            if np.sum(mask) < 1024: continue # 过滤点数太少的块
            
            chunk_folder = save_dir / f"{scene_name}_{chunk_count}"
            chunk_folder.mkdir(exist_ok=True)
            
            # 拼接 4D Geo + 768D DINO = 772 维 extra_feat
            # 统一为 float16 以在 GH200 的统一内存中占用更少空间
            extra_feat = np.hstack([geo_prior[mask].astype(np.float16), dino_prior[mask]])
            
            # 写入文件
            np.save(chunk_folder / "coord.npy", pts[mask])
            np.save(chunk_folder / "color.npy", rgb[mask].astype(np.uint8))
            np.save(chunk_folder / "segment.npy", sem[mask].astype(np.int16))
            np.save(chunk_folder / "extra_feat.npy", extra_feat)
            
            chunk_count += 1

    return f"✅ {scene_name} processed: {chunk_count} chunks generated."

if __name__ == "__main__":
    ply_files = sorted(list(RAW_DATA_ROOT.glob("**/*.ply")))
    print(f"🚀 Found {len(ply_files)} PLY files. Starting GH200-Optimized Pipeline...")
    for f in tqdm(ply_files):
        print(process_scene(f))