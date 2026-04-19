import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

def save_ensemble_attention_ply(coords, original_colors, attention_weights, query_idx, is_bike_mask, save_path):
    """保存多重曝光后的全局注意力热力图 (Max-pooling 模式)"""
    num_points = coords.shape[0]
    final_colors = np.full((num_points, 3), [200, 200, 200], dtype=np.uint8)

    # 🚀 提取有关注的区域
    valid_indices = attention_weights > 0
    valid_attn = attention_weights[valid_indices]
    
    # 局部 Min-Max 归一化 (使用 99.5% 分位数剔除极值噪点)
    if valid_attn.size > 0:
        min_a = valid_attn.min()
        max_a = np.percentile(valid_attn, 99.5) 
        if max_a > min_a:
            norm_attn = np.clip((valid_attn - min_a) / (max_a - min_a), 0, 1)
        else:
            norm_attn = np.ones_like(valid_attn)
            
        cmap = plt.get_cmap('jet')
        mapped_colors = (cmap(norm_attn)[:, :3] * 255.0).astype(np.uint8)
        final_colors[valid_indices] = mapped_colors

    final_colors[query_idx] = [255, 0, 255] # 靶心高亮为洋红
    
    vertex = np.zeros(num_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('attn_weight', 'f4'),
        ('is_bike', 'u1')
    ])
    
    vertex['x'] = coords[:, 0]
    vertex['y'] = coords[:, 1]
    vertex['z'] = coords[:, 2]
    vertex['red'] = final_colors[:, 0]
    vertex['green'] = final_colors[:, 1]
    vertex['blue'] = final_colors[:, 2]
    vertex['attn_weight'] = attention_weights
    vertex['is_bike'] = is_bike_mask 
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(save_path)
    print(f"✅ Max-pooling 全局热力图已保存: {os.path.basename(save_path)}")

def load_raw_arrays(data_dir):
    coord = np.load(os.path.join(data_dir, "coord.npy"))
    color = np.load(os.path.join(data_dir, "color.npy"))
    extra_feat = np.load(os.path.join(data_dir, "extra_feat.npy"))
    segment = np.load(os.path.join(data_dir, "segment.npy")) 
    return coord, color, extra_feat, segment

def build_point_input(coord, color, extra_feat):
    feat = np.concatenate([coord, color, extra_feat], axis=-1)
    coords_tensor = torch.tensor(coord, dtype=torch.float32).cuda()
    feats_tensor = torch.tensor(feat, dtype=torch.float32).cuda()
    
    N = coords_tensor.shape[0]
    offset = torch.tensor([N], dtype=torch.int32).cuda()
    batch = torch.zeros(N, dtype=torch.long).cuda()
    grid_size = 0.1 
    grid_coord = torch.div(coords_tensor - coords_tensor.min(0)[0], grid_size, rounding_mode="trunc").int()

    return Point({
        "coord": coords_tensor,
        "feat": feats_tensor,
        "grid_coord": grid_coord,
        "offset": offset,
        "batch": batch
    })

@torch.no_grad()
def main():
    base_test_dir = "/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/special_out/val"
    ckpt_path = "./exp/sensaturban/SensatUrban_DSGG-PT_FinalEXP/model/model_best.pth"
    target_layer_idx = -1 
    
    # ==========================================
    # 🎯 核心控制参数：多次曝光聚合 (Max-pooling)
    # ==========================================
    NUM_ENSEMBLE = 8   # 设置为 8 次，配合 Max-pooling 足够照亮整个物体
    
    print(f"🚀 初始化模型 (启用多次随机拼接以提取最高 Attention，共曝光 {NUM_ENSEMBLE} 次)...")
    model = PointTransformerV3(
        in_channels=7,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True, # 🚀 必须开启！每次推理强制重新随机切分 Patch
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False, 
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('ScanNet', 'S3DIS', 'Structured3D', 'SensatUrban')
    ).cuda()
    
    print(f"📦 加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    cleaned_state_dict = {k.replace('backbone.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()

    chunk_dirs = [os.path.join(base_test_dir, d) for d in os.listdir(base_test_dir) if d.startswith("special_")]
    
    for data_dir in chunk_dirs:
        chunk_name = os.path.basename(data_dir)
        coord, color, extra_feat, segment = load_raw_arrays(data_dir)
        
        if 11 not in segment:
            continue
            
        print(f"\n" + "="*50)
        print(f"🌟 在 {chunk_name} 启动 Max-pooling 多重曝光捕获...")
        
        is_bike_mask = (segment == 11).astype(np.uint8)
        
        bike_indices = np.where(segment == 11)[0]
        bike_coords = coord[bike_indices]
        bike_center = np.mean(bike_coords, axis=0)
        closest_to_bike_center = np.argmin(np.linalg.norm(bike_coords - bike_center, axis=1))
        query_idx = bike_indices[closest_to_bike_center]
        print(f"🎯 统一锁定 Query 点 Index: {query_idx}")

        original_colors = color * 255.0
        
        # 🚀 初始化 Max-pooling 累加器为全 0
        max_attn_ours = np.zeros(coord.shape[0], dtype=np.float32)
        max_attn_base = np.zeros(coord.shape[0], dtype=np.float32)
        
        for i in range(NUM_ENSEMBLE):
            print(f"   ⏳ 正在执行第 {i+1}/{NUM_ENSEMBLE} 次随机切块推理...")
            
            # --- Ours (DPT) ---
            point_ours = build_point_input(coord, color, extra_feat)
            point_ours["extract_attn"] = True 
            out_ours = model(point_ours)
            
            attn_ours = out_ours["attn_maps"][target_layer_idx].mean(dim=1)
            order_ours = out_ours["attn_orders"][target_layer_idx].numpy()
            p_size_ours = out_ours["attn_patch_size"]
            
            loc_ours = np.where(order_ours == query_idx)[0][0]
            pidx_ours = loc_ours // p_size_ours
            in_pidx_ours = loc_ours % p_size_ours
            patch_gidx_ours = order_ours[pidx_ours * p_size_ours : (pidx_ours + 1) * p_size_ours]
            
            # 🚀 取当前层权重与历史最高权重的最大值
            current_weights_ours = attn_ours[pidx_ours, in_pidx_ours, :].numpy()
            max_attn_ours[patch_gidx_ours] = np.maximum(max_attn_ours[patch_gidx_ours], current_weights_ours)
            
            # --- Baseline (Zero-DINO) ---
            point_base = build_point_input(coord, color, np.zeros_like(extra_feat))
            point_base["extract_attn"] = True 
            out_base = model(point_base)
            
            attn_base = out_base["attn_maps"][target_layer_idx].mean(dim=1)
            order_base = out_base["attn_orders"][target_layer_idx].numpy()
            p_size_base = out_base["attn_patch_size"]
            
            loc_base = np.where(order_base == query_idx)[0][0]
            pidx_base = loc_base // p_size_base
            in_pidx_base = loc_base % p_size_base
            patch_gidx_base = order_base[pidx_base * p_size_base : (pidx_base + 1) * p_size_base]
            
            # 🚀 取当前层权重与历史最高权重的最大值
            current_weights_base = attn_base[pidx_base, in_pidx_base, :].numpy()
            max_attn_base[patch_gidx_base] = np.maximum(max_attn_base[patch_gidx_base], current_weights_base)

        # ==========================================
        # 💾 保存双子星结果 (Max-pooling 多重曝光版)
        # ==========================================
        save_ensemble_attention_ply(coord, original_colors, max_attn_ours, query_idx, is_bike_mask, f"{chunk_name}_MAX_ENSEMBLE{NUM_ENSEMBLE}_query{query_idx}_A_OURS.ply")
        save_ensemble_attention_ply(coord, original_colors, max_attn_base, query_idx, is_bike_mask, f"{chunk_name}_MAX_ENSEMBLE{NUM_ENSEMBLE}_query{query_idx}_B_ZERODINO.ply")

if __name__ == "__main__":
    main()