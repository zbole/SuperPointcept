import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

# ==============================================================================
# 🚀 显式导入你的模型结构
# ==============================================================================
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

def save_attention_ply(coords, original_colors, attention_weights, query_idx, patch_indices, save_path):
    """保存为高对比度热力图 PLY 文件：背景灰度化，高亮 Attention 区域"""
    num_points = coords.shape[0]
    
    # 1. 默认将整个背景设为浅灰色
    final_colors = np.full((num_points, 3), [200, 200, 200], dtype=np.uint8)

    # 2. 提取有效 Patch 的注意力权重
    patch_attn = attention_weights[patch_indices]
    
    # 3. 🚀 局部 Min-Max 归一化：强行放大特征差异！
    min_a = patch_attn.min()
    max_a = patch_attn.max()
    
    if max_a > min_a:
        norm_patch_attn = (patch_attn - min_a) / (max_a - min_a)
    else:
        norm_patch_attn = np.ones_like(patch_attn)

    # 4. 映射到 Jet 色板
    cmap = plt.get_cmap('jet')
    patch_colors = (cmap(norm_patch_attn)[:, :3] * 255.0).astype(np.uint8)
    
    # 5. 涂回到全局颜色矩阵
    final_colors[patch_indices] = patch_colors
    
    # 6. 将 Query 中心点高亮为耀眼的纯洋红色，如同靶心
    final_colors[query_idx] = [255, 0, 255] 
    
    # 7. 保存
    vertex = np.zeros(num_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('attn_weight', 'f4') 
    ])
    
    vertex['x'] = coords[:, 0]
    vertex['y'] = coords[:, 1]
    vertex['z'] = coords[:, 2]
    vertex['red'] = final_colors[:, 0]
    vertex['green'] = final_colors[:, 1]
    vertex['blue'] = final_colors[:, 2]
    vertex['attn_weight'] = attention_weights
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(save_path)
    print(f"✅ [高对比度版本] Attention 热力图已保存至: {os.path.abspath(save_path)}")

def load_sensaturban_chunk(data_dir):
    """精准适配数据结构读取，增加 segment 标签"""
    print(f"📂 正在加载点云块: {data_dir}")
    
    coord_path = os.path.join(data_dir, "coord.npy")
    color_path = os.path.join(data_dir, "color.npy")
    extra_feat_path = os.path.join(data_dir, "extra_feat.npy")
    segment_path = os.path.join(data_dir, "segment.npy") 
    
    coord = np.load(coord_path)
    color = np.load(color_path)
    extra_feat = np.load(extra_feat_path) 
    segment = np.load(segment_path) 
    
    feat = np.concatenate([coord, color, extra_feat], axis=-1)
    
    coords_tensor = torch.tensor(coord, dtype=torch.float32).cuda()
    feats_tensor = torch.tensor(feat, dtype=torch.float32).cuda()
    
    N = coords_tensor.shape[0]
    offset = torch.tensor([N], dtype=torch.int32).cuda()
    batch = torch.zeros(N, dtype=torch.long).cuda()
    grid_size = 0.1 
    grid_coord = torch.div(coords_tensor - coords_tensor.min(0)[0], grid_size, rounding_mode="trunc").int()

    return {
        "coord": coords_tensor,
        "feat": feats_tensor,
        "grid_coord": grid_coord,
        "offset": offset,
        "batch": batch,
        "segment": segment 
    }

@torch.no_grad()
def main():
    # ==========================================
    # 1. 路径配置
    # ==========================================
    # 🚀 直接指向 test 目录，我们会遍历它下面的所有 special_x 文件夹
    base_test_dir = "/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/special_out/val"
    ckpt_path = "./exp/sensaturban/SensatUrban_DSGG-PT_FinalEXP/model/model_best.pth"
    target_layer_idx = -1 

    # ==========================================
    # 2. 初始化你的 SOTA 模型
    # ==========================================
    print("🚀 正在初始化 PointTransformerV3 (强制禁用 Flash Attention)...")
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
        shuffle_orders=True,
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
    
    print(f"📦 正在加载最佳权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            cleaned_state_dict[k.replace('backbone.', '')] = v
        elif k.startswith('module.'):
            cleaned_state_dict[k.replace('module.', '')] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()

    # ==========================================
    # 3. 遍历扫街：寻找包含自行车的 chunk
    # ==========================================
    # 获取所有的 special_X 文件夹
    chunk_dirs = [os.path.join(base_test_dir, d) for d in os.listdir(base_test_dir) if d.startswith("special_")]
    
    bike_found_anywhere = False

    for data_dir in chunk_dirs:
        chunk_name = os.path.basename(data_dir)
        print(f"\n" + "="*50)
        print(f"🚦 正在探索点云块: {chunk_name}")
        
        # 预检：先读取 segment 看有没有自行车
        segment_path = os.path.join(data_dir, "segment.npy")
        if os.path.exists(segment_path):
            temp_segment = np.load(segment_path)
            unique_classes = np.unique(temp_segment) # 🚀 获取当前块里所有的类别 ID
            
            print(f"📊 当前块包含的类别 IDs: {unique_classes}") # 🚀 打印出来！
            
            if 11 not in temp_segment:
                print(f"⏩ {chunk_name} 中没有自行车 (ID=11)，直接跳过推理。")
                continue
        
        # 发现了自行车！开始前向传播
        bike_found_anywhere = True
        print(f"🌟 在 {chunk_name} 中发现目标！开始执行模型推理...")
        
        data_dict = load_sensaturban_chunk(data_dir)
        point = Point(data_dict)
        point["extract_attn"] = True 
        
        output_point = model(point)
        
        # 提取 Attention Map
        attn_map = output_point["attn_maps"][target_layer_idx]
        order = output_point["attn_orders"][target_layer_idx]
        patch_size = output_point["attn_patch_size"]
        
        attn_map = attn_map.mean(dim=1) 
        coords = output_point.coord.cpu().numpy()
        order_np = order.numpy()
        N = coords.shape[0]
        
        # 锁定自行车
        segment = data_dict["segment"]
        bike_indices = np.where(segment == 11)[0]
        
        bike_coords = coords[bike_indices]
        bike_center = np.mean(bike_coords, axis=0)
        closest_to_bike_center = np.argmin(np.linalg.norm(bike_coords - bike_center, axis=1))
        query_idx = bike_indices[closest_to_bike_center]
        print(f"🚴 成功锁定自行车！总共发现 {len(bike_indices)} 个自行车点。")
        
        global_attn = np.zeros(N, dtype=np.float32)
        order_loc = np.where(order_np == query_idx)[0][0]
        
        patch_idx = order_loc // patch_size
        within_patch_idx = order_loc % patch_size
        
        print(f"🎯 最终选定 Query 点 Index: {query_idx}, 所属 Patch ID: {patch_idx}")
        
        patch_attn = attn_map[patch_idx, within_patch_idx, :].numpy()
        patch_global_indices = order_np[patch_idx * patch_size : (patch_idx + 1) * patch_size]
        global_attn[patch_global_indices] = patch_attn
        
        original_colors = data_dict['feat'][:, 3:6].cpu().numpy() * 255.0

        output_name = f"{chunk_name}_attn_layer{target_layer_idx}_query{query_idx}_BIKE.ply"
        save_attention_ply(
            coords=coords, 
            original_colors=original_colors, 
            attention_weights=global_attn, 
            query_idx=query_idx, 
            patch_indices=patch_global_indices,
            save_path=output_name
        )

    if not bike_found_anywhere:
        print("\n😭 扫街结束：你提取的所有 special_X 块里居然一辆自行车都没有！")

if __name__ == "__main__":
    main()