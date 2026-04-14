import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

# ==============================================================================
# 🚀 显式导入你的模型结构 (解决 NameError)
# ==============================================================================
from pointcept.models.utils.structure import Point
# 注意：确保这个导入路径能找到你修改过 (带 hook) 的 PointTransformerV3 类
# 如果它是放在 pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py 中，使用下面这行：
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

# 如果你把整个代码保存在了跟这个脚本同级的单独文件 (例如 model_ptv3.py)，则使用：
# from model_ptv3 import PointTransformerV3
# ==============================================================================

def save_attention_ply(coords, attention_weights, query_idx, save_path):
    """保存为带热力图颜色的 PLY 文件"""
    num_points = coords.shape[0]
    cmap = plt.get_cmap('jet')
    
    if np.any(attention_weights > 0):
        max_attn = np.percentile(attention_weights[attention_weights > 0], 99.5)
    else:
        max_attn = 1.0
        
    norm_weights = np.clip(attention_weights / (max_attn + 1e-8), 0, 1)
    colors = cmap(norm_weights)[:, :3] * 255.0  
    colors = colors.astype(np.uint8)
    
    # 将 Query 中心点高亮为耀眼的亮粉色，方便你在软件里寻找
    colors[query_idx] = [255, 0, 255] 
    
    vertex = np.zeros(num_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('attn_weight', 'f4') 
    ])
    
    vertex['x'] = coords[:, 0]
    vertex['y'] = coords[:, 1]
    vertex['z'] = coords[:, 2]
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]
    vertex['attn_weight'] = attention_weights
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(save_path)
    print(f"✅ [大功告成] Attention 热力图已保存至: {os.path.abspath(save_path)}")

def load_sensaturban_chunk(data_dir):
    """精准适配你截图里的数据结构读取"""
    print(f"📂 正在加载点云块: {data_dir}")
    
    coord_path = os.path.join(data_dir, "coord.npy")
    color_path = os.path.join(data_dir, "color.npy")
    extra_feat_path = os.path.join(data_dir, "extra_feat.npy")
    
    coord = np.load(coord_path)
    color = np.load(color_path)
    extra_feat = np.load(extra_feat_path) # 包含 1D_Z + 1024D_DINO
    
    # [XYZ(3) + RGB(3) + RelZ_and_DINO(1025)] = 1031D
    feat = np.concatenate([coord, color, extra_feat], axis=-1)
    
    coords_tensor = torch.tensor(coord, dtype=torch.float32).cuda()
    feats_tensor = torch.tensor(feat, dtype=torch.float32).cuda()
    
    N = coords_tensor.shape[0]
    offset = torch.tensor([N], dtype=torch.int32).cuda()
    batch = torch.zeros(N, dtype=torch.long).cuda()
    
    # 匹配 config 中的 grid_size=0.1
    grid_size = 0.1 
    grid_coord = torch.div(coords_tensor - coords_tensor.min(0)[0], grid_size, rounding_mode="trunc").int()

    return {
        "coord": coords_tensor,
        "feat": feats_tensor,
        "grid_coord": grid_coord,
        "offset": offset,
        "batch": batch,
    }

@torch.no_grad()
def main():
    # ==========================================
    # 1. 路径配置
    # ==========================================
    data_dir = "/lus/lfs1aip2/projects/b6ae/datasets/sensaturban/special_out/test/special_0"
    ckpt_path = "./exp/sensaturban/SensatUrban_DSGG-PT_FinalEXP/model/model_best.pth"
    target_layer_idx = -1 

    # ==========================================
    # 2. 初始化你的 SOTA 模型 (严格按照 config 参数)
    # ==========================================
    print("🚀 正在初始化 PointTransformerV3 (强制禁用 Flash Attention)...")
    
    # 依据你提供的 config 初始化参数
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
        enable_flash=False, # 🛑 强制覆盖 config 里的 True，设为 False 才能截获 Attention！
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
    
    # 提取 state_dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # 清理 DataParallel 带来的 'module.' 前缀 (如果有)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # 如果模型存的是 DefaultSegmentorV2，里面会有 'backbone.' 前缀
        if k.startswith('backbone.'):
            cleaned_state_dict[k.replace('backbone.', '')] = v
        # 或者纯粹去 module
        elif k.startswith('module.'):
            cleaned_state_dict[k.replace('module.', '')] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()

    # ==========================================
    # 3. 前向传播
    # ==========================================
    data_dict = load_sensaturban_chunk(data_dir)
    point = Point(data_dict)
    
    # 🚀 触发 Hook
    point.extract_attn = True 
    
    print(f"🧠 正在进行模型推理 (点数: {data_dict['coord'].shape[0]:,})...")
    output_point = model(point)
    
    if not hasattr(output_point, 'attn_maps'):
        raise ValueError("❌ 找不到 Attention maps！请确认代码里 `extract_attn` 的 Hook 是否注入成功。")

    # ==========================================
    # 4. 提取并映射中心点的 Attention
    # ==========================================
    print(f"🔍 正在提取第 {target_layer_idx} 层的 Attention Map...")
    attn_map = output_point.attn_maps[target_layer_idx]
    order = output_point.attn_orders[target_layer_idx]
    patch_size = output_point.attn_patch_size
    
    attn_map = attn_map.mean(dim=1) 
    
    coords = output_point.coord.cpu().numpy()
    order_np = order.numpy()
    N = coords.shape[0]
    
    center_coord = np.mean(coords, axis=0)
    query_idx = np.argmin(np.linalg.norm(coords - center_coord, axis=1))
    
    global_attn = np.zeros(N, dtype=np.float32)
    order_loc = np.where(order_np == query_idx)[0][0]
    
    patch_idx = order_loc // patch_size
    within_patch_idx = order_loc % patch_size
    
    print(f"🎯 选定 Query 中心点 Index: {query_idx}, 所属 Patch ID: {patch_idx}")
    
    patch_attn = attn_map[patch_idx, within_patch_idx, :].numpy()
    patch_global_indices = order_np[patch_idx * patch_size : (patch_idx + 1) * patch_size]
    
    global_attn[patch_global_indices] = patch_attn
    
    # ==========================================
    # 5. 保存结果
    # ==========================================
    output_name = f"special_0_attn_layer{target_layer_idx}_query{query_idx}.ply"
    save_attention_ply(coords, global_attn, query_idx, output_name)

if __name__ == "__main__":
    main()