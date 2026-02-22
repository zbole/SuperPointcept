import numpy as np
import os
import glob
from plyfile import PlyData, PlyElement
from tqdm import tqdm

# ================= 配置区域 =================
BASE_SCENE_NAME = "OCCC_points"

DATA_DIR = "/workspace/Pointcept/data/stpls3d/processed/grid0.10_chunk50x50_stride25x25/val"
PRED_DIR = "/workspace/Pointcept/exp/stpls3d/STPLS3D-PTV3-6cls/result"
OUTPUT_PLY = f"{BASE_SCENE_NAME}_FULL_PRED.ply"

# ✅ 加入你的 CategoryMapping 字典
MAPPING_DICT = {
    0: 0, 15: 0, 18: 0, 19: 0,  # Ground
    1: 1, 17: 1,                # Building
    2: 2, 3: 2, 4: 2,           # Vegetation
    5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, # Vehicle
    11: 4,                      # LightPole
    14: 5,                      # Fence
    12: 255, 13: 255            # Ignore
}

# 合并后的 6 类颜色映射
COLOR_MAP = {
    0: [178, 178, 178], # Ground: 灰色
    1: [204, 25, 25],   # Building: 红色
    2: [25, 204, 25],   # Vegetation: 绿色
    3: [25, 25, 204],   # Vehicle: 蓝色
    4: [229, 229, 25],  # LightPole: 黄色
    5: [229, 127, 25],  # Fence: 橙色
    255: [0, 0, 0]      # Ignore: 黑色
}
# ===========================================

def merge_and_export():
    print(f"🔍 正在搜寻 {BASE_SCENE_NAME} 的所有预测碎块...")
    search_pattern = os.path.join(PRED_DIR, f"val-{BASE_SCENE_NAME}_*_pred.npy")
    pred_files = glob.glob(search_pattern)
    
    if not pred_files:
        print(f"❌ 找不到预测结果！请检查路径。")
        return
        
    print(f"🧩 找到 {len(pred_files)} 个碎块！准备开始拼装...")

    all_coords = []
    all_preds = []
    all_gts = []

    for pred_file in tqdm(pred_files, desc="Merging chunks"):
        filename = os.path.basename(pred_file)
        chunk_name = filename.replace("val-", "").replace("_pred.npy", "")
        raw_chunk_dir = os.path.join(DATA_DIR, chunk_name)
        
        # 1. 读取坐标
        coord_path = os.path.join(raw_chunk_dir, "coord.npy")
        if not os.path.exists(coord_path):
            continue
        coord = np.load(coord_path)
        
        # 2. 读取原始真值 (GT) 并进行同步映射
        gt_raw = np.load(os.path.join(raw_chunk_dir, "segment.npy")).astype(np.int16)
        gt_mapped = np.full_like(gt_raw, 255, dtype=np.uint8) # 默认设为 255
        
        # ✅ 执行映射覆盖
        for raw_id, target_id in MAPPING_DICT.items():
            gt_mapped[gt_raw == raw_id] = target_id
            
        gt = gt_mapped
        
        # 3. 读取预测标签 Pred (已经是 0-5 范围了)
        pred_data = np.load(pred_file, allow_pickle=True)
        if pred_data.shape == () and isinstance(pred_data.item(), dict):
            pred = pred_data.item()['pred']
        elif isinstance(pred_data, dict) and 'pred' in pred_data:
            pred = pred_data['pred']
        else:
            pred = pred_data
        pred = pred.astype(np.uint8)
        
        assert len(coord) == len(pred), f"{chunk_name} 点数不匹配！"
        
        all_coords.append(coord)
        all_preds.append(pred)
        all_gts.append(gt)

    print("\n🌪️ 正在进行矩阵融合 (Vstack)...")
    full_coords = np.vstack(all_coords)
    full_preds = np.concatenate(all_preds)
    full_gts = np.concatenate(all_gts)
    
    total_points = len(full_coords)
    print(f"📈 拼装完成！全景总点数: {total_points:,} (约 {total_points / 1000000:.2f} M 个点)")

    print("🎨 正在为数百万个点进行 RGB 着色...")
    colors = np.zeros((total_points, 3), dtype=np.uint8)
    for label_id, color in COLOR_MAP.items():
        colors[full_preds == label_id] = color

    print("🧱 正在构建高维 PLY 结构 (XYZ + RGB + Pred + GT)...")
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('pred', 'u1'), ('gt', 'u1')
    ]
    
    vertex_data = np.empty(total_points, dtype=vertex_dtype)
    vertex_data['x'] = full_coords[:, 0]
    vertex_data['y'] = full_coords[:, 1]
    vertex_data['z'] = full_coords[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    vertex_data['pred'] = full_preds
    vertex_data['gt'] = full_gts # 这里存入的就是映射好的 GT

    print(f"💾 正在将全景点云写入 {OUTPUT_PLY} (文件较大，请耐心等待)...")
    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el], text=False).write(OUTPUT_PLY)
    
    print(f"✅ 全景导出成功！")

if __name__ == "__main__":
    merge_and_export()