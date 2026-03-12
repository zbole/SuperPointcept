import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# ========================================================
# 配置
# ========================================================
PROCESSED_ROOT = Path("../../../../../datasets/sensaturban/processed_772d")
SPLITS = ["train", "val", "test"]
EXPECTED_DIM = 772  # 4D Geo + 768D DINO

def check_data():
    print(f"🔍 Starting Data Integrity Check at: {PROCESSED_ROOT}")
    
    total_chunks = 0
    total_points = 0
    error_log = []

    # 获取所有分块文件夹
    all_chunks = []
    for split in SPLITS:
        split_path = PROCESSED_ROOT / split
        if split_path.exists():
            all_chunks.extend(list(split_path.iterdir()))

    if not all_chunks:
        print("❌ No processed chunks found! Please check OUT_DATA_ROOT path.")
        return

    print(f"📦 Total chunks found: {len(all_chunks)}")
    
    # 随机抽样 5% 的数据进行深度检查，或者全量检查（如果时间允许）
    check_sample = random.sample(all_chunks, min(len(all_chunks), 500)) 
    print(f"🧪 Sampling {len(check_sample)} chunks for detailed verification...")

    for chunk_path in tqdm(check_sample):
        try:
            # 1. 检查文件是否存在
            files = ["coord.npy", "color.npy", "segment.npy", "extra_feat.npy"]
            for f in files:
                if not (chunk_path / f).exists():
                    raise FileNotFoundError(f"Missing {f}")

            # 2. 加载数据 (使用 mmap 模式，速度极快且不占内存)
            coord = np.load(chunk_path / "coord.npy", mmap_mode='r')
            color = np.load(chunk_path / "color.npy", mmap_mode='r')
            segment = np.load(chunk_path / "segment.npy", mmap_mode='r')
            extra_feat = np.load(chunk_path / "extra_feat.npy", mmap_mode='r')

            N = coord.shape[0]
            total_points += N
            total_chunks += 1

            # 3. 验证形状一致性 (Point Alignment)
            if not (N == color.shape[0] == segment.shape[0] == extra_feat.shape[0]):
                raise ValueError(f"Shape mismatch: {N} vs {extra_feat.shape[0]}")

            # 4. 验证特征维度 (772D)
            if extra_feat.shape[1] != EXPECTED_DIM:
                raise ValueError(f"Dimension mismatch: Expected {EXPECTED_DIM}, got {extra_feat.shape[1]}")

            # 5. 验证数据类型 (Dtype Check)
            if extra_feat.dtype != np.float16:
                error_log.append(f"⚠️ {chunk_path.name}: extra_feat is {extra_feat.dtype}, not float16")

            # 6. 统计值检查 (防止全 0 或 NaN)
            # 检查 DINO 特征区 (index 4-771)
            dino_slice = extra_feat[:, 4:]
            if np.all(dino_slice == 0):
                raise ValueError("DINO features are all zeros! (Model inference failure?)")
            
            # 由于 GH200 内存极大，可以偶尔做一次 NaN 检查
            if np.isnan(extra_feat).any():
                raise ValueError("Found NaN in extra_feat!")

        except Exception as e:
            error_log.append(f"❌ Error in {chunk_path.name}: {str(e)}")

    # ========================================================
    # 报告总结
    # ========================================================
    print("\n" + "="*50)
    print("📊 DATA CHECK SUMMARY")
    print("="*50)
    print(f"✅ Successfully verified chunks: {total_chunks}")
    print(f"💎 Total points processed: {total_points:,}")
    print(f"📐 Feature Dimension: {EXPECTED_DIM} (4D Geo + 768D DINO)")
    
    if error_log:
        print(f"\n🚩 FAILED CHECKS ({len(error_log)}):")
        for err in error_log[:10]: # 只显示前 10 条
            print(err)
        if len(error_log) > 10:
            print(f"... and {len(error_log)-10} more errors.")
    else:
        print("\n✨ All checked data is PERFECT. Ready for PTV3 training!")
    print("="*50)

if __name__ == "__main__":
    check_data()