import torch
import numpy as np
import os
from pointcept.datasets.semantic_kitti import SemanticKITTIDataset
from tqdm import tqdm
import multiprocessing

def scan_worker(args):
    """单个样本的扫描函数，用于多进程"""
    dataset, idx = args
    try:
        data = dataset[idx]
        segment = data['segment']
        
        if isinstance(segment, torch.Tensor):
            unique_vals = torch.unique(segment).numpy()
        else:
            unique_vals = np.unique(segment)
            
        # 检查是否只有 -1 到 18
        # 如果出现 > 18 的数，或者 < -1 的数，就是非法的
        invalid_mask = (unique_vals > 18) | (unique_vals < -1)
        
        if np.any(invalid_mask):
            return idx, unique_vals[invalid_mask]
        return None
    except Exception as e:
        return idx, f"Error: {str(e)}"

def check_all_data():
    print(">>> 正在初始化数据集索引...")
    
    data_root = 'data/semantic_kitti'
    
    # 简单的配置，只为了读标签
    transform_list = [
        dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "strength"))
    ]
    
    dataset = SemanticKITTIDataset(
        split='train',
        data_root=data_root,
        transform=transform_list, 
        test_mode=False,
        loop=1 # 只读一次
    )
    
    total_len = len(dataset)
    print(f"✅ 准备扫描全部 {total_len} 个样本...")
    print(">>> 这可能需要几分钟，请耐心等待...")

    # 单进程扫描（为了避免序列化问题，简单直接）
    error_count = 0
    for i in tqdm(range(total_len)):
        try:
            # 只读 segment，不加载 coord 以加快速度
            # 注意：Pointcept 的 dataset 可能需要加载文件才能拿到 segment
            # 这里我们直接调 dataset[i]
            data = dataset[i]
            segment = data['segment']
            
            if isinstance(segment, torch.Tensor):
                vals = torch.unique(segment).numpy()
            else:
                vals = np.unique(segment)

            # 核心判断：我们现在的配置认为合法范围是 [-1, 0, 1, ..., 18]
            # 任何其他值都会导致 crash
            invalid_vals = vals[(vals > 18) | (vals < -1)]
            
            if len(invalid_vals) > 0:
                print(f"\n🚨 抓到了！样本 [{i}] 包含非法标签: {invalid_vals}")
                print(f"   该样本完整标签: {vals}")
                error_count += 1
                # 抓到一个就可以停了，或者继续找
                if error_count > 5: break
                
        except Exception as e:
            print(f"\n⚠️ 样本 [{i}] 读取失败: {e}")

    if error_count == 0:
        print("\n✅ 全量扫描完成，未发现非法标签。问题可能在 AMP (NaN) 或显存。")
    else:
        print(f"\n❌ 扫描完成，共发现 {error_count} 个坏样本。")

if __name__ == "__main__":
    check_all_data()