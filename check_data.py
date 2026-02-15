import torch
import numpy as np
import os
from pointcept.datasets.semantic_kitti import SemanticKITTIDataset
# 不需要 import transform 了，dataset 自己会处理

def check_data():
    print(">>> 开始检查数据标签...")
    
    # 1. 定义变换配置 (直接用字典列表)
    transform_list = [
        dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("coord", "strength"))
    ]
    
    # 2. 构建 transform -> 【删掉这一步！】
    # transform = T.Compose(transform_list) <--- 删掉这行

    # 3. 加载数据集
    data_root = 'data/semantic_kitti'
    if not os.path.exists(data_root):
        print(f"❌ 错误: 找不到数据目录 {data_root}")
        return

    print(f"正在读取 {data_root} ...")
    
    # 直接把 list 传给它
    dataset = SemanticKITTIDataset(
        split='train',
        data_root=data_root,
        transform=transform_list,  # <--- 这里直接传列表！
        test_mode=False
    )
    
    print(f"✅ 数据集加载成功，共有 {len(dataset)} 个样本")

    # 4. 抽查前 5 个样本
    print("-" * 40)
    # 检查前5个
    indices = [0, 100, 1000, 2000] 
    
    for i in indices:
        if i >= len(dataset): continue
        try:
            data = dataset[i]
            segment = data['segment']
            
            # 转成 numpy 方便看
            if isinstance(segment, torch.Tensor):
                unique_vals = torch.unique(segment).numpy()
            else:
                unique_vals = np.unique(segment)
                
            min_val = unique_vals.min()
            max_val = unique_vals.max()
            
            print(f"[样本 {i}] 标签范围: {min_val} -> {max_val}")
            
            # 5. 核心诊断逻辑
            # 正常范围：0-18，忽略值：255
            
            # 筛选出异常值
            invalid_mask = (unique_vals > 18) & (unique_vals != 255)
            
            if np.any(invalid_mask):
                print(f"❌ 发现非法标签! 值: {unique_vals[invalid_mask]}")
                print(f"   完整标签集合: {unique_vals}")
                print("   👉 这就是导致 CUDA Error 的原因！请把这个非法值告诉我。")
                return
            else:
                if 255 in unique_vals:
                    print(f"   (包含忽略标签 255，正常)")
                else:
                    print(f"   (标签纯净，无 255)")
                
        except Exception as e:
            print(f"⚠️ 加载样本 {i} 失败: {e}")
            import traceback
            traceback.print_exc()
            break
            
    print("-" * 40)
    print("🎉 检查结束。")

if __name__ == "__main__":
    check_data()