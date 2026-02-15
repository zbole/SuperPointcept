import numpy as np
import glob
import os
from tqdm import tqdm

def check_stpls3d_data(processed_path):
    # 搜寻所有 segment.npy 文件
    # 注意：STPLS3D 的切块结构通常是 processed/train/Scene_X/segment.npy
    search_path = os.path.join(processed_path, "**/segment.npy")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print(f"❌ 路径错误：在 {processed_path} 没找到任何 segment.npy。")
        return

    print(f"🔍 启动排雷：准备扫描 {len(files)} 个数据块...")
    
    # 合法标签集：0-5 是训练类，255 是忽略类
    valid_labels = {0, 1, 2, 3, 4, 5, 255}
    bad_files = []

    # 进度条
    for f in tqdm(files, desc="Checking Labels", unit="chunk"):
        try:
            # 加载标签
            labels = np.load(f)
            
            # 1. 检查是否有 NaN/Inf
            if not np.isfinite(labels).all():
                bad_files.append({"path": f, "reason": "包含 NaN 或 Inf"})
                continue
                
            # 2. 检查是否有越界标签
            unique_labels = np.unique(labels)
            offending = [l for l in unique_labels if l not in valid_labels]
            
            if offending:
                bad_files.append({
                    "path": f, 
                    "reason": f"非法标签索引: {offending}"
                })
                
        except Exception as e:
            bad_files.append({"path": f, "reason": f"文件损坏或无法读取: {str(e)}"})

    # 报告结果
    print("\n" + "="*60)
    if not bad_files:
        print("✅ 完美！所有已处理的数据标签都在 [0-5, 255] 范围内。")
    else:
        print(f"🚨 警报！共发现 {len(bad_files)} 个有问题的“毒样本”：")
        for item in bad_files[:10]: # 只展示前10个
            print(f"  - {item['path']} -> {item['reason']}")
        if len(bad_files) > 10:
            print(f"  ... 以及另外 {len(bad_files) - 10} 个错误文件。")
        
        # 自动生成清理建议
        print("\n💡 建议方案：")
        print("  1. 运行 'rm -rf' 删除上述异常文件所在目录。")
        print("  2. 检查你的 CategoryMapping 字典是否覆盖了所有原始标签。")
    print("="*60)

if __name__ == "__main__":
    # 根据你的容器映射路径修改
    DATA_PATH = "data/stpls3d/processed"
    check_stpls3d_data(DATA_PATH)