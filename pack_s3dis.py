import os
import torch
import numpy as np

# 1. 设置绝对路径，防止迷路
root_dir = os.path.abspath("data/s3dis/processed")
print(f"🔍 正在扫描数据目录: {root_dir}")

if not os.path.exists(root_dir):
    print("❌ 错误: 目录不存在！")
    exit()

# 2. 遍历 Area_1 到 Area_6
for area in sorted(os.listdir(root_dir)):
    area_path = os.path.join(root_dir, area)
    
    # 只处理 Area_ 开头的文件夹
    if not os.path.isdir(area_path) or not area.startswith("Area_"):
        continue

    print(f"\n📂 正在处理区域: {area} ...")
    room_count = 0
    
    # 3. 遍历房间
    for room in os.listdir(area_path):
        room_path = os.path.join(area_path, room)
        
        # 必须是文件夹，且里面有 coord.npy 才是有效数据
        if not os.path.isdir(room_path):
            continue
        if not os.path.exists(os.path.join(room_path, "coord.npy")):
            continue

        # 4. 目标文件路径
        save_path = os.path.join(area_path, f"{room}.pth")
        
        if os.path.exists(save_path):
            # print(f"  -> 跳过 (已存在): {room}")
            room_count += 1
            continue

        try:
            # 5. 加载 NPY 并打包为 PTH
            coord = np.load(os.path.join(room_path, "coord.npy"))
            color = np.load(os.path.join(room_path, "color.npy"))
            segment = np.load(os.path.join(room_path, "segment.npy"))
            
            instance = None
            if os.path.exists(os.path.join(room_path, "instance.npy")):
                instance = np.load(os.path.join(room_path, "instance.npy"))

            # 保存
            torch.save({
                "coord": coord, 
                "color": color, 
                "segment": segment, 
                "instance": instance
            }, save_path)
            
            print(f"  ✅ 已打包: {room}")
            room_count += 1
            
        except Exception as e:
            print(f"  ❌ 打包失败 {room}: {e}")

    print(f"  -> {area} 区域处理完毕，共找到 {room_count} 个房间。")

print("\n✨ 全部完成！")
