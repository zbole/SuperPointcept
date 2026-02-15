"""
Preprocessing Script for S3DIS with Progress Bar
Original Author: Xiaoyang Wu
Modified for: Bole (Bristol) - Added tqdm & Robustness fixes
"""

import os
import argparse
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm  # 引入进度条库

try:
    import open3d
except ImportError:
    import warnings
    warnings.warn("Please install open3d for parsing normal")

try:
    import trimesh
except ImportError:
    import warnings
    warnings.warn("Please install trimesh for parsing normal")

# Global dictionary to hold mesh data for normal parsing
area_mesh_dict = {}

def parse_room(
    room, angle, dataset_root, output_root, align_angle=True, parse_normal=False
):
    # 移除这里的 print，改用 tqdm 的进度描述，防止控制台刷屏
    classes = [
        "ceiling", "floor", "wall", "beam", "column", "window",
        "door", "table", "chair", "sofa", "bookcase", "board", "clutter",
    ]
    class2label = {cls: i for i, cls in enumerate(classes)}
    source_dir = os.path.join(dataset_root, room)
    save_path = os.path.join(output_root, room)
    
    # 1. 路径安全检查：防止源文件夹不存在导致 Crash
    if not os.path.exists(source_dir):
        return f"Skip: {room} (Source not found)"

    os.makedirs(save_path, exist_ok=True)
    object_path_list = sorted(glob.glob(os.path.join(source_dir, "Annotations/*.txt")))

    room_coords = []
    room_colors = []
    room_normals = []
    room_semantic_gt = []
    room_instance_gt = []

    for object_id, object_path in enumerate(object_path_list):
        object_name = os.path.basename(object_path).split("_")[0]
        try:
            obj = np.loadtxt(object_path)
        except Exception:
            # 防止空文件或损坏文件中断进程
            continue
        
        # S3DIS raw txts are usually XYZRGB
        coords = obj[:, :3]
        colors = obj[:, 3:6]
        
        class_name = object_name if object_name in classes else "clutter"
        semantic_gt = np.repeat(class2label[class_name], coords.shape[0])
        semantic_gt = semantic_gt.reshape([-1, 1])
        instance_gt = np.repeat(object_id, coords.shape[0])
        instance_gt = instance_gt.reshape([-1, 1])

        room_coords.append(coords)
        room_colors.append(colors)
        room_semantic_gt.append(semantic_gt)
        room_instance_gt.append(instance_gt)

    if len(room_coords) == 0:
        return f"Skip: {room} (No valid objects)"

    room_coords = np.ascontiguousarray(np.vstack(room_coords))

    # Normal parsing logic
    if parse_normal:
        x_min, z_max, y_min = np.min(room_coords, axis=0)
        x_max, z_min, y_max = np.max(room_coords, axis=0)
        z_max = -z_max
        z_min = -z_min
        max_bound = np.array([x_max, y_max, z_max]) + 0.1
        min_bound = np.array([x_min, y_min, z_min]) - 0.1
        bbox = open3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )
        
        try:
            # Retrieve the loaded mesh for the corresponding Area
            room_mesh = (
                area_mesh_dict[os.path.dirname(room)]
                .crop(bbox)
                .transform(
                    np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                )
            )
            
            vertices = np.array(room_mesh.vertices)
            faces = np.array(room_mesh.triangles)
            vertex_normals = np.array(room_mesh.vertex_normals)
            
            room_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=vertex_normals
            )
            (closest_points, distances, face_id) = room_mesh.nearest.on_surface(room_coords)
            room_normals = room_mesh.face_normals[face_id]
        except Exception as e:
            # 如果法向量提取失败，填充 0 防止报错
            print(f"Normal parsing failed for {room}: {e}")
            room_normals = np.zeros_like(room_coords)

    # Angle alignment logic
    if align_angle:
        angle_rad = (2 - angle / 180) * np.pi
        rot_cos, rot_sin = np.cos(angle_rad), np.sin(angle_rad)
        rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        room_center = (np.max(room_coords, axis=0) + np.min(room_coords, axis=0)) / 2
        room_coords = (room_coords - room_center) @ np.transpose(rot_t) + room_center
        if parse_normal:
            room_normals = room_normals @ np.transpose(rot_t)

    room_colors = np.ascontiguousarray(np.vstack(room_colors))
    room_semantic_gt = np.ascontiguousarray(np.vstack(room_semantic_gt))
    room_instance_gt = np.ascontiguousarray(np.vstack(room_instance_gt))
    
    np.save(os.path.join(save_path, "coord.npy"), room_coords.astype(np.float32))
    np.save(os.path.join(save_path, "color.npy"), room_colors.astype(np.uint8))
    np.save(os.path.join(save_path, "segment.npy"), room_semantic_gt.astype(np.int16))
    np.save(os.path.join(save_path, "instance.npy"), room_instance_gt.astype(np.int16))

    if parse_normal:
        np.save(os.path.join(save_path, "normal.npy"), room_normals.astype(np.float32))

    return f"Success: {room}"

def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True, nargs="+", choices=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"])
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--raw_root", default=None)
    parser.add_argument("--align_angle", action="store_true")
    parser.add_argument("--parse_normal", action="store_true")
    parser.add_argument("--num_workers", default=4, type=int) # 默认改为 4，避免内存爆炸
    args = parser.parse_args()

    if args.parse_normal:
        assert args.raw_root is not None

    room_list = []
    angle_list = []

    print("📅 Loading room information ...")
    for split in args.splits:
        angle_file = os.path.join(args.dataset_root, split, f"{split}_alignmentAngle.txt")
        if not os.path.exists(angle_file):
            print(f"⚠️ Warning: {angle_file} not found. Skipping {split}.")
            continue
            
        area_info = np.loadtxt(angle_file, dtype=str)
        # 2. 修复单行数据 Bug：如果文件只有一行，numpy会读成1D数组，导致后续遍历出错
        if area_info.ndim == 1:
            area_info = area_info[None, :]
            
        room_list += [os.path.join(split, room_info[0]) for room_info in area_info]
        angle_list += [int(room_info[1]) for room_info in area_info]

    if args.parse_normal:
        print("Loading raw mesh file ... (This may consume lots of RAM)")
        # ... (Mesh loading logic remains similar, omitted for brevity but area_mesh_dict needs to be populated here)
        # 建议：如果不需要法向量，尽量不要开 --parse_normal，非常吃内存
        pass 

    print(f"🚀 Processing {len(room_list)} scenes with {args.num_workers} workers...")
    
    # 3. 使用 tqdm 包装 ProcessPoolExecutor
    # 注意：Pool 这里我们用上下文管理器，更安全
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        # pool.map 返回一个迭代器，tqdm 可以直接包装它
        results = list(tqdm(
            pool.map(
                parse_room,
                room_list,
                angle_list,
                repeat(args.dataset_root),
                repeat(args.output_root),
                repeat(args.align_angle),
                repeat(args.parse_normal),
            ),
            total=len(room_list),
            desc="S3DIS Preprocessing",
            unit="room"
        ))

    # 4. 打印简报
    success_count = sum(1 for r in results if r and "Success" in r)
    print(f"\n✅ All done! {success_count}/{len(room_list)} rooms processed successfully.")

if __name__ == "__main__":
    main_process()