"""
STPLS3D Dataset
"""

import os
import numpy as np
from .defaults import DefaultDataset
from .builder import DATASETS
from .transform import TRANSFORMS  # <--- 修正这里：从 .transform 导入，而不是 .builder

# ---------------------------------------------------------------------------- #
# 自定义 Transform 用于类别映射 (20类 -> 6类)
# ---------------------------------------------------------------------------- #
@TRANSFORMS.register_module()
class CategoryMapping(object):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict
        
        # 预计算查找表 (Lookup Table)
        if mapping_dict:
            max_key = max(mapping_dict.keys())
            self.lookup = np.ones(max_key + 1, dtype=np.int64) * 255
            for k, v in mapping_dict.items():
                self.lookup[k] = v
        else:
            self.lookup = None

    def __call__(self, data_dict):
            if self.lookup is not None and "segment" in data_dict:
                seg = data_dict["segment"]
                
                # 1. 创建一个布尔掩码：找出所有在映射表范围内的有效原始标签
                # 假设你的 lookup 长度是 20 (0-19)
                valid_range_mask = (seg >= 0) & (seg < len(self.lookup))
                
                # 2. 初始化新标签数组，默认全部设为 ignore_index (255)
                new_seg = np.ones_like(seg, dtype=np.int64) * 255
                
                # 3. 只有在映射范围内的标签才执行 lookup，其余全部保持 255
                # 这能彻底过滤掉任何大于 19 或异常的标签
                new_seg[valid_range_mask] = self.lookup[seg[valid_range_mask]]
                
                data_dict["segment"] = new_seg
                
            return data_dict

# ---------------------------------------------------------------------------- #
# Dataset 定义
# ---------------------------------------------------------------------------- #
@DATASETS.register_module()
class STPLS3DDataset(DefaultDataset):
    def get_data_name(self, idx):
        """
        根据给定的索引获取数据名称
        """
        try:
            # 尝试解析路径
            remain, las_name = os.path.split(self.data_list[idx % len(self.data_list)])
            remain, data_type = os.path.split(remain)
            return f"{data_type}-{las_name}"
        except:
            # 如果解析失败，直接返回文件名
            return os.path.basename(self.data_list[idx % len(self.data_list)])