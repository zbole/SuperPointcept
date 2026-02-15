"""
TransmissionCorridor Dataset

Author: Binhan Luo (luobinhan@cug.edu.cn)
"""

import os
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class TransmissionCorridorDataset(DefaultDataset):
    def get_data_name(self, idx):
        """
        根据给定的索引获取数据名称
        参数:
            idx (int): 数据的索引
        返回:
            str: 组合的文件类型和文件名名称，格式为"数据集类型-文件名"
        """
        remain, las_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, data_type = os.path.split(remain)
        return f"{data_type}-{las_name}"

