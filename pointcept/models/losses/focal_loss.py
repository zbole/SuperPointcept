import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.builder import MODELS

@MODELS.register_module()
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, loss_weight=1.0, ignore_index=255):
        """
        gamma: 聚焦参数，越大越专注于困难样本(边界点)
        alpha: 缩放因子
        loss_weight: 在总 Loss 中的权重占比
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        pred: [N, C] (logits)
        target: [N] (labels)
        """
        # 1. 过滤掉 ignore_index (比如 255)
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if pred.numel() == 0:
            return pred.sum() * 0.0 # 防止 NaN

        # 2. 计算每个点的基础交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # 3. 计算预测正确的概率 (pt)
        pt = torch.exp(-ce_loss)
        
        # 4. 套用 Focal Loss 公式
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 返回加权后的平均 Loss
        return focal_loss.mean() * self.loss_weight