"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, keep_ratio=0.7, loss_weight=1.0, ignore_index=255):
        """
        keep_ratio: 保留 loss 最大的前多少比例的点 (0 < keep_ratio <= 1)
        """
        super(OHEMCrossEntropyLoss, self).__init__()
        self.keep_ratio = keep_ratio
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # 1. 计算 point-wise 的 raw Cross Entropy Loss (reduction='none' 保证输出形状为 (N,))
        losses = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        
        # 2. 过滤掉被 ignore 的无效点
        valid_mask = target != self.ignore_index
        valid_losses = losses[valid_mask]
        
        # 防止极端情况下 batch 内全是被 ignore 的点
        if valid_losses.numel() == 0:
            return losses.sum() * 0.0
            
        # 3. 计算需要保留的 hard examples 数量
        num_keep = int(valid_losses.numel() * self.keep_ratio)
        num_keep = max(1, num_keep) # 至少保留一个点
        
        # 4. 核心优化：使用 topk 提取 Loss 最大的点
        if num_keep < valid_losses.numel():
            topk_losses, _ = valid_losses.topk(num_keep)
        else:
            topk_losses = valid_losses
            
        # 5. 求 mean 并乘以权重
        loss = topk_losses.mean()
        
        return loss * self.loss_weight

@LOSSES.register_module()
class TargetRepulsionLoss(nn.Module):
    def __init__(self, target_classes, margin=0.0, loss_weight=1.0, ignore_index=255):
        super(TargetRepulsionLoss, self).__init__()
        self.target_classes = target_classes
        self.margin = margin
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, features, target):
        # 1. 过滤 ignore_index
        valid_mask = target != self.ignore_index
        features = features[valid_mask]
        target = target[valid_mask]

        if features.shape[0] == 0:
            return (features.sum() * 0.0).requires_grad_(True)

        # 2. 找出目标和背景
        target_mask = torch.isin(target, torch.tensor(self.target_classes, device=target.device))
        
        # 【极致优化 1】：不再需要 bg_mask 和全量 bg_feats
        # 直接拿全体点的平均特征，或者非目标点的平均特征作为 Background Prototype
        bg_feats = features[~target_mask]

        if target_mask.sum() == 0 or bg_feats.shape[0] == 0:
            return (features.sum() * 0.0).requires_grad_(True)

        target_feats = features[target_mask]

        # 【极致优化 2】：计算背景的原型 (Mean Prototype)，并阻断梯度！
        # .detach() 是灵魂！它让庞大的背景点前向传播图直接被丢弃，不计算梯度，省下海量时间和显存
        bg_prototype = bg_feats.mean(dim=0, keepdim=True).detach() 

        # 3. L2 归一化
        target_feats_norm = F.normalize(target_feats, p=2, dim=1)
        bg_prototype_norm = F.normalize(bg_prototype, p=2, dim=1)

        # 【极致优化 3】：变成 O(N) 的点乘，彻底消灭 N^2 的矩阵乘法和 randperm
        # 每个 target point 只和那个唯一的 bg_prototype 算相似度
        sim_vector = torch.sum(target_feats_norm * bg_prototype_norm, dim=1) 

        # 4. 计算排斥 Loss
        loss = torch.relu(sim_vector - self.margin).mean()

        return loss * self.loss_weight