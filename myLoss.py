# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

# dice loss的计算函数
class BinaryDiceLoss(nn.Module):

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失w
        loss = 1 - N_dice_eff.sum() / N
        return loss
