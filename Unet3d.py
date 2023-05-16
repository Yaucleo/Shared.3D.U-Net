# -*- coding: utf-8 -*-


import torch.nn as nn
import torch
from torch import autograd
from torchvision import transforms


# 使用残差网络的连续卷积
class DownDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.out_ch = out_ch
        self.in_ch = in_ch

    def forward(self, x):
        # 复制input得到与输出通过尺寸的tensor
        out = x.repeat(1, self.out_ch // self.in_ch, 1, 1, 1)
        out2 = self.conv(x)
        # 返回输出与输入的和
        return torch.add(out, out2)


# 上采样的卷积运算未使用残差网络
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        size = 32
        self.conv1 = DownDoubleConv(in_ch, 1 * size)
        self.pool1 = nn.MaxPool3d(2)  # 每次把图像尺寸缩小一半
        self.conv2 = DownDoubleConv(1 * size, 2 * size)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DownDoubleConv(2 * size, 4 * size)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = DownDoubleConv(4 * size, 8 * size)
        self.pool4 = nn.MaxPool3d(2)
        self.conv5 = DownDoubleConv(8 * size, 16 * size)

        self.up6 = nn.ConvTranspose3d(16 * size, 8 * size, 2, stride=2)
        self.conv6 = DoubleConv(16 * size, 8 * size)
        self.up7 = nn.ConvTranspose3d(8 * size, 4 * size, 2, stride=2)
        self.conv7 = DoubleConv(8 * size, 4 * size)
        self.up8 = nn.ConvTranspose3d(4 * size, 2 * size, 2, stride=2)
        self.conv8 = DoubleConv(4 * size, 2 * size)
        self.up9 = nn.ConvTranspose3d(2 * size, 1 * size, 2, stride=2)
        self.conv9 = DoubleConv(2 * size, 1 * size)
        self.conv10 = nn.Conv3d(1 * size, out_ch, 1)
        self.fnn = nn.Linear(2 * size * 2 * size * 32, 2)
        self.len_of_RIO = 96

        self.fnn = nn.Linear(32 * 32 * 8, 2)
        self.len_of_RIO = 96

    def unet(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

    def forward(self, x):
        # 提取ROI的位置
        roi_feature = self.unet(x)
        # adaptive_avg_pool3d使tensor为固定size，以便作为全连接层输入
        out1 = torch.nn.functional.adaptive_avg_pool3d(roi_feature, (8, 32, 32))
        out2 = out1.view(-1)
        # 全连接层计算出ROI位置坐标x,y，其中不包括z坐标（不对z作裁剪）
        location = self.fnn(out2)
        # 将x,y坐标转化为[0,1]，适配不同size的图片
        location = nn.Sigmoid()(location)
        # 计算当前图片准确的x,y坐标值（已保证ROI不越出图像边界）
        roi_x = ((x.shape[3] - self.len_of_RIO) * location[0]).__long__()
        roi_y = ((x.shape[4] - self.len_of_RIO) * location[1]).__long__()
        # 从输入图像中裁剪出ROI，作为分割网络的输入
        roi = x[:, :, :, roi_x:roi_x + self.len_of_RIO, roi_y:roi_y + self.len_of_RIO]
        # 加入out0中的roi裁剪结果，类似残差网络
        x3 = roi_feature[:, :, :, roi_x:roi_x + self.len_of_RIO, roi_y:roi_y + self.len_of_RIO]
        out3 = self.unet(torch.add(roi, x3))
        # 为ROI分割结果填充0像素，恢复为原图大小，以便计算损失
        pad = nn.ZeroPad2d((roi_y, x.shape[3] - roi_y - self.len_of_RIO, roi_x, x.shape[4] - roi_x - self.len_of_RIO))
        out = pad(out3)
        return out
