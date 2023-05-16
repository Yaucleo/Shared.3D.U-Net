# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:14:53 2020

@author: 陈健宇
"""

import torch.utils.data as data
import os
import PIL.Image as Image
import nrrd
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T


# data.Dataset:
# 所有子类应该override __len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)

class LiverDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None, target_transform=None):  # root表示图片路径
        n = len(os.listdir(root)) // 2  # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整
        imgs = [] # 列表？可以放入不同size的数组作为元素
        for i in range(n):
            img = os.path.join(root, "%03d_lgemri.nrrd" % i)  # os.path.join(path1[,path2[,......]]):将多个路径组合后返回
            mask = os.path.join(root, "%03d_laendo.nrrd" % i)#mask
            imgs.append([img, mask])  # append只能有一个参数，加上[]变成一个list

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index] #取出 [img, mask]？
        img_x, nrrd_options = nrrd.read(x_path)
        img_y, nrrd_options = nrrd.read(y_path)
        t = 3
        if self.transform is not None:
            img_x = self.transform(img_x[0:576:t, 0:576:t, 4:84])
            # img_x = self.transform(img_x)
            img_x = img_x.unsqueeze(0)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y[0:576:t, 0:576:t, 4:84])
            # img_y = self.target_transform(img_y)
            img_y = img_y.unsqueeze(0)
        return img_x, img_y  # 返回的矩阵

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]

if __name__ == '__main__':
    print(1)

    x_transform = T.Compose([
        T.ToTensor(),
        # 标准化至[-1,1],规定均值和标准差
        # T.Normalize(0.5, 0.5)  # torchvision.transforms.Normalize(mean, std, inplace=False)
    ])
    # mask只需要转换为tensor
    y_transform = T.ToTensor()
    liver_dataset = LiverDataset("data/t_3D", transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(liver_dataset, batch_size=1, shuffle=True)
    for x, y in dataloader:  # 分100次遍历数据集，每次遍历batch_size=4
        # optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
        inputs = x
        labels = y
        print(inputs.size())
        print(labels.size())

