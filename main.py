# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:15:06 2020

@author: 陈健宇
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
from torchvision.transforms import transforms as T
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
import Unet3d
from torch import optim
from dataset3d import LiverDataset
from torch.utils.data import DataLoader
import myLoss

# 是否使用gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_transform = T.Compose([
    T.ToTensor()
])
# mask只需要转换为tensor
y_transform = T.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=300):
    # 将历史最小的loss（取值范围是[0,1]）初始化为最大值1
    min_testloiss = 1
    for epoch in range(num_epochs):
        # 5个epoch不优化则降低学习率
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                                   threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                   eps=1e-06)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 训练集数据个数
        dataset_size = len(dataload.dataset)
        # 每个epoch的loss
        epoch_loss = 0
        # 当前epoch的当前计算数据序号
        step = 0
        # 遍历数据集，batch_size=1， 共进行num_epochs次
        for x, y in dataload:
            # 将梯度(dw,db,...)清零
            optimizer.zero_grad()
            inputs = x.to(device)
            labels = y.to(device)
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 梯度下降,计算出梯度
            loss.backward()
            # 对所有的参数进行一次更新
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        # 记录训练结果
        f = open('D:/CJY/training_log_single.txt', 'a')
        f.write(str +'\n')
        f.close()
        # 更新保存的当前权重，用于在验证集测试
        torch.save(model.state_dict(), 'weights_single.pth')
        if (epoch % 3 == 0):
            # torch.save(model.state_dict(), 'weights_%d.pth' % epoch)  # 返回模型的所有内容
            with torch.no_grad():
                # 使用保存的当前权重计算验证集上的损失
                testloss = test(epoch)
            # loss比历史最小loss小时独立保存
            if testloss < min_testloiss:
                torch.save(model.state_dict(), 'test_weights%0.3f.pth' % testloss)
    return model


# 训练模型
def train():
    # 输入与输出图片的通道数都是1
    model = Unet3d.UNet(1, 1).to(device)
    # 导入历史保存的权重作为训练初始权重
    model.load_state_dict(torch.load('test_weights0.106.pth', map_location='cpu'))  # JY11.21,加载之前的训练结果，到model中
    # batch_size设为1
    batch_size = 1
    # 两种损失函数
    # criterion = torch.nn.BCELoss()
    criterion = myLoss.BinaryDiceLoss()  # 指定损失函数为自定义
    # 梯度下降的优化器，使用默认学习率
    optimizer = optim.Adam(model.parameters())  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    liver_dataset = LiverDataset("D:/CJY/myData/trainset_0_69", transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True)
    # 开始训练
    train_model(model, criterion, optimizer, dataloader)

# 测试
def test(e):
    model = Unet3d.UNet(1, 1).to(device)
    model.load_state_dict(torch.load('weights_single.pth', map_location='cpu'))
    # 使用测试集数据进行测试
    liver_dataset = LiverDataset("D:/CJY/myData/validationSet", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)
    step = 0
    sumloss = 0
    for x, y in dataloaders:
        inputs = x.to(device)
        labels = y.to(device)
        outputs = model(inputs)  # 前向传播
        loss = myLoss.BinaryDiceLoss()(outputs, labels)
        sumloss += loss
        step += 1
        str = "%d,test_loss:%0.3f" % (step, loss.item())
        print(str+'\n')
    print("meanloss:%0.3f" % (sumloss/step))
    log("meanloss:%0.3f" % (sumloss/step))
    return sumloss/step

# 保存信息到日志中
def log(str):
    f = open('training_log_double_unet.txt', 'a')
    f.write(str + '\n')
    f.close()

if __name__ == '__main__':
    with torch.cuda.device(0):
        train()

