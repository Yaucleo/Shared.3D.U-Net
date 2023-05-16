import torch
from torchvision.transforms import transforms as T
import Unet3d
from torch import optim
from dataset3d import LiverDataset
from torch.utils.data import DataLoader
import myLoss
import imshow
from PIL import Image

# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.ToTensor(),
])

y_transform = T.ToTensor()


# 测试
class LiverDataset_flip(object):
    pass


def test():
    model = Unet3d.UNet(1, 1).to(device)
    # 导入待测试的权重
    model.load_state_dict(torch.load('weights_3DUnet_FINAL.pth', map_location='cpu'))
    # 导入测试集数据
    liver_dataset = LiverDataset("D:/CJY/myData/testSet", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)
    step = 0
    epoch_loss = 0
    for x, y in dataloaders:
        inputs = x.to(device)
        labels = y.to(device)
        outputs = model(inputs)
        # 图片可视化
        imOutput = outputs.squeeze().cpu().detach().numpy() * 255
        imLable = labels.squeeze().cpu().detach().numpy() * 255
        imshow.plot_3d(imLable, 50)
        imshow.plot_3d(imOutput, 50)
        # 计算损失函数
        # loss = torch.nn.BCELoss()(outputs, labels)
        loss = myLoss.BinaryDiceLoss()(outputs, labels)
        step += 1
        str = "%d,test_loss:%0.3f" % (step, loss.item())
        epoch_loss += loss.item()
        print(str + '\n')
    print("epoch_loss:%0.3f" % epoch_loss + '\n' + "mean_loss:%0.3f" % (epoch_loss / step))


if __name__ == '__main__':
    with torch.no_grad():
        test()
