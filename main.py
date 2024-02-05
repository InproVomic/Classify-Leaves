import os.path
import torch.optim
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_to_int=None):
        self.annotations = pd.read_csv(csv_file)
        self.annotations['image'] = self.annotations['image'].str.replace('images/', '')
        self.root_dir = root_dir
        self.transform = transform
        if label_to_int is None:
            # 创建一个从标签到整数的映射
            self.label_to_int = {label: idx for idx, label in enumerate(self.annotations['label'].unique())}
        else:
            # 使用提供的映射
            self.label_to_int = label_to_int

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations['image'][index])
        image = Image.open(image_path).convert('RGB')
        label = self.annotations['label'][index] if 'label' in self.annotations else None

        # 使用映射将标签转换为整数
        if label is not None:
            label = self.label_to_int[label]
        if self.transform:
            image = self.transform(image)
        return (image, label) if label is not None else image


train_transform = transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224 x 224的新图像
    transforms.RandomResizedCrop(128, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # 添加随机噪声
    transforms.ToTensor(),
    # 标准化图像的每个通道
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


train_data = CustomDataset(csv_file='leaves/train.csv', root_dir='leaves/images/', transform=train_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = CustomDataset(csv_file='leaves/test.csv', root_dir='leaves/images', transform=test_transform, label_to_int=train_data.label_to_int)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

resnet_50 = torchvision.models.resnet50(pretrained=True)
resnet_50.fc = nn.Linear(2048, 176)
resnet_50 = resnet_50.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

learning_rate = 0.000015
optimizer = torch.optim.Adam(resnet_50.parameters(), lr=learning_rate, weight_decay=0.001)

epoch = 10

resnet_50.train()
for i in range(epoch):
    for data in train_loader:
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        ouputs = resnet_50(imgs)
        loss = loss_fn(ouputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("第{}轮训练，loss:{}".format(i+1, str(loss.item())))
    if i+1 % 10 == 0:
        torch.save(resnet_50, 'resnet50_{}.pth'.format(epoch+1))

dict_inverse = {v: k for k, v in train_data.label_to_int.items()}
ret = []

resnet_50.eval()
with torch.no_grad():
    for images in test_loader:
        images = images.cuda()
        output = resnet_50(images)
        _, tmp = torch.max(output, 1)
        ret.append(dict_inverse[tmp.item()])

test_data_original = pd.read_csv('leaves/test.csv')
test_data_original['label'] = ret
test_data_original.to_csv('output.csv', index=False)
