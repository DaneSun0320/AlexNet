#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :AlexNet
# @FileName     :train
# @CreatTime    :2022/5/11 16:51 
# @CreatUser    :DaneSun
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import AlexNet

# 超参数
BATCH_SIZE = 32
EPOCH = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
}

# 数据集准备
img_path = './dataset'
train_dataset = datasets.ImageFolder(root=img_path + '/train',transform = data_transform['train'])
val_dataset = datasets.ImageFolder(root=img_path + '/val',transform = data_transform['val'])

train_num = len(train_dataset)
val_num = len(val_dataset)
classes = train_dataset.classes


train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
validate_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

net = AlexNet(init_weights=True).to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0002)
best_acc = 0.0

# 开始训练
for epoch in range(EPOCH):
    net.train()
    running_loss = 0.0
    for step,data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(DEVICE))
        loss = loss_func(outputs, labels.to(DEVICE))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 打印训练进度
        rate = (step + 1) / len(train_loader)
        a = '#' * int(rate * 50)
        b = '.' * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100),a,b,loss),end="")
    print()
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for date_test in validate_loader:
            test_images,test_labels = date_test
            outputs = net(test_images.to(DEVICE))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(DEVICE)).sum().item()
        val_acc = acc / val_num
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(),'./models/model.pth')
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / len(train_loader), val_acc))
print('Finished Training')