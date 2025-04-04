from torch.nn import Linear,ReLU,Conv2d
from scipy.io import loadmat
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pca import pca
import os
#模型搭建
'''
train_X大小为(194,7,20)可以视为194张7*20的图像,因而可以用CNN(卷积神经网络)处理
'''
class CNN_model(torch.nn.Module):
    def __init__(self,k):
        super(CNN_model, self).__init__()
        # 卷积层
        self.conv1 = Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        # 全连接层
        self.fc1 = Linear(32 * 7 * k, 128)
        self.fc2 = Linear(128, 7 * 1)  # 输出形状为 (batch_size, 7, 1)
        # 激活函数
        self.relu = ReLU()

    def forward(self, x):
        # 输入形状: (batch_size, 1, 7, k)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # 展平
        x = x.view(x.size(0), -1)  # 形状: (batch_size, 32 * 7 * k)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # 调整输出形状为 (batch_size, 7, 1)
        x = x.view(x.size(0), 7, 1)
        return x