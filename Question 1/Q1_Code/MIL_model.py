import torchvision
import torch
from torch.nn import Linear,Conv2d,ReLU,Module
from torchvision.models import resnet18
from readData import train_dataloader,test_dataloader

import torch
import torch.nn as nn
from torch.nn import functional as F

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out




class DistributionPoolingFilter(nn.Module):

    __constants__ = ['num_bins', 'sigma']

    def __init__(self, num_bins=21, sigma=4): # 0.0167
        super(DistributionPoolingFilter, self).__init__()

        self.num_bins = num_bins
        self.sigma = sigma
        self.alfa = 1 / math.sqrt(2 * math.pi * (sigma ** 2))
        self.beta = -1 / (2 * (sigma ** 2))

        sample_points = torch.linspace(0, 1, steps=num_bins, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sample_points', sample_points)

    def extra_repr(self):
        return 'num_bins={}, sigma={}'.format(
            self.num_bins, self.sigma
        )

    def forward(self, data):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(batch_size, num_instances, num_features, 1)
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # data.size() --> (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff ** 2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

        result = self.alfa * torch.exp(self.beta * diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,1)

        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)

        return out

class RepresentationTransformation(nn.Module):
    def __init__(self, num_features=32, num_bins=21, num_classes=1):
        super(RepresentationTransformation, self).__init__()

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(100, 64),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.fc(x)

        return out

class model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = RestNet18()
        # self.filter = DistributionPoolingFilter()
        # self.features.conv1.in_channels = 1
        self.filter = ReLU()
        self.linear = RepresentationTransformation()
        print(self.features)

    def forward(self,x):
        x = self.features(x)
        x = torch.reshape(x,[1,-1,1])
        x = self.filter(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

