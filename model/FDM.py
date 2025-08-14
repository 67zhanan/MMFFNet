import torch
import torch.nn as nn
from model.ODConv2d import ODConv2d


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(nn.Conv2d(channel, channel, 1, bias=False),
                                        nn.ReLU())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        avgout = self.shared_MLP(x)
        return self.sigmoid(avgout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        return self.sigmoid(x)


class FDM(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(FDM, self).__init__()
        self.CA = ChannelAttention(channel)

        self.conv1 = nn.Sequential(ODConv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(num_features=channel))
        self.conv2 = nn.Sequential(ODConv2d(channel, channel2, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(num_features=channel2))

        self.conv3 = nn.Sequential(ODConv2d(channel2, channel2, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(num_features=channel2))
        self.conv4 = nn.Sequential(ODConv2d(channel2, num_filters, kernel_size=1, stride=1, padding=0),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(num_features=num_filters))

        self.SA = SpatialAttention(7)
        
        self.pp = nn.Sequential(nn.ConvTranspose2d(in_channels=num_filters, out_channels=num_filters, kernel_size=4, padding=1, stride=2),
                                nn.Conv2d(num_filters, 1, 1, 1, 0),
                                nn.ReLU())

    def forward(self, x):
        x = self.CA(x) * x

        pool1 = x
        x = self.conv1(x)
        x = x + pool1
        x = self.conv2(x)
        pool2 = x
        x = self.conv3(x)
        x = x + pool2
        x = self.conv4(x)

        x = self.SA(x) * x

        x = self.pp(x)

        return x
