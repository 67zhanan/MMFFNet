import torch.nn as nn
import torch


class FEM(nn.Module):
    def __init__(self, channel):
        super(FEM, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.conv = nn.Sequential(nn.Conv2d(3 * channel, channel, 1, 1, 0),
                                  nn.ReLU())

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        return self.conv(concat) + x


class FFM(nn.Module):
    def __init__(self, channels):
        super(FFM, self).__init__()
        self.rgb_conv = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0),
                                      nn.ReLU())
        self.t_conv = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0),
                                    nn.ReLU())
        self.rgb_t = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(channels))
        self.t_rgb = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(channels))
        self.conv1 = FEM(channels)
        self.conv2 = FEM(channels)
        self.cat = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, 1, 0),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(channels))

    def forward(self, rgb, t):
        rgb = self.rgb_conv(rgb) + rgb
        t = self.t_conv(t) + t
        rgb_t_weights = torch.sigmoid(self.rgb_t(rgb))
        t_rgb_weights = torch.sigmoid(self.t_rgb(t))
        new_rgb = torch.mul(rgb, t_rgb_weights)
        new_t = torch.mul(t, rgb_t_weights)
        new_rgb = self.conv1(new_rgb) + new_rgb
        new_t = self.conv2(new_t) + new_t
        share = torch.cat([new_rgb, new_t], dim=1)
        share = self.cat(share)

        return share
