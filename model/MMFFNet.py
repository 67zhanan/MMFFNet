import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from model.FFM import FFM
from model.FDM import FDM


class MMFFNet(nn.Module):
    def __init__(self):
        super(MMFFNet, self).__init__()
        feats = list(convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.children())
        self.rgb_1 = nn.Sequential(*feats[:2])
        self.t_1 = nn.Sequential(*feats[:2])
        self.rgb_2 = nn.Sequential(*feats[2:4])
        self.t_2 = nn.Sequential(*feats[2:4])
        self.rgb_3 = nn.Sequential(*feats[4:6])
        self.t_3 = nn.Sequential(*feats[4:6])

        self.ffm = FFM(channels=384)
        self.fdm = FDM(384, 192, 32)

    def forward(self, rgb, t):
        rgb = self.rgb_1(rgb)
        t = self.t_1(t)
        rgb = self.rgb_2(rgb)
        t = self.t_2(t)
        rgb = self.rgb_3(rgb)
        t = self.t_3(t)

        share = self.ffm(rgb, t)
        x = self.fdm(share)

        B, C, H, W = x.size()
        x_sum = x.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_normed = x / (x_sum + 1e-6)

        return x, x_normed


if __name__ == '__main__':
    from thop import profile


    model = MMFFNet().cuda()
    print(model)
    model.eval()
    x1 = torch.randn((1, 3, 480, 640)).cuda()
    x2 = torch.randn((1, 3, 480, 640)).cuda()
    flops, params = profile(model, inputs=(x1, x2))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # x1 = torch.rand(2, 3, 256, 256).cuda()
    # x2 = torch.rand(2, 3, 256, 256).cuda()
    # y1, y2 = model(x1, x2)
    # print(y1.shape, y2.shape)