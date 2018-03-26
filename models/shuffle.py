import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        print(N, g, C // g, H, W)
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes // 4
        g = 1 if in_planes == 24 else groups
        self.learned = nn.Sequential(
            # conv3x3 adapted from ResNets
            nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(True),
            # Shuffle
            ShuffleBlock(groups=g),
            # Conv3x3
            nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )

        # self.shortcut = nn.Sequential()
        # if stride == 2:
        #     self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = self.learned(x)
        # res = self.shortcut(x)
        # out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out


if __name__ == "__main__":
    t = torch.randn(3, 64, 224, 224)
    d = Variable(t)

    m = Bottleneck(64, 64, 1, 4)

    out = m(d)

    print(out.size())
