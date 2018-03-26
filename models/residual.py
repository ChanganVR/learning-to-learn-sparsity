import torch
import torch.nn as nn


class ResidualUnit(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(ResidualUnit, self).__init__()

        bottle_planes = out_planes // self.expansion
        self.learned = nn.Sequential(
            # conv 1x1
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            nn.Conv2d(in_planes, bottle_planes, kernel_size=1, stride=stride, padding=0, bias=False),
            # conv 3x3
            nn.BatchNorm2d(bottle_planes),
            nn.ReLU(True),
            nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=1, padding=1, bias=False),
            # conv 1x1
            nn.BatchNorm2d(bottle_planes),
            nn.ReLU(True),
            nn.Conv2d(bottle_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.learned(x)
