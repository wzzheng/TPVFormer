"""
Most of the code in this file is taken from https://github.com/waterljwant/SSC/blob/master/models/DDR.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out


"""
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
"""


class Bottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        dilation=[1, 1, 1],
        expansion=4,
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
        bn_momentum=0.0003,
    ):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 1, 3),
            stride=(1, 1, stride),
            dilation=(1, 1, dilation[0]),
            padding=(0, 0, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 1),
            stride=(1, stride, 1),
            dilation=(1, dilation[1], 1),
            padding=(0, dilation[1], 0),
            bias=False,
        )
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            dilation=(dilation[2], 1, 1),
            padding=(dilation[2], 0, 0),
            bias=False,
        )
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False
        )
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu
