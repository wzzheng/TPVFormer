import torch
import torch.nn as nn
from kitti_ssc.models.DDR import Bottleneck3D


class ASPP(nn.Module):
    """
    ASPP 3D
    Adapt from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, planes, dilations_conv_list):
        super().__init__()

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

    def forward(self, x_in):

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        return x_in


class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    Taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

        self.conv_classes = nn.Conv3d(
            planes, nbr_classes, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x_in):

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in


class ProcessKitti(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[
                Bottleneck3D(
                    feature,
                    feature // 4,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=[i, i, i],
                )
                for i in dilations
            ]
        )

    def forward(self, x):
        return self.main(x)


class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[
                Bottleneck3D(
                    feature,
                    feature // 4,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=[i, i, i],
                )
                for i in dilations
            ]
        )

    def forward(self, x):
        return self.main(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(
            feature,
            feature // 4,
            bn_momentum=bn_momentum,
            expansion=expansion,
            stride=2,
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    feature,
                    int(feature * expansion / 4),
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                norm_layer(int(feature * expansion / 4), momentum=bn_momentum),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, x):
        return self.main(x)
