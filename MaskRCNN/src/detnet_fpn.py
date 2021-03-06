import math

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os

'''
The Detnet-59 backbone implementation is based on the Detnet-50 implementation from CS444 project 3 starter code:
https://slazebni.cs.illinois.edu/spring22/assignment3_part2.html

It is then modified by me based on Feature Pyramid Networks for Object Detection:
https://arxiv.org/pdf/1612.03144.pdf

-- Peiyan Wu
'''

# Based on Resnet implementation from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# These layers are based on DetNet: A Backbone network for Object Detection, https://arxiv.org/pdf/1804.06215.pdf
class DetnetBottleneck(nn.Module):
    # We keep the same grid size in the output. (SxS)
    # Layer structre is 1x1 conv, dilated 3x3 conv, 1x1 conv, with a skip connection
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type="A"):
        super(DetnetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            bias=False,
            dilation=2,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes or block_type == "B":
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Detnet_FPN(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(Detnet_FPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.detLayer1 = self._make_detnet_layer(in_channels=1024)
        self.detLayer2 = self._make_detnet_layer(in_channels=256)

        self.proj0 = nn.Conv2d(256, 256, 1)
        self.proj1 = nn.Conv2d(256, 256, 1)
        self.proj2 = nn.Conv2d(1024, 256, 1)
        self.proj3 = nn.Conv2d(512, 256, 1)
        self.proj4 = nn.Conv2d(256, 256, 1)

        self.upsample3 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels):
        layers = []
        layers.append(
            DetnetBottleneck(in_planes=in_channels, planes=256, block_type="B")
        )
        layers.append(DetnetBottleneck(in_planes=256, planes=256, block_type="A"))
        layers.append(DetnetBottleneck(in_planes=256, planes=256, block_type="A"))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 2 stride

        ### Bottom-up path
        C2 = self.layer1(x)     # 4 stride
        C3 = self.layer2(C2)    # 8 stride
        C4 = self.layer3(C3)    # 16 stride

        C5 = self.detLayer1(C4) # 16 stride
        C6 = self.detLayer2(C5) # 16 stride

        ### Top-down path
        M6 = self.proj0(C6)                         # 16 stride
        M5 = M6 + self.proj1(C5)                    # 16 stride
        M4 = M5 + self.proj2(C4)                    # 16 stride
        M3 = self.upsample3(M4) + self.proj3(C3)    # 8 stride
        M2 = self.upsample4(M3) + self.proj4(C2)    # 4 stride

        feature_maps = [M2, M3, M4, M5]

        return feature_maps


def update_state_dict(pretrained_state_dict, model):
    dd = model.state_dict()
    for k in pretrained_state_dict.keys():
        if k in dd.keys() and not k.startswith("fc"):
            dd[k] = pretrained_state_dict[k]
    model.load_state_dict(dd)
    return model


def detnet59_fpn(pretrained=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Detnet_FPN(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained is not None:
        if os.path.isfile(pretrained):
            pretrained_model = torch.load(pretrained)
            pretrained_state_dict = pretrained_model["state_dict"]
            model = update_state_dict(pretrained_state_dict, model)
    return model
