import math
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

class yolo_head(nn.Module):
    def __init__(self, in_channels, out_channels0, out_channels1):
        super(yolo_head, self).__init__()
        self.basicC = BasicConv(in_channels, out_channels0, 3)
        self.conv2d = nn.Conv2d(out_channels0, out_channels1, 1)

    def forward(self, x):
        x = self.basicC(x)
        x = self.conv2d(x)
        return x

class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()

        # 1x640x640 --> 32x320x320
        self.conv1 = BasicConv(1, 32, kernel_size=3, stride=2)
        # 32x320x320 --> 64x160x160
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        # 64x160x160 -->128x80x80
        self.conv3 = BasicConv(64, 128, kernel_size=3, stride=2)
        # 128x80x80 --> 256x40x40
        self.conv4 = BasicConv(128, 256, kernel_size=3, stride=2)
        # 256x40x40 --> 512x20x20
        self.conv5 = BasicConv(256, 512, kernel_size=3, stride=2)
        # 512x20x20 --> 256x20x20
        self.conv6 = BasicConv(512, 256, kernel_size=1 ,stride=1)

        self.yolo_headP4 = yolo_head(256, 512, num_anchors*(5+num_classes))

        self.up_sample = Upsample(256, 128)

        self.yolo_headP5 = yolo_head(384, 256, num_anchors*(5+num_classes))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.concat = nn.quantized.FloatFunctional()

    def forward(self, x):
        x = self.quant(x)

        # 1x640x640 --> 32x320x320
        c1 = self.conv1(x)
        # 32x320x320 --> 64x160x160
        c2 = self.conv2(c1)
        # 64x160x160 --> 128x80x80
        c3 = self.conv3(c2)
        # 128x80x80 --> 256x40x40
        c4 = self.conv4(c3)
        route = c4

        # 256x40x40 --> 512x20x20
        c5 = self.conv5(c4)
        # 512x20x20 --> 256x20x20
        c6 = self.conv6(c5)

        # 256x20x20 --> 512x20x20
        # 512x20x20 --> 18x20x20
        out0 = self.yolo_headP4(c6)

        # 256x20x20 --> 128x40x40
        upsample = self.up_sample(c6)

        # 256x40x40 + 128x40x40 --> 384x40x40
        cat = self.concat.cat([upsample, route], 1)

        # 384x40x40 --> 256x40x40
        # 256x40x40 --> 18x40x40
        out1 = self.yolo_headP5(cat)

        d_out0 = self.dequant(out0)
        d_out1 = self.dequant(out1)

        return d_out0, d_out1

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, BasicConv):
                torch.quantization.fuse_modules(m.conv, [['0', '1', '2']], inplace=True)
model = YoloBody(3, 1)
print(model)



