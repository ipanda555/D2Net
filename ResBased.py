import paddle
from paddle import nn
from paddle.nn import functional as F
# from HRNet import HRNet_W18_C


class ResNet(paddle.nn.Layer):
    def __init__(self):
        super(ResNet, self).__init__()
        r50 = paddle.vision.models.resnet50(True)
        self.cv1 = r50.conv1
        self.bn1 = r50.bn1
        self.maxp = r50.maxpool
        self.relu = r50.relu
        self.layer1 = r50.layer1
        self.layer2 = r50.layer2
        self.layer3 = r50.layer3
        self.layer4 = r50.layer4

    def forward(self, x):
        x1 = self.maxp(self.relu(self.bn1(self.cv1(x))))
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5

    def init_weight(self):
        print("bkbone finished!")


class Squeeze(nn.Layer):
    def __init__(self):
        super(Squeeze, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(256, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        self.cv2 = nn.Sequential(
            nn.Conv2D(256*2, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        self.cv3 = nn.Sequential(
            nn.Conv2D(256*4, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        self.cv4 = nn.Sequential(
            nn.Conv2D(256*8, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

    def forward(self, x1, x2, x3, x4):
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)
        x3 = self.cv3(x3)
        x4 = self.cv4(x4)
        return x1, x2, x3, x4


class ASPP(nn.Layer):
    def __init__(self):
        super(ASPP, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 2, dilation=2), # rf: 3 + (3 - 1) * (3 - 1) = 4
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2D(64, 64, 5, 1, 4, dilation=2), # rf: 5 + (5-1) * (2-1) = 9
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2D(64, 64, 7, 1, 6, dilation=2),  # rf: 7 + (7-1) * (2-1) = 13
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv4 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x)
        x4 = self.cv4(x)
        x  = paddle.concat([x1, x2, x3, x4], 1)
        return x
# class ASPP(nn.Layer):#消融实验测试
#     def __init__(self):
#         super(ASPP, self).__init__()
#         self.cv1 = nn.Sequential(
#             nn.Conv2D(64, 64, 3, 1, 2, dilation=6), # rf: 3 + (3 - 1) * (3 - 1) = 4
#             nn.BatchNorm2D(64),
#             nn.ReLU()
#         )
#         self.cv2 = nn.Sequential(
#             nn.Conv2D(64, 64, 3, 1, 4, dilation=12), # rf: 5 + (5-1) * (2-1) = 9
#             nn.BatchNorm2D(64),
#             nn.ReLU()
#         )
#         self.cv3 = nn.Sequential(
#             nn.Conv2D(64, 64, 3, 1, 6, dilation=18),  # rf: 7 + (7-1) * (2-1) = 13
#             nn.BatchNorm2D(64),
#             nn.ReLU()
#         )
#         self.cv4 = nn.Sequential(
#             nn.Conv2D(64, 64, 3, 1, 1, dilation=24),
#             nn.BatchNorm2D(64),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = self.cv2(x)
#         x3 = self.cv3(x)
#         x4 = self.cv4(x)
#         x  = paddle.concat([x1, x2, x3, x4], 1)
#         return x


class Low_Level(nn.Layer):
    def __init__(self):
        super(Low_Level, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(64, 64, 7, 1, 3),
            nn.BatchNorm2D(64),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2D(64, 64, 7, 1, 3),
            nn.BatchNorm2D(64),
        )

    def forward(self, x):
        x = F.relu(x + self.cv1(x))
        x = F.relu(x + self.cv2(x))
        return x


class AdaptivateFusion(nn.Layer):
    def __init__(self):
        super(AdaptivateFusion, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 2, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 2, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2D(64*2, 64, 3, 1, 1),
            nn.BatchNorm2D(64)
        )
        self.cv4 = nn.Sequential(
            nn.Conv2D(64, 64, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        self.cv5 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
        )
        self.cv7 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64)
        )

        self.cv6 = nn.Sequential(
            nn.Conv2D(64*2, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
        )

    def forward(self, xh, xl):#fl尺寸为88*88*128,fh尺寸为22*22*128
        ad = self.cv2(self.cv1(xl))
        xh = F.relu(self.cv3(paddle.concat([ad, xh],1)) + self.cv5(xh))
        xh = F.interpolate(xh, size=xl.shape[2:], mode='bilinear')
        xh = self.cv4(xh)
      #  xh = self.cv5(xh)
        f  = F.relu(self.cv6(paddle.concat([xl, xh],1)) + self.cv7(xl))
        return f


class HRSODNet(nn.Layer):
    def __init__(self):
        super(HRSODNet, self).__init__()
        self.bkbone   = ResNet()
       # self.bkbone.load_dict(paddle.load('external-libraries/P2Net/HRNet_W18_C_pretrained.pdparams'))
        self.squeeze  = Squeeze()
        self.high_le3 = ASPP()
        self.high_le4 = ASPP()
        self.highs    = nn.Sequential(nn.Conv2D(64 * 4 * 2, 64, 3, 1, 1), nn.BatchNorm2D(64), nn.ReLU())
        self.lows     = nn.Sequential(nn.Conv2D(64 * 2, 64, 3, 1, 1), nn.BatchNorm2D(64), nn.ReLU())
        self.low_le1  = Low_Level()
        self.low_le2  = Low_Level()
        self.adpf     = AdaptivateFusion()
        self.out      = nn.Conv2D(64, 1, 3, 1, 1)
        for p in self.bkbone.parameters():
            p.optimize_attr['learning_rate'] /= 10.0

    def forward(self, x):
        x1, x2, x3, x4 = self.bkbone(x)
        x1, x2, x3, x4 = self.squeeze(x1, x2, x3, x4)
        x3, x4         = self.high_le3(x3), self.high_le4(x4)
        x4             = F.interpolate(x4, size=x3.shape[2:], mode='bilinear')
        xh             = self.highs(paddle.concat([x3, x4], 1))#尺寸为22*22*128
        print("------------xh------------")
        print(xh)
        x1, x2         = self.low_le1(x1), self.low_le2(x2)
        x2             = F.interpolate(x2, size=x1.shape[2:], mode='bilinear')
        xl             = self.lows(paddle.concat([x1, x2], 1))#尺寸为88*88*128
        print("------------xl------------")
        print(xl);
        f              = self.adpf(xh, xl)
        out            = F.interpolate(self.out(f), size=x.shape[2:], mode='bilinear')
        return out
        # return f


if __name__ == '__main__':
    net = HRSODNet()
    paddle.flops(net, (1, 3, 352, 352))



