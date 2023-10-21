import numpy as np
import torch
import torch.nn as nn
from thop import clever_format, profile
from .backbone import Conv, SiLU, autopad


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1):
        # https://github.com/irvinxav/Efficient-HetConv-Heterogeneous-Kernel-Based-Convolutions/blob/master/hetconv.py
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class simple_Conv(Conv):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = HetConv(c1, c2, k, s, autopad(k, p), g)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())


class simple_SPP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        # 该段代码为分组卷积 + HetConv
        super(simple_SPP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, g=4)
        self.cv2 = Conv(c1, c_, 1, 1, g=4)
        self.cv3 = simple_Conv(c_, c_, 3, 1, g=4)
        self.cv4 = Conv(c_, c_, 1, 1, g=4)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1, g=4)
        self.cv6 = simple_Conv(c_, c_, 3, 1, g=4)
        self.cv7 = Conv(2 * c_, c2, 1, 1, g=4)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


if __name__ == '__main__':
    input_shape = [40, 40]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = simple_SPP(1024, 512).to(device)

    dummy_input = torch.randn(1, 1024, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(m.to(device), (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
