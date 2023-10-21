import torch
import torch.nn as nn
import math
import torch.autograd
from thop import clever_format, profile


# ---------------------------------------------------#
#   注意力机制通常使用在BackBone之后
# ---------------------------------------------------#


# ---------------------------------------------------#
# SE：通道注意力
# ---------------------------------------------------#
class SE_BLOCK(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE_BLOCK, self).__init__()
        # 在高和宽上进行全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # b,c,h,w -> b,c,1,1 -> b,c
        y = self.avg_pool(x).view(b, c)
        # b,c -> b,c // ratio -> b,c ->b,c,1,1
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ---------------------------------------------------#
# CBAM：空间-通道注意力
# ---------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("ChannelAttention/avg_pool:{}".format(self.avg_pool(x).shape))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 两个通道分别为avg和max的结果
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print("SpatialAttention/avg_out:{}".format(avg_out.shape))
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM_BLOCK(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM_BLOCK, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


# ---------------------------------------------------#
# ECA：通道注意力
# ---------------------------------------------------#
class ECA_Block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_Block, self).__init__()
        # 自适应计算卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 调整为序列的形式
        y = self.avg_pool(x).view([b, 1, c])
        # 1d卷积特征提取
        y = self.conv(y)
        y = self.sigmoid(y).view([b, c, 1, 1])
        print(y)
        return y * x


# ---------------------------------------------------#
# EPSANet：改进的通道注意力
# ---------------------------------------------------#
class SE_EPSANet(SE_BLOCK):
    def forward(self, x):
        b, c, _, _ = x.size()
        # b,c,h,w -> b,c,1,1 -> b,c
        y = self.avg_pool(x).view(b, c)
        # b,c -> b,c // ratio -> b,c ->b,c,1,1
        y = self.fc(y).view(b, c, 1, 1)
        return y


class EPSANet(nn.Module):
    def __init__(self, inplans, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(EPSANet, self).__init__()
        self.conv_1 = nn.Conv2d(inplans, inplans // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                                stride=stride, groups=conv_groups[0])
        self.conv_2 = nn.Conv2d(inplans, inplans // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                                stride=stride, groups=conv_groups[1])
        self.conv_3 = nn.Conv2d(inplans, inplans // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                                stride=stride, groups=conv_groups[2])
        self.conv_4 = nn.Conv2d(inplans, inplans // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                                stride=stride, groups=conv_groups[3])
        self.se = SE_EPSANet(inplans // 4)
        self.split_channel = inplans // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        return out


# ---------------------------------------------------#
# ICB：改进的空间-通道注意力
# ---------------------------------------------------#
# class C_Attention(nn.Module):
#     def __init__(self, channel, b=1, gamma=2):
#         super(C_Attention, self).__init__()
#         # 自适应计算卷积核大小
#         kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.conv1d = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#         inter_channel = channel // 4
#         self.conv2d_1 = nn.Conv2d(channel, inter_channel, kernel_size=1, dilation=1)
#         self.conv2d_2 = nn.Conv2d(channel, inter_channel, kernel_size=3, padding=1, dilation=1)
#         self.conv2d_3 = nn.Conv2d(channel, inter_channel, kernel_size=3, padding=3, dilation=3)
#         self.conv2d_4 = nn.Conv2d(channel, inter_channel, kernel_size=3, padding=5, dilation=5)
#
#         self.bn2 = nn.BatchNorm2d(channel, affine=True)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         # 代替特征金字塔(EPSANet)
#         x0 = self.conv2d_1(x)
#         x1 = self.conv2d_2(x)
#         x2 = self.conv2d_3(x)
#         x3 = self.conv2d_4(x)
#
#         x0 = self.max_pool(x0)
#         x1 = self.max_pool(x1)
#         x2 = self.max_pool(x2)
#         x3 = self.max_pool(x3)
#
#         ppm_x = torch.cat((x0, x1, x2, x3), dim=1).view([b, 1, c])
#
#         # 调整为序列的形式(ECA)
#         y = self.avg_pool(x).view([b, 1, c])
#         y = torch.cat((y, ppm_x), dim=1)
#
#         # 1d卷积特征提取
#         y = self.conv1d(y)
#         y = self.sigmoid(y).view([b, c, 1, 1])
#
#         return y * x
#
#
# class S_Attention(nn.Module):
#     def __init__(self, channel, kernel_size=7):
#         super(S_Attention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         # 两个通道分别为avg和max的结果
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x):
#
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

class ICS_BLOCK(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ICS_BLOCK, self).__init__()
        # 在高和宽上进行全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # b,c,h,w -> b,c,1,1 -> b,c
        y = self.avg_pool(x).view(b, c)
        # b,c -> b,c // ratio -> b,c ->b,c,1,1
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BackBoneBlock(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(BackBoneBlock, self).__init__()
        middle_channle = in_channel // ratio
        out_channel = in_channel
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channel, out_channels=middle_channle, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(middle_channle),
            nn.ReLU(),
            nn.Conv2d(in_channels=middle_channle, out_channels=middle_channle, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(middle_channle),
            nn.Conv2d(in_channels=middle_channle, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        y = self.layer(x)
        x = x + y
        return x


if __name__ == '__main__':
    model = BackBoneBlock(512)
    input = torch.randn([2, 512, 640, 640])
    flops, params = profile(model, (input,), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
