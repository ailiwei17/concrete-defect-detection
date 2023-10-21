import torch
import torch.nn as nn


class Stem(nn.Module):
    def __init__(self, in_features, out_features):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


if __name__ == '__main__':
    input_test = torch.randn([2, 128, 40, 40])
    model = Stem(128, 256)
    output = model(input_test)
