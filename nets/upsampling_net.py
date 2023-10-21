import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile


# ---------------------------------------------------#
# DUpsampling
# ---------------------------------------------------#
class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding=pad, bias=False)

        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        print(x.shape)
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)
        print(x.shape)

        return x


# ---------------------------------------------------#
# CARAFE
# ---------------------------------------------------#
class CARAFE(nn.Module):
    # CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        # print("up shape:",out_tensor.shape)
        return out_tensor


# ---------------------------------------------------#
# 不完整的SAPA
# ---------------------------------------------------#
class SAPA(nn.Module):
    def __init__(self, dim_y, dim_x=None, embedding_dim=64,
                 upscale_factor=2, qkv_bias=True, norm=nn.LayerNorm):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y

        self.embedding_dim = embedding_dim

        self.norm_y = norm(dim_y)
        self.norm_x = norm(dim_x // 2)

        self.q = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
        self.k = nn.Linear(dim_x // 2, embedding_dim, bias=qkv_bias)

        self.num_attention_heads = embedding_dim
        self.attention_head_size = int(dim_y / self.num_attention_heads)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, y, x):

        y = y.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()

        y_ = self.norm_y(y)
        x_ = self.norm_x(x)

        q = self.q(y_)
        k = self.k(x_)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        out = torch.matmul(attention_probs, x)

        return out


# ---------------------------------------------------#
# IUP_BLOCK
# ---------------------------------------------------#
class IUP_BLOCK(nn.Module):
    def __init__(self, in_channel, embedding_dim=64,
                 upscale_factor=2, qkv_bias=True, norm=nn.LayerNorm):
        super().__init__()
        dim_y = in_channel // 4
        dim_x = in_channel // 2

        self.embedding_dim = embedding_dim

        self.norm_y = norm(dim_y)
        self.norm_x = norm(dim_x)

        self.q = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
        self.k = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)

        self.num_attention_heads = embedding_dim
        self.attention_head_size = int(dim_y / self.num_attention_heads)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2d = nn.Conv2d(in_channel, dim_x, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):

        y = self.pixel_shuffle(x)
        y = y.permute(0, 2, 3, 1).contiguous()
        y_ = self.norm_y(y)
        q = self.q(y_)

        x = self.conv2d(x)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x_ = self.norm_x(x)

        k = self.k(x_)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        out = torch.matmul(attention_probs, x)
        out = out.permute(0, 3, 2, 1).contiguous()
        return out


if __name__ == '__main__':
    input_shape = [40, 40]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # m = DUpsampling(1024, 2).to(device)
    # m = CARAFE(1024, 512).to(device)
    m = IUP_BLOCK(1024).to(device)

    dummy_input = torch.randn(1, 1024, input_shape[0], input_shape[1]).to(device)
    # flops, params = profile(m.to(device), (dummy_input,), verbose=False)
    flops, params = profile(m.to(device), (dummy_input, ), verbose=False)
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
