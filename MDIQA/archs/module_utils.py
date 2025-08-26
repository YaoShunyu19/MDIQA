import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_chns, out_chns, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_chns, out_chns, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        a = torch.max(x, 1)[0].unsqueeze(1)
        b = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((a, b), dim=1)


class SpatialAttnLayer(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttnLayer, self).__init__()
        self.ChannelPool = ChannelPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # B,C,H,W
        x_compress = self.ChannelPool(x)  # B,2,H,W
        x_out = self.conv(x_compress)  # B,1,H,W
        scale = torch.sigmoid(x_out)
        return x * scale
