import torch.nn as nn
from .vmamba import LayerNorm2d

# Shared norm/act layer registries used across all ChangeMamba model files.
_NORMLAYERS = dict(
    ln=nn.LayerNorm,
    ln2d=LayerNorm2d,
    bn=nn.BatchNorm2d,
)

_ACTLAYERS = dict(
    silu=nn.SiLU,
    gelu=nn.GELU,
    relu=nn.ReLU,
    sigmoid=nn.Sigmoid,
)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
