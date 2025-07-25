import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(in_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.catconv = Conv_block(out_channels * 2, out_channels, kernel_size)

    def forward(self, x, y):
        x = self.up(x)
        x = self.conv(x)
        crop_size = int(y.shape[3] - x.shape[3]) / 2
        _, _, h, w = y.shape
        item_cropped = y[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        x = torch.cat((x, item_cropped), dim=1)
        return self.catconv(x)


class FFN(nn.Module):
    def __init__(self, in_channels, drop_rate=0.):
        super().__init__()
        ratio = 4
        self.DW = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.MLP_0 = nn.Conv2d(in_channels, ratio * in_channels, kernel_size=1)
        self.MLP_1 = nn.Conv2d(ratio * in_channels, in_channels, kernel_size=1)
        self.Norm = nn.LayerNorm(in_channels)
        self.GELU = nn.GELU()
        self.drop_path = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        skip = x
        x = self.MLP_1(self.GELU(self.MLP_0(self.Norm(self.DW(x).permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous())))
        return self.drop_path(x) + skip


class T_Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters, n_classes, i, kernel_size=3):
        super().__init__()
        self.catconv = Conv_block(out_channels * 2, out_channels, kernel_size)
        ratio = 2
        if i == 'up1':
            self.up = nn.ConvTranspose2d(n_filters * 10, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.FFN = FFN(n_filters * 10)
            self.Pre = nn.Conv2d(n_filters * 8, n_classes, kernel_size=1)
            self.queryd = nn.Conv2d(n_filters * 8, n_filters * 10, kernel_size=1)
            self.keyd = nn.Conv2d(n_filters * 10, n_filters * 10, kernel_size=1)
            self.valued = nn.Conv2d(n_filters * 10, n_filters * 10, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

        elif i == 'up2':
            self.up = nn.ConvTranspose2d(n_filters * 8, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.FFN = FFN(n_filters * 8)
            self.Pre = nn.Conv2d(n_filters * 4, n_classes, kernel_size=1)
            self.queryd = nn.Conv2d(n_filters * 4, n_filters * 8, kernel_size=1)
            self.keyd = nn.Conv2d(n_filters * 8, n_filters * 8, kernel_size=1)
            self.valued = nn.Conv2d(n_filters * 8, n_filters * 8, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

        elif i == 'up3':
            self.up = nn.ConvTranspose2d(n_filters * 4, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.FFN = FFN(n_filters * 4)
            self.Pre = nn.Conv2d(n_filters * 2, n_classes, kernel_size=1)
            self.queryd = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=1)
            self.keyd = nn.Conv2d(n_filters * 4, n_filters * 4, kernel_size=1)
            self.valued = nn.Conv2d(n_filters * 4, n_filters * 4, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

    def forward(self, x, y, z):
        crop_size = int(z.shape[3] - x.shape[3]) / 2
        _, _, h, w = z.shape
        item_cropped = z[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        B, C, H, W = x.shape
        skip = x
        Q = self.queryd(item_cropped).view(B, -1, H * W).permute(0, 2, 1).contiguous()
        K = self.keyd(x).view(B, -1, H * W)
        _, head_dim, _ = K.shape
        attention = self.softmaxd(torch.bmm(Q, K) / (head_dim ** 0.5))
        V = self.valued(x).view(B, C, H * W).permute(0, 2, 1).contiguous()
        out = torch.bmm(attention, V).permute(0, 2, 1).contiguous().view(B, C, H, W)

        x = self.FFN(self.gammad * out + skip)
        x = self.up(x)
        x = self.conv(x)

        crop_size = int(y.shape[3] - x.shape[3]) / 2
        _, _, h, w = y.shape
        item_cropped = y[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        x = torch.cat((x, item_cropped), dim=1)
        output = self.catconv(x)
        return output, self.Pre(output)


class Output(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        return self.output(x)
