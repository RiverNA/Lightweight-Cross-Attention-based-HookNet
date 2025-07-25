import torch
import torch.nn as nn
from Light_HookNet_parts import *


class Light_HookNet(nn.Module):
    def __init__(self, n_channels, n_classes, filter_size=3, n_filters=64):
        super(Light_HookNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.n_filters = n_filters

        # Context branch
        self.c_first = Conv_block(n_channels, n_filters)
        self.c_down1 = Downsample(n_filters, n_filters * 2)
        self.c_down2 = Downsample(n_filters * 2, n_filters * 4)
        self.c_down3 = Downsample(n_filters * 4, n_filters * 8)
        self.c_down4 = Downsample(n_filters * 8, n_filters * 10)

        self.c_up1 = Upsample(n_filters * 10, n_filters * 8)
        self.c_up2 = Upsample(n_filters * 8, n_filters * 4)
        self.c_up3 = Upsample(n_filters * 4, n_filters * 2)
        self.c_up4 = Upsample(n_filters * 2, n_filters)

        self.c_out = Output(n_filters, n_classes)

        # Target branch
        self.t_first = Conv_block(n_channels, n_filters)
        self.t_down1 = Downsample(n_filters, n_filters * 2)
        self.t_down2 = Downsample(n_filters * 2, n_filters * 4)
        self.t_down3 = Downsample(n_filters * 4, n_filters * 8)
        self.t_down4 = Downsample(n_filters * 8, n_filters * 10)

        self.t_up1 = T_Upsample(n_filters * 18, n_filters * 8, n_filters, n_classes, i='up1')
        self.t_up2 = T_Upsample(n_filters * 12, n_filters * 4, n_filters, n_classes, i='up2')
        self.t_up3 = T_Upsample(n_filters * 6, n_filters * 2, n_filters, n_classes, i='up3')
        self.t_up4 = Upsample(n_filters * 2, n_filters)

        self.t_out = Output(n_filters, n_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        c_residuals = []
        c_outhooks = []
        
        x = self.c_first(x)
        c_residuals.append(x)
        x = self.c_down1(x)
        c_residuals.append(x)
        x = self.c_down2(x)
        c_residuals.append(x)
        x = self.c_down3(x)
        c_residuals.append(x)
        x = self.c_down4(x)

        x = self.c_up1(x, c_residuals[-1])
        c_outhooks.append(x)
        x = self.c_up2(x, c_residuals[-2])
        c_outhooks.append(x)
        x = self.c_up3(x, c_residuals[-3])
        c_outhooks.append(x)
        x = self.c_up4(x, c_residuals[-4])
        c_outhooks.append(x)
        x = self.c_out(x)

        t_residuals = []
        
        y = self.t_first(y)
        t_residuals.append(y)
        y = self.t_down1(y)
        t_residuals.append(y)
        y = self.t_down2(y)
        t_residuals.append(y)
        y = self.t_down3(y)
        t_residuals.append(y)
        y = self.t_down4(y)

        z = []
        y, yy = self.t_up1(y, t_residuals[-1], c_outhooks[0])
        z.append(yy)
        y, yy = self.t_up2(y, t_residuals[-2], c_outhooks[1])
        z.append(yy)
        y, yy = self.t_up3(y, t_residuals[-3], c_outhooks[2])
        z.append(yy)
        y = self.t_up4(y, t_residuals[-4])
        y = self.t_out(y)

        return x, y, z
