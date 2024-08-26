import torch
import torch.nn as nn
import torch.nn.functional as F


# basic parts for Unet
class U_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch,act):
        super(U_double_conv, self).__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            self.act,
            nn.Conv2d(out_ch, out_ch, 3, padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            self.act
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, act):
        super(inconv, self).__init__()
        self.conv = U_double_conv(in_ch, out_ch, act)

    def forward(self, x):
        x = self.conv(x)
        return x


class U_down(nn.Module):
    def __init__(self, in_ch, out_ch, act):
        super(U_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            U_double_conv(in_ch, out_ch, act)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class U_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, act=None):
        super(U_up, self).__init__()
        self.act = act
        assert self.act is not None, 'act is none'
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2,bias=False)

        self.conv = U_double_conv(in_ch, out_ch, act)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x

# ResNext parts
class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cardinality, act):
        super(ResNeXtBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.act(out)
        return out

class ResNeXtInconv(nn.Module):
    def __init__(self, in_ch, out_ch, act, cardinality=32):
        super(ResNeXtInconv, self).__init__()
        self.conv = ResNeXtBlock(in_ch, out_ch, cardinality, act)

    def forward(self, x):
        x = self.conv(x)
        return x

class ResNeXtDown(nn.Module):
    def __init__(self, in_ch, out_ch, act, cardinality=32):
        super(ResNeXtDown, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNeXtBlock(in_ch, out_ch, cardinality, act)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class ResNeXtUp(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, act=None, cardinality=32):
        super(ResNeXtUp, self).__init__()
        self.act = act
        assert self.act is not None, 'act is none'
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2, bias=False)

        self.conv = ResNeXtBlock(in_ch, out_ch, cardinality, act)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0, diffX, 0))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# classifer
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x