from torch import nn
from .unet_parts import *

# basic model : no aug 0.3806
class Unet(nn.Module):
    def __init__(self, in_channels, n_classes,act):
        super(Unet, self).__init__()
        self.act = act
        self.inc = inconv(in_channels, 64, self.act)
        self.down1 = U_down(64, 128, self.act)
        self.down2 = U_down(128, 256, self.act)
        self.down3 = U_down(256, 512, self.act)
        self.down4 = U_down(512, 512, self.act)
        self.up1 = U_up(1024, 256, act=self.act)
        self.up2 = U_up(512, 128, act=self.act)
        self.up3 = U_up(256, 64, act=self.act)
        self.up4 = U_up(128, 64, act=self.act)
        self.out = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x

# resnext backbone :
class ResNeXtUnet(nn.Module):
    def __init__(self, in_channels, n_classes, act, cardinality=32):
        super(ResNeXtUnet, self).__init__()
        self.act = act
        
        self.inc = ResNeXtInconv(in_channels, 64, self.act, cardinality)
        self.down1 = ResNeXtDown(64, 128, self.act, cardinality)
        self.down2 = ResNeXtDown(128, 256, self.act, cardinality)
        self.down3 = ResNeXtDown(256, 512, self.act, cardinality)
        self.down4 = ResNeXtDown(512, 512, self.act, cardinality)
        self.up1 = ResNeXtUp(1024, 256, act=self.act, cardinality=cardinality)
        self.up2 = ResNeXtUp(512, 128, act=self.act, cardinality=cardinality)
        self.up3 = ResNeXtUp(256, 64, act=self.act, cardinality=cardinality)
        self.up4 = ResNeXtUp(128, 64, act=self.act, cardinality=cardinality)
        self.out = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x

def test_():
    def get_total_params(model):
        return sum([p.numel() for p in model.parameters()])
    model = Unet(in_channels=3,n_classes=30,act=nn.GELU())
    inp = torch.randn((1,3,256,512))
    with torch.no_grad():
        out = model(inp)
    print('{:,}'.format(get_total_params(model)))
    print(out.shape)
    model = ResNeXtUnet(in_channels=3,n_classes=30,act=nn.GELU())
    inp = torch.randn((1,3,256,512))
    with torch.no_grad():
        out = model(inp)
    print('{:,}'.format(get_total_params(model)))
    print(out.shape)

if __name__ == '__main__':
    test_()