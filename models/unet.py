import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
'''U-Net Structure'''

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x1 = self.double_conv(x)
        return x1


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, dimension=64, bilinear=True, multi_head=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, dimension)
        self.down1 = Down(dimension, dimension * 2)
        self.down2 = Down(dimension * 2, dimension * 4)
        self.down3 = Down(dimension * 4, dimension * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(dimension * 8, (dimension * 16) // factor)
        self.up1 = Up(dimension * 16, (dimension * 8) // factor, bilinear)
        self.up2 = Up(dimension * 8, (dimension * 4) // factor, bilinear)
        self.up3 = Up(dimension * 4, (dimension * 2) // factor, bilinear)
        self.up4 = Up(dimension * 2, dimension, bilinear)
        self.outc = MultiHead(dimension, n_classes) if multi_head \
            else OutConv(dimension, n_classes)

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
        logits = self.outc(x)
        return logits

class SingleHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1))

    def forward(self, x):
        return self.head(x)

class MultiHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiHead, self).__init__()
        self.head1 = SingleHead(in_channels, out_channels)
        self.head2 = SingleHead(in_channels, out_channels)
        self.head3 = SingleHead(in_channels, out_channels)
        self.head4 = SingleHead(in_channels, out_channels)
        self.head5 = SingleHead(in_channels, out_channels)

    def forward(self, x):
        y1 = self.head1(x) # pos10
        y2 = self.head2(x) # neg10
        y3 = self.head3(x) # pos40
        y4 = self.head4(x) # neg40
        y5 = self.head5(x) # others, except water (lulc=80)
        return y1, y2, y3, y4, y5


##############################################################################
if __name__ == "__main__":
    data = torch.rand((1,6,512,512),dtype=torch.float32)
    model = UNet(6,2)
    output = model(data)
    print(output.shape)