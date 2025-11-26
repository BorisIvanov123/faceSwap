import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic Blocks
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv(x)
        att = self.att(feat)
        return feat * att


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 4, out_chan, 1),
            nn.Sigmoid()
        )

    def forward(self, f_sp, f_cp):
        feat = torch.cat([f_sp, f_cp], dim=1)
        feat = self.convblk(feat)
        att = self.att(feat)
        return feat * att + feat


# -----------------------------
# BiSeNet Architecture
# -----------------------------

class BiSeNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Spatial Path
        self.sp1 = ConvBNReLU(3, 64, stride=2)
        self.sp2 = ConvBNReLU(64, 64, stride=2)
        self.sp3 = ConvBNReLU(64, 64, stride=2)

        # Context Path
        self.cp1 = ConvBNReLU(3, 64, stride=2)
        self.cp2 = ConvBNReLU(64, 128, stride=2)
        self.cp3 = ConvBNReLU(128, 256, stride=2)

        self.arm1 = AttentionRefinementModule(256, 128)
        self.arm2 = AttentionRefinementModule(128, 128)

        # Fusion
        self.ffm = FeatureFusionModule(64 + 128, 128)

        # Output segmentation head
        self.conv_out = nn.Sequential(
            ConvBNReLU(128, 128),
            nn.Conv2d(128, n_classes, 1)
        )

    def forward(self, x):

        # Spatial Path
        sp = self.sp1(x)
        sp = self.sp2(sp)
        sp = self.sp3(sp)

        # Context Path
        cp = self.cp1(x)
        cp = self.cp2(cp)
        cp = self.cp3(cp)

        cp = self.arm1(cp)
        cp = F.interpolate(cp, size=sp.shape[2:], mode='bilinear', align_corners=True)

        # Fusion
        feat = self.ffm(sp, cp)

        # Output segmentation
        out = self.conv_out(feat)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out
