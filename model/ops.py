import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F




class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out



class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        
    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


