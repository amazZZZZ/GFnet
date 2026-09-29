
import torch
from .conv import DWConv,Conv,RepConv
from .block import RepBottleneck,Bottleneck,C3k
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
# __all__ = (
#     "SS2D",
# )
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape: b, num_channels, h, w  -->  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channelshuffle
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
class EMAP(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMAP, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1, 1, 0, bias=True)
        self.fc2 = nn.Conv2d(channels // self.groups, channels // self.groups, 1, 1, 0, bias=True)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.spatt=SpatialAttention(7)
 
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)).reshape(b * self.groups, -1, h+w, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.fc1(self.agp(x1)).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.fc2(self.agp(x2)).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out=self.spatt((group_x * weights.sigmoid()).reshape(b, c, h, w))
        return channel_shuffle(out,groups=4)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f"SpatialAttention input shape: {x.shape}")
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        weight = self.sigmoid(self.conv(feat))
        # print(f"SpatialAttention output shape: {(x * weight).shape}")
        return x * weight

class DynamicGateFusion(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        # self.gate_conv = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(channel * 2, channel // 8, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(channel // 8, 2, 1),
        #     nn.Softmax(dim=1)
        # )
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 保留全局感知，也可以去掉
            nn.Conv2d(channel * 2, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel * 2, 1, bias=True),
            nn.Sigmoid()  # 改成Sigmoid，两个分支各自有权重
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        gates = self.gate_conv(combined)
        g1, g2 = gates.chunk(2, dim=1)
        return x1 * g1 + x2 * g2

class Simam_module(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)

class DynamicGateFusion_SimAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.simam1 = Simam_module()
        self.simam2 = Simam_module()

        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel * 2, channel // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        # 先分别加 SimAM 注意力
        x1 = self.simam1(x1)
        x2 = self.simam2(x2)

        # 然后再做动态融合
        combined = torch.cat([x1, x2], dim=1)
        gates = self.gate_conv(combined)
        g1, g2 = gates.chunk(2, dim=1)
        return x1 * g1 + x2 * g2
class CF(nn.Module):
    def __init__(self, in1_channels, in2_channels, out_channels, num_blocks=3, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.fc1 = Conv(in1_channels, hidden_channels, 1, 1)
        self.fc2 = Conv(in2_channels, hidden_channels, 1, 1)
        self.sa = SpatialAttention()

        self.dynamic_gate = DynamicGateFusion_SimAM(hidden_channels)

        self.fc3 = Conv(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):  # 修改点：统一接口
        assert isinstance(x, (list, tuple)) and len(x) == 2, "CCFF expects a list or tuple of two tensors."
        x1, x2 = x
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        fused = self.dynamic_gate(x1, x2)
        fused = self.sa(fused)

        return self.fc3(fused)
    
class CF1(nn.Module):
    def __init__(self, in1_channels, in2_channels, out_channels,expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.fc1 = Conv(in1_channels, hidden_channels, 1, 1)
        self.fc2 = Conv(in2_channels, hidden_channels, 1, 1)
        self.sa = SpatialAttention()

        self.dynamic_gate = DynamicGateFusion_SimAM(hidden_channels)

        self.fc3 = Conv(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()
        self.ffn = nn.Sequential(Conv(hidden_channels, hidden_channels * 2, 1), Conv(hidden_channels * 2, hidden_channels, 1, act=False))
        

    def forward(self, x):  # 修改点：统一接口
        assert isinstance(x, (list, tuple)) and len(x) == 2, "CF expects a list or tuple of two tensors."
        x1, x2 = x
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        fused = self.dynamic_gate(x1, x2)
        fused = fused + self.sa(fused)
        fused = fused + self.ffn(fused)
        return self.fc3(fused)
from .block import Attention
class CF2(nn.Module):
    def __init__(self, in1_channels, in2_channels, out_channels, num_blocks=3, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.fc1 = Conv(in1_channels, hidden_channels, 1, 1)
        self.fc2 = Conv(in2_channels, hidden_channels, 1, 1)
        
        self.dynamic_gate = DynamicGateFusion_SimAM(hidden_channels)
        
        self.att = CAA(hidden_channels)
        self.ffn = nn.Sequential(Conv(hidden_channels, hidden_channels * 2, 1), Conv(hidden_channels * 2, hidden_channels, 1, act=False))
        
        self.fc3 = Conv(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):  # 修改点：统一接口
        assert isinstance(x, (list, tuple)) and len(x) == 2, "CCFF expects a list or tuple of two tensors."
        x1, x2 = x
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        fused = self.dynamic_gate(x1, x2)
        fused = fused + self.att(fused)
        fused = fused + self.ffn(fused)
        return self.fc3(fused)



class CCFF(nn.Module):
    def __init__(self, in1_channels, in2_channels, out_channels, num_blocks=3, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.fc1 = Conv(in1_channels, hidden_channels, 1, 1)
        self.fc2 = Conv(in2_channels, hidden_channels, 1, 1)

        # self.rep_blocks = nn.Sequential(*[
        #     RepBottleneck(hidden_channels, hidden_channels, shortcut=True, g=1, e=1)
        #     for _ in range(num_blocks)
        # ])
        self.ca = ChannelAttention(hidden_channels)
        self.sa = SpatialAttention()
        self.dynamic_gate = DynamicGateFusion(hidden_channels)
        # self.ema=EMAP(hidden_channels)
        self.fc3 = Conv(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):  # 修改点：统一接口
        assert isinstance(x, (list, tuple)) and len(x) == 2, "CCFF expects a list or tuple of two tensors."
        x1, x2 = x

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        # x1 = self.rep_blocks(x1) + x1
        x1 = self.ca(x1)
        x2 = self.sa(x2)

        fused = self.dynamic_gate(x1, x2)
        fused = self.sa(fused)
        return self.fc3(fused)

class CCFF2(nn.Module):
    def __init__(self, in1_channels, in2_channels, out_channels, n=1, e=0.5):
        super().__init__()
        hidden_channels = int(out_channels * e)

        self.fc1 = Conv(in1_channels, hidden_channels, 1, 1)
        self.fc2 = Conv(in2_channels, hidden_channels, 1, 1)

        self.blocks =C3k(hidden_channels, hidden_channels, shortcut=True,n=n,g=1, e=e)
        self.block2=C3k(hidden_channels, hidden_channels, shortcut=True,n=n,g=1, e=e)
    
        self.ca = ChannelAttention(hidden_channels)
        self.sa = SpatialAttention()
        self.dynamic_gate = DynamicGateFusion(hidden_channels)
        # self.ema=EMAP(hidden_channels)
        self.fc3 = Conv(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):  # 修改点：统一接口
        assert isinstance(x, (list, tuple)) and len(x) == 2, "CCFF expects a list or tuple of two tensors."
        x1, x2 = x

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        x1 = self.blocks(x1) + x1
        x1 = self.ca(x1)
        
        x2 = self.block2(x2) + x2
        x2 = self.sa(x2)

        fused = self.dynamic_gate(x1, x2)
        fused = self.sa(fused)
        return self.fc3(fused)


class CCFF3(nn.Module):
    def __init__(self, in1_channels, in2_channels, out_channels, num_blocks=3, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.fc1 = Conv(in1_channels, hidden_channels, 1, 1)
        self.fc2 = Conv(in2_channels, hidden_channels, 1, 1)

        self.sablocks = Block(hidden_channels)
        self.cablocks = Block(hidden_channels)
        self.ca = ChannelAttention(hidden_channels)
        self.sa = SpatialAttention()
        self.dynamic_gate = DynamicGateFusion(hidden_channels)
        # self.ema=EMAP(hidden_channels)
        self.fc3 = Conv(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):  # 修改点：统一接口
        assert isinstance(x, (list, tuple)) and len(x) == 2, "CCFF expects a list or tuple of two tensors."
        x1, x2 = x

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        x1 = self.cablocks(x1) + x1
        x1 = self.ca(x1)
        x2 = self.sablocks(x2) + x2
        x2 = self.sa(x2)

        fused = self.dynamic_gate(x1, x2)
        fused = self.sa(fused)
        return self.fc3(fused)

class CCFF4(nn.Module):
    def __init__(self, in1_channels, in2_channels, out_channels, num_blocks=3, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.fc1 = Conv(in1_channels, hidden_channels, 1, 1)
        self.fc2 = Conv(in2_channels, hidden_channels, 1, 1)

        self.sablocks = Block(hidden_channels)
        self.cablocks = Block(hidden_channels)
        self.ca = ChannelAttention(hidden_channels)
        self.sa = SpatialAttention()
        self.dynamic_gate = DynamicGateFusion(hidden_channels)
        # self.ema=EMAP(hidden_channels)
        self.fc3 = Conv(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):  # 修改点：统一接口
        assert isinstance(x, (list, tuple)) and len(x) == 2, "CCFF expects a list or tuple of two tensors."
        x1, x2 = x

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        
        x1 = self.ca(x1)
        x2 = self.sa(x2)

        fused = self.dynamic_gate(x1, x2)
        fused = self.cablocks(fused) + fused
        fused = self.cablocks(fused) + fused
        fused = self.sa(fused)
        return self.fc3(fused)
    
class SPPF11(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2 # hidden channels
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv_ = Conv(c_, c_, 1, 1)
        
        self.cvb1= nn.Sequential(
            DWConv(c_,c_,k=(1,4),s=1,d=4),
            DWConv(c_,c_,k=(4,1),s=1,d=4),
            Conv(c_,c_,1,1)
        )
        self.cvb2= nn.Sequential(
            DWConv(c_,c_,k=(1,5),s=1,d=5),
            DWConv(c_,c_,k=(5,1),s=1,d=5),
            Conv(c_,c_,1,1)
        )
        self.cvb3= nn.Sequential(
            DWConv(c_,c_,k=(1,6),s=1,d=6),
            DWConv(c_,c_,k=(6,1),s=1,d=6),
            Conv(c_,c_,1,1)
        )
        self.cv2=nn.Sequential(
            Conv(5 * c_, c_, 1, 1),
            Conv(c_, c_, 3, 1)
            )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = Conv(c_ * 4, c2, 1, 1)       

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x=self.cv1(x)
        y = [self.cv2(torch.cat([self.cv_(x),self.cvb1(x),self.cvb2(x),self.cvb3(x),x],1))]
        y.extend(self.m(y[-1]) for _ in range(3))
        
        return self.cv3(torch.cat(y, 1))

class SPPF12(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2 # hidden channels
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv_ = Conv(c_, c_, 1, 1)
        
        self.cvb1= nn.Sequential(
            DWConv(c_,c_,k=(1,4),s=1,d=4),
            DWConv(c_,c_,k=(4,1),s=1,d=4),
            Conv(c_,c_,1,1)
        )
        self.cvb2= nn.Sequential(
            DWConv(c_,c_,k=(1,5),s=1,d=5),
            DWConv(c_,c_,k=(5,1),s=1,d=5),
            Conv(c_,c_,1,1)
        )
        self.cvb3= nn.Sequential(
            DWConv(c_,c_,k=(1,6),s=1,d=6),
            DWConv(c_,c_,k=(6,1),s=1,d=6),
            Conv(c_,c_,1,1)
        )
        self.cv2=nn.Sequential(
            Conv(4 * c_, c_, 1, 1),
            Conv(c_, c_, 3, 1)
            )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = Conv(c_ * 4, c2, 1, 1)       

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x=self.cv1(x)
        y = [self.cv2(torch.cat([self.cv_(x)+x,self.cvb1(x)+x,self.cvb2(x)+x,self.cvb3(x)+x],1))]
        y.extend(self.m(y[-1]) for _ in range(3))
        
        return self.cv3(torch.cat(y, 1))
class SPPF13(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2 # hidden channels
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv_ = Conv(c_, c_, 1, 1)
        
        self.cvb1= nn.Sequential(
            DWConv(c_,c_,k=(1,4),s=1,d=4),
            DWConv(c_,c_,k=(4,1),s=1,d=4),
            Conv(c_,c_,1,1)
        )
        self.cvb2= nn.Sequential(
            DWConv(c_,c_,k=(1,5),s=1,d=5),
            DWConv(c_,c_,k=(5,1),s=1,d=5),
            Conv(c_,c_,1,1)
        )
        self.cvb3= nn.Sequential(
            DWConv(c_,c_,k=(1,6),s=1,d=6),
            DWConv(c_,c_,k=(6,1),s=1,d=6),
            Conv(c_,c_,1,1)
        )
        self.cv2=nn.Sequential(
            Conv(4 * c_, 2*c_, 1, 1),
            Conv(2*c_, c_, 3, 1)
            )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = Conv(c_ * 4, c2, 1, 1)       

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x=self.cv1(x)
        y = [self.cv2(torch.cat([self.cv_(x)+x,self.cvb1(x)+x,self.cvb2(x)+x,self.cvb3(x)+x],1))]
        y.extend(self.m(y[-1]) for _ in range(3))
        
        return self.cv3(torch.cat(y, 1))
    
class SPPF14(nn.Module):
    def __init__(self, c1, c2,k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        
        self.cv1 = Conv(c1, c_, 1, 1)
        
        self.cvb0= nn.Sequential(
            DWConv(c_,c_,k=(1,3),s=1,d=3),
            DWConv(c_,c_,k=(3,1),s=1,d=3),
            Conv(c_,c_,1,1)
        )
        self.cvb1= nn.Sequential(
            DWConv(c_,c_,k=(1,4),s=1,d=4),
            DWConv(c_,c_,k=(4,1),s=1,d=4),
            Conv(c_,c_,1,1)
        )
        self.cvb2= nn.Sequential(
            DWConv(c_,c_,k=(1,5),s=1,d=5),
            DWConv(c_,c_,k=(5,1),s=1,d=5),
            Conv(c_,c_,1,1)
        )
        self.cvb3= nn.Sequential(
            DWConv(c_,c_,k=(1,6),s=1,d=6),
            DWConv(c_,c_,k=(6,1),s=1,d=6),
            Conv(c_,c_,1,1)
        )
        self.cv_=Conv(c_,c_,1,1)
        # self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * 6, c2, 1, 1)


    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x=self.cv1(x)
        y = [self.cvb0(x)+x,self.cvb1(x)+x,self.cvb2(x)+x,self.cvb3(x)+x,self.cv_(x)+x,x]
        # y.extend(self.m(y[-1]) for _ in range(3))
       
        return self.cv2(torch.cat(y, 1))
class SPPF14(nn.Module):
    def __init__(self, c1, c2,k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        
        self.cv1 = Conv(c1, c_, 1, 1)
        
        self.cvb0= nn.Sequential(
            DWConv(c_,c_,k=(1,3),s=1,d=3),
            DWConv(c_,c_,k=(3,1),s=1,d=3),
            Conv(c_,c_,1,1)
        )
        self.cvb1= nn.Sequential(
            DWConv(c_,c_,k=(1,4),s=1,d=4),
            DWConv(c_,c_,k=(4,1),s=1,d=4),
            Conv(c_,c_,1,1)
        )
        self.cvb2= nn.Sequential(
            DWConv(c_,c_,k=(1,5),s=1,d=5),
            DWConv(c_,c_,k=(5,1),s=1,d=5),
            Conv(c_,c_,1,1)
        )
        self.cvb3= nn.Sequential(
            DWConv(c_,c_,k=(1,6),s=1,d=6),
            DWConv(c_,c_,k=(6,1),s=1,d=6),
            Conv(c_,c_,1,1)
        )
        self.cv_=Conv(c_,c_,1,1)
        # self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * 6, c2, 1, 1)


    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x=self.cv1(x)
        y = [self.cvb0(x)+x,self.cvb1(x)+x,self.cvb2(x)+x,self.cvb3(x)+x,self.cv_(x)+x,x]
        # y.extend(self.m(y[-1]) for _ in range(3))
       
        return self.cv2(torch.cat(y, 1))
    
class PConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s=1):
        super().__init__()
        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
    
class SPPF15(nn.Module):
    def __init__(self, c1, c2,k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        
        self.cv1 = Conv(c1, c_, 1, 1)
        
        self.cvb1= PConv(c_,c_,k=3)
        self.cvb2= PConv(c_,c_,k=4)
        self.cvb3= PConv(c_,c_,k=5)
        self.cv_=Conv(c_,c_,1,1)
        # self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * 5, c2, 1, 1)


    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x=self.cv1(x)
        y = [self.cvb1(x)+x,self.cvb2(x)+x,self.cvb3(x)+x,self.cv_(x)+x,x]
        # y.extend(self.m(y[-1]) for _ in range(3))
       
        return self.cv2(torch.cat(y, 1))


class SPPF16(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1)

        self.cvb1 = PConv(c_, c_, k=3)
        self.cvb2 = PConv(c_, c_, k=5)
        self.cvb3 = PConv(c_, c_, k=7)  # use dilation
        self.cv_ = Conv(c_, c_, 1, 1)
        self.cv2 = Conv(c_ * 5, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        y = [self.cvb1(x)+x, self.cvb2(x) + x, self.cvb3(x) + x, self.cv_(x) + x, x]
        return self.cv2(torch.cat(y, 1))

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        # DWConv: depthwise convolution, kernel=7, groups=dim
        self.dwconv = Conv(dim, dim, k=7, g=dim)

        # Linear projection to higher dim: without BN
        self.f1 = Conv(dim, mlp_ratio * dim, act=False)  # act=False => Identity
        self.f2 = Conv(dim, mlp_ratio * dim, act=False)

        # Projection back to dim, with BN
        self.g = Conv(mlp_ratio * dim, dim)

        # DWConv again, without BN (act defaults to SiLU, we disable it)
        self.dwconv2 = Conv(dim, dim, k=7, g=dim, act=False)

        # Activation
        self.act = nn.ReLU6()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x)) + input
        return x

from typing import Optional    
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,      # 输入通道数
            out_channels: int,     # 输出通道数
            kernel_size: int,      # 卷积核大小
            stride: int = 1,       # 步长
            padding: int = 0,      # 填充
            groups: int = 1,       # 组卷积数
            norm_cfg: Optional[dict] = None,  # 归一化配置
            act_cfg: Optional[dict] = None):  # 激活函数配置
        super().__init__()
        layers = []
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # 归一化层
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # 激活层
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # 将所有层组合为一个序列层
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    # 定义获取归一化层的辅助函数
    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # 若需要其他归一化类型可以在此处添加
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    # 定义获取激活层的辅助函数
    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # 若需要其他激活类型可以在此处添加
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

# 定义上下文锚点注意力 (Context Anchor Attention) 模块
class CAA(nn.Module):
    """上下文锚点注意力模块"""
    def __init__(
            self,
            channels: int,                     # 输入通道数
            h_kernel_size: int = 11,           # 水平卷积核大小
            v_kernel_size: int = 11,           # 垂直卷积核大小
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置
            act_cfg: Optional[dict] = dict(type='SiLU')):                         # 激活函数配置
        super().__init__()
        # 平均池化层
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        # 1x1卷积模块，用于调整通道数
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 水平卷积模块，使用1xh_kernel_size的卷积核，仅在水平方向上进行卷积
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        # 垂直卷积模块，使用v_kernel_sizex1的卷积核，仅在垂直方向上进行卷积
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        # 1x1卷积模块，用于进一步调整通道数
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 使用Sigmoid激活函数
        self.act = nn.Sigmoid()

    # 前向传播函数
    def forward(self, x):
        # 通过平均池化、卷积和激活函数计算注意力系数
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        # x与生成的注意力系数相乘，生成增强后特征图
        x = x*attn_factor
        return x


