from .common_utils_mbyolo import *
from .conv import SpatialAttention
import torch.nn.functional as F
from .conv import Conv,DWConv
from .block import Bottleneck
import torch
import torch.nn.functional as F
import numpy as np

__all__ = (
    "UVSSB",
    "UVSSB2"
)

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


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out




class UEMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(UEMA, self).__init__()
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
        x1 = channel_shuffle(x1,groups=4)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.fc1(self.agp(x1)).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.fc2(self.agp(x2)).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out=self.spatt((group_x * weights.sigmoid()).reshape(b, c, h, w))
        return out


    
class LFEM(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0): 
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=5, padding=5 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act2=act_layer()
        self.fc3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm3 = nn.BatchNorm2d(hidden_features)
        self.fc4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act4=nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        
        x1 = self.norm(self.fc1(x))       
        x2 = self.act2(self.fc2(x1))        
        x3 = self.norm3(self.fc3(x2))        
        x4 = self.act4(self.fc4(x3))
        x =self.drop(x4) +x
        return x


class UVSSB(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            out_channels: int = 0,
            e : float = 0.5,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_ratio=2.0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
   
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        hidden_dim=int(out_channels*e)

        # proj
        self.proj_conv =Conv(in_channels,2*hidden_dim,1,s=1)
        
        #conv_branch
        self.att=UEMA(hidden_dim)
        self.conv_branch1=Conv(hidden_dim,hidden_dim,3,s=1)
           
        self.conv_branch2=Conv(hidden_dim//2,hidden_dim//2,3,s=1)
        
        
        #ssm_branch
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(d_model=hidden_dim,initialize=ssm_init,forward_type=forward_type,)

        self.drop_path = DropPath(drop_path)
        self.lfem = LFEM(hidden_dim, hidden_dim)
        self.final_conv=Conv(4*hidden_dim,out_channels,1,s=1)


    def forward(self, input: torch.Tensor):
        x = self.proj_conv(input)
        x1,x2=x.chunk(2,1)
        
        #ssm_branch
        x3=x1+self.drop_path(self.op(self.norm(self.lfem(x1))))
        # conv branch
        x4 = self.conv_branch1(self.att(x2))
        x4,x5=x4.chunk(2,1)
        x4 = self.conv_branch2(x4)
        
        #concat
        x=torch.cat([x1,x2,x3,x4,x5],1)
        
        return self.final_conv(x)

class UVSSB2(nn.Module):
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6)):
        
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 3 * self.c, 1, 1)
        self.cv_final = Conv((3 + n) * self.c, c2, 1)
        # self.att=UEMA(self.c)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        self.ssm=nn.Sequential(
            LFEM(self.c, self.c),
            norm_layer(self.c),
            SS2D(d_model=self.c),
        )


    def forward(self, x):
        y = self.cv_first(x)
        
        y1,y2, y3 = y.chunk(3, 1)
        y1=self.ssm(y1)
        y = list((y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))