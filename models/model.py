import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath


def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def inv_mag(x):
    fft_ = torch.fft.fft2(x)
    fft_ = torch.fft.ifft2(1 * torch.exp(1j * (fft_.angle())))
    return fft_.real


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class LKABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, linear=False, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(n_feat * 4, n_feat * 2, 3, 1, 1, bias=False))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        hidden_features = int(dim * 3)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        return self.project_out(x)


class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        k = self.kernel_size()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        return k if k % 2 else k + 1

    def forward(self, x):
        x1 = inv_mag(x)
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class FDPA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2_fft = torch.fft.fft2(x2)
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(-2, -1)).abs()
        return out * self.alpha + x * self.beta


class HybridDomainAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias=False)
        self.fpa = FDPA(dim)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.fpa(x)
        s_attn = self.conv(self.pool(self.norm1(out)))
        out = s_attn * out
        out = x + out
        return out + self.ffn(self.norm2(out))


class CSC_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, 1)
        self.dw_13 = nn.Conv2d(dim, dim, (1, ker), padding=(0, pad), groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, (ker, 1), padding=(pad, 0), groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, ker, padding=pad, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
        return self.out_conv(self.act(out))


class UIR_PolyKernel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=36, bias=False):
        super().__init__()
        self.input_embed = nn.Conv2d(in_channels, dim, 1)
        self.encoder_level1 = HybridDomainAttention(dim)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = LKABlock(dim * 2)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = LKABlock(dim * 4)
        self.bottleneck = CSC_Block(dim * 4)

        self.eca_level3 = ECA(dim * 4)
        self.eca_level2 = ECA(dim * 2)
        self.eca_level1 = ECA(dim)

        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = LKABlock(dim * 4)
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = LKABlock(dim * 2)
        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.decoder_level1 = HybridDomainAttention(dim)

        self.final_conv = nn.Conv2d(dim, out_channels, 1)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        inp = self.input_embed(x)
        out_enc_level1 = self.encoder_level1(inp)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        latent = self.bottleneck(out_enc_level3)

        out_enc_level3 = self.eca_level3(out_enc_level3)
        out_enc_level2 = self.eca_level2(out_enc_level2)
        out_enc_level1 = self.eca_level1(out_enc_level1)

        inp_dec_level3 = cat(latent, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = cat(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, out_enc_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return self.norm(self.final_conv(out_dec_level1) + x)


if __name__ == '__main__':
    from thop import profile, clever_format

    t = torch.randn(1, 3, 256, 256).cuda()
    model = UIR_PolyKernel().cuda()
    macs, params = profile(model, inputs=(t,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
