"""3D SPADE-modulated feature-volume generator -- the foreground field's encoder.

Faithful port of nerfstudio/field_components/spade_generator.py from the released
EDUS source (Miaosheng1/EDUS). One substitution: the StyleGAN2-style fused CUDA
`FusedLeakyReLU` op (nerfstudio.third_party.gsn_ops) is replaced by its well-known
plain-PyTorch equivalent (bias add + leaky_relu(0.2) + scale by sqrt(2)) to avoid
the custom CUDA extension dependency -- identical math, just portable.

Confirmed in [[edus-reproduction-project]]: this (not the plain 3D U-Net in
point_encoder.py, which is the paper's own ablation variant) is the encoder
actually instantiated in neuralpoint.py:
    self.encoder = FeatureVolumeGenerator(init_res=8, volume_res=volume_res,
                                           out_channel=16, input_channels=3,
                                           z_dim_oasis=0)
with `volume_res = (voxel.shape[2], voxel.shape[1], voxel.shape[3])`, explicitly
commented `## YXZ` in the source -- i.e. constructed from a (B,X,Y,Z,C) voxel
array. `z_dim_oasis=0` means no StyleGAN-style noise injection is actually used
in the real model, despite the class defaulting to 64 -- the SPADE conditioning
map is just the raw 3-channel volume, nothing concatenated onto it.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fused_leaky_relu(input: torch.Tensor, bias: torch.Tensor | None = None,
                      negative_slope: float = 0.2, scale: float = 2 ** 0.5) -> torch.Tensor:
    """Pure-PyTorch equivalent of StyleGAN2's fused CUDA op: bias-add, leaky_relu, scale."""
    if bias is not None:
        shape = [1, -1] + [1] * (input.dim() - 2)
        input = input + bias.view(*shape)
    return F.leaky_relu(input, negative_slope=negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel: int, bias: bool = True, negative_slope: float = 0.2, scale: float = 2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate (StyleGAN-style init/scale)."""

    def __init__(self, in_channel, out_channel, bias=True, bias_init=0, lr_mul=1, activate=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init)) if bias else None
        self.activate = activate
        self.scale = (1 / math.sqrt(in_channel)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activate:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out


class EqualConv3d(nn.Module):
    """3D convolution with equalized learning rate (StyleGAN-style init/scale)."""

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 3)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, input):
        return F.conv3d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)


class ConvLayer3d(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=True, activate=True):
        padding = kernel_size // 2
        layers = [EqualConv3d(in_channel, out_channel, kernel_size, padding=padding, stride=1,
                               bias=bias and not activate)]
        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))
        super().__init__(*layers)


class SPADE3D(nn.Module):
    """Spatially-adaptive (de)normalization: instance-normalize, then modulate
    with gamma/beta predicted from the (interpolated) conditioning volume."""

    def __init__(self, norm_nc, label_nc, nhidden=128, norm_type="instance", kernel_size=3):
        super().__init__()
        if norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        elif norm_type == "batch":
            self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
        else:
            raise ValueError(f"unsupported norm_type {norm_type}")

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            EqualConv3d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU(),
        )
        self.mlp_gamma = EqualConv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = EqualConv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode="trilinear")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class ConvResBlock3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super().__init__()
        mid_ch = min(in_channel, out_channel)
        self.conv1 = ConvLayer3d(in_channel, mid_ch, kernel_size=kernel_size, bias=True, activate=True)
        self.conv2 = ConvLayer3d(mid_ch, out_channel, kernel_size=kernel_size, bias=True, activate=True)
        if in_channel != out_channel:
            self.skip = ConvLayer3d(in_channel, out_channel, kernel_size=1, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        if hasattr(self, "skip"):
            out = (out + self.skip(input)) / math.sqrt(2)
        else:
            out = (out + input) / math.sqrt(2)
        return out


class SPADE3DResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, spade_nc, hidden_nc, kernel_size=3):
        super().__init__()
        self.learned_shortcut = in_channel != out_channel
        middle_channel = min(in_channel, out_channel)

        self.conv_0 = ConvLayer3d(in_channel, middle_channel, kernel_size=kernel_size, bias=True, activate=True)
        self.conv_1 = ConvLayer3d(middle_channel, out_channel, kernel_size=kernel_size, bias=True, activate=True)
        self.norm_0 = SPADE3D(in_channel, spade_nc, hidden_nc)
        self.norm_1 = SPADE3D(middle_channel, spade_nc, hidden_nc)
        if self.learned_shortcut:
            self.conv_s = ConvLayer3d(in_channel, out_channel, kernel_size=1, bias=False, activate=False)
            self.norm_s = SPADE3D(in_channel, spade_nc, hidden_nc)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input, seg):
        out = self.conv_0(self.actvn(self.norm_0(input, seg)))
        out = self.conv_1(self.actvn(self.norm_1(out, seg)))
        if self.learned_shortcut:
            skip = self.conv_s(self.norm_s(input, seg))
            out = (out + skip) / math.sqrt(2)
        else:
            out = (out + input) / math.sqrt(2)
        return out


class FeatureVolumeGenerator(nn.Module):
    """SPADE-modulated 3D generator: a learned coarse latent, progressively
    upsampled and modulated by the raw RGB voxel volume at every resolution,
    ending at the full `volume_res` with `out_channel` feature channels."""

    def __init__(self, init_res=8, volume_res=(64, 64, 64), max_channel=512, out_channel=16,
                 input_channels=3, spade_hidden_channel=128, noise_type="oasis",
                 z_dim=256, z_dim_oasis=64, final_unconditional_layer=True, final_tanh=False, **kwargs):
        super().__init__()
        self.h, self.w, self.d = volume_res  # (Y, X, Z) per the release's own "## YXZ" comment
        out_res = min(volume_res)
        self.out_nc = out_channel
        self.max_nc = max_channel
        self.noise_type = noise_type
        self.z_dim = z_dim
        self.final_tanh = final_tanh
        self.final_unconditional_layer = final_unconditional_layer

        res_log2 = int(math.log2(out_res) - math.log2(init_res))
        self.block_num = res_log2
        nc_list = [min(self.out_nc * (2 ** (res_log2 - i)), self.max_nc) for i in range(res_log2)]
        nc_list += [nc_list[-1], out_channel]
        self.nc_list = nc_list
        self.sw, self.sh, self.sd = self.compute_latent_vector_size3D(num_up_layers=self.block_num)

        self.z_dim_oasis = z_dim_oasis if noise_type == "oasis" else 0
        if noise_type == "oasis":
            spade_nc = input_channels + self.z_dim_oasis
            self.fc = EqualConv3d(spade_nc, nc_list[0], 3, padding=1)
        else:
            spade_nc = input_channels
            self.fc = EqualLinear(z_dim, nc_list[0] * self.sw * self.sh * self.sd)

        for i in range(self.block_num):
            self.add_module(f"spade_block{i}",
                             SPADE3DResnetBlock(in_channel=nc_list[i], out_channel=nc_list[i + 1],
                                                 spade_nc=spade_nc, hidden_nc=spade_hidden_channel))

        self.up2x = nn.Upsample(scale_factor=2)
        if final_unconditional_layer:
            self.nospade = ConvResBlock3d(nc_list[-2], nc_list[-2])
        self.out = nn.Sequential(
            nn.LeakyReLU(0.2),
            EqualConv3d(nc_list[-2], nc_list[-1], kernel_size=3, padding=1),
        )

    def compute_latent_vector_size3D(self, num_up_layers):
        sw = self.w // (2 ** num_up_layers)
        sh = round(sw / (self.w / self.h))
        sd = round(sw / (self.w / self.d))
        return sw, sh, sd

    def forward(self, input: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """input: (B, input_channels, D, H, W) with (D,H,W) matching (Z,Y,X) of volume_res."""
        B, C, D, H, W = input.shape
        input = input.permute(0, 1, 3, 4, 2)  # (B,C,D,H,W) -> (B,C,H,W,D)

        if z is None:
            z = torch.randn(B, self.z_dim, dtype=torch.float32, device=input.device)

        if self.noise_type == "oasis":
            if self.z_dim_oasis > 0:
                z = z[:, :self.z_dim_oasis].to(input.device)
                z = z.view(B, self.z_dim_oasis, 1, 1, 1).expand(B, self.z_dim_oasis, *input.shape[2:])
                input = torch.cat((z, input), dim=1)
            x = F.interpolate(input, size=(self.sh, self.sw, self.sd))
            x = self.fc(x)
        else:
            x = self.fc(z)
            x = x.view(-1, self.nc_list[0], self.sh, self.sw, self.sd)

        for i in range(self.block_num):
            block = getattr(self, f"spade_block{i}")
            x = block(x, input)
            x = self.up2x(x)

        if self.final_unconditional_layer:
            x = self.nospade(x)
        x = self.out(x)
        if self.final_tanh:
            x = torch.tanh(x)
        return x


def volume_xyz_to_generator_input(volume: torch.Tensor) -> torch.Tensor:
    """(B, C, X, Y, Z) -> (B, C, Z, Y, X), the input layout FeatureVolumeGenerator.forward
    expects (matches the release's combined effect of two rearranges we traced through
    neuralpoint.py + spade_generator.py -- see module docstring)."""
    return volume.permute(0, 1, 4, 3, 2)
