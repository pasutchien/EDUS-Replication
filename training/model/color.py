"""Foreground color/appearance branch: 2D image-based color retrieval from nearby
reference views, combined with the 3D volume feature to produce view-dependent RGB.

Faithful (plain-PyTorch) port of neuralpoint_fields.py's mlp_base / view_head /
color_mlp (tcnn.Network -> nn.Sequential, same shapes) and nerfstudio's
NeRFEncoding (standard NeRF frequency positional encoding). Appearance embedding
is omitted (the release's `else` branch -- use_individual/use_average_appearance
_embedding both off) to keep scope down; can be added later if needed.

The color-retrieval step itself isn't in the released source we have (only
compiled bytecode for the datamanager that would show the exact scheme survived
-- see [[edus-reproduction-project]]); implemented here as the standard IBR-style
scheme this family of methods uses (IBRNet/PixelNeRF/GeoNeRF): project each 3D
sample point directly into each reference camera via its known pose (no source
depth needed), bilinearly sample RGB there. Reference images include BOTH eyes
(left+right) -- confirmed the release's own neighbor search draws from whatever
poses are in the training pool with no eye filtering.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_to_image(points_world: torch.Tensor, cam_pose_c2w: torch.Tensor,
                      fx: float, fy: float, cx: float, cy: float, H: int, W: int):
    """points_world: (N,3) world points. cam_pose_c2w: (4,4), OpenGL convention
    (camera looks down -Z, +Y up) -- same convention as transforms.json/dataset.py.
    Returns (u_norm, v_norm) in [-1,1] for F.grid_sample and a validity mask
    (strictly in front of the camera and inside the image)."""
    R = cam_pose_c2w[:3, :3]
    t = cam_pose_c2w[:3, 3]
    p_local = (points_world - t) @ R  # world -> camera-local; R orthonormal so R^-1 = R^T
    x, y, z = p_local[..., 0], p_local[..., 1], p_local[..., 2]

    valid = z < 0  # in front of the camera (OpenGL convention: looks down -Z)
    z_safe = torch.where(valid, z, -torch.ones_like(z))  # avoid div-by-zero for invalid points

    u = cx - fx * x / z_safe
    v = cy + fy * y / z_safe
    valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # pixel (u,v) is already a "center-at-integer" coordinate (standard calibration
    # convention), matching align_corners=False's own convention directly.
    u_norm = (2 * u + 1) / W - 1
    v_norm = (2 * v + 1) / H - 1
    return u_norm, v_norm, valid


def retrieve_reference_colors(points_world: torch.Tensor, ref_images: torch.Tensor,
                               ref_poses: torch.Tensor, fx: float, fy: float,
                               cx: float, cy: float) -> torch.Tensor:
    """points_world: (n_rays, n_samples, 3). ref_images: (K,3,H,W) in [0,1].
    ref_poses: (K,4,4). Returns (n_rays, n_samples, 3*K): RGB retrieved from each
    of the K reference images (concatenated), zeroed out where invalid."""
    n_rays, n_samples, _ = points_world.shape
    K, _, H, W = ref_images.shape
    pts_flat = points_world.reshape(-1, 3)

    colors = []
    for k in range(K):
        u_norm, v_norm, valid = project_to_image(pts_flat, ref_poses[k], fx, fy, cx, cy, H, W)
        grid = torch.stack([u_norm, v_norm], dim=-1)[None, None, :, :]  # (1,1,N,2)
        sampled = F.grid_sample(ref_images[k:k + 1], grid, mode="bilinear",
                                 align_corners=False, padding_mode="zeros")  # (1,3,1,N)
        sampled = sampled[0, :, 0, :].permute(1, 0)  # (N,3)
        sampled = sampled * valid[:, None]
        colors.append(sampled)

    out = torch.cat(colors, dim=-1)  # (N, 3*K)
    return out.reshape(n_rays, n_samples, 3 * K)


class NeRFEncoding(nn.Module):
    """Standard NeRF frequency positional encoding (nerfstudio convention)."""

    def __init__(self, in_dim: int = 3, num_frequencies: int = 10,
                 min_freq_exp: float = 0.0, max_freq_exp: float = 9.0, include_input: bool = True):
        super().__init__()
        self.include_input = include_input
        self.register_buffer("freqs", 2.0 ** torch.linspace(min_freq_exp, max_freq_exp, num_frequencies))
        self.out_dim = in_dim * num_frequencies * 2 + (in_dim if include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x[..., None] * self.freqs  # (...,in_dim,num_freq)
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (...,in_dim,2*num_freq)
        encoded = encoded.reshape(*x.shape[:-1], -1)
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
        return encoded


class ColorBranch(nn.Module):
    """Color Decoder, matching the paper's Table 4 (App A.2) exactly: a single
    6-layer MLP, with the viewing direction d and a per-image appearance code w
    injected mid-network (after layer 2, before layer 3) rather than at the very
    start or via a separately-sized head:

        LinearRelu0: 16+9+63 -> 128   in: f3D_fg, f2D_fg, gamma(x)
        LinearRelu1: 128 -> 128
        LinearRelu2: 128 -> 128
        LinearRelu3: 128+32+3 -> 128  in: [+ d, w]
        LinearRelu4: 128 -> 64
        LinearRelu5: 64 -> 3          out: c_fg (sigmoid)

    w is a learned per-training-image code (nn.Embedding), following NeRF-in-the-
    Wild: absorbs per-image photometric differences (exposure/white-balance drift
    across KITTI-360 frames) so the underlying 3D field doesn't have to explain
    them. See [[edus-reproduction-project]]."""

    def __init__(self, feature_dim_in: int = 16, num_neighbour_select: int = 3,
                 num_frequencies: int = 10, appearance_dim: int = 32,
                 num_images: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.position_encoding = NeRFEncoding(in_dim=3, num_frequencies=num_frequencies)
        self.embedding_appearance = nn.Embedding(num_images, appearance_dim)

        in0 = feature_dim_in + 3 * num_neighbour_select + self.position_encoding.out_dim  # 16+9+63
        self.layer0 = nn.Sequential(nn.Linear(in0, hidden_dim), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        in3 = hidden_dim + appearance_dim + 3  # 128+32+3
        self.layer3 = nn.Sequential(nn.Linear(in3, hidden_dim), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(64, 3), nn.Sigmoid())

    def forward(self, feature: torch.Tensor, positions: torch.Tensor, ref_colors: torch.Tensor,
                directions: torch.Tensor, appearance_idx: torch.Tensor | None = None,
                appearance_embedding: torch.Tensor | None = None) -> torch.Tensor:
        """feature: (...,16). positions: (...,3). ref_colors: (...,3*K).
        directions: (...,3) normalized ray directions. appearance_idx: scalar or
        (...,) long tensor -- looked up via the trained per-image embedding
        table. appearance_embedding: an already-resolved (...,appearance_dim)
        (or (1,appearance_dim)) code to use directly instead -- for scenes with
        no trained appearance_idx of their own (App A.2's average-embedding
        case, see validate.py). Exactly one of the two must be given.
        Returns (...,3) RGB in [0,1]."""
        pe = self.position_encoding(positions)
        h = torch.cat([feature, ref_colors, pe], dim=-1)
        h = self.layer0(h)
        h = self.layer1(h)
        h = self.layer2(h)

        assert (appearance_idx is None) != (appearance_embedding is None), \
            "pass exactly one of appearance_idx / appearance_embedding"
        w = self.embedding_appearance(appearance_idx) if appearance_embedding is None else appearance_embedding
        if w.dim() < h.dim():
            w = w.expand(*h.shape[:-1], -1)
        h = torch.cat([h, directions, w], dim=-1)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.layer5(h)
