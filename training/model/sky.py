"""Sky field: a view-dependent environment map, per the paper's Sky Modelling
section and Table 5 (App A.3).

"Street scenes invariably contain the infinite sky region where rays do not
collide with any physical objects... we omit the positional influence and
represent the sky as a view-dependent environment map... we retrieve the 2D
image feature f2D_sky from reference frames, as we discussed in Sec. 3.1, and
then blend the sky color using a single-layer MLP."

Table 5: LinearRelu0, in=9+3(=12), out=3, "in: f2D_sky, d; out: c_sky" -- a
single linear layer, no hidden layers at all (unlike foreground's 6-layer color
decoder). Retrieval reuses the same project+bilinear-sample scheme as
color.retrieve_reference_colors, but since sky has no finite depth, points are
projected by DIRECTION only (rotate into each reference camera's local frame,
no translation) -- the standard "point at infinity" projection, scale-invariant
in depth by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_direction_to_image(directions_world: torch.Tensor, cam_pose_c2w: torch.Tensor,
                                fx: float, fy: float, cx: float, cy: float, H: int, W: int):
    """directions_world: (N,3), need not be normalized. cam_pose_c2w: (4,4),
    OpenGL convention. Returns (u_norm, v_norm) in [-1,1] and a validity mask
    (direction points into the front hemisphere of the camera). No translation
    involved -- a direction has no position, so only rotation matters, and the
    result is the same regardless of assumed distance (points "at infinity")."""
    R = cam_pose_c2w[:3, :3]
    d_local = directions_world @ R  # world direction -> camera-local direction (see color.py's project_to_image)
    x, y, z = d_local[..., 0], d_local[..., 1], d_local[..., 2]

    valid = z < 0  # OpenGL convention: camera looks down -Z
    z_safe = torch.where(valid, z, -torch.ones_like(z))

    u = cx - fx * x / z_safe
    v = cy + fy * y / z_safe
    valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_norm = (2 * u + 1) / W - 1
    v_norm = (2 * v + 1) / H - 1
    return u_norm, v_norm, valid


def retrieve_sky_colors(directions_world: torch.Tensor, ref_images: torch.Tensor,
                         ref_poses: torch.Tensor, fx: float, fy: float, cx: float, cy: float,
                         ref_sky_masks: torch.Tensor | None = None) -> torch.Tensor:
    """directions_world: (N,3). ref_images: (K,3,H,W) in [0,1]. ref_poses: (K,4,4).
    ref_sky_masks: optional (K,1,H,W), 1=sky -- if given, a retrieval is also
    zeroed out when the reference image's OWN pixel at that projected location
    isn't sky (e.g. a building/tree at that viewing angle in the reference view,
    which parallax can easily put where the target ray sees only sky). Without
    this, retrieve_sky_colors has no way to tell sky pixels from ordinary
    geometry -- it only checks FOV overlap, not content. See
    [[edus-reproduction-project]]. Returns (N, 3*K): RGB retrieved by direction
    from each of the K reference images, zeroed where invalid."""
    K, _, H, W = ref_images.shape
    colors = []
    for k in range(K):
        u_norm, v_norm, valid = project_direction_to_image(directions_world, ref_poses[k], fx, fy, cx, cy, H, W)
        grid = torch.stack([u_norm, v_norm], dim=-1)[None, None, :, :]  # (1,1,N,2)
        sampled = F.grid_sample(ref_images[k:k + 1], grid, mode="bilinear",
                                 align_corners=False, padding_mode="zeros")  # (1,3,1,N)
        sampled = sampled[0, :, 0, :].permute(1, 0)  # (N,3)

        if ref_sky_masks is not None:
            is_sky = F.grid_sample(ref_sky_masks[k:k + 1], grid, mode="bilinear",
                                    align_corners=False, padding_mode="zeros")  # (1,1,1,N)
            is_sky = is_sky[0, 0, 0, :] > 0.5  # (N,)
            valid = valid & is_sky

        sampled = sampled * valid[:, None]
        colors.append(sampled)
    return torch.cat(colors, dim=-1)  # (N, 3*K)


class SkyBranch(nn.Module):
    """Single-layer sky decoder (Table 5): [f2D_sky, d] -> c_sky, sigmoid output."""

    def __init__(self, num_neighbour_select: int = 3):
        super().__init__()
        in_dim = 3 * num_neighbour_select + 3  # f2D_sky (9) + d (3) = 12
        self.layer = nn.Sequential(nn.Linear(in_dim, 3), nn.Sigmoid())

    def forward(self, sky_colors: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """sky_colors: (...,3*K). directions: (...,3) normalized ray directions.
        Returns (...,3) RGB in [0,1]."""
        return self.layer(torch.cat([sky_colors, directions], dim=-1))
