"""Foreground field: sample the SPADE feature volume at ray-sample positions and
regress density from it.

Faithful port of the relevant parts of neuralpoint_fields.py's `get_grid_coords`,
`interpolate_features`, and the mlp_density path of `get_density_factor_fields`
from the released EDUS source. `mlp_density` there is a tcnn.Network (tiny-cuda-nn);
reproduced here as a plain nn.Sequential MLP with matching shape (feature_dim_in=16
-> hidden=64 x2 hidden layers -> 1, ReLU activations, no output activation) since
tcnn isn't a portable dependency -- the outer F.relu(...) the release applies to
the MLP's output is kept, so the final nonlinearity is identical either way.

AABB/voxel-size constants match preprocess/accumulate.py and preprocess/voxelize.py
(same normalized-world, middle-frame-anchored convention -- see [[edus-reproduction-project]]).
`self.volume_size` uses plain (X,Y,Z) = (128,64,256), matching how our voxel.npy is
stored (and matching self.voxel.shape[2:] in the release, which is computed BEFORE
the Y,X,Z-labeled rearrange used only for building the generator -- see spade_encoder.py).

Deliberate deviation from the release: get_grid_coords there reorders to
dhw[...,[2,1,0]] = [z_norm, y_norm, x_norm]. Traced FeatureVolumeGenerator's
forward() precisely (two permutes: the outer 'B C W H D -> B C D H W' in
neuralpoint.py, then 'B C D H W -> B C H W D' inside forward()) and its output
layout is verifiably (dim2=Y, dim3=X, dim4=Z) -- but F.grid_sample maps
grid[...,0]->dim4, grid[...,1]->dim3, grid[...,2]->dim2, so the release's own
ordering would feed y_norm to dim3(X-sized) and x_norm to dim2(Y-sized): X and Y
swapped. Verified empirically with a synthetic marker volume (see conversation) --
confirmed the swap is real, not a mistracing on my part, and confirmed
[2,0,1] = [z_norm, x_norm, y_norm] is the self-consistent fix. Since we train our
own weights rather than loading the release's checkpoint here, internal
consistency matters, not matching their exact ordering -- using the fixed order.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

AABB_MIN = torch.tensor([-12.8, -9.0, -20.0])
VOXEL_SIZE = torch.tensor([0.2, 0.2, 0.2])
VOLUME_SIZE = torch.tensor([128, 64, 256])  # plain (X, Y, Z), matches voxel.npy's own layout


def get_grid_coords(position_w: torch.Tensor, bounding_min: torch.Tensor = AABB_MIN,
                     voxel_size: torch.Tensor = VOXEL_SIZE, volume_size: torch.Tensor = VOLUME_SIZE):
    """position_w: (N,3) world points -> (grid_coords (N,3) in [-1,1] for
    F.grid_sample, index (N,3) integer voxel indices)."""
    pts = position_w - bounding_min.to(position_w)
    x_index = pts[..., 0] / voxel_size[0]
    y_index = pts[..., 1] / voxel_size[1]
    z_index = pts[..., 2] / voxel_size[2]

    dhw = torch.stack([x_index, y_index, z_index], dim=1)
    index = dhw.clone().long()

    dhw = dhw.clone()
    dhw[..., 0] = dhw[..., 0] / volume_size[0] * 2 - 1
    dhw[..., 1] = dhw[..., 1] / volume_size[1] * 2 - 1
    dhw[..., 2] = dhw[..., 2] / volume_size[2] * 2 - 1
    grid_coords = dhw[..., [2, 0, 1]]  # [z_norm, x_norm, y_norm] -- see module docstring
    return grid_coords, index


def interpolate_features(grid_coords: torch.Tensor, feature_volume: torch.Tensor) -> torch.Tensor:
    """grid_coords: (N,3) in [-1,1]. feature_volume: (1,C,Dz,Dy,Dx)-shaped as
    produced by FeatureVolumeGenerator. Returns (N,C) trilinearly-interpolated features.

    align_corners=False matches get_grid_coords' "edge-at-integer" index convention
    exactly (integer i = voxel i's near boundary, i+0.5 = its center) -- verified
    empirically that this correctly recovers voxel centers with no residual offset,
    and fades to zero-padding right at the AABB boundary as expected. align_corners=True
    would instead introduce a scale distortion growing to a full voxel of error at
    the far edge of the volume.
    """
    grid_coords = grid_coords[None, None, None, ...]  # (1,1,1,N,3): grid_sample wants (N,Dout,Hout,Wout,3)
    feat = F.grid_sample(feature_volume, grid_coords, mode="bilinear", align_corners=False)
    # feat: (1, C, 1, 1, N) -> (N, C)
    return feat[0, :, 0, 0, :].permute(1, 0)


class DensityMLP(nn.Module):
    """Plain-PyTorch stand-in for the release's tcnn.Network mlp_density:
    feature_dim_in(16) -> hidden(64) -> hidden(64) -> 1, ReLU, no output activation."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 64, n_hidden_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_density(positions: torch.Tensor, feature_volume: torch.Tensor, density_mlp: DensityMLP):
    """positions: (...,3) world points. feature_volume: (1,C,Dz,Dy,Dx).
    Returns (density (...,1), feature (...,C))."""
    shape = positions.shape[:-1]
    positions_flat = positions.reshape(-1, 3)

    grid_coords, index = get_grid_coords(positions_flat)
    feat = interpolate_features(grid_coords, feature_volume)  # (N, C)

    density_flat = density_mlp(feat)  # (N, 1)
    density = F.relu(density_flat).reshape(*shape, 1)
    #density = torch.nan_to_num(density)  # commented out deliberately: let NaN/inf surface
    # and crash loudly during training rather than silently clamp a real underlying bug.
    feat = feat.reshape(*shape, -1)
    return density, feat
