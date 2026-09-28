"""Background field: image-based density+color prediction for the unbounded
region beyond the foreground AABB, using scene contraction (mip-NeRF 360 style)
instead of the bounded voxel grid. Matches Table 5 (App A.3), "Background
network architecture":

  LinearRelu0: 63+9 -> 128   in: gamma(contract(x)), f2D_bg
  LinearRelu1,2,3: 128 -> 128
  LinearRelu4: 128 -> 16     out: [sigma_bg(1), fe(15)]
  LinearRelu5: 15+3 -> 64    in: [fe, d]
  LinearRelu6: 64 -> 64
  LinearRelu7: 64 -> 3       out: c_bg (sigmoid)

Same density/color split pattern as the foreground field (position-only trunk
predicts density + a geometric embedding fe, then a late head folds in the view
direction for color) -- but here there's no 3D volume at all, since the
background is unbounded; density/color come purely from positional encoding
(after contraction) plus the same 2D image-based retrieval used elsewhere
(color.retrieve_reference_colors, reused unchanged -- it's generic over which
3D points get projected). Confirmed via the released source
(nerfstudio/fields/background_model.py's own docstring): "adds scene
contraction and image embeddings to instant ngp." See [[edus-reproduction-project]].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import NeRFEncoding


def contract(x: torch.Tensor) -> torch.Tensor:
    """Mip-NeRF 360 / nerfstudio-style scene contraction: points inside the unit
    ball pass through unchanged; points outside get mapped into a ball of radius
    2 (unbounded space -> bounded), giving distant background points a sane,
    bounded input to positional encoding instead of unbounded raw coordinates."""
    mag = x.norm(dim=-1, keepdim=True)
    return torch.where(mag <= 1, x, (2 - 1 / mag) * (x / mag))


class BackgroundBranch(nn.Module):
    def __init__(self, num_neighbour_select: int = 3, num_frequencies: int = 10,
                 hidden_dim: int = 128, geo_embed_dim: int = 15):
        super().__init__()
        self.position_encoding = NeRFEncoding(in_dim=3, num_frequencies=num_frequencies)
        self.geo_embed_dim = geo_embed_dim

        in0 = self.position_encoding.out_dim + 3 * num_neighbour_select  # 63+9
        self.layer0 = nn.Sequential(nn.Linear(in0, hidden_dim), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.layer4 = nn.Linear(hidden_dim, 1 + geo_embed_dim)  # [sigma_bg, fe]; split below, no shared activation

        in5 = geo_embed_dim + 3  # fe + d
        self.layer5 = nn.Sequential(nn.Linear(in5, 64), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.layer7 = nn.Sequential(nn.Linear(64, 3), nn.Sigmoid())

    def forward(self, positions: torch.Tensor, bg_colors: torch.Tensor, directions: torch.Tensor):
        """positions: (...,3) world points (unbounded). bg_colors: (...,3*K)
        retrieved 2D reference colors. directions: (...,3) normalized ray dirs.
        Returns density (...,1) and color (...,3)."""
        x = contract(positions)
        pe = self.position_encoding(x)
        h = torch.cat([pe, bg_colors], dim=-1)
        h = self.layer0(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        out = self.layer4(h)
        sigma_raw, fe = out[..., :1], out[..., 1:]
        density = F.relu(sigma_raw)

        h2 = torch.cat([fe, directions], dim=-1)
        h2 = self.layer5(h2)
        h2 = self.layer6(h2)
        color = self.layer7(h2)
        return density, color
