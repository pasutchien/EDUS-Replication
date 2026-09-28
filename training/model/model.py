"""EdusModel: bundles the foreground encoder + density + color, background, and
sky networks into a single nn.Module (for one optimizer / one checkpoint), plus
a render() method that performs the exact forward pass we've been assembling by
hand throughout this project's tests: hierarchical sampling (80 initial +
3x16 importance rounds, App B.2) -> foreground density/color -> background
density/color -> sky color -> full compositing. Used identically by the
training loop and (later) any eval/inference code.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .background import BackgroundBranch
from .color import ColorBranch, retrieve_reference_colors
from .foreground import DensityMLP, get_grid_coords, interpolate_features
from .render import composite_full
from .sampling import (N_IMPORTANCE_PER_ROUND, N_IMPORTANCE_ROUNDS, compute_weights,
                        ray_aabb_intersection, sample_background, sample_pdf,
                        sample_uniform_and_disparity)
from .sky import SkyBranch, retrieve_sky_colors
from .spade_encoder import FeatureVolumeGenerator, volume_xyz_to_generator_input


class EdusModel(nn.Module):
    def __init__(self, num_images: int, num_neighbour_select: int = 3):
        super().__init__()
        self.encoder = FeatureVolumeGenerator(init_res=8, volume_res=(64, 128, 256),
                                               out_channel=16, input_channels=3, z_dim_oasis=0)
        self.density_mlp = DensityMLP()
        self.color_branch = ColorBranch(num_neighbour_select=num_neighbour_select, num_images=num_images)
        self.bg_branch = BackgroundBranch(num_neighbour_select=num_neighbour_select)
        self.sky_branch = SkyBranch(num_neighbour_select=num_neighbour_select)

    def encode(self, volume: torch.Tensor) -> torch.Tensor:
        """volume: (3,128,64,256) -- already masked if desired (training only,
        see masking.py). Returns the SPADE-CNN feature volume, independent of
        any particular ray batch -- split out from render() so full-image
        rendering (render_utils.py) can run the (comparatively expensive)
        encoder once and reuse it across many ray chunks."""
        gen_input = volume_xyz_to_generator_input(volume[None])
        return self.encoder(gen_input)

    def render(self, volume: torch.Tensor, batch: dict, fx: float, fy: float, cx: float, cy: float,
               deterministic: bool = False) -> dict:
        """volume: (3,128,64,256) -- already masked if desired (training only,
        see masking.py). batch: a dict from WindowData.sample_rays (ray_origins,
        ray_directions, ref_images, ref_poses, ref_sky_masks, appearance_idx).
        deterministic=False (default, training): jittered/random sample
        placement every call. deterministic=True (eval/rendering): fixed bin
        midpoints and evenly-spaced quantiles, for reproducible output -- see
        sampling.py's sample_uniform_and_disparity/sample_pdf docstrings.
        Returns composite_full's dict: rgb, rgb_fg_bg, acc, acc_fg, depth, weights."""
        feats_volume = self.encode(volume)
        return self.render_from_features(feats_volume, batch, fx, fy, cx, cy, deterministic=deterministic)

    def render_from_features(self, feats_volume: torch.Tensor, batch: dict, fx: float, fy: float,
                              cx: float, cy: float, deterministic: bool = False) -> dict:
        """Same as render(), but takes an already-encoded feature volume
        (from encode()) instead of the raw voxel volume -- lets a caller
        encode once and render many ray-batch chunks against it."""
        origins, directions = batch["ray_origins"], batch["ray_directions"]
        unit_dirs = directions / directions.norm(dim=-1, keepdim=True)
        t_near, t_far, hit = ray_aabb_intersection(origins, directions)
        # rays that miss the AABB entirely can come back with t_far < t_near
        # (even negative) -- left unclamped, sample_uniform_and_disparity's
        # disparity formula (1/(near_disp + frac*(far_disp-near_disp))) can
        # cross zero and blow up to +-inf, which later produces NaN once that
        # reaches F.grid_sample's coordinate normalization (inf-inf). Clamping
        # collapses a missed ray to a zero-length foreground interval at
        # t_near instead -- well-defined.
        t_far = torch.maximum(t_far, t_near)
        # That collapsed interval still queries ONE real (but physically
        # meaningless, out-of-box) point, and compute_weights' "pad last
        # sample's distance with 1e10" trick then lets whatever density
        # density_mlp happens to output for that point's (mostly zero, via
        # grid_sample's zero-padding) feature saturate the ray's foreground
        # opacity to ~0 or ~1 -- teaching color_branch to match sky/background
        # pixels at a meaningless location. Zeroing density for non-hit rays
        # (below) makes them contribute exactly zero foreground opacity
        # instead, so background+sky -- which is what's architecturally
        # supposed to handle non-box content -- carry these rays entirely,
        # and no gradient reaches density_mlp/color_branch from them at all.
        hit_mask = hit.to(t_far.dtype)[:, None]  # (N,1), broadcasts over samples
        t_vals = sample_uniform_and_disparity(t_near, t_far, deterministic=deterministic)

        def query_fg(t_vals):
            positions = origins[:, None, :] + directions[:, None, :] * t_vals[..., None]
            grid_coords, _ = get_grid_coords(positions.reshape(-1, 3))
            feat = interpolate_features(grid_coords, feats_volume)
            density = torch.relu(self.density_mlp(feat)).reshape(*t_vals.shape) * hit_mask
            return density, positions, feat.reshape(*t_vals.shape, -1)

        density, positions, feat = query_fg(t_vals)
        for _ in range(N_IMPORTANCE_ROUNDS):
            with torch.no_grad():
                weights = compute_weights(t_vals, density)
                new_t = sample_pdf(t_vals, weights, N_IMPORTANCE_PER_ROUND, deterministic=deterministic)
                t_vals = torch.cat([t_vals, new_t], dim=-1)
                t_vals, _ = torch.sort(t_vals, dim=-1)
            density, positions, feat = query_fg(t_vals)

        n_fg = t_vals.shape[-1]
        ref_colors = retrieve_reference_colors(positions, batch["ref_images"], batch["ref_poses"], fx, fy, cx, cy)
        dirs_fg = unit_dirs[:, None, :].expand(-1, n_fg, -1)
        if "appearance_embedding" in batch:
            # unseen scene at inference (validate.py): no trained per-image
            # code exists, use the given (already-resolved) embedding instead.
            fg_color = self.color_branch(feat, positions, ref_colors, dirs_fg,
                                          appearance_embedding=batch["appearance_embedding"])
        else:
            appearance_idx = batch["appearance_idx"].expand(positions.shape[0], n_fg)
            fg_color = self.color_branch(feat, positions, ref_colors, dirs_fg, appearance_idx=appearance_idx)

        bg_t = sample_background(t_far)
        n_bg = bg_t.shape[-1]
        bg_positions = origins[:, None, :] + directions[:, None, :] * bg_t[..., None]
        bg_ref_colors = retrieve_reference_colors(bg_positions, batch["ref_images"], batch["ref_poses"], fx, fy, cx, cy)
        dirs_bg = unit_dirs[:, None, :].expand(-1, n_bg, -1)
        bg_density, bg_color = self.bg_branch(bg_positions, bg_ref_colors, dirs_bg)
        bg_density = bg_density.squeeze(-1)

        sky_colors = retrieve_sky_colors(unit_dirs, batch["ref_images"], batch["ref_poses"], fx, fy, cx, cy,
                                          ref_sky_masks=batch.get("ref_sky_masks"))
        sky_rgb = self.sky_branch(sky_colors, unit_dirs)

        return composite_full(t_vals, density, fg_color, bg_t, bg_density, bg_color, sky_rgb)
