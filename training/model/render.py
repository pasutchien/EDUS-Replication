"""Volume rendering: composite per-sample (density, color) pairs along a ray into
one final pixel color, plus accumulated opacity and expected depth.

Standard NeRF alpha-compositing. Reuses sampling.compute_weights so the weight
formula lives in exactly one place (it's also needed standalone during
hierarchical importance resampling, see sampling.py).

Written generically over an arbitrary (t_vals, density, color) triple -- not
hardcoded to the foreground samples -- so once the background field exists, the
same function renders the combined (foreground + background) sample set with no
changes: concatenate and sort both sample sets' t_vals/density/color first, then
call this once on the merged set.
"""
from __future__ import annotations

import torch

from .sampling import compute_weights


def render_rays(t_vals: torch.Tensor, density: torch.Tensor, color: torch.Tensor) -> dict:
    """t_vals, density: (N,S). color: (N,S,3). Returns:
      rgb:     (N,3) composited pixel color
      acc:     (N,)  accumulated opacity -- 0 = ray saw nothing, 1 = fully opaque
               before reaching the end of the sampled range (e.g. sky beyond
               background samples would account for the missing 1-acc)
      depth:   (N,)  expected depth (weight-weighted mean of t_vals)
      weights: (N,S) per-sample contribution -- also what App's entropy
               regularization (penalizing semi-transparent reconstructions) will
               act on once losses are built
    """
    weights = compute_weights(t_vals, density)           # (N,S)
    rgb = (weights[..., None] * color).sum(dim=-2)        # (N,3)
    acc = weights.sum(dim=-1)                              # (N,)
    depth = (weights * t_vals).sum(dim=-1)                 # (N,)
    return {"rgb": rgb, "acc": acc, "depth": depth, "weights": weights}


def composite_full(fg_t: torch.Tensor, fg_density: torch.Tensor, fg_color: torch.Tensor,
                    bg_t: torch.Tensor, bg_density: torch.Tensor, bg_color: torch.Tensor,
                    sky_color: torch.Tensor) -> dict:
    """Combine foreground + background samples into one ray, render them
    together, then composite the sky color as the backdrop for whatever
    fraction of the ray wasn't accounted for by foreground/background
    (1 - acc) -- standard "composite over" compositing, since sky has no finite
    depth of its own to be sorted into the sample sequence.

    fg_t, fg_density: (N,S_fg). fg_color: (N,S_fg,3).
    bg_t, bg_density: (N,S_bg). bg_color: (N,S_bg,3).
    sky_color: (N,3).
    Returns render_rays' dict (from the merged fg+bg samples) plus:
      rgb_fg_bg: (N,3) the fg+bg composited color alone, before adding sky
      rgb:       (N,3) OVERWRITTEN to be the final fg+bg+sky color -- this is
                 what's directly comparable to a real ground-truth pixel
      acc_fg:    (N,)  the FOREGROUND-ONLY accumulated opacity (not fg+bg
                 combined) -- Eq. 12's entropy regularization acts on this
                 specifically, not the combined "acc" that Eq. 10's sky loss uses
    """
    acc_fg = render_rays(fg_t, fg_density, fg_color)["acc"]

    t_vals = torch.cat([fg_t, bg_t], dim=-1)
    density = torch.cat([fg_density, bg_density], dim=-1)
    color = torch.cat([fg_color, bg_color], dim=-2)  # samples axis is dim=-2 here, not -1 (color has a trailing RGB dim)

    # fg samples live within the AABB, bg samples start at t_far and extend
    # beyond it, so they're already non-overlapping and in order -- but sort
    # explicitly anyway so this stays correct even if that assumption changes.
    t_vals, sort_idx = torch.sort(t_vals, dim=-1)
    density = torch.gather(density, -1, sort_idx)
    color = torch.gather(color, -2, sort_idx[..., None].expand(*sort_idx.shape, 3))

    out = render_rays(t_vals, density, color)
    out["rgb_fg_bg"] = out["rgb"]
    out["rgb"] = out["rgb_fg_bg"] + (1.0 - out["acc"][:, None]) * sky_color
    out["acc_fg"] = acc_fg
    return out
