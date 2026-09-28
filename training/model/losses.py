"""Training losses, per Sec 4 "Loss Function" (Eq. 8, 10, 12):

    L_training     = L_rgb + lambda1*L_lidar + lambda2*L_sky + lambda3*L_entropy
    L_fine-tuning  = L_rgb +                    lambda2*L_sky + lambda3*L_entropy
    lambda1=0.1, lambda2=1, lambda3=0.002 (paper's own values, Sec 4 "Training Details")

L_lidar is omitted entirely: LiDAR supervision is deliberately deferred for this
reproduction (see [[edus-reproduction-project]]), so what we actually use is the
paper's own "fine-tuning" formula, even though we're doing generalizable
training rather than per-scene fine-tuning -- there's simply no lambda1 term to
include without LiDAR data.

L_sky (Eq. 10): BCE(1 - alpha^(fg+bg), M_sky)
  alpha^(fg+bg) is the combined foreground+background accumulated opacity
  (render.composite_full's "acc"). Sky pixels should end up with ~zero fg+bg
  opacity (so 1-acc -> 1, matching M_sky=1); everything else should be ~fully
  opaque before reaching the sky term (1-acc -> 0, matching M_sky=0).

L_entropy (Eq. 12): -(a*ln(a) + (1-a)*ln(1-a)), a = alpha^fg
  Uses ONLY the foreground's own accumulated opacity (composite_full's
  "acc_fg"), not fg+bg combined. Standard binary entropy: minimized at a=0 or
  a=1, maximal at a=0.5 -- so minimizing this loss pushes foreground opacity
  toward confidently empty/solid rather than uniformly semi-transparent
  ("StreetSurf" [10], cited as the source of this term).

L_rgb: standard NeRF photometric loss (MSE) -- not given an explicit formula in
the paper text, but this is the universal convention across the NeRF family and
nothing here suggests otherwise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LAMBDA_SKY = 1.0
LAMBDA_ENTROPY = 0.002
# LAMBDA_LIDAR = 0.1  # unused -- LiDAR supervision deferred, see module docstring


def rgb_loss(rgb_pred: torch.Tensor, rgb_gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(rgb_pred, rgb_gt)


def sky_loss(acc_fg_bg: torch.Tensor, sky_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """acc_fg_bg, sky_mask: (N,). Eq. 10: BCE(1 - acc, sky_mask)."""
    pred = (1.0 - acc_fg_bg).clamp(eps, 1 - eps)
    return F.binary_cross_entropy(pred, sky_mask)


def entropy_loss(acc_fg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """acc_fg: (N,), the FOREGROUND-ONLY accumulated opacity. Eq. 12."""
    a = acc_fg.clamp(eps, 1 - eps)
    return (-(a * torch.log(a) + (1 - a) * torch.log(1 - a))).mean()


def total_loss(rgb_pred: torch.Tensor, rgb_gt: torch.Tensor, acc_fg_bg: torch.Tensor,
                sky_mask: torch.Tensor, acc_fg: torch.Tensor,
                lambda_sky: float = LAMBDA_SKY, lambda_entropy: float = LAMBDA_ENTROPY) -> dict:
    """Returns a dict with the combined "total" loss (for backward()) plus each
    individual term (unweighted, for logging)."""
    l_rgb = rgb_loss(rgb_pred, rgb_gt)
    l_sky = sky_loss(acc_fg_bg, sky_mask)
    l_entropy = entropy_loss(acc_fg)
    total = l_rgb + lambda_sky * l_sky + lambda_entropy * l_entropy
    return {"total": total, "rgb": l_rgb, "sky": l_sky, "entropy": l_entropy}
