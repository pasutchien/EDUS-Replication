"""Input Volume Random Masking (App B.1): randomly zero out a chunk of the
input voxel volume every training iteration, so the foreground field learns to
complete occluded/missing geometry rather than only ever seeing a fully-formed
accumulated point cloud.

"we randomly select an 8m x 8m x 12m cuboid in each iteration, corresponding to
a volume of size [40 x 40 x 60], and eliminate the point cloud within this
area. Note that this is not applied in feed-forward inference nor per-scene
fine-tuning."

8/0.2=40, 12/0.2=60 -- matches this project's own voxel_size=0.2m exactly (see
preprocess/voxelize.py), confirming the [40,40,60] figure is voxel counts, not
an independent choice. Which physical axis gets which of the two "8m" vs the
one "12m" isn't stated beyond that ordering; applied here directly to our own
(X=128,Y=64,Z=256) grid axis order as the most natural reading. Not verified
against the released training script (unreleased) -- flagged as an assumption,
see [[edus-reproduction-project]].
"""
from __future__ import annotations

import random

import torch

MASK_SIZE = (40, 40, 60)  # (X, Y, Z) voxels


def mask_volume(volume: torch.Tensor, mask_size: tuple[int, int, int] = MASK_SIZE) -> torch.Tensor:
    """volume: (3, X, Y, Z). Returns a COPY with a random mask_size cuboid
    zeroed out -- a fresh random position every call, matching "in each
    iteration" (i.e. call this once per training step, not once ever)."""
    _, X, Y, Z = volume.shape
    mx, my, mz = mask_size
    x0 = random.randint(0, max(X - mx, 0))
    y0 = random.randint(0, max(Y - my, 0))
    z0 = random.randint(0, max(Z - mz, 0))
    masked = volume.clone()
    masked[:, x0:x0 + mx, y0:y0 + my, z0:z0 + mz] = 0.0
    return masked
