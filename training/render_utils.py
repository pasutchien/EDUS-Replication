"""Full-image rendering (all H*W pixels of one evaluation frame), chunked so
it fits in memory -- unlike training, which only ever renders a 4096-ray
batch. Used for periodic training-progress previews (see train.py's
render_every) and would equally serve standalone eval/inference.
"""
from __future__ import annotations

import torch

from training.dataset import WindowData


@torch.no_grad()
def render_full_image(model, wd: WindowData, img_idx: int, fx: float, fy: float, cx: float, cy: float,
                       device: str, chunk_size: int = 8192) -> torch.Tensor:
    """Returns an (H,W,3) float tensor in [0,1] on CPU. Encodes the window's
    volume once, then reuses that feature volume across ray chunks -- only
    the (cheap) ray-sampling/decoding work is repeated per chunk, not the
    SPADE-CNN encoder forward pass."""
    model.eval()
    batch = wd.full_image_rays(img_idx)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    feats_volume = model.encode(wd.volume.to(device))
    shared = {k: batch[k] for k in ("ref_images", "ref_poses", "ref_sky_masks", "appearance_idx")}

    n_rays = batch["ray_origins"].shape[0]
    rgb_chunks = []
    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)
        chunk_batch = {
            "ray_origins": batch["ray_origins"][start:end],
            "ray_directions": batch["ray_directions"][start:end],
            **shared,
        }
        out = model.render_from_features(feats_volume, chunk_batch, fx, fy, cx, cy, deterministic=True)
        rgb_chunks.append(out["rgb"].cpu())

    model.train()
    rgb_full = torch.cat(rgb_chunks, dim=0)
    return rgb_full.reshape(wd.H, wd.W, 3).clamp(0, 1)
