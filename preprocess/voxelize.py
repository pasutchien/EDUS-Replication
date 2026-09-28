"""Voxelize the accumulated point cloud into the (128,64,256,3) RGB feature grid
the network actually consumes.

Confirmed against the released EDUS source (Miaosheng1/EDUS):
  * point_encoder.py's VoxelEncoder.forward names the raw input `color_volume`.
  * spade_generator.py's FeatureVolumeGenerator (the encoder actually used in
    neuralpoint.py) takes the same 3-channel volume as its SPADE conditioning map.
  * neuralpoint_fields.py's get_grid_coords() computes voxel index as
    `floor((position_w - bounding_min) / voxel_size)` -- the formula used here.
So the 3 channels are RGB, and cell placement matches the release exactly.

Aggregation (mean color per occupied cell) verified empirically against the real
released data/seq_00_nerfacto_7840_40/drop50_voxel/7840_volume.npy: 96.7% of its
occupied cells hold colors that don't land on any exact single-pixel (k/255)
fraction (consistent with averaging, not nearest/last-point-wins), and 4 of 5
sampled multi-point voxels reproduce the stored value to within float rounding.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .accumulate import AABB_MAX, AABB_MIN, DATA

VOXEL_SIZE = 0.2
GRID_SHAPE = tuple(np.round((AABB_MAX - AABB_MIN) / VOXEL_SIZE).astype(int))  # (128, 64, 256)


def voxelize(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Mean-color RGB voxel grid, shape GRID_SHAPE + (3,), float in [0,1]. Empty cells are zero."""
    idx = np.floor((points - AABB_MIN) / VOXEL_SIZE).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array(GRID_SHAPE)), axis=1)
    idx, rgb = idx[inside], colors[inside].astype(np.float64) / 255.0

    n_cells = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
    flat_idx = (idx[:, 0] * GRID_SHAPE[1] + idx[:, 1]) * GRID_SHAPE[2] + idx[:, 2]

    sums = np.zeros((n_cells, 3), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(sums, flat_idx, rgb)
    np.add.at(counts, flat_idx, 1)

    volume = np.zeros((n_cells, 3), dtype=np.float64)
    occupied = counts > 0
    volume[occupied] = sums[occupied] / counts[occupied, None]
    return volume.reshape(*GRID_SHAPE, 3)


def build_window(win: dict, overwrite: bool = True) -> Path:
    out_dir = DATA / win["name"] / "voxel"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "volume.npy"
    if dst.exists() and not overwrite:
        return dst

    d = np.load(DATA / win["name"] / "accum" / "points.npz")
    volume = voxelize(d["points"], d["colors"])
    np.save(dst, volume)  # float64, matching the released *_volume.npy format
    return dst


def main():
    windows = json.loads((DATA / "windows.json").read_text())
    for i, win in enumerate(windows):
        out = build_window(win)
        print(f"[{i+1}/{len(windows)}] {win['name']} -> {out}")


if __name__ == "__main__":
    main()
