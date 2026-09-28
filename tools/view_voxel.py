"""Interactive Open3D viewer for a voxel volume .npy (shape (128,64,256,3), RGB in [0,1]).

Renders each occupied cell as a colored cube at its true world position.
Controls: left-drag rotate, right/middle-drag pan, scroll zoom, 'h' for full help.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

AABB_MIN = np.array([-12.8, -9.0, -20.0])
AABB_MAX = np.array([12.8, 3.8, 31.2])
VOXEL_SIZE = 0.2


def aabb_lineset(vmin, vmax) -> o3d.geometry.LineSet:
    box = o3d.geometry.AxisAlignedBoundingBox(vmin, vmax)
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
    ls.paint_uniform_color([1, 0, 0])
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy", type=Path, help="path to a voxel volume .npy (128,64,256,3)")
    ap.add_argument("--axis-size", type=float, default=5.0, help="length of the XYZ axes at the origin")
    args = ap.parse_args()

    vol = np.load(args.npy)
    occ = np.argwhere(np.any(vol != 0, axis=-1))  # (N,3) grid indices
    colors = vol[occ[:, 0], occ[:, 1], occ[:, 2]]
    total = vol.shape[0] * vol.shape[1] * vol.shape[2]
    print(f"{len(occ)} occupied cells / {total} ({100 * len(occ) / total:.2f}%)")

    grid = o3d.geometry.VoxelGrid()
    grid.voxel_size = VOXEL_SIZE
    grid.origin = AABB_MIN
    for (ix, iy, iz), c in zip(occ, colors):
        grid.add_voxel(o3d.geometry.Voxel(grid_index=np.array([ix, iy, iz]), color=c))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [grid, aabb_lineset(AABB_MIN, AABB_MAX), axes],
        window_name=str(args.npy),
        width=1280, height=800,
    )


if __name__ == "__main__":
    main()
