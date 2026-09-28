"""Interactive Open3D viewer for a point cloud: either our own .npz (points+colors,
e.g. data_train/<window>/accum/points.npz) or a released .ply (e.g.
EDUS_inferdata/<scene>/<fid>.ply).

Controls: left-drag rotate, right/middle-drag pan, scroll zoom, 'h' for full help.
Draws the foreground AABB as a wireframe box for reference.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

AABB_MIN = np.array([-12.8, -9.0, -20.0])
AABB_MAX = np.array([12.8, 3.8, 31.2])


def aabb_lineset(vmin, vmax) -> o3d.geometry.LineSet:
    box = o3d.geometry.AxisAlignedBoundingBox(vmin, vmax)
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
    ls.paint_uniform_color([1, 0, 0])
    return ls


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    if path.suffix == ".npz":
        d = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(d["points"].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((d["colors"] / 255.0).astype(np.float64))
        return pcd
    return o3d.io.read_point_cloud(str(path))  # .ply and other formats Open3D supports natively


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="path to a points.npz (points, colors) or a .ply point cloud")
    ap.add_argument("--axis-size", type=float, default=5.0, help="length of the XYZ axes at the origin")
    args = ap.parse_args()

    pcd = load_point_cloud(args.path)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [pcd, aabb_lineset(AABB_MIN, AABB_MAX), axes],
        window_name=str(args.path),
        width=1280, height=800,
    )


if __name__ == "__main__":
    main()
