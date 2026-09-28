"""Offscreen-render a point cloud (.npz or .ply) from a fixed camera pose.

Default: camera at the origin (0,0,0) looking straight down +Z, using this
project's real KITTI-360 perspective intrinsics -- i.e. what a camera literally
sitting at the box-local origin would photograph.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    if path.suffix == ".npz":
        d = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(d["points"].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((d["colors"] / 255.0).astype(np.float64))
        return pcd
    return o3d.io.read_point_cloud(str(path))


def look_at_extrinsic(eye: np.ndarray, forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """World-to-camera extrinsic (OpenCV convention: x right, y down, z forward).

    up_hint is the true up direction (e.g. (0,-1,0) for this project's Y-down
    world convention).
    """
    z = forward / np.linalg.norm(forward)
    x = np.cross(z, up_hint)  # NOT cross(up_hint, z) -- that sign flip mirrors left/right
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_cam_to_world = np.stack([x, y, z], axis=1)  # columns = camera axes in world coords
    R_world_to_cam = R_cam_to_world.T
    t = -R_world_to_cam @ eye
    E = np.eye(4)
    E[:3, :3] = R_world_to_cam
    E[:3, 3] = t
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help=".npz (points,colors) or .ply point cloud to render")
    ap.add_argument("out", type=Path, help="output image path")
    ap.add_argument("--pos", type=float, nargs=3, default=[0, 0, 0], help="camera position")
    ap.add_argument("--forward", type=float, nargs=3, default=[0, 0, 1], help="camera look direction")
    ap.add_argument("--up", type=float, nargs=3, default=[0, -1, 0], help="true up direction (default: (0,-1,0), this project's Y-down convention)")
    ap.add_argument("--point-size", type=float, default=2.0)
    args = ap.parse_args()

    from preprocess.kitti360 import load_perspective  # noqa: E402 (needs ROOT on sys.path)
    cal = load_perspective()
    w, h = cal["width"], cal["height"]

    pcd = load_point_cloud(args.path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0, 0, 0])

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, cal["fx"], cal["fy"], cal["cx"], cal["cy"])
    extrinsic = look_at_extrinsic(np.array(args.pos, dtype=float),
                                   np.array(args.forward, dtype=float),
                                   np.array(args.up, dtype=float))
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic
    params.extrinsic = extrinsic
    vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(args.out), do_render=True)
    vis.destroy_window()
    print("wrote", args.out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
