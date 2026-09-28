"""Pose normalization and transforms.json construction for one window.

Frames in play (see plan file):
  * GT world            - KITTI-360 global frame (cam0_to_world.txt)
  * normalized world    - trajectory-centred, gravity/travel-aligned; what transforms.json stores
  * box-local           - frame-0 left camera + C flip; the voxel grid / AABB frame

Key identity (verified against EDUS_inferdata):
    inv(bbx2w) @ transform_matrix[i]  ==  C @ inv(G0) @ Gi @ C
so the box-local geometry is independent of the global-frame choice.
"""
from __future__ import annotations
import numpy as np

from .kitti360 import C_FLIP, cam1_from_cam0, load_cam0_to_world, load_perspective


def build_inv_pose(cam0_to_world: list[np.ndarray]) -> np.ndarray:
    """GT-world -> normalized-world rigid transform.

    Exactly the raw pose of the window's temporal middle frame, inverted:
    inv_pose = inv(cam0_to_world[N // 2]). Verified bit-exact (<1e-13) against
    all 5 released EDUS_inferdata scenes -- not an approximation.
    """
    mid = cam0_to_world[len(cam0_to_world) // 2]
    return np.linalg.inv(mid)


def window_poses(drive: int | str, fids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (inv_pose, per-frame normalized cam0->world, per-frame normalized cam1->world)."""
    g0 = load_cam0_to_world(drive)
    G0 = [g0[f] for f in fids]
    G1 = [cam1_from_cam0(g0[f]) for f in fids]
    inv_pose = build_inv_pose(G0)
    N0 = np.array([inv_pose @ G @ C_FLIP for G in G0])
    N1 = np.array([inv_pose @ G @ C_FLIP for G in G1])
    return inv_pose, N0, N1


def make_transforms(drive: int | str, fids: list[int]) -> dict:
    """Build the transforms.json dict (frames interleaved L,R,L,R,...)."""
    cal = load_perspective()
    inv_pose, N0, N1 = window_poses(drive, fids)
    frames = []
    for k, fid in enumerate(fids):
        frames.append({"file_path": f"{fid}_00.png", "transform_matrix": N0[k].tolist()})
        frames.append({"file_path": f"{fid}_01.png", "transform_matrix": N1[k].tolist()})
    return {
        "fl_x": cal["fx"], "fl_y": cal["fy"], "cx": cal["cx"], "cy": cal["cy"],
        "w": cal["width"], "h": cal["height"],
        "scale": 1, "use_bbx": False, "aabb_scale": 16,
        "bbx2w": N0[0].tolist(),          # box anchored at frame-0 left camera
        "inv_pose": inv_pose.tolist(),    # GT world -> normalized world
        "frames": frames,
    }


def pose_in_box(G0_first: np.ndarray, G_i: np.ndarray) -> np.ndarray:
    """box-local cam->world for a frame: inv(G0) @ Gi  (frame-0 left cam = identity).

    No C_FLIP: confirmed via interactive (Open3D) inspection of the accumulated
    point cloud that this matches real scene geometry, while the C-sandwiched
    version (C @ inv(G0) @ Gi @ C) did not, despite that version's Z-distribution
    numerically matching the released EDUS_inferdata point cloud. That match is
    still unexplained -- open question, deferred.
    """

    # C_FLIP @ np.linalg.inv(G0_first) @ G_i @ C_FLIP
    return np.linalg.inv(G0_first) @ G_i
