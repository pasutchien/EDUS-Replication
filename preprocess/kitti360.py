"""KITTI-360 raw-data access: paths, calibration, poses, frame availability.

All raw data lives under raw-data/ :
  raw-data/KITTI-360/data_2d_raw/<drive>/image_0{0,1}/data_rect/<fid>.png
  raw-data/data_poses/<drive>/cam0_to_world.txt          (frame_id + 16 vals, row-major 4x4)
  raw-data/calibration/calibration/perspective.txt
  raw-data/data_2d_semantics{,_image_01}/data_2d_semantics/train/<drive>/image_0{0,1}/semantic/<fid>.png
"""
from __future__ import annotations
import functools
from pathlib import Path
import numpy as np

RAW = Path(__file__).resolve().parent.parent / "raw-data"
DRIVE_FMT = "2013_05_28_drive_{:04d}_sync"

# camera axis flip: KITTI-360 rectified-cam convention (x right, y down, z forward)
# <-> the convention EDUS stores in transforms.json. Verified against EDUS_inferdata.
C_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

SKY_SEMANTIC_ID = 23  # Cityscapes/KITTI-360 labelId for "sky"


def drive_name(drive: int | str) -> str:
    return drive if isinstance(drive, str) else DRIVE_FMT.format(int(drive))


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def load_perspective() -> dict:
    """Parse perspective.txt -> dict with P_rect_0x (3x4), R_rect_0x (3x3), S_rect_0x."""
    txt = (RAW / "calibration" / "calibration" / "perspective.txt").read_text()
    out: dict[str, np.ndarray] = {}
    for line in txt.splitlines():
        if ":" not in line:
            continue
        key, vals = line.split(":", 1)
        try:
            nums = [float(x) for x in vals.split()]
        except ValueError:
            continue
        if not nums:
            continue
        out[key.strip()] = np.array(nums)
    P0 = out["P_rect_00"].reshape(3, 4)
    P1 = out["P_rect_01"].reshape(3, 4)
    return {
        "P_rect_00": P0,
        "P_rect_01": P1,
        "R_rect_00": out["R_rect_00"].reshape(3, 3),
        "R_rect_01": out["R_rect_01"].reshape(3, 3),
        "S_rect_00": out["S_rect_00"],
        "fx": float(P0[0, 0]),
        "fy": float(P0[1, 1]),
        "cx": float(P0[0, 2]),
        "cy": float(P0[1, 2]),
        "width": int(out["S_rect_00"][0]),
        "height": int(out["S_rect_00"][1]),
        # stereo baseline (m): P_rect_01[0,3] = -fx * baseline
        "baseline": float(-P1[0, 3] / P0[0, 0]),
    }


@functools.lru_cache(maxsize=1)
def load_cam_to_pose() -> dict:
    """calib_cam_to_pose.txt -> {image_00: 4x4, image_01: 4x4} (unrectified cam -> IMU/pose)."""
    txt = (RAW / "calibration" / "calibration" / "calib_cam_to_pose.txt").read_text()
    out = {}
    for line in txt.splitlines():
        if not line.strip():
            continue
        key, vals = line.split(":", 1)
        M = np.eye(4)
        M[:3, :4] = np.array([float(x) for x in vals.split()]).reshape(3, 4)
        out[key.strip()] = M
    return out


def cam1_from_cam0(cam0_to_world: np.ndarray) -> np.ndarray:
    """Rectified cam1 -> world, given rectified cam0 -> world.

    Chain (KITTI-360 devkit): rect1 -> pose -> rect0, composed onto cam0_to_world.
    Verified against EDUS_inferdata _01 poses to < 0.4 mm.
    """
    cal = load_perspective()
    c2p = load_cam_to_pose()
    T_r0 = np.eye(4); T_r0[:3, :3] = cal["R_rect_00"]
    T_r1 = np.eye(4); T_r1[:3, :3] = cal["R_rect_01"]
    rect0_from_pose = T_r0 @ np.linalg.inv(c2p["image_00"])
    pose_from_rect1 = c2p["image_01"] @ np.linalg.inv(T_r1)
    return cam0_to_world @ (rect0_from_pose @ pose_from_rect1)


# --------------------------------------------------------------------------- #
# poses
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=16)
def load_cam0_to_world(drive: int | str) -> dict[int, np.ndarray]:
    """{frame_id: 4x4 rectified-cam0 -> world}. Not every frame is present."""
    path = RAW / "data_poses" / drive_name(drive) / "cam0_to_world.txt"
    out: dict[int, np.ndarray] = {}
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) < 17:
            continue
        out[int(v[0])] = np.array(v[1:17], dtype=float).reshape(4, 4)
    return out


# --------------------------------------------------------------------------- #
# frame availability
# --------------------------------------------------------------------------- #
def image_dir(drive: int | str, cam: int) -> Path:
    return RAW / "KITTI-360" / "data_2d_raw" / drive_name(drive) / f"image_{cam:02d}" / "data_rect"


def semantic_dir(drive: int | str, cam: int) -> Path:
    root = "data_2d_semantics" if cam == 0 else "data_2d_semantics_image_01"
    return (RAW / root / "data_2d_semantics" / "train" / drive_name(drive)
            / f"image_{cam:02d}" / "semantic")


@functools.lru_cache(maxsize=16)
def available_frames(drive: int | str) -> list[int]:
    """Frame ids with rectified L+R image, cam0_to_world pose, and L+R semantic label."""
    poses = set(load_cam0_to_world(drive))
    sets = [poses]
    for cam in (0, 1):
        sets.append({int(p.stem) for p in image_dir(drive, cam).glob("*.png")})
        sets.append({int(p.stem) for p in semantic_dir(drive, cam).glob("*.png")})
    return sorted(set.intersection(*map(set, sets)))


def image_path(drive: int | str, cam: int, fid: int) -> Path:
    return image_dir(drive, cam) / f"{fid:010d}.png"


def semantic_path(drive: int | str, cam: int, fid: int) -> Path:
    return semantic_dir(drive, cam) / f"{fid:010d}.png"


if __name__ == "__main__":
    cal = load_perspective()
    print("intrinsics:", {k: cal[k] for k in ("fx", "fy", "cx", "cy", "width", "height", "baseline")})
    for d in (3, 7, 10):
        fr = available_frames(d)
        print(f"drive {d:04d}: {len(fr)} usable frames  [{fr[0]}..{fr[-1]}]")
