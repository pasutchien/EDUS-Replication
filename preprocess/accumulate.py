"""Multi-view depth-consistency filtering + box-local point cloud accumulation
for the 80-scene training corpus.

Reference views (the accumulated point cloud fed to the 3D CNN) are every fifth
stereo pair -- App B.3's training recipe: "the accumulated point cloud we use for
training is from every fifth of the stereo pairs."

Evaluation views (the rendering-loss supervision targets) reuse the paper's own
held-out test-frame pattern (App D.1: "the left-eye image of 1, 3, 7, 9 pairs"
within every 10 consecutive pairs), rather than the paper's Drop50 stride-2
choice. Deliberate deviation: Drop50 (0,2,4,...) shares residue 0 mod 10 with the
stride-5 reference set, so frames 0/10/20/30 would be both a reference view and a
supervision target -- an easy near-identity case for the image-based rendering
branch. The odd {1,3,7,9} mod-10 pattern has no overlap with the stride-5
reference set at all (residues {0,5} vs {1,3,7,9}), so every supervised view is a
genuine held-out-relative-to-geometry render. See [[edus-reproduction-project]].

Depth consistency (Sec 3.1 / App B.3): unproject depth D_i of frame i to 3D,
reproject into a nearby frame j (its neighbor within the reference set), and mask
out pixels where the reprojected depth differs from D_j by more than sigma=0.2m.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .kitti360 import load_cam0_to_world, load_perspective
from .normalize import pose_in_box

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_train"

SIGMA_M = 0.2
AABB_MIN = np.array([-12.8, -9.0, -20.0])
AABB_MAX = np.array([12.8, 3.8, 31.2])

REFERENCE_STRIDE = 5           # App B.3 training recipe
EVAL_BLOCK = 10
EVAL_OFFSETS = (1, 3, 7, 9)    # App D.1 test-frame pattern, reused as the training eval set


def reference_indices(n_frames: int) -> list[int]:
    return list(range(0, n_frames, REFERENCE_STRIDE))


def evaluation_indices(n_frames: int) -> list[int]:
    out = []
    for block_start in range(0, n_frames, EVAL_BLOCK):
        out += [block_start + o for o in EVAL_OFFSETS if block_start + o < n_frames]
    return out


def _unproject(depth: np.ndarray, cal: dict) -> np.ndarray:
    """OpenCV pinhole depth map -> (H,W,4) homogeneous camera-space points."""
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = (us - cal["cx"]) / cal["fx"] * depth
    y = (vs - cal["cy"]) / cal["fy"] * depth
    return np.stack([x, y, depth, np.ones_like(depth)], axis=-1).astype(np.float32)


def _project(points_cam: np.ndarray, cal: dict):
    """(...,4) homogeneous camera-space points -> (u, v, z)."""
    x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cal["fx"] * x / z + cal["cx"]
        v = cal["fy"] * y / z + cal["cy"]
    return u, v, z


def _sample_nearest(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    h, w = img.shape
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros_like(u)
    out[inside] = img[vi[inside], ui[inside]]
    return out, inside


def consistency_mask(depth_i: np.ndarray, Gi: np.ndarray, depth_j: np.ndarray, Gj: np.ndarray,
                      cal: dict) -> np.ndarray:
    """Pixels of frame i whose unprojected depth reprojects consistently into frame j.

    pose_in_box(Gj, Gi) reuses the box-local formula with frame j as the reference,
    which is exactly the cam_i -> cam_j rigid transform (both sides OpenCV pinhole
    convention).
    """
    pc_i = _unproject(depth_i, cal)
    T_i_to_j = pose_in_box(Gj, Gi)
    p_cam_j = pc_i @ T_i_to_j.T
    u, v, z_proj = _project(p_cam_j, cal)
    dj, inside = _sample_nearest(depth_j, u, v)
    return (depth_i > 0) & inside & (dj > 0) & (z_proj > 0) & (np.abs(z_proj - dj) <= SIGMA_M)


def accumulate_window(win: dict, ref_idx: list[int]) -> dict:
    """Filtered box-local points/colors built from the given reference-frame indices
    (positions into win["fids"]), consistency-checked against their neighbors within
    that same set."""
    cal = load_perspective()
    poses = load_cam0_to_world(win["drive"])
    fids = win["fids"]
    base = DATA / win["name"]
    G0 = poses[fids[len(fids) // 2]]  # AABB anchored at the window's middle frame, not frame 0

    depths = {i: np.load(base / "depth" / f"{fids[i]}_00.npy") for i in ref_idx}
    Gs = {i: poses[fids[i]] for i in ref_idx}

    pts_all, cols_all = [], []
    n_valid = n_consistent = 0
    for k, i in enumerate(ref_idx):
        neighbors = ([ref_idx[k - 1]] if k > 0 else []) + ([ref_idx[k + 1]] if k < len(ref_idx) - 1 else [])
        mask = np.zeros(depths[i].shape, dtype=bool)
        for j in neighbors:
            mask |= consistency_mask(depths[i], Gs[i], depths[j], Gs[j], cal)
        n_valid += int((depths[i] > 0).sum())
        n_consistent += int(mask.sum())

        pc_i = _unproject(depths[i], cal)
        T_i_to_box = pose_in_box(G0, Gs[i])
        p_box = (pc_i @ T_i_to_box.T)[..., :3]
        inside_aabb = np.all((p_box >= AABB_MIN) & (p_box <= AABB_MAX), axis=-1)
        keep = mask & inside_aabb

        pts_all.append(p_box[keep])
        rgb = np.array(Image.open(base / f"{fids[i]}_00.png").convert("RGB"))
        cols_all.append(rgb[keep])

    points = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3), np.float32)
    colors = np.concatenate(cols_all, axis=0) if cols_all else np.zeros((0, 3), np.uint8)
    return {
        "points": points.astype(np.float32), "colors": colors.astype(np.uint8),
        "n_frames": len(ref_idx), "n_valid": n_valid, "n_consistent": n_consistent,
        "n_in_aabb": int(points.shape[0]),
    }


def build_window(win: dict, overwrite: bool = True) -> Path:
    out_dir = DATA / win["name"]
    accum_dir = out_dir / "accum"
    accum_dir.mkdir(parents=True, exist_ok=True)
    dst = accum_dir / "points.npz"

    fids = win["fids"]
    ref_idx = reference_indices(len(fids))
    eval_idx = evaluation_indices(len(fids))
    (out_dir / "frames.json").write_text(json.dumps({
        "reference_fids": [fids[i] for i in ref_idx],
        "evaluation_fids": [fids[i] for i in eval_idx],
    }, indent=1))

    if dst.exists() and not overwrite:
        return dst
    result = accumulate_window(win, ref_idx)
    np.savez(dst, points=result["points"], colors=result["colors"])
    return dst


def main():
    windows = json.loads((DATA / "windows.json").read_text())
    for i, win in enumerate(windows):
        out = build_window(win)
        print(f"[{i+1}/{len(windows)}] {win['name']} -> {out}")


if __name__ == "__main__":
    main()
