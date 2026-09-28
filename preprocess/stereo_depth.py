"""Stereo depth for one window -> depth/<fid>_00.npy (left eye, metric, float32).

Deliberate substitution for the paper's MobileStereoNet: that repo targets
Python 3.6 / PyTorch 1.4 / CUDA 10.0 (2021, Google-Drive-hosted weights) which is
impractical against this project's modern stack. OpenCV StereoSGBM needs no extra
deps/weights and produces a noisy-but-usable metric prior -- which is exactly what
EDUS's foreground pipeline is designed to tolerate (see the accumulation step's
depth-consistency filter and the input-volume random masking during training).

Invalid pixels (no disparity match, or masked as sky) are written as depth = 0.0;
downstream code must filter on depth > 0. This differs from the released
EDUS_inferdata depth maps, which are fully dense (a learned net always outputs
something) -- documented, not a bug.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

from .kitti360 import SKY_SEMANTIC_ID, image_path, load_perspective, semantic_path

MAX_DEPTH_M = 100.0
NUM_DISP = 128     # multiple of 16; fx*baseline/NUM_DISP ~= 2.6 m near-depth limit
BLOCK_SIZE = 5


def _make_matcher() -> cv2.StereoSGBM:
    b = BLOCK_SIZE
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=NUM_DISP, blockSize=b,
        P1=8 * b * b, P2=32 * b * b,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def stereo_depth(drive: int, fid: int, matcher: cv2.StereoSGBM | None = None) -> np.ndarray:
    cal = load_perspective()
    matcher = matcher or _make_matcher()
    L = cv2.imread(str(image_path(drive, 0, fid)), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(str(image_path(drive, 1, fid)), cv2.IMREAD_GRAYSCALE)
    disp = matcher.compute(L, R).astype(np.float32) / 16.0

    depth = np.zeros_like(disp)
    valid = disp > 0
    depth[valid] = cal["fx"] * cal["baseline"] / disp[valid]
    depth[depth >= MAX_DEPTH_M] = 0.0

    sem = cv2.imread(str(semantic_path(drive, 0, fid)), cv2.IMREAD_UNCHANGED)
    depth[sem == SKY_SEMANTIC_ID] = 0.0
    return depth


def build_window_depth(win: dict, out_root: Path, overwrite: bool = False) -> Path:
    out = out_root / win["name"] / "depth"
    out.mkdir(parents=True, exist_ok=True)
    matcher = _make_matcher()
    for fid in win["fids"]:
        dst = out / f"{fid}_00.npy"
        if dst.exists() and not overwrite:
            continue
        np.save(dst, stereo_depth(win["drive"], fid, matcher))
    return out


def main():
    import json
    root = Path(__file__).resolve().parent.parent
    windows = json.loads((root / "data_train" / "windows.json").read_text())
    for i, win in enumerate(windows):
        out = build_window_depth(win, root / "data_train")
        print(f"[{i+1}/{len(windows)}] {win['name']} -> {out}")


if __name__ == "__main__":
    main()
