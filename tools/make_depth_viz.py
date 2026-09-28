"""Regenerate the RGB / depth (Turbo colormap) / sky-mask stacked example viz.

Same 12 (window, fid) picks as the original SGBM viz pass (data_train/_viz/sgbm/),
so filenames line up 1:1 for a side-by-side diff against a different depth source.
Depth .npy files aren't kept on disk (deleted to save space), so this recomputes
depth on the fly for just these 12 frames via the RAFT-Stereo model.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess.raft_stereo_depth import load_model, raft_depth  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_train"
FID_OFFSETS = [0, 20, 39]
DEPTH_CLIP_M = 60.0


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    d_norm = (np.clip(depth, 0, DEPTH_CLIP_M) / DEPTH_CLIP_M * 255).astype(np.uint8)
    color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
    color[~valid] = (30, 30, 30)
    return color


def build_example(win: dict, fid: int, model, out_dir: Path) -> Path:
    base = DATA / win["name"]
    rgb = cv2.imread(str(base / f"{fid}_00.png"))
    mask = cv2.imread(str(base / "mask" / f"{fid}_00.png"), cv2.IMREAD_GRAYSCALE)
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    depth = raft_depth(win["drive"], fid, model)
    depth_vis = colorize_depth(depth)
    cv2.putText(depth_vis, f"RAFT-Stereo  valid={100 * (depth > 0).mean():.0f}%",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    stacked = np.concatenate([rgb, depth_vis, mask_vis], axis=0)
    out = out_dir / f"{win['name']}_{fid}.png"
    cv2.imwrite(str(out), stacked)
    return out


def main():
    windows = json.loads((DATA / "windows.json").read_text())
    picks = [windows[0], windows[13], windows[13 + 17], windows[-1]]

    out_dir = DATA / "_viz" / "raft"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    for win in picks:
        for off in FID_OFFSETS:
            fid = win["fids"][off]
            out = build_example(win, fid, model, out_dir)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
