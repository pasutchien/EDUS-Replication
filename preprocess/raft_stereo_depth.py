"""Stereo depth via RAFT-Stereo (github.com/princeton-vl/RAFT-Stereo, cloned to third_party/).

Upgrade over the initial OpenCV StereoSGBM pass (stereo_depth.py, kept for reference):
a real, purpose-built learned stereo network -- respects the epipolar/horizontal-only
constraint properly (unlike an optical-flow net repurposed for stereo). It's a 2021
repo pinned to torch 1.7/1.11 in its own docs but runs unmodified on our torch
2.5.1+cu121 stack (verified). No KITTI-specific checkpoint ships in their model zoo;
`middlebury` is the authors' recommended generalist checkpoint for in-the-wild images.

Unlike SGBM, this is a dense network -- it always outputs *something*, even in sky /
textureless regions (matching the released EDUS_inferdata depth maps, which are also
fully dense). We still zero out sky pixels using the semantic mask, since that region
must never contribute foreground points regardless of what disparity the net guesses.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .kitti360 import SKY_SEMANTIC_ID, image_path, load_perspective, semantic_path

ROOT = Path(__file__).resolve().parent.parent
RAFT_DIR = ROOT / "third_party" / "RAFT-Stereo"
CKPT = RAFT_DIR / "models" / "raftstereo-middlebury.pth"
for p in (RAFT_DIR, RAFT_DIR / "core"):  # repo root (for `core.xxx` imports inside raft_stereo.py)
    if str(p) not in sys.path:            # and core/ itself (for `raft_stereo`, `utils.utils`)
        sys.path.insert(0, str(p))

MAX_DEPTH_M = 100.0
VALID_ITERS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    from raft_stereo import RAFTStereo  # noqa: E402  (needs RAFT_DIR/core on sys.path)

    args = argparse.Namespace(
        hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False,
        corr_levels=4, corr_radius=4, n_downsample=2, context_norm="batch",
        slow_fast_gru=False, n_gru_layers=3, mixed_precision=(DEVICE == "cuda"),
    )
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0] if DEVICE == "cuda" else None)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    return model.module.to(DEVICE).eval()


def _load_img(path) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB")).astype(np.uint8)
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(DEVICE)


@torch.no_grad()
def raft_depth(drive: int, fid: int, model, iters: int = VALID_ITERS) -> np.ndarray:
    from utils.utils import InputPadder  # noqa: E402

    cal = load_perspective()
    L, R = _load_img(image_path(drive, 0, fid)), _load_img(image_path(drive, 1, fid))
    padder = InputPadder(L.shape, divis_by=32)
    Lp, Rp = padder.pad(L, R)
    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        _, flow_up = model(Lp, Rp, iters=iters, test_mode=True)
    disp = -padder.unpad(flow_up).squeeze().float().cpu().numpy()  # RAFT-Stereo's flow_up is negative

    depth = np.zeros_like(disp)
    valid = disp > 0
    depth[valid] = cal["fx"] * cal["baseline"] / disp[valid]
    depth[depth >= MAX_DEPTH_M] = 0.0

    sem = np.array(Image.open(semantic_path(drive, 0, fid)))
    depth[sem == SKY_SEMANTIC_ID] = 0.0
    return depth.astype(np.float32)


def build_window_depth(win: dict, out_root: Path, model, overwrite: bool = True) -> Path:
    out = out_root / win["name"] / "depth"
    out.mkdir(parents=True, exist_ok=True)
    for fid in win["fids"]:
        dst = out / f"{fid}_00.npy"
        if dst.exists() and not overwrite:
            continue
        np.save(dst, raft_depth(win["drive"], fid, model))
    return out


def main():
    import json
    import time

    windows = json.loads((ROOT / "data_train" / "windows.json").read_text())
    model = load_model()
    t0 = time.time()
    for i, win in enumerate(windows):
        out = build_window_depth(win, ROOT / "data_train", model)
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(windows)}] {win['name']} -> {out}  ({elapsed/60:.1f} min elapsed)")


if __name__ == "__main__":
    main()
