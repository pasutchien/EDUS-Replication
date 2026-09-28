"""Build one training-window folder (RGB + transforms.json + sky masks) from windows.json.

Output layout mirrors EDUS_inferdata:
    data_train/<name>/<fid>_0{0,1}.png
    data_train/<name>/mask/<fid>_0{0,1}.png     (binary 0/255 sky mask)
    data_train/<name>/transforms.json
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .kitti360 import SKY_SEMANTIC_ID, image_path, semantic_path
from .normalize import make_transforms

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = ROOT / "data_train" / "windows.json"
OUT_ROOT = ROOT / "data_train"


def sky_mask(drive: int, cam: int, fid: int) -> np.ndarray:
    sem = np.array(Image.open(semantic_path(drive, cam, fid)))
    return np.where(sem == SKY_SEMANTIC_ID, 255, 0).astype(np.uint8)


def build_window(win: dict, overwrite: bool = False) -> Path:
    out = OUT_ROOT / win["name"]
    done_marker = out / "transforms.json"
    if done_marker.exists() and not overwrite:
        return out
    (out / "mask").mkdir(parents=True, exist_ok=True)

    drive, fids = win["drive"], win["fids"]
    for fid in fids:
        for cam in (0, 1):
            shutil.copyfile(image_path(drive, cam, fid), out / f"{fid}_{cam:02d}.png")
            Image.fromarray(sky_mask(drive, cam, fid)).save(out / "mask" / f"{fid}_{cam:02d}.png")

    transforms = make_transforms(drive, fids)
    done_marker.write_text(json.dumps(transforms, indent=1))
    return out


def main():
    windows = json.loads(WINDOWS.read_text())
    for i, win in enumerate(windows):
        out = build_window(win)
        print(f"[{i+1}/{len(windows)}] {win['name']} -> {out}")


if __name__ == "__main__":
    main()
