"""Pick 80 non-overlapping 40-frame training windows from drives 0003/0007/0010.

Writes data_train/windows.json: a list of {name, drive, fids}, one per window.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .kitti360 import available_frames

WINDOW_LEN = 40
DRIVES = (3, 7, 10)
TARGET_TOTAL = 80
OUT = Path(__file__).resolve().parent.parent / "data_train" / "windows.json"


def find_windows(drive: int, length: int = WINDOW_LEN) -> list[list[int]]:
    """Greedy left-to-right scan: all non-overlapping runs of `length` consecutive
    available frame ids. Non-overlapping because each hit jumps the cursor past it."""
    avail = set(available_frames(drive))
    if not avail:
        return []
    frames = sorted(avail)
    windows: list[list[int]] = []
    i, last = frames[0], frames[-1]
    while i + length - 1 <= last:
        run = list(range(i, i + length))
        if all(f in avail for f in run):
            windows.append(run)
            i += length
        else:
            i += 1
    return windows


def _pick_evenly(candidates: list, n: int) -> list:
    """n items spread across candidates (by index), not clustered at the start."""
    if n <= 0:
        return []
    if n >= len(candidates):
        return candidates
    idxs = sorted({int(round(x)) for x in np.linspace(0, len(candidates) - 1, n)})
    pool = (i for i in range(len(candidates)) if i not in idxs)
    while len(idxs) < n:
        idxs.append(next(pool))
    return [candidates[i] for i in sorted(idxs)[:n]]


def select(target_total: int = TARGET_TOTAL) -> list[dict]:
    per_drive = {d: find_windows(d) for d in DRIVES}
    counts = {d: len(w) for d, w in per_drive.items()}
    total = sum(counts.values())
    if total < target_total:
        raise ValueError(f"only {total} candidate windows available, need {target_total}")

    alloc = {d: max(1, round(target_total * counts[d] / total)) for d in DRIVES}
    order = sorted(DRIVES, key=lambda d: counts[d], reverse=True)
    i = 0
    while sum(alloc.values()) != target_total:
        d = order[i % len(order)]
        if sum(alloc.values()) < target_total and alloc[d] < counts[d]:
            alloc[d] += 1
        elif sum(alloc.values()) > target_total and alloc[d] > 1:
            alloc[d] -= 1
        i += 1

    selected = []
    for d in DRIVES:
        for fids in _pick_evenly(per_drive[d], alloc[d]):
            selected.append({
                "name": f"drive{d:04d}_f{fids[0]:010d}_{len(fids)}",
                "drive": d,
                "fids": fids,
            })
    return selected


def main():
    windows = select()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(windows, indent=1))
    by_drive: dict[int, int] = {}
    for w in windows:
        by_drive[w["drive"]] = by_drive.get(w["drive"], 0) + 1
    print(f"selected {len(windows)} windows -> {OUT}")
    print("per drive:", by_drive)


if __name__ == "__main__":
    main()
