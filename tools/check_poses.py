"""Verify the pose pipeline against the 5 released EDUS_inferdata scenes.

For each scene we rebuild transforms.json from raw KITTI-360 GT poses and compare:
  * box-local geometry   inv(bbx2w) @ transform_matrix   -- must match to ~1e-4
  * global transform_matrix / bbx2w / inv_pose           -- may differ by a small
    rigid (our deterministic frame vs the authors' offline one); report the residual.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess.kitti360 import C_FLIP, cam1_from_cam0, load_cam0_to_world
from preprocess.normalize import make_transforms

INFER = Path(__file__).resolve().parent.parent / "EDUS_inferdata"

SCENES = {
    "seq_00_nerfacto_7840_40": (0, range(7840, 7880)),
    "seq_04_nerfacto_0382_40": (4, range(382, 422)),
    "seq_04_nerfacto_1741_40": (4, range(1741, 1781)),
    "seq_04_nerfacto_2320_40": (4, range(2320, 2360)),
    "seq_04_nerfacto_3340_40": (4, range(3340, 3380)),
}


def rel(P):
    """box-local poses: inv(pose[0]) @ pose[i] for a stack of 4x4."""
    return np.linalg.inv(P[0])[None] @ P


def main():
    for name, (drive, rng) in SCENES.items():
        ref = json.loads((INFER / name / "transforms.json").read_text())
        fids = list(rng)
        mine = make_transforms(drive, fids)

        Pr = np.array([f["transform_matrix"] for f in ref["frames"]])
        Pm = np.array([f["transform_matrix"] for f in mine["frames"]])
        assert [f["file_path"].split(".")[0] for f in ref["frames"]] == \
               [f["file_path"].split(".")[0] for f in mine["frames"]], "frame order mismatch"

        # box-local geometry (frame-0-relative) — the part that must be exact
        rel_err = np.abs(rel(Pr) - rel(Pm)).max()

        # global frame residual: best-fit rigid Pr ~ S @ Pm
        A = Pm.reshape(-1, 4, 4)
        # translation Kabsch on camera centres
        gm, gr = Pm[:, :3, 3], Pr[:, :3, 3]
        cm, cr = gm - gm.mean(0), gr - gr.mean(0)
        U, _, Vt = np.linalg.svd(cm.T @ cr)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Rfit = Vt.T @ np.diag([1, 1, d]) @ U.T
        t_res = np.linalg.norm((Rfit @ cm.T).T - cr, axis=1).max()
        ang_res = np.degrees(np.arccos(np.clip((np.trace(Rfit) - 1) / 2, -1, 1)))

        # cam1 derivation sanity (against released _01 entries)
        cam1_err = np.abs(rel(Pr)[1::2] - rel(Pm)[1::2]).max()

        print(f"{name}")
        print(f"  box-local pose err (max abs)      : {rel_err:.2e}   {'OK' if rel_err < 1e-3 else 'FAIL'}")
        print(f"  cam1 rel-pose err                 : {cam1_err:.2e}")
        print(f"  global-frame residual  rot={ang_res:6.3f} deg  trans={t_res:.3f} m")
        print(f"  intrinsics match                  : "
              f"{all(abs(ref[k]-mine[k])<1e-3 for k in ('fl_x','fl_y','cx','cy'))}")


if __name__ == "__main__":
    main()
