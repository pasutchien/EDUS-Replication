"""PyTorch Dataset for EDUS training: random camera rays (from a window's
evaluation/supervision frames) paired with that window's voxel volume.

Ray convention matches what's stored in transforms.json's `transform_matrix`
(OpenGL/NeRF: X right, Y up, Z backward -- camera looks down -Z), which is what
the C_FLIP in normalize.py was built to produce, confirmed against the real
released camera trajectory (driving forward decreases box-local Z, i.e. forward
is -Z). Supervision uses only `evaluation_fids` from frames.json (see
[[edus-reproduction-project]] for why: the mod-10 {1,3,7,9} pattern, disjoint
from the stride-5 reference/geometry frames, so every supervised ray is a
genuine held-out-relative-to-geometry render).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocess.accumulate import evaluation_indices
from training.model.sampling import AABB_MIN, AABB_MAX

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_train"

# Every window is a 40-frame sequence, so this is a fixed constant -- used to turn
# a (window_idx, local eval-frame idx) pair into a single global appearance-
# embedding index (Table 4's per-image code w, App A.2).
EVAL_PER_WINDOW = len(evaluation_indices(40))  # 16

_AABB_MIN_NP = AABB_MIN.numpy()
_AABB_MAX_NP = AABB_MAX.numpy()
# Eval frames near/past the window's temporal boundary can have a camera pose
# outside (or barely inside) the foreground AABB -- see [[edus-reproduction-project]]:
# the foreground field then contributes ~nothing for most of that frame's rays,
# leaning entirely on background+sky. Measured across all 1280 eval frames (80
# windows): 231 (18%) sit fully outside the AABB, rising to only 289 (22.6%) at
# this 2m clearance margin (vs. 510/39.8% at 3m, 1011/79% at 5m -- much steeper,
# likely just catching normal pose wobble against the tight +Y=3.8m ceiling
# rather than genuine boundary-of-window cases) -- so 2m cuts the worst
# offenders without over-pruning. At this margin every window still keeps at
# least 8 of its 16 eval frames (mean 12.4), so sample_rays always has choices.
AABB_MARGIN_M = 2.0


def _pose_clears_aabb(transform_matrix: list, margin: float = AABB_MARGIN_M) -> bool:
    pos = np.array(transform_matrix, dtype=np.float32)[:3, 3]
    return bool(np.all(pos - _AABB_MIN_NP >= margin) and np.all(_AABB_MAX_NP - pos >= margin))


class WindowData:
    """Loads one window's images, poses, sky masks and voxel volume into memory.

    Also loads *reference*-frame images (both eyes -- App B.3's stride-5 set,
    left+right per App D.1's "stereo pairs" wording, see [[edus-reproduction-project]])
    for the foreground field's 2D image-based color retrieval branch, plus the
    num_neighbour_select nearest reference views for each evaluation camera
    (nearest by camera-position distance, matching get_nearest_pose_ids in the
    released nerfstudio/data/datamanagers/utils.py, angular_dist_method='dist')."""

    def __init__(self, window_name: str, window_idx: int, num_neighbour_select: int = 3):
        self.name = window_name
        self.window_idx = window_idx
        self.dir = DATA / window_name
        self.num_neighbour_select = num_neighbour_select

        tj = json.loads((self.dir / "transforms.json").read_text())
        self.fx, self.fy = float(tj["fl_x"]), float(tj["fl_y"])
        self.cx, self.cy = float(tj["cx"]), float(tj["cy"])
        self.W, self.H = int(tj["w"]), int(tj["h"])
        frame_by_path = {f["file_path"]: f for f in tj["frames"]}

        frames_json = json.loads((self.dir / "frames.json").read_text())
        eval_fids_all = frames_json["evaluation_fids"]
        ref_fids = frames_json["reference_fids"]
        eval_fids = [fid for fid in eval_fids_all
                     if _pose_clears_aabb(frame_by_path[f"{fid}_00.png"]["transform_matrix"])]
        assert eval_fids, f"{window_name}: every eval frame's pose was filtered out by AABB_MARGIN_M"
        self.eval_fids = eval_fids  # frame id per eval image, indexed like self.images

        images, poses, masks = [], [], []
        for fid in eval_fids:
            key = f"{fid}_00.png"  # evaluation views are left-eye only (App D.1)
            images.append(np.asarray(Image.open(self.dir / key).convert("RGB"), dtype=np.float32) / 255.0)
            poses.append(np.array(frame_by_path[key]["transform_matrix"], dtype=np.float32))
            masks.append(np.asarray(Image.open(self.dir / "mask" / key), dtype=np.float32) / 255.0)

        self.images = np.stack(images)   # (N, H, W, 3)
        self.poses = np.stack(poses)     # (N, 4, 4)
        self.sky_masks = np.stack(masks)  # (N, H, W), 1 = sky

        ref_images, ref_poses, ref_masks = [], [], []
        for fid in ref_fids:
            for cam in ("00", "01"):  # both eyes
                key = f"{fid}_{cam}.png"
                ref_images.append(np.asarray(Image.open(self.dir / key).convert("RGB"), dtype=np.float32) / 255.0)
                ref_poses.append(np.array(frame_by_path[key]["transform_matrix"], dtype=np.float32))
                ref_masks.append(np.asarray(Image.open(self.dir / "mask" / key), dtype=np.float32) / 255.0)
        self.ref_images = np.stack(ref_images)  # (M, H, W, 3)
        self.ref_poses = np.stack(ref_poses)    # (M, 4, 4)
        self.ref_sky_masks = np.stack(ref_masks)  # (M, H, W), 1 = sky
        # torch, channel-first, ready for F.grid_sample once a small subset is picked
        self.ref_images_t = torch.from_numpy(self.ref_images).permute(0, 3, 1, 2).contiguous()  # (M,3,H,W)
        # (M,1,H,W) -- sky masks as a 1-channel "image" so retrieve_sky_colors can
        # grid_sample them the same way it samples RGB
        self.ref_sky_masks_t = torch.from_numpy(self.ref_sky_masks).unsqueeze(1).contiguous()  # (M,1,H,W)

        eval_pos = self.poses[:, :3, 3]                    # (N,3)
        ref_pos = self.ref_poses[:, :3, 3]                  # (M,3)
        dists = np.linalg.norm(eval_pos[:, None, :] - ref_pos[None, :, :], axis=-1)  # (N,M)
        self.nearest_ref_idx = np.argsort(dists, axis=1)[:, :num_neighbour_select]   # (N,K)

        volume = np.load(self.dir / "voxel" / "volume.npy").astype(np.float32)  # (128,64,256,3)
        self.volume = torch.from_numpy(volume).permute(3, 0, 1, 2).contiguous()  # (3,128,64,256)

    def _build_rays(self, img_idx: int, px: np.ndarray, py: np.ndarray) -> dict:
        """Shared by sample_rays (random pixel subset) and full_image_rays (every
        pixel): given an evaluation frame index and pixel coordinates, build the
        ray batch dict model.render() expects."""
        c2w = self.poses[img_idx]  # (4,4)
        dirs_cam = np.stack([
            (px - self.cx) / self.fx,
            -(py - self.cy) / self.fy,
            -np.ones_like(px),
        ], axis=-1)  # (B,3), OpenGL convention
        dirs_world = dirs_cam @ c2w[:3, :3].T
        origins = np.broadcast_to(c2w[:3, 3], dirs_world.shape)

        py_i, px_i = py.astype(np.int64), px.astype(np.int64)
        rgb = self.images[img_idx, py_i, px_i]
        sky = self.sky_masks[img_idx, py_i, px_i]

        ref_idx = self.nearest_ref_idx[img_idx]  # (K,)
        appearance_idx = self.window_idx * EVAL_PER_WINDOW + img_idx
        return {
            "ray_origins": torch.from_numpy(origins.astype(np.float32)),
            "ray_directions": torch.from_numpy(dirs_world.astype(np.float32)),
            "rgb": torch.from_numpy(rgb.astype(np.float32)),
            "sky": torch.from_numpy(sky.astype(np.float32)),
            "ref_images": self.ref_images_t[ref_idx],                            # (K,3,H,W)
            "ref_poses": torch.from_numpy(self.ref_poses[ref_idx].copy()),       # (K,4,4)
            "ref_sky_masks": self.ref_sky_masks_t[ref_idx],                      # (K,1,H,W)
            "appearance_idx": torch.tensor(appearance_idx, dtype=torch.long),  # scalar
        }

    def sample_rays(self, n_rays: int) -> dict:
        """Random rays, all from ONE randomly-chosen evaluation frame (not mixed
        across frames) so that the whole batch shares the same K nearest reference
        images -- required for the 2D retrieval branch to stay cheap: gathering a
        full reference image per individual ray would be enormous (thousands of
        full-resolution images per batch)."""
        img_idx = np.random.randint(0, len(self.images))
        px = np.random.randint(0, self.W, size=n_rays).astype(np.float32)
        py = np.random.randint(0, self.H, size=n_rays).astype(np.float32)
        return self._build_rays(img_idx, px, py)

    def full_image_rays(self, img_idx: int) -> dict:
        """Every pixel of evaluation frame img_idx, in raster order (row-major,
        i.e. flat index = py*W + px) -- for rendering a complete preview image
        rather than a random training batch."""
        us, vs = np.meshgrid(np.arange(self.W), np.arange(self.H))  # (H,W) each
        px = us.reshape(-1).astype(np.float32)
        py = vs.reshape(-1).astype(np.float32)
        return self._build_rays(img_idx, px, py)


class EdusDataset(Dataset):
    """Wraps all training windows; each __getitem__ samples a batch of rays from
    one randomly-chosen window (lazily loaded, LRU-cached) plus its voxel volume."""

    def __init__(self, rays_per_batch: int = 4096, n_batches: int = 10_000, cache_size: int = 8):
        self.windows = json.loads((DATA / "windows.json").read_text())
        self.rays_per_batch = rays_per_batch
        self.n_batches = n_batches
        self.cache_size = cache_size
        self._cache: dict[str, WindowData] = {}
        # total size the model's appearance-embedding table needs (Table 4's w)
        self.num_images = len(self.windows) * EVAL_PER_WINDOW

    def __len__(self):
        return self.n_batches

    def _get_window(self, idx: int) -> WindowData:
        name = self.windows[idx]["name"]
        if name not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[name] = WindowData(name, window_idx=idx)
        return self._cache[name]

    def __getitem__(self, idx):
        win_idx = np.random.randint(0, len(self.windows))
        wd = self._get_window(win_idx)
        batch = wd.sample_rays(self.rays_per_batch)
        batch["volume"] = wd.volume
        batch["window_name"] = wd.name
        return batch


def collate_single(batch_list):
    """DataLoader batch_size should stay 1 -- each __getitem__ already returns a
    full ray batch for one window. This just unwraps the length-1 outer list."""
    return batch_list[0]
