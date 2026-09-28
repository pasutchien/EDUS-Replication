"""Feed-forward-only validation on the 5 real released EDUS_inferdata scenes at
Drop50/80/90 reference sparsity, reporting PSNR/SSIM/LPIPS in Table 1's format.

Deliberately skips per-scene fine-tuning / test-time optimization (Table 1's
"per-scene opt." rows) -- this only reproduces the "No per-scene opt." feed-
forward numbers, per the user's explicit request, and additionally reports
Drop90 (which Table 1 itself omits -- see the conversation established in
[[edus-reproduction-project]] for why: Table 1 only benchmarks other
generalizable methods at 50%/80%, Drop90 only appears in Table 2 for
test-time-optimization baselines).

Why these 5 scenes instead of held-out windows from our own 80 training
windows: they were NEVER touched during training (unlike our own windows,
which the model has already seen), so this is a genuine test of
generalization, matching what Table 1 itself measures.

Reference/geometry sparsity uses the exact strides from App D.2: Drop50 =
every 2nd frame, Drop80 = every 5th, Drop90 = every 10th (measured within each
scene's own 40 local frame indices, 0..39). Reuses the released precomputed
drop{50,80,90}_voxel/*_volume.npy directly (already verified to match our own
voxelization format/convention -- see [[edus-preprocessing-conventions]])
rather than re-running our own accumulate.py/voxelize.py at custom strides.

Evaluation frames reuse this project's own held-out pattern (App D.1, mod-10
{1,3,7,9} within each 10-frame block) -- verified disjoint from the reference
set at all three strides (residues {even}, {0,5,15,25,35}, {0,10,20,30} vs
{1,3,7,9,11,13,...}, no overlap in any case).

Appearance embedding: these scenes are unseen during training, so no trained
appearance_idx row is meaningful for them -- uses the mean embedding across all
trained images instead (App A.2's use_average_appearance_embedding case).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from preprocess.kitti360 import load_perspective
from training.model.model import EdusModel
from training.model.sampling import ray_aabb_intersection

INFERDATA = Path(__file__).resolve().parent.parent / "EDUS_inferdata"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoint"

DROP_STRIDES = {50: 2, 80: 5, 90: 10}
EVAL_OFFSETS = (1, 3, 7, 9)
NUM_NEIGHBOUR = 3


def reference_local_indices(stride: int, n: int = 40) -> list[int]:
    return list(range(0, n, stride))


def eval_local_indices(n: int = 40) -> list[int]:
    out = []
    for block_start in range(0, n, 10):
        out += [block_start + o for o in EVAL_OFFSETS if block_start + o < n]
    return out


class InferdataScene:
    """One released EDUS_inferdata scene, loaded at a given drop rate: reference
    images (both eyes, at the drop-rate's stride) for 2D color retrieval, the
    matching precomputed voxel volume, and held-out eval frames (left-eye,
    disjoint from the reference set) as ground truth."""

    def __init__(self, scene_dir: Path, drop_rate: int, num_neighbour_select: int = NUM_NEIGHBOUR):
        self.dir = scene_dir
        self.name = scene_dir.name
        self.drop_rate = drop_rate
        stride = DROP_STRIDES[drop_rate]

        tj = json.loads((scene_dir / "transforms.json").read_text())
        self.fx, self.fy = float(tj["fl_x"]), float(tj["fl_y"])
        self.cx, self.cy = float(tj["cx"]), float(tj["cy"])
        self.W, self.H = int(tj["w"]), int(tj["h"])
        frame_by_path = {f["file_path"]: f for f in tj["frames"]}

        fids = sorted(set(int(f["file_path"].split("_")[0]) for f in tj["frames"]))
        assert len(fids) == 40, f"{scene_dir.name}: expected 40 frames, got {len(fids)}"
        start_fid = fids[0]

        ref_local = reference_local_indices(stride)
        eval_local = eval_local_indices()

        ref_images, ref_poses, ref_masks = [], [], []
        for li in ref_local:
            fid = fids[li]
            for cam in ("00", "01"):
                key = f"{fid}_{cam}.png"
                ref_images.append(np.asarray(Image.open(scene_dir / key).convert("RGB"), dtype=np.float32) / 255.0)
                ref_poses.append(np.array(frame_by_path[key]["transform_matrix"], dtype=np.float32))
                mask_path = scene_dir / "mask" / key
                if mask_path.exists():
                    # released EDUS_inferdata masks use 0=sky/255=non-sky --
                    # confirmed by inspection (zero pixels exactly trace the
                    # sky region), OPPOSITE of our own preprocessing's
                    # 255=sky convention (dataset.py/accumulate.py) -- invert
                    # so ref_sky_masks means the same thing (1=sky) either way.
                    ref_masks.append(1.0 - np.asarray(Image.open(mask_path), dtype=np.float32) / 255.0)
                else:
                    ref_masks.append(np.zeros((self.H, self.W), dtype=np.float32))
        self.ref_images = np.stack(ref_images)
        self.ref_poses = np.stack(ref_poses)
        self.ref_sky_masks = np.stack(ref_masks)
        self.ref_images_t = torch.from_numpy(self.ref_images).permute(0, 3, 1, 2).contiguous()
        self.ref_sky_masks_t = torch.from_numpy(self.ref_sky_masks).unsqueeze(1).contiguous()

        images, poses = [], []
        for li in eval_local:
            fid = fids[li]
            key = f"{fid}_00.png"
            images.append(np.asarray(Image.open(scene_dir / key).convert("RGB"), dtype=np.float32) / 255.0)
            poses.append(np.array(frame_by_path[key]["transform_matrix"], dtype=np.float32))
        self.images = np.stack(images)
        self.poses = np.stack(poses)
        self.eval_fids = [fids[li] for li in eval_local]

        eval_pos = self.poses[:, :3, 3]
        ref_pos = self.ref_poses[:, :3, 3]
        dists = np.linalg.norm(eval_pos[:, None, :] - ref_pos[None, :, :], axis=-1)
        self.nearest_ref_idx = np.argsort(dists, axis=1)[:, :num_neighbour_select]

        # volume filenames use 4-digit zero-padding (e.g. "0382_volume.npy"),
        # even though every other file uses the plain unpadded fid ("382_00.png")
        # -- confirmed by directly listing the released files, not assumed.
        volume_path = scene_dir / f"drop{drop_rate}_voxel" / f"{start_fid:04d}_volume.npy"
        volume = np.load(volume_path).astype(np.float32)  # (128,64,256,3)
        self.volume = torch.from_numpy(volume).permute(3, 0, 1, 2).contiguous()  # (3,128,64,256)

    def full_image_rays(self, img_idx: int) -> dict:
        c2w = self.poses[img_idx]
        us, vs = np.meshgrid(np.arange(self.W), np.arange(self.H))
        px = us.reshape(-1).astype(np.float32)
        py = vs.reshape(-1).astype(np.float32)
        dirs_cam = np.stack([
            (px - self.cx) / self.fx,
            -(py - self.cy) / self.fy,
            -np.ones_like(px),
        ], axis=-1)
        dirs_world = dirs_cam @ c2w[:3, :3].T
        origins = np.broadcast_to(c2w[:3, 3], dirs_world.shape)

        ref_idx = self.nearest_ref_idx[img_idx]
        return {
            "ray_origins": torch.from_numpy(origins.astype(np.float32)),
            "ray_directions": torch.from_numpy(dirs_world.astype(np.float32)),
            "ref_images": self.ref_images_t[ref_idx],
            "ref_poses": torch.from_numpy(self.ref_poses[ref_idx].copy()),
            "ref_sky_masks": self.ref_sky_masks_t[ref_idx],
        }


@torch.no_grad()
def render_scene_frame(model: EdusModel, feats_volume: torch.Tensor, scene: InferdataScene, img_idx: int,
                        appearance_embedding: torch.Tensor, device: str, chunk_size: int = 8192) -> torch.Tensor:
    """Full-image feed-forward render, using a fixed (mean) appearance
    embedding since this scene was never seen during training -- no learned
    per-image code exists for it (App A.2's average-embedding case).

    feats_volume: already-encoded (from model.encode(scene.volume)) -- passed
    in rather than recomputed here, since it's identical across all 16 eval
    frames of one (scene, drop_rate); encoding it once per scene instead of
    once per frame avoids 16x redundant SPADE-CNN forward passes."""
    batch = scene.full_image_rays(img_idx)
    batch = {k: v.to(device) for k, v in batch.items()}

    shared = {k: batch[k] for k in ("ref_images", "ref_poses", "ref_sky_masks")}
    n_rays = batch["ray_origins"].shape[0]

    rgb_chunks = []
    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)
        chunk_batch = {
            "ray_origins": batch["ray_origins"][start:end],
            "ray_directions": batch["ray_directions"][start:end],
            "appearance_embedding": appearance_embedding,
            **shared,
        }
        out = model.render_from_features(feats_volume, chunk_batch, scene.fx, scene.fy, scene.cx, scene.cy,
                                          deterministic=True)
        rgb_chunks.append(out["rgb"].cpu())

    rgb_full = torch.cat(rgb_chunks, dim=0)
    return rgb_full.reshape(scene.H, scene.W, 3).clamp(0, 1)


def validate(checkpoint: Path = CKPT_DIR / "final.pt", device: str | None = None,
             scenes: list[str] | None = None, drop_rates: tuple[int, ...] = (50, 80, 90),
             max_eval_frames: int | None = None) -> dict:
    """max_eval_frames: if given and smaller than a scene's eval-frame count
    (16), evenly subsample down to this many per (scene, drop_rate) instead of
    rendering all of them -- trades statistical depth for wall-clock time
    while still covering every scene and every drop rate (the actual axes
    Table 1 compares), rather than dropping a whole scene or sparsity level."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scene_dirs = sorted(p for p in INFERDATA.iterdir() if p.is_dir())
    if scenes is not None:
        scene_dirs = [p for p in scene_dirs if p.name in scenes]
    assert scene_dirs, f"no scenes found under {INFERDATA}"

    ckpt = torch.load(checkpoint, map_location=device)
    # num_images must match what the checkpoint's appearance-embedding table was
    # trained with -- infer it from the checkpoint itself rather than recomputing
    # from windows.json (robust to that changing, e.g. the AABB-margin filter).
    num_images = ckpt["model"]["color_branch.embedding_appearance.weight"].shape[0]
    model = EdusModel(num_images=num_images).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # 1-D (appearance_dim,), matching the shape embedding_appearance(scalar_idx)
    # would normally produce -- broadcasts against (n_rays,n_samples,dim) the
    # same way, see ColorBranch.forward's `w.dim() < h.dim()` expand.
    mean_embedding = model.color_branch.embedding_appearance.weight.mean(dim=0).to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    results = {}
    for drop_rate in drop_rates:
        psnr_metric.reset(); ssim_metric.reset(); lpips_metric.reset()
        n_frames = 0
        for scene_dir in scene_dirs:
            scene = InferdataScene(scene_dir, drop_rate)
            with torch.no_grad():
                feats_volume = model.encode(scene.volume.to(device))
            frame_indices = range(len(scene.images))
            if max_eval_frames is not None and max_eval_frames < len(scene.images):
                frame_indices = sorted(set(np.linspace(0, len(scene.images) - 1, max_eval_frames).round().astype(int).tolist()))
            for img_idx in frame_indices:
                pred = render_scene_frame(model, feats_volume, scene, img_idx, mean_embedding, device).permute(2, 0, 1)[None]
                gt = torch.from_numpy(scene.images[img_idx]).permute(2, 0, 1)[None].to(device)
                pred, gt = pred.to(device), gt
                psnr_metric.update(pred, gt)
                ssim_metric.update(pred, gt)
                lpips_metric.update(pred, gt)
                n_frames += 1
                print(f"  drop{drop_rate} {scene.name} fid={scene.eval_fids[img_idx]} "
                      f"[{img_idx+1}/{len(scene.images)}]")
        results[drop_rate] = {
            "psnr": psnr_metric.compute().item(),
            "ssim": ssim_metric.compute().item(),
            "lpips": lpips_metric.compute().item(),
            "n_frames": n_frames,
        }
        r = results[drop_rate]
        print(f"Drop{drop_rate}: PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.3f}  LPIPS={r['lpips']:.3f}  "
              f"({r['n_frames']} frames)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=CKPT_DIR / "final.pt")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--drop-rates", type=int, nargs="+", default=[50, 80, 90])
    ap.add_argument("--max-eval-frames", type=int, default=None,
                     help="evenly subsample each scene's eval frames down to this many, per drop rate "
                          "(default: use all 16) -- trades statistical depth for wall-clock time")
    args = ap.parse_args()
    validate(checkpoint=args.checkpoint, device=args.device, drop_rates=tuple(args.drop_rates),
             max_eval_frames=args.max_eval_frames)
