"""Training loop for EDUS (Sec 4 "Training Details" + App B.1).

Adam, lr=5e-3 (the paper's "initial learning rate" -- no explicit decay
schedule is given in the text we have, so it's kept constant here rather than
inventing one; revisit if training plateaus). Input Volume Random Masking
(App B.1, masking.py) is applied every step, zeroing an 8x8x12m chunk of the
window's own voxel volume before it reaches the encoder -- training-only, per
the paper's own note that it's skipped for inference/fine-tuning. Loss =
rgb + lambda_sky*sky + lambda_entropy*entropy (losses.py; LiDAR term dropped,
since LiDAR supervision is deliberately deferred for this reproduction -- see
[[edus-reproduction-project]]).

Batching matches the paper directly: "we randomly select a single image and
randomly sample 4096 pixels as a batch" -- exactly WindowData.sample_rays'
one-window/one-frame-per-batch design.
"""
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from preprocess.kitti360 import load_perspective
from training.dataset import EdusDataset, collate_single
from training.model.losses import total_loss
from training.model.masking import mask_volume
from training.model.model import EdusModel
from training.render_utils import render_full_image

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoint"
RENDER_DIR = Path(__file__).resolve().parent.parent / "renders"


def render_preview(model, ds: EdusDataset, fx, fy, cx, cy, device, step: int, render_dir: Path):
    """Picks a random window + random evaluation frame, renders the full
    image (all pixels, deterministic sampling) and saves it as
    {scene_name}_{image_name}_step_{num_step}.png, per the user's request."""
    render_dir.mkdir(parents=True, exist_ok=True)
    win_idx = random.randint(0, len(ds.windows) - 1)
    wd = ds._get_window(win_idx)
    img_idx = random.randint(0, len(wd.images) - 1)
    image_name = wd.eval_fids[img_idx]

    rgb = render_full_image(model, wd, img_idx, fx, fy, cx, cy, device)
    img = Image.fromarray((rgb.numpy() * 255.0).clip(0, 255).astype(np.uint8))
    out_path = render_dir / f"{wd.name}_{image_name}_step_{step}.png"
    img.save(out_path)
    print(f"[{step}] saved preview render {out_path}")


def train(n_steps: int = 10_000, rays_per_batch: int = 4096, lr: float = 5e-3,
          log_every: int = 50, ckpt_every: int = 1000, ckpt_dir: Path = CKPT_DIR,
          render_every: int = 500, render_dir: Path = RENDER_DIR,
          resume: Path | None = None, device: str | None = None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cal = load_perspective()  # same KITTI-360 calibration for every window
    fx, fy, cx, cy = cal["fx"], cal["fy"], cal["cx"], cal["cy"]

    ds = EdusDataset(rays_per_batch=rays_per_batch, n_batches=n_steps)
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_single, num_workers=0)

    model = EdusModel(num_images=ds.num_images).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_step = 0
    elapsed_time = 0.0  # cumulative wall-clock TRAINING time, persisted across resumes

    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        elapsed_time = ckpt.get("elapsed_time", 0.0)  # .get: older checkpoints predate this field
        print(f"resumed from {resume} at step {start_step}, {elapsed_time/3600:.2f}h trained so far")

    def save_ckpt(path: Path, step: int):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "elapsed_time": elapsed_time}, path)

    model.train()
    for step, batch in enumerate(dl, start=start_step):
        if device == "cuda":
            torch.cuda.synchronize()
        step_start = time.time()

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        volume = mask_volume(batch["volume"])  # App B.1, fresh mask every iteration

        out = model.render(volume, batch, fx, fy, cx, cy)
        losses = total_loss(out["rgb"], batch["rgb"], out["acc"], batch["sky"], out["acc_fg"])

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        step_time = time.time() - step_start
        elapsed_time += step_time

        if step % log_every == 0:
            print(f"[{step}] {batch['window_name']}  total={losses['total'].item():.4f} "
                  f"rgb={losses['rgb'].item():.4f} sky={losses['sky'].item():.4f} "
                  f"entropy={losses['entropy'].item():.4f}  "
                  f"step_time={step_time:.2f}s  elapsed={elapsed_time/3600:.2f}h")

        if step > 0 and step % ckpt_every == 0:
            save_ckpt(ckpt_dir / f"step_{step}.pt", step)

        if step > 0 and step % render_every == 0:
            render_preview(model, ds, fx, fy, cx, cy, device, step, render_dir)

    save_ckpt(ckpt_dir / "final.pt", step)  # step, not n_steps -- n_steps is steps-THIS-run, step is the true absolute step reached (matters across resumes)
    print(f"done, saved {ckpt_dir / 'final.pt'} at step {step}  total training time: {elapsed_time/3600:.2f}h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=10_000)
    ap.add_argument("--rays-per-batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--render-every", type=int, default=500)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()
    train(n_steps=args.n_steps, rays_per_batch=args.rays_per_batch, lr=args.lr,
          log_every=args.log_every, ckpt_every=args.ckpt_every, render_every=args.render_every,
          resume=args.resume)
