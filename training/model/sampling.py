"""Hierarchical ray sampling, per App B.2:

"The total number of samples located at the ray is 160 with 128 sampled from the
foreground and 32 from the background. Specifically, we first select 80 points
along the ray within the foreground volume, distributing one-half of these
samples uniformly, while the other half is allocated linearly based on disparity
spacing, following Nerfacto. Next, we iteratively perform importance sampling to
sample valid regions containing solid contents based on the coarse density
prediction following NeuS, with three iterations and 16 samples for each
iteration. For the background, we additionally uniformly sample 32 points out of
the foreground volume."

80 + 3*16 = 128 foreground samples, + 32 background = 160 total.

The foreground density decoder we built (foreground.py) outputs plain ReLU
density, not a signed distance field, so "following NeuS" is read here as
reusing NeuS's *iterative coarse-to-fine* importance-sampling strategy (resample
where accumulated alpha-compositing weight concentrates, refine repeatedly),
not NeuS's SDF-specific "s-density" formulation, which doesn't apply to a plain
density field. This is the standard NeRF hierarchical-sampling `sample_pdf`
algorithm, just run 3 times instead of once.

AABB/coordinate conventions match preprocess/accumulate.py, training/model/foreground.py
(normalized-world, middle-frame-anchored -- see [[edus-reproduction-project]]).
"""
from __future__ import annotations

import torch

AABB_MIN = torch.tensor([-12.8, -9.0, -20.0])
AABB_MAX = torch.tensor([12.8, 3.8, 31.2])

N_UNIFORM = 40
N_DISPARITY = 40
N_IMPORTANCE_ROUNDS = 3
N_IMPORTANCE_PER_ROUND = 16
N_BACKGROUND = 32
BACKGROUND_MAX_T = 200.0  # matches this project's MAX_DEPTH_M ballpark for stereo depth


def ray_aabb_intersection(origins: torch.Tensor, directions: torch.Tensor,
                           aabb_min: torch.Tensor = AABB_MIN, aabb_max: torch.Tensor = AABB_MAX):
    """origins, directions: (N,3). Returns t_near, t_far (N,), hit (N,) bool.
    Standard slab method. t_near is clamped to be positive (don't start behind
    the camera even if the AABB technically extends there)."""
    aabb_min = aabb_min.to(origins)
    aabb_max = aabb_max.to(origins)
    inv_d = 1.0 / directions
    t0 = (aabb_min - origins) * inv_d
    t1 = (aabb_max - origins) * inv_d
    t_small = torch.minimum(t0, t1)
    t_large = torch.maximum(t0, t1)
    t_near = t_small.max(dim=-1).values
    t_far = t_large.min(dim=-1).values
    hit = (t_far > t_near) & (t_far > 0)
    t_near = torch.clamp(t_near, min=1e-3)
    return t_near, t_far, hit


def _jittered_fracs(n: int, batch: int, device, deterministic: bool) -> torch.Tensor:
    """Stratified [0,1] fractions: split into n bins, one sample per bin --
    the bin's midpoint if deterministic, else a random point inside it (the
    standard NeRF/Nerfacto "stratified sampling" trick, App B.2 citing
    Nerfacto for the uniform+disparity scheme). Returns (batch, n)."""
    edges = torch.linspace(0, 1, n + 1, device=device)
    lower, upper = edges[:-1], edges[1:]
    if deterministic:
        return ((lower + upper) / 2)[None, :].expand(batch, n)
    u = torch.rand(batch, n, device=device)
    return lower[None, :] + (upper - lower)[None, :] * u


def sample_uniform_and_disparity(t_near: torch.Tensor, t_far: torch.Tensor,
                                  n_uniform: int = N_UNIFORM, n_disparity: int = N_DISPARITY,
                                  deterministic: bool = False) -> torch.Tensor:
    """t_near, t_far: (N,). Returns sorted t_vals (N, n_uniform+n_disparity).
    deterministic=False (training): each of the n_uniform/n_disparity bins gets
    a randomly-jittered sample point, not a fixed grid position, so the network
    can't exploit samples always landing on the same discrete t-values.
    deterministic=True (eval/rendering): bin midpoints, for reproducible output."""
    device = t_near.device
    N = t_near.shape[0]

    frac_u = _jittered_fracs(n_uniform, N, device, deterministic)
    t_uniform = t_near[:, None] + frac_u * (t_far - t_near)[:, None]

    frac_d = _jittered_fracs(n_disparity, N, device, deterministic)
    near_disp = 1.0 / t_near
    far_disp = 1.0 / t_far
    t_disparity = 1.0 / (near_disp[:, None] + frac_d * (far_disp - near_disp)[:, None])

    t_vals = torch.cat([t_uniform, t_disparity], dim=-1)
    t_vals, _ = torch.sort(t_vals, dim=-1)
    return t_vals


def compute_weights(t_vals: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    """t_vals, density: (N,S). Standard alpha-compositing weights (NeRF eq.).
    Returns weights (N,S): each sample's contribution to the ray's final color."""
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    alpha = 1.0 - torch.exp(-density * dists)
    ones = torch.ones_like(alpha[..., :1])
    transmittance = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    return alpha * transmittance


def sample_pdf(t_vals: torch.Tensor, weights: torch.Tensor, n_samples: int,
                deterministic: bool = False) -> torch.Tensor:
    """Standard NeRF hierarchical (inverse-CDF) resampling. t_vals, weights: (N,S)
    -- SAME length (weights[i] is sample i's own contribution, per compute_weights).
    Returns new_t_vals (N, n_samples), NOT merged with the input t_vals.

    deterministic=False (training): quantiles `u` are drawn randomly, so two
    calls with the same weights give different samples -- adds the same kind of
    stratification benefit as sample_uniform_and_disparity's jitter.
    deterministic=True (eval/rendering): u is an evenly-spaced grid of
    quantiles instead, for reproducible output.

    Note: the classic reference implementation of this algorithm expects `bins`
    with one MORE entry than `weights` (S+1 bin edges for S interval weights).
    Here t_vals/weights are the same length S instead, so `above` is clamped to
    S-1 rather than allowed to reach S -- a harmless approximation affecting only
    the single last bucket's interpolation range, avoiding an out-of-bounds
    gather into t_vals."""
    S = t_vals.shape[-1]
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (N, S+1)

    if deterministic:
        u = torch.linspace(0, 1, n_samples, device=weights.device)
        u = u[None, :].expand(*cdf.shape[:-1], n_samples).contiguous()
    else:
        u = torch.rand(*cdf.shape[:-1], n_samples, device=weights.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0, max=S - 1)
    above = torch.clamp(inds, min=0, max=S - 1)
    inds_g = torch.stack([below, above], dim=-1)  # (N, n_samples, 2), valid indices into t_vals

    cdf_trim = cdf[..., :S]  # drop the final padded entry so it matches t_vals' length
    cdf_g = torch.gather(cdf_trim[..., None, :].expand(*cdf_trim.shape[:-1], n_samples, S), -1, inds_g)
    bins_g = torch.gather(t_vals[..., None, :].expand(*t_vals.shape[:-1], n_samples, S), -1, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    frac = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + frac * (bins_g[..., 1] - bins_g[..., 0])


def sample_background(t_far: torch.Tensor, n_samples: int = N_BACKGROUND,
                       max_t: float = BACKGROUND_MAX_T) -> torch.Tensor:
    """t_far: (N,) foreground exit distance. Returns sorted t_vals (N, n_samples)
    spaced by disparity from t_far out to max_t (unbounded background, coarse)."""
    device = t_far.device
    d = torch.linspace(0, 1, n_samples, device=device)
    near_disp = 1.0 / t_far.clamp(min=1e-3)
    far_disp = 1.0 / torch.full_like(t_far, max_t)
    t_vals = 1.0 / (near_disp[:, None] + d[None, :] * (far_disp - near_disp)[:, None])
    t_vals, _ = torch.sort(t_vals, dim=-1)
    return t_vals
