"""Synthetic 'spaghetti' scene: thin coloured 3D strands, rendered with gsplat (an independent rasterizer)
from dense tangent-aligned Gaussians, 2x supersampled. Also stores the ground-truth strand polylines.

python scripts/make_synthetic.py --out data/synth_spaghetti
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image


def smooth_noise(n, sigma, rng):
    x = rng.normal(size=(n + 6 * sigma, 3))
    k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    k /= np.sqrt((k ** 2).sum())
    return np.stack([np.convolve(x[:, i], k, mode="valid")[:n] for i in range(3)], 1)


def make_strand(rng, length, step=0.005, curl=0.035, R=0.9):
    n = int(length / step)
    p = rng.normal(size=3)
    p = p / np.linalg.norm(p) * R * rng.uniform(0.2, 0.9)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    kappa = smooth_noise(n, 25, rng) * curl
    pts = [p.copy()]
    for i in range(n):
        d = d + kappa[i]
        if np.linalg.norm(p) > R:  # stay inside the ball
            d = d - 0.08 * p / np.linalg.norm(p)
        d /= np.linalg.norm(d)
        p = p + step * d
        pts.append(p.copy())
    return np.array(pts)


def frame_quats(T):
    """Quaternion (w,x,y,z) of a rotation whose x axis is the tangent T."""
    a = np.where(np.abs(T[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    N = np.cross(T, a)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B = np.cross(T, N)
    Rm = np.stack([T, N, B], 2)  # columns
    w = np.sqrt(np.clip(1 + Rm[:, 0, 0] + Rm[:, 1, 1] + Rm[:, 2, 2], 1e-12, None)) / 2
    x = (Rm[:, 2, 1] - Rm[:, 1, 2]) / (4 * w)
    y = (Rm[:, 0, 2] - Rm[:, 2, 0]) / (4 * w)
    z = (Rm[:, 1, 0] - Rm[:, 0, 1]) / (4 * w)
    return np.stack([w, x, y, z], 1)


def look_at(C, target=np.zeros(3)):
    f = target - C
    f /= np.linalg.norm(f)
    up = np.array([0, 1.0, 0]) if abs(f[1]) < 0.95 else np.array([1.0, 0, 0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    Rm = np.stack([x, y, f], 0)  # world -> camera (OpenCV: x right, y down, z forward)
    return Rm, -Rm @ C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth_spaghetti")
    ap.add_argument("--n_strands", type=int, default=48)
    ap.add_argument("--radius", type=float, default=0.006)
    ap.add_argument("--views", type=int, default=128)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import gsplat

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    palette = {"pasta": np.array([0.93, 0.76, 0.36]), "tomato": np.array([0.80, 0.18, 0.10]),
               "basil": np.array([0.20, 0.55, 0.18])}
    strands, means, quats, scales, colors, bases = [], [], [], [], [], []
    r = args.radius
    for k in range(args.n_strands):
        pts = make_strand(rng, length=rng.uniform(1.2, 2.6))
        strands.append(pts.astype(np.float32))
        kind = rng.choice(["pasta", "tomato", "basil"], p=[0.7, 0.2, 0.1])
        base = np.clip(palette[kind] * rng.uniform(0.85, 1.1), 0, 1)
        bases.append(base)
        # dense Gaussians along the polyline
        seg = np.diff(pts, axis=0)
        L = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0], np.cumsum(L)])
        h = 0.5 * r
        ss = np.arange(0, s[-1], h)
        P = np.stack([np.interp(ss, s, pts[:, i]) for i in range(3)], 1)
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        shade = 0.82 + 0.18 * np.sin(ss * 40 + rng.uniform(0, 6.28))  # mild texture along the strand
        means.append(P)
        quats.append(frame_quats(T))
        scales.append(np.tile([[0.8 * h, 0.55 * r, 0.55 * r]], (len(P), 1)))
        colors.append(base[None] * shade[:, None])
    dev = "cuda"
    to = lambda x: torch.tensor(np.concatenate(x), dtype=torch.float32, device=dev)
    means, quats, scales, colors = to(means), to(quats), to(scales), to(colors)
    opac = torch.full((means.shape[0],), 0.9, device=dev)
    bg = [0.10, 0.12, 0.16]
    print(f"{args.n_strands} strands, {means.shape[0]} GT gaussians")

    W = H = args.res
    fov = math.radians(40)
    f = 0.5 * W / math.tan(0.5 * fov)
    ss = 2  # supersampling
    Kss = torch.tensor([[f * ss, 0, W * ss / 2], [0, f * ss, H * ss / 2], [0, 0, 1]], dtype=torch.float32, device=dev)
    frames = []
    gold = math.pi * (3 - math.sqrt(5))
    for i in range(args.views):
        y = 1 - 2 * (i + 0.5) / args.views
        rad = math.sqrt(1 - y * y)
        C = 3.2 * np.array([math.cos(gold * i) * rad, y, math.sin(gold * i) * rad])
        Rm, t = look_at(C, rng.normal(size=3) * 0.05)
        view = torch.eye(4, device=dev)
        view[:3, :3] = torch.tensor(Rm, dtype=torch.float32)
        view[:3, 3] = torch.tensor(t, dtype=torch.float32)
        img, _, _ = gsplat.rasterization(means, quats, scales, opac, colors, view[None], Kss[None], W * ss, H * ss,
                                         backgrounds=torch.tensor([bg], device=dev), eps2d=0.3)
        img = torch.nn.functional.avg_pool2d(img[0].permute(2, 0, 1)[None], ss)[0].permute(1, 2, 0)
        fn = f"images/{i:03d}.png"
        Image.fromarray(img.clamp(0, 1).mul(255).round().byte().cpu().numpy()).save(os.path.join(args.out, fn))
        frames.append(dict(file=fn, R=Rm.tolist(), t=t.tolist()))
    json.dump(dict(fx=f, fy=f, cx=W / 2, cy=H / 2, W=W, H=H, bg=bg, strand_radius=r, frames=frames),
              open(os.path.join(args.out, "transforms.json"), "w"), indent=1)
    offs = np.cumsum([0] + [len(p) for p in strands])
    np.savez(os.path.join(args.out, "gt_strands.npz"), points=np.concatenate(strands), offsets=offs, radius=r)
    # Simulated SfM initialisation (stand-in for COLMAP points, shared by all methods): noisy points on the
    # strands (sigma ~1.7x strand radius, colour + noise) plus 20% uniform outliers in the bounding box.
    # (A purely random box init collapses: thin strands make every blob harmful before it can shrink.)
    allp = np.concatenate(strands)
    allc = np.concatenate([np.repeat(c[None], len(p), 0) for c, p in zip(bases, strands)])
    n_on, n_out = 8000, 2000
    pick = rng.integers(0, len(allp), n_on)
    on = allp[pick] + rng.normal(scale=0.01, size=(n_on, 3))
    on_rgb = np.clip(allc[pick] + rng.normal(scale=0.05, size=(n_on, 3)), 0, 1)
    lo, hi = allp.min(0) - 0.1, allp.max(0) + 0.1
    out = rng.uniform(lo, hi, size=(n_out, 3))
    init = np.concatenate([on, out]).astype(np.float32)
    rgb = np.concatenate([on_rgb, np.full((n_out, 3), 0.5)]).astype(np.float32)
    np.savez(os.path.join(args.out, "init_points.npz"), xyz=init, rgb=rgb)
    print("saved", args.out)


if __name__ == "__main__":
    main()
