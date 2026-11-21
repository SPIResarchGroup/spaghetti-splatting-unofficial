"""Train Spaghetti Splatting (or the point baseline) on a COLMAP or synthetic scene.

python train.py --data data/360_v2/bonsai --images images_2 --out runs/bonsai_gauss --kernel gaussian
python train.py --data data/synth_spaghetti --type synthetic --out runs/synth --kernel beta
python train.py ... --points            # same pipeline with zero-length segments (3DGS-MCMC-like baseline)
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from spaghetti.data import load_colmap, load_synthetic, scene_extent
from spaghetti.metrics import lpips, psnr, ssim
from spaghetti.model import SpaghettiModel


def expon_lr(it, lr0, lr1, max_steps):
    t = min(max(it / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr0) * (1 - t) + math.log(lr1) * t)


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--type", choices=["colmap", "synthetic"], default="colmap")
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", choices=["gaussian", "beta"], default="gaussian")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--cap", type=int, default=1_000_000)
    ap.add_argument("--points", action="store_true", help="zero-length segments (3DGS-MCMC-like baseline)")
    ap.add_argument("--no_split", action="store_true")
    ap.add_argument("--no_clone", action="store_true")
    ap.add_argument("--no_merge", action="store_true")
    ap.add_argument("--no_attract", action="store_true")
    ap.add_argument("--no_mip", action="store_true")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--bg", default=None, help="r,g,b in [0,1]; default black (colmap) / meta (synthetic)")
    ap.add_argument("--lambda_dssim", type=float, default=0.2)
    ap.add_argument("--opacity_reg", type=float, default=0.01)
    ap.add_argument("--scale_reg", type=float, default=0.01)
    ap.add_argument("--lambda_connect", type=float, default=0.01)
    ap.add_argument("--densify_from", type=int, default=500)
    ap.add_argument("--relocate_stop", type=int, default=None, help="default: iters - 5000")
    ap.add_argument("--merge_every", type=int, default=100)
    ap.add_argument("--merge_factor", type=float, default=0.5)
    ap.add_argument("--attract_factor", type=float, default=3.0)
    ap.add_argument("--split_len_factor", type=float, default=2.0)
    ap.add_argument("--init_seg_len", type=float, default=0.5)
    ap.add_argument("--noise_lr", type=float, default=5e5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--save_test_images", type=int, default=6)
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate(model, cams, bg, outdir=None, n_save=0, mip=True):
    res = {"psnr": [], "ssim": [], "lpips": [], "lpips_3dgs": []}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    for k, cam in enumerate(cams):
        rgb = model.render(cam, bg, mip=mip)[0].clamp(0, 1)
        gt = cam.image.float() / 255.0
        res["psnr"].append(psnr(rgb, gt))
        res["ssim"].append(float(ssim(rgb, gt)))
        res["lpips"].append(lpips(rgb, gt))
        res["lpips_3dgs"].append(lpips(rgb, gt, "3dgs"))
        if outdir and k < n_save:
            im = torch.cat([rgb, gt], 1).mul(255).byte().cpu().numpy()
            Image.fromarray(im).save(os.path.join(outdir, f"{k:03d}_{os.path.splitext(os.path.basename(cam.name))[0]}.png"))
    # FPS at test resolution
    for cam in cams[:3]:
        model.render(cam, bg, mip=mip)
    torch.cuda.synchronize()
    t0 = time.time()
    reps = 3
    for _ in range(reps):
        for cam in cams:
            model.render(cam, bg, mip=mip)
    torch.cuda.synchronize()
    fps = reps * len(cams) / (time.time() - t0)
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["fps"] = fps
    return out


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
    dev = "cuda"

    if args.type == "colmap":
        train_cams, test_cams, xyz, rgb = load_colmap(args.data, args.images, device=dev)
        bg = [0.0, 0.0, 0.0]
    else:
        train_cams, test_cams, xyz, rgb, meta = load_synthetic(args.data, device=dev)
        bg = meta.get("bg", [0.0, 0.0, 0.0])
    if args.bg:
        bg = [float(x) for x in args.bg.split(",")]
    bg = torch.tensor(bg, dtype=torch.float32, device=dev)
    extent = scene_extent(train_cams)
    print(f"{len(train_cams)} train / {len(test_cams)} test views, {xyz.shape[0]} init points, extent {extent:.3f}")

    mip = not args.no_mip
    model = SpaghettiModel(sh_degree=args.sh_degree, kernel=args.kernel, device=dev)
    model.split_len_factor = args.split_len_factor
    model.init_from_points(xyz, rgb, seg_len_factor=args.init_seg_len, point_mode=args.points)
    if mip:
        model.compute_filter3d(train_cams)
    lr = {"verts": 1.6e-4 * extent, "log_scales": 5e-3, "quats": 1e-3, "opacity": 5e-2, "sh_dc": 2.5e-3,
          "sh_rest": 2.5e-3 / 20, "beta": 5e-3}
    opt = model.make_optimizer(lr)
    relocate_stop = (args.relocate_stop if args.relocate_stop is not None
                     else max(args.densify_from, args.iters - min(5000, args.iters // 6)))
    topo = not args.points
    spawn_kw = dict(allow_split=not args.no_split, allow_clone=not args.no_clone)

    log = []
    order = []
    t_start = time.time()
    ema = None
    stale = len(train_cams) + 10  # one epoch (every view once) without contributing -> dead
    for it in range(1, args.iters + 1):
        model.iteration = it
        lr_xyz = expon_lr(it, 1.6e-4 * extent, 1.6e-6 * extent, args.iters)
        for g in opt.param_groups:
            if g["name"] == "verts":
                g["lr"] = lr_xyz
        if it % 1000 == 0:
            model.active_sh = min(model.active_sh + 1, model.sh_degree)
        if not order:
            order = list(range(len(train_cams)))
            random.shuffle(order)
        cam = train_cams[order.pop()]
        gt = cam.image.float() / 255.0

        rgb, alpha, info = model.render(cam, bg, mip=mip)
        l1 = (rgb - gt).abs().mean()
        loss = (1 - args.lambda_dssim) * l1 + args.lambda_dssim * (1 - ssim(rgb, gt))
        if topo and not args.no_attract and model.attract_pairs is not None:
            loss = loss + args.lambda_connect * model.attraction_loss()
        loss.backward()
        # L_MCMC (opacity + width regularisation of 3DGS-MCMC), applied only to primitives that received a
        # photometric gradient this step. Occluded primitives get exactly zero photometric gradient, so Adam
        # would normalise the tiny regulariser into full-lr steps and kill them all within ~400 iterations
        # (independent of the weight); see README.
        with torch.no_grad():
            c = model.opacity.grad != 0
            model.last_contrib[c] = it
            N = model.num_segments
            o = torch.sigmoid(model.opacity)
            model.opacity.grad[c] += args.opacity_reg * (o * (1 - o))[c] / N
            model.log_scales.grad[c] += args.scale_reg * torch.exp(model.log_scales)[c] / (3 * N)
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.add_noise(lr_xyz, args.noise_lr)

        with torch.no_grad():
            ema = float(l1) if ema is None else 0.95 * ema + 0.05 * float(l1)
            if args.densify_from <= it <= relocate_stop and it % 100 == 0:
                model.relocate(opt, stale_iters=stale, **spawn_kw)
                model.grow(opt, args.cap, **spawn_kw)
                if mip:
                    model.compute_filter3d(train_cams)
            if topo and it > relocate_stop and it % args.merge_every == 0:
                if not args.no_merge:
                    model.merge_endpoints(opt, args.merge_factor)
                if not args.no_attract:
                    model.update_attraction_pairs(args.attract_factor)

        if it % args.log_every == 0 or it == args.iters:
            el = time.time() - t_start
            _, sizes = model.components()
            with torch.no_grad():
                a, b = model.endpoints()
                ratio = (b - a).norm(dim=-1) / torch.exp(model.log_scales).max(-1).values
                lq = torch.quantile(ratio[:: max(1, ratio.numel() // 100000)], torch.tensor([0.5, 0.99], device=dev))
            rec = dict(it=it, l1=ema, psnr_train=psnr(rgb.detach(), gt), n_seg=model.num_segments,
                       n_vert=model.num_vertices, comp_mean=float(sizes.mean()), comp_max=int(sizes.max()),
                       pairs=info["n_pairs"], len_ratio_med=float(lq[0]), len_ratio_p99=float(lq[1]), time=el)
            log.append(rec)
            print(f"[{it:6d}] l1 {ema:.4f} psnr {rec['psnr_train']:.2f} segs {rec['n_seg']} verts {rec['n_vert']} "
                  f"comp {rec['comp_mean']:.2f}/{rec['comp_max']} L/s {lq[0]:.2f}/{lq[1]:.1f} "
                  f"pairs {rec['pairs']} {el / 60:.1f}min", flush=True)

    train_min = (time.time() - t_start) / 60
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    metrics = evaluate(model, test_cams, bg, os.path.join(args.out, "test"), args.save_test_images, mip=mip)
    _, sizes = model.components()
    metrics.update(n_seg=model.num_segments, n_vert=model.num_vertices, comp_mean=float(sizes.mean()),
                   comp_max=int(sizes.max()), train_minutes=train_min)
    print("TEST", json.dumps(metrics))
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
