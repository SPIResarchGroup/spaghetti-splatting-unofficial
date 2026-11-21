"""Paper-style views that expose the primitives themselves (cf. Fig. 1 right, Fig. 3):

  rgb        - the radiance field
  strokes    - every segment's Dirac support drawn as a thin stroke (sigma 0.35 px) in its own colour, white background
  shrunk     - every kernel shrunk to 30% of its size, own colour: individual primitives become visible
               (ellipsoids for the point baseline, capsules for segments)
  polylines  - each connected polyline in its own random colour (full-size kernels)
  primitives - each primitive in its own random colour (full-size kernels)

python scripts/vis_primitives.py --runs runs/synth_gauss runs/synth_points --data data/synth_spaghetti \
    --type synthetic --views 1 5 --out figures/synth --turntable 48
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spaghetti.data import load_colmap, load_synthetic  # noqa: E402
from spaghetti.model import SpaghettiModel, inverse_sigmoid  # noqa: E402
from spaghetti.render import eval_sh  # noqa: E402

VIEWS = ["rgb", "strokes", "shrunk", "polylines", "primitives"]


def load_model(run):
    m = SpaghettiModel(device="cuda")
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cuda"))
    return m


def clone(m):
    c = SpaghettiModel(m.sh_degree, m.kernel, m.device)
    c.load_state_dict(m.state_dict())
    return c


@torch.no_grad()
def true_colors(m, cam):
    a, b = m.endpoints()
    dirs = F.normalize(0.5 * (a + b) - cam.center[None], dim=-1)
    sh = torch.cat([m.sh_dc, m.sh_rest], 1)
    return torch.clamp(eval_sh(m.active_sh, sh, dirs) + 0.5, 0, 1)


def random_colors(labels, seed=0):
    rng = np.random.default_rng(seed)
    pal = rng.uniform(0.1, 0.95, size=(int(labels.max()) + 1, 3)).astype(np.float32)
    return torch.from_numpy(pal[labels]).cuda()


@torch.no_grad()
def render_view(m, cam, bg, view, min_opacity, comp_labels):
    white = torch.ones(3, device="cuda")
    if view == "rgb":
        return m.render(cam, bg)[0]
    if view in ("polylines", "primitives"):
        lab = comp_labels if view == "polylines" else np.arange(m.num_segments)
        return m.render(cam, bg, override_color=random_colors(lab, 1 if view == "primitives" else 0))[0]
    keep = torch.sigmoid(m.opacity) >= min_opacity
    v = clone(m)
    v.filter3d = None
    col = true_colors(m, cam)
    hi = inverse_sigmoid(torch.tensor(0.9, device="cuda"))
    if view == "strokes":
        a, b = v.endpoints()
        z = (0.5 * (a + b)) @ cam.R[2] + cam.t[2]
        v.log_scales = torch.log(z.clamp_min(1e-3) * 0.35 / cam.fx)[:, None].expand(-1, 3).contiguous()
        v.opacity = torch.where(keep, hi, torch.tensor(-30.0, device="cuda"))
        return v.render(cam, white, override_color=col, mip=False, eps2d=0.05)[0]
    if view == "shrunk":
        v.log_scales = (m.log_scales + np.log(0.3)).contiguous()
        o = torch.sigmoid(m.opacity)
        v.opacity = torch.where(keep, inverse_sigmoid(torch.clamp(o, 0.8, 0.99)), torch.tensor(-30.0, device="cuda"))
        return v.render(cam, white, override_color=col, mip=False, eps2d=0.05)[0]
    raise ValueError(view)


def to_img(x):
    return Image.fromarray(x.clamp(0, 1).mul(255).byte().cpu().numpy())


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 9 * len(text) + 10, 22], fill=(0, 0, 0))
    d.text((5, 4), text, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--type", choices=["colmap", "synthetic"], default="colmap")
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--views", type=int, nargs="+", default=[0])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--min_opacity", type=float, default=0.05)
    ap.add_argument("--out", required=True, help="output prefix, e.g. figures/synth")
    ap.add_argument("--turntable", type=int, default=0, help="#frames of a GIF orbit (strokes + rgb)")
    ap.add_argument("--crop", type=float, nargs=4, default=None, help="x0 y0 x1 y1 in [0,1] for a zoomed crop row")
    args = ap.parse_args()
    names = args.names or [os.path.basename(r.rstrip("/")) for r in args.runs]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.type == "colmap":
        train, test, _, _ = load_colmap(args.data, args.images)
        bg = torch.zeros(3, device="cuda")
    else:
        train, test, _, _, meta = load_synthetic(args.data)
        bg = torch.tensor(meta["bg"], device="cuda")
    models = [load_model(r) for r in args.runs]
    labels = [m.components()[0] for m in models]

    for vi in args.views:
        cam = test[vi]
        gt = cam.image.float() / 255.0
        if args.scale != 1.0:
            gt = F.interpolate(gt.permute(2, 0, 1)[None], scale_factor=args.scale, mode="area")[0].permute(1, 2, 0)
            cam = cam.scaled(args.scale)
        rows = []
        for name, m, lab in zip(names, models, labels):
            tiles = [label(to_img(gt), "GT")]
            for v in VIEWS:
                img = to_img(render_view(m, cam, bg, v, args.min_opacity, lab))
                tiles.append(label(img, f"{name}: {v}"))
            rows.append(tiles)
        W, H = rows[0][0].size
        grid = Image.new("RGB", (W * len(rows[0]), H * len(rows)), "white")
        for r, tiles in enumerate(rows):
            for c, t in enumerate(tiles):
                grid.paste(t, (c * W, r * H))
        grid.save(f"{args.out}_view{vi}_grid.jpg", quality=92)
        if args.crop:
            x0, y0, x1, y1 = args.crop
            box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
            crops = []
            for tiles in rows:
                crops.append([t.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST)
                              for t in tiles])
            cw, ch = crops[0][0].size
            cg = Image.new("RGB", (cw * len(crops[0]), ch * len(crops)), "white")
            for r, tiles in enumerate(crops):
                for c, t in enumerate(tiles):
                    cg.paste(t, (c * cw, r * ch))
            cg.save(f"{args.out}_view{vi}_crop.jpg", quality=92)
        print("saved", f"{args.out}_view{vi}_grid.jpg")

    if args.turntable:
        cams = sorted(train + test, key=lambda c: c.name)
        idx = np.linspace(0, len(cams) - 1, args.turntable).round().astype(int)
        frames = []
        for k in idx:
            cam = cams[k] if args.scale == 1.0 else cams[k].scaled(args.scale)
            tiles = []
            for name, m in zip(names, models):
                tiles.append(label(to_img(render_view(m, cam, bg, "rgb", args.min_opacity, None)), f"{name}: rgb"))
                tiles.append(label(to_img(render_view(m, cam, bg, "strokes", args.min_opacity, None)), f"{name}: strokes"))
            W, H = tiles[0].size
            fr = Image.new("RGB", (W * 2, H * len(models)), "white")
            for i, t in enumerate(tiles):
                fr.paste(t, ((i % 2) * W, (i // 2) * H))
            frames.append(fr)
        frames[0].save(f"{args.out}_turntable.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
        print("saved", f"{args.out}_turntable.gif")


if __name__ == "__main__":
    main()
