"""'Where is the spaghetti?' view: draw only the connected polylines (noodles), one colour per polyline.

Each segment is drawn as a thin line along its Dirac support (kernel shrunk to ~0.6 px); only polylines with at
least --min_len segments are kept. Panels: RGB | noodles on white | noodles over the faded RGB.

python scripts/vis_spaghetti.py --run runs/bonsai_gauss --data data/360_v2/bonsai --view 3 --scale 0.5 \
    --out figures/bonsai_spaghetti
"""
import argparse
import colorsys
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spaghetti.data import load_colmap, load_synthetic  # noqa: E402
from spaghetti.model import SpaghettiModel, inverse_sigmoid  # noqa: E402


def bright_palette(n, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 1, n)
    return np.array([colorsys.hsv_to_rgb(x, 0.85, 0.9) for x in h], np.float32)


@torch.no_grad()
def noodle_render(m, cam, labels, sizes, min_len, min_opacity, px=0.35):
    keep_comp = sizes >= min_len
    seg_keep = torch.from_numpy(keep_comp[labels]).cuda() & (torch.sigmoid(m.opacity) >= min_opacity)
    pal = torch.from_numpy(bright_palette(len(sizes))).cuda()
    col = pal[torch.from_numpy(labels).cuda()]
    v = SpaghettiModel(m.sh_degree, m.kernel, m.device)
    v.load_state_dict(m.state_dict())
    v.filter3d = None
    a, b = v.endpoints()
    z = (0.5 * (a + b)) @ cam.R[2] + cam.t[2]
    v.log_scales = torch.log(z.clamp_min(1e-3) * px / cam.fx)[:, None].expand(-1, 3).contiguous()
    v.opacity = torch.where(seg_keep, inverse_sigmoid(torch.tensor(0.9, device="cuda")),
                            torch.tensor(-30.0, device="cuda"))
    rgb, alpha, _ = v.render(cam, torch.zeros(3, device="cuda"), override_color=col, mip=False, eps2d=0.05)
    return rgb, alpha, int(keep_comp.sum()), int(sizes[keep_comp].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--type", choices=["colmap", "synthetic"], default="colmap")
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--min_len", type=int, default=4, help="minimum #segments of a polyline to draw")
    ap.add_argument("--min_opacity", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = SpaghettiModel(device="cuda")
    m.load_state_dict(torch.load(os.path.join(args.run, "model.pt"), map_location="cuda"))
    if args.type == "colmap":
        _, test, _, _ = load_colmap(args.data, args.images)
        bg = torch.zeros(3, device="cuda")
    else:
        _, test, _, _, meta = load_synthetic(args.data)
        bg = torch.tensor(meta["bg"], device="cuda")
    cam = test[args.view] if args.scale == 1.0 else test[args.view].scaled(args.scale)
    labels, sizes = m.components()
    with torch.no_grad():
        rgb = m.render(cam, bg)[0].clamp(0, 1)
        nood, al, ncomp, nseg = noodle_render(m, cam, labels, sizes, args.min_len, args.min_opacity)
    white = nood + (1 - al[..., None])  # noodles over white
    faded = 0.35 * rgb + 0.65  # washed-out photo
    over = nood + (1 - al[..., None]) * faded
    tiles = [rgb, white.clamp(0, 1), over.clamp(0, 1)]
    labs = ["RGB", f"polylines with >= {args.min_len} segments ({ncomp} polylines, {nseg} segments)",
            "same, over the faded photo"]
    ims = []
    for t, lab in zip(tiles, labs):
        im = Image.fromarray(t.mul(255).byte().cpu().numpy())
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 7 * len(lab) + 12, 20], fill=(0, 0, 0))
        d.text((6, 4), lab, fill=(255, 255, 255))
        ims.append(im)
    W, H = ims[0].size
    out = Image.new("RGB", (W * 3, H))
    for i, im in enumerate(ims):
        out.paste(im, (i * W, 0))
    out.save(args.out + ".jpg", quality=92)
    ims[1].save(args.out + "_only.png")
    print(f"saved {args.out}.jpg  ({ncomp} polylines >= {args.min_len} segments, {nseg} segments; "
          f"total segments {m.num_segments}, longest polyline {sizes.max()} segments)")


if __name__ == "__main__":
    main()
