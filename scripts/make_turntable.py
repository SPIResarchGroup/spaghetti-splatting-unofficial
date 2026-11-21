"""Orbit GIF of a trained model: RGB render | Dirac-measure sketch (paper Fig. 3 style), side by side.

The orbit is fitted to the training cameras: its centre is the point closest to all optical axes, its axis the mean
camera up vector, its radius and height the median camera distance and mean height.

python scripts/make_turntable.py --run runs/shelly_kitten --data data/shelly/kitten --type blender --out figures/kitten.gif
python scripts/make_turntable.py --run runs/bonsai_long --data data/360_v2/bonsai --out figures/bonsai.gif
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spaghetti.data import load_blender, load_colmap, load_synthetic  # noqa: E402
from spaghetti.model import SpaghettiModel  # noqa: E402
from spaghetti.render import Camera  # noqa: E402
from vis_sketch import sketch  # noqa: E402


def orbit_cameras(cams, n, width, radius_scale=1.0, height_scale=1.0, zoom=1.0, height=None):
    R = torch.stack([c.R for c in cams]).double().cpu()
    C = torch.stack([c.center for c in cams]).double().cpu()
    d = R[:, 2]  # optical axes (OpenCV: +z forward)
    # point closest to all axes: sum_i (I - d_i d_i^T)(p - C_i) = 0
    P = torch.eye(3, dtype=torch.float64)[None] - d[:, :, None] * d[:, None, :]
    center = torch.linalg.solve(P.sum(0), (P @ C[:, :, None]).sum(0))[:, 0]
    up = -R[:, 1].mean(0)  # OpenCV: +y down
    up = up / up.norm()
    rel = C - center
    h = float((rel @ up).mean()) * height_scale
    horiz = rel - (rel @ up)[:, None] * up
    r = float(horiz.norm(dim=-1).median()) * radius_scale
    e1 = horiz[0] / horiz[0].norm()
    e2 = torch.linalg.cross(up, e1)
    c0 = cams[0]
    s = width / c0.W
    W, H = width, (height or round(c0.H * s))
    fx, fy = c0.fx * s * zoom, c0.fy * s * zoom
    out = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        pos = center + h * up + r * (np.cos(th) * e1 + np.sin(th) * e2)
        z = (center - pos) / (center - pos).norm()
        y = -(up - (up @ z) * z)
        y = y / y.norm()
        x = torch.linalg.cross(y, z)
        Rw = torch.stack([x, y, z]).float()
        t = -(Rw.double() @ pos).float()
        out.append(Camera(Rw.to(c0.R.device), t.to(c0.R.device), fx, fy, W / 2, H / 2, W, H, None,
                          f"orbit_{len(out):03d}"))
    return out


def tag(im, text, size):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        f = ImageFont.load_default()
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    pad = size // 3
    d.rectangle([0, im.size[1] - (y1 - y0) - 3 * pad, x1 - x0 + 3 * pad, im.size[1]], fill=(30, 32, 38))
    d.text((1.5 * pad - x0, im.size[1] - (y1 - y0) - 1.5 * pad - y0), text, fill=(255, 255, 255), font=f)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--type", choices=["colmap", "synthetic", "blender"], default="colmap")
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=480, help="width of each panel in pixels")
    ap.add_argument("--height", type=int, default=None, help="panel height (default: training aspect ratio)")
    ap.add_argument("--zoom", type=float, default=1.0, help="focal length multiplier")
    ap.add_argument("--radius_scale", type=float, default=1.0)
    ap.add_argument("--height_scale", type=float, default=1.0)
    ap.add_argument("--duration", type=int, default=80, help="ms per frame")
    ap.add_argument("--colors", type=int, default=256, help="GIF palette size (fewer = smaller file)")
    ap.add_argument("--width_px", type=float, default=0.15, help="sketch line width (screen pixels)")
    ap.add_argument("--const_alpha", type=float, default=0.35)
    ap.add_argument("--darken", type=float, default=0.85)
    ap.add_argument("--no_labels", action="store_true")
    ap.add_argument("--preview", action="store_true", help="only write the first frame as <out>.png")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.type == "colmap":
        train, _, _, _ = load_colmap(args.data, args.images)
        bg = torch.zeros(3, device="cuda")
    elif args.type == "blender":
        train, _, _, _ = load_blender(args.data)
        bg = torch.ones(3, device="cuda")
    else:
        train, _, _, _, meta = load_synthetic(args.data)
        bg = torch.tensor(meta.get("bg", [0.0, 0.0, 0.0]), device="cuda")
    m = SpaghettiModel(device="cuda")
    m.load_state_dict(torch.load(os.path.join(args.run, "model.pt"), map_location="cuda"))

    frames = []
    cams = orbit_cameras(train, args.frames, args.width, args.radius_scale, args.height_scale, args.zoom, args.height)
    for k, cam in enumerate(cams[:1] if args.preview else cams):
        with torch.no_grad():
            rgb = m.render(cam, bg)[0].clamp(0, 1)
        sk = sketch(m, cam, width_px=args.width_px, darken=args.darken, alpha_mode="const",
                    const_alpha=args.const_alpha)
        ims = [Image.fromarray(x.mul(255).byte().cpu().numpy()) for x in (rgb, sk)]
        if not args.no_labels:
            size = max(11, args.width // 32)
            ims = [tag(ims[0], "render", size), tag(ims[1], "Dirac measure", size)]
        f = Image.new("RGB", (2 * cam.W + 4, cam.H), "white")
        f.paste(ims[0], (0, 0))
        f.paste(ims[1], (cam.W + 4, 0))
        frames.append(f)
        print(f"\rframe {k + 1}/{args.frames}", end="", flush=True)
    if args.preview:
        frames[0].save(os.path.splitext(args.out)[0] + ".png")
        print(f"\nsaved preview {os.path.splitext(args.out)[0]}.png")
        return
    pal = [f.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=args.duration, loop=0, optimize=True)
    print(f"\nsaved {args.out} ({len(frames)} frames, {frames[0].size}, {os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
