"""Render a trained model three ways (cf. paper Fig. 1 / Fig. 3):
  rgb       - the radiance field
  polylines - every connected polyline in its own random colour
  dirac     - the Dirac measure: segments drawn with a tiny kernel (thin lines; points become dots)

python scripts/visualize.py --run runs/bonsai_gauss --data data/360_v2/bonsai --views 0 3
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spaghetti.data import load_colmap, load_synthetic  # noqa: E402
from spaghetti.model import SpaghettiModel, inverse_sigmoid  # noqa: E402


def component_colors(model, seed=0):
    lab, _ = model.components()
    rng = np.random.default_rng(seed)
    pal = rng.uniform(0.15, 1.0, size=(lab.max() + 1, 3)).astype(np.float32)
    return torch.from_numpy(pal[lab]).to(model.device)


@torch.no_grad()
def render_views(model, cams, bg, out_prefix, dirac_px=0.6, min_opacity=0.05):
    cols = component_colors(model)
    # Dirac view: kernel shrunk to ~dirac_px pixels at each primitive's depth, fixed opacity, no Mip filter
    dirac = SpaghettiModel(model.sh_degree, model.kernel, model.device)
    dirac.load_state_dict(model.state_dict())
    dirac.filter3d = None
    keep = torch.sigmoid(model.opacity) >= min_opacity
    imgs = []
    for k, cam in enumerate(cams):
        rgb = model.render(cam, bg)[0].clamp(0, 1)
        poly = model.render(cam, bg, override_color=cols)[0].clamp(0, 1)
        a, b = dirac.endpoints()
        z = (0.5 * (a + b)) @ cam.R[2] + cam.t[2]
        dirac.log_scales = torch.log((z.clamp_min(1e-3) * dirac_px / cam.fx))[:, None].expand(-1, 3).contiguous()
        dirac.opacity = torch.where(keep, inverse_sigmoid(torch.tensor(0.95, device=model.device)),
                                    torch.tensor(-20.0, device=model.device))
        dline = dirac.render(cam, torch.ones(3, device=model.device), override_color=cols * 0.85, mip=False,
                             eps2d=0.0)[0].clamp(0, 1)
        row = torch.cat([rgb, poly, dline], 1)
        if cam.image is not None:
            row = torch.cat([cam.image.float() / 255.0, row], 1)
        imgs.append(row)
        Image.fromarray(row.mul(255).byte().cpu().numpy()).save(f"{out_prefix}_{k}.png")
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--type", choices=["colmap", "synthetic"], default="colmap")
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--views", type=int, nargs="+", default=[0])
    ap.add_argument("--scale", type=float, default=1.0, help="render resolution scale")
    args = ap.parse_args()
    sd = torch.load(os.path.join(args.run, "model.pt"), map_location="cuda")
    model = SpaghettiModel(device="cuda")
    model.load_state_dict(sd)
    if args.type == "colmap":
        _, test, _, _ = load_colmap(args.data, args.images)
        bg = torch.zeros(3, device="cuda")
    else:
        _, test, _, _, meta = load_synthetic(args.data)
        bg = torch.tensor(meta["bg"], device="cuda")
    cams = [test[v] for v in args.views]
    if args.scale != 1.0:
        cams = [c.scaled(args.scale) for c in cams]
    os.makedirs(os.path.join(args.run, "vis"), exist_ok=True)
    render_views(model, cams, bg, os.path.join(args.run, "vis", "view"))
    print("saved to", os.path.join(args.run, "vis"))


if __name__ == "__main__":
    main()
