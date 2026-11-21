"""Paper Fig. 3 (right) style: the Dirac measure of every primitive drawn as a thin, semi-transparent line in its
own (darkened) colour on white, depth-composited and supersampled -> a pencil-sketch look.
Points (the 3DGS baseline) become dots.

python scripts/vis_sketch.py --run runs/bonsai_gauss --data data/360_v2/bonsai --view 3 --out figures/bonsai_sketch

Paper-like look (thin lines, every live primitive drawn with the same alpha):
python scripts/vis_sketch.py ... --alpha_mode const --const_alpha 0.35 --width_px 0.15 --darken 0.85
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spaghetti.data import load_colmap, load_synthetic  # noqa: E402
from spaghetti.model import SpaghettiModel, inverse_sigmoid  # noqa: E402
from spaghetti.render import eval_sh  # noqa: E402


@torch.no_grad()
def sketch(m, cam, ss=2, width_px=0.35, alpha_gain=1.5, alpha_max=0.85, darken=0.65, min_opacity=0.005,
           alpha_mode="opacity", const_alpha=0.5):
    """alpha_mode="opacity": line alpha ~ primitive opacity (faint lines for wide, low-density kernels).
    alpha_mode="const": every live primitive drawn with the same alpha -> shows the Dirac support only, which is
    what the paper's Fig. 3 depicts (wide background kernels then appear as long hatching strokes)."""
    c2 = cam.scaled(ss)
    v = SpaghettiModel(m.sh_degree, m.kernel, m.device)
    v.load_state_dict(m.state_dict())
    v.filter3d = None
    a, b = v.endpoints()
    dirs = F.normalize(0.5 * (a + b) - cam.center[None], dim=-1)
    col = torch.clamp(eval_sh(m.active_sh, torch.cat([m.sh_dc, m.sh_rest], 1), dirs) + 0.5, 0, 1) * darken
    z = (0.5 * (a + b)) @ c2.R[2] + c2.t[2]
    v.log_scales = torch.log(z.clamp_min(1e-3) * width_px * ss / c2.fx)[:, None].expand(-1, 3).contiguous()
    o = torch.sigmoid(m.opacity)
    if alpha_mode == "const":
        target = torch.full_like(o, const_alpha)
    else:
        target = (o * alpha_gain).clamp(max=alpha_max)  # line alpha ~ primitive opacity (mass weight keeps it per length)
    v.opacity = torch.where(o >= min_opacity, inverse_sigmoid(target.clamp(1e-4, 0.999)),
                            torch.tensor(-30.0, device="cuda"))
    rgb, alpha, _ = v.render(c2, torch.zeros(3, device="cuda"), override_color=col, mip=False, eps2d=0.05)
    img = rgb + (1 - alpha[..., None])  # over white
    img = F.avg_pool2d(img.permute(2, 0, 1)[None], ss)[0].permute(1, 2, 0)
    return img.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--type", choices=["colmap", "synthetic"], default="colmap")
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--width_px", type=float, default=0.35)
    ap.add_argument("--alpha_gain", type=float, default=1.5)
    ap.add_argument("--alpha_mode", choices=["opacity", "const"], default="opacity")
    ap.add_argument("--const_alpha", type=float, default=0.5)
    ap.add_argument("--darken", type=float, default=0.65)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = SpaghettiModel(device="cuda")
    m.load_state_dict(torch.load(os.path.join(args.run, "model.pt"), map_location="cuda"))
    if args.type == "colmap":
        _, test, _, _ = load_colmap(args.data, args.images)
    else:
        _, test, _, _, _ = load_synthetic(args.data)
    cam = test[args.view] if args.scale == 1.0 else test[args.view].scaled(args.scale)
    img = sketch(m, cam, width_px=args.width_px, alpha_gain=args.alpha_gain, darken=args.darken,
                 alpha_mode=args.alpha_mode, const_alpha=args.const_alpha)
    Image.fromarray(img.mul(255).byte().cpu().numpy()).save(args.out + ".jpg", quality=93)
    print("saved", args.out + ".jpg", img.shape)


if __name__ == "__main__":
    main()
