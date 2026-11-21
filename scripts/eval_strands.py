"""Strand-geometry F1 (Sec. 4.2 criterion) of a trained model against GT strands of the synthetic scene.

Every primitive with opacity >= --min_opacity contributes a 3D segment: its Dirac support [a, b], or, if that is
shorter than half the kernel's major std (e.g. all primitives of the point baseline), the kernel's major axis
(+-1 std), as Gaussian-Haircut-style methods do. Both sets are resampled at --step and compared with
distance/angle thresholds.

python scripts/eval_strands.py --run runs/synth_gauss --gt data/synth_spaghetti/gt_strands.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spaghetti.metrics import resample_segments, single_polyline_coverage, strand_f1  # noqa: E402
from spaghetti.model import SpaghettiModel  # noqa: E402
from spaghetti.render import quat_to_rotmat  # noqa: E402


def predicted_segments(sd, min_opacity):
    V, S = sd["verts"], sd["segs"]
    o = torch.sigmoid(sd["opacity"])
    keep = o >= min_opacity
    a, b = V[S[keep, 0]], V[S[keep, 1]]
    s = torch.exp(sd["log_scales"][keep])
    Rm = quat_to_rotmat(sd["quats"][keep])
    k = s.argmax(-1)
    axis = Rm[torch.arange(len(k)), :, k]
    smax = s.max(-1).values
    L = (b - a).norm(dim=-1)
    use_axis = L < 0.5 * smax
    mid = 0.5 * (a + b)
    A = torch.where(use_axis[:, None], mid - smax[:, None] * axis, a)
    B = torch.where(use_axis[:, None], mid + smax[:, None] * axis, b)
    return A.numpy(), B.numpy(), float(use_axis.float().mean()), keep.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gt", default="data/synth_spaghetti/gt_strands.npz")
    ap.add_argument("--step", type=float, default=0.002)
    ap.add_argument("--thresholds", default="0.003:10,0.006:20,0.012:30")  # ~0.5/1/2 strand radii
    ap.add_argument("--min_opacity", type=float, default=0.05)
    args = ap.parse_args()

    sd = torch.load(os.path.join(args.run, "model.pt"), map_location="cpu")
    A, B, frac_axis, keep = predicted_segments(sd, args.min_opacity)
    P_pred, T_pred, own = resample_segments(A, B, args.step, return_owner=True)

    gt = np.load(args.gt)
    pts, offs = gt["points"], gt["offsets"]
    GA = np.concatenate([pts[offs[i]:offs[i + 1] - 1] for i in range(len(offs) - 1)])
    GB = np.concatenate([pts[offs[i] + 1:offs[i + 1]] for i in range(len(offs) - 1)])
    GS = np.concatenate([np.full(offs[i + 1] - offs[i] - 1, i) for i in range(len(offs) - 1)])
    P_gt, T_gt, gown = resample_segments(GA, GB, args.step, return_owner=True)

    m = SpaghettiModel(device="cpu")
    m.load_state_dict(sd)
    labels, sizes = m.components()
    pred_label = labels[keep][own]
    res = dict(n_prims_eval=int(len(A)), frac_major_axis=frac_axis, comp_mean=float(sizes.mean()),
               comp_max=int(sizes.max()))
    for th in args.thresholds.split(","):
        d, ang = th.split(":")
        p, r, f = strand_f1(P_pred, T_pred, P_gt, T_gt, float(d), float(ang))
        res[f"P@{d}"], res[f"R@{d}"], res[f"F1@{d}"] = p, r, f
        res[f"cover1@{d}"] = single_polyline_coverage(P_gt, T_gt, GS[gown], P_pred, T_pred, pred_label,
                                                      float(d), float(ang))
    print(json.dumps(res))
    json.dump(res, open(os.path.join(args.run, "strands.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
