"""Collect runs/*/metrics.json (+ strands.json) into markdown tables.

python scripts/summarize.py > results.md
"""
import json
import os

RUNS = "runs"


def load(name):
    d = {}
    for f in ("metrics.json", "strands.json"):
        p = os.path.join(RUNS, name, f)
        if os.path.exists(p):
            d.update(json.load(open(p)))
    return d


def fmt(v, nd):
    return "—" if v is None else f"{v:.{nd}f}"


def table(rows, cols):
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    out = [head, sep]
    for label, name in rows:
        d = load(name)
        if not d:
            continue
        out.append("| " + label + " | " + " | ".join(fmt(d.get(k), nd) for _, k, nd in cols[1:]) + " |")
    return "\n".join(out)


img_cols = [("Method", None, 0), ("PSNR↑", "psnr", 2), ("SSIM↑", "ssim", 3), ("LPIPS↓", "lpips", 3),
            ("FPS↑", "fps", 0), ("#prims", "n_seg", 0), ("mean comp.", "comp_mean", 2), ("train min", "train_minutes", 0)]
strand_cols = [("Method", None, 0), ("PSNR↑", "psnr", 2), ("F1@0.003↑", "F1@0.003", 3), ("F1@0.006↑", "F1@0.006", 3),
               ("F1@0.012↑", "F1@0.012", 3), ("1-polyline cover↑", "cover1@0.006", 3), ("mean comp.", "comp_mean", 2),
               ("max comp.", "comp_max", 0)]

print("### Mip-NeRF 360 — bonsai (images_2, every 8th view held out)\n")
print(table([("Points (3DGS-MCMC-style, same code)", "bonsai_points"), ("Ours, Gaussian kernel", "bonsai_gauss"),
             ("Ours, Beta kernel", "bonsai_beta")], img_cols))
print("\n### Synthetic spaghetti (48 strands, 112 train / 16 test views, 512²)\n")
print(table([("Points (3DGS-MCMC-style, same code)", "synth_points"), ("Ours, Gaussian kernel", "synth_gauss"),
             ("Ours, Beta kernel", "synth_beta")], img_cols))
print()
print(table([("Points (major axis as tangent)", "synth_points"), ("Ours, Gaussian kernel", "synth_gauss"),
             ("Ours, Beta kernel", "synth_beta")], strand_cols))
print("\n### Ablation (synthetic, Gaussian kernel)\n")
print(table([("w/o midpoint split", "synth_abl_nosplit"), ("w/o segment cloning", "synth_abl_noclone"),
             ("w/o polyline merging", "synth_abl_nomerge"), ("w/o attraction loss", "synth_abl_noattract"),
             ("full", "synth_gauss")], strand_cols))
