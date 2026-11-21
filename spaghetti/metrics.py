import math

import numpy as np
import torch
import torch.nn.functional as F


def _gauss_window(size=11, sigma=1.5, device="cuda"):
    g = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)


_WIN = {}


def ssim(img1, img2):
    """img: (H,W,3) in [0,1]; standard 11x11 Gaussian SSIM as used by 3DGS."""
    x = img1.permute(2, 0, 1)[None]
    y = img2.permute(2, 0, 1)[None]
    key = x.device
    if key not in _WIN:
        _WIN[key] = _gauss_window(device=x.device)
    w = _WIN[key]
    mu1 = F.conv2d(x, w, padding=5, groups=3)
    mu2 = F.conv2d(y, w, padding=5, groups=3)
    s11 = F.conv2d(x * x, w, padding=5, groups=3) - mu1 ** 2
    s22 = F.conv2d(y * y, w, padding=5, groups=3) - mu2 ** 2
    s12 = F.conv2d(x * y, w, padding=5, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return m.mean()


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return -10 * math.log10(max(float(mse), 1e-12))


_LPIPS = {}


def lpips(img1, img2, convention="standard"):
    """LPIPS-VGG. 'standard': inputs mapped to [-1,1] as LPIPS expects. '3dgs': [0,1] images fed directly,
    which is what the 3DGS / 3DGS-MCMC evaluation scripts effectively do (numbers differ by ~0.01-0.04)."""
    if convention not in _LPIPS:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        _LPIPS[convention] = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=(convention == "standard")).to(img1.device).eval()
    x = img1.permute(2, 0, 1)[None].clamp(0, 1)
    y = img2.permute(2, 0, 1)[None].clamp(0, 1)
    with torch.no_grad():
        return float(_LPIPS[convention](x, y))


# ---------------------------------------------------------------------------
# strand geometry: precision / recall / F1 with distance + tangent-angle thresholds
# (criterion of Takimoto et al. / Neural Haircut, as used in the paper's Sec. 4.2)
# ---------------------------------------------------------------------------
def resample_segments(A, B, step, return_owner=False):
    """Resample segments [A,B] at spacing `step`. Returns points and unit tangents (and owning segment)."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    L = np.linalg.norm(B - A, axis=1)
    n = np.maximum(1, np.ceil(L / step).astype(int))
    own = np.repeat(np.arange(len(A)), n)
    start = np.cumsum(n) - n
    k = np.arange(n.sum()) - start[own]
    t = (k + 0.5) / n[own]
    P = A[own] + t[:, None] * (B - A)[own]
    T = (B - A)[own] / np.maximum(L[own], 1e-12)[:, None]
    if return_owner:
        return P, T, own
    return P, T


def _matches(P1, T1, P2, T2, dist, cos_t, chunk=50000):
    """For each point of set 1: index of the nearest point of set 2 within `dist` whose tangent line is within
    the angle threshold, or -1. Exact radius search (no k-NN truncation), chunked to bound memory."""
    import itertools

    from scipy.spatial import cKDTree

    tree = cKDTree(P2)
    best = np.full(len(P1), -1, dtype=np.int64)
    for s0 in range(0, len(P1), chunk):
        nb = tree.query_ball_point(P1[s0:s0 + chunk], r=dist, workers=-1)
        lens = np.fromiter((len(x) for x in nb), dtype=np.int64, count=len(nb))
        tot = int(lens.sum())
        if tot == 0:
            continue
        flat = np.fromiter(itertools.chain.from_iterable(nb), dtype=np.int64, count=tot)
        own = np.repeat(np.arange(len(nb)), lens) + s0
        ok = np.abs((T2[flat] * T1[own]).sum(-1)) >= cos_t
        d = np.where(ok, np.linalg.norm(P2[flat] - P1[own], axis=1), np.inf)
        order = np.lexsort((d, own))
        own_s, flat_s, d_s = own[order], flat[order], d[order]
        first = np.ones(len(own_s), dtype=bool)
        first[1:] = own_s[1:] != own_s[:-1]
        sel = first & np.isfinite(d_s)
        best[own_s[sel]] = flat_s[sel]
    return best


def single_polyline_coverage(gt_P, gt_T, gt_sid, pred_P, pred_T, pred_label, dist, angle_deg):
    """Mean over GT strands of the fraction of the strand matched by ONE predicted connected component
    (the component that matches most of it). 1.0 = every strand recovered as a single connected polyline."""
    m = _matches(gt_P, gt_T, pred_P, pred_T, dist, math.cos(math.radians(angle_deg)))
    cov = []
    for s in np.unique(gt_sid):
        sel = gt_sid == s
        lab = pred_label[m[sel][m[sel] >= 0]]
        cov.append(np.bincount(lab).max() / sel.sum() if len(lab) else 0.0)
    return float(np.mean(cov))


def strand_f1(pred_P, pred_T, gt_P, gt_T, dist, angle_deg):
    """Precision / recall / F1: a point is correct if the other set has a point within `dist` whose tangent
    line differs by less than `angle_deg` (Takimoto et al.; paper Sec. 4.2)."""
    if len(pred_P) == 0:
        return 0.0, 0.0, 0.0
    cos_t = math.cos(math.radians(angle_deg))
    prec = float((_matches(pred_P, pred_T, gt_P, gt_T, dist, cos_t) >= 0).mean())
    rec = float((_matches(gt_P, gt_T, pred_P, pred_T, dist, cos_t) >= 0).mean())
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1
