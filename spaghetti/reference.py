"""Pure-PyTorch reference implementations (slow, differentiable by autograd).

* ``tau_exact_3d``      – paper Eq. (17)-(19): exact ray integral of a Gaussian-convolved 3D segment.
* ``tau_numeric_3d``    – brute-force quadrature of Eq. (13), used to validate the closed form.
* ``gauss_integral`` / ``beta_integral`` – the screen-space 1D integrals evaluated by the CUDA kernel.
* ``render_dense``      – dense per-pixel compositing that mirrors the CUDA rasterizer exactly
                          (used for forward/backward unit tests).
"""
import math

import numpy as np
import torch

SQRT_PI = math.sqrt(math.pi)
GAUSS_KEPS = 1e-4
BETA_QMAX = 4.5
_glx, _glw = np.polynomial.legendre.leggauss(8)
GL_X = [float(x) for x in _glx]
GL_W = [float(w) for w in _glw]


# ----------------------------------------------------------------------------
# exact 3D formulation (Sec. 3.3)
# ----------------------------------------------------------------------------
def tau_exact_3d(o, d, a, b, Sigma, rho=1.0):
    """Optical depth of the Gaussian-convolved segment [a,b] along rays o + t d.

    Kernel is the *unnormalised* Gaussian exp(-x^T Q x), Q = 0.5 Sigma^{-1} (as in the paper).
    o: (3,), d: (...,3) unit directions, a,b: (3,), Sigma: (3,3).
    """
    Q = 0.5 * torch.linalg.inv(Sigma)
    v = b - a
    w = o - a
    Qd = d @ Q  # (...,3) (Q symmetric)
    dQd = (Qd * d).sum(-1)
    # Q_perp = Q - Qd Qd^T / dQd  -> quadratic forms
    vQv = v @ Q @ v
    wQw = w @ Q @ w
    vQw = v @ Q @ w
    vQd = Qd @ v
    wQd = Qd @ w
    k2 = vQv - vQd * vQd / dQd  # kappa^2 = |v|^2_{Q_perp}
    vQpw = vQw - vQd * wQd / dQd
    wQpw = wQw - wQd * wQd / dQd
    point = k2 < 1e-12
    k2s = torch.where(point, torch.ones_like(k2), k2)
    ubar = vQpw / k2s
    e = wQpw - vQpw * vQpw / k2s  # |o - x(ubar)|^2_{Q_perp}
    f_ubar = torch.sqrt(math.pi / dQd) * torch.exp(-e)
    kappa = torch.sqrt(k2s)
    tau_seg = f_ubar * SQRT_PI / (2 * kappa) * (torch.erf(kappa * ubar) + torch.erf(kappa * (1 - ubar)))
    tau_pt = torch.sqrt(math.pi / dQd) * torch.exp(-wQpw)
    return rho * torch.where(point, tau_pt, tau_seg)


def tau_numeric_3d(o, d, a, b, Sigma, rho=1.0, nu=2001, nt=4001, tmax=None):
    """Brute-force double integral of Eq. (13) (float64, trapezoid rule)."""
    Q = 0.5 * torch.linalg.inv(Sigma)
    u = torch.linspace(0, 1, nu, dtype=o.dtype, device=o.device)
    x = a[None] + u[:, None] * (b - a)[None]  # (nu,3)
    if tmax is None:
        tmax = float((torch.stack([a, b]) - o).norm(dim=-1).max()) + 10 * float(torch.linalg.eigvalsh(Sigma).max().sqrt())
    t = torch.linspace(0, tmax, nt, dtype=o.dtype, device=o.device)
    out = []
    for di in d.reshape(-1, 3):
        r = o[None] + t[:, None] * di[None]  # (nt,3)
        diff = r[:, None, :] - x[None, :, :]  # (nt,nu,3)
        q = torch.einsum("tui,ij,tuj->tu", diff, Q, diff)
        f = torch.exp(-q)
        out.append(torch.trapezoid(torch.trapezoid(f, t, dim=0), u))
    return rho * torch.stack(out).reshape(d.shape[:-1])


# ----------------------------------------------------------------------------
# screen-space 1D integrals (mirror the CUDA device functions)
# ----------------------------------------------------------------------------
def erf_sum(A, B):
    """erf(A) + erf(B) for A + B > 0 without cancellation."""
    s1 = torch.erf(A) + torch.erf(B)
    s2 = torch.erfc(-A) - torch.erfc(B)
    s3 = torch.erfc(-B) - torch.erfc(A)
    return torch.where((A >= 0) & (B >= 0), s1, torch.where(A < 0, s2, s3))


def gauss_integral(P, R, K, E=None):
    """I = int_0^1 exp(-(P + 2 R l + K l^2)) dl.  E = P - R^2/K may be given (evaluated stably from vectors)."""
    K = K.clamp_min(0)
    small = K < GAUSS_KEPS
    Ks = torch.where(small, torch.ones_like(K), K)
    kappa = Ks.sqrt()
    A = -R / kappa
    B = kappa - A
    if E is None:
        E = (P - A * A).clamp_min(0)
    I_big = torch.exp(-E) * (0.5 * SQRT_PI / kappa) * erf_sum(A, B)
    I_small = torch.exp(-P) * (1 - R - K / 3 + 2 * R * R / 3)
    return torch.where(small, I_small, I_big)


BETA_KSTABLE = 1.0


@torch.no_grad()
def beta_support(P, R, K, E=None):
    if E is not None:  # mirror the CUDA kernel: stable form for long segments
        lo_s, hi_s, ok_s = _beta_support_stable(R, K, E)
        lo_o, hi_o, ok_o = beta_support(P, R, K)
        st = K >= BETA_KSTABLE
        return torch.where(st, lo_s, lo_o), torch.where(st, hi_s, hi_o), torch.where(st, ok_s, ok_o)
    c0 = P - BETA_QMAX
    degenerate = (K < 1e-12) & (R.abs() < 1e-12)
    disc = R * R - K * c0
    sq = disc.clamp_min(0).sqrt()
    qq = -(R + torch.copysign(sq, R))
    big = torch.full_like(qq, 1e30)
    r1 = torch.where(K > 1e-12, qq / torch.where(K > 1e-12, K, torch.ones_like(K)), torch.where(qq > 0, big, -big))
    safe_qq = torch.where(qq.abs() > 1e-30, qq, torch.ones_like(qq))
    r2 = torch.where(qq.abs() > 1e-30, c0 / safe_qq, torch.where(c0 < 0, -big, big))
    lo = torch.clamp(torch.minimum(r1, r2), min=0.0)
    hi = torch.clamp(torch.maximum(r1, r2), max=1.0)
    ok = (disc > 0) & (hi > lo)
    lo = torch.where(degenerate, torch.zeros_like(lo), lo)
    hi = torch.where(degenerate, torch.ones_like(hi), hi)
    ok = torch.where(degenerate, c0 < 0, ok)
    return lo, hi, ok


@torch.no_grad()
def _beta_support_stable(R, K, E):
    Ks = torch.where(K > 0, K, torch.ones_like(K))
    ls = -R / Ks
    h = ((BETA_QMAX - E).clamp_min(0) / Ks).sqrt()
    lo = (ls - h).clamp(min=0.0)
    hi = (ls + h).clamp(max=1.0)
    return lo, hi, (E < BETA_QMAX) & (hi > lo)


def beta_integral(P, R, K, beta, E=None):
    """I = int_{[0,1]} (1 - 2 q(l) / 9)_+^beta dl via 8-pt Gauss-Legendre on the support."""
    K = K.clamp_min(0)
    lo, hi, ok = beta_support(P, R, K, E)
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    st = (K >= BETA_KSTABLE) if E is not None else torch.zeros_like(K, dtype=torch.bool)
    Ks = torch.where(st, K, torch.ones_like(K))
    ls = -R / Ks
    I = torch.zeros_like(P)
    for x, w in zip(GL_X, GL_W):
        lam = mid + half * x
        q_plain = P + lam * (2 * R + lam * K)
        q_stable = (E if E is not None else P) + K * (lam - ls) ** 2
        q = torch.where(st, q_stable, q_plain)
        base = 1 - (2.0 / 9.0) * q
        pos = base > 0
        f = torch.where(pos, torch.exp(beta * torch.log(torch.where(pos, base, torch.ones_like(base)))), torch.zeros_like(base))
        I = I + half * w * f
    return torch.where(ok, I, torch.zeros_like(I))


def segment_PRK(pix, pa, pb, ca, cb, Q, with_E=False):
    """pix: (M,2) pixel centres; per-primitive tensors broadcast as (N,...) -> returns (M,N) P,R,K[,E]."""
    g0 = (pix[:, None, :] - pa[None]) * ca[None, :, None]
    g1 = (pix[:, None, :] - pb[None]) * cb[None, :, None]
    dl = g1 - g0
    qa, qb, qc = Q[:, 0][None], Q[:, 1][None], Q[:, 2][None]

    def form(x, y):
        return qa * x[..., 0] * y[..., 0] + qb * (x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]) + qc * x[..., 1] * y[..., 1]

    P, R, K = form(g0, g0), form(dl, g0), form(dl, dl)
    if not with_E:
        return P, R, K
    Ks = torch.where(K >= GAUSS_KEPS, K, torch.ones_like(K))
    ub = -R / Ks
    r = g0 + ub[..., None] * dl
    E = torch.where(K >= GAUSS_KEPS, form(r, r).clamp_min(0), torch.zeros_like(K))
    return P, R, K, E


def render_dense(pa, pb, ca, cb, Q, rho, colors, depth, W, H, bg, kernel="gaussian", beta=None,
                 alpha_min=1 / 255, T_min=1e-4, tile_mask=None):
    """Dense reference renderer with the same compositing rules as the CUDA kernel.

    tile_mask: optional (H*W, N) bool of which primitive is binned to which pixel's tile.
    """
    dev = pa.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2).to(pa.dtype) + 0.5
    order = torch.argsort(depth)
    pa, pb, ca, cb, Q, rho, colors = pa[order], pb[order], ca[order], cb[order], Q[order], rho[order], colors[order]
    if beta is not None:
        beta = beta[order]
    if tile_mask is not None:
        tile_mask = tile_mask[:, order]
    P, R, K, E = segment_PRK(pix, pa, pb, ca, cb, Q, with_E=True)
    if kernel == "gaussian":
        I = gauss_integral(P, R, K, E)
    else:
        I = beta_integral(P, R, K.clamp_min(0), beta[None].expand_as(P), E)
    tau = rho[None] * I
    alpha = torch.clamp(1 - torch.exp(-tau), max=0.99)
    keep = (alpha >= alpha_min) & (alpha > 0)
    if tile_mask is not None:
        keep = keep & tile_mask
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    # early termination (T < T_min) as in the CUDA kernel
    one_m = 1 - alpha
    T_incl = torch.cumprod(one_m, dim=1)
    T_excl = torch.cat([torch.ones_like(T_incl[:, :1]), T_incl[:, :-1]], 1)
    stop = (T_incl < T_min) & keep
    first_stop = torch.where(stop.any(1), stop.float().argmax(1), torch.full_like(stop[:, 0], stop.shape[1], dtype=torch.long))
    idx = torch.arange(alpha.shape[1], device=dev)[None]
    alive = idx < first_stop[:, None]
    alpha = torch.where(alive, alpha, torch.zeros_like(alpha))
    T_incl = torch.cumprod(1 - alpha, dim=1)
    T_excl = torch.cat([torch.ones_like(T_incl[:, :1]), T_incl[:, :-1]], 1)
    w = alpha * T_excl
    rgb = w @ colors + T_incl[:, -1:] * bg[None]
    return rgb.reshape(H, W, 3), (1 - T_incl[:, -1]).reshape(H, W)
