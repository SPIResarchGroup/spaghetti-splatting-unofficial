"""Correctness tests. Run: python tests/test_spaghetti.py  (or pytest)."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spaghetti import reference as ref  # noqa: E402

DEV = "cuda"
torch.manual_seed(0)


def rand_spd(n=3, lo=0.05, hi=0.4, dtype=torch.float64, device="cpu"):
    A = torch.randn(n, n, dtype=dtype, device=device)
    Qm, _ = torch.linalg.qr(A)
    s = torch.empty(n, dtype=dtype, device=device).uniform_(lo, hi)
    return Qm @ torch.diag(s ** 2) @ Qm.T


# ---------------------------------------------------------------------------
# 1. paper Eq. (17)-(19) against brute-force integration of Eq. (13)
# ---------------------------------------------------------------------------
def test_closed_form_vs_numeric():
    worst = 0.0
    for trial in range(4):
        Sigma = rand_spd()
        a = torch.randn(3, dtype=torch.float64) * 0.5
        b = a + torch.randn(3, dtype=torch.float64) * (0.2 + 0.5 * trial)
        o = torch.tensor([0.3, -0.2, -4.0], dtype=torch.float64)
        # rays aimed at points scattered around the segment
        tgt = a[None] + torch.rand(6, 1, dtype=torch.float64) * (b - a)[None] + 0.25 * torch.randn(6, 3, dtype=torch.float64)
        d = tgt - o
        d = d / d.norm(dim=-1, keepdim=True)
        t_cf = ref.tau_exact_3d(o, d, a, b, Sigma)
        t_nu = ref.tau_numeric_3d(o, d, a, b, Sigma)
        rel = ((t_cf - t_nu).abs() / t_nu.abs().clamp_min(1e-6)).max().item()
        worst = max(worst, rel)
    print(f"[closed form vs numeric] max rel err = {worst:.2e}")
    assert worst < 1e-4


def test_subdivision_exact_3d():
    Sigma = rand_spd()
    a = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    b = torch.tensor([1.0, 0.4, 0.3], dtype=torch.float64)
    m = 0.5 * (a + b)
    o = torch.tensor([0.1, 0.2, -3.0], dtype=torch.float64)
    d = torch.randn(64, 3, dtype=torch.float64) * 0.15 + torch.tensor([0.1, 0.0, 1.0], dtype=torch.float64)
    d = d / d.norm(dim=-1, keepdim=True)
    full = ref.tau_exact_3d(o, d, a, b, Sigma, 1.0)
    halves = ref.tau_exact_3d(o, d, a, m, Sigma, 0.5) + ref.tau_exact_3d(o, d, m, b, Sigma, 0.5)
    err = (full - halves).abs().max().item()
    print(f"[subdivision, exact 3D] max |tau - (tau1+tau2)| = {err:.2e}")
    assert err < 1e-10


def test_point_limit_3d():
    Sigma = rand_spd()
    a = torch.randn(3, dtype=torch.float64)
    o = torch.tensor([0.0, 0.0, -3.0], dtype=torch.float64)
    d = torch.randn(32, 3, dtype=torch.float64) * 0.1 + torch.tensor([0, 0, 1.0], dtype=torch.float64)
    d = d / d.norm(dim=-1, keepdim=True)
    t_pt = ref.tau_exact_3d(o, d, a, a, Sigma)
    t_tiny = ref.tau_exact_3d(o, d, a, a + 1e-5, Sigma)
    t_nu = ref.tau_numeric_3d(o, d, a, a, Sigma, nu=3)
    err = max((t_pt - t_tiny).abs().max().item(), ((t_pt - t_nu).abs() / t_nu.clamp_min(1e-6)).max().item())
    print(f"[point limit a=b == 3D Gaussian] max err = {err:.2e}")
    assert err < 1e-4


# ---------------------------------------------------------------------------
# 2. screen-space 1D integrals
# ---------------------------------------------------------------------------
def _numeric_1d(P, R, K, kern, n=200001):
    lam = torch.linspace(0, 1, n, dtype=torch.float64)
    q = P[:, None] + 2 * R[:, None] * lam[None] + K[:, None] * lam[None] ** 2
    return torch.trapezoid(kern(q), lam, dim=1)


def _rand_PRK(n, kmax):
    g0 = torch.randn(n, 2, dtype=torch.float64) * 1.5
    dl = torch.randn(n, 2, dtype=torch.float64) * torch.rand(n, 1, dtype=torch.float64) * kmax
    P = (g0 * g0).sum(-1)
    R = (g0 * dl).sum(-1)
    K = (dl * dl).sum(-1)
    return P, R, K


def test_gauss_integral():
    worst = 0.0
    for kmax in [1e-3, 0.05, 1.0, 10.0, 60.0]:
        P, R, K = _rand_PRK(200, kmax)
        I = ref.gauss_integral(P, R, K)
        In = _numeric_1d(P, R, K, lambda q: torch.exp(-q))
        worst = max(worst, ((I - In).abs() / (In.abs() + 1e-6)).max().item())
    print(f"[gaussian closed-form 1D integral] max rel err = {worst:.2e}")
    assert worst < 1e-5


def test_beta_integral():
    worst = 0.0
    for beta in [2.0, 4.5, 12.0]:
        for kmax in [1e-3, 0.5, 5.0, 40.0]:
            P, R, K = _rand_PRK(200, kmax)
            I = ref.beta_integral(P, R, K, torch.full_like(P, beta))
            In = _numeric_1d(P, R, K, lambda q: (1 - 2 * q / 9).clamp_min(0) ** beta)
            worst = max(worst, (I - In).abs().max().item())
    print(f"[beta GL8 1D integral] max abs err = {worst:.2e}")
    assert worst < 2e-3


# ---------------------------------------------------------------------------
# 3. CUDA rasterizer == dense reference (forward and backward)
# ---------------------------------------------------------------------------
def _random_screen_prims(N, W, H, kernel, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    pa = torch.stack([r(N) * W, r(N) * H], -1)
    L = torch.where(r(N) < 0.2, torch.zeros(N, dtype=torch.float64), r(N) * 0.6 * W)  # 20% points
    ang = r(N) * 2 * math.pi
    pb = pa + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * L[:, None]
    ca = 0.6 + r(N)
    cb = 0.6 + r(N)
    s1, s2, th = 1.0 + 5 * r(N), 1.0 + 3 * r(N), r(N) * math.pi
    c, s = torch.cos(th), torch.sin(th)
    sxx = c * c * s1 ** 2 + s * s * s2 ** 2
    syy = s * s * s1 ** 2 + c * c * s2 ** 2
    sxy = c * s * (s1 ** 2 - s2 ** 2)
    det = sxx * syy - sxy ** 2
    Q = 0.5 * torch.stack([syy, -sxy, sxx], -1) / det[:, None]
    rho = 0.05 + 2.5 * r(N)
    col = r(N, 3)
    depth = 1 + r(N) * 10
    beta = 1.0 + 8 * r(N) if kernel == "beta" else None
    return pa, pb, ca, cb, Q, rho, col, depth, beta


def _cuda_vs_dense(kernel):
    from spaghetti.render import rasterize
    W, H, N = 61, 45, 40
    prims = _random_screen_prims(N, W, H, kernel)
    pa, pb, ca, cb, Q, rho, col, depth, beta = prims
    bg = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    leaves64 = [x.clone().requires_grad_(True) for x in (pa, pb, ca, cb, Q, rho, col)]
    beta64 = beta.clone().requires_grad_(True) if beta is not None else None
    rgb_r, a_r = ref.render_dense(*leaves64, depth, W, H, bg, kernel=kernel, beta=beta64, alpha_min=0.0, T_min=1e-20)
    gy_rgb = torch.randn(H, W, 3, dtype=torch.float64)
    gy_a = torch.randn(H, W, dtype=torch.float64)
    ((rgb_r * gy_rgb).sum() + (a_r * gy_a).sum()).backward()

    leaves32 = [x.detach().float().to(DEV).requires_grad_(True) for x in (pa, pb, ca, cb, Q, rho, col)]
    beta32 = beta.detach().float().to(DEV).requires_grad_(True) if beta is not None else None
    radii = torch.full((N, 2), 1e4, device=DEV)
    rgb_c, a_c, _ = rasterize(*leaves32, depth.float().to(DEV), radii, W, H, bg.float().to(DEV), kernel=kernel,
                              beta=beta32, alpha_min=0.0, T_min=1e-20)
    ((rgb_c * gy_rgb.float().to(DEV)).sum() + (a_c * gy_a.float().to(DEV)).sum()).backward()

    fwd = max((rgb_c.double().cpu() - rgb_r.detach()).abs().max().item(), (a_c.double().cpu() - a_r.detach()).abs().max().item())
    names = ["pa", "pb", "ca", "cb", "Q", "rho", "color"]
    pairs = list(zip(names, leaves64, leaves32))
    if beta is not None:
        pairs.append(("beta", beta64, beta32))
    report = {}
    for nm, l64, l32 in pairs:
        g_ref = l64.grad
        g_cu = l32.grad.double().cpu()
        report[nm] = ((g_cu - g_ref).abs().max() / g_ref.abs().max().clamp_min(1e-12)).item()
    return fwd, report


def test_cuda_gaussian():
    fwd, rep = _cuda_vs_dense("gaussian")
    print(f"[CUDA vs dense, gaussian] fwd max abs err = {fwd:.2e}; grad rel err: " + ", ".join(f"{k}={v:.1e}" for k, v in rep.items()))
    assert fwd < 1e-4 and max(rep.values()) < 5e-3


def test_cuda_beta():
    fwd, rep = _cuda_vs_dense("beta")
    print(f"[CUDA vs dense, beta]     fwd max abs err = {fwd:.2e}; grad rel err: " + ", ".join(f"{k}={v:.1e}" for k, v in rep.items()))
    assert fwd < 1e-4 and max(rep.values()) < 5e-3


# ---------------------------------------------------------------------------
# 4. projection: a = b reproduces 3DGS (checked against gsplat)
# ---------------------------------------------------------------------------
def _toy_camera(W=320, H=240):
    from spaghetti.render import Camera
    ang = 0.3
    R = torch.tensor([[math.cos(ang), 0, -math.sin(ang)], [0, 1, 0], [math.sin(ang), 0, math.cos(ang)]], device=DEV)
    t = torch.tensor([0.1, -0.05, 4.0], device=DEV)
    return Camera(R, t, 300.0, 310.0, W / 2 + 3, H / 2 - 2, W, H)


def test_point_limit_matches_gsplat():
    try:
        import gsplat
    except ImportError:
        print("[gsplat] not installed, skipped")
        return
    from spaghetti.render import build_covariance, project_segments
    cam = _toy_camera()
    N = 500
    mu = torch.randn(N, 3, device=DEV) * 0.7
    ls = torch.log(torch.rand(N, 3, device=DEV) * 0.08 + 0.01)
    q = torch.randn(N, 4, device=DEV)
    q = q / q.norm(dim=-1, keepdim=True)
    Sigma, _ = build_covariance(ls, q)
    pr = project_segments(mu, mu, Sigma, cam, eps2d=0.1)
    view = torch.eye(4, device=DEV)
    view[:3, :3], view[:3, 3] = cam.R, cam.t
    Kmat = torch.tensor([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]], device=DEV)
    out = gsplat.fully_fused_projection(mu, None, q, torch.exp(ls), view[None], Kmat[None], cam.W, cam.H,
                                        eps2d=0.1, calc_compensations=True)
    radii, means2d, depths, conics, comps = out
    m = (radii[0] > 0) & pr["valid"]
    if radii.dim() == 3:
        m = (radii[0].min(-1).values > 0) & pr["valid"]
    e_mean = (means2d[0][m] - pr["pa"][m]).abs().max().item()
    e_con = ((conics[0][m] - 2 * pr["Q"][m]).abs() / conics[0][m].abs().max()).max().item()
    e_cmp = (comps[0][m] - pr["comp2d"][m]).abs().max().item()
    print(f"[a=b vs gsplat 3DGS projection] means2d {e_mean:.1e}px, conic rel {e_con:.1e}, mip comp {e_cmp:.1e}")
    assert e_mean < 1e-3 and e_con < 1e-4 and e_cmp < 1e-4


# ---------------------------------------------------------------------------
# 5. EWA screen-space approximation vs exact 3D ray integral, and subdivision in screen space
# ---------------------------------------------------------------------------
def _screen_tau(a, b, Sigma, cam, pix):
    from spaghetti.render import project_segments
    pr = project_segments(a[None], b[None], Sigma[None], cam, eps2d=0.0, mip2d=False)
    P, R, K = ref.segment_PRK(pix, pr["pa"], pr["pb"], pr["ca"], pr["cb"], pr["Q"])
    return ref.gauss_integral(P, R, K)[:, 0]


def test_ewa_vs_exact():
    cam = _toy_camera(160, 120)
    cam.R = cam.R.double()
    cam.t = cam.t.double()
    ys, xs = torch.meshgrid(torch.arange(cam.H, device=DEV), torch.arange(cam.W, device=DEV), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2).double() + 0.5
    dirs_c = torch.stack([(pix[:, 0] - cam.cx) / cam.fx, (pix[:, 1] - cam.cy) / cam.fy, torch.ones_like(pix[:, 0])], -1)
    dirs = dirs_c @ cam.R  # camera -> world (R^T applied on the right)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    o = cam.center
    lines = []
    for name, a, b in [
        ("fronto-parallel", [-0.6, 0.0, 0.0], [0.6, 0.1, 0.0]),
        ("receding (z ratio ~2)", [-0.3, 0.0, -1.2], [0.4, 0.1, 1.6]),
        ("toward camera", [0.05, 0.0, -1.0], [0.0, 0.0, 1.0]),
    ]:
        a = torch.tensor(a, dtype=torch.float64, device=DEV)
        b = torch.tensor(b, dtype=torch.float64, device=DEV)
        Sigma = torch.diag(torch.tensor([0.03, 0.02, 0.025], dtype=torch.float64, device=DEV) ** 2)
        exact = ref.tau_exact_3d(o, dirs, a, b, Sigma)
        # the screen-space kernel drops the per-ray normalisation sqrt(pi / d'Qd) (absorbed in rho, as in 3DGS)
        Qm = 0.5 * torch.linalg.inv(Sigma)
        dm = 0.5 * (a + b) - o
        dm = dm / dm.norm()
        exact = exact / math.sqrt(math.pi / float(dm @ Qm @ dm))
        approx = _screen_tau(a, b, Sigma, cam, pix)
        err = (approx - exact).abs().max().item() / exact.abs().max().item()
        lines.append(f"{name}: {err:.2%}")
    print("[EWA screen-space vs exact 3D, max err / peak] " + "; ".join(lines))


def test_subdivision_screen():
    from spaghetti.render import build_covariance, project_segments, rasterize, screen_radii
    cam = _toy_camera()
    a = torch.tensor([[-0.8, -0.2, -0.5]], device=DEV)
    b = torch.tensor([[0.9, 0.3, 0.8]], device=DEV)
    ls = torch.log(torch.tensor([[0.04, 0.03, 0.05]], device=DEV))
    q = torch.tensor([[1.0, 0.2, 0.1, 0.0]], device=DEV)
    rho = torch.tensor([3.0], device=DEV)
    col = torch.tensor([[0.9, 0.7, 0.2]], device=DEV)
    bg = torch.zeros(3, device=DEV)

    def render(A, B, n):
        Sigma, _ = build_covariance(ls.expand(n, 3), q.expand(n, 4))
        pr = project_segments(A, B, Sigma, cam)
        r = (rho / n).expand(n) * pr["comp2d"]
        rad = screen_radii(pr, r, "gaussian")
        return rasterize(pr["pa"], pr["pb"], pr["ca"], pr["cb"], pr["Q"], r, col.expand(n, 3), pr["depth"], rad,
                         cam.W, cam.H, bg)[1]

    full = render(a, b, 1)
    res = []
    for n in [2, 4, 8]:
        lam = torch.linspace(0, 1, n + 1, device=DEV)[:, None]
        pts = a + lam * (b - a)
        sub = render(pts[:-1], pts[1:], n)
        res.append(f"{n} pieces: {(sub - full).abs().max().item():.1e}")
    print("[subdivision in screen space, max |alpha diff|] " + ", ".join(res))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
