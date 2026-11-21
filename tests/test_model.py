"""Topology operations keep the rendering (nearly) unchanged and the optimizer state consistent."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spaghetti.model import SpaghettiModel  # noqa: E402
from spaghetti.render import Camera  # noqa: E402

DEV = "cuda"


def make_model(n=60, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3)).astype(np.float32) * 0.5
    m = SpaghettiModel(sh_degree=1, device=DEV)
    m.init_from_points(pts, rng.uniform(size=(n, 3)), seg_len_factor=0.5)
    with torch.no_grad():  # make half of them long
        a, b = m.segs[:, 0], m.segs[:, 1]
        d = m.verts[b] - m.verts[a]
        m.verts[b[: n // 2]] += 6 * d[: n // 2]
        m.log_scales.fill_(np.log(0.03))
        m.opacity.fill_(-0.5)  # keep tau below the 0.99 alpha clamp (the clamp is not split-invariant)
    return m


def cam():
    return Camera(torch.eye(3, device=DEV), torch.tensor([0.0, 0.0, 4.0], device=DEV), 400.0, 400.0, 160.0, 120.0, 320, 240)


def check_opt(m, opt):
    for g in opt.param_groups:
        p = g["params"][0]
        st = opt.state.get(p)
        assert getattr(m, g["name"]) is p
        if st:
            assert st["exp_avg"].shape == p.shape, g["name"]


def test_spawn_invariance():
    m = make_model()
    opt = m.make_optimizer({k: 1e-3 for k in ["verts", "log_scales", "quats", "opacity", "sh_dc", "sh_rest", "beta"]})
    # one step to create optimizer state
    rgb, _, _ = m.render(cam(), torch.zeros(3, device=DEV))
    rgb.sum().backward()
    opt.step()
    opt.zero_grad()
    bg = torch.zeros(3, device=DEV)
    with torch.no_grad():
        before = m.render(cam(), bg)[0]
    tg = torch.tensor([0, 0, 1, 2, 2, 2, 40, 41, 41], device=DEV)  # long (0,1,2) and short (40,41) targets
    ns0 = m.num_segments
    m.spawn(opt, tg)
    check_opt(m, opt)
    assert m.num_segments == ns0 + len(tg)
    with torch.no_grad():
        after = m.render(cam(), bg)[0]
    err = (after - before).abs().max().item()
    _, sizes = m.components()
    # scene level: pieces are depth-sorted individually, so their order relative to *other* overlapping
    # segments can change -> not exactly invariant (inherent to per-primitive sorting); report only.
    print(f"[spawn in a scene] max |rgb diff| = {err:.2e} (re-sorting w.r.t. overlapping segments); "
          f"max component size {sizes.max()}")
    # isolated segment: split into n+1 pieces / clone n times must leave the image unchanged
    worst = 0.0
    for idx, n in [(0, 1), (1, 3), (2, 2), (40, 1), (41, 3)]:
        torch.manual_seed(0)
        m1 = make_model()
        keep = torch.zeros(m1.num_segments, dtype=torch.bool, device=DEV)
        keep[idx] = True
        m1.remove(None, ~keep)
        with torch.no_grad():
            r0 = m1.render(cam(), bg)[1]
        m1.spawn(None, torch.zeros(n, dtype=torch.long, device=DEV))
        with torch.no_grad():
            r1 = m1.render(cam(), bg)[1]
        worst = max(worst, (r1 - r0).abs().max().item())
    print(f"[spawn, isolated segment] max |alpha diff| = {worst:.2e}")
    assert worst < 1.5e-2  # EWA is linearised at each piece's own midpoint (~1% effect)
    # remove + gc
    mask = torch.zeros(m.num_segments, dtype=torch.bool, device=DEV)
    mask[5:15] = True
    m.remove(opt, mask)
    check_opt(m, opt)
    used = torch.zeros(m.num_vertices, dtype=torch.bool, device=DEV)
    used[m.segs.reshape(-1)] = True
    assert bool(used.all())
    # merge: put two endpoints of different polylines on top of each other, in continuation (pointing apart)
    ends, own, sig = m._endpoint_info()
    oth = m._end_other
    with torch.no_grad():
        V = m.verts
        d0 = V[ends[0]] - V[oth[0]]
        dk = V[ends] - V[oth]
        lab, _ = m.components()
        lab = torch.as_tensor(lab, device=DEV)
        ok = ((dk @ d0) < 0) & (lab[own] != lab[own[0]])
        k1 = int(ok.nonzero()[0, 0])
        shift = V[ends[0]] + 1e-4 * d0 / d0.norm() - V[ends[k1]]  # move the whole segment rigidly
        V[ends[k1]] += shift
        V[oth[k1]] += shift
    nv = m.num_vertices
    k = m.merge_endpoints(opt, 0.5)
    check_opt(m, opt)
    assert k >= 1 and m.num_vertices == nv - k
    ends, own, sig = m._endpoint_info()
    oth = m._end_other
    with torch.no_grad():  # a continuation pair at ~1.7 sigma: not merged, but attracted
        V = m.verts
        d0 = V[ends[0]] - V[oth[0]]
        dk = V[ends] - V[oth]
        lab, _ = m.components()
        lab = torch.as_tensor(lab, device=DEV)
        k1 = int((((dk @ d0) < 0) & (lab[own] != lab[own[0]])).nonzero()[0, 0])
        shift = V[ends[0]] + 0.05 * d0 / d0.norm() - V[ends[k1]]
        V[ends[k1]] += shift
        V[oth[k1]] += shift
    n = m.update_attraction_pairs(3.0)
    assert n >= 2
    loss = m.attraction_loss()
    loss.backward()
    print(f"[merge] merged {k} pairs, attraction pairs {n}, loss {float(loss):.3f}")


def test_split_exact_3d():
    """Paper (exact subdivision): splitting a segment into pieces that inherit the density per unit length keeps
    the optical depth along every ray, tau_parent = sum_k tau_k, hence the same transmittance.
    Checked with the exact 3D ray integral (Eq. 17-19) on the pieces produced by model.spawn."""
    from spaghetti import reference as ref
    from spaghetti.model import mass_weight
    from spaghetti.render import build_covariance
    import torch.nn.functional as F

    torch.manual_seed(0)
    m = make_model()
    keep = torch.zeros(m.num_segments, dtype=torch.bool, device=DEV)
    keep[0] = True
    m.remove(None, ~keep)

    def taus(o, d):
        a, b = m.endpoints()
        Sig, _ = build_covariance(m.log_scales, m.quats)
        rho = F.softplus(m.opacity) * mass_weight(a, b, m.log_scales, m.quats)
        out = 0
        for k in range(m.num_segments):
            out = out + ref.tau_exact_3d(o, d, a[k].double(), b[k].double(), Sig[k].double(), rho[k].double())
        return out

    with torch.no_grad():
        a, b = m.endpoints()
        o = torch.tensor([0.0, 0.0, -4.0], dtype=torch.float64, device=DEV)
        tgt = a[0].double() + torch.rand(64, 1, dtype=torch.float64, device=DEV) * (b[0] - a[0]).double() \
            + 0.05 * torch.randn(64, 3, dtype=torch.float64, device=DEV)
        d = F.normalize(tgt - o, dim=-1)
        t0 = taus(o, d)
        m.spawn(None, torch.zeros(3, dtype=torch.long, device=DEV))  # 1 segment -> 4 connected pieces
        t1 = taus(o, d)
    err = ((t1 - t0).abs() / t0.abs().clamp_min(1e-8)).max().item()
    print(f"[exact subdivision, 1 -> {m.num_segments} pieces via spawn, exact 3D tau] max rel err = {err:.1e}")
    assert m.num_segments == 4 and err < 1e-5


def test_relocate_point_mode():
    rng = np.random.default_rng(1)
    m = SpaghettiModel(sh_degree=0, device=DEV)
    m.init_from_points(rng.normal(size=(100, 3)).astype(np.float32), rng.uniform(size=(100, 3)), point_mode=True)
    opt = m.make_optimizer({k: 1e-3 for k in ["verts", "log_scales", "quats", "opacity", "sh_dc", "sh_rest", "beta"]})
    with torch.no_grad():
        m.opacity[:30] = -10
    n = m.relocate(opt)
    check_opt(m, opt)
    assert n == 30 and m.num_segments == 100
    assert bool((m.segs[:, 0] == m.segs[:, 1]).all()) and m.num_vertices == 100
    m.grow(opt, cap=120)
    assert m.num_segments == 105 and m.num_vertices == 105
    print("[point mode relocate/grow] ok")


if __name__ == "__main__":
    test_spawn_invariance()
    test_split_exact_3d()
    test_relocate_point_mode()
    print("model tests passed")
