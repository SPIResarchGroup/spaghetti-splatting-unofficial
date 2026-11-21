"""Polyline (spaghetti) representation: shared vertices + per-segment kernels.

Topology: ``verts`` (Nv,3) and ``segs`` (Ns,2) index pairs. Connected segments share a vertex,
so a polyline is a chain of segments. Every segment carries its own kernel (scale, rotation),
opacity, appearance (SH) and, for the Beta kernel, a shape parameter.

Density convention. The paper's segment measure (Eq. 6) has unit mass, m = int_0^1 delta(x - x(u)) du, so a
longer segment spreads the same mass more thinly (lateral optical depth ~ 1/length). We keep that measure but
parameterise its density per unit *kernel length*:

    rho_eff = softplus(x) * w,   w = 1 + L / (sqrt(2 pi) * sigma_along)

(L = |b - a|, sigma_along = kernel std along the segment). w = 1 for a point (exact 3DGS limit), and for a
long segment the peak alpha across it is ~ o = sigmoid(x), independent of its length — so the optimiser has no
incentive to shrink segments to concentrate mass, and "children inherit the density of the parent" (Sec. 3.6)
holds: splitting into n pieces keeps sum(rho_eff) = rho_eff_parent with rho_child -> rho_parent for long
segments. Clones get rho / n, which is exactly invariant (alpha = 1 - exp(-tau) composes additively).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .render import build_covariance, eval_sh, project_segments, quat_to_rotmat, rasterize, screen_radii

SEG_ATTRS = ["log_scales", "quats", "opacity", "sh_dc", "sh_rest", "beta"]
SQRT_2PI = math.sqrt(2 * math.pi)


def mass_weight(a, b, log_scales, quats):
    """w = 1 + |b - a| / (sqrt(2 pi) sigma_along), sigma_along^2 = v^T Sigma v / |v|^2 (kernel without filters).
    Written as |v|^2 / sqrt(v^T Sigma v) so that it is differentiable (gradient 0) at zero length."""
    v = b - a
    proj = (quat_to_rotmat(quats).transpose(1, 2) @ v[..., None]).squeeze(-1)
    vSv = (proj * proj * torch.exp(2 * log_scales)).sum(-1)
    return 1 + (v * v).sum(-1) / (SQRT_2PI * torch.sqrt(vSv + 1e-30))


def rotmat_to_quat(Rm):
    """(N,3,3) proper rotations -> (N,4) quaternions (w,x,y,z), numerically robust branch selection."""
    m = Rm
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.empty(m.shape[0], 4, device=m.device, dtype=m.dtype)
    c0 = t > 0
    c1 = (~c0) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    c2 = (~c0) & (~c1) & (m[:, 1, 1] >= m[:, 2, 2])
    c3 = ~(c0 | c1 | c2)
    s = torch.sqrt((t + 1).clamp_min(1e-12)) * 2
    q[c0] = torch.stack([0.25 * s, (m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s,
                         (m[:, 1, 0] - m[:, 0, 1]) / s], -1)[c0]
    s = torch.sqrt((1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]).clamp_min(1e-12)) * 2
    q[c1] = torch.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, 0.25 * s, (m[:, 0, 1] + m[:, 1, 0]) / s,
                         (m[:, 0, 2] + m[:, 2, 0]) / s], -1)[c1]
    s = torch.sqrt((1 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]).clamp_min(1e-12)) * 2
    q[c2] = torch.stack([(m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 0, 1] + m[:, 1, 0]) / s, 0.25 * s,
                         (m[:, 1, 2] + m[:, 2, 1]) / s], -1)[c2]
    s = torch.sqrt((1 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]).clamp_min(1e-12)) * 2
    q[c3] = torch.stack([(m[:, 1, 0] - m[:, 0, 1]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s,
                         (m[:, 1, 2] + m[:, 2, 1]) / s, 0.25 * s], -1)[c3]
    return q / q.norm(dim=-1, keepdim=True)


MAX_CLONE_DIV = 51.0  # 3DGS-MCMC's N_max for relocation copies


def inverse_softplus(y):
    y = y.clamp_min(1e-10)
    return y + torch.log(-torch.expm1(-y))


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def rgb2sh(rgb):
    return (rgb - 0.5) / 0.28209479177387814


# Spherical Voronoi appearance (Di Sario et al., CVPR 2026), as in their 3DGS-MCMC release: K sites per primitive,
# colour(d) = relu(sum_k softmax_k(-tau_k * |s_k/|s_k| - d|) c_k + 0.5), tau_k = exp(t_k)
SV_ATTRS = ["sv_sites", "sv_tau", "sv_colors"]
SH_ATTRS = ["sh_dc", "sh_rest"]


def fibonacci_sphere(k, device=None):
    i = torch.arange(k, dtype=torch.float32, device=device) + 0.5
    y = 1 - 2 * i / k
    r = torch.sqrt((1 - y * y).clamp_min(0))
    phi = math.pi * (3 - math.sqrt(5)) * i
    return torch.stack([torch.cos(phi) * r, y, torch.sin(phi) * r], -1)


def sv_color(sites, tau, colors, dirs):
    """sites (N,K,3) unnormalised, tau (N,K) log-temperatures, colors (N,K,3), dirs (N,3) unit view directions."""
    s = F.normalize(sites, dim=-1)
    # |s - d| for unit vectors, written via the dot product (finite gradient at s = d)
    dist = torch.sqrt((2 - 2 * (s * dirs[:, None, :]).sum(-1)).clamp_min(1e-12))
    w = torch.softmax(-torch.exp(tau) * dist, dim=-1)
    return F.relu((w[..., None] * colors).sum(1) + 0.5)


class SpaghettiModel:
    def __init__(self, sh_degree=3, kernel="gaussian", device="cuda", appearance="sh", sv_sites=8):
        self.sh_degree = sh_degree
        self.active_sh = 0
        self.kernel = kernel
        self.device = device
        self.filter3d = None
        self.split_len_factor = 2.0
        self.attract_pairs = None
        self.point_mode = False
        self.iteration = 0  # set by the training loop
        self.last_contrib = None  # (Ns,) last iteration with a nonzero photometric gradient
        self.seg_attrs = list(SEG_ATTRS)  # per-segment parameters (subclasses may add more)
        self.extra_vert = []  # extra scalar per-vertex parameters (for subclasses)
        self.appearance, self.sv_k = "sh", sv_sites
        self._set_appearance(appearance, sv_sites)

    @property
    def color_attrs(self):
        return SV_ATTRS if self.appearance == "sv" else SH_ATTRS

    def _set_appearance(self, appearance, k=8):
        """Swap the appearance parameters in seg_attrs ("sh": SH coefficients, "sv": Spherical Voronoi, k sites)."""
        assert appearance in ("sh", "sv"), appearance
        old = self.color_attrs
        i = self.seg_attrs.index(old[0])
        self.seg_attrs = [a for a in self.seg_attrs if a not in old]
        self.appearance, self.sv_k = appearance, k
        self.seg_attrs[i:i] = self.color_attrs

    def view_colors(self, dirs):
        """Per-segment RGB (>= 0, unclamped above) for unit view directions dirs (camera -> segment)."""
        if self.appearance == "sv":
            return sv_color(self.sv_sites, self.sv_tau, self.sv_colors, dirs)
        sh = torch.cat([self.sh_dc, self.sh_rest], 1)
        return torch.clamp_min(eval_sh(self.active_sh, sh, dirs) + 0.5, 0.0)

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def init_from_points(self, pts, rgb, seg_len_factor=0.5, init_opacity=0.5, point_mode=False):
        from scipy.spatial import cKDTree

        pts = np.asarray(pts, dtype=np.float32)
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=4)
        scale = np.sqrt(np.clip((d[:, 1:] ** 2).mean(1), 1e-7, None)).astype(np.float32)
        N = pts.shape[0]
        dev = self.device
        P = torch.from_numpy(pts).to(dev)
        s = torch.from_numpy(scale).to(dev)
        self.point_mode = point_mode
        if point_mode:
            verts = P
            segs = torch.stack([torch.arange(N), torch.arange(N)], 1)
        else:
            dirs = F.normalize(torch.randn(N, 3, device=dev), dim=-1)
            half = 0.5 * seg_len_factor * s[:, None] * dirs
            verts = torch.cat([P - half, P + half], 0)
            segs = torch.stack([torch.arange(N), torch.arange(N) + N], 1)
        self.verts = verts.contiguous().requires_grad_(True)
        self.segs = segs.to(dev).long()
        self.log_scales = torch.log(s)[:, None].repeat(1, 3).contiguous().requires_grad_(True)
        q = torch.zeros(N, 4, device=dev)
        q[:, 0] = 1
        self.quats = q.requires_grad_(True)
        self.opacity = inverse_sigmoid(torch.full((N,), init_opacity, device=dev)).requires_grad_(True)
        rgb = torch.as_tensor(np.asarray(rgb, dtype=np.float32), device=dev)
        if self.appearance == "sv":
            # SV release init: Fibonacci sites shared by all primitives, tau = exp(0) = 1, every site's colour =
            # point colour - 0.5 (so the initial colour is the point colour in every direction)
            K = self.sv_k
            self.sv_sites = fibonacci_sphere(K, dev)[None].repeat(N, 1, 1).contiguous().requires_grad_(True)
            self.sv_tau = torch.zeros(N, K, device=dev).requires_grad_(True)
            self.sv_colors = (rgb.clamp(0, 1) - 0.5)[:, None, :].repeat(1, K, 1).contiguous().requires_grad_(True)
        else:
            self.sh_dc = rgb2sh(rgb)[:, None, :].contiguous().requires_grad_(True)
            nrest = (self.sh_degree + 1) ** 2 - 1
            self.sh_rest = torch.zeros(N, nrest, 3, device=dev).requires_grad_(True)
        self.beta = torch.zeros(N, device=dev).requires_grad_(True)
        self.last_contrib = torch.zeros(N, dtype=torch.long, device=dev)

    # ------------------------------------------------------------- accessors
    @property
    def num_segments(self):
        return self.segs.shape[0]

    @property
    def num_vertices(self):
        return self.verts.shape[0]

    def params(self):
        return {"verts": self.verts, **{k: getattr(self, k) for k in self.extra_vert},
                **{k: getattr(self, k) for k in self.seg_attrs}}

    def _vertex_features(self):
        """(Nv, 3 + #extra) positions and extra vertex attributes, interpolated/copied together by topology ops."""
        cols = [self.verts.detach()] + [getattr(self, k).detach()[:, None] for k in self.extra_vert]
        return torch.cat(cols, 1)

    def _seg_mass_weight(self, idx):
        """Mass weight w of segments idx (rho_eff = softplus(x) * w); split bookkeeping keeps sum(rho * w)."""
        V = self.verts.detach()
        return mass_weight(V[self.segs[idx, 0]], V[self.segs[idx, 1]], self.log_scales.detach()[idx],
                           self.quats.detach()[idx])

    def _long_mask(self, idx):
        """Which segments are 'long' (split at the midpoint) rather than cloned."""
        V = self.verts.detach()
        ia, ib = self.segs[idx, 0], self.segs[idx, 1]
        L = (V[ib] - V[ia]).norm(dim=-1)
        sig = torch.exp(self.log_scales.detach()[idx]).max(-1).values
        return (L > self.split_len_factor * sig) & (ia != ib)

    def get_opacity(self):
        return torch.sigmoid(self.opacity)

    def get_beta(self):
        # beta >= 1: for beta < 1 the derivative integrand is singular at the support ends and GL8 gradients
        # become inaccurate. b = 0 -> beta = 4.5 (close to the Gaussian with the same covariance).
        return 1.0 + 3.5 * torch.exp(self.beta)

    def endpoints(self):
        return self.verts[self.segs[:, 0]], self.verts[self.segs[:, 1]]

    # ---------------------------------------------------------------- render
    def render(self, cam, bg, eps2d=0.1, mip=True, override_color=None, alpha_min=1 / 255, T_min=1e-4):
        a, b = self.endpoints()
        Sigma, comp3d = build_covariance(self.log_scales, self.quats, self.filter3d if mip else None)
        pr = project_segments(a, b, Sigma, cam, eps2d=eps2d, mip2d=mip)
        rho = F.softplus(self.opacity) * mass_weight(a, b, self.log_scales, self.quats) * pr["comp2d"]
        if comp3d is not None:
            rho = rho * comp3d
        if override_color is None:
            dirs = F.normalize(0.5 * (a + b).detach() - cam.center[None], dim=-1)
            color = self.view_colors(dirs)
        else:
            color = override_color
        beta = self.get_beta() if self.kernel == "beta" else None
        rad = screen_radii(pr, rho, self.kernel, beta=beta, max_radius=2.0 * max(cam.W, cam.H))
        rgb, alpha, info = rasterize(pr["pa"], pr["pb"], pr["ca"], pr["cb"], pr["Q"], rho, color, pr["depth"], rad,
                                     cam.W, cam.H, bg, kernel=self.kernel, beta=beta, alpha_min=alpha_min, T_min=T_min)
        info["visible"] = rad[:, 0] > 0
        return rgb, alpha, info

    # --------------------------------------------------------- optimizer glue
    def make_optimizer(self, lr):
        groups = [{"params": [t], "lr": lr[name], "name": name} for name, t in self.params().items()]
        return torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def _swap(self, opt, name, new_value, state_fn):
        group = next(g for g in opt.param_groups if g["name"] == name)
        old = group["params"][0]
        st = opt.state.pop(old, None)
        p = new_value.detach().contiguous().requires_grad_(True)
        group["params"][0] = p
        if st is not None and "exp_avg" in st:
            st["exp_avg"] = state_fn(st["exp_avg"]).contiguous()
            st["exp_avg_sq"] = state_fn(st["exp_avg_sq"]).contiguous()
            opt.state[p] = st
        setattr(self, name, p)

    @torch.no_grad()
    def _rebuild(self, opt, seg_src, seg_reset, new_segs, opacity_override, vert_src, new_verts, vert_reset):
        """Generic topology change. seg_src/vert_src index the old arrays (state is gathered the same way);
        rows with *_reset get a zeroed Adam state."""
        def seg_state(x):
            y = x[seg_src].clone()
            y[seg_reset] = 0
            return y

        def vert_state(x):
            y = x[vert_src.clamp_min(0)].clone()
            y[vert_reset | (vert_src < 0)] = 0
            return y

        for name in self.seg_attrs:
            val = getattr(self, name).detach()[seg_src].clone()
            if name == "opacity" and opacity_override is not None:
                val = opacity_override
            if opt is None:
                setattr(self, name, val.contiguous().requires_grad_(True))
            else:
                self._swap(opt, name, val, seg_state)
        # new_verts: (Nv', 3 + #extra) vertex features
        vals = {"verts": new_verts[:, :3]}
        for k, name in enumerate(self.extra_vert):
            vals[name] = new_verts[:, 3 + k]
        for name, val in vals.items():
            if opt is None:
                setattr(self, name, val.contiguous().requires_grad_(True))
            else:
                self._swap(opt, name, val, vert_state)
        self.segs = new_segs.contiguous()
        if self.filter3d is not None:
            self.filter3d = self.filter3d[seg_src].contiguous()
        if self.last_contrib is not None:
            self.last_contrib = self.last_contrib[seg_src].contiguous()
        self.attract_pairs = None

    @torch.no_grad()
    def _gc_vertices(self, opt):
        """Drop vertices no segment uses. Returns the old->new index map (-1 = dropped), or None if none dropped."""
        used = torch.zeros(self.num_vertices, dtype=torch.bool, device=self.device)
        used[self.segs.reshape(-1)] = True
        if bool(used.all()):
            return None
        vsrc = used.nonzero().squeeze(1)
        remap = torch.full((self.num_vertices,), -1, dtype=torch.long, device=self.device)
        remap[vsrc] = torch.arange(vsrc.numel(), device=self.device)
        segs = remap[self.segs]
        Ns = self.num_segments
        ar = torch.arange(Ns, device=self.device)
        self._rebuild(opt, ar, torch.zeros(Ns, dtype=torch.bool, device=self.device), segs, None, vsrc,
                      self._vertex_features()[vsrc], torch.zeros(vsrc.numel(), dtype=torch.bool, device=self.device))
        return remap

    # ------------------------------------------------------------- MCMC ops
    @torch.no_grad()
    def spawn(self, opt, targets, allow_split=True, allow_clone=True):
        """Create len(targets) new segments from (repeated) targets.
        Long targets are subdivided into n+1 connected pieces, short ones are cloned n times.
        The pieces share the parent's total density, which leaves the image unchanged (exactly for the 3D model;
        up to the per-segment EWA linearisation in screen space)."""
        dev = self.device
        Ns, Nv = self.num_segments, self.num_vertices
        if targets.numel() == 0:
            return
        counts = torch.bincount(targets, minlength=Ns)
        tgt = counts.nonzero().squeeze(1)
        n = counts[tgt]
        V = self._vertex_features()
        ia, ib = self.segs[tgt, 0], self.segs[tgt, 1]
        is_long = self._long_mask(tgt)
        if not allow_split:
            is_long = torch.zeros_like(is_long)
        if not allow_clone:  # everything that cannot be cloned is split (points are always cloned)
            is_long = ia != ib

        # density bookkeeping: clones share rho equally; split pieces share rho_eff = rho * w equally
        rho = F.softplus(self.opacity.detach()[tgt])
        w_par = self._seg_mass_weight(tgt)
        n1 = n.float() + 1
        w_child = 1 + (w_par - 1) / n1
        # Clones: as in 3DGS-MCMC the divisor is capped (N_max = 51 there). Without the cap, a burst of dead
        # primitives sampled onto few high-opacity targets splits each into hundreds of clones whose density
        # falls below the dead threshold; they get the full MCMC noise kick, die, are relocated again, and the
        # whole model collapses (seen on Shelly/kitten with points). Splits stay exact (geometry, not copies).
        rho_new = torch.where(is_long, rho * w_par / (n1 * w_child), rho / n1.clamp(max=MAX_CLONE_DIV))
        logit_new = inverse_softplus(rho_new)
        logit_of = torch.zeros(Ns, device=dev)
        logit_of[tgt] = logit_new

        segs = self.segs.clone()
        # --- long: subdivide into n+1 pieces
        tl, nl = tgt[is_long], n[is_long]
        tot_l = int(nl.sum())
        if tot_l > 0:
            own = torch.repeat_interleave(torch.arange(tl.numel(), device=dev), nl)
            start = torch.cumsum(nl, 0) - nl
            i_in = torch.arange(tot_l, device=dev) - start[own] + 1
            frac = (i_in.float() / (nl[own].float() + 1))[:, None]
            A, B = V[segs[tl, 0]], V[segs[tl, 1]]
            pos_l = A[own] + frac * (B[own] - A[own])
            vid = Nv + torch.arange(tot_l, device=dev)
            b_old = segs[tl, 1].clone()
            segs[tl, 1] = Nv + start
            nxt = torch.where(i_in < nl[own], vid + 1, b_old[own])
            segs_l = torch.stack([vid, nxt], 1)
            src_l = tl[own]
        else:
            pos_l = V.new_zeros(0, V.shape[1])
            segs_l = segs.new_zeros(0, 2)
            src_l = segs.new_zeros(0)
        # --- short: clone n times with fresh vertices
        ts, ns = tgt[~is_long], n[~is_long]
        tot_s = int(ns.sum())
        if tot_s > 0:
            own = torch.repeat_interleave(torch.arange(ts.numel(), device=dev), ns)
            src_s = ts[own]
            pa_, pb_ = V[self.segs[src_s, 0]], V[self.segs[src_s, 1]]
            base = Nv + tot_l + 2 * torch.arange(tot_s, device=dev)
            is_pt = self.segs[src_s, 0] == self.segs[src_s, 1]
            segs_s = torch.stack([base, torch.where(is_pt, base, base + 1)], 1)
            pos_s = torch.stack([pa_, pb_], 1).reshape(-1, V.shape[1])
        else:
            pos_s = V.new_zeros(0, V.shape[1])
            segs_s = segs.new_zeros(0, 2)
            src_s = segs.new_zeros(0)

        seg_src = torch.cat([torch.arange(Ns, device=dev), src_l, src_s])
        new_segs = torch.cat([segs, segs_l, segs_s])
        op = self.opacity.detach()[seg_src].clone()
        touched = torch.zeros(seg_src.numel(), dtype=torch.bool, device=dev)
        touched[tgt] = True
        touched[Ns:] = True
        op[touched] = logit_of[seg_src[touched]]
        new_verts = torch.cat([V, pos_l, pos_s])
        vert_src = torch.cat([torch.arange(Nv, device=dev), torch.full((tot_l + 2 * tot_s,), -1, device=dev, dtype=torch.long)])
        self._rebuild(opt, seg_src, touched, new_segs, op, vert_src, new_verts,
                      torch.zeros(new_verts.shape[0], dtype=torch.bool, device=dev))
        if self.last_contrib is not None:  # new pieces get a fresh grace period
            self.last_contrib[Ns:] = self.iteration
        self._gc_vertices(opt)

    @torch.no_grad()
    def remove(self, opt, mask):
        keep = (~mask).nonzero().squeeze(1)
        Nv = self.num_vertices
        self._rebuild(opt, keep, torch.zeros(keep.numel(), dtype=torch.bool, device=self.device), self.segs[keep],
                      None, torch.arange(Nv, device=self.device), self._vertex_features(),
                      torch.zeros(Nv, dtype=torch.bool, device=self.device))
        self._gc_vertices(opt)

    @torch.no_grad()
    def relocate(self, opt, min_opacity=0.005, stale_iters=None, **kw):
        """MCMC relocation of dead segments. Dead = opacity <= min_opacity, or (with stale_iters) no photometric
        gradient for stale_iters iterations, i.e. contributed to no training view during the last epoch: such
        segments sit below the alpha cut-off (alpha < 1/255) but above min_opacity and would otherwise hold
        budget forever (the gated L_MCMC never touches them)."""
        o = self.get_opacity().detach()
        dead = (o <= min_opacity) | ~torch.isfinite(o)
        if stale_iters is not None and self.last_contrib is not None:
            dead |= (self.iteration - self.last_contrib) > stale_iters
        # safety net: anything non-finite is dead (and is never used as a relocation target)
        vbad = ~torch.isfinite(self.verts.detach()).all(-1)
        for name in ["log_scales", "quats"] + self.color_attrs:
            dead |= ~torch.isfinite(getattr(self, name).detach()).reshape(self.num_segments, -1).all(-1)
        dead |= vbad[self.segs[:, 0]] | vbad[self.segs[:, 1]]
        o = torch.where(dead, torch.zeros_like(o), o)
        nd = int(dead.sum())
        if nd == 0 or nd == o.numel():
            return 0
        alive = (~dead).nonzero().squeeze(1)
        pick = torch.multinomial(o[alive], nd, replacement=True)
        self.spawn(opt, alive[pick], **kw)
        mask = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
        mask[: dead.numel()] = dead
        self.remove(opt, mask)
        return nd

    @torch.no_grad()
    def grow(self, opt, cap, rate=0.05, **kw):
        Ns = self.num_segments
        n_add = max(0, min(cap, int((1 + rate) * Ns)) - Ns)
        if n_add == 0:
            return 0
        o = self.get_opacity().detach()
        self.spawn(opt, torch.multinomial(o, n_add, replacement=True), **kw)
        return n_add

    @torch.no_grad()
    def add_noise(self, lr_xyz, noise_lr=5e5):
        """3DGS-MCMC SGLD noise (Sigma @ eps * noise_lr * lr, gated by low opacity) as a *rigid translation* of
        isolated segments only. The MCMC step is large (~Sigma, not sigma: units of the scene for big kernels);
        applied to individual vertices it stretches segments — a chain's shared vertex gets averaged noise while
        a free end gets the full kick — into scene-sized streaks that cover the screen. Segments inside
        polylines are left to relocation instead."""
        o = self.get_opacity().detach()
        gate = torch.sigmoid(100 * ((1 - o) - 0.995))
        Nv = self.num_vertices
        s0, s1 = self.segs[:, 0], self.segs[:, 1]
        deg = torch.bincount(self.segs.reshape(-1), minlength=Nv)
        is_pt = s0 == s1
        isolated = torch.where(is_pt, deg[s0] == 2, (deg[s0] == 1) & (deg[s1] == 1))
        idx = (isolated & (gate > 1e-4)).nonzero().squeeze(1)
        if idx.numel() == 0:
            return
        Sigma, _ = build_covariance(self.log_scales.detach()[idx], self.quats.detach()[idx])
        n = (Sigma @ torch.randn(idx.numel(), 3, 1, device=self.device)).squeeze(-1)
        n = n * (gate[idx] * noise_lr * lr_xyz)[:, None]
        self.verts.index_add_(0, s0[idx], n)
        nb = ~is_pt[idx]
        self.verts.index_add_(0, s1[idx][nb], n[nb])

    # ------------------------------------------------------------ Mip filter
    @torch.no_grad()
    def compute_filter3d(self, cams):
        a, b = self.endpoints()
        pts = torch.stack([a, b, 0.5 * (a + b)], 0).reshape(-1, 3)
        dist = torch.full((pts.shape[0],), float("inf"), device=self.device)
        seen = torch.zeros_like(dist, dtype=torch.bool)
        focal = 0.0
        for c in cams:
            pc = pts @ c.R.T + c.t
            z = pc[:, 2]
            x = pc[:, 0] / z * c.fx + c.cx
            y = pc[:, 1] / z * c.fy + c.cy
            ok = (z > 0.2) & (x >= -0.15 * c.W) & (x <= 1.15 * c.W) & (y >= -0.15 * c.H) & (y <= 1.15 * c.H)
            dist = torch.where(ok, torch.minimum(dist, z), dist)
            seen |= ok
            focal = max(focal, c.fx)
        if seen.any():
            dist[~seen] = dist[seen].max()
        else:
            dist[:] = 1.0
        d = dist.reshape(3, -1).min(0).values
        self.filter3d = (d / focal * math.sqrt(0.2)).contiguous()

    # ----------------------------------------------------- polyline topology
    def _endpoint_info(self):
        Nv = self.num_vertices
        deg = torch.bincount(self.segs.reshape(-1), minlength=Nv)
        ends = (deg == 1).nonzero().squeeze(1)
        owner = torch.full((Nv,), -1, dtype=torch.long, device=self.device)
        other = torch.full((Nv,), -1, dtype=torch.long, device=self.device)
        ar = torch.arange(self.num_segments, device=self.device)
        owner[self.segs[:, 0]] = ar
        owner[self.segs[:, 1]] = ar
        other[self.segs[:, 0]] = self.segs[:, 1]
        other[self.segs[:, 1]] = self.segs[:, 0]
        self._end_other = other[ends]
        sig = torch.exp(self.log_scales.detach()).mean(-1)
        return ends, owner[ends], sig[owner[ends]]

    @torch.no_grad()
    def _candidate_pairs(self, factor, k=8):
        """Endpoint pairs (i<j) closer than factor * min(kernel scale), among each endpoint's k nearest
        endpoints (bounded cost; a global radius search explodes when a few kernels are huge)."""
        from scipy.spatial import cKDTree

        ends, own, sig = self._endpoint_info()
        fin = torch.isfinite(self.verts.detach()[ends]).all(-1) & torch.isfinite(sig)
        ends, own, sig, oth = ends[fin], own[fin], sig[fin], self._end_other[fin]
        if ends.numel() < 2:
            return None
        P = self.verts.detach()[ends].cpu().numpy()
        s = sig.cpu().numpy()
        tree = cKDTree(P)
        kk = min(k + 1, len(P))
        d, nb = tree.query(P, k=kk, workers=-1)
        i = np.repeat(np.arange(len(P)), kk - 1)
        j = nb[:, 1:].reshape(-1)
        d = d[:, 1:].reshape(-1)
        ok = np.isfinite(d) & (j < len(P))
        i, j, d = i[ok], j[ok], d[ok]
        # admissible joins: (1) close relative to the kernels, (2) different connected components (never close a
        # polyline into a loop / contract it), (3) an end-to-end *continuation*: the two segments point away from
        # each other at the joint. (3) rejects a clone and its parent, which sit on top of each other with the
        # same orientation and would otherwise be fused into a pair of duplicate parallel edges.
        # (1) first: it is cheap and leaves few pairs for the gathers of (2) and (3)
        ok = d < factor * np.minimum(s[i], s[j])
        i, j, d = i[ok], j[ok], d[ok]
        comp_lab, _ = self.components()
        lab = comp_lab[own.cpu().numpy()]
        V = self.verts.detach()
        dirv = (V[ends] - V[oth]).cpu().numpy()  # direction of the owning segment towards the endpoint
        # continuation angle: cos of the bend at the joint (1 = straight). merge_cos = 0 accepts any bend < 90 deg
        nrm = np.linalg.norm(dirv, axis=1) + 1e-12
        straight = -(dirv[i] * dirv[j]).sum(1) / (nrm[i] * nrm[j])
        ok = (lab[i] != lab[j]) & (straight > getattr(self, "merge_cos", 0.0))
        i, j, d = i[ok], j[ok], d[ok]
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        key = lo.astype(np.int64) * len(P) + hi
        _, first = np.unique(key, return_index=True)
        if len(first) == 0:
            return None
        return ends.cpu().numpy(), lo[first], hi[first], d[first], s, lab

    def _end_dist(self, ends, u, w):
        V = self.verts.detach()
        return (V[torch.as_tensor(ends[u], device=self.device)]
                - V[torch.as_tensor(ends[w], device=self.device)]).norm(dim=-1).cpu().numpy()

    @staticmethod
    def _acyclic(lab, a, b, d):
        """Filter a merge matching (a[k] ~ b[k], endpoints with component labels lab) so that the merges of one call
        never join the same two components twice (Kruskal: closest first, union-find over labels).
        Two merges between the same components close a cycle. The common case is a segment lying on top of a clone
        of itself: each end's nearest *continuation* is the far end of the other copy (a~d, b~c), and merging both
        puts all four vertices on the midpoint, i.e. both segments collapse to zero length. They then stay
        degenerate (symmetric gradients), and relocation keeps cloning them (seen: 13% of the budget at
        merge_factor 4). With one merge the pair becomes a straight two-segment polyline instead."""
        if len(a) == 0:
            return a, b
        parent = {}

        def find(x):
            root = x
            while parent.get(root, root) != root:
                root = parent[root]
            while parent.get(x, x) != root:
                parent[x], x = root, parent[x]
            return root

        ok = np.zeros(len(a), dtype=bool)
        for k in np.argsort(d, kind="stable"):
            ra, rb = find(int(lab[a[k]])), find(int(lab[b[k]]))
            if ra != rb:
                parent[ra] = rb
                ok[k] = True
        return a[ok], b[ok]

    @staticmethod
    def _one_per_component_pair(lab, u, w, d):
        """Keep only the closest attraction pair between any two components. Pulling a polyline towards another one
        at two places folds it onto the other (for a segment on a clone of itself, a->d and b->c shrink both)."""
        if len(u) == 0:
            return u, w
        la, lb = lab[u].astype(np.int64), lab[w].astype(np.int64)
        key = np.minimum(la, lb) * (int(lab.max()) + 1) + np.maximum(la, lb)
        order = np.lexsort((d, key))
        first = np.ones(len(order), dtype=bool)
        first[1:] = key[order][1:] != key[order][:-1]
        sel = np.sort(order[first])
        return u[sel], w[sel]

    @staticmethod
    def _nearest_partner(i, j, d):
        """For candidate pairs (i, j, d): every endpoint that has a candidate, with its closest partner."""
        u = np.concatenate([i, j])
        w = np.concatenate([j, i])
        dd = np.concatenate([d, d])
        order = np.lexsort((dd, u))
        u, w = u[order], w[order]
        first = np.ones(len(u), dtype=bool)
        first[1:] = u[1:] != u[:-1]
        return u[first], w[first]

    @staticmethod
    def _mutual_matching(n, i, j, d):
        """Mutual nearest neighbours among the candidates -> a matching (each endpoint used at most once)."""
        u, w = SpaghettiModel._nearest_partner(i, j, d)
        best = np.full(n, -1)
        best[u] = w
        cand_u = np.nonzero(best >= 0)[0]
        mutual = cand_u[(best[best[cand_u]] == cand_u) & (cand_u < best[cand_u])]
        return mutual, best[mutual]

    @torch.no_grad()
    def merge_endpoints(self, opt, factor=0.5):
        """Merge pairs of polyline endpoints closer than factor * kernel scale (greedy, closest first)."""
        cand = self._candidate_pairs(factor)
        if cand is None:
            return 0
        ends, i, j, d, _, lab = cand
        a, b = self._mutual_matching(len(ends), i, j, d)
        a, b = self._acyclic(lab, a, b, self._end_dist(ends, a, b))
        if len(a) == 0:
            return 0
        self._merge_vertices(opt, ends[a], ends[b])
        return len(a)

    @torch.no_grad()
    def _merge_vertices(self, opt, keep_i, drop_j):
        """Identify vertex drop_j[k] with keep_i[k] (placed at their midpoint). Returns the old->new vertex index map
        of the compaction (None if no vertex was removed)."""
        ki = torch.as_tensor(keep_i, device=self.device)
        dj = torch.as_tensor(drop_j, device=self.device)
        V = self._vertex_features().clone()
        V[ki] = 0.5 * (V[ki] + V[dj])
        remap = torch.arange(self.num_vertices, device=self.device)
        remap[dj] = ki
        segs = remap[self.segs]
        Ns, Nv = self.num_segments, self.num_vertices
        reset = torch.zeros(Nv, dtype=torch.bool, device=self.device)
        reset[ki] = True
        self._rebuild(opt, torch.arange(Ns, device=self.device), torch.zeros(Ns, dtype=torch.bool, device=self.device),
                      segs, None, torch.arange(Nv, device=self.device), V, reset)
        return self._gc_vertices(opt)

    @torch.no_grad()
    def update_attraction_pairs(self, factor=3.0):
        """For each endpoint, its nearest other endpoint (different segment) within factor * kernel scale."""
        cand = self._candidate_pairs(factor)
        if cand is None:
            self.attract_pairs = None
            return 0
        ends, i, j, d, s, lab = cand
        if len(i) == 0:
            self.attract_pairs = None
            return 0
        u, w = self._nearest_partner(i, j, d)
        u, w = self._one_per_component_pair(lab, u, w, self._end_dist(ends, u, w))
        sc = np.minimum(s[u], s[w])
        self.attract_pairs = (torch.as_tensor(ends[u], device=self.device), torch.as_tensor(ends[w], device=self.device),
                              torch.as_tensor(sc, device=self.device, dtype=torch.float32))
        return len(u)

    def attraction_loss(self):
        if self.attract_pairs is None:
            return torch.zeros((), device=self.device)
        pi, pj, sc = self.attract_pairs
        return ((self.verts[pi] - self.verts[pj]).norm(dim=-1) / sc).mean()

    @torch.no_grad()
    def components(self):
        """Connected components of the segment graph. Returns (labels per segment, sizes in #segments)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        Nv = self.num_vertices
        s = self.segs.cpu().numpy()
        g = coo_matrix((np.ones(len(s)), (s[:, 0], s[:, 1])), shape=(Nv, Nv))
        _, lab = connected_components(g, directed=False)
        seg_lab = lab[s[:, 0]]
        _, inv, sizes = np.unique(seg_lab, return_inverse=True, return_counts=True)
        return inv, sizes

    # ------------------------------------------------------------- save/load
    def state_dict(self):
        d = {k: v.detach().cpu() for k, v in self.params().items()}
        d.update(segs=self.segs.cpu(), kernel=self.kernel, sh_degree=self.sh_degree, active_sh=self.active_sh,
                 appearance=self.appearance, sv_k=self.sv_k,
                 point_mode=self.point_mode, filter3d=None if self.filter3d is None else self.filter3d.cpu(),
                 last_contrib=None if self.last_contrib is None else self.last_contrib.cpu())
        return d

    def load_state_dict(self, d):
        self._set_appearance(d.get("appearance", "sh"), d.get("sv_k", 8))
        for k in ["verts"] + self.extra_vert + self.seg_attrs:
            setattr(self, k, d[k].to(self.device).requires_grad_(True))
        self.segs = d["segs"].to(self.device)
        self.kernel, self.sh_degree, self.active_sh = d["kernel"], d["sh_degree"], d["active_sh"]
        self.point_mode = d.get("point_mode", False)
        self.filter3d = None if d.get("filter3d") is None else d["filter3d"].to(self.device)
        lc = d.get("last_contrib")
        self.last_contrib = lc.to(self.device) if lc is not None else torch.zeros(
            self.num_segments, dtype=torch.long, device=self.device)
