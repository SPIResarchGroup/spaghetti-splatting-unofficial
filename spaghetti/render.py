"""Projection of 3D segments to screen space (PyTorch autograd) + CUDA rasterization."""
import math
from dataclasses import dataclass

import torch

from ._backend import _C

KERNELS = {"gaussian": 0, "beta": 1}


@dataclass
class Camera:
    R: torch.Tensor  # (3,3) world -> camera rotation
    t: torch.Tensor  # (3,)  world -> camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    image: torch.Tensor = None  # (H,W,3) in [0,1]
    name: str = ""

    @property
    def center(self):
        return -self.R.T @ self.t

    def to(self, device):
        img = self.image.to(device) if self.image is not None else None
        return Camera(self.R.to(device), self.t.to(device), self.fx, self.fy, self.cx, self.cy, self.W, self.H, img, self.name)

    def scaled(self, s):
        """Camera for an image rescaled by factor s (s<1: lower resolution)."""
        W, H = max(1, round(self.W * s)), max(1, round(self.H * s))
        sx, sy = W / self.W, H / self.H
        return Camera(self.R, self.t, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, W, H, None, self.name)


# ----------------------------------------------------------------------------
# spherical harmonics (same convention as 3DGS)
# ----------------------------------------------------------------------------
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435]


def eval_sh(deg, sh, dirs):
    """sh: (N,(deg+1)^2,3), dirs: (N,3) unit."""
    out = C0 * sh[:, 0]
    if deg < 1:
        return out
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    out = out - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if deg < 2:
        return out
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    out = (out + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5] + C2[2] * (2 * zz - xx - yy) * sh[:, 6]
           + C2[3] * xz * sh[:, 7] + C2[4] * (xx - yy) * sh[:, 8])
    if deg < 3:
        return out
    out = (out + C3[0] * y * (3 * xx - yy) * sh[:, 9] + C3[1] * xy * z * sh[:, 10]
           + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11] + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
           + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + C3[5] * z * (xx - yy) * sh[:, 14]
           + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return out


def quat_to_rotmat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def build_covariance(log_scales, quats, filter3d=None):
    """Sigma = R S^2 R^T (+ filter3d^2 I). Returns Sigma and the Mip-Splatting 3D opacity compensation."""
    Rm = quat_to_rotmat(quats)
    s2 = torch.exp(2 * log_scales)
    comp = None
    if filter3d is not None:
        f2 = filter3d[:, None] ** 2
        comp = torch.sqrt(torch.prod(s2 / (s2 + f2), -1))
        s2 = s2 + f2
    Sigma = (Rm * s2[:, None, :]) @ Rm.transpose(1, 2)
    return Sigma, comp


# ----------------------------------------------------------------------------
# projection (Sec. 3.4: EWA at the segment midpoint, scaled by inverse depth)
# ----------------------------------------------------------------------------
def project_segments(a_w, b_w, Sigma, cam, near=0.2, eps2d=0.1, mip2d=True):
    R, t = cam.R, cam.t
    a_c = a_w @ R.T + t
    b_c = b_w @ R.T + t
    za, zb = a_c[:, 2], b_c[:, 2]
    valid = (za > near) & (zb > near)
    one = torch.ones_like(za)
    za = torch.where(valid, za, one)
    zb = torch.where(valid, zb, one)
    m_c = 0.5 * (a_c + b_c)
    zm = torch.where(valid, m_c[:, 2], one)

    fx, fy, cx, cy, W, H = cam.fx, cam.fy, cam.cx, cam.cy, cam.W, cam.H
    # 3DGS / gsplat clamp: 0.3 * tan(fov/2) = 0.15 * W / fx beyond the image border
    lim_xp, lim_xn = (W - cx) / fx + 0.15 * W / fx, cx / fx + 0.15 * W / fx
    lim_yp, lim_yn = (H - cy) / fy + 0.15 * H / fy, cy / fy + 0.15 * H / fy
    tx = torch.clamp(m_c[:, 0] / zm, -lim_xn, lim_xp) * zm
    ty = torch.clamp(m_c[:, 1] / zm, -lim_yn, lim_yp) * zm
    zero = torch.zeros_like(zm)
    J = torch.stack([fx / zm, zero, -fx * tx / zm ** 2, zero, fy / zm, -fy * ty / zm ** 2], -1).reshape(-1, 2, 3)
    T = J @ R  # (N,2,3)
    cov2d = T @ Sigma @ T.transpose(1, 2)
    c00, c01, c11 = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det0 = c00 * c11 - c01 * c01
    c00, c11 = c00 + eps2d, c11 + eps2d
    det1 = (c00 * c11 - c01 * c01).clamp_min(1e-12)
    comp2d = torch.sqrt((det0 / det1).clamp_min(1e-12)) if mip2d else torch.ones_like(det1)
    Q = 0.5 * torch.stack([c11, -c01, c00], -1) / det1[:, None]

    pa = torch.stack([fx * a_c[:, 0] / za + cx, fy * a_c[:, 1] / za + cy], -1)
    pb = torch.stack([fx * b_c[:, 0] / zb + cx, fy * b_c[:, 1] / zb + cy], -1)
    ca, cb = za / zm, zb / zm

    with torch.no_grad():
        lam_max = 0.5 * (c00 + c11) + torch.sqrt((0.25 * (c00 - c11) ** 2 + c01 ** 2).clamp_min(0))
        sig = lam_max.clamp_min(0).sqrt()
    return dict(pa=pa, pb=pb, ca=ca, cb=cb, Q=Q, comp2d=comp2d, depth=zm, valid=valid, sigma_max=sig)


def screen_radii(proj, rho, kernel, valid_extra=None, beta=None):
    """Pixel radii at both endpoints: the Mahalanobis radius beyond which tau < 1/255 for a point
    (the segment's tau is bounded by the point value of its nearest kernel), i.e. alpha < alpha_min."""
    with torch.no_grad():
        if kernel == "gaussian":  # rho exp(-q) >= 1/255, q = k^2/2
            k = torch.sqrt(2 * torch.log((255.0 * rho).clamp_min(1.0))).clamp(max=4.0)
        else:  # rho (1 - 2q/9)^beta >= 1/255, compact support k <= 3
            frac = 1 - torch.pow((255.0 * rho).clamp_min(1.0), -1.0 / beta.clamp_min(1e-3))
            k = torch.sqrt(9.0 * frac.clamp(0, 1))
        ra = k * proj["sigma_max"] / proj["ca"]
        rb = k * proj["sigma_max"] / proj["cb"]
        ok = proj["valid"] & (k > 0)
        if valid_extra is not None:
            ok = ok & valid_extra
        rad = torch.stack([ra, rb], -1)
        rad = torch.where(ok[:, None], rad, torch.zeros_like(rad))
        rad = torch.nan_to_num(rad, nan=0.0, posinf=0.0, neginf=0.0)
    return rad.contiguous()


# ----------------------------------------------------------------------------
# rasterization
# ----------------------------------------------------------------------------
class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pa, pb, ca, cb, Q, rho, colors, beta, bg, ranges, plist, W, H, kernel, alpha_min, T_min):
        rgb, alpha, finT, finN = _C.rasterize_fwd(ranges, plist, pa, pb, ca, cb, Q, rho, colors, beta, bg,
                                                  W, H, kernel, alpha_min, T_min)
        ctx.save_for_backward(pa, pb, ca, cb, Q, rho, colors, beta, bg, ranges, plist, finT, finN)
        ctx.meta = (W, H, kernel, alpha_min)
        return rgb, alpha

    @staticmethod
    def backward(ctx, d_rgb, d_alpha):
        pa, pb, ca, cb, Q, rho, colors, beta, bg, ranges, plist, finT, finN = ctx.saved_tensors
        W, H, kernel, alpha_min = ctx.meta
        if d_rgb is None:
            d_rgb = torch.zeros(H, W, 3, device=pa.device)
        if d_alpha is None:
            d_alpha = torch.zeros(H, W, device=pa.device)
        g = _C.rasterize_bwd(ranges, plist, pa, pb, ca, cb, Q, rho, colors, beta, bg, W, H, kernel, alpha_min,
                             finT, finN, d_rgb.contiguous(), d_alpha.contiguous())
        g_pa, g_pb, g_ca, g_cb, g_Q, g_rho, g_col, g_beta = g
        return (g_pa, g_pb, g_ca, g_cb, g_Q, g_rho, g_col, g_beta if kernel == KERNELS["beta"] else None,
                None, None, None, None, None, None, None, None)


def rasterize(pa, pb, ca, cb, Q, rho, colors, depth, radii, W, H, bg, kernel="gaussian", beta=None,
              alpha_min=1.0 / 255.0, T_min=1e-4):
    """All inputs are per-primitive screen-space quantities; returns rgb (H,W,3), alpha (H,W), info."""
    kid = KERNELS[kernel]
    f = lambda x: x.float().contiguous()
    pa, pb, ca, cb, Q, rho, colors = map(f, (pa, pb, ca, cb, Q, rho, colors))
    if beta is None:
        beta = torch.zeros_like(rho)
    beta = f(beta)
    bg = f(bg)
    ranges, plist, counts = _C.bin_segments(pa.detach(), pb.detach(), radii, depth.detach().float().contiguous(), W, H)
    rgb, alpha = _Rasterize.apply(pa, pb, ca, cb, Q, rho, colors, beta, bg, ranges, plist, W, H, kid,
                                  float(alpha_min), float(T_min))
    return rgb, alpha, dict(tiles_per_prim=counts, n_pairs=int(plist.numel()))
