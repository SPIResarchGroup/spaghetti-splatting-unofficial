"""Dataset loading: COLMAP (Mip-NeRF 360) and the synthetic format written by scripts/make_synthetic.py."""
import json
import os
import struct

import numpy as np
import torch
from PIL import Image

from .render import Camera

CAMERA_MODEL_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, model, w, h = _read(f, "iiQQ")
            params = _read(f, "d" * CAMERA_MODEL_NPARAMS[model])
            cams[cid] = (model, w, h, params)
    return cams


def read_images_bin(path):
    imgs = []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (npts,) = _read(f, "Q")
            f.seek(24 * npts, 1)
            imgs.append((iid, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]), cid, name.decode()))
    return imgs


def read_points3d_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    (n,) = struct.unpack_from("<Q", data, 0)
    off = 8
    rec = struct.Struct("<QdddBBBdQ")  # id, xyz, rgb, error, track length (51 bytes)
    xyz = np.empty((n, 3), np.float64)
    rgb = np.empty((n, 3), np.uint8)
    for k in range(n):
        vals = rec.unpack_from(data, off)
        xyz[k] = vals[1:4]
        rgb[k] = vals[4:7]
        off += rec.size + 8 * vals[8]  # track: (image_id int32, point2D_idx int32) per element
    assert off == len(data), "points3D.bin parse error"
    return xyz, rgb


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def _load_image(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).contiguous()  # uint8 (H,W,3)


def load_colmap(root, images="images_2", llffhold=8, device="cuda"):
    sp = os.path.join(root, "sparse", "0")
    cams = read_cameras_bin(os.path.join(sp, "cameras.bin"))
    imgs = sorted(read_images_bin(os.path.join(sp, "images.bin")), key=lambda x: x[4])
    out = []
    for iid, q, t, cid, name in imgs:
        model, w0, h0, params = cams[cid]
        if model == 0:  # SIMPLE_PINHOLE
            f, cx, cy = params
            fx = fy = f
        elif model in (2, 3):  # SIMPLE_RADIAL / RADIAL (distortion ignored)
            f, cx, cy = params[:3]
            fx = fy = f
        else:  # PINHOLE / OPENCV ... (distortion ignored)
            fx, fy, cx, cy = params[:4]
        img = _load_image(os.path.join(root, images, name))
        H, W = img.shape[:2]
        sx, sy = W / w0, H / h0
        R = torch.tensor(qvec2rotmat(q), dtype=torch.float32)
        cam = Camera(R.to(device), torch.tensor(t, dtype=torch.float32, device=device), fx * sx, fy * sy, cx * sx,
                     cy * sy, W, H, img.to(device), name)
        out.append(cam)
    test = [c for i, c in enumerate(out) if i % llffhold == 0]
    train = [c for i, c in enumerate(out) if i % llffhold != 0]
    xyz, rgb = read_points3d_bin(os.path.join(sp, "points3D.bin"))
    return train, test, xyz.astype(np.float32), rgb.astype(np.float32) / 255.0


def load_synthetic(root, device="cuda", llffhold=8):
    meta = json.load(open(os.path.join(root, "transforms.json")))
    out = []
    for fr in meta["frames"]:
        img = _load_image(os.path.join(root, fr["file"]))
        H, W = img.shape[:2]
        out.append(Camera(torch.tensor(fr["R"], dtype=torch.float32, device=device),
                          torch.tensor(fr["t"], dtype=torch.float32, device=device),
                          meta["fx"], meta["fy"], meta["cx"], meta["cy"], W, H, img.to(device), fr["file"]))
    test = [c for i, c in enumerate(out) if i % llffhold == 0]
    train = [c for i, c in enumerate(out) if i % llffhold != 0]
    init = np.load(os.path.join(root, "init_points.npz"))
    return train, test, init["xyz"].astype(np.float32), init["rgb"].astype(np.float32), meta


def scene_extent(cams):
    C = torch.stack([c.center for c in cams])
    return 1.1 * float((C - C.mean(0)).norm(dim=-1).max())
