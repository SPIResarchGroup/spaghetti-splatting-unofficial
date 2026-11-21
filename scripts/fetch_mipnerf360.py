"""Extract selected scenes from the Mip-NeRF 360 zip using HTTP range requests.

The official archive is 12.5 GB; this downloads only the members we need
(central directory + requested files), e.g. ~250 MB for bonsai/images_2.

usage: python scripts/fetch_mipnerf360.py --scene bonsai --images images_2 --out data/360_v2
"""
import argparse
import io
import os
import urllib.request
import zipfile

URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests with a small read-ahead cache."""

    def __init__(self, url, block=4 << 20):
        self.url = url
        self.block = block
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache_start, self.cache = -1, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _fetch(self, start, end):  # inclusive end
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs = self.cache_start
        if cs >= 0 and cs <= self.pos and self.pos + n <= cs + len(self.cache):
            out = self.cache[self.pos - cs: self.pos - cs + n]
        elif n >= self.block:
            out = self._fetch(self.pos, self.pos + n - 1)
        else:
            end = min(self.size, self.pos + self.block) - 1
            self.cache_start, self.cache = self.pos, self._fetch(self.pos, end)
            out = self.cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", nargs="+", default=["bonsai"])
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--out", default="data/360_v2")
    args = ap.parse_args()

    f = HTTPRangeFile(URL)
    zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = zf.namelist()
    for scene in args.scene:
        want = [
            n for n in names
            if n.startswith(f"{scene}/")
            and not n.endswith("/")
            and (n.startswith(f"{scene}/{args.images}/") or n.startswith(f"{scene}/sparse/")
                 or n == f"{scene}/poses_bounds.npy")
        ]
        total = sum(zf.getinfo(n).file_size for n in want)
        print(f"{scene}: {len(want)} files, {total / 2**20:.1f} MB")
        for i, n in enumerate(want):
            dst = os.path.join(args.out, n)
            if os.path.exists(dst) and os.path.getsize(dst) == zf.getinfo(n).file_size:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(n) as src, open(dst + ".part", "wb") as o:
                o.write(src.read())
            os.replace(dst + ".part", dst)
            if i % 25 == 0:
                print(f"  [{i + 1}/{len(want)}] {n}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
