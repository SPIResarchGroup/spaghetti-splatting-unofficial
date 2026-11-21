"""Download the Shelly dataset (Wang et al., Adaptive Shells; CC BY-NC-SA 4.0) from its public Google Drive folder,
file by file, with retries. Needs `gdown` (pip install gdown).

python scripts/fetch_shelly.py --out data/shelly
"""
import argparse
import json
import os
import time

import gdown

FOLDER = "https://drive.google.com/drive/folders/1VyNLOx5qAJQmIQ_Iw1JSoQ4rnkCWg5uk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/shelly")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lst = os.path.join(args.out, "filelist.json")
    if os.path.exists(lst):
        files = json.load(open(lst))
    else:
        files = [(f.path, f.id) for f in gdown.download_folder(FOLDER, skip_download=True, quiet=True)]
        json.dump(files, open(lst, "w"))
    if args.scenes:
        files = [f for f in files if f[0].split("/")[0] in args.scenes or "/" not in f[0]]
    for i, (path, fid) in enumerate(files):
        dst = os.path.join(args.out, path)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(5):
            try:
                gdown.download(id=fid, output=dst + ".part", quiet=True)
                os.replace(dst + ".part", dst)
                break
            except Exception as e:  # rate limits etc.
                print(f"retry {attempt} {path}: {e}", flush=True)
                time.sleep(20 * (attempt + 1))
        if i % 50 == 0:
            print(f"[{i + 1}/{len(files)}] {path}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
