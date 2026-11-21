"""JIT-compiles the CUDA rasterizer on first import (cached in spaghetti/csrc/build)."""
import os
import shutil
import sys

# Pick the CUDA toolkit *before* importing torch.utils.cpp_extension (it reads CUDA_HOME at import).
# Prefer the nvcc shipped inside the active python env (Blackwell needs nvcc >= 12.8).
if "CUDA_HOME" not in os.environ:
    _env = os.path.dirname(os.path.dirname(sys.executable))
    if os.path.exists(os.path.join(_env, "bin", "nvcc")):
        os.environ["CUDA_HOME"] = _env
    elif shutil.which("nvcc"):
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(shutil.which("nvcc")))

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, "csrc")
_build = os.environ.get("SPAGHETTI_BUILD_DIR", os.path.join(_src, "build"))
os.makedirs(_build, exist_ok=True)

if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

_inc = []
_tgt = os.path.join(os.environ.get("CUDA_HOME", ""), "targets", "x86_64-linux", "include")
if os.path.isdir(_tgt):  # conda-packaged CUDA keeps headers here
    _inc.append(_tgt)

_C = load(
    name="spaghetti_cuda",
    sources=[os.path.join(_src, "spaghetti.cu")],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    extra_include_paths=_inc,
    build_directory=_build,
    verbose=bool(int(os.environ.get("SPAGHETTI_VERBOSE_BUILD", "0"))),
)
