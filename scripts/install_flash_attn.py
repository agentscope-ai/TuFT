import os
import platform
import subprocess
import sys

import torch

FLASH_VERSION = "2.8.1"

# Get torch version
TORCH_VERSION_RAW = torch.__version__
torch_major, torch_minor = TORCH_VERSION_RAW.split(".")[:2]
torch_version = f"{torch_major}.{torch_minor}"

# Get python version
python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

# Get platform name
platform_name = platform.system().lower() + "_" + platform.machine()

# Get cxx11_abi
cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

# Is ROCM
IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None  # type: ignore[attr-defined]

if IS_ROCM:
    print("We currently do not host ROCm wheels for flash-attn.")
    sys.exit(1)
else:
    torch_cuda_version = torch.version.cuda  # type: ignore[attr-defined]
    cuda_major = torch_cuda_version.split(".")[0] if torch_cuda_version else None
    if cuda_major != "12":
        print("Only CUDA 12 wheels are hosted for flash-attn.")
        sys.exit(1)
    cuda_version = "12"
    wheel_filename = (
        f"flash_attn-{FLASH_VERSION}%2Bcu{cuda_version}torch{torch_version}"
        f"cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    )
    local_filename = (
        f"flash_attn-{FLASH_VERSION}-{python_version}-{python_version}-{platform_name}.whl"
    )

wheel_url = (
    "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com"
    f"/AgentScope/download/flash-attn/{FLASH_VERSION}/{wheel_filename}"
)

print(f"wheel_url: {wheel_url}")
print(f"target_local_file: {local_filename}")

local_path = f"/tmp/{local_filename}"

# avoid downloading multiple times in case of retrys
if os.path.exists(local_path):
    print(f"{local_path} already exists, removing the old file.")
    os.remove(local_path)

subprocess.run(["wget", wheel_url, "-O", local_path], check=True)
subprocess.run(["uv", "pip", "install", local_path], check=True)

# Try to import flash_attn
try:
    import flash_attn

    print(f"flash_attn {flash_attn.__version__} imported successfully!")
except ImportError as e:
    print("Failed to import flash_attn:", e)
    sys.exit(2)
