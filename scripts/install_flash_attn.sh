#!/bin/bash

# flash-attention version
flash_version=2.8.1

# get torch version
torch_version_raw=$(python -c "import torch; print(torch.__version__)")
torch_major=$(echo $torch_version_raw | cut -d. -f1)
torch_minor=$(echo $torch_version_raw | cut -d. -f2)
torch_version="${torch_major}.${torch_minor}"

# get python version
python_version="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"

# get platform name
platform_name=$(python -c "import platform; print(platform.system().lower() + '_' + platform.machine())")

# get cxx11_abi
cxx11_abi=$(python -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")

# is ROCM
IS_ROCM=$(python -c "import torch; print('rocm' in torch.version.__dict__ and torch.version.hip is not None)")


if [ "$IS_ROCM" = "True" ]; then
    echo "We currently do not host ROCm wheels for flash-attn."
    exit 1
else
    torch_cuda_version=$(python -c "import torch; print(torch.version.cuda)")
    cuda_major=$(echo $torch_cuda_version | cut -d. -f1)
    if [ "$cuda_major" = "12" ]; then
        cuda_version="12"
    else
        echo "Only CUDA 12 wheels are hosted for flash-attn."
        exit 1
    fi
    cuda_version="12"
    wheel_filename="flash_attn-${flash_version}%2Bcu${cuda_version}torch${torch_version}cxx11abi${cxx11_abi}-${python_version}-${python_version}-${platform_name}.whl"
    local_filename="flash_attn-${flash_version}-${python_version}-${python_version}-${platform_name}.whl"
fi


wheel_url="https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/AgentScope/download/flash-attn/${flash_version}/${wheel_filename}"

echo "wheel_url: $wheel_url"
echo "target_local_file: $local_filename"

# avoid downloading multiple times in case of retrys
if [ -f /tmp/$local_filename ]; then
    echo "/tmp/$local_filename already exists, removing the old file."
    rm /tmp/$local_filename
fi
wget $wheel_url -O /tmp/$local_filename
uv pip install /tmp/$local_filename