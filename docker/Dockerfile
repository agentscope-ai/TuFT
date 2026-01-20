# This Dockerfile sets up a TuFT environment using uv.
# Build and run the docker image with the following command:
#
# cd <TuFT root dir>
# docker build -f docker/Dockerfile -t tuft:latest .
# docker run -it --gpus all --shm-size="128g" --rm -v $PWD:/workspace -v <host_dir>:/data tuft:latest
#
# Note:
# This Dockerfile uses 'uv' to create a virtual environment for better package management.
# The uv virtual environment is created at `/opt/venv`, use `source /opt/venv/bin/activate` to activate it.
# Make sure to use `uv pip` to install packages within the virtual environment.

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace

RUN chmod 1777 /tmp && apt update && apt install -y --no-install-recommends \
    build-essential \
    curl git wget vim tmux net-tools \
    python3 python3-pip python3-dev python3-packaging python3-venv \
    libomp-dev infiniband-diags libibverbs-dev librdmacm-dev rdma-core perftest \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

ENV VIRTUAL_ENV=/opt/venv

# copy the TuFT dir into the workspace
COPY . .

# For Aliyun users: update pip mirror to aliyun to speed up pip install
# ENV UV_DEFAULT_INDEX=http://mirrors.cloud.aliyuncs.com/pypi/simple/

# For China users: uncomment the following line to use a Hugging Face mirror
# ENV HF_ENDPOINT=https://hf-mirror.com

# Install uv
RUN pip install uv && uv venv /opt/venv --python=python3.12

# Install minimal TuFT
RUN . /opt/venv/bin/activate && uv sync --all-extras

# Install flash_attn
RUN . /opt/venv/bin/activate && uv pip install flash_attn==2.8.1 --no-build-isolation

ENTRYPOINT ["/bin/bash", "-c", "source /opt/venv/bin/activate && exec \"$@\"", "--"]
CMD ["bash"]
