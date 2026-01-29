# This Dockerfile sets up a TuFT environment using uv.
# Build and run the docker image with the following command:
#
# cd <TuFT root dir>
# docker build -f docker/Dockerfile -t tuft:latest .
# docker run -it --gpus all --shm-size="128g" --rm -v $PWD:/workspace -v <host_dir>:/data tuft:latest
#
# Note:
# This Dockerfile uses 'uv' to create a virtual environment for better package management.
# The uv virtual environment is created at `/root/.tuft/venv`, use `source /root/.tuft/venv/bin/activate` to activate it.
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

ENV VIRTUAL_ENV=/root/.tuft/venv

# copy the TuFT dir into the workspace
COPY ./pyproject.toml .
COPY ./LICENSE .
COPY ./README.md .
COPY ./src ./src
COPY ./scripts ./scripts

#  Uncomment the following line if you want to use AliCloud Mirror to speed up pip install
# ENV UV_DEFAULT_INDEX=http://mirrors.cloud.aliyuncs.com/pypi/simple/

# Uncomment the following line to use a Hugging Face mirror if you have network connection problem with Hugging Face
# ENV HF_ENDPOINT=https://hf-mirror.com

# Install
RUN bash ./scripts/install.sh --local-source /workspace \
    && . $HOME/.local/bin/env \
    && . /root/.tuft/venv/bin/activate \
    && uv pip install .[dev]

ENTRYPOINT ["/bin/bash", "-c", "source ${VIRTUAL_ENV}/bin/activate && exec \"$@\"", "--"]
CMD ["bash"]
