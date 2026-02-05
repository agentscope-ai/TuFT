FROM python:3.10-slim

WORKDIR /workspace/


COPY ./console/ ./

# If you cannot access pypi, enable the following line to use Mirror to speed up pip install
# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN pip install uv

# The following lines are used to enable mirrors. If you have network connection problems, you can enable them.
# ENV UV_PYTHON_INSTALL_MIRROR=https://python-standalone.org/mirror/astral-sh/python-build-standalone
# ENV UV_INDEX=https://mirrors.aliyun.com/pypi/simple/
# ENV HF_ENDPOINT=https://hf-mirror.com


RUN uv venv --python 3.12
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv pip install fastapi gradio tinker pytz requests

EXPOSE 10613

ENTRYPOINT ["bash", "scripts/start_user_console.sh"]
CMD ["--server-url", "http://host.docker.internal:10610", "--gui-port", "10613", "--backend-port", "10713"]
