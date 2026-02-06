# 控制台

TuFT 提供了一个仪表盘，供用户查看训练运行和检查点的详细信息，同时内置推理试验场，可直接试用微调后的模型。

## 环境准备

**安装 uv**

我们推荐使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理。

**激活虚拟环境**

如果您已在 uv 环境中，可跳过此步骤。

```bash
uv venv --python 3.12
source .venv/bin/activate
```

**安装依赖**

```bash
uv pip install fastapi gradio tinker pytz requests
```

## 快速开始

启动 TuFT 服务器后，运行以下命令启动用户控制台：

- `--server-url`：TuFT 服务器的 URL
- `--gui-port`：用户控制台的端口
- `--backend-port`：控制台后端的端口

推理标签页需要控制台服务器从 Hugging Face 加载 tokenizer。如果无法访问 Hugging Face，请设置镜像以启用推理标签页：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后启动控制台服务器和控制台 GUI：

```bash
cd src/tuft/console/
bash scripts/start_user_console.sh --server-url http://localhost:10610 --gui-port 10613 --backend-port 10713
```

您可以通过 [http://0.0.0.0:10613](http://0.0.0.0:10613) 访问用户控制台。

## 使用 Docker 部署

TuFT 还提供了 Dockerfile 用于快速部署。

TuFT 在 Dockerfile 中将服务器 URL 设置为 `http://host.docker.internal:10610`，这要求 TuFT 服务器在同一主机上运行。

我们还在 Dockerfile 中提供了网络不可用场景的相关建议，如有需要请启用。

```bash
cd src/tuft/
docker build -t tuft/user-console -f console/docker/Dockerfile .
docker run -d --name user-console-app -p 10613:10613 --add-host=host.docker.internal:host-gateway tuft/user-console
```
