# User console

TuFT provides a dashboard for users to know the details of their traning runs and checkpoints. It also provides a sampling playground for users to try out the finetuned models.

## prerequisites

**Install uv**

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

**Activate virtual environment**

If are already in the uv environment, you can skip this step.
```shell
uv venv --python 3.12
source .venv/bin/activate
```

**Install the dependencies**

```shell
uv pip install fastapi gradio tinker pytz requests
```

## Quick start
After you start the TuFT server, you can run the following command to start the user console:

--server-url: the URL of the TuFT server

--gui-port: the port of the user console

--backend-port: the port of the console backend

The sampling tab needs the console server to load the tokenizer from the huggingface. If you cannot access HF, you need setup the mirror to enable the sampling tab.

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

Then start the console server and the console gui.
```shell
cd src/tuft/console/
bash scripts/start_user_console.sh --server-url http://localhost:10610 --gui-port 10613 --backend-port 10713
```

You can access the user console in http://0.0.0.0:10613

## Deploy with docker

TuFT also provide the docker file for quick deployment. 

TuFT set the server url to http://host.docker.internal:10610 in the Dockerfile which requires the TuFT server to be running on the same host.

We also provie some suggestions when the network is not available in the dockerfile. Enable them if you need.
```shell
cd src/tuft/
docker build -t tuft/user-console -f console/docker/Dockerfile .
docker run -d --name user-console-app -p 10613:10613 --add-host=host.docker.internal:host-gateway tuft/user-console
```