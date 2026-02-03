# User console

## prerequisites

**Install uv**

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

**Activate virtual environment**
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

```shell
cd src/tuft/user_console/
bash scripts/start_user_console.sh --server-url http://localhost:10610 --gui-port 10613 --backend-port 10712
```

You can access the user console in http://0.0.0.0:10613

## Deploy with docker

We also provide the docker file for quick deployment. 

We set the server url to http://host.docker.internal:10610 in the Dockerfile which requires the TuFT server to be running on the same host.
```shell
cd src/tuft/user_console/
docker build -t tuft/user-console .
docker run -d --name user-console-app -p 10613:10613 --add-host=host.docker.internal:host-gateway tuft/user-console
```