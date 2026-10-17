#!/usr/bin/env python
"""
Run TuFT as a Hugging Face Job FROM A CONFIG FILE — mirrors deploy/modal/launch.py.

You edit a standard tuft_config.yaml (the same file `tuft launch --config` uses) and run
this script. It validates the config, passes it to the Job as a secret, starts the stock
TuFT image on an HF Jobs GPU with the API port exposed, waits for /healthz, and prints how
to connect. To change models / keys / options, edit the YAML and re-run.

Verbs (flag-based, like the Modal launch.py — default action is "launch"):
    python deploy/hf_jobs/launch.py --config my.yaml                 # launch a Job
    python deploy/hf_jobs/launch.py --config my.yaml --flavor l4x1   # override hardware
    python deploy/hf_jobs/launch.py --down --config my.yaml          # cancel it (or --job-id)
    python deploy/hf_jobs/launch.py --status                         # list TuFT Jobs

Infra (flavor, timeout, ...) can be set as CLI flags or in an `hf_jobs:` section of the YAML;
precedence is flag > `hf_jobs:` > default. TuFT ignores that section (it's stripped before the
server sees the config), so the same YAML still works with `tuft launch --config` locally.

Storage: checkpoint_dir is pinned to a private Storage Bucket (default
`<namespace>/tuft-checkpoints`, created if missing), so adapters outlive the Job and other
Jobs can mount them. Models given as HF repo ids are mounted read-only from the Hub instead
of downloaded (`--no-mount-models` to download at startup instead).

Auth: log in with `hf auth login` (a token that can start Jobs). Needs `pip install pyyaml
huggingface_hub`. The exposed port requires an HF token with read access to the Job's
namespace (sent as `Authorization: Bearer`) ON TOP of your `tml-` X-API-Key. `--public`
drops the HF token layer, leaving only the TuFT key.

Billing: HF Jobs have no scale-to-zero. The Job is billed until it hits --timeout or you
run --down.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


TUFT_PORT = 10610
HEALTHZ_PATH = "/api/v1/healthz"
CKPT_DIR = "/data/checkpoints"
MODELS_DIR = "/models"
REMOTE_CONFIG = "/tmp/tuft_config.yaml"
VENV_PY = "/root/.tuft/venv/bin/python"
JOB_LABELS = {"app": "tuft"}
_HF_REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
_TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}

# Infra knobs and their built-in defaults. Same keys are used for CLI flags and the
# YAML `hf_jobs:` section.
DEFAULTS = {
    "flavor": "a100-large",  # `hf jobs hardware` lists them; l4x1 fits Qwen3-1.7B
    "name": "tuft-server",
    "timeout": "2h",  # hard stop; billed until then unless you --down
    "namespace": "",  # "" = the logged-in user
    "ckpt_bucket": "tuft-checkpoints",
    "mount_models": True,
    "public": False,
    "network_group": "",
    "hf_token": False,  # forward your HF token into the Job (gated models without mounting)
    "image": "ghcr.io/agentscope-ai/tuft:latest",
    "startup_timeout": 900,
}
_BOOL_KEYS = {"mount_models", "public", "hf_token"}

# Decode the config from the secret and run the standard server, bound to 0.0.0.0 so the
# Jobs proxy can reach it; --checkpoint-dir pins checkpoints to the bucket mount. Mounted
# weights are read once sequentially first: vLLM's mmap reads on a cold Hub mount can take
# minutes or stall (seen with tensor_parallel_size=2), while a warm page cache loads in seconds.
SERVER_CMD = (
    f"if [ -d {MODELS_DIR} ]; then echo 'prefetching mounted model weights'; "
    f"find {MODELS_DIR} -name '*.safetensors' -exec cat {{}} + > /dev/null; fi; "
    f'echo "$TUFT_CONFIG_B64" | base64 -d > {REMOTE_CONFIG} && '
    f"exec {VENV_PY} -m tuft launch --host 0.0.0.0 --port {TUFT_PORT} "
    f"--config {REMOTE_CONFIG} --checkpoint-dir {CKPT_DIR}"
)


def hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is required: pip install -U huggingface_hub")
    if "expose" not in inspect.signature(HfApi.run_job).parameters:
        sys.exit("huggingface_hub is too old for exposed Job ports: pip install -U huggingface_hub")
    return HfApi()


def load_config(path: Path) -> dict:
    """Parse + validate the TuFT config. Returns the parsed dict (incl. any `hf_jobs:` section)."""
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml is required: pip install pyyaml")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        sys.exit("config error: top level must be a YAML mapping")
    models = data.get("supported_models") or []
    if not models:
        sys.exit("config error: supported_models is empty — add at least one model")
    if len(models) > 1 and any((m or {}).get("colocate") for m in models):
        sys.exit("config error: colocate: true is only allowed with a SINGLE model")
    users = data.get("authorized_users") or {}
    if not users:
        sys.exit("config error: authorized_users is empty — add at least one 'tml-...' key")
    bad = [k for k in users if not str(k).startswith("tml-")]
    if bad:
        sys.exit(f"config error: authorized_users keys must start with 'tml-': {bad}")
    if any("CHANGE-ME" in str(k) for k in users):
        print("[warn] authorized_users still contains a placeholder key — set a real tml- key")
    print(f"[ok] config valid: {len(models)} model(s), {len(users)} api key(s)")
    return data


def resolve_infra(args: argparse.Namespace, file_infra: dict) -> dict:
    """Merge infra knobs: CLI flag (if given) > YAML `hf_jobs:` section > DEFAULTS."""
    unknown = set(file_infra) - set(DEFAULTS)
    if unknown:
        print(f"[warn] unknown keys in `hf_jobs:` section ignored: {sorted(unknown)}")
    infra = {}
    for key, dflt in DEFAULTS.items():
        cli_val = getattr(args, key)
        if cli_val is not None:
            infra[key] = cli_val
        elif file_infra.get(key) is not None:
            infra[key] = file_infra[key]
        else:
            infra[key] = dflt
    for key in _BOOL_KEYS:
        infra[key] = bool(infra[key])
    infra["startup_timeout"] = int(infra["startup_timeout"])
    return infra


def prepare_server_config(data: dict, mount_models: bool) -> tuple[str, list[str]]:
    """Strips deploy sections, points Hub-hosted model_paths at read-only mounts, and returns
    (base64 config, repo ids to mount)."""
    import yaml

    clean = {k: v for k, v in data.items() if k not in ("hf_jobs", "modal", "lambda")}
    repos: list[str] = []
    if mount_models:
        for m in clean["supported_models"]:
            path = str(m.get("model_path") or m.get("model_name") or "")
            if _HF_REPO_ID.match(path) and not Path(path).expanduser().exists():
                m["model_path"] = f"{MODELS_DIR}/{path}"
                if path not in repos:
                    repos.append(path)
    payload = yaml.safe_dump(clean, sort_keys=False).encode("utf-8")
    return base64.b64encode(payload).decode("ascii"), repos


def get_hf_token() -> str:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        sys.exit("no HF token found — run `hf auth login` first")
    return token


def wait_healthy(api, job_id: str, namespace: str, url: str, headers: dict, timeout: int) -> bool:
    """Waits for the Job to run, then for TuFT's /healthz through the Jobs proxy."""
    deadline = time.time() + timeout
    print(f"[wait] Job {job_id}: waiting for the image pull and TuFT startup ...", flush=True)
    last_stage = ""
    while time.time() < deadline:
        stage = str(api.inspect_job(job_id=job_id, namespace=namespace).status.stage)
        stage = stage.rsplit(".", 1)[-1]
        if stage != last_stage:
            print(f"[wait] stage={stage}", flush=True)
            last_stage = stage
        if stage in _TERMINAL:
            return False
        if stage == "RUNNING":
            req = urllib.request.Request(url + HEALTHZ_PATH, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                pass
        time.sleep(10)
    return False


def banner(url: str, model: str, public: bool, job_id: str, namespace: str, stop: str) -> None:
    headers, note = "", ""
    if not public:
        headers = '\n        default_headers={"Authorization": f"Bearer {HF_TOKEN}"},'
        note = "   # HF_TOKEN: any HF token with read access to the Job's namespace\n"
    print(f"""
✅ TuFT is up: {url}
   model={model}  job={namespace}/{job_id}

   import tinker
   client = tinker.ServiceClient(
        base_url="{url}",
        api_key="tml-...",{headers}
   )
   # api_key: your authorized_users key
{note}
   logs:  hf jobs logs -f {namespace}/{job_id}
   stop:  {stop}   (billed until then, or until --timeout)
""")


def file_infra(config: str | None) -> dict:
    """The `hf_jobs:` section of --config, so --down/--status find Jobs where launch put them."""
    if not config:
        return {}
    try:
        import yaml

        doc = yaml.safe_load(Path(config).expanduser().read_text()) or {}
    except Exception:
        return {}
    return doc.get("hf_jobs") or {}


def down(api, args: argparse.Namespace) -> None:
    infra = file_infra(args.config)
    namespace = args.namespace or infra.get("namespace") or None
    if args.job_id:
        ids = [args.job_id]
    else:
        name = args.name or infra.get("name") or DEFAULTS["name"]
        jobs = api.list_jobs(labels={**JOB_LABELS, "name": name}, namespace=namespace)
        ids = [j.id for j in jobs if str(j.status.stage).rsplit(".", 1)[-1] not in _TERMINAL]
        if not ids:
            where = namespace or "your namespace"
            sys.exit(f"no running TuFT Job named '{name}' in {where} — see `--status`")
    for job_id in ids:
        api.cancel_job(job_id=job_id, namespace=namespace)
        print(f"== canceled Job {job_id} ==")


def status(api, args: argparse.Namespace) -> None:
    namespace = args.namespace or file_infra(args.config).get("namespace") or None
    jobs = list(api.list_jobs(labels=JOB_LABELS, namespace=namespace))
    if not jobs:
        print("no TuFT Jobs")
    for j in jobs:
        stage = str(j.status.stage).rsplit(".", 1)[-1]
        urls = " ".join(getattr(j.status, "expose_urls", None) or [])
        print(f"{j.id}  {(j.labels or {}).get('name', '-'):<20} {stage:<11} {j.flavor}  {urls}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", help="Path to a tuft_config.yaml (required to launch)")
    # Infra flags default to None so we can tell "explicitly set" from "use file/default".
    ap.add_argument("--flavor", default=None, help="Jobs hardware: a100-large | l4x1 | h200 ...")
    ap.add_argument("--name", default=None, help="Job name (also used by --down)")
    ap.add_argument("--timeout", default=None, help="hard stop, e.g. 30m | 2h | 1d")
    ap.add_argument("--namespace", default=None, help="user or org to run (and bill) the Job")
    ap.add_argument(
        "--ckpt-bucket",
        dest="ckpt_bucket",
        default=None,
        help="Storage Bucket for checkpoint_dir (name or namespace/name)",
    )
    ap.add_argument(
        "--no-mount-models",
        dest="mount_models",
        action="store_const",
        const=False,
        default=None,
        help="download models at startup instead of mounting them from the Hub",
    )
    ap.add_argument(
        "--public",
        action="store_const",
        const=True,
        default=None,
        help="expose the port WITHOUT the HF token layer (only the tml- key protects it)",
    )
    ap.add_argument(
        "--network-group",
        dest="network_group",
        default=None,
        help="join a Jobs network group; other members reach TuFT at <prefix>tuft:10610",
    )
    ap.add_argument(
        "--hf-token",
        dest="hf_token",
        action="store_const",
        const=True,
        default=None,
        help="forward your HF token into the Job as HF_TOKEN (gated models, --no-mount-models)",
    )
    ap.add_argument("--image", default=None)
    ap.add_argument("--startup-timeout", dest="startup_timeout", type=int, default=None)
    ap.add_argument("--no-wait", action="store_true", help="return once the Job is submitted")
    ap.add_argument("--down", "--stop", dest="down", action="store_true", help="cancel the Job")
    ap.add_argument("--job-id", dest="job_id", default=None, help="Job to cancel with --down")
    ap.add_argument("--status", action="store_true", help="list TuFT Jobs and exit")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the resolved Job but do not submit it"
    )
    args = ap.parse_args()

    if args.status:
        return status(hf_api(), args)
    if args.down:
        return down(hf_api(), args)

    if not args.config:
        sys.exit("--config is required to launch (or use --down / --status)")
    cfg = Path(args.config).expanduser().resolve()
    if not cfg.is_file():
        sys.exit(f"config not found: {cfg}")

    data = load_config(cfg)
    infra = resolve_infra(args, data.get("hf_jobs") or {})
    config_b64, repos = prepare_server_config(data, infra["mount_models"])

    api = hf_api()
    from huggingface_hub import Volume

    namespace = infra["namespace"] or api.whoami()["name"]
    bucket = infra["ckpt_bucket"]
    bucket = bucket if "/" in bucket else f"{namespace}/{bucket}"
    volumes = [Volume(type="bucket", source=bucket, mount_path=CKPT_DIR)]
    volumes += [Volume(type="model", source=r, mount_path=f"{MODELS_DIR}/{r}") for r in repos]
    secrets = {"TUFT_CONFIG_B64": config_b64}
    if infra["hf_token"]:
        secrets["HF_TOKEN"] = get_hf_token()
    port_kwargs = {"expose_public": [TUFT_PORT]} if infra["public"] else {"expose": [TUFT_PORT]}
    net_kwargs = (
        {"network_group": infra["network_group"], "network_aliases": ["tuft"]}
        if infra["network_group"]
        else {}
    )

    print(
        f"[job] {namespace}/{infra['name']}  flavor={infra['flavor']} timeout={infra['timeout']} "
        f"image={infra['image']}\n"
        f"      checkpoints=hf://buckets/{bucket}  mounted models={repos or 'none'}  "
        f"port={TUFT_PORT} ({'public' if infra['public'] else 'HF token required'})"
    )
    if args.dry_run:
        print("[dry-run] not submitting.")
        return

    api.create_bucket(bucket, private=True, exist_ok=True)
    from huggingface_hub.errors import HfHubHTTPError

    try:
        job = api.run_job(
            image=infra["image"],
            command=["bash", "-c", SERVER_CMD],
            secrets=secrets,
            flavor=infra["flavor"],
            timeout=infra["timeout"],
            name=infra["name"],
            labels=JOB_LABELS,
            volumes=volumes,
            namespace=namespace,
            **port_kwargs,
            **net_kwargs,
        )
    except HfHubHTTPError as e:
        # Mounting a gated model you haven't been granted is rejected here, before billing.
        sys.exit(f"Job submission failed: {e}\n(gated model? request access on its Hub page)")
    url = f"https://{job.id}--{TUFT_PORT}.hf.jobs"
    model = data["supported_models"][0].get("model_name", "?")
    stop = f"python {Path(sys.argv[0]).as_posix()} --down --job-id {job.id} --namespace {namespace}"
    print(f"[job] submitted {job.id}: {job.url}")
    if args.no_wait:
        print(f"[job] TuFT will be at {url} once healthy — `hf jobs logs -f {namespace}/{job.id}`")
        print(f"[job] stop it with `{stop}`")
        return
    headers = {} if infra["public"] else {"Authorization": f"Bearer {get_hf_token()}"}
    if not wait_healthy(api, job.id, namespace, url, headers, infra["startup_timeout"]):
        sys.exit(
            f"TuFT did not become healthy — check `hf jobs logs {namespace}/{job.id}`, then "
            f"`{stop}`"
        )
    banner(url, model, infra["public"], job.id, namespace, stop)


if __name__ == "__main__":
    main()
