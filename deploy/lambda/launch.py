#!/usr/bin/env python
"""
Run TuFT on a Lambda Cloud GPU instance FROM A CONFIG FILE — mirrors deploy/modal/launch.py.

You edit a standard tuft_config.yaml (the same file `tuft launch --config` uses) and run
this script. It validates the config, embeds it, launches a Lambda GPU instance that
self-bootstraps TuFT (Docker), and prints how to connect. To change models/keys/options,
edit the YAML and re-run.

Verbs (flag-based, like the Modal launch.py — default action is "launch"):
    python deploy/lambda/launch.py --config my.yaml            # launch a new instance
    python deploy/lambda/launch.py --config my.yaml --instance-id <id>   # reuse an instance
    python deploy/lambda/launch.py --down   --config my.yaml   # terminate (by name) / --instance-id / --all
    python deploy/lambda/launch.py --status                    # list instances

Auto instance selection (so you don't think about hardware): if you don't pin `--instance-type`,
it picks a 1xGPU type with capacity, PREFERRING a100 — the cheaper a10 has a known training
issue (see the a10 note in deploy/README.md). Filter with a `--gpu` hint, e.g. `--gpu a100`.
Infra can also live in a `lambda:` section in the YAML
(stripped before the server sees the config); precedence is flag > `lambda:` > default.

Bootstrap: a NEW instance self-bootstraps via cloud-init `user_data` (no SSH needed). An
EXISTING instance (`--instance-id`) is bootstrapped over SSH (cloud-init only runs at first
boot). Checkpoints + HF cache live under a host dir bind-mounted at /data; pass `--filesystem`
for a Lambda persistent filesystem so they survive termination.

Auth: export LAMBDA_API_KEY (Lambda Cloud console → API keys). Needs `pip install pyyaml`.

Security: port 10610 is the TuFT API, protected by your `tml-` X-API-Key. Prefer reaching it
over an SSH tunnel (`ssh -N -L 10610:localhost:10610 ubuntu@<ip>`) rather than the public IP.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_API_BASE = "https://cloud.lambda.ai/api/v1"
TUFT_IMAGE = "ghcr.io/agentscope-ai/tuft:latest"
TUFT_PORT = 10610
HEALTHZ_PATH = "/api/v1/healthz"

# Container paths (host <dir> bind-mounted at /data; matches the README docker-run convention).
CONTAINER_DATA_DIR = "/data"
CONTAINER_CONFIG_PATH = "/data/tuft_config.yaml"
CONTAINER_HF_CACHE = "/data/hf-cache"
CONTAINER_CHECKPOINT_DIR = "/data/checkpoints"

# Infra knobs (CLI flags and the YAML `lambda:` section share these keys) + defaults.
DEFAULTS = {
    "instance_type": "",   # full Lambda type, e.g. gpu_1x_a100_sxm4; "" = auto-pick
    "gpu": "",             # family hint(s), e.g. "a100"; "" = auto (prefers a100; a10 last)
    "region": "",          # "" = auto (a region with capacity)
    "ssh_key": "",         # "" = use the account's sole key if exactly one
    "name": "tuft-server",
    "filesystem": "",      # Lambda persistent filesystem name; "" = root disk (ephemeral)
    "image": TUFT_IMAGE,
    "shm_size": "64g",     # bump to 128g on big boxes
    "hf_token": "",        # for gated models; also read from env HF_TOKEN
}
_INFRA_KEYS = list(DEFAULTS)

# Auto-select order when no --gpu hint: PREFER a100 over the cheaper a10. gpu_1x_a10 (sm_86)
# has a known forward_backward bug in the current image (serving/inference are fine, but
# TRAINING returns null logprobs), so a10 is the last resort.
_DEFAULT_PREF = ["a100", "h100", "gh200", "l40", "a6000", "rtx6000", "a10"]


# -----------------------------------------------------------------------------
# Lambda Cloud API client (stdlib only)
# -----------------------------------------------------------------------------
class LambdaAPIError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        self.status, self.body, self.url = status, body, url
        super().__init__(f"Lambda API {status} for {url}: {body}")


class LambdaClient:
    def __init__(self, api_key: str, api_base: str = DEFAULT_API_BASE, use_bearer: bool = False):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.use_bearer = use_bearer

    def _auth_header(self) -> Dict[str, str]:
        if self.use_bearer:
            return {"Authorization": f"Bearer {self.api_key}"}
        token = base64.b64encode(f"{self.api_key}:".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.api_base}{path}"
        data = None
        # A non-default User-Agent is required: Lambda's API sits behind a WAF that 403s
        # the stdlib "Python-urllib/x.y" UA.
        headers = {"Accept": "application/json", "User-Agent": "tuft-launch/1.0", **self._auth_header()}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise LambdaAPIError(e.code, e.read().decode("utf-8", errors="replace"), url) from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error calling {url}: {e.reason}") from None
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}

    def instance_types(self) -> Dict[str, Any]:
        return self.request("GET", "/instance-types").get("data", {})

    def ssh_keys(self) -> List[Dict[str, Any]]:
        return self.request("GET", "/ssh-keys").get("data", [])

    def launch(self, payload: Dict[str, Any]) -> List[str]:
        return self.request("POST", "/instance-operations/launch", payload).get("data", {}).get("instance_ids", [])

    def list_instances(self) -> List[Dict[str, Any]]:
        return self.request("GET", "/instances").get("data", [])

    def get_instance(self, instance_id: str) -> Dict[str, Any]:
        return self.request("GET", f"/instances/{instance_id}").get("data", {})

    def terminate(self, instance_ids: List[str]) -> Dict[str, Any]:
        return self.request("POST", "/instance-operations/terminate", {"instance_ids": instance_ids})


# -----------------------------------------------------------------------------
# instance-types parsing + auto-selection
# -----------------------------------------------------------------------------
def _price_of(entry: Dict[str, Any]) -> int:
    it = entry.get("instance_type", entry)
    if not isinstance(it, dict):
        return 0
    try:
        return int(it.get("price_cents_per_hour", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _regions_of(entry: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for r in entry.get("regions_with_capacity_available", []):
        if isinstance(r, dict) and r.get("name"):
            out.append(r["name"])
        elif isinstance(r, str):
            out.append(r)
    return out


def resolve_instance(client: LambdaClient, instance_type: str, gpu: str, region: str) -> Tuple[str, str, int]:
    """Return (instance_type_name, region_name, price_cents). Auto-picks a 1xGPU type with
    capacity, preferring a100 over a10 (see _DEFAULT_PREF), unless `instance_type` is pinned or
    `gpu` gives a comma-separated family hint."""
    types = client.instance_types()
    if instance_type:
        entry = types.get(instance_type)
        regions = _regions_of(entry) if isinstance(entry, dict) else []
        chosen = region or (regions[0] if regions else "")
        if not chosen:
            sys.exit(f"No capacity for {instance_type}"
                     + (f" in {region}" if region else " in any region") + ". Try later or another type.")
        return instance_type, chosen, _price_of(entry or {})

    # available 1xGPU candidates: (price, name, regions)
    avail = []
    for name, entry in types.items():
        if not isinstance(entry, dict) or not name.startswith("gpu_1x_"):
            continue
        regions = _regions_of(entry)
        if region:
            regions = [r for r in regions if r == region]
        if regions:
            avail.append((_price_of(entry), name, regions))
    if not avail:
        sys.exit("No 1xGPU capacity available right now"
                 + (f" in {region}" if region else " in any region")
                 + ". Try again later, widen --region, or pin --instance-type.")

    # With an explicit --gpu hint, honor it (cheapest matching each, in order). Otherwise use
    # the default preference (a100 first, a10 last) so training-by-default lands on a good GPU.
    prefs = [h.strip() for h in gpu.split(",") if h.strip()] if gpu else _DEFAULT_PREF
    for hint in prefs:
        matches = sorted(c for c in avail if hint in c[1])
        if matches:
            price, name, regions = matches[0]
            return name, regions[0], price
    if gpu:
        sys.exit(f"No available 1xGPU matching --gpu '{gpu}'"
                 + (f" in {region}" if region else "") + ". Available: "
                 + ", ".join(sorted(c[1] for c in avail)))
    # None of the preferred families have capacity -> cheapest of whatever is available.
    avail.sort()
    price, name, regions = avail[0]
    return name, regions[0], price


# -----------------------------------------------------------------------------
# config (mirror launch.py): validate, resolve infra, strip `lambda:` section
# -----------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    try:
        import yaml
    except Exception:
        sys.exit("pyyaml is required to read the config: pip install pyyaml")
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
    unknown = set(file_infra) - set(DEFAULTS)
    if unknown:
        print(f"[warn] unknown keys in `lambda:` section ignored: {sorted(unknown)}")
    infra = {}
    for key, dflt in DEFAULTS.items():
        cli = getattr(args, key, None)
        infra[key] = cli if cli not in (None, "") else (
            file_infra[key] if file_infra.get(key) not in (None, "") else dflt)
    if not infra["hf_token"]:
        infra["hf_token"] = os.environ.get("HF_TOKEN", "")
    return infra


_DEPLOY_SECTIONS = ("modal", "lambda")  # deploy-infra sections; stripped before the server sees the config


def config_text(path: Path, data: dict) -> str:
    """The tuft_config.yaml the server should see, with deploy-infra sections stripped."""
    if any(s in data for s in _DEPLOY_SECTIONS):
        import yaml
        clean = {k: v for k, v in data.items() if k not in _DEPLOY_SECTIONS}
        return yaml.safe_dump(clean, sort_keys=False)
    return path.read_text()


# -----------------------------------------------------------------------------
# bootstrap renderers (cloud-init for new instances; SSH for existing ones)
# -----------------------------------------------------------------------------
def _docker_run(host_data_dir: str, hf_token: str, shm_size: str, image: str) -> str:
    hf = f"-e HF_TOKEN={hf_token} " if hf_token else ""
    return (
        f"docker run -d --name tuft --restart unless-stopped --gpus all --shm-size={shm_size} "
        f"-p {TUFT_PORT}:{TUFT_PORT} {hf}-e HF_HOME={CONTAINER_HF_CACHE} "
        f"-e TUFT_CHECKPOINT_DIR={CONTAINER_CHECKPOINT_DIR} -v {host_data_dir}:{CONTAINER_DATA_DIR} "
        f"{image} tuft launch --host 0.0.0.0 --port {TUFT_PORT} "
        f"--config {CONTAINER_CONFIG_PATH} --checkpoint-dir {CONTAINER_CHECKPOINT_DIR}"
    )


def build_user_data(cfg_yaml: str, host_data_dir: str, hf_token: str, shm_size: str, image: str) -> str:
    cfg_b64 = base64.b64encode(cfg_yaml.encode("utf-8")).decode("ascii")
    run = _docker_run(host_data_dir, hf_token, shm_size, image)
    return (
        "#cloud-config\n"
        "# TuFT bootstrap generated by launch.py\n"
        "runcmd:\n"
        f"  - mkdir -p {host_data_dir} {host_data_dir}/hf-cache {host_data_dir}/checkpoints\n"
        f"  - echo {cfg_b64} | base64 -d > {host_data_dir}/tuft_config.yaml\n"
        "  - 'which docker || (curl -fsSL https://get.docker.com | sh)'\n"
        f"  - docker pull {image}\n"
        f"  - {run}\n"
    )


def ssh_bootstrap(ip: str, cfg_yaml: str, host_data_dir: str, hf_token: str, shm_size: str, image: str) -> List[str]:
    cfg_b64 = base64.b64encode(cfg_yaml.encode("utf-8")).decode("ascii")
    remote = (
        f"sudo mkdir -p {host_data_dir} {host_data_dir}/hf-cache {host_data_dir}/checkpoints && "
        f"echo {cfg_b64} | base64 -d | sudo tee {host_data_dir}/tuft_config.yaml >/dev/null && "
        f"(which docker || (curl -fsSL https://get.docker.com | sudo sh)) && "
        f"sudo docker rm -f tuft 2>/dev/null; sudo docker pull {image} && sudo {_docker_run(host_data_dir, hf_token, shm_size, image)}"
    )
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", f"ubuntu@{ip}", remote]


def _ip(inst: Dict[str, Any]) -> str:
    return inst.get("ip") or inst.get("public_ip") or inst.get("private_ip") or ""


def _host_data_dir(filesystem: str) -> str:
    # Lambda mounts a persistent filesystem at /home/ubuntu/<name>; root disk is ephemeral.
    return f"/home/ubuntu/{filesystem}/tuft-data" if filesystem else "/home/ubuntu/tuft-data"


def wait_active(client: LambdaClient, instance_id: str, timeout: float = 900.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        inst = client.get_instance(instance_id)
        if inst.get("status") == "active" and _ip(inst):
            return inst
        print(f"   ...{inst.get('status', '?')} (waiting for active + IP)", flush=True)
        time.sleep(10)
    raise SystemExit(f"instance {instance_id} did not become active within {timeout:.0f}s")


def _banner(ip: str, api_key: str, model: str, instance_id: str, name: str) -> None:
    print("\n" + "=" * 70)
    print(f"✅ TuFT launching on Lambda  (instance {instance_id}, name '{name}', {ip})")
    print("=" * 70)
    print("The instance is bootstrapping (docker pull + model load) — this takes a few minutes.")
    print("\nConnect securely over an SSH tunnel (recommended; keeps :10610 off the public net):")
    print(f"    ssh -N -L {TUFT_PORT}:localhost:{TUFT_PORT} ubuntu@{ip}")
    print(f"    # then health-gate:  curl http://localhost:{TUFT_PORT}{HEALTHZ_PATH}")
    print("\nThen drive training from your laptop:")
    print(f"    python examples/personality_sft/train.py \\")
    print(f"        --base-url http://localhost:{TUFT_PORT} --api-key {api_key} --model {model}")
    print("\nStop billing when done (Lambda has no scale-to-zero — you MUST terminate):")
    print(f"    python deploy/lambda/launch.py --down --instance-id {instance_id}")
    print("=" * 70)


def _first_key(users: dict) -> str:
    return next(iter(users), "tml-...")


def _model(data: dict) -> str:
    return (data.get("supported_models") or [{}])[0].get("model_name", "")


# -----------------------------------------------------------------------------
# actions
# -----------------------------------------------------------------------------
def do_launch(client: LambdaClient, args: argparse.Namespace, data: dict, infra: dict, cfg_path: Path) -> int:
    cfg_yaml = config_text(cfg_path, data)
    host_dir = _host_data_dir(infra["filesystem"])
    api_key = _first_key(data["authorized_users"])
    model = _model(data)

    # Reuse an existing instance: bootstrap over SSH (cloud-init only runs at first boot).
    if args.instance_id:
        inst = client.get_instance(args.instance_id)
        ip = _ip(inst)
        if not ip:
            sys.exit(f"instance {args.instance_id} has no IP (status {inst.get('status')})")
        cmd = ssh_bootstrap(ip, cfg_yaml, host_dir, infra["hf_token"], infra["shm_size"], infra["image"])
        print(f"[reuse] bootstrapping TuFT on existing instance {args.instance_id} ({ip}) over SSH")
        if args.dry_run:
            print("[dry-run]", " ".join(cmd[:-1]), repr(cmd[-1])[:120] + "...")
            return 0
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit("SSH bootstrap failed (check your SSH key / instance reachability)")
        _banner(ip, api_key, model, args.instance_id, inst.get("name", "?"))
        return 0

    # Launch a NEW instance with auto-selected hardware + cloud-init bootstrap.
    itype, region, price = resolve_instance(client, infra["instance_type"], infra["gpu"], infra["region"])
    ssh_key = infra["ssh_key"]
    if not ssh_key:
        keys = [k.get("name") for k in client.ssh_keys() if k.get("name")]
        if len(keys) == 1:
            ssh_key = keys[0]
        else:
            sys.exit(f"Specify --ssh-key (account has {len(keys)} keys: {keys}). "
                     "Register one in the Lambda console first if none.")
    user_data = build_user_data(cfg_yaml, host_dir, infra["hf_token"], infra["shm_size"], infra["image"])
    payload: Dict[str, Any] = {
        "region_name": region,
        "instance_type_name": itype,
        "ssh_key_names": [ssh_key],
        "name": infra["name"],
        "user_data": user_data,
    }
    if infra["filesystem"]:
        payload["file_system_names"] = [infra["filesystem"]]

    print(f"[launch] {itype} in {region} (~${price/100:.2f}/hr), ssh_key={ssh_key}, name={infra['name']}"
          + (f", filesystem={infra['filesystem']}" if infra["filesystem"] else " (ephemeral root disk)"))
    if args.dry_run:
        print("[dry-run] would POST /instance-operations/launch; user_data:")
        print(user_data)
        return 0
    ids = client.launch(payload)
    if not ids:
        sys.exit("launch returned no instance id (capacity may have just vanished — retry)")
    inst = wait_active(client, ids[0])
    _banner(_ip(inst), api_key, model, ids[0], infra["name"])
    return 0


def do_down(client: LambdaClient, args: argparse.Namespace) -> int:
    instances = client.list_instances()
    if args.instance_id:
        ids = [args.instance_id]
    elif args.all:
        ids = [i["id"] for i in instances]
    else:
        name = args.name
        if not name and args.config:
            try:
                import yaml
                name = (yaml.safe_load(Path(args.config).read_text()).get("lambda") or {}).get("name")
            except Exception:
                pass
        name = name or DEFAULTS["name"]
        ids = [i["id"] for i in instances if i.get("name") == name]
        if not ids:
            sys.exit(f"No running instance named '{name}'. Use --instance-id, --all, or --status to inspect.")
    print(f"[down] terminating: {ids}")
    if args.dry_run:
        print("[dry-run] not terminating")
        return 0
    client.terminate(ids)
    print("Terminated. (Verify with --status; persistent filesystems, if any, keep billing.)")
    return 0


def do_status(client: LambdaClient) -> int:
    instances = client.list_instances()
    if not instances:
        print("No running instances.")
        return 0
    print(f"{'ID':24} {'NAME':18} {'TYPE':20} {'STATUS':10} IP")
    for i in instances:
        print(f"{i.get('id',''):24} {str(i.get('name','')):18} "
              f"{i.get('instance_type', {}).get('name',''):20} {i.get('status',''):10} {_ip(i)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="Path to a tuft_config.yaml (required to launch)")
    # infra (flag > `lambda:` section > default); default None so we can tell if set
    ap.add_argument("--instance-type", dest="instance_type", default=None,
                    help="pin a Lambda type, e.g. gpu_1x_a100_sxm4 (default: auto-pick)")
    ap.add_argument("--gpu", default=None,
                    help="family hint, e.g. a100 (default auto-pick prefers a100; a10 is last resort)")
    ap.add_argument("--region", default=None, help="pin a region (default: auto from capacity)")
    ap.add_argument("--ssh-key", dest="ssh_key", default=None,
                    help="registered Lambda SSH key name (default: the account's sole key)")
    ap.add_argument("--name", default=None, help="instance name (default: tuft-server)")
    ap.add_argument("--filesystem", default=None, help="Lambda persistent filesystem name (durable storage)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--shm-size", dest="shm_size", default=None, help="docker --shm-size (default 64g; 128g on big boxes)")
    ap.add_argument("--hf-token", dest="hf_token", default=None, help="HuggingFace token for gated models (or env HF_TOKEN)")
    # existing instance + verbs
    ap.add_argument("--instance-id", dest="instance_id", default=None, help="reuse/target an existing instance id")
    ap.add_argument("--down", "--stop", dest="down", action="store_true", help="terminate instance(s) and exit")
    ap.add_argument("--all", action="store_true", help="with --down: terminate ALL instances")
    ap.add_argument("--status", action="store_true", help="list instances and exit")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, don't call Lambda")
    ap.add_argument("--api-base", dest="api_base", default=DEFAULT_API_BASE)
    ap.add_argument("--auth-bearer", dest="auth_bearer", action="store_true",
                    help="use Bearer auth instead of HTTP Basic")
    args = ap.parse_args(argv)

    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        sys.exit("ERROR: set LAMBDA_API_KEY (Lambda Cloud console → API keys): export LAMBDA_API_KEY=secret_...")
    client = LambdaClient(api_key, args.api_base, use_bearer=args.auth_bearer)

    if args.status:
        return do_status(client)
    if args.down:
        return do_down(client, args)

    if not args.config:
        sys.exit("--config is required to launch (or use --down / --status)")
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        sys.exit(f"config not found: {cfg_path}")
    data = load_config(cfg_path)
    infra = resolve_infra(args, (data.get("lambda") or {}))
    return do_launch(client, args, data, infra, cfg_path)


if __name__ == "__main__":
    sys.exit(main())
