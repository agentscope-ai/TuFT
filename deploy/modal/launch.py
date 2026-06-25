#!/usr/bin/env python
"""
Deploy a TuFT server to Modal FROM A CONFIG FILE — no Python edits.

You edit a standard tuft_config.yaml (the same file `tuft launch --config` uses) and run
this script. It validates the config, embeds it, renders a self-contained Modal app, and
`modal deploy`s it. To change models / keys / options, edit the YAML and re-run.

Modal infra (GPU, scale settings, ...) can be set TWO ways, in precedence order:
    1. CLI flags          (highest)         --gpu H100 --min-containers 1 ...
    2. a `modal:` section inside the YAML    (so one file fully describes the deploy)
    3. built-in defaults  (lowest)
TuFT ignores the `modal:` section (it's stripped before the server sees the config), so
the same YAML still works with `tuft launch --config` locally / in Docker.

Usage:
    python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"   # make a key

    # everything from the file (gpu etc. come from its `modal:` section):
    python deploy/modal/launch.py --config my.yaml

    # or override per-run with flags:
    python deploy/modal/launch.py --config my.yaml --gpu L4 --serve          # ephemeral
    python deploy/modal/launch.py --config my.yaml --gpu H100:2 \
        --hf-secret huggingface --min-containers 1

Durability is baked in: HF cache + checkpoint_dir live on Modal Volumes, and
checkpoint_dir is pinned to the Volume via --checkpoint-dir regardless of the YAML.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
GENERATED_APP = HERE / "_generated_app.py"

# Infra knobs and their built-in defaults. Same keys are used for CLI flags and the
# YAML `modal:` section.
DEFAULTS = {
    "gpu": "H100",
    "name": "tuft-server",
    "scaledown": 1200,
    "startup_timeout": 600,
    "min_containers": 0,
    "max_containers": 1,
    "max_concurrent": 100,
    "proxy_auth": False,
    "hf_secret": "",
    "hf_volume": "tuft-hf-cache",
    "ckpt_volume": "tuft-checkpoints",
    "image": "ghcr.io/agentscope-ai/tuft:latest",
    "registry_secret": "",
}
_INT_KEYS = {"scaledown", "startup_timeout", "min_containers", "max_containers", "max_concurrent"}

# Fixed body of the generated Modal app (plain string — the variables it references are
# emitted as literals in the header by render_app()). No env vars at import time, so it
# imports identically at deploy time and inside the container.
APP_BODY = """
PORT = 10610
HF_HOME = "/cache/hf"
CKPT_DIR = "/data/checkpoints"
REMOTE_CONFIG = "/etc/tuft/tuft_config.yaml"
VENV_PY = "/root/.tuft/venv/bin/python"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)
ckpts = modal.Volume.from_name(CKPT_VOLUME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        TUFT_IMAGE,
        secret=modal.Secret.from_name(REGISTRY_SECRET) if USE_REGISTRY_SECRET else None,
        add_python="3.11",
    )
    .entrypoint([])
    .env({"HF_HOME": HF_HOME})
)

_secrets = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []


@app.cls(
    image=image,
    gpu=GPU,
    volumes={HF_HOME: hf_cache, CKPT_DIR: ckpts},
    secrets=_secrets,
    timeout=86400,
    scaledown_window=SCALEDOWN,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class TuftServer:
    @modal.enter()
    def warm(self):
        hf_cache.reload()
        ckpts.reload()
        os.makedirs(os.path.dirname(REMOTE_CONFIG), exist_ok=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        # The user's tuft_config.yaml, embedded at deploy time and written here.
        with open(REMOTE_CONFIG, "w") as f:
            f.write(base64.b64decode(CONFIG_B64).decode("utf-8"))

    @modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT, requires_proxy_auth=PROXY_AUTH)
    def serve(self):
        # MUST bind 0.0.0.0; --checkpoint-dir pins durability to the Volume (overrides the YAML).
        subprocess.Popen(
            [VENV_PY, "-m", "tuft", "launch", "--host", "0.0.0.0", "--port", str(PORT),
             "--config", REMOTE_CONFIG, "--checkpoint-dir", CKPT_DIR],
            env={**os.environ, "HF_HOME": HF_HOME},
        )

    @modal.exit()
    def shutdown(self):
        try:
            ckpts.commit()
        except Exception:
            pass


@app.local_entrypoint()
def _info():
    print(f"TuFT app '{APP_NAME}': gpu={GPU} proxy_auth={PROXY_AUTH} "
          f"min/max containers={MIN_CONTAINERS}/{MAX_CONTAINERS}")
"""


def load_config(path: Path) -> dict:
    """Parse + validate the TuFT config. Returns the parsed dict (incl. any `modal:` section)."""
    try:
        import yaml
    except Exception:
        print("[warn] pyyaml not available; skipping validation (and `modal:` section support)")
        return {}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        sys.exit("config error: top level must be a YAML mapping")
    models = data.get("supported_models") or []
    if not models:
        sys.exit("config error: supported_models is empty — add at least one model")
    if len(models) > 1 and any((m or {}).get("colocate") for m in models):
        sys.exit(
            "config error: colocate: true is only allowed with a SINGLE model "
            "(see deploy/README.md → 'Multiple base models')"
        )
    users = data.get("authorized_users") or {}
    if not users:
        sys.exit("config error: authorized_users is empty — add at least one 'tml-...' key")
    bad = [k for k in users if not str(k).startswith("tml-")]
    if bad:
        sys.exit(
            f"config error: authorized_users keys must start with 'tml-' "
            f"(Tinker SDK requirement): {bad}"
        )
    if any("CHANGE-ME" in str(k) for k in users):
        print("[warn] authorized_users still contains a placeholder key — set a real tml- key")
    print(f"[ok] config valid: {len(models)} model(s), {len(users)} api key(s)")
    return data


def resolve_infra(args: argparse.Namespace, file_infra: dict) -> dict:
    """Merge infra knobs: CLI flag (if given) > YAML `modal:` section > DEFAULTS."""
    unknown = set(file_infra) - set(DEFAULTS)
    if unknown:
        print(f"[warn] unknown keys in `modal:` section ignored: {sorted(unknown)}")
    infra = {}
    for key, dflt in DEFAULTS.items():
        cli_val = getattr(args, key)
        if cli_val is not None:
            infra[key] = cli_val
        elif key in file_infra and file_infra[key] is not None:
            infra[key] = file_infra[key]
        else:
            infra[key] = dflt
    for key in _INT_KEYS:
        infra[key] = int(infra[key])
    infra["proxy_auth"] = bool(infra["proxy_auth"])
    return infra


def embed_config(path: Path, data: dict) -> str:
    """Base64 of the config to ship to the container, with any `modal:` section stripped.
    If there's nothing to strip we embed the original bytes verbatim (preserves comments)."""
    if data and ("modal" in data or "lambda" in data):
        import yaml

        clean = {k: v for k, v in data.items() if k not in ("modal", "lambda")}
        payload = yaml.safe_dump(clean, sort_keys=False).encode("utf-8")
    else:
        payload = path.read_bytes()
    return base64.b64encode(payload).decode("ascii")


def render_app(infra: dict, config_b64: str) -> str:
    header = "\n".join(
        [
            "# AUTO-GENERATED by deploy/modal/launch.py — DO NOT EDIT.",
            "# Edit your tuft_config.yaml and re-run launch.py instead.",
            "import base64, os, subprocess",
            "import modal",
            "",
            f"APP_NAME = {infra['name']!r}",
            f"TUFT_IMAGE = {infra['image']!r}",
            f"GPU = {infra['gpu']!r}",
            f"SCALEDOWN = {infra['scaledown']}",
            f"STARTUP_TIMEOUT = {infra['startup_timeout']}",
            f"MIN_CONTAINERS = {infra['min_containers']}",
            f"MAX_CONTAINERS = {infra['max_containers']}",
            f"MAX_CONCURRENT = {infra['max_concurrent']}",
            f"PROXY_AUTH = {infra['proxy_auth']}",
            f"HF_SECRET = {infra['hf_secret']!r}",
            f"HF_VOLUME = {infra['hf_volume']!r}",
            f"CKPT_VOLUME = {infra['ckpt_volume']!r}",
            f"USE_REGISTRY_SECRET = {bool(infra['registry_secret'])}",
            f"REGISTRY_SECRET = {(infra['registry_secret'] or 'github-registry')!r}",
            f"CONFIG_B64 = {config_b64!r}",
        ]
    )
    return header + "\n" + APP_BODY


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", help="Path to a tuft_config.yaml (required unless --down)")
    # Infra flags default to None so we can tell "explicitly set" from "use file/default".
    ap.add_argument("--gpu", default=None, help="Modal GPU: H100 | A100-80GB | L4 | H100:2 ...")
    ap.add_argument("--name", default=None, help="Modal app name")
    ap.add_argument("--scaledown", type=int, default=None, help="idle seconds before scale-to-zero")
    ap.add_argument("--startup-timeout", dest="startup_timeout", type=int, default=None)
    ap.add_argument(
        "--min-containers",
        dest="min_containers",
        type=int,
        default=None,
        help="1 keeps it warm (no scale-to-zero) — useful for long laptop-driven loops",
    )
    ap.add_argument(
        "--max-containers",
        dest="max_containers",
        type=int,
        default=None,
        help="keep at 1 — TuFT state lives in one container",
    )
    ap.add_argument("--max-concurrent", dest="max_concurrent", type=int, default=None)
    ap.add_argument(
        "--proxy-auth",
        dest="proxy_auth",
        action="store_const",
        const=True,
        default=None,
        help="add Modal gateway auth IN FRONT of X-API-Key (breaks the plain Tinker SDK)",
    )
    ap.add_argument(
        "--no-proxy-auth",
        dest="proxy_auth",
        action="store_const",
        const=False,
        help="force proxy auth off (overrides a `modal:` section)",
    )
    ap.add_argument(
        "--hf-secret",
        dest="hf_secret",
        default=None,
        help="Modal secret name holding HF_TOKEN (only for gated models)",
    )
    ap.add_argument("--hf-volume", dest="hf_volume", default=None)
    ap.add_argument("--ckpt-volume", dest="ckpt_volume", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument(
        "--registry-secret",
        dest="registry_secret",
        default=None,
        help="Modal secret name for a PRIVATE image registry (REGISTRY_USERNAME/PASSWORD)",
    )
    ap.add_argument(
        "--serve",
        "--foreground",
        dest="serve",
        action="store_true",
        help="FOREGROUND/ephemeral (modal serve): stays up while running, Ctrl-C tears it down",
    )
    ap.add_argument(
        "--down",
        "--stop",
        dest="down",
        action="store_true",
        help="DETACHED shutdown: stop a deployed app (modal app stop <name>) and exit",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="render/print the command but do not run it"
    )
    args = ap.parse_args()

    # `down` mode: stop a previously-deployed app and exit (no config needed).
    if args.down:
        name = args.name
        if not name and args.config:
            try:
                import yaml

                doc = yaml.safe_load(Path(args.config).read_text()) or {}
                name = (doc.get("modal") or {}).get("name")
            except Exception:
                pass
        name = name or DEFAULTS["name"]
        cmd = [sys.executable, "-m", "modal", "app", "stop", "--yes", name]
        print(f"== modal app stop {name} ==")
        if args.dry_run:
            print("[dry-run]", " ".join(cmd))
            return
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(
                "Could not stop it — run `modal app list` to find the exact app name/id.",
                file=sys.stderr,
            )
        sys.exit(rc)

    if not args.config:
        sys.exit("--config is required to deploy (or use --down [--name <app>] to stop one)")
    cfg = Path(args.config).expanduser().resolve()
    if not cfg.is_file():
        sys.exit(f"config not found: {cfg}")

    data = load_config(cfg)
    infra = resolve_infra(args, (data.get("modal") or {}) if data else {})
    config_b64 = embed_config(cfg, data)

    GENERATED_APP.write_text(render_app(infra, config_b64))
    verb = "serve" if args.serve else "deploy"
    print(
        f"[render] {GENERATED_APP.name}  (app={infra['name']} gpu={infra['gpu']} "
        f"proxy_auth={infra['proxy_auth']} "
        f"min/max={infra['min_containers']}/{infra['max_containers']})"
    )

    if args.dry_run:
        print("[dry-run] not deploying.")
        return

    rc = subprocess.run([sys.executable, "-m", "modal", verb, str(GENERATED_APP)]).returncode
    if rc == 0 and not args.serve:
        print("\n✅ Deployed. Copy the URL printed above (…serve.modal.run) and connect:")
        print(
            "   tinker.ServiceClient(base_url='<url>', api_key='tml-...')  "
            "# your authorized_users key"
        )
    sys.exit(rc)


if __name__ == "__main__":
    main()
