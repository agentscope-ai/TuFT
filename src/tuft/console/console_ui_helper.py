import logging
import random
from typing import List

import console_config
import requests


logger = logging.getLogger(__name__)


def generate_api_key():
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    key_part = "".join(random.choice(characters) for _ in range(16))
    return f"tml-{key_part}"


def fetch_from_console(endpoint: str, api_key: str) -> dict:
    try:
        resp = requests.get(
            f"{console_config.CONSOLE_SERVER_URL}{endpoint}",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        logger.debug(f"{console_config.CONSOLE_SERVER_URL}{endpoint}")
        if resp.status_code == 200:
            return resp.json()
        else:
            logger.debug(f"Error {resp.status_code}: {resp.text}")
            return {}
    except Exception as e:
        logger.debug(f"Request failed: {e}")
        return {}


def load_runs(api_key: str) -> list:
    if not api_key:
        return []
    data = fetch_from_console("/runs", api_key)
    return data.get("runs", [])


def load_ckpts(api_key: str) -> list:
    if not api_key:
        return []
    data = fetch_from_console("/checkpoints", api_key)
    return data.get("checkpoints", [])


def load_models(api_key: str) -> List[str]:
    if not api_key:
        return []
    data = fetch_from_console("/models", api_key)
    return [m["model_name"] for m in data.get("models", [])]


def fetch_run_detail(run_id: str, api_key: str) -> dict:
    """Fetch the training run details"""
    try:
        resp = requests.get(
            f"{console_config.CONSOLE_SERVER_URL}/runs/{run_id}",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("run_detail", {})
        else:
            logger.error(f"Error {resp.status_code}: {resp.text}")
            return {}
    except Exception as e:
        logger.error(f"Detail request failed: {e}")
        return {}


def run_sample(payload: dict, api_key: str):
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        resp = requests.post(
            f"{console_config.CONSOLE_SERVER_URL}/sample", json=payload, headers=headers, timeout=60
        )
        if resp.status_code == 200:
            samples = resp.json().get("samples", [])
            return "\n".join(samples) if samples else "No output generated."
        else:
            detail = resp.json().get("detail", "Unknown error")
            return f"❌ API Error ({resp.status_code}): {detail}"
    except Exception as e:
        return f"❌ Request failed: {str(e)}"
