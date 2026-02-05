import logging

from console_config import CONSOLE_SERVER_PROXY
from console_server.services.base_proxy import DataProxy
from console_server.services.proxy_registry import PROXY_REGISTRY
from fastapi import Header, HTTPException


logger = logging.getLogger(__name__)

_proxy_instances: dict[str, DataProxy] = {}


def get_data_proxy() -> DataProxy:
    proxy_type = CONSOLE_SERVER_PROXY

    if proxy_type not in PROXY_REGISTRY:
        available = ", ".join(PROXY_REGISTRY.keys())
        raise ValueError(f"Unsupported proxy type: '{proxy_type}'. Available options: {available}")

    if proxy_type not in _proxy_instances:
        cls = PROXY_REGISTRY[proxy_type]
        logger.info(f"Initializing {cls.__name__} as data proxy")
        _proxy_instances[proxy_type] = cls()

    return _proxy_instances[proxy_type]


def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if not x_api_key or not x_api_key.strip():
        raise HTTPException(status_code=400, detail="Missing X-API-Key header")
    return x_api_key
