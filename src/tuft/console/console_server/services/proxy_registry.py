from typing import Dict, Type

from .base_proxy import DataProxy
from .tinker_proxy import TinkerProxy


# from .database_proxy import DatabaseProxy  # TODO: add database proxy


PROXY_REGISTRY: Dict[str, Type[DataProxy]] = {
    "tinker": TinkerProxy,
    # "database": DatabaseProxy,
    # "mock": MockProxy,
}
