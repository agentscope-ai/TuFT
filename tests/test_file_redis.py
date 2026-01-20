from __future__ import annotations

import time
from pathlib import Path

from tuft.persistence.file_redis import FileRedis


def test_file_redis_persists_between_instances(tmp_path: Path) -> None:
    file_path = tmp_path / "file_redis.json"
    store = FileRedis(file_path=file_path)
    assert store.get("missing") is None

    store.set("alpha", "1")
    store.set("beta", "2")
    assert store.get("alpha") == "1"
    assert store.exists("beta") == 1

    store.close()

    reloaded = FileRedis(file_path=file_path)
    assert reloaded.get("alpha") == "1"
    assert reloaded.get("beta") == "2"


def test_file_redis_ttl_expires(tmp_path: Path) -> None:
    file_path = tmp_path / "ttl_store.json"
    store = FileRedis(file_path=file_path)

    store.setex("expiring", 0.01, "value")
    time.sleep(0.02)
    assert store.get("expiring") is None
    assert store.exists("expiring") == 0


def test_file_redis_pipeline_and_scan(tmp_path: Path) -> None:
    file_path = tmp_path / "pipeline_store.json"
    store = FileRedis(file_path=file_path)

    with store.pipeline() as pipe:
        pipe.set("key:1", "a")
        pipe.set("key:2", "b")
        pipe.set("other:1", "c")
        pipe.delete("key:2")

    keys = sorted(store.scan_iter(match="key:*"))
    assert keys == ["key:1"]
    assert store.get("key:1") == "a"
    assert store.get("key:2") is None

