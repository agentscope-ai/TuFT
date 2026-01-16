import os
import tempfile
from pathlib import Path

import pytest

TEST_REDIS_DB = 15
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", f"redis://localhost:6379/{TEST_REDIS_DB}")

# Global variable to store the temporary redislite db path for tests
_test_redislite_dir: Path | None = None


def _get_test_redislite_path() -> Path:
    """Get the path to the test redislite database file."""
    global _test_redislite_dir
    if _test_redislite_dir is None:
        _test_redislite_dir = Path(tempfile.mkdtemp(prefix="tuft_test_redis_"))
    return _test_redislite_dir / "test_redis.db"


def _redislite_available() -> bool:
    """Check if redislite is available for testing."""
    import importlib.util

    return importlib.util.find_spec("redislite") is not None


def _redis_available() -> bool:
    """Check if external Redis is available for testing."""
    try:
        import redis

        r = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


def _clear_redislite_db(db_path: Path) -> None:
    """Clear redislite database by flushing it."""
    try:
        import redislite

        if db_path.exists():
            r = redislite.Redis(str(db_path), decode_responses=True)
            r.flushdb()
            r.close()
    except Exception:
        pass


def _clear_redis_db(redis_url: str) -> None:
    """Clear all keys in the specified Redis database."""
    try:
        import redis

        r = redis.Redis.from_url(redis_url, decode_responses=True)
        r.flushdb()
        r.close()
    except Exception:
        pass


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="run tests that require GPU")
    parser.addoption(
        "--persistence",
        action="store_true",
        default=False,
        help="enable persistence for tests (uses redislite by default)",
    )
    parser.addoption(
        "--use-external-redis",
        action="store_true",
        default=False,
        help="use external Redis instead of redislite for persistence tests",
    )
    parser.addoption(
        "--no-redis-clean",
        action="store_true",
        default=False,
        help="skip Redis database cleanup before/after tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "persistence: mark test as requiring persistence")


@pytest.fixture(autouse=True, scope="session")
def set_cpu_env(request):
    if not request.config.getoption("--gpu"):
        os.environ["TUFT_CPU_TEST"] = "1"


@pytest.fixture(autouse=True, scope="function")
def configure_persistence(request):
    """Configure persistence settings for all tests.

    By default, tests use redislite for persistence when --persistence is enabled.
    Use --use-external-redis to use an external Redis server instead.
    """
    from tuft.persistence import PersistenceConfig, get_redis_store

    store = get_redis_store()
    store.reset()

    enable_persistence = request.config.getoption("--persistence", default=False)
    use_external_redis = request.config.getoption("--use-external-redis", default=False)

    if "persistence" in [marker.name for marker in request.node.iter_markers()]:
        enable_persistence = True

    if enable_persistence:
        if use_external_redis and _redis_available():
            # Use external Redis server
            redis_url = os.getenv("REDIS_URL", TEST_REDIS_URL)

            if not request.config.getoption("--no-redis-clean"):
                _clear_redis_db(redis_url)

            store.configure(PersistenceConfig.from_redis_url(redis_url, namespace="tuft_test"))
        elif _redislite_available():
            # Use lightweight redislite (default for tests)
            redislite_path = _get_test_redislite_path()

            if not request.config.getoption("--no-redis-clean"):
                _clear_redislite_db(redislite_path)

            store.configure(PersistenceConfig.from_redislite(redislite_path, namespace="tuft_test"))
        else:
            # No persistence available
            store.configure(PersistenceConfig.disabled(namespace="tuft_test"))
    else:
        store.configure(PersistenceConfig.disabled(namespace="tuft_test"))

    yield

    if store.is_enabled:
        store.close()
    store.reset()


@pytest.fixture(scope="function")
def clean_redis():
    """Explicit fixture for tests that need guaranteed clean Redis state.

    Uses redislite by default for tests.
    """
    if _redislite_available():
        _clear_redislite_db(_get_test_redislite_path())
    elif _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
    yield
    if _redislite_available():
        _clear_redislite_db(_get_test_redislite_path())
    elif _redis_available():
        _clear_redis_db(TEST_REDIS_URL)


@pytest.fixture(scope="function")
def enable_persistence():
    """Fixture to enable persistence for a specific test.

    Uses redislite by default for lightweight embedded Redis.
    """
    from tuft.persistence import PersistenceConfig, get_redis_store

    store = get_redis_store()
    store.reset()

    if _redislite_available():
        # Use redislite (default for tests)
        redislite_path = _get_test_redislite_path()
        _clear_redislite_db(redislite_path)
        store.configure(PersistenceConfig.from_redislite(redislite_path, namespace="tuft_test"))
    elif _redis_available():
        # Fall back to external Redis if redislite not available
        _clear_redis_db(TEST_REDIS_URL)
        store.configure(PersistenceConfig.from_redis_url(TEST_REDIS_URL, namespace="tuft_test"))
    else:
        pytest.skip("Neither redislite nor Redis available")

    yield

    store.close()
    if _redislite_available():
        _clear_redislite_db(_get_test_redislite_path())
    elif _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
    store.reset()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Persistence tests can run with redislite (default) or external Redis
    has_persistence_backend = _redislite_available() or _redis_available()
    if not config.getoption("--persistence") and not has_persistence_backend:
        skip_persistence = pytest.mark.skip(
            reason="need --persistence option and redislite/Redis to run persistence tests"
        )
        for item in items:
            if "persistence" in item.keywords:
                item.add_marker(skip_persistence)
