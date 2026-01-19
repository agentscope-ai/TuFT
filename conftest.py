import os

import pytest

TEST_REDIS_DB = 15
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", f"redis://localhost:6379/{TEST_REDIS_DB}")


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
        help="enable persistence for tests (requires external Redis)",
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

    Uses external Redis server for persistence tests.
    """
    from tuft.persistence import PersistenceConfig, get_redis_store

    store = get_redis_store()
    store.reset()

    enable_persistence = request.config.getoption("--persistence", default=False)

    if "persistence" in [marker.name for marker in request.node.iter_markers()]:
        enable_persistence = True

    if enable_persistence:
        if _redis_available():
            # Use external Redis server
            redis_url = os.getenv("REDIS_URL", TEST_REDIS_URL)

            if not request.config.getoption("--no-redis-clean"):
                _clear_redis_db(redis_url)

            store.configure(PersistenceConfig.from_redis_url(redis_url, namespace="tuft_test"))
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
    """Explicit fixture for tests that need guaranteed clean Redis state."""
    if _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
    yield
    if _redis_available():
        _clear_redis_db(TEST_REDIS_URL)


@pytest.fixture(scope="function")
def enable_persistence():
    """Fixture to enable persistence for a specific test.

    Uses external Redis server.
    """
    from tuft.persistence import PersistenceConfig, get_redis_store

    store = get_redis_store()
    store.reset()

    if _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
        store.configure(PersistenceConfig.from_redis_url(TEST_REDIS_URL, namespace="tuft_test"))
    else:
        pytest.skip("Redis not available")

    yield

    store.close()
    if _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
    store.reset()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Persistence tests require external Redis
    has_persistence_backend = _redis_available()
    if not config.getoption("--persistence") and not has_persistence_backend:
        skip_persistence = pytest.mark.skip(
            reason="need --persistence option and Redis to run persistence tests"
        )
        for item in items:
            if "persistence" in item.keywords:
                item.add_marker(skip_persistence)
