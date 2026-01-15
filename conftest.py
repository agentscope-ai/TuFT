import os

import pytest

from llm_rpc.persistence import (
    REDIS_AVAILABLE,
    PersistenceSettings,
    RedisConnection,
)

# Redis for test
TEST_REDIS_DB = 15
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", f"redis://localhost:6379/{TEST_REDIS_DB}")


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="run tests that require GPU")
    parser.addoption(
        "--no-redis-clean",
        action="store_true",
        default=False,
        help="skip Redis database cleanup before/after tests",
    )
    parser.addoption(
        "--persistence",
        action="store_true",
        default=False,
        help="enable persistence for tests (requires Redis)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "persistence: mark test as requiring persistence")


@pytest.fixture(autouse=True, scope="session")
def set_cpu_env(request):
    if not request.config.getoption("--gpu"):
        os.environ["LLM_RPC_CPU_TEST"] = "1"


def _clear_redis_db(redis_url: str) -> None:
    """Clear all keys in the specified Redis database."""
    if not REDIS_AVAILABLE:
        return

    try:
        import redis

        r = redis.Redis.from_url(redis_url, decode_responses=True)
        r.flushdb()
        r.close()
    except Exception:
        # Redis not available, skip cleanup
        pass


@pytest.fixture(autouse=True, scope="function")
def configure_persistence(request):
    """Configure persistence settings for all tests.

    - By default, persistence is DISABLED for tests
    - Use --persistence flag to enable persistence testing
    - Uses a dedicated test database (db 15 by default) to avoid conflicts
    - Clears the database before each test for isolation
    - Can be skipped with --no-redis-clean flag
    """
    # Reset persistence settings before each test
    PersistenceSettings.reset()

    # Check if persistence should be enabled for this test
    enable_persistence = request.config.getoption("--persistence", default=False)

    # Also check for persistence marker on the test
    if "persistence" in [marker.name for marker in request.node.iter_markers()]:
        enable_persistence = True

    if enable_persistence and REDIS_AVAILABLE:
        # Persistence tests set their own REDIS_URL, don't override it
        if "REDIS_URL" not in os.environ:
            redis_url = TEST_REDIS_URL
        else:
            redis_url = os.environ["REDIS_URL"]

        # Clean Redis before each test (unless disabled)
        if not request.config.getoption("--no-redis-clean"):
            _clear_redis_db(redis_url)

        # Configure persistence
        PersistenceSettings.configure(
            enabled=True,
            redis_url=redis_url,
            namespace="llm_rpc_test",
            instance_id=None,
        )
        RedisConnection.configure(redis_url)
    else:
        # Persistence disabled
        PersistenceSettings.configure(
            enabled=False,
            namespace="llm_rpc_test",
            instance_id=None,
        )

    yield

    # Clean up after test
    if PersistenceSettings.is_enabled():
        RedisConnection.close()
    PersistenceSettings.reset()


@pytest.fixture(scope="function")
def clean_redis():
    """Explicit fixture for tests that need guaranteed clean Redis state.

    Use this fixture when you need to ensure Redis is clean before your test,
    especially for persistence-related tests.
    """
    if REDIS_AVAILABLE:
        _clear_redis_db(TEST_REDIS_URL)
    yield
    if REDIS_AVAILABLE:
        _clear_redis_db(TEST_REDIS_URL)


@pytest.fixture(scope="function")
def enable_persistence():
    """Fixture to enable persistence for a specific test.

    Use this fixture for tests that require persistence to be enabled.
    """
    if not REDIS_AVAILABLE:
        pytest.skip("Redis dependencies not installed")

    PersistenceSettings.reset()
    _clear_redis_db(TEST_REDIS_URL)
    PersistenceSettings.configure(
        enabled=True,
        redis_url=TEST_REDIS_URL,
        namespace="llm_rpc_test",
        instance_id=None,
    )
    RedisConnection.configure(TEST_REDIS_URL)

    yield

    RedisConnection.close()
    _clear_redis_db(TEST_REDIS_URL)
    PersistenceSettings.reset()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip persistence tests if --persistence flag is not set and Redis is not available
    if not config.getoption("--persistence") and not REDIS_AVAILABLE:
        skip_persistence = pytest.mark.skip(
            reason="need --persistence option and Redis to run persistence tests"
        )
        for item in items:
            if "persistence" in item.keywords:
                item.add_marker(skip_persistence)
