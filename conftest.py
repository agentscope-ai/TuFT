import os

import pytest
import redis

from llm_rpc.persistence import RedisConnection

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


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


@pytest.fixture(autouse=True, scope="session")
def set_cpu_env(request):
    if not request.config.getoption("--gpu"):
        os.environ["LLM_RPC_CPU_TEST"] = "1"


def _clear_redis_db(redis_url: str) -> None:
    """Clear all keys in the specified Redis database."""
    try:
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        r.flushdb()
        r.close()
    except redis.ConnectionError:
        # Redis not available, skip cleanup
        pass


@pytest.fixture(autouse=True, scope="function")
def configure_redis(request):
    """Configure Redis connection for all tests.

    - Uses a dedicated test database (db 15 by default) to avoid conflicts
    - Clears the database before each test for isolation
    - Can be skipped with --no-redis-clean flag
    """
    # Persistence tests set their own REDIS_URL, don't override it
    if "REDIS_URL" not in os.environ:
        redis_url = TEST_REDIS_URL
    else:
        redis_url = os.environ["REDIS_URL"]

    # Clean Redis before each test (unless disabled)
    if not request.config.getoption("--no-redis-clean"):
        _clear_redis_db(redis_url)

    RedisConnection.configure(redis_url)
    yield

    # Clean up after test
    RedisConnection.close()


@pytest.fixture(scope="function")
def clean_redis():
    """Explicit fixture for tests that need guaranteed clean Redis state.

    Use this fixture when you need to ensure Redis is clean before your test,
    especially for persistence-related tests.
    """
    _clear_redis_db(TEST_REDIS_URL)
    yield
    _clear_redis_db(TEST_REDIS_URL)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
