import os
import tempfile
import warnings
from pathlib import Path

import pytest


TEST_REDIS_DB = 15
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL")

_test_temp_dir: Path | None = None


def _get_file_redis_path(test_name: str) -> Path:
    """Get a unique FileRedis path for a specific test in a session-scoped temp directory."""
    global _test_temp_dir
    if _test_temp_dir is None:
        _test_temp_dir = Path(tempfile.mkdtemp(prefix="tuft_test_"))
    return _test_temp_dir / f"{test_name}.json"


def _redis_available() -> bool:
    """Check if external Redis is available for testing."""
    if TEST_REDIS_URL is None:
        return False
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
    except Exception as e:
        warnings.warn(
            f"[tuft tests] Failed to clear Redis database: {e}",
            RuntimeWarning,
            stacklevel=2,
        )


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="run tests that require GPU")
    parser.addoption(
        "--no-persistence",
        action="store_true",
        default=False,
        help="disable persistence tests (uses FileRedis fallback if Redis unavailable)",
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

    Persistence is ALWAYS enabled by default unless --no-persistence is specified.
    - If Redis is available (TEST_REDIS_URL is set), uses external Redis server
    - If Redis is not available, falls back to FileRedis (file-backed storage)

    Each test uses a unique temporary file for FileRedis to ensure test isolation.
    Use --no-persistence to disable persistence entirely.
    """
    from tuft.persistence import PersistenceConfig, get_redis_store

    store = get_redis_store()
    store.reset()

    # Check if persistence should be disabled
    no_persistence = request.config.getoption("--no-persistence", default=False)

    if no_persistence:
        store.configure(PersistenceConfig(namespace="tuft_test"))  # mode=disabled by default
    else:
        # Persistence enabled - use Redis if available, otherwise FileRedis
        if _redis_available() and TEST_REDIS_URL is not None:
            # Use external Redis server
            # Clear test DB before test
            _clear_redis_db(TEST_REDIS_URL)
            store.configure(PersistenceConfig.from_redis_url(TEST_REDIS_URL, namespace="tuft_test"))
        else:
            # Redis not available - fall back to FileRedis with unique temp file per test
            test_name = request.node.name
            file_path = _get_file_redis_path(test_name)
            warnings.warn(
                f"[tuft tests] Redis unavailable; falling back to FileRedis at {file_path}",
                RuntimeWarning,
                stacklevel=2,
            )
            store.configure(
                PersistenceConfig.from_file_redis(
                    file_path=file_path,
                    namespace="tuft_test",
                )
            )

    yield

    if store.is_enabled:
        store.close()

    # Always clear storage after test for isolation
    if not no_persistence:
        if _redis_available() and TEST_REDIS_URL is not None:
            _clear_redis_db(TEST_REDIS_URL)

    store.reset()


@pytest.fixture(scope="function")
def clean_redis():
    """Explicit fixture for tests that need guaranteed clean Redis state.

    Note: For FileRedis, each test already uses a unique temp file, so no explicit
    cleanup is needed.
    """
    if _redis_available() and TEST_REDIS_URL is not None:
        _clear_redis_db(TEST_REDIS_URL)
    yield
    if _redis_available() and TEST_REDIS_URL is not None:
        _clear_redis_db(TEST_REDIS_URL)


@pytest.fixture(scope="function")
def enable_persistence(request):
    """Fixture to enable persistence for a specific test.

    Uses Redis if available, otherwise falls back to FileRedis with unique temp file.
    """
    from tuft.persistence import PersistenceConfig, get_redis_store

    store = get_redis_store()
    store.reset()

    if _redis_available() and TEST_REDIS_URL is not None:
        _clear_redis_db(TEST_REDIS_URL)
        store.configure(PersistenceConfig.from_redis_url(TEST_REDIS_URL, namespace="tuft_test"))
    else:
        test_name = request.node.name
        file_path = _get_file_redis_path(f"enable_persistence_{test_name}")
        store.configure(
            PersistenceConfig.from_file_redis(
                file_path=file_path,
                namespace="tuft_test",
            )
        )

    yield

    store.close()

    # Cleanup
    if _redis_available() and TEST_REDIS_URL is not None:
        _clear_redis_db(TEST_REDIS_URL)
    # FileRedis temp files are automatically cleaned up by the OS

    store.reset()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_runtest_logstart(location):
    print(f"\n[pytest] Running {location[0]}:{location[1]}:{location[2]}")
