import os
import warnings
from pathlib import Path

import pytest

TEST_REDIS_DB = 15
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", f"redis://localhost:6379/{TEST_REDIS_DB}")

# Default file path for FileRedis fallback
DEFAULT_FILE_REDIS_PATH = Path.home() / ".cache" / "tuft" / "test_file_redis.json"


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


def _clear_file_redis(file_path: Path | None = None) -> None:
    """Clear the FileRedis JSON file.

    Args:
        file_path: Path to the FileRedis JSON file. Uses default if None.
    """
    path = file_path or DEFAULT_FILE_REDIS_PATH
    try:
        if path.exists():
            path.unlink()
        # Also remove temp file if exists
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass


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
    - If Redis is available, uses external Redis server (DB 15 for tests)
    - If Redis is not available, falls back to FileRedis (file-backed storage)

    Storage is ALWAYS cleared before and after each test to ensure test isolation.
    Use --no-persistence to disable persistence entirely.
    """
    from tuft.persistence import PersistenceConfig, get_redis_store

    store = get_redis_store()
    store.reset()

    # Check if persistence should be disabled
    no_persistence = request.config.getoption("--no-persistence", default=False)

    if no_persistence:
        store.configure(PersistenceConfig.disabled(namespace="tuft_test"))
    else:
        # Persistence enabled - use Redis if available, otherwise FileRedis
        if _redis_available():
            # Use external Redis server (always use TEST_REDIS_URL which points to DB 15)
            # Clear test DB before test
            _clear_redis_db(TEST_REDIS_URL)
            store.configure(PersistenceConfig.from_redis_url(TEST_REDIS_URL, namespace="tuft_test"))
        else:
            # Redis not available - fall back to FileRedis
            # Clear test file before test
            warnings.warn(
                (
                    "[tuft tests] Redis unavailable; falling back to FileRedis at "
                    f"{DEFAULT_FILE_REDIS_PATH}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            _clear_file_redis(DEFAULT_FILE_REDIS_PATH)
            store.configure(
                PersistenceConfig.from_file_redis(
                    file_path=DEFAULT_FILE_REDIS_PATH,
                    namespace="tuft_test",
                )
            )

    yield

    if store.is_enabled:
        store.close()

    # Always clear storage after test for isolation
    if not no_persistence:
        if _redis_available():
            _clear_redis_db(TEST_REDIS_URL)
        else:
            _clear_file_redis(DEFAULT_FILE_REDIS_PATH)

    store.reset()


@pytest.fixture(scope="function")
def clean_redis():
    """Explicit fixture for tests that need guaranteed clean Redis/FileRedis state."""
    if _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
    else:
        _clear_file_redis(DEFAULT_FILE_REDIS_PATH)
    yield
    if _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
    else:
        _clear_file_redis(DEFAULT_FILE_REDIS_PATH)


@pytest.fixture(scope="function")
def enable_persistence():
    """Fixture to enable persistence for a specific test.

    Uses Redis if available, otherwise falls back to FileRedis.
    """
    from tuft.persistence import PersistenceConfig, get_redis_store

    store = get_redis_store()
    store.reset()

    if _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
        store.configure(PersistenceConfig.from_redis_url(TEST_REDIS_URL, namespace="tuft_test"))
    else:
        # Fall back to FileRedis
        _clear_file_redis(DEFAULT_FILE_REDIS_PATH)
        store.configure(
            PersistenceConfig.from_file_redis(
                file_path=DEFAULT_FILE_REDIS_PATH,
                namespace="tuft_test",
            )
        )

    yield

    store.close()

    # Cleanup
    if _redis_available():
        _clear_redis_db(TEST_REDIS_URL)
    else:
        _clear_file_redis(DEFAULT_FILE_REDIS_PATH)

    store.reset()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
