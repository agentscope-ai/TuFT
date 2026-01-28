# How to Write and Run Tests

TuFT supports running tests on both CPU and GPU devices. The test suite includes unit tests, integration tests, and persistence tests. Below are the instructions for writing and running tests.

## Quick Start

Run all tests (CPU mode, no GPU tests):
```bash
uv run pytest
```

Run tests with verbose output:
```bash
uv run pytest -v -s
```

Skip integration tests:
```bash
uv run pytest -m "not integration"
```

## Test Configuration

The test configuration is defined in [pyproject.toml](../pyproject.toml) under `[tool.pytest.ini_options]`:

- **Test paths**: `tests/`
- **Test file pattern**: `test_*.py`
- **Async mode**: Auto-configured via pytest-asyncio
- **Markers**:
  - `integration`: Marks tests as integration tests (skip with `-m "not integration"`)
  - `gpu`: Marks tests as requiring GPU (skip automatically unless `--gpu` is provided)

## Running Tests on CPU

To run tests on CPU devices, no special configuration is needed. Simply run pytest:

```bash
uv run pytest tests -v -s
```

The [conftest.py](../tests/conftest.py) automatically sets the environment variable `TUFT_CPU_TEST=1` when `--gpu` is not specified, which configures backends to use CPU-compatible implementations during testing. Tests marked with `@pytest.mark.gpu` are automatically skipped.

### Skipping Integration Tests

Integration tests may take longer to run. To skip them:

```bash
uv run pytest -m "not integration"
```

## Running Tests on GPU

To write tests that run on GPU devices, use the `@pytest.mark.gpu` decorator:

```python
import pytest

@pytest.mark.gpu
def test_gpu_functionality():
    # Your test code that requires GPU
    pass
```

To run GPU tests, you need to:

1. Set a model path via the `TUFT_TEST_MODEL` environment variable
2. Start a Ray cluster
3. Run pytest with the `--gpu` option

```bash
export TUFT_TEST_MODEL=/path/to/your/model
ray start --head
uv run pytest tests -v -s --gpu
```

This will execute all tests, including those marked with `@pytest.mark.gpu`, on GPU devices.

## Persistence Testing

TuFT tests support persistence via Redis or FileRedis. By default, persistence is **enabled** for all tests:

- **External Redis**: If `TEST_REDIS_URL` environment variable is set and Redis is available, tests use the external Redis server
- **FileRedis fallback**: If Redis is not available, tests automatically fall back to FileRedis (file-backed storage) with unique temporary files per test

### Using External Redis for Tests

To run tests with an external Redis server:

```bash
# Start Redis (example using Docker)
docker run -d --name tuft-test-redis -p 6379:6379 redis:7-alpine

# Set the Redis URL and run tests
export TEST_REDIS_URL=redis://localhost:6379/15
uv run pytest
```

The test suite uses database 15 by default to avoid conflicts with other Redis usage.

### Disabling Persistence in Tests

To disable persistence entirely:

```bash
uv run pytest --no-persistence
```

### Writing Persistence Tests

Tests automatically have persistence configured via the `configure_persistence` fixture. For tests that need explicit persistence control, use the `enable_persistence` fixture:

```python
def test_with_persistence(enable_persistence):
    # This test runs with persistence enabled
    pass
```

## Continuous Integration

The test suite runs in CI with different configurations:

### CPU Tests (checks.yml)

Runs on every push and pull request with:
- Python versions: 3.11, 3.12, 3.13
- Redis service for persistence tests
- All linting and type checking
- CPU-only pytest execution

```bash
uv run pytest
```

### GPU Tests (unittest.yml)

Triggered by `/unittest` comment on pull requests (for collaborators/members):
- Runs on self-hosted GPU runners
- Uses Docker Compose for multi-node Ray setup
- Executes full test suite including GPU tests

```bash
pytest tests -v -s --gpu --basetemp /mnt/checkpoints
```

## Test Markers and Fixtures

### Markers

- `@pytest.mark.gpu`: Marks a test as requiring GPU hardware
- `@pytest.mark.integration`: Marks a test as an integration test

### Fixtures

- `set_cpu_env`: (autouse) Sets `TUFT_CPU_TEST=1` when not running in GPU mode
- `configure_persistence`: (autouse) Configures persistence for each test
- `enable_persistence`: Explicitly enables persistence for a specific test
- `clean_redis`: Ensures clean Redis state before and after a test

## Writing Tests

### Basic Test Structure

```python
def test_example():
    # Your test code
    assert True
```

### GPU Test

```python
import pytest

@pytest.mark.gpu
def test_gpu_feature():
    # This test only runs with --gpu flag
    pass
```

### Integration Test

```python
import pytest

@pytest.mark.integration
def test_integration_workflow():
    # This test is skipped with -m "not integration"
    pass
```

### Async Test

```python
import pytest

async def test_async_function():
    # pytest-asyncio automatically handles async tests
    await some_async_function()
```

## Common Test Commands

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with stdout/stderr output
uv run pytest -s

# Run specific test file
uv run pytest tests/test_server.py

# Run specific test function
uv run pytest tests/test_server.py::test_function_name

# Run tests matching a pattern
uv run pytest -k "test_pattern"

# Skip integration tests
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m integration

# Run GPU tests (requires GPU and model)
export TUFT_TEST_MODEL=/path/to/model
ray start --head
uv run pytest --gpu

# Disable persistence
uv run pytest --no-persistence

# Show test durations
uv run pytest --durations=10
```
