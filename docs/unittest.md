# How to Write and Run Unit Tests

LLM-RPC supports running unit tests on both CPU and GPU devices. Below are the instructions for each setup.

## On CPU devices

To write and run unit tests on CPU devices, you do not need any special configuration. Simply write your tests as usual and run them using pytest:

```bash
pytest tests -v -s
```

The `conftest.py` will automatically set the environment variable `LLM_RPC_CPU_TEST=1`, which will configure the backends to use CPU-compatible implementations during testing. And the tests marked with `@pytest.mark.gpu` will be skipped automatically.

## On GPU devices

To write tests that run on GPU devices, you can use the `@pytest.mark.gpu` decorator to mark those tests. For example:

```python
import pytest

@pytest.mark.gpu
def test_gpu_functionality():
    # Your test code that requires GPU
    pass
```

To run the GPU tests, you need set a model path via the environment variable `LLM_RPC_TEST_MODEL`, start a Ray cluster, and then execute pytest with the `--gpu` option:

```bash
export LLM_RPC_TEST_MODEL=/path/to/your/model
ray start --head
pytest tests -v -s --gpu
```

This will execute all tests, including those marked with `@pytest.mark.gpu`, on GPU devices.
