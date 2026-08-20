# Development

This section covers development setup, testing, and contributing to TuFT.

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Testing Guide
:link: testing
:link-type: doc
:shadow: none

How to run tests on CPU and GPU, including persistence testing.
:::
```

## Setup Development Environment

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Install dev dependencies:

    ```bash
    uv sync --extra dev
    ```

3. Set up pre-commit hooks:

    ```bash
    uv run pre-commit install
    ```

## Linting and Type Checking

Run the linter:

```bash
uv run ruff check .
uv run ruff format .
```

Run the type checker:

```bash
uv run pyright
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write docstrings for public APIs
- Keep line length under 100 characters

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request with a clear description

```{toctree}
:maxdepth: 1
:hidden:

testing
```
