# 开发

本节涵盖开发环境设置、测试和为 TuFT 做贡献。

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 测试指南
:link: testing
:link-type: doc
:shadow: none

如何在 CPU 和 GPU 上运行测试，包括持久化测试。
:::
```

## 设置开发环境

1. 如果尚未安装 [uv](https://github.com/astral-sh/uv)，请先安装：

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. 安装开发依赖：

    ```bash
    uv sync --extra dev
    ```

3. 设置 pre-commit hooks：

    ```bash
    uv run pre-commit install
    ```

## 代码检查和类型检查

运行代码检查器：

```bash
uv run ruff check .
uv run ruff format .
```

运行类型检查器：

```bash
uv run pyright
```

## 代码风格

- 遵循 PEP 8 指南
- 为所有函数签名使用类型提示
- 为公共 API 编写文档字符串
- 保持行长度在 100 个字符以内

## Pull Request 流程

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 运行测试和代码检查
5. 提交带有清晰描述的 pull request

```{toctree}
:maxdepth: 1
:hidden:

testing
```
