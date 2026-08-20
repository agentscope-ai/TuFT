"""Unit tests for scripts/verify_runtime_versions.py.

The script lives outside the package, so it is loaded by file path. These
tests cover the PEP 440-aware pin comparison (local build suffixes such as
``+cu130`` must satisfy public pins), backend extraction from local version
segments, and the backend consistency checks.
"""

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_runtime_versions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_runtime_versions", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vrv = _load_module()


def _write_pyproject(tmp_path: Path, dependencies: list[str]) -> Path:
    deps = ",\n    ".join(f'"{dep}"' for dep in dependencies)
    content = f'[project]\nname = "demo"\nversion = "0.0.0"\ndependencies = [\n    {deps},\n]\n'
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(content)
    return pyproject


@pytest.mark.parametrize(
    ("installed", "pinned", "expected"),
    [
        ("2.11.0", "2.11.0", True),
        ("2.11.0+cu129", "2.11.0", True),
        ("2.11.0+cu130", "2.11.0", True),
        ("2.11.0+cpu", "2.11.0", True),
        ("2.11.1", "2.11.0", False),
        ("2.11.0", "2.11.1", False),
        ("2.11.0+cu129", "2.11.0+cu129", True),
        ("2.11.0+cu130", "2.11.0+cu129", False),
        ("2.11.0", "2.11.0+cu129", False),
        ("0.24.0", "0.24.0", True),
        ("0.24.0.post1", "0.24.0", False),
    ],
)
def test_version_satisfies_pin(installed: str, pinned: str, expected: bool):
    assert vrv.version_satisfies_pin(installed, pinned) is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.11.0", None),
        ("2.11.0+cu129", "cu129"),
        ("2.11.0+cpu", "cpu"),
        ("not-a-version+weird", "weird"),
    ],
)
def test_local_segment(version: str, expected: str | None):
    assert vrv.local_segment(version) == expected


@pytest.mark.parametrize(
    ("local", "expected"),
    [
        (None, None),
        ("cu129", "cu129"),
        ("CU130", "cu130"),
        ("cpu", "cpu"),
        ("cpu.something", "cpu"),
        ("rocm6.2", "rocm6.2"),
    ],
)
def test_backend_from_local(local: str | None, expected: str | None):
    assert vrv.backend_from_local(local) == expected


@pytest.mark.parametrize(
    ("package", "version", "expected"),
    [
        ("torch", "2.11.0", None),
        ("torch", "2.11.0+cu130", "cu130"),
        ("vllm", "0.24.0", "cu130"),
        ("vllm", "0.24.0+cu128", "cu128"),
    ],
)
def test_package_backend(package: str, version: str, expected: str | None):
    assert vrv.package_backend(package, version) == expected


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("cu129", "12"), ("cu130", "13"), ("cu118", "11"), ("cpu", None), ("rocm6.2", None)],
)
def test_cuda_major(backend: str, expected: str | None):
    assert vrv.cuda_major(backend) == expected


def test_exact_project_pins_parses_versions_and_markers(tmp_path: Path):
    pyproject = _write_pyproject(
        tmp_path,
        [
            "numpy>=2.0.0",
            "torch==2.11.0",
            "vllm==0.24.0; sys_platform == 'linux'",
        ],
    )
    pins = vrv.exact_project_pins(pyproject)
    assert pins["torch"] == vrv.Pin("2.11.0", None)
    assert pins["vllm"] == vrv.Pin("0.24.0", "sys_platform == 'linux'")


def test_exact_project_pins_requires_all_pins(tmp_path: Path):
    pyproject = _write_pyproject(tmp_path, ["torch==2.11.0"])
    with pytest.raises(RuntimeError, match="vllm"):
        vrv.exact_project_pins(pyproject)


class TestBackendConsistency:
    def test_no_expectation_and_no_local_tags_is_clean(self):
        installed = {"torch": "2.11.0", "vllm": "0.24.0"}
        assert vrv.check_backend_consistency(installed, None) == []
        assert vrv.check_backend_consistency(installed, "default") == []

    def test_expected_cuda_backend_matches(self):
        installed = {"torch": "2.11.0+cu130", "vllm": "0.24.0"}
        assert vrv.check_backend_consistency(installed, "cu130") == []

    def test_untagged_vllm_cuda13_build_rejects_cuda12_stack(self):
        installed = {"torch": "2.11.0+cu129", "vllm": "0.24.0"}
        problems = vrv.check_backend_consistency(installed, "cu129")
        assert len(problems) == 1
        assert "CUDA major version mismatch" in problems[0]
        assert "vllm (+cu130)" in problems[0]

    def test_expected_cuda_backend_mismatch(self):
        installed = {"torch": "2.11.0+cu130", "vllm": "0.24.0"}
        problems = vrv.check_backend_consistency(installed, "cu129")
        assert len(problems) == 1
        assert "+cu130" in problems[0]

    def test_expected_cuda_backend_but_default_build(self):
        installed = {"torch": "2.11.0", "vllm": "0.24.0"}
        problems = vrv.check_backend_consistency(installed, "cu129")
        assert len(problems) == 1
        assert "no local build tag" in problems[0]

    def test_expected_cpu_backend(self):
        assert vrv.check_backend_consistency({"torch": "2.11.0+cpu"}, "cpu") == []
        problems = vrv.check_backend_consistency({"torch": "2.11.0+cu129"}, "cpu")
        assert len(problems) == 1
        assert "CPU-only" in problems[0]

    def test_unknown_expected_backend(self):
        problems = vrv.check_backend_consistency({"torch": "2.11.0"}, "gpu")
        assert len(problems) == 1
        assert "unknown expected backend" in problems[0]

    def test_cross_package_cuda_major_mismatch(self):
        installed = {"torch": "2.11.0+cu129", "vllm": "0.24.0+cu130"}
        problems = vrv.check_backend_consistency(installed, None)
        assert len(problems) == 1
        assert "CUDA major version mismatch" in problems[0]

    def test_cross_package_cuda_same_major_is_clean(self):
        installed = {"torch": "2.11.0+cu129", "vllm": "0.24.0+cu128"}
        assert vrv.check_backend_consistency(installed, None) == []

    def test_cpu_torch_with_cuda_package(self):
        installed = {"torch": "2.11.0+cpu", "vllm": "0.24.0+cu130"}
        problems = vrv.check_backend_consistency(installed, None)
        assert len(problems) == 1
        assert "CPU-only" in problems[0]


class TestVerifyRuntimeVersions:
    def _pyproject(self, tmp_path: Path) -> Path:
        return _write_pyproject(
            tmp_path,
            ["torch==2.11.0", "vllm==0.24.0; sys_platform == 'linux'"],
        )

    def test_accepts_local_build_suffixes(self, tmp_path: Path, capsys):
        versions = {"torch": "2.11.0+cu130", "vllm": "0.24.0"}
        vrv.verify_runtime_versions(self._pyproject(tmp_path), get_version=versions.__getitem__)
        out = capsys.readouterr().out
        assert "Verified torch==2.11.0+cu130" in out
        assert "(build: +cu130)" in out

    def test_rejects_version_mismatch(self, tmp_path: Path):
        versions = {"torch": "2.10.0+cu130", "vllm": "0.24.0"}
        with pytest.raises(RuntimeError, match="torch version mismatch"):
            vrv.verify_runtime_versions(self._pyproject(tmp_path), get_version=versions.__getitem__)

    def test_rejects_backend_mismatch(self, tmp_path: Path):
        versions = {"torch": "2.11.0+cu129", "vllm": "0.24.0"}
        with pytest.raises(RuntimeError, match="Backend consistency check failed"):
            vrv.verify_runtime_versions(
                self._pyproject(tmp_path),
                expected_backend="cu130",
                get_version=versions.__getitem__,
            )

    def test_skips_packages_whose_marker_does_not_apply(self, tmp_path: Path, capsys):
        pyproject = _write_pyproject(
            tmp_path,
            ["torch==2.11.0", "vllm==0.24.0; sys_platform == 'not_a_real_platform'"],
        )

        def get_version(package: str) -> str:
            if package == "vllm":
                raise importlib.metadata.PackageNotFoundError(package)
            return "2.11.0"

        vrv.verify_runtime_versions(pyproject, get_version=get_version)
        assert "Skipped vllm==0.24.0" in capsys.readouterr().out

    def test_missing_package_with_applicable_marker_fails(self, tmp_path: Path):
        def get_version(package: str) -> str:
            if package == "vllm":
                raise importlib.metadata.PackageNotFoundError(package)
            return "2.11.0"

        pyproject = _write_pyproject(
            tmp_path,
            ["torch==2.11.0", "vllm==0.24.0; python_version >= '3'"],
        )
        with pytest.raises(RuntimeError, match="vllm is pinned"):
            vrv.verify_runtime_versions(pyproject, get_version=get_version)

    def test_expected_backend_success_message(self, tmp_path: Path, capsys):
        versions = {"torch": "2.11.0+cu130", "vllm": "0.24.0"}
        vrv.verify_runtime_versions(
            self._pyproject(tmp_path),
            expected_backend="cu130",
            get_version=versions.__getitem__,
        )
        assert "Backend consistency OK (expected: cu130)" in capsys.readouterr().out
