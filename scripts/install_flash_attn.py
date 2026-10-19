"""Fetch the FlashAttention-2 kernel used by attn_implementation: flash_attention_2.

When the flash_attn package is not installed, TuFT loads FlashAttention-2 from
the Hugging Face kernel hub (kernels-community/flash-attn2, pinned to a
validated commit) through the `kernels` package, which picks the build matching
the installed torch/CUDA/ABI (a cu129 torch uses the cu128 build). This script
fetches that build ahead of time, loads it, and runs a forward/backward smoke
test with grouped-query attention, so a missing or broken kernel shows up at
install time rather than on the first training request.

Sources:
  huggingface (default)  Download into the Hugging Face cache, where
                         transformers looks for it at runtime.
  modelscope             Download the matching build from the ModelScope mirror
                         into --kernels-dir, verify its sigstore signature
                         (huggingface/kernels-community build workflow) and file
                         digests, and record a LOCAL_KERNELS override in
                         $TUFT_HOME/local-kernels. The `tuft` wrapper exports it;
                         other launchers must export LOCAL_KERNELS themselves.

Exit codes: 0 kernel ready, 1 kernel unavailable for this environment.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import torch

from tuft.backends.attention import (
    FLASH_ATTN2_KERNEL,
    FLASH_ATTN2_KERNEL_REVISION,
    flash_attn_installed,
)


SOURCES = ("huggingface", "modelscope")
# ModelScope mirror commit whose torch211 builds match FLASH_ATTN2_KERNEL_REVISION.
MODELSCOPE_REVISION = "9cc36782d0673051d2e21f5c3985e111c353cf3a"


def environment() -> str:
    return (
        f"torch {torch.__version__}, CUDA {torch.version.cuda or 'none'}, "  # type: ignore[attr-defined]
        f"cp{sys.version_info.major}{sys.version_info.minor}, "
        f"cxx11abi={torch.compiled_with_cxx11_abi()}"
    )


def fetch_huggingface(tuft_home: Path):
    from kernels import get_kernel

    # A ModelScope override from an earlier run would shadow the Hub kernel.
    stale = tuft_home / "local-kernels"
    if stale.exists():
        stale.unlink()
        print(f"Removed the ModelScope kernel override {stale}")
    pinned = f"{FLASH_ATTN2_KERNEL}@{FLASH_ATTN2_KERNEL_REVISION}"
    print(f"Fetching {pinned} from the Hugging Face Hub...")
    return get_kernel(FLASH_ATTN2_KERNEL, revision=FLASH_ATTN2_KERNEL_REVISION)


def _modelscope_files(api: str, root: str, recursive: bool) -> list[dict]:
    query = urllib.parse.urlencode(
        {"Root": root, "Revision": MODELSCOPE_REVISION, "Recursive": str(recursive).lower()}
    )
    url = f"{api}/repo/files?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    files = (payload.get("Data") or {}).get("Files")
    if not payload.get("Success") or files is None:
        raise RuntimeError(f"ModelScope listing failed: {payload.get('Message')} ({url})")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, sha256: str) -> None:
    if dest.exists() and _sha256(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    actual = _sha256(partial)
    if actual != sha256:
        partial.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {dest.name}: expected {sha256}, got {actual}")
    partial.replace(dest)


def fetch_modelscope(kernels_dir: Path, tuft_home: Path):
    from kernels import get_local_kernel
    from kernels.variants import parse_variant, resolve_variant
    from kernels.verify import VerificationResult, verify_variant

    domain = os.environ.get("MODELSCOPE_DOMAIN", "www.modelscope.cn")
    api = f"https://{domain}/api/v1/models/{FLASH_ATTN2_KERNEL}"
    print(f"Fetching {FLASH_ATTN2_KERNEL}@{MODELSCOPE_REVISION} from ModelScope ({domain})...")

    variants = []
    for entry in _modelscope_files(api, "build", recursive=False):
        if entry["Type"] != "tree":
            continue
        try:
            variants.append(parse_variant(entry["Name"]))
        except ValueError:
            continue
    variant, _ = resolve_variant(variants)
    if variant is None:
        raise RuntimeError(f"ModelScope has no {FLASH_ATTN2_KERNEL} build for this environment")
    print(f"Selected build variant: {variant.variant_str}")

    repo_dir = kernels_dir / FLASH_ATTN2_KERNEL
    variant_dir = repo_dir / "build" / variant.variant_str
    for entry in _modelscope_files(api, f"build/{variant.variant_str}", recursive=True):
        if entry["Type"] != "blob":
            continue
        query = urllib.parse.urlencode({"Revision": MODELSCOPE_REVISION, "FilePath": entry["Path"]})
        _download(f"{api}/repo?{query}", repo_dir / entry["Path"], entry["Sha256"])

    # The mirror is only a transport: require the Hub build's signature.
    result = verify_variant(variant_dir)
    if not isinstance(result, VerificationResult.Success):
        raise RuntimeError(f"Signature verification of {variant_dir} failed: {result}")
    print("Verified the sigstore signature and file digests of the downloaded build")

    override = f"{FLASH_ATTN2_KERNEL}={repo_dir}"
    (tuft_home / "local-kernels").write_text(override + "\n")
    os.environ["LOCAL_KERNELS"] = override
    print(f"Recorded LOCAL_KERNELS={override} in {tuft_home / 'local-kernels'}")
    return get_local_kernel(repo_dir)


def smoke_test(kernel) -> None:
    """Forward and backward with grouped-query attention against SDPA."""

    if not torch.cuda.is_available():
        print("No CUDA device visible; skipping the FlashAttention smoke test.")
        return
    torch.manual_seed(0)
    shapes = ((1, 128, 8, 64), (1, 128, 2, 64), (1, 128, 2, 64))  # q, k, v (GQA 4:1)
    q, k, v = (torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in shapes)

    def run(attention):
        inputs = [t.clone().requires_grad_() for t in (q, k, v)]
        out = attention(*inputs)
        out.float().square().sum().backward()
        return [out, *(t.grad for t in inputs)]

    def flash(q, k, v):
        out = kernel.flash_attn_func(q, k, v, causal=True)
        return out[0] if isinstance(out, tuple) else out

    def sdpa(q, k, v):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True
        )
        return out.transpose(1, 2)

    for name, got, ref in zip(("out", "dq", "dk", "dv"), run(flash), run(sdpa), strict=True):
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        if rel > 2e-2:
            raise RuntimeError(f"FlashAttention {name} differs from SDPA (rel. error {rel:.3g})")
    device = torch.cuda.get_device_name()
    print(f"FlashAttention forward/backward smoke test (GQA) passed on {device}")


def main() -> int:
    tuft_home = Path(os.environ.get("TUFT_HOME", Path.home() / ".tuft"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=os.environ.get("TUFT_KERNEL_SOURCE", "huggingface"),
        help="Where to download the kernel from (default: $TUFT_KERNEL_SOURCE or huggingface)",
    )
    parser.add_argument(
        "--kernels-dir",
        type=Path,
        default=tuft_home / "kernels",
        help="Download directory for --source modelscope (default: $TUFT_HOME/kernels)",
    )
    args = parser.parse_args()

    print(f"Environment: {environment()}")
    if flash_attn_installed(2):
        import flash_attn

        print(f"flash_attn {flash_attn.__version__} is installed; transformers will use it.")
        return 0

    try:
        if args.source == "modelscope":
            tuft_home.mkdir(parents=True, exist_ok=True)
            kernel = fetch_modelscope(args.kernels_dir, tuft_home)
        else:
            kernel = fetch_huggingface(tuft_home)
        print(f"Loaded {FLASH_ATTN2_KERNEL} from {Path(kernel.__file__ or '.').parent}")
        smoke_test(kernel)
    except Exception as exc:
        other = "huggingface" if args.source == "modelscope" else "modelscope"
        print(
            f"FlashAttention-2 is unavailable for this environment ({environment()}): {exc}\n"
            "Models that set attn_implementation: flash_attention_2 cannot use it. "
            f"Retry with --source {other}, use the cu130 torch backend, "
            "or set attn_implementation: sdpa.",
            file=sys.stderr,
        )
        return 1
    print("FlashAttention-2 ready for attn_implementation: flash_attention_2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
