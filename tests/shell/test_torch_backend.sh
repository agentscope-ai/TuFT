#!/usr/bin/env bash
# Shell tests for the installer's torch backend selection logic.
#
# Run directly:  bash tests/shell/test_torch_backend.sh
# (Also run via pytest: tests/test_install_backend_selection.py.)
#
# The tests source scripts/install.sh (which is guarded against executing main
# when sourced) and exercise the shared tuft_* helpers with stubbed nvidia-smi
# and uname binaries, so they are deterministic on both GPU and GPU-less hosts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# Sourcing the installer turns on `set -euo pipefail`; relax -e afterwards so
# failed assertions report instead of aborting the whole suite.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/install.sh"
set +e
set -u

TESTS=0
FAILURES=0

# make_stub_path <cuda-version|fail> [uname-output]
# Creates a directory with a stubbed nvidia-smi (reporting the given CUDA
# version, or failing when "fail") and a stubbed uname (Linux by default), and
# prints the directory path for use as a PATH prefix. Stubbing uname by default
# keeps the Linux backend tests deterministic when this suite runs on macOS.
# The directory comes from
# mktemp because this function runs in command substitution (a subshell), so
# a counter variable would not persist across calls.
make_stub_path() {
    local cuda="$1"
    local uname_out="${2:-Linux}"
    local dir
    dir="$(mktemp -d "$WORKDIR/stub-XXXXXX")"

    if [ "$cuda" = "fail" ]; then
        printf '#!/bin/sh\nexit 1\n' > "$dir/nvidia-smi"
    else
        cat > "$dir/nvidia-smi" << STUB_EOF
#!/bin/sh
if [ "\${1:-}" = "--query-gpu=driver_version" ]; then
    echo "580.65.06"
    exit 0
fi
cat << 'SMI_EOF'
Wed Aug 20 00:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: $cuda     |
|-----------------------------------------+------------------------+----------------------+
SMI_EOF
STUB_EOF
    fi
    chmod +x "$dir/nvidia-smi"

    printf '#!/bin/sh\necho "%s"\n' "$uname_out" > "$dir/uname"
    chmod +x "$dir/uname"

    echo "$dir"
}

# expect_resolve <description> <stub-path> <requested> <want-status> <want-output> [skip-gpu-checks]
expect_resolve() {
    local desc="$1" stub="$2" requested="$3" want_status="$4" want_output="$5" skip="${6:-0}"
    TESTS=$((TESTS + 1))

    local out status
    out="$(PATH="$stub:$PATH" TUFT_SKIP_GPU_CHECKS="$skip" \
        tuft_resolve_torch_backend "$requested" 2> "$WORKDIR/stderr")"
    status=$?

    if [ "$status" -ne "$want_status" ]; then
        echo "FAIL: $desc — exit status $status (wanted $want_status), output '$out'"
        sed 's/^/      stderr: /' "$WORKDIR/stderr"
        FAILURES=$((FAILURES + 1))
    elif [ -n "$want_output" ] && [ "$out" != "$want_output" ]; then
        echo "FAIL: $desc — output '$out' (wanted '$want_output')"
        sed 's/^/      stderr: /' "$WORKDIR/stderr"
        FAILURES=$((FAILURES + 1))
    else
        echo "ok: $desc"
    fi
}

# expect_equal <description> <actual> <expected>
expect_equal() {
    local desc="$1" actual="$2" expected="$3"
    TESTS=$((TESTS + 1))
    if [ "$actual" != "$expected" ]; then
        echo "FAIL: $desc — got '$actual' (wanted '$expected')"
        FAILURES=$((FAILURES + 1))
    else
        echo "ok: $desc"
    fi
}

# expect_status <description> <want-status> <command...>
expect_status() {
    local desc="$1" want_status="$2"
    shift 2
    TESTS=$((TESTS + 1))
    "$@" > /dev/null 2>&1
    local status=$?
    if [ "$status" -ne "$want_status" ]; then
        echo "FAIL: $desc — exit status $status (wanted $want_status)"
        FAILURES=$((FAILURES + 1))
    else
        echo "ok: $desc"
    fi
}

echo "=== tuft_backend_cuda_version ==="
expect_equal "cu130 maps to 13.0" "$(tuft_backend_cuda_version cu130)" "13.0"
expect_equal "cu129 maps to 12.9" "$(tuft_backend_cuda_version cu129)" "12.9"
expect_equal "cu118 maps to 11.8" "$(tuft_backend_cuda_version cu118)" "11.8"

echo "=== tuft_version_ge ==="
expect_status "13.0 >= 12.9" 0 tuft_version_ge "13.0" "12.9"
expect_status "12.9 >= 12.9" 0 tuft_version_ge "12.9" "12.9"
expect_status "12.4 >= 12.9 fails" 1 tuft_version_ge "12.4" "12.9"
expect_status "12.10 >= 12.9 (numeric, not lexicographic)" 0 tuft_version_ge "12.10" "12.9"
expect_status "12.9 >= 13.0 fails" 1 tuft_version_ge "12.9" "13.0"

echo "=== tuft_runtime_import_test ==="
PYTHON_STUB="$WORKDIR/python-stub"
cat > "$PYTHON_STUB" << 'PYTHON_STUB_EOF'
#!/bin/sh
cat > /dev/null
printf '%s\n' "$2" > "$TUFT_IMPORT_BACKEND_FILE"
exit "${TUFT_IMPORT_STUB_STATUS:-0}"
PYTHON_STUB_EOF
chmod +x "$PYTHON_STUB"
BACKEND_FILE="$WORKDIR/import-backend"
TUFT_IMPORT_BACKEND_FILE="$BACKEND_FILE" \
    expect_status "runtime import helper returns success" 0 \
    tuft_runtime_import_test "$PYTHON_STUB" cu130
expect_equal "runtime import helper forwards backend" "$(cat "$BACKEND_FILE")" cu130
TUFT_IMPORT_BACKEND_FILE="$BACKEND_FILE" TUFT_IMPORT_STUB_STATUS=7 \
    expect_status "runtime import helper propagates import failure" 7 \
    tuft_runtime_import_test "$PYTHON_STUB" cu130

echo "=== tuft_detect_driver_cuda_version ==="
STUB_130="$(make_stub_path "13.0")"
STUB_129="$(make_stub_path "12.9")"
STUB_124="$(make_stub_path "12.4")"
STUB_132="$(make_stub_path "13.2")"
STUB_NONE="$(make_stub_path "fail")"
expect_equal "parses CUDA version from nvidia-smi output" \
    "$(PATH="$STUB_130:$PATH" tuft_detect_driver_cuda_version)" "13.0"
expect_status "fails when nvidia-smi fails" 1 \
    env PATH="$STUB_NONE:$PATH" bash -c "$(declare -f tuft_detect_driver_cuda_version); tuft_detect_driver_cuda_version"

echo "=== tuft_resolve_torch_backend: auto ==="
expect_resolve "auto picks cu130 for a CUDA 13.0 driver" "$STUB_130" auto 0 cu130
expect_resolve "auto picks cu130 for a newer CUDA 13.2 driver" "$STUB_132" auto 0 cu130
expect_resolve "auto rejects an unvalidated CUDA 12.9 driver" "$STUB_129" auto 1 ""
expect_resolve "auto fails for a too-old CUDA 12.4 driver" "$STUB_124" auto 1 ""
expect_resolve "auto falls back to default without a driver" "$STUB_NONE" auto 0 default
expect_resolve "auto + skip-gpu-checks degrades too-old driver to default" "$STUB_124" auto 0 default 1
expect_resolve "auto + skip-gpu-checks degrades CUDA 12.9 to default" "$STUB_129" auto 0 default 1

echo "=== tuft_resolve_torch_backend: explicit ==="
expect_resolve "cpu resolves to cpu" "$STUB_NONE" cpu 0 cpu
expect_resolve "explicit cu130 accepted with matching driver" "$STUB_130" cu130 0 cu130
expect_resolve "unvalidated explicit cu129 accepted with newer driver" "$STUB_130" cu129 0 cu129
expect_resolve "explicit cu130 rejected with older 12.9 driver" "$STUB_129" cu130 1 ""
expect_resolve "explicit cu130 + skip-gpu-checks accepted with older driver" "$STUB_129" cu130 0 cu130 1
expect_resolve "unvalidated explicit cu129 accepted without a driver" "$STUB_NONE" cu129 0 cu129
expect_resolve "unvalidated cu126 accepted with warning" "$STUB_130" cu126 0 cu126

echo "=== tuft_resolve_torch_backend: invalid values ==="
expect_resolve "two-digit cu13 rejected" "$STUB_130" cu13 1 ""
expect_resolve "arbitrary word rejected" "$STUB_130" gpu 1 ""
expect_resolve "empty value behaves like auto (no driver -> default)" "$STUB_NONE" "" 0 default

echo "=== tuft_resolve_torch_backend: non-Linux ==="
STUB_DARWIN="$(make_stub_path "13.0" "Darwin")"
expect_resolve "auto resolves to default on macOS" "$STUB_DARWIN" auto 0 default
expect_resolve "explicit backend ignored (default) on macOS" "$STUB_DARWIN" cu130 0 default

echo "=== generated backend library (declare -f serialization) ==="
TESTS=$((TESTS + 1))
LIB_HOME="$WORKDIR/tuft-home"
mkdir -p "$LIB_HOME/scripts"
TUFT_HOME="$LIB_HOME" write_torch_backend_lib > /dev/null
LIB_PATH="$LIB_HOME/scripts/torch_backend.sh"
if [ ! -f "$LIB_PATH" ]; then
    echo "FAIL: write_torch_backend_lib did not create $LIB_PATH"
    FAILURES=$((FAILURES + 1))
else
    out="$(PATH="$STUB_130:$PATH" bash -c "source '$LIB_PATH' && tuft_resolve_torch_backend auto" 2>/dev/null)"
    if [ "$out" != "cu130" ]; then
        echo "FAIL: sourcing the generated library in a fresh shell resolved '$out' (wanted 'cu130')"
        FAILURES=$((FAILURES + 1))
    else
        echo "ok: generated library resolves backends in a fresh shell"
    fi
fi

echo ""
echo "$((TESTS - FAILURES))/$TESTS tests passed"
if [ "$FAILURES" -gt 0 ]; then
    exit 1
fi
