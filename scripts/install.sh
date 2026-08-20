#!/bin/bash
# TuFT Installation Script
# Usage: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TUFT_HOME="${TUFT_HOME:-$HOME/.tuft}"
TUFT_BIN="$TUFT_HOME/bin"
TUFT_VENV="${TUFT_VENV:-$TUFT_HOME/venv}"
PYTHON_VERSION="3.12"
TUFT_PYPI_REQUIREMENT="${TUFT_PYPI_REQUIREMENT:-tuft[backend,persistence]>=0.1.8}"
TUFT_GIT_REPO="https://github.com/agentscope-ai/tuft.git"
INSTALL_FROM_SOURCE=false
LOCAL_SOURCE_PATH=""
CLEAN_INSTALL=false
# Torch/vLLM wheel variant: auto (detect from NVIDIA driver), cpu, or an
# explicit CUDA backend such as cu130/cu129 (see tuft_resolve_torch_backend).
TORCH_BACKEND="${TUFT_TORCH_BACKEND:-auto}"
# Set to 1 (or pass --skip-gpu-checks) to degrade GPU preflight and CUDA
# smoke-test failures to warnings. Exported so the shared backend helpers
# below and the generated `tuft` wrapper see the same setting.
TUFT_SKIP_GPU_CHECKS="${TUFT_SKIP_GPU_CHECKS:-0}"
export TUFT_SKIP_GPU_CHECKS
# Filled in by preflight_gpu_check: the concrete backend ("default" means the
# plain PyPI wheels) and the matching uv arguments.
RESOLVED_TORCH_BACKEND="default"
TORCH_BACKEND_UV_ARGS=""

# Print functions
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from-source)
                INSTALL_FROM_SOURCE=true
                shift
                ;;
            --local-source)
                LOCAL_SOURCE_PATH="$2"
                shift 2
                ;;
            --clean)
                CLEAN_INSTALL=true
                shift
                ;;
            --torch-backend)
                if [[ $# -lt 2 ]]; then
                    print_error "--torch-backend requires a value (auto, cpu, or cuNNN such as cu130)"
                    exit 1
                fi
                TORCH_BACKEND="$2"
                shift 2
                ;;
            --skip-gpu-checks)
                TUFT_SKIP_GPU_CHECKS=1
                export TUFT_SKIP_GPU_CHECKS
                shift
                ;;
            --help|-h)
                echo "TuFT Installation Script"
                echo ""
                echo "Usage: install.sh [options]"
                echo ""
                echo "Options:"
                echo "  --from-source          Install from GitHub instead of PyPI"
                echo "  --local-source PATH    Install from local source directory (for development/CI)"
                echo "  --clean                Remove existing installation before installing"
                echo "  --torch-backend VALUE  Torch/vLLM wheel variant: auto (default), cpu, or an"
                echo "                         explicit CUDA backend such as cu130 or cu129. 'auto'"
                echo "                         inspects the NVIDIA driver before downloading anything"
                echo "                         and picks a compatible backend; without a driver it"
                echo "                         falls back to the default PyPI wheels."
                echo "  --skip-gpu-checks      Turn GPU preflight and CUDA smoke-test failures into"
                echo "                         warnings instead of errors"
                echo "  --help, -h             Show this help message"
                echo ""
                echo "The script installs TuFT with full backend support (GPU, persistence, flash-attn)."
                echo ""
                echo "Environment Variables:"
                echo "  TUFT_HOME             Installation directory (default: ~/.tuft)"
                echo "  TUFT_VENV             Virtual environment location (default: \$TUFT_HOME/venv)"
                echo "  TUFT_TORCH_BACKEND    Default value for --torch-backend"
                echo "  TUFT_SKIP_GPU_CHECKS  Set to 1 to skip GPU preflight/smoke-test enforcement"
                echo "  TUFT_PYPI_REQUIREMENT Override the default PyPI requirement"
                echo ""
                echo "uv passthrough (network/filesystem controls, read natively by uv):"
                echo "  UV_CACHE_DIR          Cache directory used for downloads and wheels"
                echo "  UV_LINK_MODE          Package link mode: clone, copy, hardlink, or symlink"
                echo "  UV_SYSTEM_CERTS       Set to true to use the system certificate store"
                echo "  UV_DEFAULT_INDEX      Default package index URL (e.g. a PyPI mirror)"
                echo "  UV_INDEX              Additional package index URLs"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# --- Torch backend helpers ---------------------------------------------------
# The tuft_* functions below are shared with the generated `tuft` wrapper:
# write_torch_backend_lib serializes them (declare -f) into
# $TUFT_HOME/scripts/torch_backend.sh, which `tuft upgrade` sources. That keeps
# install and upgrade on the exact same backend-resolution logic. They must
# stay self-contained: no colors, no print_* helpers, diagnostics to stderr.

tuft_supported_cuda_backends() {
    # Newest first. TuFT's pinned torch/vLLM stack is built against the CUDA
    # 13.0 wheel ABI (see docker/Dockerfile); cu129 covers older 12.9 drivers.
    echo "cu130 cu129"
}

tuft_backend_cuda_version() {
    # Map a cuNNN backend to the CUDA version it needs: cu130 -> 13.0,
    # cu129 -> 12.9, cu118 -> 11.8.
    local digits="${1#cu}"
    local major="${digits%?}"
    local minor="${digits#"$major"}"
    printf '%s.%s\n' "$major" "$minor"
}

tuft_version_ge() {
    # Numeric major.minor comparison: succeeds when $1 >= $2 (12.10 >= 12.9).
    awk -v a="$1" -v b="$2" 'BEGIN {
        split(a, x, "."); split(b, y, ".");
        if (x[1] + 0 != y[1] + 0) exit !(x[1] + 0 > y[1] + 0);
        exit !(x[2] + 0 >= y[2] + 0);
    }'
}

tuft_detect_driver_cuda_version() {
    # Print the highest CUDA version the installed NVIDIA driver supports
    # (e.g. "13.0"). Fails silently when there is no working driver.
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local out version
    out="$(nvidia-smi 2>/dev/null)" || return 1
    version="$(printf '%s\n' "$out" \
        | sed -n 's/.*CUDA Version[^0-9]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    if [ -z "$version" ]; then
        out="$(nvidia-smi -q 2>/dev/null)" || return 1
        version="$(printf '%s\n' "$out" \
            | sed -n 's/.*CUDA Version[^0-9]*\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    fi
    [ -n "$version" ] || return 1
    printf '%s\n' "$version"
}

tuft_uv_supports_torch_backend() {
    uv pip install --help 2>/dev/null | grep -q -- '--torch-backend'
}

tuft_resolve_torch_backend() {
    # Usage: tuft_resolve_torch_backend <auto|cpu|cuNNN>
    # Prints the resolved backend on stdout: "cpu", "cuNNN", or "default"
    # (= plain PyPI wheels, no uv --torch-backend argument). Returns non-zero
    # with a message on stderr when the request cannot be satisfied.
    # TUFT_SKIP_GPU_CHECKS=1 degrades driver-compatibility errors to warnings.
    local requested="${1:-auto}"
    local skip_checks="${TUFT_SKIP_GPU_CHECKS:-0}"
    local driver_cuda="" backend

    # uv only applies --torch-backend on Linux; macOS wheels have one variant.
    if [ "$(uname -s)" != "Linux" ]; then
        if [ "$requested" != "auto" ]; then
            echo "Warning: --torch-backend only applies on Linux; using the default wheels." >&2
        fi
        echo "default"
        return 0
    fi

    case "$requested" in
        cpu)
            echo "cpu"
            return 0
            ;;
        cu[0-9][0-9][0-9])
            case " $(tuft_supported_cuda_backends) " in
                *" $requested "*) ;;
                *)
                    echo "Warning: torch backend '$requested' is not validated against TuFT's pinned" >&2
                    echo "         torch/vLLM stack (validated: $(tuft_supported_cuda_backends))." >&2
                    ;;
            esac
            driver_cuda="$(tuft_detect_driver_cuda_version || true)"
            if [ -z "$driver_cuda" ]; then
                echo "Warning: no working NVIDIA driver detected; installing $requested wheels anyway" >&2
                echo "         (fine for container/image builds targeting GPU machines)." >&2
            elif ! tuft_version_ge "$driver_cuda" "$(tuft_backend_cuda_version "$requested")"; then
                if [ "$skip_checks" = "1" ]; then
                    echo "Warning: the NVIDIA driver only supports CUDA <= $driver_cuda but $requested was" >&2
                    echo "         requested; continuing because GPU checks are skipped." >&2
                else
                    echo "Error: the NVIDIA driver on this machine supports CUDA <= $driver_cuda, which is" >&2
                    echo "       too old for torch backend '$requested' (needs CUDA $(tuft_backend_cuda_version "$requested"))." >&2
                    echo "       Upgrade the driver, request an older backend, or pass --skip-gpu-checks" >&2
                    echo "       (TUFT_SKIP_GPU_CHECKS=1) to install anyway." >&2
                    return 1
                fi
            fi
            echo "$requested"
            return 0
            ;;
        auto)
            driver_cuda="$(tuft_detect_driver_cuda_version || true)"
            if [ -z "$driver_cuda" ]; then
                echo "Warning: no working NVIDIA driver detected (torch backend 'auto')." >&2
                echo "         Falling back to the default PyPI wheels; GPU execution will not work" >&2
                echo "         on this machine. Pass --torch-backend cpu or cuNNN (e.g. cu130) to" >&2
                echo "         make the choice explicit." >&2
                echo "default"
                return 0
            fi
            for backend in $(tuft_supported_cuda_backends); do
                if tuft_version_ge "$driver_cuda" "$(tuft_backend_cuda_version "$backend")"; then
                    echo "$backend"
                    return 0
                fi
            done
            if [ "$skip_checks" = "1" ]; then
                echo "Warning: the NVIDIA driver only supports CUDA <= $driver_cuda, older than all" >&2
                echo "         supported backends ($(tuft_supported_cuda_backends)); using the default wheels" >&2
                echo "         because GPU checks are skipped." >&2
                echo "default"
                return 0
            fi
            echo "Error: the NVIDIA driver on this machine supports CUDA <= $driver_cuda, but TuFT's" >&2
            echo "       pinned GPU stack requires one of: $(tuft_supported_cuda_backends)." >&2
            echo "       Upgrade the NVIDIA driver, install a CPU-only environment with" >&2
            echo "       --torch-backend cpu, or pass --skip-gpu-checks to use the default wheels." >&2
            return 1
            ;;
        *)
            echo "Error: invalid torch backend '$requested' (expected: auto, cpu, or cuNNN such as cu130)." >&2
            return 1
            ;;
    esac
}

tuft_cuda_smoke_test() {
    # Usage: tuft_cuda_smoke_test <python-binary>
    # Verifies that the installed torch build can initialize CUDA and run a
    # minimal operation on the GPU.
    "$1" - <<'TUFT_SMOKE_EOF'
import sys

import torch

print(f"torch {torch.__version__} (CUDA build: {torch.version.cuda})")
if not torch.cuda.is_available():
    print("CUDA smoke test failed: torch.cuda.is_available() is False.", file=sys.stderr)
    sys.exit(1)
try:
    x = torch.ones(8, device="cuda:0")
    total = float((x + x).sum().item())
except Exception as exc:
    print(f"CUDA smoke test failed while running a CUDA op: {exc}", file=sys.stderr)
    sys.exit(1)
if total != 16.0:
    print(f"CUDA smoke test failed: unexpected result {total}.", file=sys.stderr)
    sys.exit(1)
count = torch.cuda.device_count()
print(f"CUDA smoke test OK: {count} GPU(s), device 0: {torch.cuda.get_device_name(0)}")
TUFT_SMOKE_EOF
}

# --- End of shared torch backend helpers -------------------------------------

# Detect OS and architecture
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)
            PLATFORM="linux"
            ;;
        Darwin)
            PLATFORM="macos"
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac

    case "$ARCH" in
        x86_64|amd64)
            ARCH="x86_64"
            ;;
        arm64|aarch64)
            ARCH="aarch64"
            ;;
        *)
            print_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    print_step "Detected platform: $PLATFORM ($ARCH)"
}

# Resolve the torch backend and show the install plan BEFORE downloading
# anything, so driver/backend incompatibilities surface early.
preflight_gpu_check() {
    print_step "Running GPU preflight check..."

    local driver_cuda="" driver_version=""
    if driver_cuda="$(tuft_detect_driver_cuda_version)"; then
        driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
            | head -n 1 || true)"
        print_step "NVIDIA driver detected: ${driver_version:-unknown} (supports CUDA <= $driver_cuda)"
    else
        print_warning "No working NVIDIA driver detected (nvidia-smi missing or failing)."
    fi

    if ! RESOLVED_TORCH_BACKEND="$(tuft_resolve_torch_backend "$TORCH_BACKEND")"; then
        print_error "Could not resolve a torch backend for this machine (requested: $TORCH_BACKEND)."
        exit 1
    fi
    if [ "$RESOLVED_TORCH_BACKEND" != "default" ]; then
        TORCH_BACKEND_UV_ARGS="--torch-backend $RESOLVED_TORCH_BACKEND"
    fi

    print_step "Install plan:"
    echo "      torch backend:    $TORCH_BACKEND -> $RESOLVED_TORCH_BACKEND"
    echo "      virtualenv:       $TUFT_VENV"
    [ -n "${UV_CACHE_DIR:-}" ] && echo "      uv cache dir:     $UV_CACHE_DIR"
    [ -n "${UV_LINK_MODE:-}" ] && echo "      uv link mode:     $UV_LINK_MODE"
    [ -n "${UV_SYSTEM_CERTS:-}" ] && echo "      uv system certs:  $UV_SYSTEM_CERTS"
    [ -n "${UV_DEFAULT_INDEX:-}" ] && echo "      uv default index: $UV_DEFAULT_INDEX"
    [ -n "${UV_INDEX:-}" ] && echo "      uv extra indexes: $UV_INDEX"
    return 0
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install uv if not present
install_uv() {
  if command_exists uv; then
    print_step "uv is already installed"
    return
  fi

  print_step "Installing uv (Python package manager)..."

  # Primary: official installer (fast path)
  if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
    print_warning "uv install via curl failed. Falling back to pip install uv."
    print_warning "If you are in a restricted network, consider configuring a PyPI mirror."

    local PYTHON_BIN=""
    if command_exists python3; then
      PYTHON_BIN="python3"
    elif command_exists python; then
      PYTHON_BIN="python"
    else
      print_error "Python not found; cannot install uv. Please install python3 and re-run."
      exit 1
    fi

    "$PYTHON_BIN" -m pip install --user --upgrade uv
  fi

  # Source env files if present (for curl installer case)
  if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
  elif [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
  fi

  # Ensure PATH contains common locations
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

  if ! command_exists uv; then
    print_error "Failed to install uv. Please install it manually and re-run."
    exit 1
  fi

  print_success "uv installed successfully"
}

# A torch backend was requested, so uv must understand --torch-backend.
ensure_uv_supports_torch_backend() {
    if tuft_uv_supports_torch_backend; then
        return 0
    fi
    print_error "The installed uv ($(uv --version 2>/dev/null || echo unknown)) does not support --torch-backend."
    print_error "Upgrade uv (curl -LsSf https://astral.sh/uv/install.sh | sh) and re-run."
    exit 1
}

# Create tuft directory structure
create_directories() {
    print_step "Creating TuFT directory structure..."
    mkdir -p "$TUFT_HOME"
    mkdir -p "$TUFT_BIN"
    mkdir -p "$TUFT_HOME/checkpoints"
    mkdir -p "$TUFT_HOME/configs"
    mkdir -p "$TUFT_HOME/scripts"
}

# Create Python virtual environment and install tuft
install_tuft() {
    print_step "Creating Python $PYTHON_VERSION virtual environment at $TUFT_VENV..."

    # Remove existing venv if present
    if [ -d "$TUFT_VENV" ]; then
        rm -rf "$TUFT_VENV"
    fi

    mkdir -p "$(dirname "$TUFT_VENV")"
    uv venv --python "$PYTHON_VERSION" "$TUFT_VENV"

    print_step "Installing TuFT package..."

    # Determine package source
    if [ -n "$LOCAL_SOURCE_PATH" ]; then
        print_step "Installing from local source: $LOCAL_SOURCE_PATH"
        PACKAGE_SPEC="${LOCAL_SOURCE_PATH}[backend,persistence]"
    elif [ "$INSTALL_FROM_SOURCE" = true ]; then
        print_step "Installing from GitHub: $TUFT_GIT_REPO"
        PACKAGE_SPEC="git+${TUFT_GIT_REPO}#egg=tuft[backend,persistence]"
    else
        print_step "Installing from PyPI: $TUFT_PYPI_REQUIREMENT"
        PACKAGE_SPEC="$TUFT_PYPI_REQUIREMENT"
    fi

    if [ -n "$TORCH_BACKEND_UV_ARGS" ]; then
        print_step "Using torch backend: $RESOLVED_TORCH_BACKEND"
    fi

    # PACKAGE_SPEC already includes the backend and persistence extras.
    # TORCH_BACKEND_UV_ARGS is deliberately unquoted: it is either empty or
    # "--torch-backend <value>" with a validated, whitespace-free value.
    # shellcheck disable=SC2086
    uv pip install --python "$TUFT_VENV/bin/python" $TORCH_BACKEND_UV_ARGS "$PACKAGE_SPEC"

    print_success "TuFT installed successfully"
}

# Record the resolved backend so `tuft upgrade` resolves packages the same way.
record_torch_backend() {
    printf '%s\n' "$RESOLVED_TORCH_BACKEND" > "$TUFT_HOME/torch-backend"
    print_step "Recorded torch backend '$RESOLVED_TORCH_BACKEND' in $TUFT_HOME/torch-backend"
}

# URL for the flash-attn installation script
FLASH_ATTN_SCRIPT_URL="https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install_flash_attn.py"

# Install flash-attn from precompiled wheels (avoids lengthy compilation)
# Also stores the script locally for later use by install-backend command
install_flash_attn() {
    print_step "Installing flash-attn from precompiled wheels..."

    local script_path="$TUFT_HOME/scripts/install_flash_attn.py"

    # Copy or download the flash-attn install script to local storage
    if [ -n "$LOCAL_SOURCE_PATH" ] && [ -f "$LOCAL_SOURCE_PATH/scripts/install_flash_attn.py" ]; then
        print_step "Using local flash-attn install script"
        cp "$LOCAL_SOURCE_PATH/scripts/install_flash_attn.py" "$script_path"
    else
        # Download the script from GitHub and store locally
        if ! curl -fsSL "$FLASH_ATTN_SCRIPT_URL" -o "$script_path"; then
            print_warning "Could not download flash-attn install script, skipping"
            return
        fi
    fi

    # Run the script and check exit code
    if "$TUFT_VENV/bin/python" "$script_path"; then
        print_success "flash-attn installation complete"
    else
        print_warning "flash-attn installation failed. This is optional, so installation will continue."
    fi
}

# Validate the freshly created environment: dependency metadata coherence,
# imports, and (when a driver is present) a CUDA smoke test.
post_install_checks() {
    print_step "Running post-install checks..."
    local python_bin="$TUFT_VENV/bin/python"

    print_step "Checking installed package compatibility (uv pip check)..."
    if ! uv pip check --python "$python_bin"; then
        print_error "Installed packages have incompatible dependencies (see above)."
        exit 1
    fi

    print_step "Verifying core imports..."
    if ! "$python_bin" -c 'import tuft, torch; print("imports OK: tuft with torch %s (CUDA build: %s)" % (torch.__version__, torch.version.cuda))'; then
        print_error "The installed environment failed to import tuft/torch."
        exit 1
    fi

    if [ "${TUFT_SKIP_GPU_CHECKS:-0}" = "1" ]; then
        print_warning "Skipping CUDA smoke test (GPU checks disabled)."
        return 0
    fi
    if [ "$RESOLVED_TORCH_BACKEND" = "cpu" ]; then
        print_step "CPU backend selected; skipping CUDA smoke test."
        return 0
    fi
    if [ -z "$(tuft_detect_driver_cuda_version || true)" ]; then
        print_warning "No NVIDIA driver detected; skipping CUDA smoke test. GPU features will not work on this machine."
        return 0
    fi

    print_step "Running CUDA smoke test..."
    if tuft_cuda_smoke_test "$python_bin"; then
        print_success "CUDA smoke test passed"
    else
        print_error "CUDA smoke test failed: the installed torch build cannot use this machine's GPU."
        print_error "Pick a backend matching your driver (--torch-backend auto|cpu|cu129|cu130),"
        print_error "or re-run with --skip-gpu-checks to keep the installation anyway."
        exit 1
    fi
}

# Install the shared backend helpers so the `tuft` wrapper (upgrade command)
# resolves torch backends with exactly the installer's logic. The function
# bodies are serialized from this script -- the single source of truth.
write_torch_backend_lib() {
    local lib_path="$TUFT_HOME/scripts/torch_backend.sh"
    print_step "Installing shared torch backend helpers to $lib_path"
    {
        echo "#!/bin/bash"
        echo "# TuFT torch backend helpers (GENERATED by install.sh -- do not edit)."
        echo "# The source of truth is scripts/install.sh; the tuft wrapper sources"
        echo "# this file so 'tuft upgrade' shares the installer's backend logic."
        echo ""
        declare -f \
            tuft_supported_cuda_backends \
            tuft_backend_cuda_version \
            tuft_version_ge \
            tuft_detect_driver_cuda_version \
            tuft_uv_supports_torch_backend \
            tuft_resolve_torch_backend \
            tuft_cuda_smoke_test
    } > "$lib_path"
}

# Create the tuft wrapper script
# Note: The wrapper is intentionally embedded in this install script (heredocs)
# rather than being a separate file. This ensures the wrapper is always in sync
# with the install script version and simplifies distribution. When updating the
# wrapper, edit the heredocs below. The wrapper provides CLI commands (launch,
# version, upgrade, etc.) that delegate to the Python module while handling
# configuration defaults and environment setup. Backend-resolution logic is NOT
# duplicated here: the wrapper sources $TUFT_HOME/scripts/torch_backend.sh,
# which write_torch_backend_lib generates from this script's functions.
create_wrapper() {
    print_step "Creating tuft command wrapper..."

    # Head segment: expanded at install time to bake in install-time choices.
    cat > "$TUFT_BIN/tuft" << WRAPPER_HEAD_EOF
#!/bin/bash
# TuFT CLI Wrapper
# This script provides a convenient interface to the TuFT server
# Generated by install.sh - edit the heredocs in install.sh to modify

set -e

# Virtualenv location recorded at install time (override with TUFT_VENV).
TUFT_INSTALL_VENV="${TUFT_VENV}"
WRAPPER_HEAD_EOF

    # Body: quoted heredoc, no expansion at install time.
    cat >> "$TUFT_BIN/tuft" << 'WRAPPER_EOF'

TUFT_HOME="${TUFT_HOME:-$HOME/.tuft}"
# TUFT_VENV precedence: explicit env var > location recorded at install time.
TUFT_VENV="${TUFT_VENV:-${TUFT_INSTALL_VENV:-$TUFT_HOME/venv}}"
TUFT_PYTHON="$TUFT_VENV/bin/python"
TUFT_PYPI_REQUIREMENT="${TUFT_PYPI_REQUIREMENT:-tuft[backend,persistence]>=0.1.8}"

# Verify installation
if [ ! -f "$TUFT_PYTHON" ]; then
    echo "Error: TuFT installation not found at $TUFT_VENV"
    echo "Please reinstall TuFT using:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"'
    exit 1
fi

# Handle commands
case "${1:-}" in
    launch)
        shift
        # Pass all arguments directly to the CLI (single source of truth)
        exec "$TUFT_PYTHON" -m tuft launch "$@"
        ;;

    version|--version|-v)
        "$TUFT_PYTHON" -c "import tuft; print(f'TuFT version: {tuft.__version__}')" 2>/dev/null || \
        "$TUFT_PYTHON" -c "from importlib.metadata import version; print(f'TuFT version: {version(\"tuft\")}')"
        ;;

    upgrade)
        shift
        # Parse upgrade options
        UPGRADE_FROM_SOURCE=false
        UPGRADE_LOCAL_SOURCE=""
        UPGRADE_TORCH_BACKEND=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --from-source)
                    UPGRADE_FROM_SOURCE=true
                    shift
                    ;;
                --local-source)
                    UPGRADE_LOCAL_SOURCE="$2"
                    shift 2
                    ;;
                --torch-backend)
                    if [[ $# -lt 2 ]]; then
                        echo "Error: --torch-backend requires a value (auto, cpu, or cuNNN such as cu130)"
                        exit 1
                    fi
                    UPGRADE_TORCH_BACKEND="$2"
                    shift 2
                    ;;
                *)
                    echo "Unknown option: $1"
                    echo "Usage: tuft upgrade [--from-source | --local-source PATH] [--torch-backend auto|cpu|cuNNN]"
                    exit 1
                    ;;
            esac
        done

        # Torch backend precedence: --torch-backend flag > TUFT_TORCH_BACKEND
        # env var > value recorded at install time. Resolution logic is shared
        # with the installer via the generated helper library.
        BACKEND_LIB="$TUFT_HOME/scripts/torch_backend.sh"
        BACKEND_STATE="$TUFT_HOME/torch-backend"
        if [ -f "$BACKEND_LIB" ]; then
            # shellcheck disable=SC1090
            source "$BACKEND_LIB"
        fi
        REQUESTED_BACKEND="${UPGRADE_TORCH_BACKEND:-${TUFT_TORCH_BACKEND:-}}"
        RESOLVED_BACKEND=""
        if [ -n "$REQUESTED_BACKEND" ]; then
            if type tuft_resolve_torch_backend >/dev/null 2>&1; then
                RESOLVED_BACKEND="$(tuft_resolve_torch_backend "$REQUESTED_BACKEND")" || exit 1
            elif [ "$REQUESTED_BACKEND" = "auto" ]; then
                echo "Error: backend helpers not found at $BACKEND_LIB (older installation),"
                echo "       so 'auto' cannot detect the driver. Pass an explicit backend"
                echo "       (e.g. --torch-backend cu130) or re-run the installer."
                exit 1
            else
                case "$REQUESTED_BACKEND" in
                    cpu|cu[0-9][0-9][0-9])
                        RESOLVED_BACKEND="$REQUESTED_BACKEND"
                        ;;
                    *)
                        echo "Error: invalid torch backend '$REQUESTED_BACKEND' (expected: auto, cpu, or cuNNN such as cu130)."
                        exit 1
                        ;;
                esac
            fi
        elif [ -f "$BACKEND_STATE" ]; then
            RESOLVED_BACKEND="$(head -n 1 "$BACKEND_STATE" 2>/dev/null || true)"
            case "$RESOLVED_BACKEND" in
                default|cpu|cu[0-9][0-9][0-9]) ;;
                *)
                    if [ -n "$RESOLVED_BACKEND" ]; then
                        echo "Warning: ignoring unrecognized torch backend '$RESOLVED_BACKEND' recorded in $BACKEND_STATE"
                    fi
                    RESOLVED_BACKEND=""
                    ;;
            esac
        fi

        # TORCH_BACKEND_UV_ARGS is deliberately unquoted below: it is either
        # empty or "--torch-backend <value>" with a whitespace-free value.
        TORCH_BACKEND_UV_ARGS=""
        if [ -n "$RESOLVED_BACKEND" ] && [ "$RESOLVED_BACKEND" != "default" ]; then
            if type tuft_uv_supports_torch_backend >/dev/null 2>&1 && ! tuft_uv_supports_torch_backend; then
                echo "Error: the installed uv does not support --torch-backend; upgrade uv first:"
                echo "       curl -LsSf https://astral.sh/uv/install.sh | sh"
                exit 1
            fi
            TORCH_BACKEND_UV_ARGS="--torch-backend $RESOLVED_BACKEND"
            echo "Using torch backend: $RESOLVED_BACKEND"
        fi

        echo "Upgrading TuFT..."
        # shellcheck disable=SC2086
        if [ -n "$UPGRADE_LOCAL_SOURCE" ]; then
            echo "Upgrading from local source: $UPGRADE_LOCAL_SOURCE"
            uv pip install --python "$TUFT_PYTHON" --upgrade $TORCH_BACKEND_UV_ARGS "${UPGRADE_LOCAL_SOURCE}[backend,persistence]"
        elif [ "$UPGRADE_FROM_SOURCE" = true ]; then
            # Repo is overridable (default: upstream main) so CI / advanced users
            # can exercise the real VCS clone+build+resolve path against a
            # specific checkout, e.g. TUFT_GIT_URL="file://$GITHUB_WORKSPACE@$GITHUB_SHA".
            TUFT_GIT_URL="${TUFT_GIT_URL:-https://github.com/agentscope-ai/tuft.git}"
            echo "Upgrading from Git: git+${TUFT_GIT_URL}"
            uv pip install --python "$TUFT_PYTHON" --upgrade $TORCH_BACKEND_UV_ARGS "git+${TUFT_GIT_URL}#egg=tuft[backend,persistence]"
        else
            uv pip install --python "$TUFT_PYTHON" --upgrade $TORCH_BACKEND_UV_ARGS "$TUFT_PYPI_REQUIREMENT"
        fi

        # Remember the backend so the next upgrade resolves the same way.
        if [ -n "$RESOLVED_BACKEND" ]; then
            printf '%s\n' "$RESOLVED_BACKEND" > "$BACKEND_STATE"
        fi

        # Also update flash-attn
        echo ""
        echo "Updating flash-attn..."
        FLASH_SCRIPT_PATH="$TUFT_HOME/scripts/install_flash_attn.py"
        if [ -n "$UPGRADE_LOCAL_SOURCE" ] && [ -f "$UPGRADE_LOCAL_SOURCE/scripts/install_flash_attn.py" ]; then
            cp "$UPGRADE_LOCAL_SOURCE/scripts/install_flash_attn.py" "$FLASH_SCRIPT_PATH"
        elif [ ! -f "$FLASH_SCRIPT_PATH" ]; then
            FLASH_SCRIPT_URL="https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install_flash_attn.py"
            mkdir -p "$TUFT_HOME/scripts"
            curl -fsSL "$FLASH_SCRIPT_URL" -o "$FLASH_SCRIPT_PATH" 2>/dev/null || true
        fi
        if [ -f "$FLASH_SCRIPT_PATH" ]; then
            "$TUFT_PYTHON" "$FLASH_SCRIPT_PATH" || echo "Warning: flash-attn update failed (optional)"
        fi

        # Post-upgrade checks: dependency metadata coherence plus a CUDA smoke
        # test when a driver is present and a GPU-capable backend is installed.
        echo ""
        echo "Running post-upgrade checks..."
        if ! uv pip check --python "$TUFT_PYTHON"; then
            echo "Error: installed packages have incompatible dependencies after upgrade (see above)."
            exit 1
        fi
        if [ "${TUFT_SKIP_GPU_CHECKS:-0}" != "1" ] && [ "$RESOLVED_BACKEND" != "cpu" ] \
            && type tuft_cuda_smoke_test >/dev/null 2>&1 \
            && type tuft_detect_driver_cuda_version >/dev/null 2>&1 \
            && [ -n "$(tuft_detect_driver_cuda_version || true)" ]; then
            echo "Running CUDA smoke test..."
            if ! tuft_cuda_smoke_test "$TUFT_PYTHON"; then
                echo "Error: CUDA smoke test failed after upgrade. Re-run with an explicit"
                echo "       backend (tuft upgrade --torch-backend auto|cpu|cu129|cu130) or set"
                echo "       TUFT_SKIP_GPU_CHECKS=1 to skip this check."
                exit 1
            fi
        fi

        echo ""
        echo "TuFT upgraded successfully!"
        ;;

    uninstall)
        echo "Uninstalling TuFT..."
        read -p "This will remove $TUFT_HOME. Are you sure? [y/N] " -n 1 -r
        echo
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            rm -rf "$TUFT_HOME"
            case "$TUFT_VENV" in
                "$TUFT_HOME"/*) ;;
                *)
                    if [ -d "$TUFT_VENV" ]; then
                        echo "Note: the custom virtualenv at $TUFT_VENV was NOT removed; delete it manually if desired."
                    fi
                    ;;
            esac
            echo "TuFT uninstalled. Please remove $TUFT_HOME/bin from your PATH."
        else
            echo "Uninstall cancelled."
        fi
        ;;

    help|--help|-h)
        echo "TuFT - Tenant-unified Fine-Tuning Server"
        echo ""
        echo "Usage: tuft <command> [options]"
        echo ""
        echo "Commands:"
        echo "  launch            Start the TuFT server"
        echo "  version           Show TuFT version"
        echo "  upgrade           Upgrade TuFT to the latest version"
        echo "                    Options: --from-source, --local-source PATH,"
        echo "                             --torch-backend auto|cpu|cuNNN"
        echo "  uninstall         Remove TuFT installation"
        echo "  help              Show this help message"
        echo ""
        echo "Launch options: Run 'tuft launch --help' for all available options."
        echo ""
        echo "Environment Variables:"
        echo "  TUFT_HOME            Installation directory (default: ~/.tuft)"
        echo "  TUFT_VENV            Virtual environment location (default: recorded at install)"
        echo "  TUFT_CONFIG          Default config file path"
        echo "  TUFT_HOST            Default host for launch command"
        echo "  TUFT_PORT            Default port for launch command"
        echo "  TUFT_CHECKPOINT_DIR  Default checkpoint directory"
        echo "  TUFT_LOG_LEVEL       Default log level"
        echo "  TUFT_PYPI_REQUIREMENT Override the PyPI package requirement"
        echo "  TUFT_TORCH_BACKEND   Torch backend for upgrade (auto|cpu|cuNNN)"
        echo "  TUFT_SKIP_GPU_CHECKS Set to 1 to skip GPU checks during upgrade"
        echo ""
        echo "uv settings such as UV_CACHE_DIR, UV_LINK_MODE, UV_SYSTEM_CERTS,"
        echo "UV_DEFAULT_INDEX and UV_INDEX are passed through to uv."
        echo ""
        echo "Examples:"
        echo "  tuft launch --config tuft_config.yaml"
        echo "  tuft launch --port 10610 --config /path/to/tuft_config.yaml"
        echo "  tuft launch  # uses default config at ~/.tuft/configs/tuft_config.yaml"
        echo "  tuft upgrade"
        echo "  tuft upgrade --torch-backend cu130"
        echo ""
        echo "Documentation: https://github.com/agentscope-ai/tuft"
        ;;

    "")
        # No command provided, show help
        "$0" help
        ;;

    *)
        # Pass through to the tuft module for any other commands
        exec "$TUFT_PYTHON" -m tuft "$@"
        ;;
esac
WRAPPER_EOF

    chmod +x "$TUFT_BIN/tuft"
    print_success "Wrapper script created at $TUFT_BIN/tuft"
}

# Create example configuration
create_example_config() {
    if [ ! -f "$TUFT_HOME/configs/tuft_config.yaml.example" ]; then
        print_step "Creating example configuration..."
        cat > "$TUFT_HOME/configs/tuft_config.yaml.example" << 'CONFIG_EOF'
# TuFT Server Configuration Example
# Copy this file to tuft_config.yaml and customize for your setup

model_owner: local

supported_models:
  - model_name: Qwen/Qwen3-8B
    model_path: Qwen/Qwen3-8B  # HuggingFace model ID or local path
    max_model_len: 32768
    tensor_parallel_size: 1
    temperature: 0.7
    top_p: 1.0
    top_k: -1

  # Add more models as needed:
  # - model_name: meta-llama/Llama-2-7b-hf
  #   model_path: /path/to/local/model
  #   max_model_len: 4096
  #   tensor_parallel_size: 1

# API Key authentication
# Format: api_key: user_identifier
authorized_users:
  my-api-key: default
  # Add more API keys as needed:
  # another-key: another-user

# Optional: Persistence configuration
# persistence:
#   mode: DISABLE  # Options: DISABLE, REDIS, FILE
#   redis_url: "redis://localhost:6379/0"
#   namespace: "persistence-tuft-server"
CONFIG_EOF
    fi
}

# Update shell configuration to add tuft to PATH
update_shell_config() {
    print_step "Configuring shell PATH..."

    SHELL_NAME="$(basename "$SHELL")"
    SHELL_CONFIG=""

    case "$SHELL_NAME" in
        bash)
            if [ -f "$HOME/.bash_profile" ]; then
                SHELL_CONFIG="$HOME/.bash_profile"
            else
                SHELL_CONFIG="$HOME/.bashrc"
            fi
            ;;
        zsh)
            SHELL_CONFIG="$HOME/.zshrc"
            ;;
        fish)
            SHELL_CONFIG="$HOME/.config/fish/config.fish"
            ;;
        *)
            print_warning "Unknown shell: $SHELL_NAME. Please add $TUFT_BIN to your PATH manually."
            return
            ;;
    esac

    # Check if PATH is already configured
    if [ -n "$SHELL_CONFIG" ] && [ -f "$SHELL_CONFIG" ]; then
        if grep -q "TUFT_HOME" "$SHELL_CONFIG" 2>/dev/null; then
            print_step "PATH already configured in $SHELL_CONFIG"
            return
        fi
    fi

    # Add to shell config
    # Use $HOME literal so the config remains portable
    if [ -n "$SHELL_CONFIG" ]; then
        if [ "$SHELL_NAME" = "fish" ]; then
            mkdir -p "$(dirname "$SHELL_CONFIG")"
            echo "" >> "$SHELL_CONFIG"
            echo "# TuFT" >> "$SHELL_CONFIG"
            echo 'set -gx TUFT_HOME $HOME/.tuft' >> "$SHELL_CONFIG"
            echo 'fish_add_path $TUFT_HOME/bin' >> "$SHELL_CONFIG"
        else
            echo "" >> "$SHELL_CONFIG"
            echo "# TuFT" >> "$SHELL_CONFIG"
            echo 'export TUFT_HOME="$HOME/.tuft"' >> "$SHELL_CONFIG"
            echo 'export PATH="$TUFT_HOME/bin:$PATH"' >> "$SHELL_CONFIG"
        fi
        print_success "Added TuFT to PATH in $SHELL_CONFIG"
    fi
}

# Print completion message
print_completion() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  TuFT installation complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "Installation directory: $TUFT_HOME"
    echo "Virtual environment:    $TUFT_VENV"
    echo "Torch backend:          $RESOLVED_TORCH_BACKEND"
    echo ""
    echo "To get started:"
    echo ""
    echo "  1. Restart your terminal or run:"
    echo "     source ~/.$(basename "$SHELL")rc"
    echo ""
    echo "  2. Create a server configuration file:"
    echo "     cp $TUFT_HOME/configs/tuft_config.yaml.example $TUFT_HOME/configs/tuft_config.yaml"
    echo "     # Edit the file to configure your models and API keys"
    echo ""
    echo "  3. Launch the TuFT server:"
    echo "     tuft launch"
    echo ""
    echo "For more information:"
    echo "  tuft help"
    echo "  https://github.com/agentscope-ai/tuft"
    echo ""
}

# Main installation flow
main() {
    parse_args "$@"

    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  TuFT Installer${NC}"
    echo -e "${BLUE}  Tenant-unified Fine-Tuning Server${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""

    print_step "Installing with full backend support (GPU, persistence, flash-attn)"

    if [ -n "$LOCAL_SOURCE_PATH" ]; then
        print_step "Installing from local source: $LOCAL_SOURCE_PATH"
    elif [ "$INSTALL_FROM_SOURCE" = true ]; then
        print_step "Installing from GitHub (source)"
    else
        print_step "Installing from PyPI"
    fi

    # Clean existing installation if requested
    if [ "$CLEAN_INSTALL" = true ] && [ -d "$TUFT_HOME" ]; then
        print_step "Cleaning existing installation at $TUFT_HOME..."
        rm -rf "$TUFT_HOME"
        print_success "Existing installation removed"
    fi

    detect_platform
    preflight_gpu_check
    install_uv
    if [ -n "$TORCH_BACKEND_UV_ARGS" ]; then
        ensure_uv_supports_torch_backend
    fi
    create_directories
    install_tuft
    record_torch_backend
    install_flash_attn
    post_install_checks
    create_wrapper
    write_torch_backend_lib
    create_example_config
    update_shell_config
    print_completion
}

# Run main unless this file is being sourced (the shell tests source it to
# exercise the backend-resolution functions directly). Note: when piped into
# bash (curl | bash or bash -c), BASH_SOURCE is empty and main must run.
if [ -z "${BASH_SOURCE[0]:-}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
