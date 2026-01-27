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
TUFT_VENV="$TUFT_HOME/venv"
PYTHON_VERSION="3.12"
TUFT_PYPI_PACKAGE="tuft"
TUFT_GIT_REPO="https://github.com/agentscope-ai/tuft.git"
INSTALL_BACKEND=true
INSTALL_FROM_SOURCE=false
LOCAL_SOURCE_PATH=""

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
            --with-backend)
                # Default behavior, kept for backwards compatibility
                INSTALL_BACKEND=true
                shift
                ;;
            --without-backend)
                INSTALL_BACKEND=false
                shift
                ;;
            --from-source)
                INSTALL_FROM_SOURCE=true
                shift
                ;;
            --local-source)
                LOCAL_SOURCE_PATH="$2"
                shift 2
                ;;
            --help|-h)
                echo "TuFT Installation Script"
                echo ""
                echo "Usage: install.sh [options]"
                echo ""
                echo "Options:"
                echo "  --without-backend     Skip GPU backend and install minimal dependencies only"
                echo "  --from-source         Install from GitHub instead of PyPI"
                echo "  --local-source PATH   Install from local source directory (for development/CI)"
                echo "  --help, -h            Show this help message"
                echo ""
                echo "By default, the script installs TuFT with all extras (GPU backend, persistence,"
                echo "flash-attn) for full functionality. Use --without-backend for minimal install."
                echo ""
                echo "Environment Variables:"
                echo "  TUFT_HOME             Installation directory (default: ~/.tuft)"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

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
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the env file to get uv in PATH
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    # Add to current PATH if still not found
    if ! command_exists uv; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi

    if ! command_exists uv; then
        print_error "Failed to install uv. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    print_success "uv installed successfully"
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
    print_step "Creating Python $PYTHON_VERSION virtual environment..."

    # Remove existing venv if present
    if [ -d "$TUFT_VENV" ]; then
        rm -rf "$TUFT_VENV"
    fi

    uv venv --python "$PYTHON_VERSION" "$TUFT_VENV"

    print_step "Installing TuFT package..."

    # Determine package source
    if [ -n "$LOCAL_SOURCE_PATH" ]; then
        print_step "Installing from local source: $LOCAL_SOURCE_PATH"
        PACKAGE_SPEC="$LOCAL_SOURCE_PATH"
    elif [ "$INSTALL_FROM_SOURCE" = true ]; then
        print_step "Installing from GitHub: $TUFT_GIT_REPO"
        PACKAGE_SPEC="git+${TUFT_GIT_REPO}"
    else
        print_step "Installing from PyPI: $TUFT_PYPI_PACKAGE"
        PACKAGE_SPEC="$TUFT_PYPI_PACKAGE"
    fi

    # Install with or without extras
    if [ "$INSTALL_BACKEND" = true ]; then
        # Install tuft with all extras (backend, persistence)
        uv pip install --python "$TUFT_VENV/bin/python" "${PACKAGE_SPEC}[backend,persistence]"
    else
        # Install tuft with minimal dependencies
        uv pip install --python "$TUFT_VENV/bin/python" "$PACKAGE_SPEC"
    fi

    print_success "TuFT installed successfully"
}

# URL for the flash-attn installation script
FLASH_ATTN_SCRIPT_URL="https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install_flash_attn.py"

# Install flash-attn from precompiled wheels (avoids lengthy compilation)
# Also stores the script locally for later use by install-backend command
install_flash_attn() {
    if [ "$INSTALL_BACKEND" != true ]; then
        return
    fi

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

# Create the tuft wrapper script
create_wrapper() {
    print_step "Creating tuft command wrapper..."

    cat > "$TUFT_BIN/tuft" << 'WRAPPER_EOF'
#!/bin/bash
# TuFT CLI Wrapper
# This script provides a convenient interface to the TuFT server

set -e

TUFT_HOME="${TUFT_HOME:-$HOME/.tuft}"
TUFT_VENV="$TUFT_HOME/venv"
TUFT_PYTHON="$TUFT_VENV/bin/python"

# Verify installation
if [ ! -f "$TUFT_PYTHON" ]; then
    echo "Error: TuFT installation not found at $TUFT_HOME"
    echo "Please reinstall TuFT using:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"'
    exit 1
fi

# Handle commands
case "${1:-}" in
    launch)
        shift
        # Default configuration
        HOST="${TUFT_HOST:-127.0.0.1}"
        PORT="${TUFT_PORT:-8000}"
        CHECKPOINT_DIR="${TUFT_CHECKPOINT_DIR:-$TUFT_HOME/checkpoints}"
        MODEL_CONFIG=""
        LOG_LEVEL="info"
        EXTRA_ARGS=()

        # Parse arguments
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --host)
                    HOST="$2"
                    shift 2
                    ;;
                --port|-p)
                    PORT="$2"
                    shift 2
                    ;;
                --model-config)
                    MODEL_CONFIG="$2"
                    shift 2
                    ;;
                --checkpoint-dir)
                    CHECKPOINT_DIR="$2"
                    shift 2
                    ;;
                --log-level)
                    LOG_LEVEL="$2"
                    shift 2
                    ;;
                *)
                    EXTRA_ARGS+=("$1")
                    shift
                    ;;
            esac
        done

        # Check for model config
        if [ -z "$MODEL_CONFIG" ]; then
            # Check for default config
            if [ -f "$TUFT_HOME/configs/models.yaml" ]; then
                MODEL_CONFIG="$TUFT_HOME/configs/models.yaml"
            else
                echo "Error: No model configuration provided."
                echo ""
                echo "Please provide a model configuration file:"
                echo "  tuft launch --model-config /path/to/models.yaml"
                echo ""
                echo "Or create a default configuration at:"
                echo "  $TUFT_HOME/configs/models.yaml"
                echo ""
                echo "Example models.yaml:"
                echo "  model_owner: local"
                echo "  supported_models:"
                echo "    - model_name: Qwen/Qwen3-8B"
                echo "      model_path: Qwen/Qwen3-8B"
                echo "      max_model_len: 32768"
                echo "      tensor_parallel_size: 1"
                echo "  authorized_users:"
                echo "    my-api-key: default"
                exit 1
            fi
        fi

        echo "Starting TuFT server..."
        echo "  Host: $HOST"
        echo "  Port: $PORT"
        echo "  Model Config: $MODEL_CONFIG"
        echo "  Checkpoint Dir: $CHECKPOINT_DIR"
        echo ""

        exec "$TUFT_PYTHON" -m tuft \
            --host "$HOST" \
            --port "$PORT" \
            --model-config "$MODEL_CONFIG" \
            --checkpoint-dir "$CHECKPOINT_DIR" \
            --log-level "$LOG_LEVEL" \
            "${EXTRA_ARGS[@]}"
        ;;

    version|--version|-v)
        "$TUFT_PYTHON" -c "import tuft; print(f'TuFT version: {tuft.__version__}')" 2>/dev/null || \
        "$TUFT_PYTHON" -c "from importlib.metadata import version; print(f'TuFT version: {version(\"tuft\")}')"
        ;;

    upgrade)
        echo "Upgrading TuFT..."
        uv pip install --python "$TUFT_PYTHON" --upgrade tuft
        echo "TuFT upgraded successfully!"
        ;;

    install-backend)
        echo "Installing GPU backend dependencies..."
        echo "This includes trinity-rft, peft, vLLM, flash-attn, and Redis persistence"
        echo ""

        # Install all extras (backend + persistence)
        uv pip install --python "$TUFT_PYTHON" "tuft[backend,persistence]"

        # Install flash-attn from precompiled wheels using local script
        echo ""
        echo "Installing flash-attn from precompiled wheels..."
        FLASH_SCRIPT_PATH="$TUFT_HOME/scripts/install_flash_attn.py"

        # Download script if not present locally
        if [ ! -f "$FLASH_SCRIPT_PATH" ]; then
            FLASH_SCRIPT_URL="https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install_flash_attn.py"
            mkdir -p "$TUFT_HOME/scripts"
            if ! curl -fsSL "$FLASH_SCRIPT_URL" -o "$FLASH_SCRIPT_PATH"; then
                echo "Warning: Could not download flash-attn install script, skipping"
                echo ""
                echo "Backend installation complete (flash-attn skipped)!"
                exit 0
            fi
        fi

        if "$TUFT_PYTHON" "$FLASH_SCRIPT_PATH"; then
            echo "flash-attn installation complete"
        else
            echo "Warning: flash-attn installation failed. This is optional, so installation will continue."
        fi
        echo ""
        echo "Backend installation complete!"
        ;;

    uninstall)
        echo "Uninstalling TuFT..."
        read -p "This will remove $TUFT_HOME. Are you sure? [y/N] " -n 1 -r
        echo
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            rm -rf "$TUFT_HOME"
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
        echo "  install-backend   Install all extras (GPU backend, persistence, flash-attn)"
        echo "  uninstall         Remove TuFT installation"
        echo "  help              Show this help message"
        echo ""
        echo "Launch Options:"
        echo "  --host          Host to bind to (default: 127.0.0.1)"
        echo "  --port, -p      Port to bind to (default: 8000)"
        echo "  --model-config  Path to model configuration YAML (required)"
        echo "  --checkpoint-dir Directory for checkpoints (default: ~/.tuft/checkpoints)"
        echo "  --log-level     Log level: debug, info, warning, error (default: info)"
        echo ""
        echo "Environment Variables:"
        echo "  TUFT_HOME            Installation directory (default: ~/.tuft)"
        echo "  TUFT_HOST            Default host for launch command"
        echo "  TUFT_PORT            Default port for launch command"
        echo "  TUFT_CHECKPOINT_DIR  Default checkpoint directory"
        echo ""
        echo "Examples:"
        echo "  tuft launch --model-config models.yaml"
        echo "  tuft launch --port 8080 --model-config /path/to/models.yaml"
        echo "  tuft upgrade"
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
    if [ ! -f "$TUFT_HOME/configs/models.yaml.example" ]; then
        print_step "Creating example configuration..."
        cat > "$TUFT_HOME/configs/models.yaml.example" << 'CONFIG_EOF'
# TuFT Model Configuration Example
# Copy this file to models.yaml and customize for your setup

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
#   mode: disabled  # Options: disabled, redis_url, file_redis
#   redis_url: "redis://localhost:6379/0"
#   namespace: "tuft"
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
    if [ -n "$SHELL_CONFIG" ]; then
        if [ "$SHELL_NAME" = "fish" ]; then
            mkdir -p "$(dirname "$SHELL_CONFIG")"
            echo "" >> "$SHELL_CONFIG"
            echo "# TuFT" >> "$SHELL_CONFIG"
            echo "set -gx TUFT_HOME $TUFT_HOME" >> "$SHELL_CONFIG"
            echo "fish_add_path $TUFT_BIN" >> "$SHELL_CONFIG"
        else
            echo "" >> "$SHELL_CONFIG"
            echo "# TuFT" >> "$SHELL_CONFIG"
            echo "export TUFT_HOME=\"$TUFT_HOME\"" >> "$SHELL_CONFIG"
            echo "export PATH=\"\$TUFT_HOME/bin:\$PATH\"" >> "$SHELL_CONFIG"
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
    echo ""
    echo "To get started:"
    echo ""
    echo "  1. Restart your terminal or run:"
    echo "     source ~/.$(basename "$SHELL")rc"
    echo ""
    echo "  2. Create a model configuration file:"
    echo "     cp $TUFT_HOME/configs/models.yaml.example $TUFT_HOME/configs/models.yaml"
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

    if [ "$INSTALL_BACKEND" = true ]; then
        print_step "Installing with full backend support (GPU, persistence, flash-attn)"
    else
        print_step "Installing minimal dependencies (no GPU backend)"
    fi

    if [ -n "$LOCAL_SOURCE_PATH" ]; then
        print_step "Installing from local source: $LOCAL_SOURCE_PATH"
    elif [ "$INSTALL_FROM_SOURCE" = true ]; then
        print_step "Installing from GitHub (source)"
    else
        print_step "Installing from PyPI"
    fi

    detect_platform
    install_uv
    create_directories
    install_tuft
    install_flash_attn
    create_wrapper
    create_example_config
    update_shell_config
    print_completion
}

# Run main
main "$@"
