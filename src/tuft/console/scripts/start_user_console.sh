#!/bin/bash

# tuft_user_console.sh - Launch the TuFT user console

set -e  # Exit immediately if a command exits with a non-zero status

# Default values
DEFAULT_GUI_PORT=10613
DEFAULT_BACKEND_PORT=10713

# Display help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Launch the TuFT user console (including backend server and frontend GUI).

Options:
    --server-url URL     URL of the TuFT server (required)
    --gui-port PORT      Port for the GUI (default: $DEFAULT_GUI_PORT)
    --backend-port PORT  Port for the backend service (default: $DEFAULT_BACKEND_PORT)
    -h, --help           Show this help message

Example:
    $0 --server-url http://localhost:10610 --gui-port 10613
EOF
}

# Parse command-line arguments
TUFT_SERVER_URL=""
GUI_PORT=$DEFAULT_GUI_PORT
BACKEND_PORT=$DEFAULT_BACKEND_PORT

while [[ $# -gt 0 ]]; do
    case $1 in
        --server-url)
            TUFT_SERVER_URL="$2"
            shift 2
            ;;
        --gui-port)
            GUI_PORT="$2"
            shift 2
            ;;
        --backend-port)
            BACKEND_PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

# Check required argument
if [[ -z "$TUFT_SERVER_URL" ]]; then
    echo "Error: --server-url is required" >&2
    show_help >&2
    exit 1
fi

# Validate that ports are valid numbers
if ! [[ "$GUI_PORT" =~ ^[0-9]+$ ]] || ! [[ "$BACKEND_PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: Port numbers must be valid integers" >&2
    exit 1
fi

echo "Configuration:"
echo "  TuFT Server URL: $TUFT_SERVER_URL"
echo "  Backend Port: $BACKEND_PORT"
echo "  GUI Port: $GUI_PORT"
echo

# Start backend service
echo "Starting user console backend service..."
export TUFT_SERVER_URL="$TUFT_SERVER_URL"
export TUFT_GUI_URL="http://localhost:$GUI_PORT"
uvicorn console_server.main:app --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Wait for backend to be ready with health check
echo "Waiting for backend service to start..."
timeout=30
count=0
while [ $count -lt $timeout ]; do
    if python3 -c "
import urllib.request, sys;
try:
    r = urllib.request.urlopen('http://127.0.0.1:${BACKEND_PORT}/api/v1/health', timeout=2);
    sys.exit(0 if r.getcode() == 200 else 1)
except Exception as e:
    sys.exit(1)
    "; then
        echo "Backend service is ready!"
        break
    fi
    sleep 1
    count=$((count + 1))
done

if [ $count -eq $timeout ]; then
    echo "Error: Backend service failed to start within $timeout seconds"
    kill $BACKEND_PID
    exit 1
fi

# Start UI
echo "Starting user console UI..."
python -m console_ui --port "$GUI_PORT" --backend_port "$BACKEND_PORT" &
GUI_PID=$!

echo
echo "User console is now running!"
echo "Open your browser at: http://localhost:$GUI_PORT"
echo
echo "Press Ctrl+C to stop all services..."

# Cleanup function
cleanup() {
    echo
    echo "Stopping services..."
    kill $BACKEND_PID $GUI_PID 2>/dev/null
    wait $BACKEND_PID $GUI_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Trap interrupt signals
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait $BACKEND_PID $GUI_PID
