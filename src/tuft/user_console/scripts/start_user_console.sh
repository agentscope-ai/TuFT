#!/bin/bash

# tuft_user_console.sh - Launch the TuFT user console

set -e  # Exit immediately if a command exits with a non-zero status

# Default values
DEFAULT_GUI_PORT=10715
DEFAULT_BACKEND_PORT=10710

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
    $0 --server-url http://localhost:10610 --gui-port 10711
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

# Check if src directory exists
if [[ ! -d "src" ]]; then
    echo "Error: 'src' directory not found. Please run this script from the project root directory." >&2
    exit 1
fi

echo "Configuration:"
echo "  TuFT Server URL: $TUFT_SERVER_URL"
echo "  Backend Port: $BACKEND_PORT"
echo "  GUI Port: $GUI_PORT"
echo

# Start backend service (run in background)
echo "Starting user console backend service..."

export TUFT_SERVER_URL="$TUFT_SERVER_URL"
uvicorn src.console_server.main:app --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Wait a few seconds for the backend to start
sleep 3

# Start GUI
echo "Starting user console GUI..."
python -m src.console_gui --port "$GUI_PORT" --backend_port "$BACKEND_PORT" &
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
