#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Realistic test URLs
# -----------------------------

GITHUB_CURL_TEST_URL="https://github.com/"
GITHUB_TEST_URL="https://github.com/astral-sh/uv/releases/download/0.9.30/source.tar.gz"
PYTHON_STANDALONE_CURL_TEST_URL="https://python-standalone.org/"
PYTHON_STANDALONE_TEST_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260203/cpython-3.10.19+20260203-aarch64-apple-darwin-install_only.tar.gz"
PYPI_CURL_TEST_URL="https://pypi.org/simple/"
PYPI_TEST_URL="https://files.pythonhosted.org/packages/4e/a0/63cea38fe839fb89592728b91928ee6d15705f1376a7940fee5bbc77fea0/uv-0.9.30.tar.gz"
HF_CURL_TEST_URL="https://huggingface.co/"
HF_TEST_URL="https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json"

TTFB_ITERATIONS=3
# -----------------------------
# Pre-check: ensure curl exists
# -----------------------------
if ! command -v curl >/dev/null 2>&1; then
  cat <<'EOF'
curl was not found, so network connectivity/latency checks cannot be performed.
Please install curl and re-run this script.

Common install commands:
  Ubuntu/Debian:   sudo apt-get update && sudo apt-get install -y curl
  CentOS/RHEL:     sudo yum install -y curl        # or sudo dnf install -y curl
  Alpine:          sudo apk add curl
  macOS(Homebrew): brew install curl
EOF
  exit 1
fi

# -----------------------------
# Config
# -----------------------------
LATENCY_THRESHOLD_MS="${LATENCY_THRESHOLD_MS:-1000}"
SPEED_THRESHOLD_KBPS="${SPEED_THRESHOLD_KBPS:-1000}"

CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-3}"
MAX_TIME="${MAX_TIME:-10}"

UA="mirror-check/1.0"

# Enable debug? Set to 1 to see raw curl output
DEBUG="${DEBUG:-0}"

# -----------------------------
# Helpers
# -----------------------------

debug_echo() {
  if [[ "$DEBUG" == "1" ]]; then
    echo "DEBUG: $*" >&2
  fi
}

echo_step() {
  echo "🔍 Checking $1 ($2)..."
}

check_url_ttfb() {
  local url="$1"
  local attempts=$TTFB_ITERATIONS
  local success_count=0
  local total_ms=0
  local last_result=""

  for ((i=1; i<=attempts; i++)); do
    local out
    out="$(curl -A "$UA" -L -o /dev/null \
      --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
      -w "code=%{http_code} ttfb=%{time_starttransfer}" \
      -sS "$url" 2>/dev/null || true)"

    debug_echo "TTFB attempt $i for $url → out=<$out>"

    if [[ -z "$out" ]]; then
      last_result="FAIL curl_error"
      continue
    fi

    local code ttfb
    code="$(sed -E -n 's/.*code=([0-9]{3}).*/\1/p' <<<"$out")"
    ttfb="$(sed -E -n 's/.*ttfb=([0-9.]+).*/\1/p' <<<"$out")"

    if [[ -z "${code:-}" || -z "${ttfb:-}" ]]; then
      last_result="FAIL parse_error"
      continue
    fi

    if [[ "$code" == "000" ]]; then
      last_result="FAIL unreachable"
      continue
    fi

    if (( code >= 400 )); then
      last_result="FAIL http_$code"
      continue
    fi

    local ms
    ms="$(awk -v t="$ttfb" 'BEGIN{printf("%d", (t*1000)+0.5)}')"
    total_ms=$((total_ms + ms))
    success_count=$((success_count + 1))
    last_result="OK_TTFB ${ms}ms"
  done

  if (( success_count == 0 )); then
    echo "$last_result"
    return
  fi

  local avg_ms=$((total_ms / success_count))

  if (( avg_ms > LATENCY_THRESHOLD_MS )); then
    echo "SLOW_TTFB ${avg_ms}ms (avg of $success_count)"
  else
    echo "OK_TTFB ${avg_ms}ms (avg of $success_count)"
  fi
}

test_download_speed() {
  local url="$1"
  local out
  out="$(curl -A "$UA" -L -o /dev/null \
    --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
    -w "speed=%{speed_download} total_time=%{time_total} code=%{http_code}" \
    -sS "$url" 2>/dev/null || true)"

  debug_echo "Speed test for $url → out=<$out>"

  if [[ -z "$out" ]]; then
    echo "FAIL curl_error"
    return
  fi

  local speed code
  speed="$(sed -E -n 's/.*speed=([0-9.]+).*/\1/p' <<<"$out")"
  code="$(sed -E -n 's/.*code=([0-9]{3}).*/\1/p' <<<"$out")"

  if [[ -z "$speed" ]] || [[ -z "$code" ]] || [[ "$code" == "000" ]] || (( ${code:-0} >= 400 )); then
    echo "FAIL download_error"
    return
  fi

  local kbps
  kbps="$(awk -v s="$speed" 'BEGIN{printf("%.1f", s/1024)}')"

if [[ $(awk -v s="$speed" -v t="$SPEED_THRESHOLD_KBPS" 'BEGIN{if (s < t * 1024) print "true"}') == "true" ]]; then
    echo "SLOW_SPEED ${kbps}KBps"
  else
    echo "OK_SPEED ${kbps}KBps"
  fi
}

print_hint() {
  local title="$1"
  local hint="$2"
  echo "------------------------------------------------------------------------------------------------"
  echo "[MIRROR SUGGESTED] $title"
  echo "$hint"
  echo
}



# -----------------------------
# Evaluate & collect mirror suggestions
# -----------------------------

declare -a MIRROR_SUGGESTIONS=()

evaluate_result() {
  local name="$1"
  local ttfb_result="$2"
  local speed_result="$3"
  local hint_text="$4"
  local mirror_cmd="$5"

  local show_hint=false
  local reason=""

  if [[ "$ttfb_result" != OK_TTFB* ]]; then
    show_hint=true
    reason="$ttfb_result"
  elif [[ "$speed_result" == SLOW_SPEED* ]]; then
    show_hint=true
    reason="$speed_result"
  elif [[ "$speed_result" == FAIL* ]]; then
    show_hint=true
    reason="$speed_result"
  fi

  if [[ "$show_hint" == true ]]; then
    print_hint "$name is slow/unreachable: $reason" "$hint_text"
    if [[ -n "$mirror_cmd" ]]; then
      MIRROR_SUGGESTIONS+=("$mirror_cmd")
    fi
  fi
}

test_endpoint() {
  local name="$1"
  local ttfb_url="$2"
  local speed_url="$3"
  local full_advice="$4"
  local quick_fix="$5"

  echo_step "$name" "connection (TTFB)"
  local ttfb_result
  ttfb_result="$(check_url_ttfb "$ttfb_url")"

  local speed_result="SKIPPED"
  if [[ "$ttfb_result" == OK_TTFB* ]]; then
    if [[ -n "$speed_url" ]]; then
      echo_step "$name" "download speed"
      speed_result="$(test_download_speed "$speed_url" 2>/dev/null || echo "FAIL url_may_be_expired")"
    fi
  else
    if [[ -n "$speed_url" ]]; then
      speed_result="SKIPPED_DUE_TO_TTFB_FAILURE"
    fi
  fi

  evaluate_result "$name" "$ttfb_result" "$speed_result" "$full_advice" "$quick_fix"
}

# 1) UV installation: GitHub
test_endpoint \
  "UV installation (GitHub)" \
  "$GITHUB_CURL_TEST_URL" \
  "$GITHUB_TEST_URL" \
  "Consider installing uv via pip and configuring a PyPI mirror:
  python -m pip install -U pip
  pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  pip install -U uv" \
  "pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/"

# 2) uv python install: python-standalone
test_endpoint \
  "uv bootstrapping / Python install (python-standalone)" \
  "$PYTHON_STANDALONE_CURL_TEST_URL" \
  "$PYTHON_STANDALONE_TEST_URL" \
  "Consider setting a mirror for uv's Python standalone builds, e.g.:
  uv python install 3.12 --mirror https://python-standalone.org/mirror/astral-sh/python-build-standalone

Or set an environment variable (applies to subsequent commands):
  export UV_PYTHON_INSTALL_MIRROR=https://python-standalone.org/mirror/astral-sh/python-build-standalone" \
  "export UV_PYTHON_INSTALL_MIRROR=https://python-standalone.org/mirror/astral-sh/python-build-standalone"

# 3) uv sync: PyPI
test_endpoint \
  "uv sync (PyPI)" \
  "$PYPI_CURL_TEST_URL" \
  "$PYPI_TEST_URL" \
  "Consider setting a PyPI mirror (index used by uv):
  export UV_INDEX=https://mirrors.aliyun.com/pypi/simple/" \
  "export UV_INDEX=https://mirrors.aliyun.com/pypi/simple/"

# 4) Start TuFT: HuggingFace
test_endpoint \
  "HuggingFace access" \
  "$HF_CURL_TEST_URL" \
  "$HF_TEST_URL" \
  "Consider setting a HuggingFace mirror endpoint:
  export HF_ENDPOINT=https://hf-mirror.com" \
  "export HF_ENDPOINT=https://hf-mirror.com"


# -----------------------------
# Final summary of mirror commands
# -----------------------------
if [[ ${#MIRROR_SUGGESTIONS[@]} -gt 0 ]]; then
  echo "================================================================================================"
  echo "✅ Recommended mirror settings (copy & paste to apply):"
  echo
  for cmd in "${MIRROR_SUGGESTIONS[@]}"; do
    echo "  $cmd"
  done
  echo
  echo "💡 Tip: Add these to your ~/.bashrc or ~/.zshrc to make them persistent."
  echo "================================================================================================"
else
  echo "✅ All services appear responsive. No mirror needed at this time."
fi