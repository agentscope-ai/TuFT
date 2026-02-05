#!/usr/bin/env bash
set -euo pipefail

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
  local out
  out="$(curl -A "$UA" -L -o /dev/null \
    --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
    -w "code=%{http_code} ttfb=%{time_starttransfer}" \
    -sS "$url" 2>/dev/null || true)"

  debug_echo "TTFB check for $url → out=<$out>"

  if [[ -z "$out" ]]; then
    echo "FAIL curl_error"
    return
  fi

  local code ttfb
  code="$(sed -E -n 's/.*code=([0-9]{3}).*/\1/p' <<<"$out")"
  ttfb="$(sed -E -n 's/.*ttfb=([0-9.]+).*/\1/p' <<<"$out")"

  if [[ -z "${code:-}" || -z "${ttfb:-}" ]]; then
    echo "FAIL parse_error"
    return
  fi

  if [[ "$code" == "000" ]]; then
    echo "FAIL unreachable"
    return
  fi

  if (( code >= 400 )); then
    echo "FAIL http_$code"
    return
  fi

  local ms
  ms="$(awk -v t="$ttfb" 'BEGIN{printf("%d", (t*1000)+0.5)}')"

  if (( ms > LATENCY_THRESHOLD_MS )); then
    echo "SLOW_TTFB ${ms}ms"
  else
    echo "OK_TTFB ${ms}ms"
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
# Realistic test URLs
# -----------------------------

GITHUB_TEST_URL="https://github.com/astral-sh/uv/releases/download/0.9.30/source.tar.gz"
PYTHON_STANDALONE_TEST_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260203/cpython-3.10.19+20260203-aarch64-apple-darwin-install_only.tar.gz"
PYPI_TEST_URL="https://files.pythonhosted.org/packages/4e/a0/63cea38fe839fb89592728b91928ee6d15705f1376a7940fee5bbc77fea0/uv-0.9.30.tar.gz"
HF_TEST_URL="https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json"

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

# --- 1) UV installation: GitHub ---
echo_step "GitHub-uv installation" "connection (TTFB)"
r1_ttfb="$(check_url_ttfb "https://github.com/")"
r1_speed="SKIPPED"
if [[ "$r1_ttfb" == OK_TTFB* ]]; then
  echo_step "GitHub-uv installation" "download speed"
  r1_speed="$(test_download_speed "$GITHUB_TEST_URL")"
fi
evaluate_result "UV installation (GitHub)" "$r1_ttfb" "$r1_speed" \
"Consider installing uv via pip and configuring a PyPI mirror:
  python -m pip install -U pip
  pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  pip install -U uv" \
"pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/"

# --- 2) uv python install: python-standalone ---
echo_step "python-standalone.org" "connection (TTFB)"
r2_ttfb="$(check_url_ttfb "https://python-standalone.org/")"
r2_speed="SKIPPED"
if [[ "$r2_ttfb" == OK_TTFB* ]]; then
  echo_step "python-standalone (release asset)" "download speed"
  r2_speed="$(test_download_speed "$PYTHON_STANDALONE_TEST_URL")"
else
  r2_speed="SKIPPED_DUE_TO_TTFB_FAILURE"
fi
evaluate_result "uv bootstrapping / Python install (python-standalone)" "$r2_ttfb" "$r2_speed" \
"Consider setting a mirror for uv's Python standalone builds, e.g.:
  uv python install 3.12 --mirror https://python-standalone.org/mirror/astral-sh/python-build-standalone

Or set an environment variable (applies to subsequent commands):
  export UV_PYTHON_INSTALL_MIRROR=https://python-standalone.org/mirror/astral-sh/python-build-standalone" \
"export UV_PYTHON_INSTALL_MIRROR=https://python-standalone.org/mirror/astral-sh/python-build-standalone"

# --- 3) uv sync: PyPI ---
echo_step "PyPI" "connection (TTFB)"
r3_ttfb="$(check_url_ttfb "https://pypi.org/simple/")"
r3_speed="SKIPPED"
if [[ "$r3_ttfb" == OK_TTFB* ]]; then
  echo_step "PyPI" "download speed"
  r3_speed="$(test_download_speed "$PYPI_TEST_URL" || echo "FAIL url_may_be_expired")"
fi
evaluate_result "uv sync (PyPI)" "$r3_ttfb" "$r3_speed" \
"Consider setting a PyPI mirror (index used by uv):
  export UV_INDEX=https://mirrors.aliyun.com/pypi/simple/" \
"export UV_INDEX=https://mirrors.aliyun.com/pypi/simple/"

# --- 4) Start TuFT: HuggingFace ---
echo_step "HuggingFace" "connection (TTFB)"
r4_ttfb="$(check_url_ttfb "https://huggingface.co/")"
r4_speed="SKIPPED"
if [[ "$r4_ttfb" == OK_TTFB* ]]; then
  echo_step "HuggingFace" "download speed"
  r4_speed="$(test_download_speed "$HF_TEST_URL")"
fi
evaluate_result "HuggingFace access" "$r4_ttfb" "$r4_speed" \
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
