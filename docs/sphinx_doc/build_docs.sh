#!/bin/bash
# Build script for TuFT documentation
# Usage:
#   ./build_docs.sh                # Build all versions from switcher.json
#   ./build_docs.sh --debug-current # Build docs from current branch only

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_EN="source"
SRC_ZH="source_zh"
OUT_ROOT="${ROOT_DIR}/build/html"
OUT_EN="${OUT_ROOT}/en"
OUT_ZH="${OUT_ROOT}/zh"
SWITCHER_JSON="${ROOT_DIR}/docs/sphinx_doc/switcher.json"

if [[ -x "${ROOT_DIR}/.venv/bin/sphinx-build" ]]; then
  SPHINX_BUILD=( "${ROOT_DIR}/.venv/bin/sphinx-build" )
else
  SPHINX_BUILD=( uv run sphinx-build )
fi

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

get_versions() {
  "$PYTHON_BIN" - <<'PY' "$1"
import json
import sys

switcher_path = sys.argv[1]
with open(switcher_path, "r", encoding="utf-8") as f:
    items = json.load(f)

for item in items:
    print(f"{item['version']} {item.get('tag','')}")
PY
}

DEBUG_CURRENT=0
if [[ "${1:-}" == "--debug-current" ]]; then
  DEBUG_CURRENT=1
  DEBUG_VERSION="latest"
  echo "[build_docs] Debug build on current branch -> ${DEBUG_VERSION}"
  mkdir -p "${OUT_EN}/${DEBUG_VERSION}" "${OUT_ZH}/${DEBUG_VERSION}"
  "${SPHINX_BUILD[@]}" -b html "${SRC_EN}" "${OUT_EN}/${DEBUG_VERSION}"
  "${SPHINX_BUILD[@]}" -b html "${SRC_ZH}" "${OUT_ZH}/${DEBUG_VERSION}"
fi

NEED_STASH=0
SWITCHER_SOURCE="${SWITCHER_JSON}"
if ! git -C "${ROOT_DIR}" diff --quiet || ! git -C "${ROOT_DIR}" diff --cached --quiet; then
  echo "[build_docs] Working tree is not clean. Stashing changes for tag builds..."
  if [[ -f "${SWITCHER_JSON}" ]]; then
    SWITCHER_TMP="$(mktemp)"
    cp "${SWITCHER_JSON}" "${SWITCHER_TMP}"
    SWITCHER_SOURCE="${SWITCHER_TMP}"
  fi
  git -C "${ROOT_DIR}" stash push -u -m "build_docs_temp" >/dev/null
  NEED_STASH=1
fi

ORIG_REF="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || git -C "${ROOT_DIR}" rev-parse HEAD)"

cleanup() {
  if [[ -n "${ORIG_REF:-}" ]]; then
    git -C "${ROOT_DIR}" checkout "${ORIG_REF}" >/dev/null 2>&1 || true
  fi
  if [[ "${NEED_STASH}" == "1" ]]; then
    git -C "${ROOT_DIR}" stash pop >/dev/null 2>&1 || true
  fi
  if [[ -n "${SWITCHER_TMP:-}" && -f "${SWITCHER_TMP}" ]]; then
    rm -f "${SWITCHER_TMP}"
  fi
}

trap cleanup EXIT

echo "[build_docs] Building versions from switcher.json"
while read -r VERSION TAG; do
  if [[ "${DEBUG_CURRENT}" == "1" && "${VERSION}" == "latest" ]]; then
    continue
  fi
  if [[ -z "${TAG}" && "${VERSION}" == "latest" ]]; then
    TAG="main"
  fi
  if [[ -z "${TAG}" ]]; then
    echo "[build_docs] Missing tag for version ${VERSION} in switcher.json"
    exit 1
  fi
  if [[ "${TAG}" != "main" ]]; then
    if ! git -C "${ROOT_DIR}" rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
      echo "[build_docs] Tag ${TAG} not found. Please create it or update switcher.json."
      exit 1
    fi
  fi
  git -C "${ROOT_DIR}" checkout "${TAG}" >/dev/null
  tree $ROOT_DIR/docs
  if [[ ! -d "${ROOT_DIR}/docs/sphinx_doc/${SRC_EN}" || ! -d "${ROOT_DIR}/docs/sphinx_doc/${SRC_ZH}" ]]; then
    echo "[build_docs] Missing docs sources after checkout ${TAG}. Ensure docs/sphinx_doc exists on this ref."
    exit 1
  fi
  echo "[build_docs] Building English documentation -> ${OUT_EN}/${VERSION}"
  mkdir -p "${OUT_EN}/${VERSION}"
  "${SPHINX_BUILD[@]}" -b html "${SRC_EN}" "${OUT_EN}/${VERSION}"

  echo "[build_docs] Building Chinese documentation -> ${OUT_ZH}/${VERSION}"
  mkdir -p "${OUT_ZH}/${VERSION}"
  "${SPHINX_BUILD[@]}" -b html "${SRC_ZH}" "${OUT_ZH}/${VERSION}"
done < <(get_versions "$SWITCHER_SOURCE")

# Copy switcher.json to output root for local development
echo "[build_docs] Copying switcher.json..."
cp "${SWITCHER_SOURCE}" "${OUT_ROOT}/switcher.json"

echo "[build_docs] Done. Output at ${OUT_ROOT} (subdirs: en/, zh/)"
