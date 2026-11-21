#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[CI] 1/4 Guardrails & Frozen Evals"
node tools/check-guardrails.mjs

echo "[CI] 2/4 Feature Map Audit"
if [ -f .agents/skills/verify-m1triton/scripts/audit-feature-map.mjs ]; then
  node .agents/skills/verify-m1triton/scripts/audit-feature-map.mjs
fi

echo "[CI] 3/4 C++ Configure, Build, CTest"
CXX_BIN="/usr/bin/clang++"
if [ ! -x "$CXX_BIN" ]; then
  CXX_BIN="$(command -v clang++)"
fi
PY_BIN="$(command -v python3)"
PYBIND11_CMAKE_DIR="$("$PY_BIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DCMAKE_OBJCXX_COMPILER="$CXX_BIN" \
  -DPython3_EXECUTABLE="$PY_BIN" \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"
cmake --build build -j
ctest --test-dir build --output-on-failure

echo "[CI] 4/4 Python Tests"
PYTHONPATH="$ROOT_DIR/python:$ROOT_DIR/build${PYTHONPATH:+:$PYTHONPATH}" pytest -q tests/python

echo "[CI] ALL GREEN"
