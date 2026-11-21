#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/out}
mkdir -p "$out"
${CXX:-g++} -O2 -std=c++20 -o "$out/e0bench" "$here/e0bench.cpp" ${VULKAN_FLAGS:-} ${VULKAN_LIB:--lvulkan}
echo "ok $out/e0bench"
