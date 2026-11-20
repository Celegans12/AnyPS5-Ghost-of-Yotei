#!/usr/bin/env bash
set -uo pipefail
build=${1:?usage: lavapipe_tests.sh <build directory> [output directory]}
out=${2:-$build/lavapipe}
root=$(cd "$(dirname "$0")/.." && pwd)
skip_file=$root/tools/lavapipe_skip.txt
known_file=$root/tools/lavapipe_validation_known.txt
mkdir -p "$out"

icd=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json 2>/dev/null | head -n 1)
if [ -z "$icd" ]; then
  echo "error: the lavapipe driver (mesa-vulkan-drivers) is not installed" >&2
  exit 2
fi
export VK_ICD_FILENAMES=$icd
export ANYPS5_REQUIRE_VULKAN=1
export ANYPS5_NO_SHADER_CACHE=1
export GALLIUM_OVERRIDE_CPU_CAPS=avx
unset DISPLAY WAYLAND_DISPLAY

vulkaninfo --summary > "$out/vulkaninfo.txt" 2>&1 || true
if ! grep -q llvmpipe "$out/vulkaninfo.txt"; then
  echo "error: the Vulkan device is not llvmpipe" >&2
  cat "$out/vulkaninfo.txt" >&2
  exit 2
fi
grep -E 'deviceName|apiVersion|driverName|driverInfo|subgroupSize' "$out/vulkaninfo.txt" | head -n 8

tests='^(kyty_shim_.*|guest_gpu_fault|guest_host_write_filter|agc_kyty_shaders|agc_driver_frame_replay_.*)$'
excluded=$(grep -vE '^\s*(#|$)' "$skip_file" 2>/dev/null | cut -f1 | paste -sd'|')
exclude_args=()
if [ -n "$excluded" ]; then
  exclude_args=(-E "^(${excluded})$")
fi

sysctl_args=(-w vm.unprivileged_userfaultfd=1 vm.max_map_count=2147483642 kernel.core_pattern=core)
if [ "$(id -u)" = 0 ]; then
  sysctl "${sysctl_args[@]}" >/dev/null 2>&1 || true
elif sudo -n true 2>/dev/null; then
  sudo -n sysctl "${sysctl_args[@]}" >/dev/null 2>&1 || true
fi
ulimit -c 0

status=0
echo "::group::functional pass on lavapipe"
ctest --test-dir "$build" -R "$tests" "${exclude_args[@]}" --output-on-failure --timeout 300 --no-tests=error \
  --output-junit "$out/ctest.xml" --output-log "$out/ctest.log" || status=1
echo "::endgroup::"
python3 -I "$root/tools/check_skips.py" "$out/ctest.xml" | tee "$out/skips.txt" || status=1
if [ -n "$excluded" ]; then
  echo "tests left out because lavapipe lacks what they need (tools/lavapipe_skip.txt):"
  grep -vE '^\s*(#|$)' "$skip_file" | cut -f1,2
fi

echo "::group::validation pass on lavapipe (Khronos validation layer, synchronization validation)"
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
export KYTY_VULKAN_VALIDATION=1
ctest --test-dir "$build" -R "$tests" "${exclude_args[@]}" --output-on-failure --timeout 600 --no-tests=error \
  --output-junit "$out/validation.xml" --output-log "$out/validation.log" > "$out/validation.out" 2>&1
echo "::endgroup::"
python3 -I "$root/tools/lavapipe_validation_report.py" "$out/validation.xml" "$out/validation.log" --known "$known_file" --summary "$out/validation-summary.md" || status=1
cat "$out/validation-summary.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$out/validation-summary.md" >> "$GITHUB_STEP_SUMMARY"
fi
exit $status
