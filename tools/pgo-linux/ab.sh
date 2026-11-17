#!/usr/bin/env bash
set -uo pipefail
dev=${APS5_DEV:-$HOME/aps5}
here=$(cd "$(dirname "$0")" && pwd)
tree=${TREE:-src-w10}
seed=${SEED:-sub-long-15}
first=${1%%=*}
for pass in "$@"; do
  label=${pass%%=*}; bdir=${pass#*=}
  echo "== $label ($bdir)"
  BUILD_DIR=$bdir bash "$here/bench.sh" "$tree" "$seed" "$label" --draws 1000 --mode varied --seconds 30 --compare-to "$first" 2>&1 | grep -E '^\[aps5\] (banc|[0-9]+ images|µs/dessin|IMAGE)'
done
