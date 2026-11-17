#!/usr/bin/env bash
set -uo pipefail
dev=${APS5_DEV:-$HOME/aps5}
here=$(cd "$(dirname "$0")" && pwd)
tree=${TREE:-src-w10}
cd "$dev"
for pass in "$@"; do
  name=${pass%%=*}; bdir=${pass#*=}
  echo "== $name ($bdir)"
  timeout 260 bash AnyPS5/linux-dev/headless.sh bash -c "python3 AnyPS5/linux-dev/probe.py $name --perf 0 --no-shot > logs/probe-$name.log 2>&1 & bash AnyPS5/linux-dev/core-ingame.sh $name \"\" 6 --build $tree/$bdir; wait" 2>&1 | grep -E 'img/s|arrêt|jamais|non atteint|aucune'
done
python3 -I "$here/session.py" $(for pass in "$@"; do echo "Build/Subnautica/core-linux/bench/${pass%%=*}"; done)
