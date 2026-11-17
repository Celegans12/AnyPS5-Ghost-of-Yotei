#!/usr/bin/env bash
set -uo pipefail
dev=${APS5_DEV:-$HOME/aps5}
here=$(cd "$(dirname "$0")" && pwd)
tree=${TREE:-src-w10}
dir=$dev/$tree
profile=${PROFILE:-$dir/build-profile}
seed=${SEED:-sub-long-15}
atomic=(-DCMAKE_C_FLAGS=-fprofile-update=atomic -DCMAKE_CXX_FLAGS=-fprofile-update=atomic)
use_args=(-DANYPS5_PGO=use "-DANYPS5_PGO_DIR=$profile")
borrow() {
  mkdir -p "$dir/$1/bench-tool"
  cp "$dir/build/bench-tool/agc_frame_replay" "$dir/$1/bench-tool/agc_frame_replay"
}
step=${1:?tool|gen|train|use|lto|all}
case $step in
  tool) bash "$dev/AnyPS5/linux-dev/build-bench-tool.sh" "$tree" ;;
  gen)
    bash "$here/build.sh" build-pgogen -DANYPS5_PGO=generate "-DANYPS5_PGO_DIR=$profile" "${atomic[@]}" && borrow build-pgogen ;;
  train)
    rm -rf "$profile"
    python3 "$dev/AnyPS5/machine_slot.py" run --kind measure --owner "${OWNER:-pgo}" -- \
      env BUILD_DIR=build-pgogen bash "$here/bench.sh" "$tree" "$seed" pgo-train --draws 1000 --mode varied --seconds 30
    code=$?
    echo "profil : $(find "$profile" -name '*.gcda' 2>/dev/null | wc -l) fichiers .gcda dans $profile"
    exit $code ;;
  use)
    NO_CCACHE=1 bash "$here/build.sh" build-pgo "${use_args[@]}" && borrow build-pgo ;;
  lto)
    NO_CCACHE=1 bash "$here/build.sh" build-pgolto "${use_args[@]}" -DANYPS5_LTO=ON "-DANYPS5_LTO_EXCLUDE=${LTO_EXCLUDE:-libc}" && borrow build-pgolto ;;
  game)
    [ -d "$dir/build-profile-bench" ] || cp -a "$profile" "$dir/build-profile-bench"
    name=${NAME:-pgo-train-game}
    cd "$dev" && python3 AnyPS5/machine_slot.py run --kind measure --owner "${OWNER:-pgo}" -- \
      timeout 420 bash AnyPS5/linux-dev/headless.sh bash -c "python3 AnyPS5/linux-dev/probe.py $name --perf 0 --no-shot > logs/probe-$name.log 2>&1 & bash AnyPS5/linux-dev/core-ingame.sh $name \"\" ${INTERVALS:-3} --build $tree/build-pgogen; wait"
    code=$?
    echo "profil : $(find "$profile" -name '*.gcda' | wc -l) fichiers .gcda, $(du -sk "$profile" | cut -f1) Kio (avant : $(du -sk "$dir/build-profile-bench" | cut -f1) Kio)"
    exit $code ;;
  use2)
    NO_CCACHE=1 bash "$here/build.sh" build-pgo2 "${use_args[@]}" && borrow build-pgo2 ;;
  all) bash "$0" gen && bash "$0" train && bash "$0" use && bash "$0" lto ;;
  *) echo "usage : pgo.sh tool|gen|train|use|lto|game|use2|all"; exit 2 ;;
esac
