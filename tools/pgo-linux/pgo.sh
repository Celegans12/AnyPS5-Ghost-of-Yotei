#!/usr/bin/env bash
# PGO de la tête d'un arbre AnyPS5 sous Linux (gcc-14). Une commande refait tout, profil pris EN JEU :
#   OWNER=<nom> bash <arbre>/tools/pgo-linux/pgo.sh recipe
# Étapes (chacune se lance seule) : tool gen shim game train use lto verify use2 recipe all
#   tool    outil du banc (build-bench-tool.sh : compile aussi <arbre>/build, sans PGO)
#   gen     <arbre>/build-pgogen : compilation instrumentée (-fprofile-generate, mises à jour atomiques)
#   shim    <arbre>/build-pgogen/libgcov-preload.so (gcov-preload.sh) : sans elle les .prx instrumentées ne se chargent pas
#   game    entraînement EN JEU : une session headless.sh sur build-pgogen avec LD_PRELOAD=libgcov-preload.so ; efface
#           d'abord le profil (GAME_KEEP=1 pour compléter celui qui existe) ; INTERVALS (12 × 5 s) ; DUMP_EVERY (20 s)
#   train   entraînement au banc (repli) : graine sub-long-15, 1000 dessins varied, 30 s
#   use     <arbre>/build-pgo : compilation PGO sans ccache
#   verify  image identique : bench.sh sub-creative sur build-pgo contre sen-282c268d7b-sub (sous machine_slot, measure)
#   recipe  tool (si absent) → gen → shim → game (repli train si aucun .gcda) → use → verify
# Variables : TREE (défaut : l'arbre qui porte ce script), OWNER (nom pour machine_slot), PROFILE, SEED, NAME, INTERVALS.
set -uo pipefail
dev=${APS5_DEV:-$HOME/aps5}
here=$(cd "$(dirname "$0")" && pwd)
tree=${TREE:-$(basename "$(cd "$here/../.." && pwd)")}
dir=$dev/$tree
profile=${PROFILE:-$dir/build-profile}
seed=${SEED:-sub-long-15}
owner=${OWNER:-pgo}
atomic=(-DCMAKE_C_FLAGS=-fprofile-update=atomic -DCMAKE_CXX_FLAGS=-fprofile-update=atomic)
use_args=(-DANYPS5_PGO=use "-DANYPS5_PGO_DIR=$profile")
export TREE=$tree OWNER=$owner
borrow() {
  mkdir -p "$dir/$1/bench-tool"
  cp "$dir/build/bench-tool/agc_frame_replay" "$dir/$1/bench-tool/agc_frame_replay"
}
count() { find "$profile" -name '*.gcda' 2>/dev/null | wc -l; }
step=${1:?tool|gen|shim|game|train|use|lto|verify|use2|recipe|all}
case $step in
  tool) bash "$dev/AnyPS5/linux-dev/build-bench-tool.sh" "$tree" ;;
  gen)
    bash "$here/build.sh" build-pgogen -DANYPS5_PGO=generate "-DANYPS5_PGO_DIR=$profile" "${atomic[@]}" && borrow build-pgogen ;;
  shim) bash "$here/gcov-preload.sh" "$dir/build-pgogen/libgcov-preload.so" ;;
  game)
    [ -f "$dir/build-pgogen/libgcov-preload.so" ] || bash "$here/gcov-preload.sh" "$dir/build-pgogen/libgcov-preload.so" || exit 1
    [ "${GAME_KEEP:-0}" = 1 ] || rm -rf "$profile"
    name=${NAME:-pgo-train-game}
    vars="LD_PRELOAD=$dir/build-pgogen/libgcov-preload.so;APS5_GCOV_DUMP_EVERY=${DUMP_EVERY:-20}"
    cd "$dev" && python3 AnyPS5/machine_slot.py run --kind measure --owner "$owner" -- \
      timeout 900 bash AnyPS5/linux-dev/headless.sh bash -c "python3 AnyPS5/linux-dev/probe.py $name --perf 0 --no-shot > logs/probe-$name.log 2>&1 & bash AnyPS5/linux-dev/core-ingame.sh $name '$vars' ${INTERVALS:-12} --build $tree/build-pgogen; wait"
    code=$?
    echo "profil : $(count) fichiers .gcda, $(du -sk "$profile" 2>/dev/null | cut -f1) Kio dans $profile"
    exit $code ;;
  train)
    rm -rf "$profile"
    python3 "$dev/AnyPS5/machine_slot.py" run --kind measure --owner "$owner" -- \
      env BUILD_DIR=build-pgogen bash "$here/bench.sh" "$tree" "$seed" pgo-train --draws 1000 --mode varied --seconds 30
    code=$?
    echo "profil : $(count) fichiers .gcda dans $profile"
    exit $code ;;
  use)
    NO_CCACHE=1 bash "$here/build.sh" build-pgo "${use_args[@]}" && borrow build-pgo ;;
  lto)
    NO_CCACHE=1 bash "$here/build.sh" build-pgolto "${use_args[@]}" -DANYPS5_LTO=ON "-DANYPS5_LTO_EXCLUDE=${LTO_EXCLUDE:-libc}" && borrow build-pgolto ;;
  verify)
    python3 "$dev/AnyPS5/machine_slot.py" run --kind measure --owner "$owner" -- \
      env BUILD_DIR=${VERIFY_BUILD:-build-pgo} bash "$here/bench.sh" "$tree" sub-creative "${NAME:-pgo-verify}" --compare-to sen-282c268d7b-sub ;;
  use2)
    NO_CCACHE=1 bash "$here/build.sh" build-pgo2 "${use_args[@]}" && borrow build-pgo2 ;;
  recipe)
    [ -f "$dir/build/bench-tool/agc_frame_replay" ] || bash "$0" tool || exit 1
    bash "$0" gen || exit 1
    bash "$0" shim || exit 1
    bash "$0" game
    if [ "$(count)" -eq 0 ]; then echo "[pgo] aucun .gcda de la session en jeu : repli sur le banc d'entraînement"; bash "$0" train || exit 1; fi
    bash "$0" use || exit 1
    bash "$0" verify ;;
  all) bash "$0" gen && bash "$0" train && bash "$0" use && bash "$0" lto ;;
  *) echo "usage : pgo.sh tool|gen|shim|game|train|use|lto|verify|use2|recipe|all"; exit 2 ;;
esac
