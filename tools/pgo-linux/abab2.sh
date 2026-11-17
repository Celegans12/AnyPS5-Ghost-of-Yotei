#!/usr/bin/env bash
# A/B/A/B en jeu de deux compilations, sous UNE prise de la place de mesure (copie durcie de AnyPS5/linux-dev/abab.sh) :
#   python3 AnyPS5/machine_slot.py run --kind measure --owner <nom> -- bash <arbre>/tools/pgo-linux/abab2.sh <préfixe> <build A> <build B> [tours=2] [intervalles=6]
# Ajouts : le jeu resté en vie après une session (arrêt bloqué) est tué avant la suivante, sans quoi core-ingame.sh refuse les suivantes
# (« Subnautica tourne déjà ») ; les échecs de session sont imprimés ; un échantillonneur note, toutes les 3 s, les compilations qui tournent sur la
# machine (une compilation pendant une mesure la fausse) et chaque session attend (QUIET_MAX=900 s) que plus aucune compilation ne tourne ; les chiffres viennent d'une fenêtre fixe (abab-read.py), pas des 6 derniers intervalles.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dev=${APS5_DEV:-$HOME/aps5}
ldev=$dev/AnyPS5/linux-dev
prefix=${1:?préfixe}; build_a=${2:?build A}; build_b=${3:?build B}; turns=${4:-2}; intervals=${5:-6}
cd "$dev"
work=$dev/logs/$prefix-abab; mkdir -p "$work"; : > "$work/load.log"; : > "$work/times.log"
( while :; do
    n=$(ps -eo comm | grep -cE '^(cc1plus|cc1|ld|ld.bfd|collect2|lto1|rustc|clang)$')
    echo "$(date +%s) $n $(cut -d' ' -f1 /proc/loadavg)" >> "$work/load.log"; sleep 3
  done ) &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
compiling() { ps -eo comm | grep -cE '^(cc1plus|cc1|ld|ld.bfd|collect2|lto1|rustc|clang)$'; }
quiet() {
  local waited=0 calm=0
  while [ $waited -lt "${QUIET_MAX:-900}" ]; do
    if [ "$(compiling)" -eq 0 ]; then calm=$((calm + 1)); [ $calm -ge 5 ] && return 0; else calm=0; fi
    sleep 3; waited=$((waited + 3))
  done
  echo "[abab2] machine jamais calme avant $1 ($(compiling) compilations) : session lancée quand même"
}
cleanup() {
  pkill -f "probe.py $1 " 2>/dev/null
  if pgrep -x Subnautica > /dev/null; then
    echo "[abab2] jeu resté en vie après $1 : arrêt"; pkill -TERM -x Subnautica; sleep 8; pkill -KILL -x Subnautica 2>/dev/null; sleep 1
  fi
}
for i in $(seq 1 "$turns"); do
  for arm in a b; do
    name="$prefix-$arm$i"; b=$build_a; [ $arm = b ] && b=$build_b
    cleanup "$name"
    quiet "$name"
    t0=$(date +%s)
    timeout 480 bash "$ldev/headless.sh" bash -c "python3 '$ldev/probe.py' '$name' --perf 0 --no-shot > logs/probe-$name.log 2>&1 & bash '$ldev/core-ingame.sh' '$name' '' $intervals --build '$b'; wait" 2>&1 | grep -E "^$name |arrêt"
    echo "$name $t0 $(date +%s)" >> "$work/times.log"
    cleanup "$name"
  done
done
python3 "$here/abab-read.py" "$dev/Build/Subnautica/core-linux/bench" "$prefix" "$turns" "$work/load.log"
