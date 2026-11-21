#!/usr/bin/env bash
set -uo pipefail
usage() {
  echo "usage: bench.sh --bench <e0bench> --list <programs.txt> --recipes <dir written by emit.sh> [--captures <dir>] [--rounds N] [--gpu <name>]" >&2
  exit 64
}
bench=""; list=""; recipes=""; captures=""; rounds=30; gpu=""
while [ $# -gt 0 ]; do
  case $1 in
    --bench) bench=$2; shift 2;;
    --list) list=$2; shift 2;;
    --recipes) recipes=$2; shift 2;;
    --captures) captures=$2; shift 2;;
    --rounds) rounds=$2; shift 2;;
    --gpu) gpu=$2; shift 2;;
    *) usage;;
  esac
done
[ -n "$bench" ] && [ -n "$list" ] && [ -n "$recipes" ] || usage
while read -r hash batch size capture; do
  [ -n "$hash" ] || continue
  case $size in *x*) ;; *) capture=$size; size="";; esac
  if [ -n "$capture" ] && [ -d "$recipes/captured/$capture" ]; then
    "$bench" --recipe "$recipes/captured/$capture/recipe.txt" --mem-root "$captures/$capture" --batch "$batch" --rounds "$rounds" \
      --barrier 2 --check 2 ${gpu:+--gpu "$gpu"}
    continue
  fi
  for dir in "$recipes"/synthetic/*-"$hash"-m*; do
    [ -f "$dir/recipe.txt" ] || continue
    if ls "$dir"/mem-*.bin >/dev/null 2>&1; then
      "$bench" --recipe "$dir/recipe.txt" --mem-root "$dir" --batch "$batch" --rounds "$rounds" --barrier 2 --check 2 \
        ${size:+--size "$size"} ${gpu:+--gpu "$gpu"}
    else
      "$bench" --recipe "$dir/recipe.txt" --stats-only ${gpu:+--gpu "$gpu"}
    fi
  done
done < "$list"
