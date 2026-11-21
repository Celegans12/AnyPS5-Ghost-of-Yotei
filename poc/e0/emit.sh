#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "usage: emit.sh --tool <kyty_e0_emit> --code <store.code> --kpm <store.kpm> --list <programs.txt> --out <dir> [--captures <dir>] [--pattern one|half|rand|zero]" >&2
  echo "  programs.txt: one line per program: <hash> <batch> [WxH] [capture directory name]" >&2
  exit 64
}
tool=""; code=""; kpm=""; list=""; out=""; captures=""; pattern=one
while [ $# -gt 0 ]; do
  case $1 in
    --tool) tool=$2; shift 2;;
    --code) code=$2; shift 2;;
    --kpm) kpm=$2; shift 2;;
    --list) list=$2; shift 2;;
    --out) out=$2; shift 2;;
    --captures) captures=$2; shift 2;;
    --pattern) pattern=$2; shift 2;;
    *) usage;;
  esac
done
[ -n "$tool" ] && [ -n "$code" ] && [ -n "$kpm" ] && [ -n "$list" ] && [ -n "$out" ] || usage
mkdir -p "$out/captured" "$out/synthetic"
while read -r hash batch size capture; do
  [ -n "$hash" ] || continue
  case $size in *x*) ;; *) capture=$size; size="";; esac
  if [ -n "$capture" ] && [ -n "$captures" ] && [ -d "$captures/$capture" ]; then
    "$tool" --code "$code" --kpm "$kpm" --capture "$captures/$capture" --out "$out/captured" | tail -1
  else
    "$tool" --code "$code" --kpm "$kpm" --module "$hash" --synth --pattern "$pattern" --out "$out/synthetic" | tail -3
  fi
done < "$list"
