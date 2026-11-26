#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && -r "${APS5_FORBIDDEN_WORDS:-}" ]] || exit 2
pattern=$(sed -e '/^[[:space:]]*$/d' -e 's/[.]/\\./g' "$APS5_FORBIDDEN_WORDS" | paste -sd'|')
[[ -n "$pattern" ]] || exit 2
status=0
LC_ALL=C.UTF-8 grep -aiE "(^|[^[:alnum:]])($pattern|PPSA[[:digit:]]+|CUSA[[:digit:]]+)" "$1" > /dev/null || status=$?
case "$status" in
  0) exit 1 ;;
  1) exit 0 ;;
  *) exit 2 ;;
esac
