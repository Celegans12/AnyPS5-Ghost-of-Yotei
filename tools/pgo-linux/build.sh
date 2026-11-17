#!/usr/bin/env bash
set -uo pipefail
dev=${APS5_DEV:-$HOME/aps5}
. "$dev/AnyPS5/linux-dev/env.sh"
tree=${TREE:-src-w10}
name=${1:?dossier de compilation relatif a l arbre, par exemple build-pgogen}; shift
dir=$dev/$tree
build=$dir/$name
ffmpeg=$(sed -n 's/^FFMPEG_PREBUILT_DIR:[A-Z]*=//p' "$dir/build/CMakeCache.txt" 2>/dev/null | head -1)
ffmpeg_args=(); [ -n "$ffmpeg" ] && ffmpeg_args=(-DFFMPEG_PREBUILT_DIR="$ffmpeg")
launcher=ccache; [ "${NO_CCACHE:-0}" = 1 ] && launcher=
mkdir -p "$dev/logs"
log=$dev/logs/pgo-$name-$(date +%H%M%S).log
cmake -S "$dir" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_WERROR=OFF \
  -DANYPS5_RELINKER_ONLY=OFF -DANYPS5_RELINKER_CLI=ON -DANYPS5_USE_SYSTEM_DEPS=OFF \
  -DCMAKE_C_COMPILER="$APS5_CC" -DCMAKE_CXX_COMPILER="$APS5_CXX" \
  -DCMAKE_C_COMPILER_LAUNCHER=$launcher -DCMAKE_CXX_COMPILER_LAUNCHER=$launcher \
  "${ffmpeg_args[@]}" "$@" > "$log.configure" 2>&1 || { tail -30 "$log.configure"; aps5_die "configuration echouee ($log.configure)"; }
grep -E 'CMake Error|Configuring (done|incomplete)|Generating done' "$log.configure" | tail -3
t1=$(date +%s)
python3 "$dev/AnyPS5/machine_slot.py" run --kind build --owner "${OWNER:-pgo}" -- \
  cmake --build "$build" --target libs agc_frame_replay --parallel "$(aps5_jobs)" > "$log" 2>&1
code=$?
grep -E 'error:|FAILED|ninja: build stopped|undefined reference|internal compiler error' "$log" | head -14 | cut -c1-320
aps5_say "compilation de $name : sortie $code en $(( $(date +%s) - t1 )) s (journal $log)"
exit $code
