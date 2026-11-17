#!/usr/bin/env bash
set -uo pipefail
APS5_DEV=${APS5_DEV:-$HOME/aps5}
here=$APS5_DEV/AnyPS5/linux-dev
. "$here/env.sh"
tree=${1:?arbre}; seedname=${2:?graine}; label=${3:?étiquette}; shift 3
draws=3000; seconds=10; mode=varied; verify=4; images=2; extra=(); ref=""
while [ $# -gt 0 ]; do
  case $1 in
    --draws) draws=$2; shift 2;; --seconds) seconds=$2; shift 2;; --mode) mode=$2; shift 2;;
    --verify) verify=$2; shift 2;; --verify-images) images=$2; shift 2;; --env) extra+=("$2"); shift 2;;
    --compare-to) ref=$2; shift 2;;
    *) aps5_die "argument inconnu : $1";;
  esac
done
src="$APS5_DEV/$tree"; bdir=${BUILD_DIR:-build}; libs="$src/$bdir/core/libs/libs/unpatched"
exe="$src/$bdir/core/libs/prx/libSceAgcDriver/agc_frame_replay"
[ -d "$libs" ] || aps5_die "bibliothèques absentes : build-tree.sh $tree libs agc_frame_replay"
if ! grep -q -- '"--bench"' "$src/core/libs/prx/libSceAgcDriver/tools/FrameReplay.cpp" 2>/dev/null; then
  exe="$src/$bdir/bench-tool/agc_frame_replay"
  aps5_say "$tree ne connaît pas --bench : outil emprunté ($exe)"
fi
[ -x "$exe" ] || aps5_die "outil du banc absent : $exe (build-bench-tool.sh $tree)"
seed="$APS5_DEV/bench-seeds/$seedname"; [ -d "$seed" ] || aps5_die "graine absente : $seed"
work="$APS5_DEV/bench-seeds/runs/$label"
mkdir -p "$work/trace"; rm -f "$work"/trace/* "$work"/verify-*.bmp
for v in $(env | grep -oE '^(APS5|ANYPS5)_[A-Z0-9_]+'); do
  case $v in APS5_DEV|APS5_CC|APS5_CXX|APS5_WIN_ROOT) ;; *) unset "$v";; esac
done
export APS5_TRACE_SUMMARY_EVERY=5 APS5_TRACE_DIR="$work/trace" APS5_PIPELINE_STATS=2 ANYPS5_VOLUME=0 \
       APS5_DRAW_PIPELINE=1 APS5_PIPELINE_DISPATCH=1 APS5_PIPELINE_DISPATCH_MISSES=1 ANYPS5_NO_SHADER_CACHE=1
for kv in "${extra[@]}"; do export "$kv"; done
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-offscreen} SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
cd "$work"
t0=$(date +%s)
timeout 600 "$exe" "$seed" --bench --draws "$draws" --seconds "$seconds" --mode "$mode" --rewrite 0 --ahead 0 \
  --verify-frames "$verify" --verify-images "$images" > bench.out 2> bench.err
code=$?
echo "[aps5] banc : sortie $code en $(( $(date +%s) - t0 )) s ($work)"
grep -m3 -iE 'write watch|write-watch\]' bench.err | cut -c1-200
python3 -I - "$work/bench.out" "$ref" "$APS5_DEV" <<'P'
import json, os, sys
def load(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith('{"draw_bench"'):
            return json.loads(line)
out, ref, dev = sys.argv[1:4]
d = load(out)
if d is None:
    print("[aps5] pas de ligne draw_bench dans bench.out"); sys.exit(1)
roles = d.get("roles", {}); draws = d["per_frame"]["draws"] or 1; frames = max(1, d["frames"])
print(f"[aps5] {d['frames']} images, {d['fps']:.2f} img/s, {draws:.0f} dessins/image, GPU {d['gpu']['busy_ms'] if d['gpu']['measured'] else 0:.1f} ms/image (mesuré : {d['gpu']['measured']})")
print("[aps5] µs/dessin " + " · ".join(f"{r} {1000.0 * roles.get(r, {}).get('work_ms', 0.0) / frames / draws:.1f}" for r in ("decode", "finisher", "committer", "walker")))
mine = {v["frame"]: v for v in d.get("verify", [])}
for f in sorted(mine): print("[aps5]   image", f, mine[f]["width"], "x", mine[f]["height"], " ".join(mine[f]["buffers"]))
if ref:
    path = ref if os.path.isdir(ref) else os.path.join(dev, "bench-seeds", "runs", ref)
    r = load(os.path.join(path, "bench.out"))
    if r is None: print("[aps5] référence illisible :", path); sys.exit(2)
    theirs = {v["frame"]: v for v in r.get("verify", [])}
    common = [f for f in sorted(mine) if f in theirs and f >= 1]
    same = [f for f in common if mine[f]["buffers"] == theirs[f]["buffers"]]
    if not common: print("[aps5] aucune image commune à comparer"); sys.exit(2)
    if len(same) == len(common): print(f"[aps5] IMAGE IDENTIQUE à {os.path.basename(path)} (images {common})"); sys.exit(0)
    print(f"[aps5] IMAGE DIFFÉRENTE de {os.path.basename(path)} : identiques {same}, différentes {[f for f in common if f not in same]}"); sys.exit(3)
P
rc=$?
[ $code -ne 0 ] && { grep -v '^\[exception\]' bench.err | tail -15; exit $code; }
exit $rc
