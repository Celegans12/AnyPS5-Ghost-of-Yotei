#!/usr/bin/env python3
"""Lit les sessions d'un A/B/A/B en jeu (aps5-summary.jsonl + stderr.log de chaque session) et imprime, par session et par bras,
img/s, ms par image, dessins par image, ms de décodage / finisher / committer par image (mêmes clés que ingame.py), GPU et µs par commit
de dessin. Fenêtre FIXE : WINDOW intervalles de 5 s qui suivent de SETTLE intervalles la première ligne « en jeu » (plus de 400 dessins par
image, image max < 400 ms), donc la même durée de jeu pour tous les bras quelle que soit la durée de la session.
Usage : abab-read.py <dossier bench> <préfixe> <tours> [load.log]   (load.log : lignes « epoque compilations charge » de abab2.sh)"""
import json, os, re, sys

bench, prefix, turns = sys.argv[1], sys.argv[2], int(sys.argv[3])
loadlog = sys.argv[4] if len(sys.argv) > 4 else None
WINDOW, SETTLE = int(os.environ.get("WINDOW", 8)), int(os.environ.get("SETTLE", 3))
loads = []
if loadlog and os.path.exists(loadlog):
    for line in open(loadlog):
        p = line.split()
        if len(p) == 3: loads.append((int(p[0]), int(p[1]), float(p[2])))
times = {}
tpath = os.path.join(os.path.dirname(loadlog), "times.log") if loadlog else None
if tpath and os.path.exists(tpath):
    for line in open(tpath):
        p = line.split()
        if len(p) == 3: times[p[0]] = (int(p[1]), int(p[2]))

def stage(rows, key):
    pres = max(1, sum(r["presents"] for r in rows))
    return sum(r.get("phase_totals_ms", {}).get(key, 0.0) for r in rows) / pres

def read(name):
    d = os.path.join(bench, name)
    try:
        rows = [json.loads(l) for l in open(os.path.join(d, "aps5-summary.jsonl")) if l.strip()]
    except OSError:
        return None
    ing = [i for i, r in enumerate(rows) if r["presents"] > 0 and (r.get("draws") or 0) / r["presents"] > 400 and r["frame_ms"]["max"] < 400]
    if not ing: return {"fail": "pas en jeu"}
    lo = ing[0] + SETTLE
    win = rows[lo:lo + WINDOW]
    if len(win) < WINDOW: return {"fail": f"{len(win)} intervalles seulement après le chargement"}
    if any(r["frame_ms"]["max"] >= 400 or r["presents"] == 0 for r in win): freeze = max(r["frame_ms"]["max"] for r in win)
    else: freeze = 0
    p = sum(r["presents"] for r in win); ms = sum(r["window"]["ms"] for r in win); dr = sum(r.get("draws") or 0 for r in win)
    t0, t1 = win[0]["interval_start_s"], win[-1]["interval_end_s"]
    n = c = 0.0
    try:
        k = 0
        for line in open(os.path.join(d, "stderr.log"), errors="replace"):
            m = re.match(r"^\[commits\] by kind \(10 s\): draw (\d+) \(([\d.]+) us\)", line)
            if not m: continue
            k += 1
            if t0 < 10 * k <= t1 + 5: n += int(m.group(1)); c += int(m.group(1)) * float(m.group(2))
    except OSError: pass
    out = {"imgs": p * 1000 / ms, "ms": ms / p, "dpi": dr / p, "dec": stage(win, "Driver.Packet"), "fin": stage(win, "Driver.DrawFinish"),
           "com": stage(win, "Vulkan.DrawFromRecipe") + stage(win, "Graphics.Draw"),
           "gpu": sum(r.get("gpu_busy_ms_per_present", 0) * r["presents"] for r in win) / p, "us": (c / n if n else None), "freeze": freeze, "n_commit": n,
           "all": [r["presents"] for r in rows]}
    if name in times and loads:
        a, b = times[name]; s = [x for x in loads if a <= x[0] <= b]
        out["compil"] = (max(x[1] for x in s), sum(x[2] for x in s) / len(s)) if s else None
    return out

cols = ("imgs", "ms", "dpi", "dec", "fin", "com", "gpu", "us")
arms = {"a": [], "b": []}
print(f"{'session':10} {'img/s':>6} {'ms/img':>7} {'dess/img':>8} {'décod':>6} {'finish':>6} {'commit':>6} {'GPU':>5} {'µs/commit':>9}  compilations(max)/charge")
for i in range(1, turns + 1):
    for arm in "ab":
        name = f"{prefix}-{arm}{i}"; r = read(name)
        if r is None: print(f"{name:10} absent"); continue
        if "fail" in r: print(f"{name:10} ÉCHEC : {r['fail']}"); continue
        arms[arm].append(r)
        comp = r.get("compil"); cs = f"{comp[0]}/{comp[1]:.1f}" if comp else "?"
        print(f"{name:10} {r['imgs']:6.1f} {r['ms']:7.2f} {r['dpi']:8.0f} {r['dec']:6.2f} {r['finish'] if False else r['fin']:6.2f} {r['com']:6.2f} {r['gpu']:5.1f} {(r['us'] or 0):9.2f}  {cs}"
              + (f"  GEL {r['freeze']:.0f} ms" if r["freeze"] else ""))
mean = {}
for arm, rows in arms.items():
    if not rows: continue
    mean[arm] = {c: sum((r[c] or 0) for r in rows) / len(rows) for c in cols}
    m = mean[arm]
    print(f"{'moy ' + arm.upper() + ' (' + str(len(rows)) + ')':10} {m['imgs']:6.1f} {m['ms']:7.2f} {m['dpi']:8.0f} {m['dec']:6.2f} {m['fin']:6.2f} {m['com']:6.2f} {m['gpu']:5.1f} {m['us']:9.2f}")
if len(mean) == 2:
    a, b = mean["a"], mean["b"]
    print(f"{'B / A':10} " + " ".join(f"{(b[c] / a[c] - 1) * 100:+6.1f}%" if a[c] else "   -   " for c in cols))
    print("par dessin (B / A, la scène varie) : ms/image ÷ dessins/image " + f"{((b['ms'] / b['dpi']) / (a['ms'] / a['dpi']) - 1) * 100:+.1f} % ; décodage {((b['dec'] / b['dpi']) / (a['dec'] / a['dpi']) - 1) * 100:+.1f} % ; "
          f"committer {((b['com'] / b['dpi']) / (a['com'] / a['dpi']) - 1) * 100:+.1f} %")
