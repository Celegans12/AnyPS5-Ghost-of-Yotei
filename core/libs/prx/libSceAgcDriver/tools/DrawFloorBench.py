import argparse
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
ROOT = os.path.dirname(TREE)
PYTHON = os.path.join(ROOT, "toolchain", "python", "python.exe")
TOOLCHAIN = os.path.join(ROOT, "toolchain", "mingw64", "bin")
OUT = os.path.join(TREE, "build", "draw-floor")
LAYER = "VK_LAYER_APS5_draw_floor"
RUNS = os.path.join(ROOT, "bench-seeds", "runs")
MANIFEST = {
    "file_format_version": "1.2.0",
    "layer": {
        "name": LAYER,
        "type": "GLOBAL",
        "library_path": ".\\DrawFloorLayer.dll",
        "api_version": "1.3.0",
        "implementation_version": "1",
        "description": "AnyPS5 draw floor: re-records the submitted draws with cached objects and times them",
    },
}

FLOOR_KEYS = {
    "raw": r"\| raw ([\d.]+) min ([\d.]+)",
    "strict": r"\| strict ([\d.]+) record ([\d.]+) min ([\d.]+)",
    "state": r"\| state ([\d.]+) record ([\d.]+) min ([\d.]+)",
    "redone": r"\| redone ([\d.]+) min ([\d.]+)",
    "kept": r"\| kept ([\d.]+) min ([\d.]+)",
    "shared": r"\| shared ([\d.]+) min ([\d.]+)",
    "overhead": r"buffer overhead ([\d.]+) us",
}


def build(args):
    os.makedirs(OUT, exist_ok=True)
    compiler = os.path.join(TOOLCHAIN, "g++.exe")
    command = [compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wno-missing-field-initializers", "-shared", "-static", "-Wl,--exclude-libs,ALL", "-I", os.path.join(TREE, "3rdparty", "Vulkan-Headers", "include"),
               os.path.join(HERE, "DrawFloorLayer.cpp"), "-o", os.path.join(OUT, "DrawFloorLayer.dll")]
    env = dict(os.environ)
    env["PATH"] = TOOLCHAIN + os.pathsep + env.get("PATH", "")
    slot = [PYTHON, os.path.join(ROOT, "AnyPS5", "machine_slot.py"), "run", "--kind", "build", "--owner", args.owner, "--"]
    code = subprocess.run(slot + command, cwd=ROOT, env=env).returncode
    if code != 0:
        return code
    with open(os.path.join(OUT, "DrawFloorLayer.json"), "w", encoding="utf-8") as manifest:
        json.dump(MANIFEST, manifest, indent=4)
    print(f"couche : {os.path.join(OUT, 'DrawFloorLayer.dll')}")
    return 0


def run(args):
    command = [PYTHON, os.path.join(ROOT, "AnyPS5", "draw_bench.py"), "--build", os.path.join(TREE, "build"), "--seed", os.path.join(ROOT, "bench-seeds", args.seed),
               "--draws", str(args.draws), "--label", args.label, "--verify", "0", "--seconds", str(args.seconds), "--mode", args.mode]
    if args.half is not None:
        command += ["--half", str(args.half)]
    if not args.baseline:
        command += ["--env", f"VK_ADD_LAYER_PATH={OUT}", "--env", f"VK_INSTANCE_LAYERS={LAYER}", "--env", f"APS5_MEASURE_DRAW_FLOOR={args.every}"]
    for pair in args.env:
        command += ["--env", pair]
    code = subprocess.run(command, cwd=ROOT).returncode
    summarize(args.label)
    return code


def parse_floor(text):
    periods = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("[draw-floor] t="):
            continue
        head = re.search(r"t=([\d.]+) period=([\d.]+) submits=(\d+) buffers=(\d+) replayed=(\d+) draws=(\d+) passes=(\d+)>(\d+)", line)
        if head is None:
            continue
        entry = {"t": float(head.group(1)), "period": float(head.group(2)), "submits": int(head.group(3)), "buffers": int(head.group(4)),
                 "replayed": int(head.group(5)), "draws": int(head.group(6)), "passes_raw": int(head.group(7)), "passes_kept": int(head.group(8))}
        for key, pattern in FLOOR_KEYS.items():
            found = re.search(pattern, line)
            if found:
                entry[key] = [float(value) for value in found.groups()]
        tail = re.search(r"per draw pipeline ([\d.]+)>([\d.]+) sets ([\d.]+)>([\d.]+) push ([\d.]+) vertex ([\d.]+)>([\d.]+) index ([\d.]+)>([\d.]+) dynamic ([\d.]+)>([\d.]+) executes (\d+)", line)
        if tail:
            entry["per_draw"] = [float(value) for value in tail.groups()]
        for follow in lines[index + 1:index + 5]:
            situ = re.search(r"\[draw-floor-insitu\] thread=(\S+) draws=(\d+) \| all ([\d.]+) draw ([\d.]+) dynamic ([\d.]+) other ([\d.]+) sets ([\d.]+) buffers ([\d.]+) submit ([\d.]+) objects ([\d.]+) poll ([\d.]+) wait ([\d.]+) us/draw \| submits (\d+) ([\d.]+) us each", follow)
            if situ:
                values = situ.groups()
                entry["insitu_" + values[0]] = {"draws": int(values[1]), "all": float(values[2]), "draw": float(values[3]), "dynamic": float(values[4]), "other": float(values[5]),
                                                 "sets": float(values[6]), "buffers": float(values[7]), "submit": float(values[8]), "objects": float(values[9]),
                                                 "poll": float(values[10]), "wait": float(values[11]), "submits": int(values[12]), "submit_us": float(values[13])}
            slots = re.match(r"\[draw-floor-slots\] thread=(\S+)(.*)", follow)
            if slots:
                entry["slots_" + slots.group(1)] = slots.group(2).strip()
        periods.append(entry)
    return periods


def drawn(entry):
    return entry["draws"] > 0 or (entry.get("insitu_draw-commit") or {}).get("draws", 0) > 0


def steady(periods):
    usable = [entry for entry in periods if entry["period"] >= 4.0 and drawn(entry)]
    if len(usable) > 2:
        usable = usable[1:]
    return usable


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def summarize(label):
    work = os.path.join(RUNS, label)
    result_path = os.path.join(work, "result.json")
    result = json.load(open(result_path, encoding="utf-8")) if os.path.exists(result_path) else {}
    text = open(os.path.join(work, "bench.err"), encoding="utf-8", errors="replace").read() if os.path.exists(os.path.join(work, "bench.err")) else ""
    periods = parse_floor(text)
    use = steady(periods)
    summary = {"label": label, "seed": result.get("seed"), "env": result.get("env"), "fps": result.get("fps"), "draws_per_frame": (result.get("per_frame") or {}).get("draws"),
               "submits_per_frame": (result.get("per_frame") or {}).get("submits"), "committer_work_us_per_draw": (result.get("work_us_per_draw") or {}).get("committer"),
               "decode_work_us_per_draw": (result.get("work_us_per_draw") or {}).get("decode"), "committer_us_per_draw_commit": result.get("committer_us_per_draw_commit"),
               "driver_prx_mtime": result.get("driver_prx_mtime"), "periods": len(periods), "steady_periods": len(use)}
    replays = [entry for entry in use if entry["draws"] > 0]
    if replays:
        for key in FLOOR_KEYS:
            if key in replays[0]:
                width = len(replays[0][key])
                summary[key] = [median([entry[key][i] for entry in replays if key in entry]) for i in range(width)]
        summary["vulkan_draws_per_period"] = median([entry["draws"] for entry in replays])
        summary["replayed_share"] = median([entry["replayed"] / entry["buffers"] for entry in replays if entry["buffers"]])
        summary["draws_per_buffer"] = median([entry["draws"] / entry["replayed"] for entry in replays if entry["replayed"]])
        summary["passes_per_draw"] = median([entry["passes_raw"] / entry["draws"] for entry in replays])
        summary["merged_passes_per_draw"] = median([entry["passes_kept"] / entry["draws"] for entry in replays])
        if "per_draw" in replays[0]:
            names = ["pipeline_raw", "pipeline_kept", "sets_raw", "sets_kept", "push", "vertex_raw", "vertex_kept", "index_raw", "index_kept", "dynamic_raw", "dynamic_kept", "executes"]
            summary["commands_per_draw"] = {name: median([entry["per_draw"][i] for entry in replays if "per_draw" in entry]) for i, name in enumerate(names)}
    if use:
        summary["submits_per_period"] = median([entry["submits"] for entry in use])
        for thread in ("draw-commit", "others"):
            key = "insitu_" + thread
            rows = [entry[key] for entry in use if key in entry]
            if rows:
                summary[key] = {name: median([row[name] for row in rows]) for name in rows[0]}
        slots = [entry.get("slots_draw-commit") for entry in use if entry.get("slots_draw-commit")]
        if slots:
            summary["slots_draw_commit_last"] = slots[-1]
    with open(os.path.join(work, "draw-floor.json"), "w", encoding="utf-8") as out:
        json.dump(summary, out, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def report(args):
    for label in args.labels:
        summarize(label)
    return 0


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("build")
    making.add_argument("--owner", default="plancher")
    running = commands.add_parser("run")
    running.add_argument("--seed", required=True)
    running.add_argument("--draws", type=int, default=3000)
    running.add_argument("--label", required=True)
    running.add_argument("--seconds", type=float, default=20.0)
    running.add_argument("--mode", choices=["repeat", "varied"], default="varied")
    running.add_argument("--half", type=int, choices=[0, 1], default=0)
    running.add_argument("--every", type=int, default=1)
    running.add_argument("--baseline", action="store_true")
    running.add_argument("--env", action="append", default=[])
    reading = commands.add_parser("report")
    reading.add_argument("labels", nargs="+")
    args = parser.parse_args()
    if args.command == "build":
        return build(args)
    if args.command == "run":
        return run(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
