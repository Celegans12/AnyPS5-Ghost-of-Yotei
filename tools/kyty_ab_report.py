import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

DESCRIPTION = """Per-arm summary of an in-run A/B of the Kyty backend (KYTY_AB). The game prints "[kyty-ab] phase A|B <k>
begins at present <n> (t <s> s, wall <w>)" when a phase starts, "[kyty-ab] phase A|B <k> settled at present <n> (...)"
KYTY_AB_SETTLE seconds later, and "[kyty-ab] phase A|B <k> settled for <s> s: <n> presents, <f> img/s, ..." when the
phase ends, for the part after its settle. Only phases with that last line count, so the phase the session stopped in is
left out. A session folder of planb/runs/ also gives vkcount.log (img/s and draws per image of each window, placed in
time by start.json), threads.jsonl (the wait field: time the threads spent runnable but not running) and phases.jsonl
(the session's steps). A phase is kept when its settled part starts after the step --from, and is set aside as disturbed
when one threads.jsonl window inside it has a CPU-weighted wait above --wait-max percent. --skip seconds after each
settled line are also left out of the vkcount and threads windows. A periodic "[kyty-*] summary of the last <s> s" line
counts for phase k only when the previous line of the same kind came after phase k's settled line and the line came
before phase k+1 began, so it covers time inside the settled part only; the counters print the mean of each number of
such lines per arm, A|B. The drift-cancelled difference compares each kept B phase with the mean of the kept A phases
just before and after it."""

BEGIN = re.compile(r"\[kyty-ab\] phase ([AB]) (\d+) begins at present (\d+) \(t ([\d.]+) s, wall ([\d.]+)\): (.*)$")
SETTLED_AT = re.compile(r"\[kyty-ab\] phase ([AB]) (\d+) settled at present (\d+) \(t ([\d.]+) s, wall ([\d.]+)\)")
RESULT = re.compile(r"\[kyty-ab\] phase ([AB]) (\d+) settled for ([\d.]+) s: (\d+) presents, ([\d.]+) img/s, longest frame "
                    r"([\d.]+) ms, (\d+) frames of 50 ms or more; presents (\d+) to (\d+), wall ([\d.]+) to ([\d.]+)")
IN_RUN = re.compile(r"\[kyty-ab\] in-run A/B: phases of ([\d.]+) s from (.*), settle ([\d.]+) s; A: (.*); B: (.*)$")
SUMMARY = re.compile(r"\[(kyty-[a-z0-9-]+)\] (.*summary of the last [\d.]+ s.*)$")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
VK = re.compile(r"\[vkcount\] t=([\d.]+) s dt=([\d.]+) fps=([\d.]+) ms_per_frame=[\d.]+ draw_calls_per_frame=([\d.]+)")
MIN_THREAD_PCT = 10.0


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite A/B statistic")
    return value


def read_lines(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def parse_log(lines):
    phases, order, arms = {}, [], None
    counters = []
    current, settled, previous = None, False, {}
    for line in lines:
        match = IN_RUN.search(line)
        if match:
            arms = {"phase_seconds": float(match.group(1)), "from": match.group(2), "settle_seconds": float(match.group(3)),
                    "A": match.group(4), "B": match.group(5)}
            continue
        match = BEGIN.search(line)
        if match:
            index = int(match.group(2))
            phases[index] = {"phase": index, "arm": match.group(1), "begin_present": int(match.group(3)),
                             "begin_wall": float(match.group(5)), "settings": match.group(6)}
            order.append(index)
            current, settled = index, False
            continue
        match = SETTLED_AT.search(line)
        if match:
            index = int(match.group(2))
            if index in phases:
                phases[index]["settled_wall"] = float(match.group(5))
            current, settled = index, True
            continue
        match = RESULT.search(line)
        if match:
            index = int(match.group(2))
            if index in phases:
                phases[index].update({"settled_seconds": float(match.group(3)), "presents": int(match.group(4)),
                                      "fps": finite(float(match.group(5))), "longest_ms": float(match.group(6)),
                                      "long_frames": int(match.group(7)), "first_present": int(match.group(8)),
                                      "last_present": int(match.group(9)), "settled_wall": float(match.group(10)),
                                      "end_wall": float(match.group(11))})
            continue
        match = SUMMARY.search(line)
        if match:
            kind, text = match.group(1), match.group(2)
            if kind == "kyty-ab":
                continue
            state = (current, settled)
            if current is not None and settled and previous.get(kind) == state:
                counters.append({"phase": current, "kind": kind, "template": NUMBER.sub("{}", text),
                                 "values": [float(value) for value in NUMBER.findall(text)]})
            previous[kind] = state
    return arms, [phases[index] for index in order], counters


def load_vkcount(folder):
    path, start = folder / "vkcount.log", folder / "start.json"
    if not path.is_file() or not start.is_file():
        return None
    epoch = json.loads(start.read_text(encoding="utf-8")).get("vk_epoch")
    if epoch is None:
        return None
    windows = []
    for line in read_lines(path):
        match = VK.search(line)
        if match:
            t, dt, fps, dpf = (float(value) for value in match.groups())
            windows.append({"start": epoch + t - dt, "end": epoch + t, "dt": dt, "fps": fps, "dpf": dpf})
    return windows


def load_threads(folder):
    path = folder / "threads.jsonl"
    if not path.is_file():
        return None
    windows = []
    for line in read_lines(path):
        if not line.strip():
            continue
        row = json.loads(line)
        busy = [thread for thread in row.get("threads", []) if thread.get("pct", 0) >= MIN_THREAD_PCT]
        weight = sum(thread["pct"] for thread in busy)
        wait = sum(thread.get("wait", 0) * thread["pct"] for thread in busy) / weight if weight else 0.0
        windows.append({"start": row["wall"] - row.get("dt", 0), "end": row["wall"], "wait": finite(wait)})
    return windows


def load_steps(folder):
    path = folder / "phases.jsonl"
    if not path.is_file():
        return {}
    steps = {}
    for line in read_lines(path):
        if line.strip():
            row = json.loads(line)
            steps.setdefault(row["phase"], row["wall"])
    return steps


def inside(windows, start, end):
    return [window for window in windows if window["start"] >= start - 1e-6 and window["end"] <= end + 1e-6]


def mean(values):
    return statistics.fmean(values) if values else None


def spread(values):
    return statistics.stdev(values) if len(values) > 1 else None


def analyse(folder, log, start_step="measure", skip=0.0, wait_max=10.0):
    arms, phases, counters = parse_log(read_lines(log))
    vkcount = load_vkcount(folder) if folder else None
    threads = load_threads(folder) if folder else None
    steps = load_steps(folder) if folder else {}
    from_wall, from_step = None, "all"
    if start_step != "all":
        for name in ((start_step, "game") if start_step == "measure" else (start_step,)):
            if name in steps:
                from_wall, from_step = steps[name], name
                break
    for phase in phases:
        if "end_wall" not in phase:
            phase["kept"], phase["why"] = False, "no settled line (the session stopped in it)"
            continue
        start, end = phase["settled_wall"] + skip, phase["end_wall"]
        if from_wall is not None and phase["settled_wall"] < from_wall:
            phase["kept"], phase["why"] = False, f"before the {from_step} step"
            continue
        if vkcount is not None:
            frames = inside(vkcount, start, end)
            weight = sum(window["fps"] * window["dt"] for window in frames)
            phase["vk_fps"] = sum(window["fps"] * window["dt"] for window in frames) / sum(window["dt"] for window in frames) if frames else None
            phase["draws_per_image"] = sum(window["dpf"] * window["fps"] * window["dt"] for window in frames) / weight if weight else None
            phase["vk_windows"] = len(frames)
        if threads is not None:
            waits = [window["wait"] for window in inside(threads, start, end)]
            phase["wait_max"] = max(waits) if waits else None
            if waits and max(waits) > wait_max:
                phase["kept"], phase["why"] = False, f"disturbed (wait {max(waits):.1f} % > {wait_max:g} %)"
                continue
        phase["kept"], phase["why"] = True, ""
    kept = {phase["phase"]: phase for phase in phases if phase["kept"]}
    per_arm = {}
    for arm in "AB":
        chosen = [phase for phase in kept.values() if phase["arm"] == arm]
        fps = [phase["fps"] for phase in chosen]
        vk = [phase["vk_fps"] for phase in chosen if phase.get("vk_fps") is not None]
        draws = [phase["draws_per_image"] for phase in chosen if phase.get("draws_per_image") is not None]
        per_arm[arm] = {"phases": len(chosen), "fps": mean(fps), "fps_sd": spread(fps), "vk_fps": mean(vk), "vk_fps_sd": spread(vk),
                        "draws_per_image": mean(draws), "draws_sd": spread(draws),
                        "longest_ms": max((phase["longest_ms"] for phase in chosen), default=None),
                        "long_frames": sum(phase["long_frames"] for phase in chosen),
                        "settings": arms[arm] if arms else (chosen[0]["settings"] if chosen else None)}
    differences = []
    for index, phase in kept.items():
        if phase["arm"] != "B":
            continue
        neighbours = [kept[other]["fps"] for other in (index - 1, index + 1) if other in kept and kept[other]["arm"] == "A"]
        if len(neighbours) == 2:
            differences.append(phase["fps"] - statistics.fmean(neighbours))
    drift = None
    if differences:
        error = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else None
        drift = {"mean": statistics.fmean(differences), "standard_error": error, "count": len(differences)}
    groups = {}
    for record in counters:
        if record["phase"] not in kept:
            continue
        arm = kept[record["phase"]]["arm"]
        key = (record["kind"], record["template"])
        groups.setdefault(key, {"A": [], "B": []})[arm].append(record["values"])
    counter_means = []
    for (kind, template), by_arm in sorted(groups.items()):
        means = {arm: [statistics.fmean(column) for column in zip(*rows)] if rows else None for arm, rows in by_arm.items()}
        counter_means.append({"kind": kind, "template": template, "lines": {arm: len(rows) for arm, rows in by_arm.items()},
                              "A": means["A"], "B": means["B"]})
    return {"log": str(log), "arms": arms, "from_step": from_step, "from_wall": from_wall, "skip": skip, "wait_max": wait_max,
            "vkcount": vkcount is not None, "threads": threads is not None, "phases": phases, "per_arm": per_arm,
            "drift_cancelled": drift, "counters": counter_means}


def number_text(value):
    if value is None:
        return "-"
    if abs(value - round(value)) < 1e-9 and abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.3g}" if abs(value) < 10 else f"{value:.1f}"


def fill(template, a_values, b_values):
    parts = template.split("{}")
    text = parts[0]
    for index, part in enumerate(parts[1:]):
        a = number_text(a_values[index]) if a_values else "-"
        b = number_text(b_values[index]) if b_values else "-"
        text += (a if a == b else f"{a}|{b}") + part
    return text


def optional(value, digits=2, suffix=""):
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def render(result, out):
    arms = result["arms"]
    out.write(f"kyty_ab_report: {result['log']}\n")
    if arms:
        out.write(f"  A: {arms['A']}\n  B: {arms['B']}\n  phases of {arms['phase_seconds']:g} s from {arms['from']}, "
                  f"settle {arms['settle_seconds']:g} s\n")
    origin = "every phase" if result["from_wall"] is None else f"phases settled after the {result['from_step']} step"
    sources = [name for name, present in (("vkcount.log", result["vkcount"]), ("threads.jsonl", result["threads"])) if present]
    out.write(f"  kept: {origin}; extra skip {result['skip']:g} s; disturbed above {result['wait_max']:g} % wait; "
              f"windows from {', '.join(sources) if sources else 'the log only'}\n\n")
    out.write("  phase arm  settled s  img/s  vk img/s  draws/img  wait %  longest ms  >=50 ms  kept\n")
    for phase in result["phases"]:
        out.write(f"  {phase['phase']:>5} {phase['arm']:>3}  {optional(phase.get('settled_seconds'), 1):>9}  "
                  f"{optional(phase.get('fps')):>5}  {optional(phase.get('vk_fps')):>8}  {optional(phase.get('draws_per_image'), 0):>9}  "
                  f"{optional(phase.get('wait_max'), 1):>6}  {optional(phase.get('longest_ms'), 1):>10}  "
                  f"{phase.get('long_frames', '-'):>7}  {'yes' if phase['kept'] else 'no: ' + phase['why']}\n")
    out.write("\n")
    per_arm = result["per_arm"]
    for arm in "AB":
        data = per_arm[arm]
        out.write(f"  {arm}: {data['phases']} phase(s); {optional(data['fps'])} img/s (sd {optional(data['fps_sd'])}); "
                  f"vkcount {optional(data['vk_fps'])} img/s (sd {optional(data['vk_fps_sd'])}); "
                  f"{optional(data['draws_per_image'], 0)} draws per image (sd {optional(data['draws_sd'], 0)}); "
                  f"longest frame {optional(data['longest_ms'], 1)} ms, {data['long_frames']} frames of 50 ms or more\n")
    a, b = per_arm["A"], per_arm["B"]
    if a["fps"] is not None and b["fps"] is not None:
        out.write(f"  B - A: {b['fps'] - a['fps']:+.2f} img/s ({(b['fps'] / a['fps'] - 1) * 100:+.1f} %)\n")
    drift = result["drift_cancelled"]
    if drift:
        out.write(f"  drift-cancelled B - A over {drift['count']} B phase(s): {drift['mean']:+.2f} img/s"
                  f" (standard error {optional(drift['standard_error'])})\n")
    if a["draws_per_image"] and b["draws_per_image"]:
        ratio = b["draws_per_image"] / a["draws_per_image"]
        verdict = "same scene" if abs(ratio - 1) <= 0.10 else "scenes differ by more than 10 %: the arms do not compare"
        out.write(f"  scene: draws per image B/A = {ratio:.3f} ({verdict})\n")
    if result["counters"]:
        out.write("\n  counters: mean of each [kyty-*] summary wholly inside the kept settled parts, A|B where they differ\n")
        for counter in result["counters"]:
            lines = counter["lines"]
            out.write(f"  [{counter['kind']}] ({lines['A']}|{lines['B']} lines) {fill(counter['template'], counter['A'], counter['B'])}\n")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("session", help="session folder (planb/runs/<name>, its stderr.log is read) or a log file")
    parser.add_argument("--log", help="log file inside the session folder (default stderr.log)")
    parser.add_argument("--from", dest="start_step", default="measure",
                        help="session step after which a settled part must start: measure (falls back to game), game, or all")
    parser.add_argument("--skip", type=float, default=0.0, help="extra seconds after each settled line left out of the windows")
    parser.add_argument("--wait-max", type=float, default=10.0, help="CPU-weighted wait percent above which a window is disturbed")
    parser.add_argument("--json", help="also write the result to this JSON file")
    args = parser.parse_args(argv)
    path = Path(args.session)
    folder, log = (path, path / (args.log or "stderr.log")) if path.is_dir() else (path.parent, path)
    if not log.is_file():
        print(f"error: {log} does not exist", file=sys.stderr)
        return 1
    try:
        result = analyse(folder, log, args.start_step, args.skip, args.wait_max)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not result["phases"]:
        print(f"error: {log} has no [kyty-ab] phase: was KYTY_AB set?", file=sys.stderr)
        return 2
    render(result, sys.stdout)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0 if result["per_arm"]["A"]["phases"] and result["per_arm"]["B"]["phases"] else 2


if __name__ == "__main__":
    sys.exit(main())
