import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import sys

from benchmark_compare import MAX_FILE_BYTES, UINT64_MAX, benchmark_integer, reject_benchmark_constant
from performance_compare import unique_json_object


def read_videoout_flip_data(path):
    if not path.is_file():
        raise ValueError(f"{path}: input must be a regular file")
    with path.open("rb") as source:
        data = source.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{path}: input exceeds {MAX_FILE_BYTES} bytes")
    return data


def videoout_flip_records(path, *, data=None):
    if data is None:
        data = read_videoout_flip_data(path)
    if data and not data.endswith(b"\n"):
        raise ValueError(f"{path}: incomplete final JSONL record")
    for number, raw in enumerate(io.BytesIO(data), 1):
        location = f"{path}:{number}"
        try:
            record = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_json_object,
                                parse_constant=reject_benchmark_constant)
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{location}: {error}") from error
        if (not isinstance(record, dict) or type(record.get("v")) is not int
                or record["v"] != 1 or not isinstance(record.get("ev"), str)):
            raise ValueError(f"{location}: expected a version 1 VideoOut event")
        yield location, record


def read_videoout_flip_samples(path, *, data=None):
    records = videoout_flip_records(path, data=data)
    location, contract = next(records, (str(path), {}))
    if (contract.get("ev") != "contract" or contract.get("contract") != "v1-pad"
            or contract.get("prx") != "libSceVideoOut"):
        raise ValueError(f"{location}: expected the VideoOut contract as the first record")
    commit = contract.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{12,40}", commit):
        raise ValueError(f"{location}: expected a known build revision")
    first = benchmark_integer(contract.get("flip_start"), f"{location} flip_start")
    count = benchmark_integer(contract.get("flip_count"), f"{location} flip_count", 100000)
    if first < 1 or count < 2 or first + count - 1 > UINT64_MAX:
        raise ValueError(f"{location}: expected a valid completed-flip window")

    generation = None
    samples = []
    drops = 0
    for location, record in records:
        event = record["ev"]
        if event == "contract":
            raise ValueError(f"{location}: multiple telemetry sessions; use one new directory per run")
        if event == "drops":
            drops = max(drops, benchmark_integer(record.get("dropped"), f"{location} dropped"))
        if event != "flip_complete":
            continue
        current = benchmark_integer(record.get("generation"), f"{location} generation")
        if current == 0 or (generation is not None and current != generation):
            raise ValueError(f"{location}: expected one nonzero output generation")
        generation = current
        sequence = benchmark_integer(record.get("n"), f"{location} n")
        expected = first + len(samples)
        if len(samples) == count:
            raise ValueError(f"{location}: extra flip {sequence} outside the declared window")
        if sequence != expected:
            raise ValueError(f"{location}: expected flip {expected}, got {sequence}")
        timestamp = benchmark_integer(record.get("process_us"), f"{location} process_us")
        if samples and timestamp <= samples[-1][0]:
            raise ValueError(f"{location}: process_us must increase between completed flips")
        samples.append((timestamp, record.get("t_ns")))

    if len(samples) != count:
        raise ValueError(f"{path}: incomplete window; expected {count} flips, found {len(samples)}")
    return {
        "commit": commit,
        "first_flip": first,
        "flip_count": count,
        "generation": generation,
        "observed_telemetry_drops": drops,
    }, samples


def read_videoout_flip_window(path, *, data=None):
    window, samples = read_videoout_flip_samples(path, data=data)
    drops = window.pop("observed_telemetry_drops")
    timestamps = [timestamp for timestamp, _ in samples]
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    duration = timestamps[-1] - timestamps[0]
    return {
        **window,
        "elapsed_us": duration,
        "mean_fps": len(intervals) * 1_000_000 / duration,
        "p50_ms": statistics.median(intervals) / 1000,
        "p99_ms": sorted(intervals)[(99 * len(intervals) - 1) // 100] / 1000,
        "observed_telemetry_drops": drops,
    }


def compare_videoout_flip_windows(anyps5_path, kyty_path):
    if anyps5_path.samefile(kyty_path):
        raise ValueError("supply separate AnyPS5 and Kyty telemetry files")
    anyps5 = read_videoout_flip_window(anyps5_path)
    kyty = read_videoout_flip_window(kyty_path)
    if anyps5["commit"] != kyty["commit"]:
        raise ValueError("both telemetry files must report the same build revision")
    if anyps5["flip_count"] != kyty["flip_count"]:
        raise ValueError("both telemetry windows must declare the same flip count")
    return {"format": "aps5-videoout-flip-comparison/1", "anyps5": anyps5, "kyty": kyty}


def read_pinned_videoout_flip_window(path, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{path}: expected a full 40-digit lowercase hexadecimal revision")
    data = read_videoout_flip_data(path)
    window = read_videoout_flip_window(path, data=data)
    if not revision.startswith(window["commit"]):
        raise ValueError(f"{path}: reported revision {window['commit']} does not match {revision}")
    return {**window, "expected_revision": revision, "sha256": hashlib.sha256(data).hexdigest()}


def compare_videoout_build_windows(baseline_path, candidate_path, baseline_revision, candidate_revision):
    if baseline_path.samefile(candidate_path):
        raise ValueError("supply separate baseline and candidate telemetry files")
    baseline = read_pinned_videoout_flip_window(baseline_path, baseline_revision)
    candidate = read_pinned_videoout_flip_window(candidate_path, candidate_revision)
    if baseline["sha256"] == candidate["sha256"]:
        raise ValueError("baseline and candidate telemetry are byte-identical")
    fields = ("first_flip", "flip_count")
    if any(baseline[field] != candidate[field] for field in fields):
        raise ValueError("baseline and candidate must declare the same first flip and flip count")
    if baseline_revision != candidate_revision and any(
        baseline_revision.startswith(run["commit"]) and candidate_revision.startswith(run["commit"])
        for run in (baseline, candidate)
    ):
        raise ValueError("a reported revision prefix matches both expected revisions")
    metrics = ("mean_fps", "p50_ms", "p99_ms")
    return {
        "format": "aps5-videoout-build-comparison/1",
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": {metric: candidate[metric] - baseline[metric] for metric in metrics},
    }


def build_comparison_markdown(report):
    lines = ["| Arm | Revision | First flip | Flips | Mean FPS | p50 ms | p99 ms | Telemetry drops observed |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ("baseline", "candidate"):
        run = report[arm]
        lines.append(f"| {arm.capitalize()} | `{run['expected_revision']}` | {run['first_flip']} | "
                     f"{run['flip_count']} | {run['mean_fps']:.2f} | {run['p50_ms']:.3f} | "
                     f"{run['p99_ms']:.3f} | {run['observed_telemetry_drops']} |")
    delta = report["candidate_minus_baseline"]
    lines.append(f"| Candidate minus baseline | | | | {delta['mean_fps']:+.2f} | "
                 f"{delta['p50_ms']:+.3f} | {delta['p99_ms']:+.3f} | |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare completed VideoOut flip windows.")
    parser.add_argument("anyps5", type=Path, help="default-backend or pinned baseline videoout.jsonl")
    parser.add_argument("kyty", type=Path, help="Kyty-backend or pinned candidate videoout.jsonl")
    parser.add_argument("--baseline-revision", help="full Git revision expected for the first input")
    parser.add_argument("--candidate-revision", help="full Git revision expected for the second input")
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args(argv)
    if (args.baseline_revision is None) != (args.candidate_revision is None):
        parser.error("--baseline-revision and --candidate-revision must be used together")
    build_mode = args.baseline_revision is not None
    try:
        if build_mode:
            report = compare_videoout_build_windows(args.anyps5, args.kyty, args.baseline_revision,
                                                   args.candidate_revision)
        else:
            report = compare_videoout_flip_windows(args.anyps5, args.kyty)
    except (OSError, ValueError) as error:
        print(f"videoout_flip_compare: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    elif build_mode:
        print(build_comparison_markdown(report))
    else:
        print(f"Build revision: {report['anyps5']['commit']}")
        print()
        print("| Backend | First flip | Flips | Mean FPS | p50 ms | p99 ms | Telemetry drops observed |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for label, backend in (("AnyPS5", "anyps5"), ("Kyty", "kyty")):
            run = report[backend]
            print(f"| {label} | {run['first_flip']} | {run['flip_count']} | {run['mean_fps']:.2f} | "
                  f"{run['p50_ms']:.3f} | {run['p99_ms']:.3f} | {run['observed_telemetry_drops']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
