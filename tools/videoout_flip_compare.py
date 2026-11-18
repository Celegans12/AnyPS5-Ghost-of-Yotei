import argparse
import io
import json
from pathlib import Path
import re
import statistics
import sys

from benchmark_compare import MAX_FILE_BYTES, UINT64_MAX, benchmark_integer, reject_benchmark_constant
from performance_compare import unique_json_object


def videoout_flip_records(path):
    if not path.is_file():
        raise ValueError(f"{path}: input must be a regular file")
    with path.open("rb") as source:
        data = source.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{path}: input exceeds {MAX_FILE_BYTES} bytes")
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


def read_videoout_flip_window(path):
    records = videoout_flip_records(path)
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
    timestamps = []
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
        expected = first + len(timestamps)
        if len(timestamps) == count:
            raise ValueError(f"{location}: extra flip {sequence} outside the declared window")
        if sequence != expected:
            raise ValueError(f"{location}: expected flip {expected}, got {sequence}")
        timestamp = benchmark_integer(record.get("process_us"), f"{location} process_us")
        if timestamps and timestamp <= timestamps[-1]:
            raise ValueError(f"{location}: process_us must increase between completed flips")
        timestamps.append(timestamp)

    if len(timestamps) != count:
        raise ValueError(f"{path}: incomplete window; expected {count} flips, found {len(timestamps)}")
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    duration = timestamps[-1] - timestamps[0]
    return {
        "commit": commit,
        "first_flip": first,
        "flip_count": count,
        "generation": generation,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare completed VideoOut flip windows from the same AnyKyty revision.")
    parser.add_argument("anyps5", type=Path, help="videoout.jsonl from the default AnyPS5 backend")
    parser.add_argument("kyty", type=Path, help="videoout.jsonl from the Kyty backend")
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args(argv)
    try:
        report = compare_videoout_flip_windows(args.anyps5, args.kyty)
    except (OSError, ValueError) as error:
        print(f"videoout_flip_compare: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
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
