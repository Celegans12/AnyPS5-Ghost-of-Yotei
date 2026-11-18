import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import sys

from benchmark_compare import MAX_FILE_BYTES, benchmark_integer
from videoout_flip_compare import read_videoout_flip_samples


MAX_GAPS = 100
MAX_TICK = (1 << 63) - 1
CLOCK_NS = 1_000_000_000
SHADER = r"0x(?P<shader>[0-9a-fA-F]{16})"
END = r" @(?P<end>[0-9]{1,19})"
PIPELINE_ROWS = {
    "stall": re.compile(r"(?P<duration>[0-9]{1,10}\.[0-9]{3}) s (?P<stall_type>\S+) "
                        + SHADER + r"(?: (?P<route>.*?))?" + END),
    "translate": re.compile(r"(?P<duration>[0-9]{1,10}\.[0-9]{3}) s " + SHADER + END),
    "compute-driver": re.compile(r"(?P<duration>[0-9]{1,10}\.[0-9]{6}) s " + SHADER
                                 + r" module=(?P<module>[0-9]{1,20}) spirv_words=(?P<spirv_words>[0-9]{1,10})"
                                 + r" mode=(?P<mode>lookup|create) result=(?P<result>\S(?:.*\S)?)" + END),
}
LABELS = {"stall": "Pipeline wait", "translate": "Translation",
          "compute-create": "Compute creation", "compute-lookup": "Compute lookup"}
NOTES = [
    "Supply both inputs from one run of a build that uses the nanosecond steady clock for both logs.",
    "The flip gap length uses process_us and the overlap uses the nearby telemetry emission timestamp t_ns.",
    "Recorded overlaps show timing evidence. Background work and execute waits can overlap.",
    "The stall threshold and unfinished calls limit coverage. Wait and translation durations are rounded to milliseconds.",
]


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    kind: str
    shader: str
    end_ns: int
    duration_ns: int
    details: dict


def read_kyty_pipeline_events(path):
    if not path.is_file():
        raise ValueError(f"{path}: input must be a regular file")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{path}: input exceeds {MAX_FILE_BYTES} bytes")
    if data and not data.endswith(b"\n"):
        raise ValueError(f"{path}: incomplete final log line")

    events = []
    session_headers = 0
    for number, raw in enumerate(io.BytesIO(data), 1):
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        location = f"{path}:{number}"
        if line.startswith(("--- Session ", "--- Relaunched ")):
            session_headers += 1
            if session_headers > 1:
                raise ValueError(f"{location}: multiple Kyty sessions; use a new log for the measured run")
        _, _, row = line.partition("[kyty-gpl] ")
        kind, _, payload = row.partition(" ")
        if kind not in PIPELINE_ROWS:
            continue
        match = PIPELINE_ROWS[kind].fullmatch(payload) if len(payload) <= 4096 else None
        if match is None:
            raise ValueError(f"{location}: malformed Kyty {kind} record")
        fields = match.groupdict()
        end = benchmark_integer(int(fields.pop("end")), f"{location} pipeline timestamp", MAX_TICK)
        whole, fraction = fields.pop("duration").split(".")
        duration = int(whole) * CLOCK_NS + int(fraction) * 10 ** (9 - len(fraction))
        if duration > end:
            raise ValueError(f"{location}: pipeline duration exceeds its end timestamp")
        shader = fields.pop("shader").lower()

        if kind == "stall":
            fields["route"] = fields["route"] or ""
        elif kind == "compute-driver":
            fields["module"] = benchmark_integer(int(fields["module"]), f"{location} module")
            fields["spirv_words"] = benchmark_integer(int(fields["spirv_words"]), f"{location} spirv_words", (1 << 32) - 1)
            kind = f"compute-{fields['mode']}"
        events.append(PipelineEvent(kind, shader, end, duration, fields))
    return events


def pipeline_overlap(event, start_ns, end_ns):
    if event.duration_ns == 0:
        return 0 if start_ns < event.end_ns <= end_ns else None
    overlap = min(end_ns, event.end_ns) - max(start_ns, event.end_ns - event.duration_ns)
    return overlap if overlap > 0 else None


def correlate_kyty_pipeline_stalls(videoout_path, session_path, limit=10):
    benchmark_integer(limit, "top gap count", MAX_GAPS)
    if limit == 0:
        raise ValueError("top gap count must be positive")
    if videoout_path.samefile(session_path):
        raise ValueError("supply separate VideoOut and Kyty log files from the same run")

    window, samples = read_videoout_flip_samples(videoout_path)
    first = window["first_flip"]
    for offset, (_, timestamp) in enumerate(samples):
        benchmark_integer(timestamp, f"{videoout_path} t_ns for flip {first + offset}", MAX_TICK)
        if offset and timestamp <= samples[offset - 1][1]:
            raise ValueError(f"{videoout_path}: t_ns must increase between completed flips")

    events = read_kyty_pipeline_events(session_path)
    in_window = [event for event in events if pipeline_overlap(event, samples[0][1], samples[-1][1]) is not None]
    gap_lengths = (right[0] - left[0] for left, right in zip(samples, samples[1:]))
    gaps = sorted(enumerate(gap_lengths), key=lambda gap: gap[1], reverse=True)[:limit]

    report_gaps = []
    for offset, duration_us in gaps:
        start_ns, end_ns = samples[offset][1], samples[offset + 1][1]
        overlaps = {}
        for event in in_window:
            overlap = pipeline_overlap(event, start_ns, end_ns)
            if overlap is None:
                continue
            count, largest, largest_overlap = overlaps.get(event.kind, (0, event, -1))
            if overlap > largest_overlap:
                largest, largest_overlap = event, overlap
            overlaps[event.kind] = (count + 1, largest, largest_overlap)

        report_gaps.append({
            "from_flip": first + offset,
            "to_flip": first + offset + 1,
            "frame_ms": duration_us / 1000,
            "overlaps": {
                kind: {"count": count, "largest": {
                    "shader": event.shader, "duration_ms": event.duration_ns / 1_000_000,
                    "overlap_ms": overlap / 1_000_000, **event.details,
                }}
                for kind, (count, event, overlap) in overlaps.items()
            },
        })

    return {
        "format": "aps5-kyty-pipeline-stalls/1",
        "window": window,
        "pipeline_events": len(events),
        "pipeline_events_in_window": len(in_window),
        "gaps": report_gaps,
        "notes": NOTES,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Correlate Kyty pipeline events with completed flips from the same run.")
    parser.add_argument("videoout", type=Path, help="videoout.jsonl from the measured Kyty run")
    parser.add_argument("session_log", type=Path, help="KYTY_SESSION_LOG text from the same run")
    parser.add_argument("--top", type=int, default=10, help="number of longest flip gaps to report (1-100)")
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args(argv)
    try:
        report = correlate_kyty_pipeline_stalls(args.videoout, args.session_log, limit=args.top)
    except (OSError, ValueError) as error:
        print(f"kyty_pipeline_stalls: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0

    window = report["window"]
    print(f"Build {window['commit']}, {window['flip_count']} flips from {window['first_flip']}, "
          f"{report['pipeline_events_in_window']}/{report['pipeline_events']} recorded pipeline events in the window.")
    print()
    print("| Flips | Gap ms | " + " | ".join(LABELS.values()) + " |")
    print("| --- | ---: | --- | --- | --- | --- |")
    for gap in report["gaps"]:
        cells = [f"{gap['from_flip']}-{gap['to_flip']}", f"{gap['frame_ms']:.3f}"]
        for kind in LABELS:
            group = gap["overlaps"].get(kind)
            if group is None:
                cells.append("-")
                continue
            event = group["largest"]
            cells.append(f"{group['count']} / 0x{event['shader']} / {event['overlap_ms']:.3f} ms")
        print("| " + " | ".join(cells) + " |")
    print()
    for note in report["notes"]:
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
