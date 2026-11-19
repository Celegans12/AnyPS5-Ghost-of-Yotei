import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import videoout_flip_compare as tool


BASELINE_REVISION = "0123456789ab" + "0" * 28
CANDIDATE_REVISION = "fedcba987654" + "1" * 28


def event(kind, **fields):
    return {"v": 1, "t_ns": 100, "ev": kind, **fields}


def run_records(timestamps, first=31, generation=4, pid=10, revision="0123456789ab"):
    contract = event("contract", contract="v1-pad", prx="libSceVideoOut", commit=revision,
                     pid=pid, flip_start=first, flip_count=len(timestamps))
    return [contract, *(event("flip_complete", n=first + index, generation=generation, process_us=timestamp)
                        for index, timestamp in enumerate(timestamps))]


class VideoOutFlipCompareTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, name, records):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        return path

    def test_pair_uses_adjacent_process_times_and_accepts_separate_run_origins(self):
        anyps5_rows = run_records([1_000, 11_000, 41_000])
        anyps5_rows.insert(2, event("present", n=60, fps=999))
        anyps5_rows.insert(3, event("pad", n=60, buttons=0))
        origin = (1 << 64) - 10_001
        anyps5 = self.write("anyps5.jsonl", anyps5_rows)
        kyty = self.write("kyty.jsonl", run_records([origin, origin + 4000, origin + 10000],
                                                   first=700, generation=8, pid=11))
        report = tool.compare_videoout_flip_windows(anyps5, kyty)
        self.assertEqual(report["anyps5"]["mean_fps"], 50.0)
        self.assertEqual(report["anyps5"]["p50_ms"], 20.0)
        self.assertEqual(report["anyps5"]["p99_ms"], 30.0)
        self.assertEqual(report["kyty"]["mean_fps"], 200.0)
        self.assertEqual(report["kyty"]["p50_ms"], 5.0)
        self.assertEqual(report["kyty"]["p99_ms"], 6.0)
        self.assertEqual(report["kyty"]["first_flip"], 700)
        self.assertEqual(report["kyty"]["generation"], 8)

    def test_p99_uses_nearest_rank(self):
        records = run_records([index * (index + 1) * 500 for index in range(101)])
        run = tool.read_videoout_flip_window(self.write("capture.jsonl", records))
        self.assertEqual(run["p99_ms"], 99)
        self.assertEqual(run["p50_ms"], 50.5)

    def test_two_flips_at_uint64_boundary(self):
        maximum = (1 << 64) - 1
        records = run_records([maximum - 1000, maximum], first=maximum - 1)
        run = tool.read_videoout_flip_window(self.write("capture.jsonl", records))
        self.assertEqual(run["mean_fps"], 1000)
        self.assertEqual(run["p99_ms"], 1)

    def test_telemetry_drops_are_visible_when_the_flip_window_is_complete(self):
        records = run_records([100, 200])
        records.extend([event("drops", dropped=3), event("drops", dropped=3)])
        run = tool.read_videoout_flip_window(self.write("capture.jsonl", records))
        self.assertEqual(run["observed_telemetry_drops"], 3)

    def test_incomplete_or_invalid_sequences_are_rejected(self):
        records = run_records([10, 20, 30])
        cases = (
            ("missing", records[:-1], "incomplete window"),
            ("gap", [*records[:2], records[3]], "expected flip"),
            ("duplicate", [*records[:2], records[1], records[3]], "expected flip"),
            ("reversed", [records[0], records[2], records[1], records[3]], "expected flip"),
            ("extra", [*records, records[3]], "outside the declared window"),
            ("generation", [*records[:3], {**records[3], "generation": 5}], "one nonzero output generation"),
            ("duplicate-time", [*records[:3], {**records[3], "process_us": 20}], "process_us must increase"),
            ("time-reversal", [*records[:3], {**records[3], "process_us": 19}], "process_us must increase"),
        )
        for name, rows, pattern in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, pattern):
                tool.read_videoout_flip_window(self.write("capture.jsonl", rows))

    def test_requires_one_initial_contract_with_supported_revision_and_bounds(self):
        records = run_records([0, 1])
        cases = (
            ("empty", [], "first record"),
            ("missing-contract", records[1:], "first record"),
            ("concatenated", records + records, "multiple telemetry sessions"),
            ("unknown-revision", [{**records[0], "commit": "unknown"}, *records[1:]], "known build revision"),
            ("wrong-contract", [{**records[0], "contract": "other"}, *records[1:]], "first record"),
            ("no-window", [event("contract", contract="v1-pad", prx="libSceVideoOut", commit="0123456789ab")],
             "flip_start must be an integer"),
            ("overflow", [{**records[0], "flip_start": (1 << 64) - 1}, *records[1:]], "valid completed-flip window"),
            ("too-many", [{**records[0], "flip_count": 100001}, *records[1:]], "flip_count must be an integer"),
            ("too-few", [{**records[0], "flip_count": 1}, *records[1:]], "valid completed-flip window"),
        )
        for name, rows, pattern in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, pattern):
                tool.read_videoout_flip_window(self.write("capture.jsonl", rows))

    def test_requires_integer_flip_counters_and_version(self):
        records = run_records([100, 200])
        for key, value, pattern in (
            ("n", 31.0, "n must be an integer"),
            ("generation", True, "generation must be an integer"),
            ("generation", 0, "one nonzero output generation"),
            ("process_us", -1, "process_us must be an integer"),
            ("v", True, "version 1 VideoOut event"),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, pattern):
                tool.read_videoout_flip_window(self.write("capture.jsonl", [records[0],
                                                                                 {**records[1], key: value},
                                                                                 records[2]]))

    def test_rejects_bad_trailing_records_after_the_complete_window(self):
        records = run_records([100, 200])
        for raw, pattern in (
            (b'{"v":1', "incomplete final JSONL record"),
            (b'{"v":1,"v":1,"ev":"present"}\n', "duplicate JSON field"),
            (b'{"v":1,"ev":"present","fps":NaN}\n', "non-finite JSON constant"),
            (b'{"v":1,"ev":"drops","dropped":true}\n', "dropped must be an integer"),
            (b'\xff\n', "utf-8"),
        ):
            with self.subTest(raw=raw):
                path = self.write("capture.jsonl", records)
                with path.open("ab") as output:
                    output.write(raw)
                with self.assertRaisesRegex(ValueError, pattern):
                    tool.read_videoout_flip_window(path)

    def test_pair_requires_same_revision_same_sample_count_and_distinct_inputs(self):
        anyps5 = self.write("anyps5.jsonl", run_records([1, 2]))
        for rows, pattern in ((run_records([1, 2], revision="fedcba987654"), "same build revision"),
                              (run_records([1, 2, 3]), "same flip count")):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                tool.compare_videoout_flip_windows(anyps5, self.write("kyty.jsonl", rows))
        with self.assertRaisesRegex(ValueError, "separate"):
            tool.compare_videoout_flip_windows(anyps5, anyps5)

    def test_bounds_file_size_and_accepts_crlf(self):
        path = self.write("capture.jsonl", run_records([1, 2]))
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertEqual(tool.read_videoout_flip_window(path)["flip_count"], 2)
        with mock.patch.object(tool, "MAX_FILE_BYTES", 10), self.assertRaisesRegex(ValueError, "exceeds 10 bytes"):
            tool.read_videoout_flip_window(path)

    def test_cli_reports_table_json_and_clear_failure(self):
        anyps5 = self.write("default.jsonl", run_records([1000, 21000]))
        kyty = self.write("kyty.jsonl", run_records([1000, 11000]))
        command = [sys.executable, "-B", tool.__file__, str(anyps5), str(kyty)]
        env = {**os.environ, "PYTHONUTF8": "1"}
        table = subprocess.run(command, check=True, capture_output=True, text=True, env=env).stdout
        self.assertIn("| AnyPS5 | 31 | 2 | 50.00 | 20.000 | 20.000 | 0 |", table)
        result = subprocess.run([*command, "--json"], check=True, capture_output=True, text=True, env=env)
        report = json.loads(result.stdout)
        self.assertEqual(set(report), {"format", "anyps5", "kyty"})
        self.assertEqual(report["format"], "aps5-videoout-flip-comparison/1")
        self.assertEqual(report["kyty"]["mean_fps"], 100)
        self.assertNotIn(str(self.root), result.stdout)
        self.write("kyty.jsonl", run_records([1000, 11000])[:-1])
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("incomplete window", result.stderr)

    def test_cli_compares_two_pinned_1200_flip_build_windows(self):
        baseline = self.write("baseline.jsonl", run_records(
            [50_000_000 + index * 47_000 for index in range(1200)], first=9000, revision=BASELINE_REVISION[:12]))
        candidate = self.write("candidate.jsonl", run_records(
            [2_000_000 + index * 28_000 for index in range(1200)], first=9000,
            generation=8, pid=11, revision=CANDIDATE_REVISION[:12]))
        command = [sys.executable, "-B", tool.__file__, str(baseline), str(candidate)]
        legacy = subprocess.run([*command, "--json"], capture_output=True, text=True)
        self.assertEqual(legacy.returncode, 2)
        self.assertIn("same build revision", legacy.stderr)
        command.extend(("--baseline-revision", BASELINE_REVISION, "--candidate-revision", CANDIDATE_REVISION))
        result = subprocess.run([*command, "--json"], check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["format"], "aps5-videoout-build-comparison/1")
        for arm, path, revision, frame_ms in (("baseline", baseline, BASELINE_REVISION, 47),
                                              ("candidate", candidate, CANDIDATE_REVISION, 28)):
            row = report[arm]
            self.assertEqual(row["expected_revision"], revision)
            self.assertEqual(row["commit"], revision[:12])
            self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual((row["first_flip"], row["flip_count"]), (9000, 1200))
            self.assertEqual((row["p50_ms"], row["p99_ms"]), (frame_ms, frame_ms))
        self.assertAlmostEqual(report["candidate_minus_baseline"]["mean_fps"], 1000 / 28 - 1000 / 47)
        self.assertEqual(report["candidate_minus_baseline"]["p99_ms"], -19)
        table = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        self.assertIn(f"| Baseline | `{BASELINE_REVISION}` | 9000 | 1200 |", table)
        self.assertIn(f"| Candidate | `{CANDIDATE_REVISION}` | 9000 | 1200 |", table)
        self.assertIn("Candidate minus baseline", table)

    def test_build_mode_accepts_distinct_runs_of_one_build_and_rejects_copies(self):
        baseline = self.write("baseline.jsonl", run_records([100, 200], revision=BASELINE_REVISION))
        candidate = self.write("candidate.jsonl", run_records([100, 120], pid=11, revision=BASELINE_REVISION))
        report = tool.compare_videoout_build_windows(baseline, candidate, BASELINE_REVISION, BASELINE_REVISION)
        self.assertEqual(report["candidate"]["mean_fps"], 50_000)
        with self.assertRaisesRegex(ValueError, "separate"):
            tool.compare_videoout_build_windows(baseline, baseline, BASELINE_REVISION, BASELINE_REVISION)
        candidate.write_bytes(baseline.read_bytes())
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            tool.compare_videoout_build_windows(baseline, candidate, BASELINE_REVISION, BASELINE_REVISION)

    def test_build_mode_enforces_revisions_and_complete_matching_windows(self):
        baseline = self.write("baseline.jsonl", run_records([100, 200], revision=BASELINE_REVISION[:12]))
        good = run_records([100, 300], revision=CANDIDATE_REVISION[:12])
        cases = (
            (good, BASELINE_REVISION[:12], CANDIDATE_REVISION, "full 40-digit"),
            (good, BASELINE_REVISION, BASELINE_REVISION, "does not match"),
            (good[:-1], BASELINE_REVISION, CANDIDATE_REVISION, "incomplete window"),
            (run_records([100, 300], first=40, revision=CANDIDATE_REVISION),
             BASELINE_REVISION, CANDIDATE_REVISION, "same first flip and flip count"),
            (run_records([100, 300, 500], revision=CANDIDATE_REVISION),
             BASELINE_REVISION, CANDIDATE_REVISION, "same first flip and flip count"),
            (run_records([100, 300], revision=BASELINE_REVISION[:12]),
             BASELINE_REVISION, BASELINE_REVISION[:12] + "2" * 28, "prefix matches both"),
        )
        for rows, expected_baseline, expected_candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                candidate = self.write("candidate.jsonl", rows)
                tool.compare_videoout_build_windows(baseline, candidate, expected_baseline, expected_candidate)

    def test_cli_requires_both_revision_options(self):
        for option in ("--baseline-revision", "--candidate-revision"):
            with self.subTest(option=option):
                command = [sys.executable, "-B", tool.__file__, "baseline.jsonl", "candidate.jsonl", option, BASELINE_REVISION]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("must be used together", result.stderr)


if __name__ == "__main__":
    unittest.main()
