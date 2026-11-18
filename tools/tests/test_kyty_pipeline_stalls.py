import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kyty_pipeline_stalls as tool


class KytyPipelineStallTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def telemetry(self, times, expected_count=None):
        contract = {
            "v": 1, "ev": "contract", "t_ns": 1, "contract": "v1-pad",
            "prx": "libSceVideoOut", "commit": "0123456789ab", "pid": 10,
            "flip_start": 31, "flip_count": len(times) if expected_count is None else expected_count,
        }
        records = [contract, *(
            {"v": 1, "ev": "flip_complete", "n": 31 + index, "generation": 5,
             "t_ns": timestamp, "process_us": process_us}
            for index, (timestamp, process_us) in enumerate(times)
        )]
        path = self.root / "videoout.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    def session_log(self, text):
        path = self.root / "session.log"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_pipeline_events_from_current_producers(self):
        log = self.session_log(
            "--- Session 2026-11-18 10:24:07 ---\n"
            "Kyty build 0123456789ab\n"
            "[kyty-gpl] stalls 3 total 1.200 s max 0.700 s >=0.1s 3 >=0.5s 1; driver pipelines 8\n"
            "[kyty-gpl] stall 0.250 s graphics-sync 0x0123456789abcdef linked fs=queued pr=building @1234567890\n"
            "[kyty-gpl] stall 0.002 s compute-sync 0x23456789abcdef01  @1235000000\n"
            "[kyty-gpl] translate 0.123 s 0x3456789abcdef012 @1236000000\n"
            "[kyty-gpl] compute-driver 0.037411 s 0x456789abcdef0123 module=42 spirv_words=4370 mode=create result=Success @1237000000\n"
            "[kyty-gpl] compute-driver 0.000007 s 0x56789abcdef01234 module=9 spirv_words=815 mode=lookup result=PipelineCompileRequired @1238000000\n"
        )
        events = list(tool.read_kyty_pipeline_events(log))
        self.assertEqual([(event.kind, event.shader, event.end_ns, event.duration_ns) for event in events], [
            ("stall", "0123456789abcdef", 1_234_567_890, 250_000_000),
            ("stall", "23456789abcdef01", 1_235_000_000, 2_000_000),
            ("translate", "3456789abcdef012", 1_236_000_000, 123_000_000),
            ("compute-create", "456789abcdef0123", 1_237_000_000, 37_411_000),
            ("compute-lookup", "56789abcdef01234", 1_238_000_000, 7_000),
        ])
        self.assertEqual(events[0].details, {"stall_type": "graphics-sync", "route": "linked fs=queued pr=building"})
        self.assertEqual(events[1].details, {"stall_type": "compute-sync", "route": ""})
        self.assertEqual(events[3].details, {"module": 42, "spirv_words": 4370, "mode": "create", "result": "Success"})
        self.assertEqual(events[4].details, {"module": 9, "spirv_words": 815, "mode": "lookup",
                                             "result": "PipelineCompileRequired"})

    def test_accepts_headerless_stderr_with_a_complete_record(self):
        log = self.session_log(
            "WARNING: [Loader Message] debug output before Vulkan pipeline work\n"
            "[kyty-gpl] compute-driver 0.005000 s 0x0123456789abcdef module=3 spirv_words=19 mode=create result=Success @3000000000\n"
        )
        event, = tool.read_kyty_pipeline_events(log)
        self.assertEqual(event.kind, "compute-create")
        self.assertEqual(event.end_ns, 3_000_000_000)
        self.assertEqual(event.duration_ns, 5_000_000)

    def test_rejects_malformed_or_truncated_recognized_rows(self):
        cases = (
            "[kyty-gpl] stall 0.050 s graphics-sync 0x0123456 driver-cache @2000000000\n",
            "[kyty-gpl] translate NaN s 0x0123456789abcdef @2000000000\n",
            "[kyty-gpl] translate 3.000 s 0x0123456789abcdef @2000000000\n",
            "[kyty-gpl] compute-driver 0.000003 s 0x0123456789abcdef module=8 spirv_words=101 mode=unknown result=Success @2000000000\n",
            "[kyty-gpl] translate 0.010 s 0x0123456789abcdef @",
            "[kyty-gpl] translate 0.010 s 0x0123456789abcdef @2000000000",
        )
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                list(tool.read_kyty_pipeline_events(self.session_log(row)))

    def test_rejects_multiple_session_headers(self):
        log = self.session_log(
            "--- Session 2026-11-18 10:24:07 ---\n"
            "[kyty-gpl] translate 0.050 s 0x0123456789abcdef @2000000000\n"
            "--- Relaunched 2026-11-18 10:25:07 ---\n"
            "[kyty-gpl] translate 0.050 s 0x0123456789abcdef @3000000000\n"
        )
        with self.assertRaises(ValueError):
            list(tool.read_kyty_pipeline_events(log))

    def test_ranks_process_gaps_and_matches_steady_intervals(self):
        telemetry = self.telemetry([(100_000_000, 1_000), (140_000_000, 41_000),
                                    (200_000_000, 66_000), (210_000_000, 106_000)])
        log = self.session_log(
            "[kyty-gpl] stall 0.060 s graphics-sync 0x1111111111111111 whole @195000000\n"
            "[kyty-gpl] stall 0.020 s graphics-sync 0x8888888888888888 driver-cache @122000000\n"
            "[kyty-gpl] translate 0.010 s 0x2222222222222222 @150000000\n"
            "[kyty-gpl] compute-driver 0.020300 s 0x3333333333333333 module=8 spirv_words=101 mode=create result=Success @212000000\n"
            "[kyty-gpl] compute-driver 0.000000 s 0x4444444444444444 module=8 spirv_words=101 mode=lookup result=Success @140000000\n"
            "[kyty-gpl] compute-driver 0.000000 s 0x5555555555555555 module=8 spirv_words=101 mode=lookup result=Success @210000000\n"
            "[kyty-gpl] translate 0.010 s 0x6666666666666666 @100000000\n"
            "[kyty-gpl] compute-driver 0.000000 s 0x7777777777777777 module=8 spirv_words=101 mode=lookup result=Success @100000000\n"
        )
        report = tool.correlate_kyty_pipeline_stalls(telemetry, log)
        self.assertEqual(report["pipeline_events"], 8)
        self.assertEqual(report["pipeline_events_in_window"], 6)
        self.assertEqual([(gap["from_flip"], gap["to_flip"], gap["frame_ms"]) for gap in report["gaps"]],
                         [(31, 32, 40), (33, 34, 40), (32, 33, 25)])
        first, last, middle = (gap["overlaps"] for gap in report["gaps"])
        self.assertEqual(set(first), {"stall", "compute-lookup"})
        self.assertEqual(set(last), {"compute-create", "compute-lookup"})
        self.assertEqual(set(middle), {"stall", "translate", "compute-create"})
        self.assertEqual(first["stall"]["count"], 2)
        self.assertEqual(first["stall"]["largest"]["shader"], "8888888888888888")
        self.assertEqual(first["stall"]["largest"]["overlap_ms"], 20)
        self.assertEqual(middle["stall"]["largest"]["route"], "whole")
        self.assertEqual(middle["stall"]["largest"]["duration_ms"], 60)
        self.assertEqual(middle["stall"]["largest"]["overlap_ms"], 55)
        self.assertEqual(first["compute-lookup"]["count"], 1)
        self.assertEqual(first["compute-lookup"]["largest"]["shader"], "4444444444444444")
        self.assertEqual(first["compute-lookup"]["largest"]["overlap_ms"], 0)
        self.assertEqual(last["compute-lookup"]["count"], 1)
        self.assertEqual(last["compute-lookup"]["largest"]["shader"], "5555555555555555")
        self.assertAlmostEqual(middle["compute-create"]["largest"]["overlap_ms"], 8.3)
        self.assertEqual(last["compute-create"]["largest"]["overlap_ms"], 10)
        self.assertEqual(tool.correlate_kyty_pipeline_stalls(telemetry, log, limit=2)["gaps"], report["gaps"][:2])

    def test_rejects_incomplete_or_nonmonotonic_flip_windows(self):
        log = self.session_log("")
        cases = (
            ("missing-flip", [(100, 1), (200, 2)], 3),
            ("same-steady-clock", [(100, 1), (100, 2)], 2),
            ("reversed-steady-clock", [(100, 1), (99, 2)], 2),
            ("reversed-process-clock", [(100, 2), (200, 1)], 2),
        )
        for name, times, expected_count in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                tool.correlate_kyty_pipeline_stalls(self.telemetry(times, expected_count), log)

    def test_reports_gaps_when_no_pipeline_events_overlap(self):
        telemetry = self.telemetry([(100_000_000, 1_000), (120_000_000, 6_000)])
        log = self.session_log(
            "[kyty-gpl] stalls 1 total 0.001 s max 0.001 s >=0.1s 0 >=0.5s 0; driver pipelines 1\n"
        )
        report = tool.correlate_kyty_pipeline_stalls(telemetry, log)
        self.assertEqual(report["pipeline_events"], 0)
        gap, = report["gaps"]
        self.assertEqual(gap["frame_ms"], 5)
        self.assertEqual(gap["overlaps"], {})

    def test_cli_json_and_input_failure(self):
        telemetry = self.telemetry([(100_000_000, 1_000), (120_000_000, 6_000), (160_000_000, 46_000)])
        log = self.session_log("")
        command = [sys.executable, "-B", tool.__file__, str(telemetry), str(log), "--top", "1", "--json"]
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        report = json.loads(result.stdout)
        gap, = report["gaps"]
        self.assertEqual((gap["from_flip"], gap["to_flip"], gap["frame_ms"]), (32, 33, 40))
        self.session_log("[kyty-gpl] translate 0.010 s 0x0123456789abcdef @")
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr)


if __name__ == "__main__":
    unittest.main()
