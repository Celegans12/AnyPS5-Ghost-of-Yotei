import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kyty_ab_report

EPOCH = 1000.0
PHASE = 20.0
SETTLE = 4.0
FPS = {1: 55.0, 2: 50.0, 3: 56.0, 4: 52.0, 5: 70.0}
SETTING = {"A": "KYTY_WRITE_FAULT_FAST_PATH=1", "B": "KYTY_WRITE_FAULT_FAST_PATH=0"}


def arm(index):
    return "B" if index % 2 else "A"


def summary(faults):
    return (f"[kyty-write-blocks] summary of the last 10 s: {faults} guest write faults, 44 opened their 64 KiB block "
            "(188 clean pages marked written with them), 394 left to the page path; 48.5 MiB of buffer uploads")


def session_lines():
    events = [(EPOCH, f"[kyty-ab] in-run A/B: phases of {PHASE:.1f} s from present 0, settle {SETTLE:.1f} s; "
                      f"A: {SETTING['A']}; B: {SETTING['B']}")]
    present = 1
    for index in range(7):
        begin = EPOCH + PHASE * index
        side = arm(index)
        events.append((begin, f"[kyty-ab] phase {side} {index} begins at present {present} (t {begin - EPOCH:.3f} s, "
                              f"wall {begin:.3f}): {SETTING[side]}"))
        events.append((begin + 2, summary(9999)))
        settled = begin + SETTLE
        events.append((settled, f"[kyty-ab] phase {side} {index} settled at present {present + 100} "
                                f"(t {settled - EPOCH:.3f} s, wall {settled:.3f})"))
        events.append((begin + 8, summary(8888)))
        events.append((begin + 18, summary(300 if side == "B" else 400)))
        if index < 6:
            fps = FPS.get(index, 40.0)
            presents = round(fps * (PHASE - SETTLE))
            end = begin + PHASE
            events.append((end - 1e-3, f"[kyty-ab] phase {side} {index} settled for {PHASE - SETTLE:.3f} s: {presents} presents, "
                                       f"{fps:.2f} img/s, longest frame 30.0 ms, 0 frames of 50 ms or more; presents "
                                       f"{present + 100} to {present + 100 + presents}, wall {settled:.3f} to {end:.3f}"))
        present += 1000
    events.sort(key=lambda event: event[0])
    return [text for _, text in events]


def phase_at(wall):
    index = int((wall - EPOCH) // PHASE)
    return index, wall - EPOCH - PHASE * index


def write_session(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stderr.log").write_text("\n".join(["[kyty] runtime started"] + session_lines()) + "\n", encoding="utf-8")
    (folder / "start.json").write_text(json.dumps({"vk_epoch": EPOCH}), encoding="utf-8")
    (folder / "phases.jsonl").write_text("\n".join(json.dumps({"phase": name, "wall": wall}) for name, wall in
                                                   (("launch", EPOCH), ("game", EPOCH + 10), ("measure", EPOCH + 15))) + "\n",
                                         encoding="utf-8")
    vk, threads = [], []
    for step in range(1, 71):
        t = step * 2.0
        index, into = phase_at(EPOCH + t - 1)
        settled = into >= SETTLE
        fps = FPS.get(index, 40.0) if settled else 10.0
        dpf = (3030.0 if arm(index) == "B" else 3000.0) if settled else 100.0
        vk.append(f"[vkcount] t={t:.1f} s dt=2.00 fps={fps:.2f} ms_per_frame={1000 / fps:.2f} draw_calls_per_frame={dpf:.1f} "
                  f"draw_calls_per_s={round(fps * dpf)} draw=0 presents={round(fps * 2)}")
        wait = 55.0 if index == 5 and 10 <= into < 12 else 0.5
        threads.append(json.dumps({"t": t, "wall": EPOCH + t, "dt": 2.0, "threads": [
            {"tid": 1, "name": "kytyGpuExecute", "pct": 80.0, "wait": wait},
            {"tid": 2, "name": "idle", "pct": 5.0, "wait": 90.0}]}))
    (folder / "vkcount.log").write_text("[vkcount] layer loaded, period 2.0 s\n" + "\n".join(vk) + "\n", encoding="utf-8")
    (folder / "threads.jsonl").write_text("\n".join(threads) + "\n", encoding="utf-8")


class KytyAbReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name) / "kab-session"
        write_session(self.folder)

    def tearDown(self):
        self.temporary.cleanup()

    def analyse(self, **options):
        return kyty_ab_report.analyse(self.folder, self.folder / "stderr.log", **options)

    def test_phases_kept_and_set_aside(self):
        result = self.analyse()
        why = {phase["phase"]: (phase["kept"], phase["why"]) for phase in result["phases"]}
        self.assertEqual(why[0], (False, "before the measure step"))
        self.assertEqual([index for index, (kept, _) in why.items() if kept], [1, 2, 3, 4])
        self.assertTrue(why[5][1].startswith("disturbed (wait 55.0 % > 10 %)"))
        self.assertEqual(why[6], (False, "no settled line (the session stopped in it)"))

    def test_per_arm_rates_and_scene(self):
        result = self.analyse()
        a, b = result["per_arm"]["A"], result["per_arm"]["B"]
        self.assertEqual((a["phases"], b["phases"]), (2, 2))
        self.assertAlmostEqual(a["fps"], 51.0)
        self.assertAlmostEqual(a["fps_sd"], 2 ** 0.5)
        self.assertAlmostEqual(b["fps"], 55.5)
        self.assertAlmostEqual(a["vk_fps"], 51.0)
        self.assertAlmostEqual(b["vk_fps"], 55.5)
        self.assertAlmostEqual(a["draws_per_image"], 3000.0)
        self.assertAlmostEqual(b["draws_per_image"], 3030.0)
        self.assertEqual(a["settings"], SETTING["A"])
        drift = result["drift_cancelled"]
        self.assertEqual(drift["count"], 1)
        self.assertAlmostEqual(drift["mean"], 5.0)

    def test_settle_windows_are_left_out(self):
        phase = {phase["phase"]: phase for phase in self.analyse()["phases"]}[2]
        self.assertEqual(phase["vk_windows"], 8)
        self.assertAlmostEqual(phase["draws_per_image"], 3000.0)
        skipped = {phase["phase"]: phase for phase in self.analyse(skip=4.0)["phases"]}[2]
        self.assertEqual(skipped["vk_windows"], 6)

    def test_counters_count_only_whole_settled_windows(self):
        counters = self.analyse()["counters"]
        self.assertEqual(len(counters), 1)
        counter = counters[0]
        self.assertEqual(counter["kind"], "kyty-write-blocks")
        self.assertEqual(counter["lines"], {"A": 2, "B": 2})
        self.assertEqual(counter["A"][1], 400.0)
        self.assertEqual(counter["B"][1], 300.0)
        self.assertEqual(kyty_ab_report.fill(counter["template"], counter["A"], counter["B"])[:63],
                         "summary of the last 10 s: 400|300 guest write faults, 44 opened")

    def test_from_all_keeps_the_first_phase(self):
        kept = [phase["phase"] for phase in self.analyse(start_step="all")["phases"] if phase["kept"]]
        self.assertEqual(kept, [0, 1, 2, 3, 4])

    def test_lines_printed_by_the_game(self):
        lines = ["[kyty-ab] in-run A/B: phases of 2.0 s from present 3, settle 0.5 s; A: KYTY_FLAT_DEFER=0, "
                 "KYTY_GENERIC_COMPUTE_VARIANT=1; B: KYTY_FLAT_DEFER=1, KYTY_GENERIC_COMPUTE_VARIANT=0",
                 "[kyty-ab] phase A 0 begins at present 3 (t 0.250 s, wall 1000.250): KYTY_FLAT_DEFER=0, "
                 "KYTY_GENERIC_COMPUTE_VARIANT=1",
                 "[kyty-ab] phase A 0 settled at present 7 (t 0.750 s, wall 1000.750)",
                 "[kyty-ab] phase A 0 settled for 1.500 s: 12 presents, 8.00 img/s, longest frame 125.0 ms, 12 frames of "
                 "50 ms or more; presents 7 to 19, wall 1000.750 to 1002.250",
                 "[kyty-ab] phase B 1 begins at present 19 (t 2.250 s, wall 1002.250): KYTY_FLAT_DEFER=1, "
                 "KYTY_GENERIC_COMPUTE_VARIANT=0"]
        arms, phases, _ = kyty_ab_report.parse_log(lines)
        self.assertEqual(arms["B"], "KYTY_FLAT_DEFER=1, KYTY_GENERIC_COMPUTE_VARIANT=0")
        self.assertEqual(phases[0]["fps"], 8.0)
        self.assertEqual(phases[0]["long_frames"], 12)
        self.assertEqual(phases[0]["end_wall"], 1002.25)
        self.assertNotIn("end_wall", phases[1])

    def test_command_line(self):
        output = io.StringIO()
        target = Path(self.temporary.name) / "report.json"
        with redirect_stdout(output):
            code = kyty_ab_report.main([str(self.folder), "--json", str(target)])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("A: 2 phase(s); 51.00 img/s", text)
        self.assertIn("B - A: +4.50 img/s (+8.8 %)", text)
        self.assertIn("scene: draws per image B/A = 1.010 (same scene)", text)
        self.assertIn("[kyty-write-blocks] (2|2 lines) summary of the last 10 s: 400|300 guest write faults", text)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["per_arm"]["B"]["phases"], 2)

    def test_log_without_phases(self):
        empty = Path(self.temporary.name) / "empty.log"
        empty.write_text("[kyty] runtime started\n", encoding="utf-8")
        errors = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(errors):
            self.assertEqual(kyty_ab_report.main([str(empty)]), 2)
        self.assertIn("has no [kyty-ab] phase: was KYTY_AB set?", errors.getvalue())

    def test_markdown_summary_and_counters(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = kyty_ab_report.main([str(self.folder), "--markdown"])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("| A | 2 | 51.00 (sd 1.41) | 51.00 (sd 1.41) | 3000 (sd 0) | 30.0 | 0 |", text)
        self.assertIn("| B | 2 | 55.50 (sd 0.71) | 55.50 (sd 0.71) | 3030 (sd 0) | 30.0 | 0 |", text)
        self.assertIn("| B - A | - | +4.50 (+8.8 %) |", text)
        self.assertIn("| Drift-cancelled B - A | 1 | +5.00 (standard error -) |", text)
        self.assertLess(text.index("| B - A |"), text.index("| Drift-cancelled B - A |"))
        self.assertIn(SETTING["A"], text)
        self.assertIn(SETTING["B"], text)
        self.assertIn("Phases kept: 4; set aside: 3.", text)
        self.assertIn("draws per image B/A = 1.010 (same scene)", text)
        self.assertIn("| kyty-write-blocks | 2 | 2 | summary of the last 10 s: 400\\|300 guest write faults", text)
        rows = [line for line in text.splitlines() if line.startswith("|")]
        for row in rows:
            self.assertNotIn("400|300", row)

    def test_markdown_keeps_json_unchanged(self):
        targets = [Path(self.temporary.name) / name for name in ("text.json", "markdown.json")]
        for options, target in zip(([], ["--markdown"]), targets):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(kyty_ab_report.main([str(self.folder), *options, "--json", str(target)]), 0)
        self.assertEqual(targets[0].read_bytes(), targets[1].read_bytes())

    def test_markdown_drift_standard_error(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = kyty_ab_report.main([str(self.folder), "--markdown", "--from", "all", "--wait-max", "100"])
        self.assertEqual(code, 0)
        self.assertIn("| Drift-cancelled B - A | 2 | +7.50 (standard error 2.50) |", output.getvalue())

    def test_markdown_plain_log_and_missing_arm(self):
        log = Path(self.temporary.name) / "plain.log"
        log.write_text("\n".join(session_lines()) + "\n", encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(kyty_ab_report.main([str(log), "--markdown"]), 0)
        self.assertIn("| A | 3 | 47.33 (sd 6.43) | - (sd -) | - (sd -) | 30.0 | 0 |", output.getvalue())
        self.assertIn("Phases kept: 6; set aside: 1.", output.getvalue())
        log.write_text("\n".join(session_lines()[:6]) + "\n", encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(kyty_ab_report.main([str(log), "--markdown"]), 2)
        self.assertIn("| B | 0 | - (sd -) |", output.getvalue())
        self.assertIn("| Drift-cancelled B - A | 0 | - |", output.getvalue())

    def test_markdown_escapes_table_cells(self):
        result = self.analyse()
        result["per_arm"]["A"]["settings"] = "a|b\n<setting>"
        output = io.StringIO()
        kyty_ab_report.render_markdown(result, output)
        self.assertIn("| A | a\\|b<br>&lt;setting&gt; |", output.getvalue())

    def test_help_describes_session_contents(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as exited:
            kyty_ab_report.main(["--help"])
        self.assertEqual(exited.exception.code, 0)
        text = " ".join(output.getvalue().split())
        self.assertNotIn("planb/runs", text)
        for name in ("stderr.log", "vkcount.log", "threads.jsonl", "phases.jsonl", "plain log file", "--markdown"):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
