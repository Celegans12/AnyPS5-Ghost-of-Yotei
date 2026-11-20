import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rt_survey


ROWS = [
    "program 0198d95d18d42cb2 stage=cs wave=64 group=64,1,1 lds=0 scratch=0 modules=3 source=manifest status=ok "
    "bvh=12 native=0 oracle=12 lane=0 empty=0 emulated=0 spirv=valid",
    "program 1111111111111111 back=2222222222222222 stage=cs wave=32 group=8,4,1 lds=256 scratch=0 modules=1 source=manifest "
    "status=ok bvh=7 native=3 oracle=1 lane=1 empty=1 emulated=1 spirv=invalid",
    "program 3333333333333333 stage=cs wave=64 group=8,8,1 lds=8192 scratch=16384 modules=0 source=code status=stop "
    "bvh=0 native=0 oracle=0 lane=0 empty=0 emulated=0 spirv=none phase=translate "
    "stop=resource_tracking:_hash=0x00112233aabbccdd_at_/home/someone/src/Tracking.cpp:772_C:\\\\Users\\\\someone\\\\x.cpp",
    "program 4444444444444444 stage=ps wave=64 group=0,0,0 lds=0 scratch=0 modules=0 source=code status=no_record "
    "bvh=3 native=0 oracle=0 lane=0 empty=0 emulated=0 spirv=none",
    "program 5555555555555555 stage=cs wave=64 group=64,1,1 lds=0 scratch=0 modules=2 source=manifest status=calls "
    "bvh=2 native=0 oracle=0 lane=0 empty=0 emulated=0 spirv=none",
]

FAKE_PROBE = """
import sys
args = sys.argv[1:]
if "--fail" in args:
    sys.exit(3)
def value(name):
    return args[args.index(name) + 1] if name in args else "none"
print("rt-survey 1 build=abc123def456+modified oracle=1 subgroup=%s lds_limit=%s" % (value("--subgroup"), value("--lds-limit")))
print("note __NOTE__")
for line in __ROWS__:
    print(line)
print("summary records=9 manifest_modules=4 manifest_skipped=0 code_only=3 programs=5 intersections=24 missing_code=1 "
      "unreadable=0 undecodable=0 truncated=0 seconds=0.5")
"""

REPORT = {
    "anyps5": {"commit": "0123456789ab", "release": None, "modified": False},
    "game": {"titleId": "PPSA00001"},
    "system": {"gpu": {"vendor": "NVIDIA", "name": "NVIDIA GeForce RTX 4080 SUPER", "driver": "570.153.02", "vulkan": "1.4.303"}},
}


def make_probe(directory, rows=ROWS, note="a note from the probe"):
    script = directory / "fake_probe.py"
    script.write_text(FAKE_PROBE.replace("__ROWS__", repr(rows)).replace("__NOTE__", note), encoding="utf-8")
    if os.name == "nt":
        launcher = directory / "kyty_rt_probe.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = directory / "kyty_rt_probe"
        launcher.write_text(f"#!{sys.executable}\n" + script.read_text(encoding="utf-8"), encoding="utf-8")
        launcher.chmod(0o755)
    return launcher


def make_cache(game, name="PPSA00001_01.000.000.code", age=0):
    folder = game / "_PipelineCache"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"KTYCODE1" + bytes(range(256)) * 4)
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


class ParseTests(unittest.TestCase):
    def test_rows_carry_every_count(self):
        header, rows, summary, notes = rt_survey.parse_probe("rt-survey 1 build=abc oracle=1 subgroup=32 lds_limit=49152\n" + "\n".join(ROWS) + "\n")
        self.assertEqual(header["subgroup"], "32")
        self.assertEqual(len(rows), 5)
        first = rows[0]
        self.assertEqual((first["program"], first["stage"], first["wave"], first["group"]), ("0198d95d18d42cb2", "cs", 64, [64, 1, 1]))
        self.assertEqual((first["intersections"], first["oracle"], first["spirv"]), (12, 12, True))
        self.assertEqual(rows[1]["back"], "2222222222222222")
        self.assertIs(rows[1]["spirv"], False)
        self.assertIsNone(rows[2]["spirv"])

    def test_totals_split_by_source(self):
        _, rows, _, _ = rt_survey.parse_probe("\n".join(ROWS))
        totals = rt_survey.totals_of(rows)
        self.assertEqual(totals["programs"], 5)
        self.assertEqual(totals["intersections"], 24)
        self.assertEqual((totals["native"], totals["oracle"], totals["lane"], totals["empty"], totals["emulated"]), (3, 13, 1, 1, 1))
        self.assertEqual((totals["spirvValid"], totals["spirvInvalid"]), (1, 1))
        self.assertEqual(totals["unclassified"], 5)
        self.assertEqual(totals["bySource"]["manifest"]["programs"], 3)
        self.assertEqual(totals["bySource"]["code"]["programs"], 2)
        self.assertEqual(totals["statuses"], {"calls": 1, "no_record": 1, "ok": 2, "stop": 1})

    def test_stop_text_keeps_no_hash_and_no_path(self):
        _, rows, _, _ = rt_survey.parse_probe("\n".join(ROWS))
        stop = rows[2]["stop"]
        self.assertIn("resource_tracking", stop)
        self.assertNotIn("00112233aabbccdd", stop)
        self.assertNotIn("/home", stop)
        self.assertNotIn("Users", stop)

    def test_notes_are_kept(self):
        _, _, _, notes = rt_survey.parse_probe("note one\nnote two\n")
        self.assertEqual(notes, ["one", "two"])

    def test_only_allowlisted_positive_diagnostic_counts_are_reported(self):
        diagnostic = " diagnostics=missing_tlas_source:2,unknown:/home/someone,local_flat_scratch:1,no_form_or_host_query:4294967296"
        _, rows, _, _ = rt_survey.parse_probe("\n".join([ROWS[1] + diagnostic, ROWS[0]]))
        self.assertEqual(rows[0]["diagnostics"], {"missing_tlas_source": 2, "local_flat_scratch": 1})
        self.assertEqual(rows[1]["diagnostics"], {})

    def test_text_lists_the_unserved_first(self):
        _, rows, _, _ = rt_survey.parse_probe("\n".join(ROWS))
        document = {
            "format": rt_survey.FORMAT,
            "title": {"id": "PPSA00001", "version": "01.000.000"},
            "anyKyty": {"commit": "abc", "modified": False},
            "host": {"gpu": {"name": "GPU", "driver": "1", "vulkan": "1.4", "source": "log"}, "subgroup": 32, "subgroupSource": "log",
                     "os": "Linux", "machine": "x86_64"},
            "totals": rt_survey.totals_of(rows),
            "notes": [],
            "programs": rows,
        }
        text = rt_survey.render_text(document)
        lines = text.splitlines()
        table = [line for line in lines if line[:16].strip() and all(c in "0123456789abcdef" for c in line[:16])]
        self.assertEqual(table[0][:16], "1111111111111111")
        self.assertIn("INVALID", table[0])
        self.assertIn("guessed inputs", text)
        self.assertIn("No probe diagnostics captured for this program.", text)


class SurveyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.game = self.root / "game"
        self.out = self.root / "out"
        self.cache = make_cache(self.game)
        (self.root / "reports").mkdir()
        self.report = self.root / "reports" / "anyps5-report.json"
        self.report.write_text(json.dumps(REPORT), encoding="utf-8")
        self.probe = make_probe(self.root)
        self.cwd = os.getcwd()
        os.chdir(self.root)
        self.patches = [patch.object(rt_survey, "data_folder", return_value=None), patch.object(rt_survey, "run_text", return_value=None)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        os.chdir(self.cwd)
        self.temp.cleanup()

    def run_main(self, *extra):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = rt_survey.main(["--game", str(self.game), "--probe", str(self.probe), "--out-dir", str(self.out), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_writes_the_json_and_the_text(self):
        code, stdout, _ = self.run_main("--report", str(self.report))
        self.assertEqual(code, 0)
        document = json.loads((self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8"))
        self.assertEqual(document["format"], "rt-survey/1")
        self.assertEqual(document["title"], {"id": "PPSA00001", "version": "01.000.000"})
        self.assertEqual(document["anyKyty"]["commit"], "0123456789ab")
        self.assertEqual(document["anyKyty"]["commitSource"], "run-report")
        self.assertEqual(document["anyKyty"]["probeBuild"], "abc123def456")
        self.assertTrue(document["anyKyty"]["probeBuildModified"])
        self.assertEqual(document["host"]["gpu"]["name"], "NVIDIA GeForce RTX 4080 SUPER")
        self.assertEqual(document["host"]["gpu"]["driver"], "570.153.02")
        self.assertEqual(document["host"]["gpu"]["source"], "run-report")
        self.assertEqual(len(document["programs"]), 5)
        self.assertEqual(document["totals"]["intersections"], 24)
        self.assertEqual(document["notes"], ["a note from the probe"])
        text = (self.out / "rt-survey-PPSA00001-01.000.000.txt").read_text(encoding="utf-8")
        self.assertIn("PPSA00001", text)
        self.assertIn("NVIDIA GeForce RTX 4080 SUPER", text)
        self.assertIn("rt-survey-PPSA00001-01.000.000.json", stdout)

    def test_keeps_diagnostics_with_the_reported_wave_and_input_source(self):
        recorded = ROWS[1] + " diagnostics=missing_tlas_source:2,no_form_or_host_query:1"
        guessed = ROWS[1].replace("wave=32", "wave=64").replace("source=manifest", "source=code")
        guessed += " diagnostics=local_flat_scratch:1,unrecognized:/home/private/raw_shader"
        self.probe = make_probe(self.root, [recorded, guessed])
        code, output, error = self.run_main("--report", str(self.report))
        self.assertEqual(code, 0, error)
        document = json.loads((self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8"))
        self.assertEqual([(row["source"], row["wave"], row["diagnostics"]) for row in document["programs"]], [
            ("manifest", 32, {"missing_tlas_source": 2, "no_form_or_host_query": 1}),
            ("code", 64, {"local_flat_scratch": 1}),
        ])
        self.assertIn("TLAS source could not be derived (2)", output)
        self.assertIn("Local FLAT access requires DWORD data and per-thread scratch storage (1)", output)
        self.assertNotIn("raw_shader", json.dumps(document) + output)

    def test_nothing_of_this_machine_is_written(self):
        self.run_main("--report", str(self.report))
        for path in self.out.iterdir():
            content = path.read_text(encoding="utf-8")
            self.assertNotIn(str(self.root), content)
            self.assertNotIn(str(Path.home()), content)
            self.assertNotIn("_PipelineCache", content)
            self.assertNotIn(".code", content)

    def test_an_ipv6_address_in_a_note_is_written_to_neither_format(self):
        address = "2001:db8:85a3::8a2e:370:7334"
        note = f"peer [{address}]:8080 link fe80::1%eth0 kept C++20 shader:phase 00:11"
        self.probe = make_probe(self.root, note=note)
        code, output, error = self.run_main("--report", str(self.report))
        self.assertEqual(code, 0, error)
        written = (self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8")
        written += (self.out / "rt-survey-PPSA00001-01.000.000.txt").read_text(encoding="utf-8")
        for text in (written, output):
            self.assertNotIn("2001:db8", text)
            self.assertNotIn("fe80", text)
            self.assertNotIn("eth0", text)
        document = json.loads((self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8"))
        self.assertEqual(document["notes"], ["peer <ip> link <ip> kept C++20 shader:phase 00:11"])
        self.assertIn("peer <ip> link <ip> kept C++20 shader:phase 00:11", written)

    def test_host_settings_reach_the_probe(self):
        self.run_main("--report", str(self.report), "--subgroup", "64", "--lds-limit", "65536")
        document = json.loads((self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8"))
        self.assertEqual(document["host"]["subgroup"], 64)
        self.assertEqual(document["host"]["subgroupSource"], "option")
        self.assertEqual(document["probe"], {"oracle": 1, "subgroup": 64, "ldsLimitBytes": 65536})

    def test_subgroup_and_gpu_come_from_a_log(self):
        log = self.root / "run.log"
        log.write_text(
            "Select device: AMD Radeon RX 7900 XTX\nVulkan subgroup: default=64 min=32 max=64 stages=0x00000020 size_control=true wave64=true\n",
            encoding="utf-8",
        )
        code, _, _ = self.run_main("--log", str(log))
        self.assertEqual(code, 0)
        document = json.loads((self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8"))
        self.assertEqual(document["host"]["gpu"]["name"], "AMD Radeon RX 7900 XTX")
        self.assertEqual(document["host"]["gpu"]["source"], "log")
        self.assertEqual((document["host"]["subgroup"], document["host"]["subgroupSource"]), (64, "log"))
        self.assertEqual(document["host"]["ldsLimitBytes"], 65536)
        self.assertEqual(document["anyKyty"]["commitSource"], "probe")

    def test_vendor_decides_the_subgroup_without_a_log(self):
        code, _, _ = self.run_main()
        self.assertEqual(code, 0)
        document = json.loads((self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8"))
        self.assertEqual(document["host"]["gpu"]["name"], None)
        self.assertEqual((document["host"]["subgroup"], document["host"]["subgroupSource"]), (32, "default"))

    def test_the_report_next_to_the_game_is_found(self):
        (self.game / "anyps5-report.json").write_text(json.dumps(REPORT), encoding="utf-8")
        code, _, _ = self.run_main()
        self.assertEqual(code, 0)
        document = json.loads((self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8"))
        self.assertEqual(document["anyKyty"]["commit"], "0123456789ab")
        self.assertEqual(document["host"]["gpu"]["name"], "NVIDIA GeForce RTX 4080 SUPER")

    def test_a_report_of_another_title_is_ignored(self):
        other = dict(REPORT, game={"titleId": "PPSA99999"})
        (self.game / "anyps5-report.json").write_text(json.dumps(other), encoding="utf-8")
        code, _, _ = self.run_main()
        self.assertEqual(code, 0)
        document = json.loads((self.out / "rt-survey-PPSA00001-01.000.000.json").read_text(encoding="utf-8"))
        self.assertEqual(document["anyKyty"]["commitSource"], "probe")

    def test_the_newest_cache_is_chosen_and_the_others_are_named(self):
        make_cache(self.game, "PPSA00002_01.000.000.code", age=3600)
        code, stdout, _ = self.run_main("--report", str(self.report))
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "rt-survey-PPSA00001-01.000.000.json").is_file())
        self.assertFalse((self.out / "rt-survey-PPSA00002-01.000.000.json").exists())
        self.assertIn("1 older cache file(s) were left out", stdout)

    def test_a_title_can_be_chosen(self):
        make_cache(self.game, "PPSA00002_02.001.000.code", age=3600)
        code, _, _ = self.run_main("--title", "PPSA00002", "--report", str(self.report))
        self.assertEqual(code, 0)
        self.assertTrue((self.out / "rt-survey-PPSA00002-02.001.000.json").is_file())

    def test_a_shared_manifest_name_is_not_a_cache(self):
        (self.game / "_PipelineCache" / "PPSA00001_01.000.000.shared.kpm").write_text("x", encoding="utf-8")
        (self.game / "_PipelineCache" / "PPSA00001.shaders").write_text("x", encoding="utf-8")
        self.assertEqual([path.name for path in rt_survey.cache_files(self.game / "_PipelineCache")], ["PPSA00001_01.000.000.code"])

    def test_no_cache_says_what_to_do(self):
        empty = self.root / "empty"
        empty.mkdir()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = rt_survey.main(["--game", str(empty), "--probe", str(self.probe), "--out-dir", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("Play the game on AnyKyty", stderr.getvalue())
        self.assertFalse(self.out.exists())

    def test_a_failing_probe_writes_nothing(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(rt_survey, "run_probe", side_effect=rt_survey.SurveyError("the probe failed with exit code 3: x")):
                code = rt_survey.main(["--game", str(self.game), "--probe", str(self.probe), "--out-dir", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("exit code 3", stderr.getvalue())
        self.assertFalse(self.out.exists())

    def test_the_probe_exit_code_is_reported(self):
        with patch("rt_survey.subprocess.run") as run:
            run.return_value.returncode = 3
            run.return_value.stdout = ""
            run.return_value.stderr = "boom\n"
            with self.assertRaises(rt_survey.SurveyError) as caught:
                rt_survey.run_probe(self.probe, self.cache, {"subgroup": 32, "ldsLimitBytes": 49152}, None, 5)
        self.assertIn("exit code 3", str(caught.exception))

    def test_a_report_that_would_leak_a_path_is_refused(self):
        with self.assertRaises(rt_survey.SurveyError):
            rt_survey.audit("written to " + str(self.cache.parent), self.cache)

    def test_a_missing_probe_names_the_build_command(self):
        with patch.dict(os.environ, {"PATH": str(self.root / "nowhere")}), patch.dict(os.environ, {"KYTY_RT_PROBE": ""}):
            with self.assertRaises(rt_survey.SurveyError) as caught:
                rt_survey.find_probe(None, self.root)
        self.assertIn("cmake --build build --target kyty_rt_probe", str(caught.exception))

    def test_the_probe_is_found_in_the_build_folder(self):
        folder = self.root / "build" / "tests"
        folder.mkdir(parents=True)
        probe = folder / rt_survey.PROBE_NAME
        probe.write_text("x", encoding="utf-8")
        with patch.dict(os.environ, {"KYTY_RT_PROBE": ""}):
            self.assertEqual(rt_survey.find_probe(None, self.root), probe)


class TextCleaningTests(unittest.TestCase):
    def test_home_user_host_and_ip_are_replaced(self):
        who = {"home": "/home/ana", "names": {"ana"}, "host": "serverfestivo"}
        text = rt_survey.clean_text("Driver at /home/ana/x on serverfestivo by ana from 192.168.1.20", who)
        self.assertNotIn("ana/", text)
        self.assertNotIn("serverfestivo", text)
        self.assertNotIn("192.168", text)

    def test_ipv6_addresses_are_replaced_in_every_written_form(self):
        who = {"home": "/home/ana", "names": set(), "host": ""}
        cases = {
            "peer 2001:db8::1 up": "peer <ip> up",
            "peer 2001:0db8:85a3:0000:0000:8a2e:0370:7334.": "peer <ip>.",
            "bracketed [2001:db8::1]:8080 and [fe80::1]": "bracketed <ip> and <ip>",
            "zone fe80::1%eth0 and fe80::a:b%12": "zone <ip> and <ip>",
            "loopback ::1 and mapped ::ffff:192.168.1.20 and 64:ff9b::192.0.2.33": "loopback <ip> and mapped <ip> and <ip>",
            "mixed 10.0.0.1 and 2001:db8::2": "mixed <ip> and <ip>",
        }
        for text, expected in cases.items():
            self.assertEqual(rt_survey.clean_text(text, who), expected, text)

    def test_text_that_only_looks_like_an_address_is_kept(self):
        who = {"home": "/home/ana", "names": set(), "host": ""}
        for text in ("C++20", "shader:phase", "00:11", "10:30:45", "12:34:56.789", "std::string", "Foo::Bar", "a:b", "ratio 1:2",
                     "MAC aa:bb:cc:dd:ee:ff", "::", "hash=0x00112233aabbccdd", "driver 595.99.02", "RTX 4080: ok"):
            self.assertEqual(rt_survey.clean_text(text, who), text)

    def test_stop_text_keeps_the_source_file_name_only(self):
        self.assertEqual(rt_survey.clean_stop("a_/x/y/File.cpp:12_b"), "a_File.cpp:12_b")
        self.assertEqual(rt_survey.clean_stop("hash=0x00112233aabbccdd"), "hash=0x<n>")


if __name__ == "__main__":
    unittest.main()
