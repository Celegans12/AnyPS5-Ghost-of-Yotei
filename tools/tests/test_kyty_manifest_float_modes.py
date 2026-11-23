import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kyty_manifest_float_modes

MANIFEST = """# kyty pipeline manifest v1
M 000000000001 vs code=1111222233334444 ud=4
M 000000000002 ps code=5555666677778888
M 000000000003 ls code=0000000000000003
M 000000000004 hs code=0000000000000004
M 000000000005 tes code=0000000000000005
M 000000000006 cs code=0000000000000006
M 000000000007 cs code=0000000000000007 fm=f0
P v=000000000001 p=000000000002 fv=c0.0.1 fp=f0.1.1 cc=1
P v=000000000001 fv=c0.0.0
P v=000000000003 h=000000000004 d=000000000005 p=000000000002 fv=c0.1.0 fh=c0.1.0 fd=c0.0.0 fp=c0.0.0
P v=000000000001 p=000000000002 cc=2
T 0000000000000001 icp=3
"""


class ManifestFloatModesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text):
        path = self.folder / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_counts_per_stage(self):
        result = kyty_manifest_float_modes.count(self.write("CUSA00001_01.000.000.kpm", MANIFEST))
        self.assertEqual(result["title"], "CUSA00001_01.000.000")
        self.assertEqual(result["pipelines"], 4)
        self.assertEqual(result["pipelines_recorded"], 3)
        stages = result["stages"]
        self.assertEqual(list(stages), ["vs", "ls", "hs", "tes", "ps"])
        self.assertEqual(stages["vs"]["stages"], 3)
        self.assertEqual(stages["vs"]["recorded"], 2)
        self.assertEqual(stages["vs"]["float_mode"], 0)
        self.assertEqual(stages["vs"]["dx10_clamp"], 1)
        self.assertEqual(stages["vs"]["values"], {"c0.0.0": 1, "c0.0.1": 1})
        self.assertEqual(stages["ps"]["stages"], 3)
        self.assertEqual(stages["ps"]["recorded"], 2)
        self.assertEqual(stages["ps"]["float_mode"], 1)
        self.assertEqual(stages["ps"]["ieee"], 1)
        self.assertEqual(stages["ls"]["ieee"], 1)
        self.assertEqual(stages["hs"]["values"], {"c0.1.0": 1})
        self.assertEqual(stages["tes"]["values"], {"c0.0.0": 1})
        self.assertEqual(result["compute"], {"modules": 2, "float_mode": 1, "values": {"c0": 1, "f0": 1}})

    def test_shared_manifest_title(self):
        path = self.write("CUSA00001_01.000.000.shared.kpm", MANIFEST)
        self.assertEqual(kyty_manifest_float_modes.count(path)["title"], "CUSA00001_01.000.000")

    def test_folder_and_json(self):
        self.write("A_1.kpm", MANIFEST)
        (self.folder / "sub").mkdir()
        self.write("sub/B_2.kpm", MANIFEST.replace("fp=f0.1.1", "fp=c0.0.1"))
        out = io.StringIO()
        self.assertEqual(kyty_manifest_float_modes.main([str(self.folder), "--json"], out), 0)
        results = json.loads(out.getvalue())
        self.assertEqual([result["title"] for result in results], ["A_1", "B_2"])
        self.assertEqual([result["stages"]["ps"]["float_mode"] for result in results], [1, 0])

    def test_text(self):
        out = io.StringIO()
        self.assertEqual(kyty_manifest_float_modes.main([str(self.write("A_1.kpm", MANIFEST))], out), 0)
        text = out.getvalue()
        self.assertIn("A_1 (", text)
        self.assertIn("4 graphics pipelines, 3 with float modes", text)
        self.assertIn("c0.0.0 x1, f0.1.1 x1", text)
        self.assertIn("c0 x1, f0 x1 (modules)", text)

    def test_rejects_bad_input(self):
        for name, text in (("bad_header.kpm", "# something else\n"),
                           ("bad_mode.kpm", MANIFEST.replace("fp=f0.1.1", "fp=f0.2.1")),
                           ("short_mode.kpm", MANIFEST.replace("fp=f0.1.1", "fp=f0.1")),
                           ("wide_mode.kpm", MANIFEST.replace("fp=f0.1.1", "fp=100.1.1"))):
            with self.subTest(name=name):
                error = io.StringIO()
                with redirect_stderr(error):
                    code = kyty_manifest_float_modes.main([str(self.write(name, text))], io.StringIO())
                self.assertEqual(code, 2)
                self.assertIn("kyty_manifest_float_modes:", error.getvalue())


if __name__ == "__main__":
    unittest.main()
