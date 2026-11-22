import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_kyty_settings

DOC = """# Kyty code in AnyPS5

## Settings of the kyty build

| Variable | Values |
|---|---|
| `KYTY_ALPHA` | `0` or `1` |
| `KYTY_BETA`, `KYTY_GAMMA` | number |
| `KYTY_DELTA` | path |

Prose that names `KYTY_PROSE` is not a row.

## Updating

| `KYTY_AFTER` | a row of another section |
"""

CODE = """#include <cstdlib>
static bool a = std::getenv("KYTY_ALPHA") != nullptr;
static bool b = Switch("KYTY_BETA");
static const Flag Flags[] = {{"KYTY_GAMMA", &Options::gamma}};
static int d = load("KYTY_DELTA", value);
static const char* e = u8"KYTY_DELTA";
static const char* f = "KYTY_ALPHA must be 0 or 1";
// std::getenv("KYTY_COMMENTED")
static const char* g = "not KYTY_INSIDE";
static const char* h = std::getenv("PATH");
static const char* i = R"(KYTY_RAW)";
"""


class CheckKytySettingsTests(unittest.TestCase):
    def tree(self, files, doc=DOC):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for name, text in files.items():
            path = root / "core" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (root / "doc.md").write_text(doc, encoding="utf-8")
        return root

    def run_check(self, root):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            status = check_kyty_settings.main(["--root", str(root / "core"), "--doc", str(root / "doc.md")])
        return status, out.getvalue()

    def test_reads_by_any_function_are_found(self):
        found, prefixes = check_kyty_settings.reads(self.tree({"a.cpp": CODE}) / "core")
        self.assertEqual(sorted(found), ["KYTY_ALPHA", "KYTY_BETA", "KYTY_DELTA", "KYTY_GAMMA"])
        self.assertEqual(prefixes, {})

    def test_documented_tree_passes(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE}))
        self.assertEqual(status, 0, text)
        self.assertIn("4 KYTY_* setting(s)", text)

    def test_missing_row_fails_with_the_site(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE + 'static int j = Helper("KYTY_NEW");\n'}))
        self.assertEqual(status, 1)
        self.assertIn("KYTY_NEW is read at", text)
        self.assertIn("a.cpp:12", text)
        self.assertIn("1 setting(s) undocumented", text)

    def test_stale_row_fails(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE.replace('static int d = load("KYTY_DELTA", value);\n', "").replace(
            'static const char* e = u8"KYTY_DELTA";\n', "")}))
        self.assertEqual(status, 1)
        self.assertIn("KYTY_DELTA has a row", text)
        self.assertIn("1 stale", text)

    def test_names_outside_the_section_and_prose_are_not_rows(self):
        names = check_kyty_settings.documented(self.tree({}) / "doc.md")
        self.assertEqual(names, {"KYTY_ALPHA", "KYTY_BETA", "KYTY_GAMMA", "KYTY_DELTA"})

    def test_name_built_from_a_prefix_fails(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE + 'static auto k = std::string("KYTY_") + name;\n'}))
        self.assertEqual(status, 1)
        self.assertIn("builds an environment name from the literal 'KYTY_'", text)

    def test_prefix_checks_do_not_read_or_build_a_setting(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE + 'bool allowed = name.starts_with("KYTY_");\n'}))
        self.assertEqual(status, 0, text)
        self.assertIn("4 KYTY_* setting(s)", text)

    def test_prefix_check_argument_built_from_a_prefix_fails(self):
        for argument in ('"KYTY_" + suffix', '"KYTY_" "ALPHA"'):
            with self.subTest(argument=argument):
                code = CODE + f'bool allowed = name.starts_with({argument});\n'
                status, text = self.run_check(self.tree({"a.cpp": code}))
                self.assertEqual(status, 1, text)
                self.assertIn("builds an environment name from the literal 'KYTY_'", text)

    def test_tests_and_third_party_directories_are_skipped(self):
        files = {"a.cpp": CODE, "libs/tests/t.cpp": 'x("KYTY_ONLY_IN_TESTS");\n', "3rdparty/z.cpp": 'x("KYTY_THIRD");\n',
                 "libs/UnitTests.cpp": 'x("KYTY_UNIT");\n', "libs/Table.inc": 'x("KYTY_DELTA");\n'}
        status, text = self.run_check(self.tree(files))
        self.assertEqual(status, 0, text)

    def test_missing_section_is_an_error(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE}, doc="# Nothing\n"))
        self.assertEqual(status, 1)
        self.assertIn("has no '## Settings of the kyty build' section", text)

    def test_missing_root_is_an_error(self):
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            status = check_kyty_settings.main(["--root", "no-such-directory"])
        self.assertEqual(status, 1)

    def test_list_prints_each_name_once(self):
        out = io.StringIO()
        root = self.tree({"a.cpp": CODE})
        with contextlib.redirect_stdout(out):
            status = check_kyty_settings.main(["--root", str(root / "core"), "--doc", str(root / "doc.md"), "--list"])
        self.assertEqual(status, 0)
        self.assertEqual([line.split("\t")[0] for line in out.getvalue().splitlines()],
                         ["KYTY_ALPHA", "KYTY_BETA", "KYTY_DELTA", "KYTY_GAMMA"])


if __name__ == "__main__":
    unittest.main()
