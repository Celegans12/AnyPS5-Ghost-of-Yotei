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

LIST = """constexpr std::string_view Prefix = "KYTY_";
constexpr std::string_view Names[] = {"KYTY_ALPHA", "KYTY_BETA", "KYTY_DELTA", "KYTY_GAMMA"};
"""

IGNORED = """# Settings

## What AnyKyty does not read

AnyKyty does not read `KYTY_FULLSCREEN` or `KYTY_VR`.

## Other

`KYTY_ALPHA` is read.
"""


class CheckKytySettingsTests(unittest.TestCase):
    def tree(self, files, doc=DOC, names=LIST, ignored=IGNORED):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for name, text in {**files, "libs/Settings.cpp": names}.items():
            path = root / "core" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (root / "doc.md").write_text(doc, encoding="utf-8")
        (root / "ignored.md").write_text(ignored, encoding="utf-8")
        return root

    def arguments(self, root):
        return ["--root", str(root / "core"), "--doc", str(root / "doc.md"), "--names",
                str(root / "core" / "libs" / "Settings.cpp"), "--ignored-doc", str(root / "ignored.md")]

    def run_check(self, root):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            status = check_kyty_settings.main(self.arguments(root))
        return status, out.getvalue()

    def test_reads_by_any_function_are_found(self):
        root = self.tree({"a.cpp": CODE})
        found, prefixes = check_kyty_settings.reads(root / "core", root / "core" / "libs" / "Settings.cpp")
        self.assertEqual(sorted(found), ["KYTY_ALPHA", "KYTY_BETA", "KYTY_DELTA", "KYTY_GAMMA"])
        self.assertEqual(prefixes, {})

    def test_list_holds_the_names_and_not_the_prefix(self):
        names = check_kyty_settings.listed(self.tree({}) / "core" / "libs" / "Settings.cpp")
        self.assertEqual(names, ["KYTY_ALPHA", "KYTY_BETA", "KYTY_DELTA", "KYTY_GAMMA"])

    def test_read_name_missing_from_the_list_fails(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE}, names=LIST.replace('"KYTY_BETA", ', "")))
        self.assertEqual(status, 1)
        self.assertIn("KYTY_BETA is read at", text)
        self.assertIn("is not in the list of read names", text)
        self.assertIn("1 list mismatch(es)", text)

    def test_listed_name_the_code_does_not_read_fails(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE}, names=LIST.replace('"KYTY_GAMMA"', '"KYTY_GAMMA", "KYTY_FLAT_DEFER"')))
        self.assertEqual(status, 1)
        self.assertIn("KYTY_FLAT_DEFER is in the list of read names", text)
        self.assertIn("the code no longer reads it", text)

    def test_name_listed_twice_fails(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE}, names=LIST.replace('"KYTY_BETA"', '"KYTY_BETA", "KYTY_BETA"')))
        self.assertEqual(status, 1)
        self.assertIn("KYTY_BETA is in the list of read names", text)
        self.assertIn("more than once", text)

    def test_name_documented_as_not_read_must_not_be_read(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE}, ignored=IGNORED.replace("KYTY_VR", "KYTY_GAMMA")))
        self.assertEqual(status, 1)
        self.assertIn("KYTY_GAMMA is in the list of read names", text)
        self.assertIn("says AnyKyty does not read it", text)

    def test_missing_list_is_an_error(self):
        root = self.tree({"a.cpp": CODE})
        (root / "core" / "libs" / "Settings.cpp").unlink()
        status, text = self.run_check(root)
        self.assertEqual(status, 1)
        self.assertIn("cannot read the list of read names", text)

    def test_missing_ignored_section_is_an_error(self):
        status, text = self.run_check(self.tree({"a.cpp": CODE}, ignored="# Nothing\n"))
        self.assertEqual(status, 1)
        self.assertIn("has no '## What AnyKyty does not read' section", text)

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
            status = check_kyty_settings.main(self.arguments(root) + ["--list"])
        self.assertEqual(status, 0)
        self.assertEqual([line.split("\t")[0] for line in out.getvalue().splitlines()],
                         ["KYTY_ALPHA", "KYTY_BETA", "KYTY_DELTA", "KYTY_GAMMA"])


if __name__ == "__main__":
    unittest.main()
