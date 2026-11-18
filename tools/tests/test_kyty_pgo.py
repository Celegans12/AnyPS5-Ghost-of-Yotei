from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kyty_pgo as tool


def profile_record(name, counters, version=1):
    return (
        f"{name}\n# Func Hash:\n{version}\n# Num Counters:\n{len(counters)}\n"
        "# Counter Values:\n" + "".join(f"{count}\n" for count in counters)
    )


def mismatch(name, discarded=None, file="shader.cpp"):
    count = f" up to {discarded} count discarded" if discarded is not None else ""
    return f"{file}: function control flow change detected (hash mismatch) {name} Hash = 7{count}\n"


class KytyPgoTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def profile(self, body):
        path = self.root / "fixture.proftext"
        source = "core/kyty/common/common.h"
        header = f"# kyty-pgo-trained fixture\n# kyty-pgo-source {tool.fingerprint(source)} {source}\n:ir\n"
        path.write_text(header + body, encoding="utf-8")
        return path

    def report(self, records, *logs):
        profile = self.profile("\n".join(profile_record(name, counts) for name, counts in records))
        paths = []
        for index, log in enumerate(logs):
            path = self.root / f"build{index}.log"
            path.write_text(log, encoding="utf-8")
            paths.append(path)
        summary = self.root / "summary.md"
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-I", str(Path(tool.__file__)), "report", "--annotate",
             "--profile", str(profile), "--summary", str(summary), *map(str, paths)],
            capture_output=True, encoding="utf-8", timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        markdown = result.stdout.partition("\n\n::warning")[0].rstrip("\n") + "\n"
        self.assertEqual(summary.read_text(encoding="utf-8"), markdown)
        return result.stdout

    def row(self, output, name, file="shader.cpp"):
        def cells(line):
            return tuple(cell.strip() for cell in line.split("|")[1:-1])

        lines = output.splitlines()
        columns = cells(next(line for line in lines if line.startswith("| Rank |")))
        self.assertIn("Clang discarded count", columns)
        rows = [cells(line) for line in lines if line.startswith("| ") and line.endswith("|")]
        row, = [row for row in rows if row[-2:] == (f"`{name}`", file)]
        return dict(zip(columns, row, strict=True))

    def test_profile_uses_declared_counters_and_preserves_repeated_peak(self):
        value_sites = (
            "# Num Value Kinds:\n2\n# ValueKind = IPVK_IndirectCallTarget:\n0\n"
            "# NumValueSites:\n1\n2\nrender.cpp;target.llvm.4:95000\n** External Symbol **:97000\n"
            "# ValueKind = IPVK_MemOPSize:\n1\n# NumValueSites:\n1\n1\n512:99000\n"
        )
        path = self.profile(
            profile_record("render.cpp;caller", (3, 19)) + value_sites + "\n"
            + profile_record("render.cpp;caller", (8, 0), version=2) + "\n"
            + profile_record("42", (7, 4)) + "\n"
            + profile_record("pseudo_hot", (2**64 - 1,)) + "\n"
            + profile_record("pseudo_warm", (2**64 - 2,))
        )
        self.assertEqual(tool.profile_weights(path), {"render.cpp;caller": 19, "42": 7})

    def test_discarded_sum_and_duplicate_logs_do_not_inflate_losses(self):
        records = [(f"active_{index}", (120,)) for index in range(11)] + [("slow", (1,))]
        warning = mismatch("slow", 1300)
        output = self.report(records, warning, warning)
        self.assertIn("**1 functions lost their counts**", output)
        self.assertIn("0.08 % of the profile's weight", output)
        self.assertNotIn("::warning", output)
        self.assertEqual(self.row(output, "slow"), {
            "Rank": "12", "Weight": "1", "Share": "0.08 %", "Clang discarded count": "1,300",
            "Function": "`slow`", "File": "shader.cpp",
        })

    def test_genuine_top_ten_loss_warns_below_share_threshold(self):
        records = [(f"active_{index}", (99,)) for index in range(101)] + [("hottest", (100,))]
        output = self.report(records, mismatch("hottest"))
        self.assertIn("0.99 % of the profile's weight", output)
        self.assertIn("1 of the 10 hottest functions lost their counts", output)
        self.assertIn("::warning file=", output)
        row = self.row(output, "hottest")
        self.assertEqual((row["Rank"], row["Weight"], row["Clang discarded count"]), ("1", "100", "—"))

    def test_genuine_combined_share_warns_without_top_ten_loss(self):
        records = [(f"active_{index}", (18,)) for index in range(11)] + [("first", (1,)), ("second", (1,))]
        output = self.report(records, mismatch("first") + mismatch("second"))
        self.assertIn("hold 1.00 % of the profile's weight (limit 1 %)", output)
        self.assertNotIn("hottest functions lost their counts", output)
        self.assertIn("::warning file=", output)
        self.assertEqual(self.row(output, "first")["Rank"], "12")
        self.assertEqual(self.row(output, "second")["Rank"], "12")

    def test_unmatched_function_retains_discarded_count_without_warning(self):
        output = self.report([("active", (20,))], mismatch("missing", 90000))
        self.assertNotIn("::warning", output)
        row = self.row(output, "missing")
        self.assertEqual((row["Rank"], row["Weight"], row["Share"], row["Clang discarded count"]),
                         ("—", "—", "—", "90,000"))

    def test_file_local_functions_are_ranked_separately(self):
        records = [("One.cpp;helper", (15,)), ("Two.cpp;helper", (10,))]
        output = self.report(records, mismatch("helper", 500, "code/One.cpp") + mismatch("helper", 700, "code/Two.cpp"))
        self.assertIn("**2 functions lost their counts**", output)
        one = self.row(output, "helper", "One.cpp")
        two = self.row(output, "helper", "Two.cpp")
        self.assertEqual((one["Rank"], one["Weight"], one["Clang discarded count"]), ("1", "15", "500"))
        self.assertEqual((two["Rank"], two["Weight"], two["Clang discarded count"]), ("2", "10", "700"))

    def test_ambiguous_plain_and_qualified_records_are_unranked(self):
        records = [("helper", (20,)), ("One.cpp;helper", (15,))]
        output = self.report(records, mismatch("helper", 500, "code/One.cpp"))
        self.assertNotIn("::warning", output)
        row = self.row(output, "helper", "One.cpp")
        self.assertEqual((row["Rank"], row["Weight"], row["Share"], row["Clang discarded count"]),
                         ("—", "—", "—", "500"))


if __name__ == "__main__":
    unittest.main()
