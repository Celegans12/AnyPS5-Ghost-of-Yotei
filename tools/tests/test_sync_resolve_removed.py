import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "sync_resolve_removed.py"
RULES = "# removed with the driver\nlegacy/\nlib/Old.cpp\n"
KEEP = "docs/ENV.md\n"
LINES = "".join(f"line {number}\n" for number in range(1, 41))


def git(root, *args, check=True):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *args],
                          cwd=root, capture_output=True, text=True, check=check)


def write(root, path, text):
    target = Path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def commit(root, message):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


class SyncResolveRemovedTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="anyps5-sync-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        git(self.root, "init", "-q", "-b", "main")
        write(self.root, ".github/sync-removed-paths.txt", RULES)
        write(self.root, ".github/sync-keep-ours-paths.txt", KEEP)
        for path in ("legacy/a.cpp", "legacy/keep.hpp", "lib/Old.cpp", "lib/Shared.cpp", "docs/ENV.md", "other/Real.cpp"):
            write(self.root, path, LINES)
        commit(self.root, "base")
        git(self.root, "checkout", "-q", "-b", "upstream")
        write(self.root, "legacy/a.cpp", LINES.replace("line 3\n", "line 3 upstream\n"))
        write(self.root, "legacy/keep.hpp", LINES.replace("line 30\n", "line 30 upstream\n"))
        write(self.root, "lib/Old.cpp", LINES.replace("line 3\n", "line 3 upstream\n"))
        write(self.root, "lib/Shared.cpp", LINES.replace("line 3\n", "line 3 upstream\n"))
        write(self.root, "legacy/new.cpp", LINES)
        write(self.root, "other/Real.cpp", LINES.replace("line 20\n", "line 20 upstream\n"))
        write(self.root, "docs/ENV.md", LINES.replace("line 5\n", "line 5 upstream\n").replace("line 35\n", "line 35 upstream\n"))
        commit(self.root, "upstream")
        git(self.root, "checkout", "-q", "main")
        git(self.root, "rm", "-q", "legacy/a.cpp", "lib/Old.cpp")
        write(self.root, "other/Real.cpp", LINES.replace("line 20\n", "line 20 ours\n"))
        write(self.root, "docs/ENV.md", LINES.replace("line 5\n", "line 5 ours\n"))
        commit(self.root, "ours")

    def resolve(self, *arguments):
        git(self.root, "merge", "--no-commit", "--no-ff", "upstream", check=False)
        return subprocess.run([sys.executable, "-I", str(TOOL), *arguments], cwd=self.root, capture_output=True, text=True)

    def test_a_modified_removed_file_stays_deleted(self):
        result = self.resolve()
        self.assertIn("kept deleted: legacy/a.cpp", result.stdout)
        self.assertIn("kept deleted: lib/Old.cpp", result.stdout)
        self.assertFalse(Path(self.root, "legacy/a.cpp").exists())
        self.assertFalse(Path(self.root, "lib/Old.cpp").exists())

    def test_a_file_added_under_a_removed_folder_is_dropped(self):
        result = self.resolve()
        self.assertIn("dropped: legacy/new.cpp", result.stdout)
        self.assertFalse(Path(self.root, "legacy/new.cpp").exists())

    def test_a_kept_file_under_a_removed_folder_takes_the_upstream_change(self):
        self.resolve()
        self.assertIn("line 30 upstream", Path(self.root, "legacy/keep.hpp").read_text(encoding="utf-8"))

    def test_a_shared_file_merges_normally(self):
        self.resolve()
        self.assertIn("line 3 upstream", Path(self.root, "lib/Shared.cpp").read_text(encoding="utf-8"))

    def test_a_real_conflict_stays_and_fails_the_run(self):
        result = self.resolve()
        self.assertEqual(result.returncode, 1)
        self.assertIn("conflict: other/Real.cpp", result.stdout)
        self.assertNotIn("conflict: legacy", result.stdout)
        self.assertNotIn("conflict: lib/Old.cpp", result.stdout)

    def test_conflicting_hunks_of_a_listed_file_go_to_our_side_and_keep_the_rest(self):
        result = self.resolve()
        text = Path(self.root, "docs/ENV.md").read_text(encoding="utf-8")
        self.assertIn("line 5 ours", text)
        self.assertNotIn("line 5 upstream", text)
        self.assertIn("line 35 upstream", text)
        self.assertNotIn("conflict: docs/ENV.md", result.stdout)

    def test_without_a_real_conflict_the_run_succeeds(self):
        git(self.root, "checkout", "-q", "-b", "clean", "main~1")
        git(self.root, "rm", "-q", "legacy/a.cpp", "lib/Old.cpp")
        commit(self.root, "ours without other edits")
        git(self.root, "merge", "--no-commit", "--no-ff", "upstream", check=False)
        result = subprocess.run([sys.executable, "-I", str(TOOL)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("conflict:", result.stdout)
        git(self.root, "commit", "-q", "-m", "merge")


if __name__ == "__main__":
    unittest.main()
