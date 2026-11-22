import fnmatch
import re
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
TOUCHES = re.compile(r"touches\(([^)]*)\)", re.S)
LITERAL = re.compile(r"'([^']+)'")
REMOVED = REPOSITORY / ".github/sync-removed-paths.txt"


def removed_rules():
    return [line.strip() for line in REMOVED.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def kept_deleted(path, rules):
    for rule in rules:
        if rule.endswith("/"):
            if path.startswith(rule):
                return True
        elif any(character in rule for character in "*?["):
            if fnmatch.fnmatchcase(path, rule):
                return True
        elif path == rule:
            return True
    return False


class EligibilityPathTests(unittest.TestCase):
    def test_every_path_the_gate_names_exists_in_the_tree(self):
        text = (REPOSITORY / ".github/workflows/build.yml").read_text(encoding="utf-8")
        paths = [path for call in TOUCHES.findall(text) for path in LITERAL.findall(call)]
        self.assertGreater(len(paths), 20)
        rules = removed_rules()
        missing = []
        for path in paths:
            if kept_deleted(path, rules):
                continue
            present = (REPOSITORY / path).is_dir() if path.endswith("/") else (REPOSITORY / path).exists()
            if not present:
                missing.append(path)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
