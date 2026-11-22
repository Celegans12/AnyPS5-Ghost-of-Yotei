import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

RULES = ".github/sync-removed-paths.txt"
FRAGMENTS = "docs/dev/debt/changes/"
DEBT_FOLD = "tools/debt_fold.py"
RESOLVE_OURS = ".github/sync-keep-ours-paths.txt"


def git(*args, check=True, text=True, stdin=None):
    result = subprocess.run(["git", *args], capture_output=True, text=text, input=stdin)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def load_rules(path):
    rules = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rules.append(line)
    return rules


def matches(rules, path):
    for rule in rules:
        if rule.endswith("/"):
            if path.startswith(rule):
                return True
        elif any(char in rule for char in "*?["):
            if fnmatch.fnmatchcase(path, rule):
                return True
        elif path == rule:
            return True
    return False


def head_files():
    return set(git("ls-tree", "-r", "--name-only", "-z", "HEAD").stdout.split("\0"))


def unmerged():
    stages = {}
    for entry in git("ls-files", "-u", "-z").stdout.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        stage = int(meta.split()[2])
        stages.setdefault(path, set()).add(stage)
    return stages


def staged_additions():
    names = git("diff", "--cached", "--name-only", "--diff-filter=A", "-z", "HEAD").stdout.split("\0")
    return [name for name in names if name]


def remove(paths):
    for path in paths:
        git("rm", "-q", "-f", "--", path)


def resolve_removed(rules):
    present = head_files()
    removed = []
    for path, stages in sorted(unmerged().items()):
        if path in present or not matches(rules, path):
            continue
        if 2 in stages:
            continue
        removed.append(path)
    remove(removed)
    added = [path for path in staged_additions() if path not in present and matches(rules, path)]
    remove(added)
    return removed, added


def keep_ours(paths):
    resolved = []
    stages = unmerged()
    for path in paths:
        if path not in stages or not {1, 2, 3} <= stages[path]:
            continue
        base, ours, theirs = (git("show", f":{stage}:{path}", text=False).stdout for stage in (1, 2, 3))
        folder = Path(".git-sync-keep-ours")
        folder.mkdir(exist_ok=True)
        try:
            for name, data in (("base", base), ("ours", ours), ("theirs", theirs)):
                (folder / name).write_bytes(data)
            subprocess.run(["git", "merge-file", "--ours", str(folder / "ours"), str(folder / "base"), str(folder / "theirs")],
                           capture_output=True)
            Path(path).write_bytes((folder / "ours").read_bytes())
        finally:
            for child in folder.iterdir():
                child.unlink()
            folder.rmdir()
        git("add", "--", path)
        resolved.append(path)
    return resolved


def prune_fragments(present):
    removed = []
    for _ in range(1000):
        result = subprocess.run([sys.executable, "-I", DEBT_FOLD, "--check"], capture_output=True, text=True)
        findings = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
        target = None
        for line in findings:
            name = line.split(":", 1)[0]
            if name.startswith(FRAGMENTS) and name not in present and Path(name).is_file():
                target = name
                break
        if target is None:
            break
        remove([target])
        removed.append(target)
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="finish a merge of upstream by keeping the deletion of the paths this fork removed")
    parser.add_argument("--rules", default=RULES, help="removed paths, one per line (default: %(default)s)")
    parser.add_argument("--keep-ours", default=RESOLVE_OURS,
                        help="files whose conflicting hunks only concern removed code, resolved to our side (default: %(default)s)")
    parser.add_argument("--prune-fragments", action="store_true",
                        help="also remove debt fragments added by the merge that no longer apply")
    args = parser.parse_args(argv)
    rules = load_rules(args.rules)
    removed, added = resolve_removed(rules)
    ours = keep_ours(load_rules(args.keep_ours)) if Path(args.keep_ours).is_file() else []
    pruned = prune_fragments(head_files()) if args.prune_fragments else []
    for label, paths in (("deleted by us, modified by upstream: kept deleted", removed),
                         ("added by upstream under a removed path: dropped", added),
                         ("conflicting hunks resolved to our side", ours),
                         ("debt fragments that no longer apply: dropped", pruned)):
        for path in paths:
            print(f"{label}: {path}")
    remaining = sorted(unmerged())
    for path in remaining:
        print(f"conflict: {path}")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
