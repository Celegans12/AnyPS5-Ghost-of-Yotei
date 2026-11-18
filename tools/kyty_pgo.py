#!/usr/bin/env python3
"""Profile of the Kyty core for profile-guided optimization (docs/dev/KYTY_PGO.md).

  kyty_pgo.py merge [--out core/kyty/pgo/kyty_gpu.proftext] [--note TEXT] [--llvm-profdata PATH] <.profraw files or directories>...
  kyty_pgo.py check [<profile>]

merge: merges the raw profiles an instrumented build wrote (-DANYPS5_KYTY_PGO_GENERATE=ON, LLVM_PROFILE_FILE) into one sparse
text profile, the form the repository keeps, headed by the commit, the date, the note and one fingerprint per Kyty source.
check: lists the Kyty sources that changed since the profile was trained, as the configure step does; exits 1 when any did.
"""

import argparse
import datetime
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
KYTY = ROOT / "core" / "kyty"
DEFAULT_PROFILE = KYTY / "pgo" / "kyty_gpu.proftext"
EXTRA_SOURCES = (
    "core/libs/prx/libSceAgcDriver/KytyShim/src/MemoryBacking.cpp",
    "core/libs/prx/libSceAgcDriver/KytyShim/src/MemoryGpu.cpp",
)
SOURCE_LINE = re.compile(r"^# kyty-pgo-source ([0-9a-f]+) (.+)$")
TRAINED_LINE = re.compile(r"^# kyty-pgo-trained (.*)$")


def sources():
    """Same list as kyty_pgo_source_list in core/kyty/cmake/KytyPgo.cmake."""
    found = []
    for directory in ("graphics", "common"):
        for path in (KYTY / directory).rglob("*"):
            if path.is_file() and path.suffix in (".cpp", ".h", ".inc"):
                found.append(path.relative_to(ROOT).as_posix())
    found.extend(EXTRA_SOURCES)
    return sorted(found)


def fingerprint(relative):
    """SHA-1 of the file with its carriage returns removed, as string(SHA1) after string(REPLACE "\\r" "") in CMake."""
    return hashlib.sha1((ROOT / relative).read_bytes().replace(b"\r", b"")).hexdigest()


def find_profdata(explicit):
    if explicit:
        return explicit
    for compiler in ("clang++", "clang"):
        if shutil.which(compiler):
            path = subprocess.run([compiler, "-print-prog-name=llvm-profdata"], capture_output=True, text=True).stdout.strip()
            if os.path.isabs(path) and os.path.exists(path):
                return path
    path = shutil.which("llvm-profdata")
    if path:
        return path
    sys.exit("kyty_pgo: no llvm-profdata beside clang or on the path (install llvm-<clang major>, or pass --llvm-profdata)")


def raw_profiles(inputs):
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.profraw") if p.stat().st_size > 0))
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f"kyty_pgo: {item} does not exist")
    if not files:
        sys.exit("kyty_pgo: no .profraw file to merge")
    return files


def commit():
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:
        return "unknown"
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--", "core/kyty/graphics", "core/kyty/common", *EXTRA_SOURCES],
                           capture_output=True, text=True).stdout.strip()
    return result.stdout.strip() + ("+changes" if dirty else "")


def merge(args):
    profdata = find_profdata(args.llvm_profdata)
    files = raw_profiles(args.inputs)
    out = Path(args.out)
    with tempfile.TemporaryDirectory() as temporary:
        text = Path(temporary) / "merged.proftext"
        command = [profdata, "merge", "--instr", "--sparse", "--text", "-o", str(text), *map(str, files)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"kyty_pgo: {' '.join(command[:6])} failed:\n{result.stderr}")
        body = text.read_text(encoding="utf-8")
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    note = " ".join(args.note.split())
    header = [f"# kyty-pgo-trained {commit()} {date}{(' ' + note) if note else ''}"]
    header.extend(f"# kyty-pgo-source {fingerprint(source)} {source}" for source in sources())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + "\n" + body, encoding="utf-8", newline="\n")
    functions = sum(1 for line in body.splitlines() if line == "# Func Hash:")
    print(f"kyty_pgo: {len(files)} raw profiles -> {out} ({functions} functions with counts, {out.stat().st_size / 1048576:.1f} MiB)")


def check(args):
    profile = Path(args.profile)
    trained = "unknown"
    known = {}
    with profile.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# kyty-pgo-"):
                if line.startswith(":") or not line.startswith("#"):
                    break
                continue
            line = line.rstrip("\n")
            if match := TRAINED_LINE.match(line):
                trained = match.group(1)
            elif match := SOURCE_LINE.match(line):
                known[match.group(2)] = match.group(1)
    if not known:
        sys.exit(f"kyty_pgo: {profile} carries no source fingerprints")
    current = sources()
    changed = [source for source in current if source in known and fingerprint(source) != known[source]]
    added = [source for source in current if source not in known]
    removed = sorted(set(known) - set(current))
    print(f"kyty_pgo: {profile} trained {trained}: {len(changed)} of {len(current)} sources changed, {len(added)} new, {len(removed)} removed")
    for label, items in (("changed", changed), ("new", added), ("removed", removed)):
        for item in items:
            print(f"  {label:8s} {item}")
    return 1 if changed or added or removed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    merging = commands.add_parser("merge", help="merge raw profiles into the text profile")
    merging.add_argument("inputs", nargs="+")
    merging.add_argument("--out", default=str(DEFAULT_PROFILE))
    merging.add_argument("--note", default="", help="what trained the profile, one line")
    merging.add_argument("--llvm-profdata", default=os.environ.get("KYTY_LLVM_PROFDATA", ""))
    checking = commands.add_parser("check", help="list the sources that changed since the profile was trained")
    checking.add_argument("profile", nargs="?", default=str(DEFAULT_PROFILE))
    args = parser.parse_args()
    if args.command == "merge":
        merge(args)
        return 0
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
