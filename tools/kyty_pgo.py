#!/usr/bin/env python3
"""Profile of the Kyty core for profile-guided optimization (docs/dev/KYTY_PGO.md).

  kyty_pgo.py merge [--out core/kyty/pgo/kyty_gpu.proftext] [--note TEXT] [--llvm-profdata PATH] <.profraw files or directories>...
  kyty_pgo.py check [<profile>]
  kyty_pgo.py report [--profile <profile>] [--top N] [--share PERCENT] [--summary FILE] [--annotate] <build log>...

merge: merges the raw profiles an instrumented build wrote (-DANYPS5_KYTY_PGO_GENERATE=ON, LLVM_PROFILE_FILE) into one sparse
text profile, the form the repository keeps, headed by the commit, the date, the note and one fingerprint per Kyty source.
check: lists the Kyty sources that changed since the profile was trained, as the configure step does; exits 1 when any did.
report: reads the log of a build that used the profile, lists the functions Clang compiled without their counts ("function control
flow change detected"), with their peak counter weight and rank in the profile and Clang's separate discarded count, as Markdown
on the standard output (and appended to --summary, the job summary of CI). When one of the --top hottest functions is among them (10 by default),
or when together they hold --share percent of the profile's weight (1 by default), --annotate adds a GitHub warning annotation. Always exits 0 once the log is read: a stale profile costs speed, never the build.
"""

import argparse
import bisect
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
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOST_LINE = re.compile(
    r"(?:(?P<file>\S+): )?function (?:control flow|basic block count) change detected \((?:hash|counter) mismatch\) "
    r"(?P<name>\S+) Hash = \d+(?: up to (?P<count>\d+) count discarded)?"
)
MAX_PROFILE_COUNT = (1 << 64) - 1
SUMMARY_ROWS = 25


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


def read_header(profile):
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
    return trained, known


def stale_sources(known):
    current = sources()
    changed = [source for source in current if source in known and fingerprint(source) != known[source]]
    added = [source for source in current if source not in known]
    removed = sorted(set(known) - set(current))
    return current, changed, added, removed


def check(args):
    profile = Path(args.profile)
    trained, known = read_header(profile)
    current, changed, added, removed = stale_sources(known)
    print(f"kyty_pgo: {profile} trained {trained}: {len(changed)} of {len(current)} sources changed, {len(added)} new, {len(removed)} removed")
    for label, items in (("changed", changed), ("new", added), ("removed", removed)):
        for item in items:
            print(f"  {label:8s} {item}")
    return 1 if changed or added or removed else 0


def profile_weights(profile):
    lines = [line.strip() for line in profile.read_text(encoding="utf-8").splitlines()]
    weights = {}
    for index, line in enumerate(lines[1:-4], start=1):
        if line != "# Func Hash:":
            continue
        name, _, func_hash, count_header, count_text, values_header = lines[index - 1:index + 5]
        if (not name or not func_hash.isdecimal() or count_header != "# Num Counters:"
                or not count_text.isdecimal() or values_header != "# Counter Values:"):
            continue
        count = int(count_text)
        counter_lines = lines[index + 5:index + 5 + count]
        if not counter_lines or len(counter_lines) != count or not all(value.isdecimal() for value in counter_lines):
            continue
        counters = list(map(int, counter_lines))
        if counters[0] in (MAX_PROFILE_COUNT, MAX_PROFILE_COUNT - 1):
            continue
        peak = max((value for value in counters if value < MAX_PROFILE_COUNT), default=0)
        weights[name] = max(weights.get(name, 0), peak)
    return weights


def lost_functions(logs):
    found = {}
    for log in logs:
        with open(log, encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                match = LOST_LINE.search(ANSI.sub("", raw))
                if match:
                    file = Path((match["file"] or "").replace("\\", "/")).name
                    key = match["name"], file
                    if found.get(key) is None:
                        found[key] = int(match["count"]) if match["count"] else None
    return found


def demangle(names):
    tool = shutil.which("c++filt") or shutil.which("llvm-cxxfilt")
    if not tool or not names:
        return {name: name for name in names}
    result = subprocess.run([tool], input="\n".join(names) + "\n", capture_output=True, text=True)
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != len(names):
        return {name: name for name in names}
    return dict(zip(names, lines))


def shorten(text, width=110):
    return text if len(text) <= width else text[: width - 1] + "…"


def escape_property(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def report(args):
    profile = Path(args.profile)
    trained, known = read_header(profile)
    _, changed, added, removed = stale_sources(known)
    weights = profile_weights(profile)
    ordered = sorted(weights.values())
    total = sum(ordered) or 1
    lost = lost_functions(args.logs)
    names = demangle(sorted({name for name, _ in lost}))
    rows = []
    seen = set()
    for (name, file), count in lost.items():
        keys = [key for key in (name, f"{file};{name}") if key in weights]
        if len(keys) != 1:
            rows.append((None, None, names[name], file, count))
            continue
        key = keys[0]
        if key in seen:
            continue
        seen.add(key)
        weight = weights[key]
        rank = len(ordered) - bisect.bisect_right(ordered, weight) + 1
        rows.append((rank, weight, names[name], file, count))
    rows.sort(key=lambda row: (row[0] is None, row[0], row[2]))
    ranked = [row for row in rows if row[0] is not None]
    lost_weight = sum(row[1] for row in ranked)
    hot = [row for row in ranked if row[1] > 0 and row[0] <= args.top]
    lost_share = 100.0 * lost_weight / total
    reasons = []
    if hot:
        reasons.append(f"{len(hot)} of the {args.top} hottest functions lost their counts")
    if lost_weight > 0 and lost_share >= args.share:
        reasons.append(f"the functions that lost their counts hold {lost_share:.2f} % of the profile's weight (limit {args.share:g} %)")
    stale = bool(reasons)
    lines = ["### Kyty PGO profile", ""]
    lines.append(f"Trained on `{shorten(trained, 140)}`; {len(changed)} of the Kyty sources changed since, {len(added)} new, {len(removed)} removed.")
    if not rows:
        if changed or added:
            lines.append("")
            lines.append("No function lost its counts in this build log. Either the changes kept every function's control flow, or the core was not compiled in this job (compiler cache): the log cannot tell which.")
        else:
            lines.append("")
            lines.append("Every function of the profile matches its code.")
    else:
        lines.append("")
        lines.append(f"**{len(rows)} functions lost their counts** and were compiled without the profile, {lost_share:.2f} % of the profile's weight "
                     f"(weight = highest counter of the function; rank among the {len(ordered)} functions the profile counts).")
        if len(ranked) != len(rows):
            lines.append(f"{len(rows) - len(ranked)} losses have no unambiguous profile weight.")
        if stale:
            lines.append("")
            lines.append(f"> [!WARNING]\n> {'; '.join(reasons)}. The build is not affected, but it is slower than the recorded profile promises: "
                         "record the profile again after this series of changes (docs/dev/KYTY_PGO.md, \"Recording the profile\").")
        lines.append("")
        lines.append("| Rank | Weight | Share | Clang discarded count | Function | File |")
        lines.append("|---:|---:|---:|---:|---|---|")
        for rank, weight, demangled, file, count in rows[:SUMMARY_ROWS]:
            stats = f"{rank} | {weight:,} | {100.0 * weight / total:.2f} %" if rank is not None else "— | — | —"
            discarded = f"{count:,}" if count is not None else "—"
            function = shorten(demangled).replace("|", "\\|")
            lines.append(f"| {stats} | {discarded} | `{function}` | {file} |")
        if len(rows) > SUMMARY_ROWS:
            rest = rows[SUMMARY_ROWS:]
            lines.append("")
            ranked_rest = ranked[SUMMARY_ROWS:]
            rest_share = 100.0 * sum(row[1] for row in ranked_rest) / total
            lines.append(f"and {len(rest)} more ({len(ranked_rest)} ranked, together {rest_share:.2f} % of the weight).")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    if args.annotate and stale:
        shown = ", ".join(f"{shorten(row[2], 70)} (rank {row[0]})" for row in ranked[:4])
        message = (f"{'; '.join(reasons)}: {shown}. The build is not affected, but it is slower than the recorded profile promises. "
                   "Record the profile again after this series of changes (docs/dev/KYTY_PGO.md).")
        print(f"::warning file={escape_property(profile.relative_to(ROOT).as_posix()) if profile.is_relative_to(ROOT) else escape_property(str(profile))},"
              f"title=Kyty PGO profile is stale on the hot path::{escape_property(message)}")
    return 0


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
    reporting = commands.add_parser("report", help="list the hot functions that lost their counts in a build log")
    reporting.add_argument("logs", nargs="+")
    reporting.add_argument("--profile", default=str(DEFAULT_PROFILE))
    reporting.add_argument("--top", type=int, default=10, help="warn when one of this many hottest functions lost its counts")
    reporting.add_argument("--share", type=float, default=1.0, help="warn when the functions that lost their counts hold this percent of the profile's weight")
    reporting.add_argument("--summary", default="", help="file the Markdown is appended to (GITHUB_STEP_SUMMARY)")
    reporting.add_argument("--annotate", action="store_true", help="print a GitHub warning annotation when a hot function lost its counts")
    args = parser.parse_args()
    if args.command == "merge":
        merge(args)
        return 0
    if args.command == "report":
        return report(args)
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
