import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nid_names import CPP_TOKEN

ROOT = "core"
DOC = "docs/dev/THIRD_PARTY_KYTY.md"
HEADING = "## Settings of the kyty build"
EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".ipp"}
SKIPPED_DIRECTORIES = {"3rdparty", "build", "tests"}
SKIPPED_SUFFIXES = ("Tests.cpp",)
NAME = re.compile(r"KYTY_[A-Z0-9_]*")
LITERAL = re.compile(r'(?:u8|u|U|L)?"(?P<text>(?:\\.|[^"\\])*)"', re.S)
ROW_NAMES = re.compile(r"`(KYTY_[A-Z0-9_]+)`")


def source_files(root):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.suffix not in EXTENSIONS or path.name.endswith(SKIPPED_SUFFIXES):
            continue
        if SKIPPED_DIRECTORIES.intersection(path.relative_to(root).parts):
            continue
        yield path


def literals(text):
    for match in CPP_TOKEN.finditer(text):
        if match.group("string") is not None:
            literal = LITERAL.fullmatch(match.group("string"))
            if literal is not None:
                yield match.start(), literal.group("text")


def reads(root):
    found = {}
    prefixes = {}
    for path in source_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"cannot read {path.as_posix()}: {error}") from error
        if "KYTY_" not in text:
            continue
        for start, value in literals(text):
            if NAME.fullmatch(value) is None:
                continue
            site = f"{path.as_posix()}:{text.count(chr(10), 0, start) + 1}"
            if value.endswith("_"):
                if re.search(r'\.\s*starts_with\s*\(\s*$', text[:start]) is None:
                    prefixes.setdefault(value, []).append(site)
            else:
                found.setdefault(value, []).append(site)
    return found, prefixes


def documented(doc):
    text = Path(doc).read_text(encoding="utf-8")
    start = text.find("\n" + HEADING + "\n")
    if start < 0:
        raise ValueError(f"{doc} has no '{HEADING}' section")
    section = text[start + 1:]
    following = re.search(r"^## ", section[len(HEADING):], re.M)
    if following is not None:
        section = section[:len(HEADING) + following.start()]
    names = set()
    for line in section.split("\n"):
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) > 1:
            names.update(ROW_NAMES.findall(cells[1]))
    return names


def check(found, prefixes, names, doc, out):
    missing = sorted(set(found) - names)
    stale = sorted(names - set(found))
    for name in missing:
        out.write(f"error: {name} is read at {found[name][0]} but has no row in {doc}\n")
    for name in stale:
        out.write(f"error: {name} has a row in {doc} but the code no longer reads it\n")
    for prefix, sites in sorted(prefixes.items()):
        out.write(f"error: {sites[0]} builds an environment name from the literal '{prefix}': "
                  "read each name through a literal of its own so that it can be documented\n")
    if missing or stale or prefixes:
        out.write(f"{len(missing)} setting(s) undocumented, {len(stale)} stale, {len(prefixes)} built at run time\n")
        return 1
    out.write(f"{len(found)} KYTY_* setting(s), all documented in {doc}\n")
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description="fail when a KYTY_* name the code under core/ reads has no row in the settings tables, or the reverse")
    parser.add_argument("--root", default=ROOT, help="directory to scan (default: %(default)s)")
    parser.add_argument("--doc", default=DOC, help="document that holds the tables (default: %(default)s)")
    parser.add_argument("--list", action="store_true", help="print each name with its first reading site and exit")
    args = parser.parse_args(argv)
    if not Path(args.root).is_dir():
        print(f"error: scan root {args.root} is not a directory", file=sys.stderr)
        return 1
    try:
        found, prefixes = reads(args.root)
        if args.list:
            for name in sorted(found):
                print(f"{name}\t{found[name][0]}")
            return 0
        names = documented(args.doc)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return check(found, prefixes, names, args.doc, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
