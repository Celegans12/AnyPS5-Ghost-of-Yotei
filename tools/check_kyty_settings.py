import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nid_names import CPP_TOKEN

ROOT = "core"
DOC = "docs/dev/THIRD_PARTY_KYTY.md"
HEADING = "## Settings of the kyty build"
NAMES = "core/libs/prx/libSceAgcDriver/KytyShim/src/Settings.cpp"
IGNORED_DOC = "docs/dev/ANYKYTY_SETTINGS.md"
IGNORED_HEADING = "## What AnyKyty does not read"
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
                yield match.start(), match.end(), literal.group("text")


def reads(root, skipped=None):
    found = {}
    prefixes = {}
    skipped = Path(skipped).resolve() if skipped is not None else None
    for path in source_files(root):
        if path.resolve() == skipped:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"cannot read {path.as_posix()}: {error}") from error
        if "KYTY_" not in text:
            continue
        for start, end, value in literals(text):
            if NAME.fullmatch(value) is None:
                continue
            site = f"{path.as_posix()}:{text.count(chr(10), 0, start) + 1}"
            if value.endswith("_"):
                if (re.search(r'\.\s*starts_with\s*\(\s*$', text[:start]) is None or
                        re.match(r'\s*\)', text[end:]) is None):
                    prefixes.setdefault(value, []).append(site)
            else:
                found.setdefault(value, []).append(site)
    return found, prefixes


def listed(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read the list of read names {Path(path).as_posix()}: {error}") from error
    names = [value for *_, value in literals(text) if NAME.fullmatch(value) is not None and not value.endswith("_")]
    if not names:
        raise ValueError(f"{Path(path).as_posix()} lists no KYTY_* name")
    return names


def section(doc, heading):
    text = Path(doc).read_text(encoding="utf-8")
    start = text.find("\n" + heading + "\n")
    if start < 0:
        raise ValueError(f"{doc} has no '{heading}' section")
    part = text[start + 1:]
    following = re.search(r"^## ", part[len(heading):], re.M)
    if following is not None:
        part = part[:len(heading) + following.start()]
    return part


def ignored(doc):
    return set(ROW_NAMES.findall(section(doc, IGNORED_HEADING)))


def documented(doc):
    names = set()
    for line in section(doc, HEADING).split("\n"):
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) > 1:
            names.update(ROW_NAMES.findall(cells[1]))
    return names


def check(found, prefixes, names, doc, out, read_list=None, names_file=NAMES, ignored_names=frozenset(),
          ignored_doc=IGNORED_DOC):
    missing = sorted(set(found) - names)
    stale = sorted(names - set(found))
    for name in missing:
        out.write(f"error: {name} is read at {found[name][0]} but has no row in {doc}\n")
    for name in stale:
        out.write(f"error: {name} has a row in {doc} but the code no longer reads it\n")
    for prefix, sites in sorted(prefixes.items()):
        out.write(f"error: {sites[0]} builds an environment name from the literal '{prefix}': "
                  "read each name through a literal of its own so that it can be documented\n")
    unlisted, unread, twice, contradicted = [], [], [], []
    if read_list is not None:
        unlisted = sorted(set(found) - set(read_list))
        unread = sorted(set(read_list) - set(found))
        twice = sorted({name for name in read_list if read_list.count(name) > 1})
        contradicted = sorted(set(read_list) & set(ignored_names))
    for name in unlisted:
        out.write(f"error: {name} is read at {found[name][0]} but is not in the list of read names in {names_file}\n")
    for name in unread:
        out.write(f"error: {name} is in the list of read names in {names_file} but the code no longer reads it\n")
    for name in twice:
        out.write(f"error: {name} is in the list of read names in {names_file} more than once\n")
    for name in contradicted:
        out.write(f"error: {name} is in the list of read names in {names_file} but {ignored_doc} says AnyKyty "
                  "does not read it\n")
    mismatches = len(unlisted) + len(unread) + len(twice) + len(contradicted)
    if missing or stale or prefixes or mismatches:
        out.write(f"{len(missing)} setting(s) undocumented, {len(stale)} stale, {len(prefixes)} built at run time, "
                  f"{mismatches} list mismatch(es)\n")
        return 1
    listed_in = f" and listed in {names_file}" if read_list is not None else ""
    out.write(f"{len(found)} KYTY_* setting(s), all documented in {doc}{listed_in}\n")
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description="fail when a KYTY_* name the code under core/ reads has no row in the settings tables or is not "
                    "in the build's list of read names, or the reverse")
    parser.add_argument("--root", default=ROOT, help="directory to scan (default: %(default)s)")
    parser.add_argument("--doc", default=DOC, help="document that holds the tables (default: %(default)s)")
    parser.add_argument("--names", default=NAMES,
                        help="source file that holds the build's list of read names; the scan skips it (default: %(default)s)")
    parser.add_argument("--ignored-doc", default=IGNORED_DOC,
                        help=f"document whose '{IGNORED_HEADING}' section names what is not read (default: %(default)s)")
    parser.add_argument("--list", action="store_true", help="print each name with its first reading site and exit")
    args = parser.parse_args(argv)
    if not Path(args.root).is_dir():
        print(f"error: scan root {args.root} is not a directory", file=sys.stderr)
        return 1
    try:
        found, prefixes = reads(args.root, args.names)
        if args.list:
            for name in sorted(found):
                print(f"{name}\t{found[name][0]}")
            return 0
        names = documented(args.doc)
        read_list = listed(args.names)
        ignored_names = ignored(args.ignored_doc)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return check(found, prefixes, names, args.doc, sys.stdout, read_list, Path(args.names).as_posix(), ignored_names,
                 Path(args.ignored_doc).as_posix())


if __name__ == "__main__":
    sys.exit(main())
