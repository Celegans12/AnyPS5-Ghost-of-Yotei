import argparse
import io
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

C_LIKE = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inc", ".inl", ".comp", ".frag", ".vert", ".task", ".mesh", ".geom", ".glsl"}
TREES = ("graphics", "common")
SKIP = {"common/CMakeLists.txt"}
CONTRACT = (
    "kernel/eventQueue.h",
    "kernel/fileSystem.h",
    "kernel/memory.h",
    "kernel/pthread.h",
    "libs/agc.h",
    "libs/controller.h",
    "libs/errno.h",
    "libs/hmd2.h",
    "libs/libs.h",
    "loader/demonsSoulsPageFree.h",
    "loader/demonsSoulsTitle.h",
    "loader/guestOodle.h",
    "loader/systemContent.h",
    "loader/timer.h",
)
SCRIPTS = ("embed_spirv.cmake", "cmake_config.h.in", "kytyGitVersion.h.in")
KYTY_DIR = PurePosixPath("core/kyty")
CONTRACT_DIR = PurePosixPath("core/libs/prx/libSceAgcDriver/KytyShim/include")
RAW_PREFIXES = {"R", "u8R", "uR", "UR", "LR"}
STRING_PREFIXES = {"u8", "u", "U", "L"}
LICENSE_MARKER = re.compile(
    r"SPDX-License-Identifier|Copyright\s*(?:\([cC]\)|©|\d{4})|Permission is hereby granted"
    r"|Permission to use,? copy,? modify|Redistribution and use in source and binary forms",
    re.I)
MARK = "\x00"
WORD = re.compile(r"[A-Za-z0-9_]")
NO_SPACE_NEIGHBOURS = set("()[]{},;")


class StripError(Exception):
    pass


class Stripper:
    def __init__(self, text):
        self.s = text
        self.n = len(text)
        self.out = []
        self.removed = 0
        self.continued_line_comments = []

    def last_char(self):
        for chunk in reversed(self.out):
            stripped = chunk.replace(MARK, "")
            if stripped:
                return stripped[-1]
        return "\n"

    def line_of(self, index):
        return self.s.count("\n", 0, index) + 1

    def line_comment_end(self, i):
        s = self.s
        j = i + 2
        while True:
            k = s.find("\n", j)
            if k < 0:
                return self.n
            back = k - 1
            if back >= 0 and s[back] == "\r":
                back -= 1
            while back >= j and s[back] in " \t":
                back -= 1
            if back >= j and s[back] == "\\":
                self.continued_line_comments.append(self.line_of(i))
                j = k + 1
                continue
            return k

    def block_replacement(self, nxt):
        prev = self.last_char()
        if prev in " \t\n\r" or nxt in " \t\n\r" or nxt == "":
            return ""
        if prev in NO_SPACE_NEIGHBOURS or nxt in NO_SPACE_NEIGHBOURS:
            return ""
        return " "

    def quoted(self, i, quote):
        s = self.s
        j = i + 1
        while j < self.n:
            c = s[j]
            if c == "\\":
                j += 2
                continue
            if c == quote:
                return j + 1
            if c == "\n":
                return j
            j += 1
        return self.n

    def raw(self, i):
        s = self.s
        open_paren = s.find("(", i + 1)
        if open_paren < 0 or open_paren - i - 1 > 16:
            raise StripError(f"raw string without delimiter at line {self.line_of(i)}")
        delimiter = s[i + 1:open_paren]
        end = s.find(")" + delimiter + '"', open_paren + 1)
        if end < 0:
            raise StripError(f"unterminated raw string at line {self.line_of(i)}")
        return end + len(delimiter) + 2

    def number_end(self, i):
        s = self.s
        j = i + 1
        while j < self.n:
            c = s[j]
            if c in "+-" and s[j - 1] in "eEpP":
                j += 1
            elif WORD.match(c) or c == ".":
                j += 1
            elif c == "'" and j + 1 < self.n and WORD.match(s[j + 1]):
                j += 2
            else:
                break
        return j

    def run(self):
        s = self.s
        i = 0
        while i < self.n:
            c = s[i]
            two = s[i:i + 2]
            if two == "//":
                i = self.line_comment_end(i)
                self.out.append(MARK)
                self.removed += 1
                continue
            if two == "/*":
                end = s.find("*/", i + 2)
                if end < 0:
                    raise StripError(f"unterminated block comment at line {self.line_of(i)}")
                i = end + 2
                nxt = s[i] if i < self.n else ""
                if nxt in " \t" and self.last_char() in " \t":
                    while i < self.n and s[i] in " \t":
                        i += 1
                    nxt = s[i] if i < self.n else ""
                    if nxt in "\n\r" or nxt == "":
                        self.out.append(MARK)
                        self.removed += 1
                        continue
                    self.out.append(MARK)
                    self.removed += 1
                    continue
                self.out.append(MARK + self.block_replacement(nxt))
                self.removed += 1
                continue
            if c == '"':
                j = self.quoted(i, '"')
                self.out.append(s[i:j])
                i = j
                continue
            if c == "'":
                j = self.quoted(i, "'")
                self.out.append(s[i:j])
                i = j
                continue
            if c.isdigit() or (c == "." and i + 1 < self.n and s[i + 1].isdigit()):
                j = self.number_end(i)
                self.out.append(s[i:j])
                i = j
                continue
            if c.isalpha() or c == "_":
                j = i + 1
                while j < self.n and WORD.match(s[j]):
                    j += 1
                word = s[i:j]
                if j < self.n and s[j] == '"' and word in RAW_PREFIXES:
                    k = self.raw(j)
                    self.out.append(s[i:k])
                    i = k
                    continue
                if j < self.n and s[j] in "\"'" and word in STRING_PREFIXES:
                    k = self.quoted(j, s[j])
                    self.out.append(s[i:k])
                    i = k
                    continue
                self.out.append(word)
                i = j
                continue
            self.out.append(c)
            i += 1
        return "".join(self.out)


def tidy(marked):
    lines = marked.split("\n")
    kept = []
    dropped_since_kept = False
    for line in lines:
        if MARK not in line:
            if not line.strip() and dropped_since_kept and kept and not kept[-1].strip():
                continue
            kept.append(line)
            if line.strip():
                dropped_since_kept = False
            continue
        clean = line.replace(MARK, "").rstrip()
        if clean.strip():
            kept.append(clean)
            dropped_since_kept = False
            continue
        if kept and kept[-1].rstrip().endswith("\\"):
            kept.append("")
            continue
        dropped_since_kept = True
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept) + "\n"


def leading_license(text):
    match = re.match(r"\A((?:[ \t]*//[^\n]*\n)+|[ \t]*/\*.*?\*/[^\n]*\n)", text, re.S)
    if match and LICENSE_MARKER.search(match.group(1)):
        return match.group(1), text[match.end():]
    return "", text


def strip_comments(text):
    text = text.replace("\r\n", "\n")
    notice, body = leading_license(text)
    stripper = Stripper(body)
    stripped = tidy(stripper.run())
    if notice:
        stripped = notice + ("\n" if stripped.strip() else "") + stripped
    return stripped, stripper.removed, stripper.continued_line_comments, bool(notice)


TOKEN = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<d>[^\s()\\]{0,16})\([\s\S]*?\)(?P=d)")'
    r'|(?P<string>(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*"?)'
    r"|(?P<char>(?:u8|u|U|L)?'(?:\\.|[^'\\\n])*'?)"
    r"|(?P<number>(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[A-Za-z0-9_.]|'[A-Za-z0-9_])*)"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<splice>\\[ \t]*\n)"
    r"|(?P<newline>\n)"
    r"|(?P<space>[ \t\r\f\v]+)"
    r"|(?P<punct>.)")


def token_stream(text):
    tokens = []
    in_directive = False
    at_line_start = True
    pending_space = False
    for match in TOKEN.finditer(text):
        kind = match.lastgroup
        value = match.group()
        if kind == "splice":
            pending_space = True
            continue
        if kind == "newline":
            if in_directive:
                tokens.append("<NL>")
            in_directive = False
            at_line_start = True
            pending_space = False
            continue
        if kind == "space":
            pending_space = True
            continue
        if at_line_start and value == "#":
            in_directive = True
        if kind == "punct" and pending_space and tokens and len(tokens[-1]) == 1 and not WORD.match(tokens[-1]):
            tokens.append("<SP>")
        tokens.append(value)
        at_line_start = False
        pending_space = False
    return tokens


GLSL_DIRECTIVE = re.compile(r"^([ \t]*#[ \t]*)(version|extension|cmakedefine01|cmakedefine)\b", re.M)


def preprocessed(text, compiler):
    text = GLSL_DIRECTIVE.sub(r"\1pragma \2", text)
    result = subprocess.run(
        [compiler, "-x", "c++", "-std=c++20", "-fpreprocessed", "-dD", "-E", "-P", "-w", "-"],
        input=text.encode(), capture_output=True)
    if result.returncode != 0:
        raise StripError(result.stderr.decode(errors="replace")[:400])
    return result.stdout.decode(errors="replace")


def same_tokens(original, stripped, compiler):
    a = token_stream(preprocessed(original.replace("\r\n", "\n"), compiler))
    b = token_stream(preprocessed(stripped, compiler))
    if a == b:
        return None
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"token {index}: {a[max(0, index - 6):index + 6]} != {b[max(0, index - 6):index + 6]}"
    return f"length {len(a)} != {len(b)}"


LEFTOVER = re.compile(r'(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)'
                      r'|(?:u8|u|U|L)?R"(?P<d>[^\s()\\]{0,16})\([\s\S]*?\)(?P=d)"'
                      r'|(?:u8|u|U|L)?"(?:\\[\s\S]|[^"\\\n])*"'
                      r"|(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[A-Za-z0-9_.]|'[A-Za-z0-9_])*"
                      r"|(?:u8|u|U|L)?'(?:\\[\s\S]|[^'\\\n])*'")


def leftover_comments(text):
    notice, body = leading_license(text)
    return [body.count("\n", 0, m.start()) + 1 for m in LEFTOVER.finditer(body) if m.group("comment")]


def read_tag(ref, tag, prefixes):
    archive = subprocess.run(["git", "-C", str(ref), "archive", "--format=tar", tag, *prefixes],
                             capture_output=True, check=True).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                files[member.name] = tar.extractfile(member).read()
    return files


def plan(ref, tag):
    prefixes = [f"src/{tree}" for tree in TREES] + [f"src/{h}" for h in CONTRACT] + [f"src/{s}" for s in SCRIPTS]
    files = read_tag(ref, tag, prefixes)
    jobs = []
    for name, data in sorted(files.items()):
        rel = PurePosixPath(name).relative_to("src")
        if str(rel) in SKIP:
            continue
        if str(rel) in CONTRACT:
            dest = CONTRACT_DIR / rel
        elif str(rel) in SCRIPTS:
            dest = KYTY_DIR / "cmake" / rel
        else:
            dest = KYTY_DIR / rel
        jobs.append((str(rel), dest, data))
    return jobs


def strip_cmake(text):
    out = []
    for line in text.replace("\r\n", "\n").split("\n"):
        code = re.sub(r'"(?:\\.|[^"\\])*"', lambda m: "\x01" * len(m.group()), line)
        cut = code.find("#")
        if cut >= 0:
            line = line[:cut].rstrip()
            if not line.strip():
                continue
        out.append(line)
    return "\n".join(out).strip("\n") + "\n"


def main():
    parser = argparse.ArgumentParser(description="Historical tool: import graphics/ and common/ of a Kyty source tree without comments")
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--tag", default="v24")
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--compiler", default="g++-14")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    jobs = plan(args.ref, args.tag)
    failures = []
    stats = {"files": 0, "comments": 0, "notices": [], "continued": [], "copied": []}
    for rel, dest, data in jobs:
        target = args.dest / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = PurePosixPath(rel).suffix
        if rel.endswith(".cmake"):
            text = strip_cmake(data.decode("utf-8"))
            target.write_text(text, encoding="utf-8", newline="\n")
            stats["files"] += 1
            continue
        if suffix not in C_LIKE and not rel.endswith(".h.in"):
            target.write_bytes(data)
            stats["copied"].append(rel)
            continue
        original = data.decode("utf-8")
        try:
            stripped, removed, continued, notice = strip_comments(original)
        except StripError as error:
            failures.append(f"{rel}: {error}")
            continue
        target.write_text(stripped, encoding="utf-8", newline="\n")
        stats["files"] += 1
        stats["comments"] += removed
        if notice:
            stats["notices"].append(rel)
        if continued:
            stats["continued"].append(f"{rel}:{continued}")
        left = leftover_comments(stripped)
        if left:
            failures.append(f"{rel}: comment left at lines {left[:5]}")
        if not args.no_verify:
            try:
                difference = same_tokens(original, stripped, args.compiler)
            except StripError as error:
                difference = str(error)
            if difference and not continued:
                failures.append(f"{rel}: {difference}")
            elif difference:
                stats["continued"][-1] += f" ({difference})"
    print(f"files {stats['files']}, comments removed {stats['comments']}")
    for rel in stats["notices"]:
        print(f"licence notice kept: {rel}")
    for item in stats["continued"]:
        print(f"line comment continued by a backslash: {item}")
    for rel in stats["copied"]:
        print(f"copied as is: {rel}")
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
