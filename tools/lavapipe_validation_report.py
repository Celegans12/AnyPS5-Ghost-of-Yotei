import argparse
import collections
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MESSAGE = re.compile(r"\[Vulkan\]\[(?P<level>[EW])\]\[\d+\]: (?P<text>.*)")
VUID = re.compile(r"(VUID-[A-Za-z0-9_-]+|SYNC-HAZARD-[A-Z_]+|UNASSIGNED-[A-Za-z0-9_-]+)")


def known_entries(path):
    if not path or not Path(path).exists():
        return {}
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, _, reason = line.partition("\t")
            entries[name.strip()] = reason.strip()
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("junit", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--known")
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    known = known_entries(args.known)
    failed = []
    passed = skipped = 0
    for case in ET.parse(args.junit).getroot().iter("testcase"):
        if case.find("failure") is not None or case.get("status") == "fail":
            output = case.findtext("system-out") or ""
            message = next((m for m in MESSAGE.finditer(output) if m.group("level") == "E"), None)
            ids = VUID.findall(message.group("text")) if message else []
            failed.append((case.get("name"), ids[0] if ids else "", (message.group("text") if message else output.strip().splitlines()[-1] if output.strip() else "")[:240]))
        elif case.find("skipped") is not None:
            skipped += 1
        else:
            passed += 1
    warnings = collections.Counter()
    if args.log.exists():
        for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = MESSAGE.search(line)
            if match and match.group("level") == "W":
                found = VUID.search(match.group("text"))
                warnings[found.group(1) if found else match.group("text")[:90]] += 1
    unknown = [item for item in failed if item[0] not in known]
    lines = ["### lavapipe: validation layer pass", "",
             f"{passed} tests passed, {skipped} skipped, {len(failed)} failed under the Khronos validation layer with synchronization validation ({len(failed) - len(unknown)} known).", ""]
    if failed:
        lines += ["| test | message id | message |", "|---|---|---|"]
        for name, vuid, text in failed:
            lines.append(f"| `{name}`{' (known)' if name in known else ''} | `{vuid}` | {text.replace('|', '/')} |")
        lines.append("")
    if warnings:
        lines += ["Warnings by message id:", ""] + [f"- `{key}` x{count}" for key, count in warnings.most_common(10)] + [""]
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, vuid, text in unknown:
        print(f"error: {name} fails under the validation layer: {vuid} {text}")
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())
