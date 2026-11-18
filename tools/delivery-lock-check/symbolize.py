import argparse
import bisect
import collections
import os
import re
import subprocess
import sys

SITE = re.compile(r"\[delivery-lock-check\] site=(0x[0-9a-f]+) (\w+) lock=(0x[0-9a-f]+) ran=(\d+) deferred=(\d+)")
FRESH = re.compile(r"\[delivery-lock-check\] (ran|deferred) tid=\d+ us=\d+((?: \| \w+ lock=0x[0-9a-f]+ site=0x[0-9a-f]+)+)")
HELD = re.compile(r"(\w+) lock=(0x[0-9a-f]+) site=(0x[0-9a-f]+)")
SUMMARY = re.compile(r"\[delivery-lock-check\] arrivals=(\d+) clean=(\d+) ran-with-locks=(\d+) deferred-with-locks=(\d+) untracked=(\d+)(?: in-guest-code=(\d+) in-host-code=(\d+))?")
STACK = re.compile(r"\[delivery-lock-check\] stack id=(\d+) frames=([0-9a-fx,]+)")
STACK_COUNT = re.compile(r"\[delivery-lock-check\] stack-count id=(\d+) arrivals=(\d+) ran=(\d+) deferred=(\d+)")


def read_maps(path):
    maps = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split(maxsplit=5)
            if len(parts) < 6 or not parts[5].startswith("/"):
                continue
            start, end = (int(value, 16) for value in parts[0].split("-"))
            maps.append((start, end, int(parts[2], 16), parts[5].strip()))
    bases = {}
    for start, _, offset, path in maps:
        if offset == 0 and path not in bases:
            bases[path] = start
    return maps, bases


class Symbols:
    def __init__(self, libs):
        self.libs = libs
        self.cache = {}

    def resolve(self, path):
        name = os.path.basename(path)
        for directory in self.libs:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate
        return path if os.path.exists(path) else None

    def table(self, path):
        if path not in self.cache:
            entries = []
            resolved = self.resolve(path)
            if resolved is not None:
                output = subprocess.run(["nm", "-C", "-n", "--defined-only", resolved], capture_output=True, text=True, check=False).stdout
                if not output.strip():
                    output = subprocess.run(["nm", "-C", "-n", "-D", "--defined-only", resolved], capture_output=True, text=True, check=False).stdout
                for line in output.splitlines():
                    parts = line.split(" ", 2)
                    if len(parts) == 3 and parts[1] in "tTwWdDbBrRvV" and parts[0]:
                        entries.append((int(parts[0], 16), parts[2]))
            self.cache[path] = ([address for address, _ in entries], [name for _, name in entries])
        return self.cache[path]

    def name(self, maps, bases, address):
        owner = None
        for start, end, _, path in maps:
            if start <= address < end:
                owner = (start, path)
                break
            if start <= address and address - end < (64 << 20) and (owner is None or start > owner[0]):
                owner = (start, path)
        if owner is not None:
            start, path = owner
            relative = address - bases.get(path, start)
            addresses, names = self.table(path)
            index = bisect.bisect_right(addresses, relative) - 1
            symbol = f"{names[index]}+0x{relative - addresses[index]:x}" if index >= 0 else f"0x{relative:x}"
            return f"{os.path.basename(path)}!{symbol}"
        return f"0x{address:x}"


def main():
    parser = argparse.ArgumentParser(description="name the lock sites a delivery-lock-check run reported")
    parser.add_argument("log", help="stderr of the checked process")
    parser.add_argument("maps", help="DELIVERY_LOCK_CHECK_OUT.maps written by the check")
    parser.add_argument("--libs", action="append", default=[], help="directory holding the libraries the process loaded, when the paths in maps are gone")
    parser.add_argument("--frames", type=int, default=8, help="frames shown for each host stack")
    args = parser.parse_args()
    maps, bases = read_maps(args.maps)
    symbols = Symbols(args.libs)
    sites = collections.OrderedDict()
    stacks = {}
    counts = {}
    summary = None
    with open(args.log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = SITE.search(line)
            if match:
                sites[int(match.group(1), 16)] = (match.group(2), int(match.group(3), 16), int(match.group(4)), int(match.group(5)))
                continue
            match = STACK.search(line)
            if match:
                stacks[int(match.group(1))] = [int(value, 16) for value in match.group(2).split(",")]
                continue
            match = STACK_COUNT.search(line)
            if match:
                counts[int(match.group(1))] = tuple(int(value) for value in match.group(2, 3, 4))
                continue
            match = FRESH.search(line)
            if match:
                for kind, lock, site in HELD.findall(match.group(2)):
                    if int(site, 16) not in sites:
                        ran = 1 if match.group(1) == "ran" else 0
                        sites[int(site, 16)] = (kind, int(lock, 16), ran, 1 - ran)
                continue
            match = SUMMARY.search(line)
            if match:
                summary = match.groups()
    if summary is None:
        print("no delivery-lock-check summary in the log")
        return 1
    print(f"arrivals {summary[0]}, without a tracked lock {summary[1]}, ran with locks held {summary[2]}, deferred with locks held {summary[3]}, untracked {summary[4]}")
    if summary[5] is not None:
        print(f"interrupted in guest code {summary[5]}, in host code {summary[6]}")
    for site, (kind, lock, ran, deferred) in sorted(sites.items(), key=lambda item: (-item[1][2], -item[1][3])):
        print(f"ran={ran:<6} deferred={deferred:<6} {kind:<5} {symbols.name(maps, bases, site)}  lock {symbols.name(maps, bases, lock)}")
    for identifier, frames in sorted(stacks.items(), key=lambda item: -counts.get(item[0], (0, 0, 0))[0]):
        arrivals, ran, deferred = counts.get(identifier, (1, 0, 0))
        print(f"host stack {identifier}: arrivals={arrivals} ran={ran} deferred={deferred}")
        for depth, address in enumerate(frames[:args.frames]):
            print(f"    {symbols.name(maps, bases, address if depth == 0 else address - 1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
