import argparse
import datetime
import ipaddress
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


FORMAT = "rt-survey/1"
CACHE_NAME = re.compile(r"^([A-Z]{4}[0-9]{5})(?:_([0-9A-Za-z.\-]+))?\.code$")
PROBE_NAME = "kyty_rt_probe.exe" if os.name == "nt" else "kyty_rt_probe"
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6 = re.compile(
    r"(?<![\w:.])\[?((?:[0-9A-Fa-f:]*:(?:\d{1,3}\.){3}\d{1,3})|(?:[0-9A-Fa-f:]*:[0-9A-Fa-f:]*))(?:%[\w.-]+)?(?:\](?::\d+)?|(?![\w:]))"
)
HEX_RUN = re.compile(r"[0-9A-Fa-f]{5,}")
CLASSES = ("native", "oracle", "lane", "empty", "emulated")
DIAGNOSTICS = {
    "missing_tlas_source": "TLAS source could not be derived",
    "no_form_or_host_query": "No recognized walk or host query served the intersections",
    "local_flat_scratch": "Local FLAT access requires DWORD data and per-thread scratch storage",
}
COLUMNS = (
    ("program", 16, "<"),
    ("stage", 5, "<"),
    ("wave", 4, ">"),
    ("group", 9, "<"),
    ("isects", 6, ">"),
    ("native", 6, ">"),
    ("oracle", 6, ">"),
    ("lane", 4, ">"),
    ("empty", 5, ">"),
    ("emulated", 8, ">"),
    ("spirv", 7, "<"),
    ("status", 0, "<"),
)
BUILD_COMMAND = "cmake --build build --target kyty_rt_probe"


class SurveyError(Exception):
    pass


def data_folder():
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) / "AnyPS5" if base else None
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AnyPS5"
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base) / "AnyPS5" if base else Path.home() / ".local" / "share" / "AnyPS5"


def identity():
    names = set()
    try:
        names.add(Path.home().name)
    except (RuntimeError, KeyError):
        pass
    for key in ("USER", "USERNAME", "LOGNAME"):
        if os.environ.get(key):
            names.add(os.environ[key])
    host = platform.node()
    return {"home": str(Path.home()), "names": {name for name in names if len(name) >= 3}, "host": host if len(host) >= 3 else ""}


def redact_ipv6(text):
    def replace(match):
        candidate = match.group(1)
        if not re.search(r"[0-9A-Fa-f]", candidate) or candidate.count(":") < 2:
            return match.group(0)
        try:
            ipaddress.IPv6Address(candidate)
        except ValueError:
            return match.group(0)
        return "<ip>"

    return IPV6.sub(replace, text)


def clean_text(text, who=None, limit=200):
    if not isinstance(text, str):
        return None
    who = who or identity()
    text = text.strip()
    for home in {who["home"], who["home"].replace("\\", "/")}:
        if len(home) >= 3:
            text = text.replace(home, "~")
    for name in who["names"]:
        text = re.sub(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", "<user>", text, flags=re.IGNORECASE)
    if who["host"]:
        text = re.sub(r"(?<![\w-])" + re.escape(who["host"]) + r"(?![\w-])", "<host>", text, flags=re.IGNORECASE)
    text = redact_ipv6(text)
    text = IPV4.sub("<ip>", text)
    text = re.sub(r"[A-Za-z]:[\\/][^\s]*|(?<![\w.])/[^\s]+", "<path>", text)
    return text[:limit]


def clean_stop(text):
    words = []
    for word in text.split("_"):
        if "/" in word or "\\" in word:
            word = re.split(r"[\\/]", word)[-1]
        words.append(HEX_RUN.sub("<n>", word))
    return "_".join(words)[:160]


def candidate_probe_dirs(root):
    found = []
    for build in sorted(root.glob("build*")):
        if build.is_dir():
            found.extend([build / "tests", build, build / "core" / "libs" / "prx" / "libSceAgcDriver"])
    return found


def find_probe(explicit, root):
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            path = path / PROBE_NAME
        if not path.is_file():
            raise SurveyError(f"the probe {path.name} is not at the path given")
        return path
    if os.environ.get("KYTY_RT_PROBE"):
        path = Path(os.environ["KYTY_RT_PROBE"])
        if path.is_file():
            return path
    for folder in candidate_probe_dirs(root):
        if (folder / PROBE_NAME).is_file():
            return folder / PROBE_NAME
    on_path = shutil.which(PROBE_NAME)
    if on_path:
        return Path(on_path)
    raise SurveyError(
        f"{PROBE_NAME} was not found. Build it from the AnyKyty folder with: {BUILD_COMMAND}\n"
        "then run this command again, or pass its path with --probe."
    )


def cache_files(folder):
    return [path for path in folder.glob("*.code") if CACHE_NAME.match(path.name)]


def find_caches(places, depth=4):
    found = {}
    for place in places:
        if not place or not place.is_dir():
            continue
        base = len(place.parts)
        for current, dirs, _ in os.walk(place):
            level = len(Path(current).parts) - base
            if level >= depth:
                dirs[:] = []
            dirs[:] = [name for name in dirs if name not in (".git", "node_modules", "3rdparty")]
            folder = Path(current)
            if folder.name == "_PipelineCache":
                for path in cache_files(folder):
                    found[path.resolve()] = path
                dirs[:] = []
    return sorted(found.values(), key=lambda path: path.stat().st_mtime, reverse=True)


def choose_cache(args, root):
    if args.cache:
        path = Path(args.cache)
        if path.is_dir():
            options = cache_files(path) or cache_files(path / "_PipelineCache")
            if not options:
                raise SurveyError("there is no <title id>_<version>.code file in that folder")
            options.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            path = options[0]
        if not path.is_file():
            raise SurveyError("the cache file given does not exist")
        return path, []
    places = [Path(args.game)] if args.game else [Path.cwd(), root]
    options = find_caches(places)
    if args.title:
        options = [path for path in options if CACHE_NAME.match(path.name).group(1) == args.title]
    if not options:
        raise SurveyError(
            "no _PipelineCache/<title id>_<version>.code was found. Play the game on AnyKyty for a few minutes, "
            "then run this from the game's folder, or pass --game <folder> or --cache <file>.code."
        )
    return options[0], options[1:]


def title_of(path):
    match = CACHE_NAME.match(path.name)
    if not match:
        return "unknown", "unknown"
    return match.group(1), match.group(2) or "unknown"


def run_text(command, timeout=20):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def vendor_of(name):
    lowered = (name or "").lower()
    if any(word in lowered for word in ("nvidia", "geforce", "quadro", "rtx", "gtx")):
        return "nvidia"
    if any(word in lowered for word in ("amd", "radeon", "ati ")):
        return "amd"
    if "intel" in lowered:
        return "intel"
    return None


def read_report(path, title):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("anyps5"), dict):
        return None
    reported = (data.get("game") or {}).get("titleId")
    if isinstance(reported, str) and title != "unknown" and reported != title:
        return None
    return data


def find_report(explicit, cache, game):
    if explicit:
        data = read_report(Path(explicit), "unknown")
        if data is None:
            raise SurveyError("the run report given is not an anyps5-report.json")
        return data
    title = title_of(cache)[0]
    folders = [cache.parent.parent, Path.cwd()]
    if game:
        folders.insert(0, Path(game))
    candidates = [folder / "anyps5-report.json" for folder in folders]
    data_dir = data_folder()
    if data_dir and (data_dir / "reports").is_dir():
        candidates.extend(sorted((data_dir / "reports").glob("*/anyps5-report.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5])
    reports = [path for path in candidates if path.is_file()]
    reports.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in reports:
        data = read_report(path, title)
        if data is not None:
            return data
    return None


def parse_log(path):
    info = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        raise SurveyError("the log file given cannot be read")
    device = re.findall(r"Select device: (.+)", text)
    if device:
        info["name"] = device[-1].strip()
    groups = re.findall(r"Vulkan subgroup: default=(\d+) min=(\d+) max=(\d+) .*?wave64=(true|false)", text)
    if groups:
        info["subgroup"] = 64 if groups[-1][3] == "true" else 32
    return info


def vulkan_devices():
    text = run_text(["vulkaninfo", "--summary"])
    if not text:
        return []
    devices = []
    for block in re.split(r"\nGPU\d+:", "\n" + text)[1:]:
        fields = dict(re.findall(r"^\s*(\w+)\s*=\s*(.+?)\s*$", block, re.MULTILINE))
        if "deviceName" in fields:
            devices.append(
                {
                    "name": fields["deviceName"],
                    "driver": " ".join(part for part in (fields.get("driverName"), fields.get("driverInfo")) if part) or None,
                    "vulkan": fields.get("apiVersion"),
                    "discrete": "DISCRETE" in fields.get("deviceType", ""),
                }
            )
    return devices


def gpu_info(report, log, subgroup_arg, lds_arg):
    who = identity()
    gpu = {"name": None, "vendor": None, "driver": None, "vulkan": None, "source": None}
    candidates = []
    reported = (report or {}).get("system", {}).get("gpu") if report else None
    if isinstance(reported, dict) and reported.get("name"):
        gpu.update({key: reported.get(key) for key in ("name", "vendor", "driver", "vulkan")})
        gpu["source"] = "run-report"
    logged = parse_log(log) if log else {}
    if logged.get("name") and not gpu["name"]:
        gpu["name"] = logged["name"]
        gpu["source"] = "log"
    if not gpu["name"]:
        devices = vulkan_devices()
        if devices:
            chosen = next((device for device in devices if device["discrete"]), devices[0])
            gpu.update({key: chosen[key] for key in ("name", "driver", "vulkan")})
            gpu["source"] = "vulkaninfo"
            candidates = [device["name"] for device in devices]
    if not gpu["name"] and os.name == "nt":
        listing = run_text(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.DriverVersion }"]
        )
        if listing and listing.strip():
            names = [line.split("|") for line in listing.strip().splitlines() if line.strip()]
            gpu["name"] = names[0][0].strip()
            gpu["driver"] = names[0][1].strip() if len(names[0]) > 1 else None
            gpu["source"] = "windows"
            candidates = [entry[0].strip() for entry in names]
    if not gpu["name"]:
        smi = run_text(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if smi and smi.strip():
            first = smi.strip().splitlines()[0].split(",")
            gpu["name"] = first[0].strip()
            gpu["driver"] = f"NVIDIA {first[1].strip()}" if len(first) > 1 else None
            gpu["source"] = "nvidia-smi"
    vendor = vendor_of(gpu["name"]) or vendor_of(str(gpu.get("vendor") or ""))
    gpu["vendor"] = vendor or (gpu.get("vendor") if isinstance(gpu.get("vendor"), str) else None)
    if subgroup_arg:
        subgroup, source = subgroup_arg, "option"
    elif logged.get("subgroup"):
        subgroup, source = logged["subgroup"], "log"
    elif vendor == "amd":
        subgroup, source = 64, "vendor"
    else:
        subgroup, source = 32, "vendor" if vendor else "default"
    if lds_arg:
        lds = lds_arg
    else:
        lds = 65536 if vendor in ("amd", "intel") else 49152
    for key in ("name", "driver", "vulkan", "vendor"):
        gpu[key] = clean_text(gpu[key], who, 120) if isinstance(gpu[key], str) else None
    return gpu, {"subgroup": subgroup, "subgroupSource": source, "ldsLimitBytes": lds}, candidates


def tokens(line):
    head, _, stop = line.partition(" stop=")
    fields = {}
    for part in head.split()[1:]:
        key, sep, value = part.partition("=")
        if sep:
            fields[key] = value
    if stop:
        fields["stop"] = stop
    return fields


def number(fields, key):
    try:
        return int(fields.get(key, "0"))
    except ValueError:
        return 0


def diagnostic_counts(value):
    counts = {}
    for entry in value.split(","):
        code, _, count = entry.partition(":")
        if code in DIAGNOSTICS and 0 < len(count) <= 10 and count.isascii() and count.isdecimal():
            count = int(count)
            if 0 < count <= 0xFFFFFFFF:
                counts[code] = count
    return counts


def parse_probe(text):
    header = {}
    rows = []
    summary = {}
    notes = []
    for line in text.splitlines():
        if line.startswith("rt-survey "):
            header = tokens(line)
        elif line.startswith("note "):
            notes.append(clean_text(line[5:]))
        elif line.startswith("summary "):
            summary = {key: value for key, value in tokens(line).items()}
        elif line.startswith("program "):
            parts = line.split()
            fields = tokens(line)
            group = [int(value) for value in fields.get("group", "0,0,0").split(",") if value.isdigit()]
            row = {
                "program": parts[1],
                "stage": fields.get("stage", "unknown"),
                "wave": number(fields, "wave"),
                "group": group if len(group) == 3 else [0, 0, 0],
                "lds": number(fields, "lds"),
                "scratch": number(fields, "scratch"),
                "modules": number(fields, "modules"),
                "source": fields.get("source", "manifest"),
                "status": fields.get("status", "unknown"),
                "intersections": number(fields, "bvh"),
                "spirv": {"valid": True, "invalid": False}.get(fields.get("spirv")),
                "diagnostics": diagnostic_counts(fields.get("diagnostics", "")),
            }
            for name in CLASSES:
                row[name] = number(fields, name)
            if "back" in fields:
                row["back"] = fields["back"]
            if "phase" in fields:
                row["phase"] = fields["phase"]
            if "stop" in fields:
                row["stop"] = clean_stop(fields["stop"])
            rows.append(row)
    return header, rows, summary, notes


def block_of(rows):
    totals = {"programs": len(rows), "intersections": sum(row["intersections"] for row in rows)}
    for name in CLASSES:
        totals[name] = sum(row[name] for row in rows)
    totals["unclassified"] = totals["intersections"] - sum(totals[name] for name in CLASSES)
    translated = [row for row in rows if row["spirv"] is not None]
    totals["spirvValid"] = sum(1 for row in translated if row["spirv"])
    totals["spirvInvalid"] = sum(1 for row in translated if not row["spirv"])
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    totals["statuses"] = dict(sorted(statuses.items()))
    return totals


def totals_of(rows):
    totals = block_of(rows)
    totals["bySource"] = {
        "manifest": block_of([row for row in rows if row["source"] == "manifest"]),
        "code": block_of([row for row in rows if row["source"] != "manifest"]),
    }
    return totals


def split_build(text):
    if not text or text == "unknown":
        return None, None
    return text.replace("+modified", ""), text.endswith("+modified")


def build_report(cache, probe_text, report, gpu, host, candidates, now):
    header, rows, summary, notes = parse_probe(probe_text)
    title, version = title_of(cache)
    build, build_modified = split_build(header.get("build"))
    anyps5 = (report or {}).get("anyps5") or {}
    commit = anyps5.get("commit") or build
    who = identity()
    summary_numbers = {}
    for key, value in summary.items():
        try:
            summary_numbers[key] = float(value) if "." in value else int(value)
        except ValueError:
            continue
    document = {
        "format": FORMAT,
        "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title": {"id": title, "version": version},
        "anyKyty": {
            "commit": commit,
            "commitSource": "run-report" if anyps5.get("commit") else ("probe" if build else None),
            "modified": anyps5.get("modified") if anyps5.get("commit") else build_modified,
            "probeBuild": build,
            "probeBuildModified": build_modified,
        },
        "host": {
            "os": clean_text(f"{platform.system()} {platform.release()}", who, 80),
            "machine": platform.machine(),
            "gpu": gpu,
            "gpuCandidates": [clean_text(name, who, 120) for name in candidates],
            **host,
        },
        "probe": {
            "oracle": number(header, "oracle"),
            "subgroup": number(header, "subgroup"),
            "ldsLimitBytes": number(header, "lds_limit"),
        },
        "cache": summary_numbers,
        "totals": totals_of(rows),
        "notes": notes,
        "programs": rows,
    }
    return document


def fmt_row(values):
    out = []
    for (name, width, align), value in zip(COLUMNS, values):
        text = str(value)
        out.append(text if width == 0 else (text.ljust(width) if align == "<" else text.rjust(width)))
    return " ".join(out).rstrip()


def render_text(document, limit=60):
    totals = document["totals"]
    anykyty = document["anyKyty"]
    gpu = document["host"]["gpu"]
    commit = anykyty["commit"] or "unknown"
    if anykyty.get("modified"):
        commit += " (modified)"
    lines = [
        f"AnyKyty RT survey ({document['format']})",
        f"Title {document['title']['id']} version {document['title']['version']}",
        f"AnyKyty commit {commit}",
        f"GPU {gpu['name'] or 'unknown'}, driver {gpu['driver'] or 'unknown'}, Vulkan {gpu['vulkan'] or 'unknown'} "
        f"(source: {gpu['source'] or 'none'}); host subgroup {document['host']['subgroup']} ({document['host']['subgroupSource']})",
        f"OS {document['host']['os']} {document['host']['machine']}",
        "",
    ]
    labels = (
        ("manifest", "Programs the game ran (pipeline manifest, inputs as the game set them)"),
        ("code", "Programs only in the code cache (inputs guessed, both wave sizes tried)"),
    )
    for source, label in labels:
        part = totals["bySource"][source]
        served = part["native"] + part["oracle"] + part["lane"]
        lines.append(
            f"{label}: {part['programs']} programs, {part['intersections']} BVH intersections: {served} served by ray queries "
            f"(native {part['native']}, oracle {part['oracle']}, lane {part['lane']}), {part['empty']} empty, "
            f"{part['emulated']} emulated, {part['unclassified']} not classified (programs that call functions or stop); "
            f"SPIR-V valid in {part['spirvValid']} of {part['spirvValid'] + part['spirvInvalid']}."
        )
    lines.append("Programs by status: " + (", ".join(f"{name} {count}" for name, count in totals["statuses"].items()) or "none") + ".")
    for note in document["notes"]:
        lines.append(f"Note: {note}")
    rows = sorted(document["programs"], key=lambda row: (-(row["emulated"] + row["empty"]), -row["intersections"], row["program"]))
    if rows:
        lines += [""]
        lines.append(fmt_row([name for name, _, _ in COLUMNS]))
        for row in rows[:limit]:
            spirv = "-" if row["spirv"] is None else ("valid" if row["spirv"] else "INVALID")
            lines.append(
                fmt_row(
                    [
                        row["program"],
                        row["stage"],
                        row["wave"],
                        "x".join(str(size) for size in row["group"]),
                        row["intersections"],
                        row["native"],
                        row["oracle"],
                        row["lane"],
                        row["empty"],
                        row["emulated"],
                        spirv,
                        row["status"] + ("" if row["source"] == "manifest" else " (guessed inputs)"),
                    ]
                )
            )
            diagnostics = row["diagnostics"]
            if diagnostics:
                lines.append("  Probe diagnostics: " + ", ".join(
                    f"{DIAGNOSTICS[code]} ({count})" for code, count in diagnostics.items()
                ))
            elif row["empty"]:
                lines.append("  No probe diagnostics captured for this program.")
        if len(rows) > limit:
            lines.append(f"... {len(rows) - limit} more programs in the JSON file")
    else:
        lines += ["", "No program with BVH intersections is in the cache yet."]
    return "\n".join(lines) + "\n"


def audit(text, cache):
    who = identity()
    leaks = [str(cache), str(cache.parent), str(cache.parent.parent), who["home"], who["home"].replace("\\", "/")]
    for leak in leaks:
        if len(leak) > 3 and (leak in text or leak.replace("\\", "\\\\") in text):
            raise SurveyError("the report would contain a path from this machine; nothing was written")


def run_probe(probe, cache, host, oracle, timeout):
    command = [str(probe), "--cache", str(cache), "--all", "--subgroup", str(host["subgroup"]), "--lds-limit", str(host["ldsLimitBytes"])]
    if oracle is not None:
        command += ["--oracle", str(oracle)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired:
        raise SurveyError(f"the probe did not finish in {timeout} seconds")
    except OSError as error:
        raise SurveyError(f"the probe could not start: {clean_text(str(error))}")
    if done.returncode != 0:
        raise SurveyError(f"the probe failed with exit code {done.returncode}: {clean_text(done.stderr.strip().splitlines()[-1] if done.stderr.strip() else '')}")
    return done.stdout


def parse_args(argv):
    parser = argparse.ArgumentParser(description="List every ray tracing program of a game and how AnyKyty serves it, without any game code.")
    parser.add_argument("--game", help="the game's folder (where _PipelineCache is); default: the current folder, then the AnyKyty folder")
    parser.add_argument("--cache", help="a _PipelineCache/<title id>_<version>.code file, or a folder that holds one")
    parser.add_argument("--title", help="keep only the cache of this title id")
    parser.add_argument("--probe", help="the kyty_rt_probe program; default: found in build*/tests or on PATH")
    parser.add_argument("--report", help="an anyps5-report.json to take the commit and GPU from")
    parser.add_argument("--log", help="a log of a run, to take the GPU and subgroup size from")
    parser.add_argument("--out-dir", default=".", help="where rt-survey-<title>-<version>.json goes (default: the current folder)")
    parser.add_argument("--oracle", type=int, help="KYTY_RT_ORACLE to survey with (default: the build's default)")
    parser.add_argument("--subgroup", type=int, choices=(32, 64), help="host subgroup size of your GPU (default: from the log, or by vendor)")
    parser.add_argument("--lds-limit", type=int, help="maxComputeSharedMemorySize of your GPU in bytes")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds the probe may run")
    return parser.parse_args(argv)


def main(argv=None, now=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    try:
        cache, others = choose_cache(args, root)
        probe = find_probe(args.probe, root)
        report = find_report(args.report, cache, args.game)
        gpu, host, candidates = gpu_info(report, args.log, args.subgroup, args.lds_limit)
        probe_text = run_probe(probe, cache, host, args.oracle, args.timeout)
        document = build_report(cache, probe_text, report, gpu, host, candidates, now or datetime.datetime.now(datetime.timezone.utc))
        text = render_text(document)
        encoded = json.dumps(document, indent=2) + "\n"
        audit(encoded + text, cache)
        title, version = title_of(cache)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"rt-survey-{title}-{version}.json"
        text_path = out / f"rt-survey-{title}-{version}.txt"
        json_path.write_text(encoded, encoding="utf-8")
        text_path.write_text(text, encoding="utf-8")
    except SurveyError as error:
        print(f"rt_survey: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(text)
    if others:
        print(f"{len(others)} older cache file(s) were left out; pass --cache or --title to survey another title.")
    print(f"\nWrote {json_path} and {text_path}.")
    print("Paste the text above in the issue and attach the JSON file. Both hold no game code: only hashes, counts and your GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
