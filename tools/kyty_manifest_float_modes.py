import argparse
from collections import Counter
import json
from pathlib import Path
import sys


HEADER = "# kyty pipeline manifest v"
DEFAULT_FLOAT_MODE = 0xC0
STAGE_KEYS = ("v", "h", "d", "p")
STAGE_ORDER = ("vs", "mesh", "ls", "hs", "tes", "ps")
MAX_FILE_BYTES = 1 << 29


def title_of(path):
    for suffix in (".shared.kpm", ".kpm"):
        if path.name.endswith(suffix):
            return path.name[:-len(suffix)]
    return path.name


def fields(tokens):
    return dict(token.partition("=")[::2] for token in tokens)


def parse_mode(text, where):
    parts = text.split(".")
    try:
        if len(parts) != 3:
            raise ValueError
        mode = (int(parts[0], 16), int(parts[1], 10), int(parts[2], 10))
    except ValueError:
        raise ValueError(f"{where}: float mode {text!r} is not <FLOAT_MODE hex>.<IEEE>.<DX10_CLAMP>") from None
    if mode[0] > 0xFF or mode[1] not in (0, 1) or mode[2] not in (0, 1):
        raise ValueError(f"{where}: float mode {text!r} is out of range")
    return mode


def read_manifest(path):
    if not path.is_file():
        raise ValueError(f"{path}: input must be a regular file")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{path}: larger than {MAX_FILE_BYTES} bytes")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith(HEADER):
        raise ValueError(f"{path}: not a kyty pipeline manifest")
    modules = {}
    compute = []
    pipelines = []
    for number, line in enumerate(lines[1:], start=2):
        tokens = line.split()
        if len(tokens) >= 3 and tokens[0] == "M":
            modules[tokens[1]] = tokens[2]
            if tokens[2] == "cs":
                compute.append(int(fields(tokens[3:]).get("fm", f"{DEFAULT_FLOAT_MODE:x}"), 16))
        elif tokens and tokens[0] == "P":
            pipelines.append((number, fields(tokens[1:])))
    stages = []
    recorded = 0
    for number, pipeline in pipelines:
        modes = 0
        for key in STAGE_KEYS:
            module = pipeline.get(key)
            if module is None:
                continue
            mode = pipeline.get("f" + key)
            if mode is not None:
                mode = parse_mode(mode, f"{path}:{number}")
                modes += 1
            stages.append((modules.get(module, "?"), mode))
        recorded += modes > 0
    return {"pipelines": len(pipelines), "recorded": recorded, "stages": stages, "compute": compute}


def stage_rank(stage):
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


def count(path):
    manifest = read_manifest(path)
    stages = {}
    for stage, mode in manifest["stages"]:
        row = stages.setdefault(stage, {"stages": 0, "recorded": 0, "float_mode": 0, "ieee": 0,
                                        "dx10_clamp": 0, "values": Counter()})
        row["stages"] += 1
        if mode is None:
            continue
        row["recorded"] += 1
        row["float_mode"] += mode[0] != DEFAULT_FLOAT_MODE
        row["ieee"] += mode[1]
        row["dx10_clamp"] += mode[2]
        row["values"][f"{mode[0]:x}.{mode[1]}.{mode[2]}"] += 1
    compute = manifest["compute"]
    return {
        "title": title_of(path),
        "file": str(path),
        "pipelines": manifest["pipelines"],
        "pipelines_recorded": manifest["recorded"],
        "stages": {stage: dict(row, values=dict(sorted(row["values"].items())))
                   for stage, row in sorted(stages.items(), key=lambda item: stage_rank(item[0]))},
        "compute": {"modules": len(compute),
                    "float_mode": sum(1 for mode in compute if mode != DEFAULT_FLOAT_MODE),
                    "values": dict(sorted(Counter(f"{mode:x}" for mode in compute).items()))},
    }


def manifest_paths(inputs):
    paths = []
    for name in inputs:
        path = Path(name)
        paths.extend(sorted(path.rglob("*.kpm")) if path.is_dir() else [path])
    return paths


def write_text(results, out):
    for result in results:
        out.write(f"{result['title']} ({result['file']}): {result['pipelines']} graphics pipelines, "
                  f"{result['pipelines_recorded']} with float modes\n")
        out.write(f"  {'stage':<6}{'stages':>8}{'recorded':>10}{'FLOAT_MODE!=c0':>16}{'IEEE=1':>8}"
                  f"{'DX10_CLAMP=1':>14}  values (FLOAT_MODE.IEEE.DX10_CLAMP)\n")
        for stage, row in result["stages"].items():
            values = ", ".join(f"{value} x{number}" for value, number in row["values"].items()) or "-"
            out.write(f"  {stage:<6}{row['stages']:>8}{row['recorded']:>10}{row['float_mode']:>16}"
                      f"{row['ieee']:>8}{row['dx10_clamp']:>14}  {values}\n")
        compute = result["compute"]
        values = ", ".join(f"{value} x{number}" for value, number in compute["values"].items()) or "-"
        out.write(f"  {'cs':<6}{compute['modules']:>8}{compute['modules']:>10}{compute['float_mode']:>16}"
                  f"{'-':>8}{'-':>14}  {values} (modules)\n")


def main(argv=None, out=sys.stdout):
    parser = argparse.ArgumentParser(
        description="Count the FLOAT_MODE, IEEE and DX10_CLAMP values that Kyty pipeline manifests (.kpm) record "
                    "per graphics stage, and the FLOAT_MODE of their compute modules, per title.")
    parser.add_argument("inputs", nargs="+", help=".kpm files, or folders searched for them")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    paths = manifest_paths(args.inputs)
    if not paths:
        parser.error("no .kpm file found")
    try:
        results = [count(path) for path in paths]
    except (OSError, UnicodeDecodeError, ValueError) as error:
        print(f"kyty_manifest_float_modes: {error}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(results, out, indent=2)
        out.write("\n")
    else:
        write_text(results, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
