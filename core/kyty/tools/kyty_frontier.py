import argparse
import collections
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

Row = collections.namedtuple("Row", "kind origin package symbol users declared")

SYSTEM_LIBRARIES = (
    "/usr/lib/x86_64-linux-gnu/libc.so.6",
    "/usr/lib/x86_64-linux-gnu/libm.so.6",
    "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    "/usr/lib/x86_64-linux-gnu/libgcc_s.so.1",
    "/lib64/ld-linux-x86-64.so.2",
)
SYSTEM_EXTRA = {"__dso_handle"}
HEADER_PACKAGE = {
    "kernel/memory.h": "K5",
    "loader/demonsSoulsPageFree.h": "K5",
    "kernel/eventQueue.h": "K9",
    "kernel/pthread.h": "K9",
    "kernel/guestPause.h": "K9",
    "kernel/fileSystem.h": "K9",
    "loader/systemContent.h": "K9",
    "loader/timer.h": "K9",
    "loader/guestOodle.h": "K9",
    "loader/demonsSoulsTitle.h": "K9",
    "libs/agc.h": "K8",
    "libs/errno.h": "K9",
    "libs/hmd2.h": "K4",
    "libs/controller.h": "K4",
    "libs/dialog.h": "K4",
    "libs/ime.h": "K4",
    "libs/imeCommon.h": "K4",
    "libs/imeDialog.h": "K4",
    "libs/libs.h": "K4",
}
PACKAGE_BY_PATH = (
    ("kernel/memory", "K5"),
    ("loader/demonsSoulsPageFree", "K5"),
    ("kernel/eventQueue", "K9"),
    ("kernel/pthread", "K9"),
    ("kernel/guestPause", "K9"),
    ("kernel/fileSystem", "K9"),
    ("kernel/", "K9"),
    ("libs/agc", "K8"),
    ("libs/errno", "K9"),
    ("libs/videoOut", "K4"),
    ("libs/controller", "K4"),
    ("libs/dialog", "K4"),
    ("libs/ime", "K4"),
    ("libs/hmd2", "K4"),
    ("libs/libHmd2", "K4"),
    ("libs/libs", "K4"),
    ("libs/", "K9"),
    ("loader/runtimeLinker", "K6"),
    ("loader/", "K9"),
    ("emulator", "K9"),
    ("graphics/presentation/", "K4"),
    ("graphics/guest_gpu/", "K1"),
    ("graphics/host_gpu/renderer/cache/", "K2a"),
    ("graphics/host_gpu/renderer/", "K2b"),
    ("graphics/host_gpu/", "K2a"),
    ("graphics/shader/", "K3"),
    ("common/", "K9"),
)
THIRD_PARTY_PREFIXES = (
    ("vk", "Vulkan (chargeur)"),
    ("SDL_", "SDL3"),
    ("ImGui", "imgui"),
    ("Im", "imgui"),
    ("xr", "OpenXR"),
    ("ffx", "FidelityFX"),
    ("tracy::", "tracy"),
    ("___tracy", "tracy"),
    ("spdlog::", "spdlog"),
    ("nlohmann::", "nlohmann_json"),
    ("cpuinfo", "cpuinfo"),
)


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def archive_symbols(archive):
    defined = set()
    undefined = collections.defaultdict(set)
    for line in run(["nm", "-A", "-P", "--defined-only", str(archive)]).splitlines():
        head, _, rest = line.rpartition(": ")
        defined.add(rest.split()[0])
    for line in run(["nm", "-A", "-P", "-u", str(archive)]).splitlines():
        head, _, rest = line.rpartition(": ")
        member = re.search(r"\[(.*)\]", head).group(1)
        undefined[rest.split()[0]].add(member)
    return defined, undefined


def library_symbols(path, dynamic):
    command = ["nm", "-P", "--defined-only"] + (["-D"] if dynamic else ["-A"]) + [str(path)]
    names = set()
    for line in run(command).splitlines():
        rest = line.rpartition(": ")[2] if not dynamic else line
        if not rest.strip():
            continue
        names.add(rest.split()[0].split("@")[0])
    return names


def demangle(names):
    names = list(names)
    output = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout
    return dict(zip(names, output.splitlines()))


def strip_template_and_args(name):
    depth = 0
    out = []
    for c in name:
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        elif depth == 0:
            out.append(c)
    text = "".join(out)
    for prefix in ("vtable for ", "typeinfo for ", "typeinfo name for ", "guard variable for ", "VTT for ",
                   "construction vtable for ", "TLS init function for ", "TLS wrapper function for ",
                   "non-virtual thunk to ", "virtual thunk to "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = re.sub(r"\s*(const|volatile|&|&&|\[abi:[^\]]*\])\s*$", "", text.strip())
    text = text.split(" ")[-1]
    return text


def upstream_sources(ref, tag):
    data = subprocess.run(["git", "-C", str(ref), "archive", "--format=tar", tag, "src"], capture_output=True, check=True).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith((".cpp", ".h", ".inc")):
                files[str(PurePosixPath(member.name).relative_to("src"))] = tar.extractfile(member).read().decode("utf-8", "replace")
    return files


def package_of(path):
    for prefix, package in PACKAGE_BY_PATH:
        if path.startswith(prefix):
            return package
    return "?"


def definition_pattern(owner, last):
    qualified = (re.escape(owner) + r"\s*::\s*") if owner else ""
    return re.compile(
        r"^[ \t]*(?:template\s*<[^;{}]*>\s*)?(?:[\w:<>,*&~]+[ \t*&]+)*" + qualified + r"(?<![\w~])" + re.escape(last)
        + r"\s*\((?:[^;{}()]|\([^;{}()]*\))*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:->[^;{]*)?(?::[^;]*?)?\{", re.M)


def namespaces_of(text):
    names = set()
    for match in re.finditer(r"\b(?:namespace|class|struct)\s+([\w:]+)", text):
        names.update(match.group(1).split("::"))
    return names


def defining_files(qualified, sources, compiled):
    parts = [p for p in qualified.split("::") if p]
    if not parts:
        return []
    last = parts[-1]
    owners = parts[:-1]
    if last.startswith("operator"):
        operator = re.compile(r"\b" + re.escape(owners[-1]) + r"\s*::\s*operator\b[^;{]*\{") if owners else None
        hits = sorted(path for path, text in sources.items()
                      if operator and path not in compiled and path.endswith(".cpp") and operator.search(text))
        return hits
    owner = owners[-1] if owners else None
    word = re.compile(r"(?<![\w~])" + re.escape(last) + r"\b")
    qualified_definition = definition_pattern(owner, last) if owner else None
    plain_definition = definition_pattern(None, last)
    scored = []
    for path, text in sources.items():
        if path in compiled or not path.endswith(".cpp") or not word.search(text):
            continue
        if qualified_definition and qualified_definition.search(text):
            kind = 3
        elif plain_definition.search(text):
            kind = 2 if owner and owner in namespaces_of(text) else 1
        else:
            continue
        context = len(set(owners) & namespaces_of(text))
        scored.append(((kind, context), path))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return sorted(path for score, path in scored if score == best)


def declaring_headers(qualified, headers):
    parts = [p for p in qualified.split("::") if p]
    last = parts[-1]
    owner = parts[-2] if len(parts) > 1 else None
    word = re.compile(r"(?<![\w~])" + re.escape(last) + r"\b")
    found = []
    for path, text in sorted(headers.items()):
        if word.search(text) and (owner is None or owner in namespaces_of(text)):
            found.append(path)
    return found


def main():
    parser = argparse.ArgumentParser(description="Undefined symbols of kyty_gpu, grouped by origin and package")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--kyty", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--tag", default="v24")
    parser.add_argument("--library", action="append", default=[], help="name=archive of a third-party library linked with kyty_gpu")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--cxx", default="g++-14", help="the GCC that builds AnyPS5, for its libgcc.a")
    args = parser.parse_args()

    defined, undefined = archive_symbols(args.archive)
    external = {name: members for name, members in undefined.items() if name not in defined}
    system_by_library = {}
    for library in SYSTEM_LIBRARIES:
        system_by_library[Path(library).name] = library_symbols(library, True)
    libgcc = run([args.cxx, "-print-libgcc-file-name"]).strip()
    system_by_library["libgcc.a"] = library_symbols(libgcc, False)
    system_by_library["crt"] = set(SYSTEM_EXTRA)
    system = set().union(*system_by_library.values())
    third = {}
    for item in args.library:
        name, _, path = item.partition("=")
        for symbol in library_symbols(path, False):
            third.setdefault(symbol, name)
    names = demangle(external)
    sources = upstream_sources(args.ref, args.tag)
    compiled = {str(p.relative_to(args.kyty)) for p in args.kyty.rglob("*.cpp")}
    headers = {str(p.relative_to(args.contract)): p.read_text() for p in args.contract.rglob("*.h")}

    rows = []
    for mangled, members in sorted(external.items(), key=lambda kv: names[kv[0]]):
        pretty = names[mangled]
        users = sorted(re.sub(r"\.cpp\.o$|\.o$", "", m) for m in members)
        if mangled in system:
            rows.append(Row("système", system_owner(mangled, system_by_library), "—", pretty, users, []))
            continue
        if mangled in third:
            rows.append(Row("tiers", third[mangled], "K7", pretty, users, []))
            continue
        qualified = strip_template_and_args(pretty)
        origin = defining_files(qualified, sources, compiled)
        declared = declaring_headers(qualified, headers)
        header_package = next((HEADER_PACKAGE[h] for h in declared if h in HEADER_PACKAGE), None)
        if origin:
            rows.append(Row("kyty", " | ".join(origin), header_package or package_of(origin[0]), pretty, users, declared))
            continue
        prefix_match = next((lib for prefix, lib in THIRD_PARTY_PREFIXES if qualified.startswith(prefix)), None)
        if prefix_match:
            rows.append(Row("tiers", prefix_match, "K7", pretty, users, []))
        elif declared:
            rows.append(Row("kyty", "déclaré seulement : " + declared[0], header_package or "?", pretty, users, declared))
        else:
            rows.append(Row("inconnu", "?", "?", pretty, users, declared))

    if args.json:
        args.json.write_text(json.dumps([row._asdict() for row in rows], ensure_ascii=False, indent=1))
    if args.markdown:
        args.markdown.write_text(markdown(rows, args), encoding="utf-8")
    counts = collections.Counter((row.kind, row.package) for row in rows)
    for (kind, package), count in sorted(counts.items()):
        print(f"{kind}\t{package}\t{count}")
    return 1 if counts.get(("inconnu", "?")) or any(row.package == "?" for row in rows) else 0


def system_owner(symbol, system_by_library):
    for library, names in system_by_library.items():
        if symbol in names:
            return library
    return "crt"


def cell(text):
    return text.replace("|", "\\|")


def users_cell(users, limit=6):
    shown = ", ".join(f"`{u}`" for u in users[:limit])
    return shown + (f" et {len(users) - limit} autres" if len(users) > limit else "")


def markdown(rows, args):
    out = []
    kyty = [row for row in rows if row.kind == "kyty"]
    third = [row for row in rows if row.kind == "tiers"]
    system = [row for row in rows if row.kind == "système"]
    unknown = [row for row in rows if row.kind == "inconnu"]
    out.append("| Paquet | Symboles | Fichiers d'origine |")
    out.append("|---|---|---|")
    by_package = collections.defaultdict(list)
    for row in kyty + third:
        by_package[row.package].append(row)
    for package in sorted(by_package, key=lambda p: (len(p), p)):
        origins = sorted({row.origin for row in by_package[package]})
        out.append(f"| {package} | {len(by_package[package])} | {len(origins)} |")
    out.append(f"| système (aucun) | {len(system)} | — |")
    out.append("")
    for package in sorted({row.package for row in kyty}, key=lambda p: (len(p), p)):
        out.append(f"### {package}")
        out.append("")
        by_origin = collections.defaultdict(list)
        for row in kyty:
            if row.package == package:
                by_origin[row.origin].append(row)
        for origin in sorted(by_origin):
            out.append(f"**`{origin}`**" if not origin.startswith("déclaré") else f"**{origin}**")
            out.append("")
            out.append("| Symbole | Déclaré dans (contrat) | Appelé depuis (objets de kyty_gpu) |")
            out.append("|---|---|---|")
            for row in by_origin[origin]:
                declared = ", ".join(f"`{h}`" for h in row.declared) or "—"
                out.append(f"| `{cell(row.symbol)}` | {declared} | {users_cell(row.users)} |")
            out.append("")
    out.append("### K7 : bibliothèques tierces à lier")
    out.append("")
    by_library = collections.defaultdict(list)
    for row in third:
        by_library[row.origin].append(row)
    for library in sorted(by_library):
        out.append(f"**{library}** ({len(by_library[library])}) : " + ", ".join(f"`{cell(row.symbol)}`" for row in by_library[library]))
        out.append("")
    if unknown:
        out.append("### Sans origine trouvée")
        out.append("")
        for row in unknown:
            out.append(f"- `{cell(row.symbol)}` : {users_cell(row.users)}")
        out.append("")
    out.append("### Système (résolu par l'édition des liens, aucun paquet)")
    out.append("")
    by_library = collections.defaultdict(list)
    for row in system:
        by_library[row.origin].append(row)
    for library in sorted(by_library):
        out.append(f"**{library}** ({len(by_library[library])}) : " + ", ".join(f"`{cell(row.symbol)}`" for row in by_library[library]))
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
