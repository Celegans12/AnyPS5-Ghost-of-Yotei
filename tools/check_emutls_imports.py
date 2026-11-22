import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SYMBOL = "__emutls_get_address"
LIBGCC = re.compile(r"^libgcc_s_.*\.dll$", re.IGNORECASE)
DLL_NAME = re.compile(r"^\s*DLL Name:\s*(\S+)\s*$", re.IGNORECASE)
SYMBOL_NAME = re.compile(rf"\b{SYMBOL}\b")

ALLOWED = {
    "libSceAvPlayer.native.prx",
    "libSceAvPlayer.prx",
    "libSceFiber.prx",
    "libSceFios2.prx",
    "libSceFont-module.prx",
    "libSceFont.native.prx",
    "libSceFont.prx",
    "libSceIcu.prx",
    "libSceJobManager.prx",
    "libSceJpegDec.prx",
    "libSceJpegEnc.prx",
    "libSceJpegParser.prx",
    "libSceJsc.prx",
    "libSceJson2.prx",
    "libSceLibcInternal.prx",
    "libSceLibreSsl.prx",
    "libSceNKWebKit.prx",
    "libSceNKWebKitRequirements.prx",
    "libSceNet.prx",
    "libSceNgs2.native.prx",
    "libSceNgs2.prx",
    "libSceNpCommon.prx",
    "libScePngDec.prx",
    "libScePngEnc.prx",
    "libScePosixForWebKit.prx",
    "libSceRudp.prx",
    "libSceS3da.prx",
    "libSceSsl.prx",
    "libSceUlt.prx",
    "libSceUsbd.prx",
    "libSceUserService.prx",
    "libSceVcodec.prx",
    "libSceVideoOut.prx",
    "libSceZlib.prx",
    "libc.prx",
    "libcairo.prx",
    "libharfbuzz.prx",
    "libkernel.prx",
    "libkernel_sys.prx",
    "libwebrtc.prx",
}


def imported_from(text):
    dll = None
    for line in text.splitlines():
        match = DLL_NAME.match(line)
        if match:
            dll = match.group(1)
        elif dll and LIBGCC.match(dll) and SYMBOL_NAME.search(line):
            return dll
    return None


def module_imports(path, objdump):
    result = subprocess.run([objdump, "-p", str(path)], capture_output=True)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{path}: objdump -p failed: {message}")
    return imported_from(result.stdout.decode("utf-8", errors="replace"))


def pe_modules(directory):
    if not directory.is_dir():
        raise RuntimeError(f"{directory}: not a directory")
    modules = sorted(directory.glob("*.prx"))
    if not modules:
        raise RuntimeError(f"{directory}: no .prx modules; build the libs target first")
    found = []
    for module in modules:
        with module.open("rb") as handle:
            if handle.read(2) == b"MZ":
                found.append(module)
    if not found:
        raise RuntimeError(f"{directory}: no PE modules; this check needs a Windows build")
    return found


def find_objdump(value):
    candidate = Path(value)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(value)
    if found is None:
        raise RuntimeError(f"objdump not found: {value}")
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description="fail when a module imports the GCC emulated TLS helper")
    parser.add_argument("directory", nargs="?", type=Path, default=Path("build/core/libs/libs"))
    parser.add_argument("--objdump", default=os.environ.get("OBJDUMP", "objdump"))
    parser.add_argument("--print-allowed", action="store_true", help="print the allow-list of the modules found")
    args = parser.parse_args(argv)
    try:
        objdump = find_objdump(args.objdump)
        importers = {module.name: dll for module in pe_modules(args.directory) if (dll := module_imports(module, objdump))}
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.print_allowed:
        print("ALLOWED = {")
        for name in sorted(importers):
            print(f'    "{name}",')
        print("}")
        return 0
    new = sorted(set(importers) - ALLOWED)
    stale = sorted(ALLOWED - set(importers))
    for name in new:
        print(f"{name} imports {SYMBOL} from {importers[name]}", file=sys.stderr)
    if stale:
        print(f"note: allow-listed modules that no longer import {SYMBOL}: {', '.join(stale)}")
    if new:
        print(f"error: {len(new)} module(s) import {SYMBOL} and are not allow-listed", file=sys.stderr)
        return 1
    print(f"{len(importers)} module(s) import {SYMBOL}; all are allow-listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
