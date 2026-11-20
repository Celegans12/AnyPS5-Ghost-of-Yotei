from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from test_optional_plt import fixture

LOAD_RVA = 0x10000
MARK_OFFSET = 0x28
LEAF = 0x240
GROWING_LEAF = 0x280
PROLOGUE = bytes.fromhex("53 41 56 41 57")
EXIT = bytes.fromhex("41 5f 41 5e 5b c3")
RECIPROCAL = bytes.fromhex("c5 f8 53 c1")
STORE_BELOW_RSP = bytes.fromhex("c5 f8 11 44 24 e0")
STORE_BELOW_RSP_MOVED = bytes.fromhex("c5 f8 11 44 24 00")
LOAD_FAR_ABOVE_RSP = bytes.fromhex("c5 f8 10 54 24 50")
LOAD_BELOW_RSP = bytes.fromhex("48 8b 44 24 e0")
STORE_DEEPER_BELOW_RSP = bytes.fromhex("48 89 7c 24 c0")
LOAD_DEEPER_BELOW_RSP = bytes.fromhex("48 8b 44 24 c0")


def call(source, target):
    return b"\xe8" + struct.pack("<i", target - (source + 5))


def leaf():
    return PROLOGUE + RECIPROCAL + STORE_BELOW_RSP + LOAD_BELOW_RSP + EXIT


def growing_leaf():
    return PROLOGUE + STORE_DEEPER_BELOW_RSP + RECIPROCAL + LOAD_FAR_ABOVE_RSP + LOAD_DEEPER_BELOW_RSP + EXIT


def build(growing):
    image = fixture()
    entry = call(0x210, LEAF)
    if growing:
        entry += call(0x215, GROWING_LEAF)
    entry += b"\xb8\x2a\x00\x00\x00\xc3"
    image[0x210:0x210 + len(entry)] = entry
    image[0x220:0x240] = b"\xcc" * 0x20
    image[LEAF:LEAF + len(leaf())] = leaf()
    image[LEAF + len(leaf()):GROWING_LEAF] = b"\xcc" * (GROWING_LEAF - LEAF - len(leaf()))
    if growing:
        image[GROWING_LEAF:GROWING_LEAF + len(growing_leaf())] = growing_leaf()
    return image


def sections(data):
    pe = struct.unpack_from("<I", data, 0x3c)[0]
    count, = struct.unpack_from("<H", data, pe + 6)
    table = pe + 24 + struct.unpack_from("<H", data, pe + 20)[0]
    found = {}
    for index in range(count):
        header = table + index * 40
        name = data[header:header + 8].rstrip(b"\0").decode()
        size, rva, raw_size, raw = struct.unpack_from("<IIII", data, header + 8)
        found.setdefault(name, []).append((rva, size, data[raw:raw + min(size, raw_size)]))
    return found


def convert(work, name, image, options):
    source = work / (name + ".elf")
    output = work / (name + ".exe")
    source.write_bytes(image)
    result = subprocess.run([str(relinker), "--skip-sce-module", "--windows", *options, str(source), str(output)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, (name, result.returncode, result.stdout, result.stderr)
    return result.stdout + result.stderr, output.read_bytes()


def mark(data):
    return data[MARK_OFFSET:MARK_OFFSET + 8], struct.unpack_from("<II", data, MARK_OFFSET + 8)


def main():
    global relinker
    relinker = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="anyps5-red-zone-ranges-") as directory:
        work = Path(directory)

        log, data = convert(work, "swallowed", build(False), ["--to-intel", "--frame-red-zone"])
        assert "Intel conversion: 0 in place, 1 stubs, 0 kept" in log, log
        assert "skipped" not in log, log
        assert "1 rsp operands that Intel conversion stubs took in from the instructions after an instruction they lower were moved in those stubs" in log, log
        assert "the executable is marked as framed (0 leaves or code ranges left); 0 of 0 guest modules are not marked" in log, log
        assert mark(data)[0] == b"APS5RZ01" and ".rzgap" not in sections(data)
        stub = sections(data)[".amdstub"][0][2]
        assert stub.count(STORE_BELOW_RSP_MOVED) == 1 and stub.count(STORE_BELOW_RSP) == 0, "the stub that took the store in does not carry the moved displacement"

        log, data = convert(work, "growing", build(True), ["--to-intel", "--frame-red-zone"])
        assert "Intel conversion: 0 in place, 2 stubs, 0 kept" in log, log
        assert "skipped 1: overlaps another stub (at 0x280 after \"rsp operand in another stub that cannot move\")" in log, log
        assert "1 code ranges (31 bytes) keep their red zone and go into the mark of the image" in log, log
        assert "the executable is marked as framed except the ranges it lists (1 leaves or code ranges left, 1 listed in the mark); 0 of 0 guest modules are not marked" in log, log
        assert "libc arms no page tracking block" not in log, log
        identifier, (rva, count) = mark(data)
        assert identifier == b"APS5RZ02" and count == 1, (identifier, rva, count)
        table = sections(data)[".rzgap"]
        assert len(table) == 1 and table[0][0] == rva and table[0][1] == 8, table
        assert struct.unpack("<II", table[0][2][:8]) == (LOAD_RVA + GROWING_LEAF, LOAD_RVA + GROWING_LEAF + len(growing_leaf()))
        pe = struct.unpack_from("<I", data, 0x3c)[0]
        assert struct.unpack_from("<I", data, pe + 24 + 56)[0] == rva + 0x1000
        stub = sections(data)[".amdstub"][0][2]
        assert stub.count(STORE_BELOW_RSP_MOVED) == 1 and stub.count(LOAD_FAR_ABOVE_RSP) == 1, "the stub of the unframed leaf was edited"

        log, data = convert(work, "plain", build(True), ["--frame-red-zone"])
        assert "skipped" not in log and mark(data)[0] == b"APS5RZ01" and ".rzgap" not in sections(data), log

        log, data = convert(work, "unmarked", build(True), ["--to-intel", "--no-frame-red-zone"])
        assert mark(data)[0] == bytes(8) and ".rzgap" not in sections(data)
    print("Windows red-zone ranges test passed")


if __name__ == "__main__":
    main()
