import json
import re
import sys

INTERVALS = 6
KIND = re.compile(r"(\w+) (\d+) \(([\d.]+) us\)")


def summarize(directory):
    rows = [json.loads(line) for line in open(f"{directory}/aps5-summary.jsonl", encoding="utf-8") if line.startswith("{")]
    take = rows[-INTERVALS:]
    presents = sum(r["presents"] for r in take)
    ms = sum(r["window"]["ms"] for r in take)
    draws = sum(r["draws"] for r in take)
    gpu = sum(r["gpu_busy_ms_per_present"] * r["presents"] for r in take) / max(1, presents)
    fps = presents * 1000.0 / ms
    lines = []
    for line in open(f"{directory}/stderr.log", encoding="utf-8", errors="replace"):
        if line.startswith("[commits] by kind (10 s):"):
            lines.append({k: (int(n), float(u)) for k, n, u in KIND.findall(line)})
    last = lines[-3:]
    draw_n = sum(l["draw"][0] for l in last if "draw" in l)
    draw_us = sum(l["draw"][0] * l["draw"][1] for l in last if "draw" in l) / max(1, draw_n)
    all_us = sum(n * u for l in last for n, u in l.values())
    commit_ms_image = all_us / (len(last) * 10.0 * fps) / 1000.0
    return {"img_s": fps, "ms_image": 1000.0 / fps, "draws_image": draws / max(1, presents), "draws_s": draws * 1000.0 / ms,
            "gpu_ms": gpu, "us_commit_draw": draw_us, "commit_ms_image": commit_ms_image,
            "decode_ms": sum(r["phase_totals_ms"].get("Driver.Packet", 0.0) for r in take) / max(1, presents)}


for directory in sys.argv[1:]:
    print(directory.rstrip("/").split("/")[-1], json.dumps({k: round(v, 2) for k, v in summarize(directory).items()}))
