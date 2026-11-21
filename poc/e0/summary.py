#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict

FORMS = ["b", "bs", "bw", "bwr", "bwd", "bv", "b0", "bt", "bts", "bq", "bp"]


def load(paths):
    checks = defaultdict(dict)
    times = defaultdict(dict)
    stats = defaultdict(dict)
    order = []
    gpu = ""
    for path in paths:
        for line in open(path, encoding="utf-8", errors="replace"):
            at = line.find("[e0")
            if at < 0:
                continue
            line = line[at:].strip()
            m = re.match(r'\[e0\] gpu "([^"]+)" driver (\S+)', line)
            if m:
                gpu = f"{m.group(1)}, pilote {m.group(2)}"
                continue
            m = re.match(r"\[e0-check\] (\S+) (\S+) vs a: (\w+), (\d+) dwords differ of (\d+)", line)
            if m:
                checks[m.group(1)][m.group(2)] = (int(m.group(4)), int(m.group(5)))
                continue
            m = re.match(r"\[e0-time\] (\S+) form (\S+) batch (\d+) barrier (\d+) rounds (\d+) us_per_dispatch (\S+) p10 (\S+) p90 (\S+) "
                         r"ratio (\S+) compile_ms (\S+)", line)
            if m:
                if m.group(1) not in order:
                    order.append(m.group(1))
                times[m.group(1)][(m.group(2), int(m.group(4)))] = {"us": float(m.group(6)), "p10": float(m.group(7)),
                                                                     "p90": float(m.group(8)), "batch": int(m.group(3)),
                                                                     "compile_ms": float(m.group(10))}
                continue
            m = re.match(r"\[e0-stats\] (\S+) form (\S+) exe=(\S+) (.*)", line)
            if m:
                if m.group(1) not in order:
                    order.append(m.group(1))
                fields = dict(item.split("=", 1) for item in m.group(4).split() if "=" in item)
                if int(fields.get("stages", "0"), 16) == 1:
                    continue
                stats[m.group(1)][m.group(2)] = fields
    return gpu, order, checks, times, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--weights", help="fichier « programme ms_par_image » pour la moyenne pondérée")
    args = parser.parse_args()
    gpu, order, checks, times, stats = load(args.logs)
    weights = {}
    if args.weights:
        for line in open(args.weights, encoding="utf-8"):
            parts = line.split()
            if len(parts) >= 2 and not parts[0].startswith("#"):
                weights[parts[0]] = float(parts[1])
    print(f"GPU : {gpu}\n")
    head = ["programme", "lot", "A µs (série)", "A µs (chevauché)"]
    for form in FORMS:
        head += [f"{form}/A série", f"{form}/A chev."]
    head += ["sorties b", "sorties bp", "A deux fois"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    sums = {(form, barrier): [0.0, 0.0] for form in FORMS for barrier in (0, 1)}
    for name in order:
        t = times.get(name, {})
        if not t:
            continue
        a1, a0 = t.get(("a", 1)), t.get(("a", 0))
        row = [name, str((a1 or a0)["batch"]), f"{a1['us']:.2f}" if a1 else "—", f"{a0['us']:.2f}" if a0 else "—"]
        weight = next((w for key, w in weights.items() if key in name), None)
        for form in FORMS:
            for barrier, base in ((1, a1), (0, a0)):
                other = t.get((form, barrier))
                if base and other and base["us"] > 0:
                    ratio = other["us"] / base["us"]
                    row.append(f"{ratio:.3f}")
                    if weight is not None:
                        sums[(form, barrier)][0] += weight * ratio
                        sums[(form, barrier)][1] += weight
                else:
                    row.append("—")

        def chk(form):
            value = checks.get(name, {}).get(form)
            return "—" if value is None else ("identiques" if value[0] == 0 else f"{value[0]} mots sur {value[1]}")

        row += [chk("b"), chk("bp"), chk("a-again")]
        print("| " + " | ".join(row) + " |")
    if weights:
        row = ["**pondéré par le temps de Yōtei**", "", "", ""]
        for form in FORMS:
            for barrier in (1, 0):
                total, weight = sums[(form, barrier)]
                row.append(f"**{total / weight:.3f}**" if weight > 0 else "—")
        row += ["", "", ""]
        print("| " + " | ".join(row) + " |")
    print()
    head = ["programme"]
    for key, label in (("Register_Count", "registres"), ("Binary_Size", "code (octets)"), ("Local_Memory_Size", "débordement (octets)"),
                       ("Shared_Memory_Size", "partagée (octets)")):
        head += [f"{label} A", "B", "Bp"]
    head += ["compilation A ms", "B ms"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name in order:
        s = stats.get(name, {})
        if not s:
            continue
        row = [name]
        for key in ("Register_Count", "Binary_Size", "Local_Memory_Size", "Shared_Memory_Size"):
            for form in ("a", "b", "bp"):
                value = s.get(form, {}).get(key)
                if value is None:
                    row.append("—")
                else:
                    number = int(float(value))
                    row.append(str(number - (1 << 36) if key == "Local_Memory_Size" and number >= (1 << 36) else number))
        t = times.get(name, {})
        for form in ("a", "b"):
            entry = t.get((form, 1)) or t.get((form, 0))
            row.append(f"{entry['compile_ms']:.0f}" if entry else "—")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
