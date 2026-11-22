"""Medians of results/next/*_raw.csv -> results/next/shared.csv (one row per label).

  python scripts/summarize_next.py results/next/shared_raw.csv [more_raw.csv ...] --out results/next/shared.csv
"""
import argparse
import csv
import statistics
from collections import OrderedDict


def output_ok(label):
    """0 if a saved response of the label is garbage (mostly question marks, or empty), "" if no response is saved

    A speed measured while the model answers "????..." says nothing, so every row carries this check.
    """
    import glob
    import json
    import os
    work = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = glob.glob(os.path.join(work, "runs", "bench", label, "response-*.json"))
    if not paths:
        return ""
    for p in paths:
        m = json.load(open(p, encoding="utf-8"))["choices"][0]["message"]
        t = (m.get("reasoning_content") or "") + (m.get("content") or "")
        # garbage seen so far is one character repeated ("????", "GGGG"); normal text has no character above ~20%
        if not t.strip() or max(t.count(c) for c in set(t)) / len(t) > 0.5:
            return 0
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--superseded", default="", help="labels of the first raw file that a later file measures again; kept with the suffix -first")
    a = ap.parse_args()
    superseded = set(filter(None, a.superseded.split(",")))
    groups = OrderedDict()
    for i, path in enumerate(a.raw):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if i == 0 and r["label"] in superseded:
                r["label"] += "-first"
            groups.setdefault(r["label"], []).append(r)
    rows = []
    for label, rs in groups.items():
        ev = [float(r["eval_tps"]) for r in rs]
        pr = [float(r["prompt_tps"]) for r in rs]
        parts = label.split("-")  # N-<off|on>-<size>-cpu<share>[-suffix]
        rows.append({
            "label": label, "share_slots": parts[1], "cache_mib": rs[0]["cache_mib"], "cpu_share": rs[0]["cpu_share"],
            "runs": len(rs), "prompt_tps": round(statistics.median(pr), 1), "eval_tps": round(statistics.median(ev), 2),
            "eval_spread_pct": round((max(ev) - min(ev)) / statistics.median(ev) * 100, 1),
            "hit_rate_small_pct": rs[0]["hit_rate_small_pct"], "uploaded_small_mib": rs[0]["uploaded_small_mib"],
            "misses_cpu_small": rs[0]["misses_cpu_small"],
            "output_ok": output_ok(label[:-6] if label.endswith("-first") else label),
            "vram_before_mib": rs[0]["vram_before_mib"], "vram_delta_mib": rs[0]["vram_delta_mib"],
            "server_args": rs[0]["server_args"],
        })
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['label']:28s} eval {r['eval_tps']:6.2f} (spread {r['eval_spread_pct']}%)  prompt {r['prompt_tps']:6.1f}  hit {r['hit_rate_small_pct']}%  vram0 {r['vram_before_mib']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
