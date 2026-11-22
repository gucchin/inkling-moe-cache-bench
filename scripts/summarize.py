"""Medians per configuration from the raw run CSVs.

Writes results/comparison.csv, results/cache_sweep.csv and results/summary.json.
"""

import csv
import json
import os
import statistics

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(WORK, "results")


def load(name):
    path = os.path.join(RES, name)
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.isfile(path) else []


def medians(rows):
    out = {}
    for r in rows:
        out.setdefault(r["label"], []).append(r)
    res = []
    for label, rs in out.items():
        ev = [float(r["eval_tps"]) for r in rs]
        pr = [float(r["prompt_tps"]) for r in rs]
        r0 = rs[0]
        res.append({
            "label": label, "server_args": r0["server_args"], "runs": len(rs),
            "prompt_tps": round(statistics.median(pr), 1), "eval_tps": round(statistics.median(ev), 2),
            "eval_spread_pct": round((max(ev) - min(ev)) / statistics.median(ev) * 100, 1),
            "vram_before_mib": r0["vram_before_mib"], "vram_delta_mib": r0["vram_delta_mib"],
            "shared_growth_mib": r0["shared_growth_mib"], "rss_peak_mib": r0["rss_peak_mib"],
            "cache_mib": r0["cache_mib"], "cpu_share": r0["cpu_share"],
            "hit_rate_pct": r0.get("hit_rate_small_pct") or "",
        })
    return res


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    base = medians(load("baseline.csv"))
    strata = medians(load("strata.csv"))
    sweep = medians(load("cache_sweep_raw.csv"))
    comparison = base + strata
    write("comparison.csv", comparison)

    by = {r["label"]: r for r in comparison}
    pinned = medians(load("cache_sweep_pinned_raw.csv"))
    sweep_rows = [{"host_memory": "pinned (-lm none)" if "-lm none" in r["server_args"] else "mmap (default)",
                   "cache_mib": int(r["cache_mib"]), "cpu_share": float(r["cpu_share"]), "hit_rate_pct": r["hit_rate_pct"],
                   "eval_tps": r["eval_tps"], "prompt_tps": r["prompt_tps"], "vram_delta_mib": r["vram_delta_mib"],
                   "eval_spread_pct": r["eval_spread_pct"], "label": r["label"]}
                  for r in sweep + pinned + [by[k] for k in ("S1", "S1-share0", "S2-lm-none-share0", "S2-lm-none") if k in by]]
    sweep_rows.sort(key=lambda r: (r["host_memory"], r["cpu_share"], r["cache_mib"]))
    sweep += pinned
    write("cache_sweep.csv", sweep_rows)

    def ratio(a, b):
        return round(by[a]["eval_tps"] / by[b]["eval_tps"], 2) if a in by and b in by else None

    summary = {
        "eval_tps": {k: v["eval_tps"] for k, v in by.items()},
        "prompt_tps": {k: v["prompt_tps"] for k, v in by.items()},
        "hit_rate_pct": {k: v["hit_rate_pct"] for k, v in by.items() if v["hit_rate_pct"]},
        "speedup": {
            "S1_vs_B1": ratio("S1", "B1"), "S1_vs_B2": ratio("S1", "B2"),
            "S2_vs_B1": ratio("S2-lm-none", "B1"), "S2_vs_B2": ratio("S2-lm-none", "B2"), "S2_vs_S1": ratio("S2-lm-none", "S1"),
            "B1_vs_B2": ratio("B1", "B2"),
            "S2_vs_B1_lm_none": ratio("S2-lm-none", "B1-lm-none"), "S2_share0_vs_B1_lm_none": ratio("S2-lm-none-share0", "B1-lm-none"),
        },
        "max_eval_spread_pct": max(r["eval_spread_pct"] for r in comparison + sweep),
    }
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))
    for r in sweep_rows:
        print(r)


if __name__ == "__main__":
    main()
