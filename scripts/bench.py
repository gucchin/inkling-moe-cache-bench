"""Measure one llama-server configuration: N identical chat requests, timings from the server.

  python scripts/bench.py --label B2 --csv results/baseline.csv -- -ngl 99 -cmoe

Common arguments (model, context, threads, --parallel 1, --jinja) are added here so that every
configuration shares them. Appends one row per run to the CSV and prints the median.
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys

from llsrv import MODEL, WORK, Server, gpu_shared_mib, http

COMMON = ["-m", MODEL, "-c", "32768", "--parallel", "1", "--jinja", "-t", "16", "-lv", "4"]  # -lv 4: the loader and the MoE cache log at this level
SEED = 1234
MAX_TOKENS = 512

# Fixed prompt. About 1k prompt tokens so that the prompt eval speed is also measured.
PROMPT_FILE = os.path.join(WORK, "scripts", "workloads", "bench_prompt.txt")

FIELDS = ["label", "run", "server_args", "prompt_n", "prompt_tps", "predicted_n", "eval_tps",
          "vram_before_mib", "vram_peak_total_mib", "vram_delta_mib", "shared_growth_mib", "rss_peak_mib", "private_peak_mib",
          "cache_mib", "cpu_share", "hit_rate_small_pct", "hits_small", "misses_small", "misses_cpu_small", "uploaded_small_mib",
          "finish_reason"]

STATS_RE = re.compile(r"llama_moe_cache: ubatch <= 8: hits = (\d+), misses = (\d+) \((\d+) on the CPU\), hit rate = ([0-9.]+)%, uploaded = ([0-9.]+) MiB")


def chat(port, prompt, max_tokens=MAX_TOKENS, seed=SEED):
    return http(port, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": seed,
        "cache_prompt": False,
        "stream": False,
    })


def arg_value(args, name, default=""):
    return next((args[i + 1] for i, a in enumerate(args[:-1]) if a == name), default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("server_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = a.server_args[1:] if a.server_args[:1] == ["--"] else a.server_args

    prompt = open(PROMPT_FILE, encoding="utf-8").read()
    run_dir = os.path.join(WORK, "runs", "bench", a.label)
    os.makedirs(run_dir, exist_ok=True)
    srv = Server(COMMON + extra, os.path.join(run_dir, "server.log"))
    rows = []
    shared0 = gpu_shared_mib()
    shared_growth = None
    try:
        srv.start()
        if not a.no_warmup:
            # short request with another prompt: loads the weights from disk, does not count
            chat(srv.port, "Say hello in one short sentence.", max_tokens=32, seed=1)
        for k in range(a.runs):
            r = chat(srv.port, prompt, a.max_tokens)
            t = r["timings"]
            json.dump(r, open(os.path.join(run_dir, f"response-{k + 1}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            rows.append({
                "label": a.label, "run": k + 1, "server_args": " ".join(extra),
                "prompt_n": t["prompt_n"], "prompt_tps": round(t["prompt_per_second"], 2),
                "predicted_n": t["predicted_n"], "eval_tps": round(t["predicted_per_second"], 2),
                "finish_reason": r["choices"][0]["finish_reason"],
            })
            print(f"{a.label} run {k + 1}: prompt {t['prompt_n']} tok {t['prompt_per_second']:.1f} t/s, "
                  f"eval {t['predicted_n']} tok {t['predicted_per_second']:.2f} t/s", flush=True)
        shared_growth = round(gpu_shared_mib() - shared0)
    finally:
        srv.stop()

    m = STATS_RE.search(srv.log())
    for row in rows:
        row.update({
            "vram_before_mib": srv.vram_before_mib,
            "vram_peak_total_mib": srv.peak["vram_total_mib"],
            "vram_delta_mib": srv.peak["vram_total_mib"] - srv.vram_before_mib,
            "shared_growth_mib": shared_growth, "rss_peak_mib": round(srv.peak["rss_mib"]),
            "private_peak_mib": round(srv.peak["private_mib"]),
            "cache_mib": arg_value(extra, "--moe-cache-mib", "0"),
            "cpu_share": arg_value(extra, "--moe-cache-cpu-share", "0"),
        })
        if m:
            row.update({"hits_small": m.group(1), "misses_small": m.group(2), "misses_cpu_small": m.group(3),
                        "hit_rate_small_pct": m.group(4), "uploaded_small_mib": m.group(5)})

    new = not os.path.isfile(a.csv)
    with open(a.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)

    ev = [r["eval_tps"] for r in rows]
    pr = [r["prompt_tps"] for r in rows]
    spread = (max(ev) - min(ev)) / statistics.median(ev) * 100
    print(f"{a.label} median: prompt {statistics.median(pr):.1f} t/s, eval {statistics.median(ev):.2f} t/s, "
          f"spread {spread:.1f}%, VRAM +{rows[0]['vram_delta_mib']} MiB, shared +{shared_growth} MiB, RSS peak {rows[0]['rss_peak_mib']} MiB"
          + (f", hit rate {m.group(4)}%" if m else ""))
    if spread > 20:
        print("WARNING: spread above 20%")
        sys.exit(2)


if __name__ == "__main__":
    main()
