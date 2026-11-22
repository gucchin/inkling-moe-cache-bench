"""Correctness check of LLAMA_MOE_CACHE_SHARE: same greedy output with and without it, and the trace replays to the server hit rate.

  python scripts/check_shared.py            -> runs/check_shared/*, results/next/correctness.csv

Per (share off/on) x (cpu share 0 / 0.95): one server, greedy requests, the trace and the exit log.
"""
import csv
import json
import os
import re
import sys

from llsrv import WORK, Server, http
from bench import COMMON, STATS_RE

sys.path.insert(0, os.path.join(WORK, "scripts"))
import replay_check as rc  # noqa: E402

at = rc.at

BASE = ["-ngl", "99", "-cmoe", "-ub", "32", "--moe-cache-mib", "7296", "-lm", "none"]
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list. Then explain its time complexity.",
    "日本の四季について、それぞれの季節の特徴を一つずつ挙げて、短く説明してください。",
    "Explain how a hash table handles collisions. Compare chaining and open addressing.",
]
MAX_TOKENS = 320
OUT = os.path.join(WORK, "runs", "check_shared")


def run(share_slots, cpu_share):
    name = f"{'on' if share_slots else 'off'}-cpu{cpu_share}"
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    trace = os.path.join(d, "cache.trace")
    env = {"LLAMA_MOE_CACHE_TRACE": trace, "LLAMA_MOE_CACHE_SHARE": "1" if share_slots else ""}
    args = COMMON + BASE + (["--moe-cache-cpu-share", str(cpu_share)] if cpu_share > 0 else [])
    srv = Server(args, os.path.join(d, "server.log"), env=env)
    texts, tps = [], []
    try:
        srv.start()
        for i, p in enumerate(PROMPTS):
            r = http(srv.port, "/v1/chat/completions", {
                "messages": [{"role": "user", "content": p}], "max_tokens": MAX_TOKENS,
                "temperature": 0, "seed": 1, "cache_prompt": False, "stream": False,
            })
            json.dump(r, open(os.path.join(d, f"response-{i + 1}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            msg = r["choices"][0]["message"]
            texts.append((msg.get("reasoning_content") or "") + "\n---\n" + (msg.get("content") or ""))
            tps.append(round(r["timings"]["predicted_per_second"], 2))
    finally:
        srv.stop()
    log = srv.log()
    m = STATS_RE.search(log)
    host = re.search(r"MoE cache size =\s+([0-9.]+) MiB for ([0-9.]+) MiB of host experts", log)
    groups = re.findall(r"llama_moe_cache.*?: +(\d+ layers.*slots.*)", log)

    # replay the trace of this server with the upstream simulator
    n_layer, n_used, tgroups, records = at.read_trace_fast(trace)
    sim = rc.replay((n_layer, n_used, tgroups, None), records, 7296, cpu_share, "slot")
    return {
        "name": name, "share_slots": int(share_slots), "cpu_share": cpu_share,
        "host_experts_mib": host.group(2) if host else "", "cache_mib": host.group(1) if host else "",
        "groups": " | ".join(groups),
        "server_hit_small_pct": m.group(4) if m else "", "replay_hit_small_pct": sim["hit_small_pct"],
        "trace_layers": "+".join(str(len(g.layers)) for g in tgroups), "trace_records": len(records),
        "eval_tps": "/".join(str(x) for x in tps),
    }, texts


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def main():
    rows, texts = [], {}
    for cpu_share in (0, 0.95):
        for share_slots in (False, True):
            row, t = run(share_slots, cpu_share)
            texts[(share_slots, cpu_share)] = t
            rows.append(row)
            print(row, flush=True)
    for row in rows:
        ref = texts[(False, row["cpu_share"])]
        mine = texts[(bool(row["share_slots"]), row["cpu_share"])]
        diffs = [first_diff(a, b) for a, b in zip(ref, mine)]
        row["same_text_as_share_off"] = "/".join("same" if x < 0 else f"differs at char {x} of {len(a)}" for x, a in zip(diffs, ref))
    # across cpu shares, share off
    d = [first_diff(a, b) for a, b in zip(texts[(False, 0)], texts[(False, 0.95)])]
    print("share off, cpu share 0 vs 0.95:", d)
    out = os.path.join(WORK, "results", "next", "correctness.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(row["name"], row["same_text_as_share_off"], "server", row["server_hit_small_pct"], "replay", row["replay_hit_small_pct"])
    print("wrote", out)


if __name__ == "__main__":
    main()
