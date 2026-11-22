"""Step 6: record the routing of each workload with LLAMA_MOE_CACHE_TRACE.

  python scripts/run_trace.py --tag gen -- -ngl 99 -cmoe --moe-cache-mib 4096
  python scripts/run_trace.py --tag all --ubatch 8 -- -ngl 99 -cmoe --moe-cache-mib 4096

One server per workload, so each trace holds one kind of input. The trace only has the ubatches
of at most 32 tokens: with the default ubatch that is the generated tokens, with --ubatch 8 the
prompt tokens are recorded too.
"""

import argparse
import csv
import json
import os

from bench import COMMON, STATS_RE
from llsrv import WORK, Server, http

WORKLOADS = ["coding", "ja_chat", "long_read"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ubatch", type=int)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("server_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = a.server_args[1:] if a.server_args[:1] == ["--"] else a.server_args
    if a.ubatch:
        extra += ["-ub", str(a.ubatch)]

    out_dir = os.path.join(WORK, "runs", "trace", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name in a.workloads.split(","):
        prompts = [json.loads(l) for l in open(os.path.join(WORK, "scripts", "workloads", name + ".jsonl"), encoding="utf-8")]
        trace = os.path.join(out_dir, name + ".trace")
        srv = Server(COMMON + extra, os.path.join(out_dir, name + ".log"), env={"LLAMA_MOE_CACHE_TRACE": trace})
        with srv:
            for p in prompts:
                r = http(srv.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": p["prompt"]}],
                                                            "max_tokens": a.max_tokens, "seed": 1234, "cache_prompt": False})
                json.dump(r, open(os.path.join(out_dir, p["id"] + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                t = r["timings"]
                rows.append({"tag": a.tag, "workload": name, "id": p["id"], "prompt_n": t["prompt_n"], "predicted_n": t["predicted_n"],
                             "prompt_tps": round(t["prompt_per_second"], 2), "eval_tps": round(t["predicted_per_second"], 2),
                             "finish_reason": r["choices"][0]["finish_reason"]})
                print(rows[-1], flush=True)
        m = STATS_RE.search(srv.log())
        for row in rows:
            if row["workload"] == name:
                row["server_hit_rate_small_pct"] = m.group(4) if m else ""
        print(name, "trace bytes:", os.path.getsize(trace), "hit rate:", m.group(4) if m else "?", flush=True)

    path = os.path.join(WORK, "results", f"trace_runs_{a.tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
