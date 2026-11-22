"""Which settings make llama-server answer with garbage ("????...") on Inkling 2 Mini with host experts.

  python scripts/diag_garbage.py   -> results/next/garbage_diag.csv

Each case: one server, the 1150-token bench prompt, 96 generated tokens, greedy. Default load mode (mmap), so the
shared GPU memory counter shows a VRAM overflow.
"""
import collections
import csv
import os
import re

from llsrv import MODEL, WORK, Server, gpu_shared_mib, http

BASE = ["-m", MODEL, "--parallel", "1", "--jinja", "-t", "16", "-lv", "4", "-ngl", "99"]
CASES = [
    ("cmoe ub32 c4096", ["-cmoe", "-ub", "32", "-c", "4096"]),
    ("cmoe ub33 c4096", ["-cmoe", "-ub", "33", "-c", "4096"]),
    ("cmoe ub64 c4096", ["-cmoe", "-ub", "64", "-c", "4096"]),
    ("cmoe ub512 c4096", ["-cmoe", "-ub", "512", "-c", "4096"]),
    ("cmoe ub64 c4096 cache2048", ["-cmoe", "-ub", "64", "-c", "4096", "--moe-cache-mib", "2048"]),
    ("ncmoe13 ub64 c4096", ["-ncmoe", "13", "-ub", "64", "-c", "4096"]),
    ("ncmoe13 ub512 c4096", ["-ncmoe", "13", "-ub", "512", "-c", "4096"]),
    ("cmoe ub64 c4096 short prompt", ["-cmoe", "-ub", "64", "-c", "4096"]),
]


def main():
    prompt = open(os.path.join(WORK, "scripts", "workloads", "bench_prompt.txt"), encoding="utf-8").read()
    rows = []
    for name, args in CASES:
        d = os.path.join(WORK, "runs", "diag_garbage", name.replace(" ", "_"))
        os.makedirs(d, exist_ok=True)
        srv = Server(BASE + args, os.path.join(d, "server.log"), env={"LLAMA_MOE_CACHE_SHARE": ""})
        shared0 = gpu_shared_mib()
        row = {"case": name, "args": " ".join(args)}
        try:
            srv.start()
            p = "Explain in three sentences how a binary search works." if "short prompt" in name else prompt
            r = http(srv.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 96,
                                                        "temperature": 0, "seed": 1, "cache_prompt": False})
            m = r["choices"][0]["message"]
            text = (m.get("reasoning_content") or "") + (m.get("content") or "")
            row.update({
                "prompt_n": r["timings"]["prompt_n"], "prompt_tps": round(r["timings"]["prompt_per_second"], 1),
                "eval_tps": round(r["timings"]["predicted_per_second"], 2),
                "top_char_share_of_head": round(collections.Counter(text[:60]).most_common(1)[0][1] / max(len(text[:60]), 1), 2) if text else 1.0,
                "garbage": int(not text.strip() or collections.Counter(text[:60]).most_common(1)[0][1] / len(text[:60]) > 0.5),
                "text_head": text[:60].replace("\n", " "),
                "shared_growth_mib": round(gpu_shared_mib() - shared0),
            })
        except Exception as e:  # keep going, a crash is also a result
            row.update({"garbage": "", "text_head": f"ERROR {type(e).__name__}: {e}"[:80]})
        finally:
            srv.stop()
        log = srv.log()
        m = re.search(r"CUDA0 compute buffer size =\s+([0-9.]+) MiB", log)
        row["compute_buffer_mib"] = m.group(1) if m else ""
        rows.append(row)
        print(row, flush=True)
    out = os.path.join(WORK, "results", "next", "garbage_diag.csv")
    fields = ["case", "args", "garbage", "top_char_share_of_head", "prompt_n", "prompt_tps", "eval_tps", "compute_buffer_mib", "shared_growth_mib", "text_head"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
