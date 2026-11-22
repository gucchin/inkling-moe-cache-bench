"""Second pass of diag_garbage.py: context size x sampling, on the cache configuration used for the measurements.

  python scripts/diag_garbage2.py   -> results/next/garbage_diag2.csv
"""
import collections
import csv
import os

from llsrv import MODEL, WORK, Server, http

BASE = ["-m", MODEL, "--parallel", "1", "--jinja", "-t", "16", "-lv", "4", "-ngl", "99", "-cmoe", "-ub", "32",
        "--moe-cache-mib", "7296", "-lm", "none"]


def main():
    prompt = open(os.path.join(WORK, "scripts", "workloads", "bench_prompt.txt"), encoding="utf-8").read()
    rows = []
    for ctx in ("32768", "8192", "4096"):
        d = os.path.join(WORK, "runs", "diag_garbage2", "c" + ctx)
        os.makedirs(d, exist_ok=True)
        with Server(BASE + ["-c", ctx], os.path.join(d, "server.log"), env={"LLAMA_MOE_CACHE_SHARE": ""}) as srv:
            for name, extra in (("temp 0", {"temperature": 0}), ("default sampling", {}), ("temp 0, half prompt", {"temperature": 0})):
                p = prompt[: len(prompt) // 2] if "half" in name else prompt
                r = http(srv.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 96,
                                                            "seed": 1234, "cache_prompt": False, **extra})
                m = r["choices"][0]["message"]
                text = (m.get("reasoning_content") or "") + (m.get("content") or "")
                top = collections.Counter(text).most_common(1)[0] if text else ("", 0)
                rows.append({"ctx": ctx, "sampling": name, "prompt_n": r["timings"]["prompt_n"],
                             "top_char_share": round(top[1] / max(len(text), 1), 2), "garbage": int(top[1] / max(len(text), 1) > 0.5),
                             "text_head": text[:70].replace("\n", " ")})
                print(rows[-1], flush=True)
    out = os.path.join(WORK, "results", "next", "garbage_diag2.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
