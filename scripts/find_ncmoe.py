"""B1: binary search for the smallest -ncmoe N that fits in the VRAM of this machine.

"Fits" means: the server loads, generates 64 tokens, and the shared GPU memory (the Windows
overflow of dedicated VRAM into system RAM) does not grow by more than 256 MiB.
Each probe is logged to results/ncmoe_search.csv.
"""

import csv
import os
import sys

from bench import COMMON, PROMPT_FILE, chat
from llsrv import WORK, Server, gpu_shared_mib, gpu_used_mib

LIMIT_SHARED_GROWTH_MIB = 256
N_MAX = 24
UBATCH = sys.argv[1] if len(sys.argv) > 1 else "512"


def probe(n):
    shared0 = gpu_shared_mib()
    srv = Server(COMMON + ["-ngl", "99", "-ncmoe", str(n), "-ub", UBATCH], os.path.join(WORK, "runs", "ncmoe", f"ub{UBATCH}_n{n}.log"))
    row = {"ubatch": UBATCH, "ncmoe": n, "vram_before_mib": gpu_used_mib(), "shared_before_mib": round(shared0)}
    try:
        srv.start()
        r = chat(srv.port, open(PROMPT_FILE, encoding="utf-8").read(), max_tokens=128)
        shared1 = gpu_shared_mib()
        row.update(loaded=1, eval_tps=round(r["timings"]["predicted_per_second"], 2),
                   vram_peak_total_mib=max(srv.peak["vram_total_mib"], gpu_used_mib()),
                   shared_growth_mib=round(shared1 - shared0))
        row["fits"] = int(row["shared_growth_mib"] <= LIMIT_SHARED_GROWTH_MIB)
    except Exception as e:  # load failure counts as "does not fit"
        row.update(loaded=0, fits=0, error=str(e)[:200])
    finally:
        srv.stop()
    print(row, flush=True)
    return row


def main():
    rows = []
    lo, hi = 0, N_MAX  # hi is assumed to fit (all experts on the CPU), checked first
    r = probe(hi)
    rows.append(r)
    if not r["fits"]:
        lo = hi = -1  # nothing fits, still write the probe
    while lo < hi:
        mid = (lo + hi) // 2
        r = probe(mid)
        rows.append(r)
        if r["fits"]:
            hi = mid
        else:
            lo = mid + 1
    fields = ["ubatch", "ncmoe", "loaded", "fits", "eval_tps", "vram_before_mib", "vram_peak_total_mib", "shared_before_mib", "shared_growth_mib", "error"]
    with open(os.path.join(WORK, "results", f"ncmoe_search_ub{UBATCH}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("smallest N that fits:", hi)


if __name__ == "__main__":
    main()
