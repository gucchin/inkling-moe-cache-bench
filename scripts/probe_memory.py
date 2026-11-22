"""Load the server with several ubatch / cache settings and record the device buffers it reserves.

Loading only, no generation. Writes results/memory_probe.csv.
"""

import csv
import os
import re
import sys

from bench import COMMON
from llsrv import WORK, Server, gpu_shared_mib, gpu_used_mib

CONFIGS = [
    # label, extra args
    ("cmoe ub512 cache0", ["-ngl", "99", "-cmoe"]),
    ("cmoe ub512 cache2048", ["-ngl", "99", "-cmoe", "--moe-cache-mib", "2048"]),
    ("cmoe ub64 cache0", ["-ngl", "99", "-cmoe", "-ub", "64"]),
    ("cmoe ub32 cache0", ["-ngl", "99", "-cmoe", "-ub", "32"]),
    ("cmoe ub32 cache2048", ["-ngl", "99", "-cmoe", "-ub", "32", "--moe-cache-mib", "2048"]),
    ("cmoe ub16 cache0", ["-ngl", "99", "-cmoe", "-ub", "16"]),
    ("cmoe ub16 cache2048", ["-ngl", "99", "-cmoe", "-ub", "16", "--moe-cache-mib", "2048"]),
]


def grab(log, pattern):
    return round(sum(float(x) for x in re.findall(pattern, log)), 1)


def main():
    rows = []
    for label, extra in CONFIGS:
        shared0 = gpu_shared_mib()
        before = gpu_used_mib()
        srv = Server(COMMON + extra, os.path.join(WORK, "runs", "memory_probe", label.replace(" ", "_") + ".log"))
        row = {"config": label, "server_args": " ".join(extra)}
        try:
            srv.start()
            row["vram_delta_mib"] = gpu_used_mib() - before
            row["shared_growth_mib"] = round(gpu_shared_mib() - shared0)
        except RuntimeError as e:
            row["error"] = str(e)[:120]
        finally:
            srv.stop()
        log = srv.log()
        row["model_mib"] = grab(log, r"CUDA0 model buffer size =\s+([0-9.]+) MiB")
        row["kv_mib"] = grab(log, r"CUDA0 KV buffer size =\s+([0-9.]+) MiB")
        row["compute_mib"] = grab(log, r"CUDA0 compute buffer size =\s+([0-9.]+) MiB")
        row["moe_cache_mib"] = grab(log, r"CUDA0 MoE cache size =\s+([0-9.]+) MiB")
        m = re.search(r"- (\d+) MiB free", log)
        row["free_at_load_mib"] = m.group(1) if m else ""
        rows.append(row)
        print(row, flush=True)
        for line in re.findall(r"llama_moe_cache.*|.*MoE cache.*", log)[:8]:
            print("   ", line[-160:])
    fields = ["config", "server_args", "free_at_load_mib", "model_mib", "kv_mib", "compute_mib", "moe_cache_mib", "vram_delta_mib", "shared_growth_mib", "error"]
    with open(os.path.join(WORK, "results", "memory_probe.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
