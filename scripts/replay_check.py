"""Replay the recorded traces with the server conditions and compare with the measured hit rates.

Mirrors the server counters: per unique expert of a lookup, hit / miss (uploaded or computed on the CPU),
in two buckets (ubatch <= 8 tokens, 9..32 tokens). The upstream simulator classes are reused unchanged.

Usage: python scripts/replay_check.py --size 7296 --share 0.95 --out results/next/replay_check.csv
"""
import argparse
import csv
import math
import os
import sys

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORK, "scripts"))
import analyze_trace as at  # noqa: E402

tune = at.tune

MEASURED = {"coding": 74.22, "ja_chat": 77.44, "long_read": 74.72}  # results/trace_runs_s2.csv, ubatch <= 8


def shared_groups(groups, rescale_bytes):
    """what-if groups: one cache entry per physical expert"""
    out = []
    for g in groups:
        g2 = tune.Group()
        g2.__dict__.update(g.__dict__)
        g2.layers = sorted({at.layer_phys(il) for il in g.layers})
        if rescale_bytes:
            g2.host_bytes = g.host_bytes * len(g2.layers) // len(g.layers)
        out.append(g2)
    if rescale_bytes:
        dev = sum(g.host_bytes for g in out)
        for g in out:
            g.dev_host_bytes = dev
    return out


def replay(trace, records, size_mib, share, mode):
    """mode: 'slot' (upstream), 'shared' (budget split as upstream), 'shared_rescaled' (budget split by physical bytes)"""
    n_layer, n_expert_used, groups, _ = trace
    share = tune.f32(share)
    share_q = int(math.floor(tune.f32(share * 65536.0) + 0.5))
    n_zero = min(n_expert_used, 64) if share > 0.0 else 0
    sim_groups = groups if mode == "slot" else shared_groups(groups, mode == "shared_rescaled")
    key = (lambda il: il) if mode == "slot" else at.layer_phys
    caches, group_of = [], {}
    for ig, g in enumerate(sim_groups):
        n = tune.n_slots_of(g, size_mib, n_zero, n_expert_used)
        caches.append(tune.ClassLRU(g.layers, g.n_expert, n) if n > 0 else None)
        group_of.update((il, ig) for il in groups[ig].layers)

    acc = {}
    stat = {"small": [0, 0, 0], "large": [0, 0, 0]}  # hits, uploaded, cpu
    skipped = 0
    for il, n_used, ids in records:
        n_tokens = len(ids) // n_used
        ig = group_of[il]
        g, c = sim_groups[ig], caches[ig]
        if c is None or n_tokens > tune.MAX_BATCH or min(n_tokens * n_used, g.n_expert) > c.n_slots:
            skipped += 1
            continue
        k = key(il)

        def n_cpu(n_miss):
            v = n_miss * (share_q if c.n_filled == c.n_slots else 0) + acc.get(k, 0)
            acc[k] = v & 0xFFFF
            return v >> 16

        ids_l = ids.tolist()
        n_uniq = len(set(ids_l))
        n_fill, n_c = c.plan(k, ids_l, n_used, n_zero, n_cpu)
        s = stat["small" if n_tokens <= 8 else "large"]
        s[0] += n_uniq - n_fill - n_c
        s[1] += n_fill
        s[2] += n_c

    def rate(s):
        return round(100 * s[0] / max(sum(s), 1), 2)

    both = [a + b for a, b in zip(stat["small"], stat["large"])]
    return {
        "mode": mode, "size_mib": size_mib, "cpu_share": float(share),
        "n_slots": "+".join(str(c.n_slots if c else 0) for c in caches),
        "hit_small_pct": rate(stat["small"]), "hit_large_pct": rate(stat["large"]), "hit_all_pct": rate(both),
        "lookups_small": sum(stat["small"]), "lookups_large": sum(stat["large"]),
        "uploaded_small": stat["small"][1], "cpu_small": stat["small"][2], "records_not_cached": skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="7296")
    ap.add_argument("--share", default="0.95")
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--out", default=os.path.join(WORK, "results", "next", "replay_check.csv"))
    args = ap.parse_args()
    rows = []
    for name in ("coding", "ja_chat", "long_read"):
        path = os.path.join(WORK, "runs", "trace", args.tag, f"{name}.trace")
        n_layer, n_used, groups, records = at.read_trace_fast(path)
        trace = (n_layer, n_used, groups, None)
        for size in [int(x) for x in args.size.split(",")]:
            for share in [float(x) for x in args.share.split(",")]:
                for mode in ("slot", "shared", "shared_rescaled"):
                    r = replay(trace, records, size, share, mode)
                    row = {"workload": name, **r}
                    row["measured_small_pct"] = MEASURED[name] if (mode == "slot" and size == 7296 and share == 0.95) else ""
                    rows.append(row)
                    print(row, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
