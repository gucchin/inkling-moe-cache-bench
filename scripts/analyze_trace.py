"""Routing analysis of MoE cache traces (LLAMA_MOE_CACHE_TRACE, format "MCT1").

  python scripts/analyze_trace.py --trace coding=runs/trace/coding.trace --trace ja_chat=... --trace long_read=... \
      --sizes 1024,2048,4096,6144 --out results

Writes routing_summary.json and CSV files for the plots:
  (a) concentration curve          routing_concentration.csv
  (b) loop vs outer layers         routing_layers.csv
  (c) overlap between the passes   routing_pass_jaccard.csv
  (d) overlap between workloads    routing_workload_overlap.csv
  (e) hit rate by cache size       routing_hit_replay.csv

The trace reader and the cache simulator come from upstream scripts/moe-cache-tune.py.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import struct
from collections import Counter

import numpy as np

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("moe_cache_tune", os.path.join(WORK, "llama.cpp", "scripts", "moe-cache-tune.py"))
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)

# Inkling 2 Mini: 24 layers, layers [6, 18) run 3 times (model card, general.architecture inkling2_loop)
N_PHYS = 24
CORE_START = 6
CORE_END = 18


def layer_phys(il):
    """llama_model_inkling2_loop::layer_phys"""
    return il if il < N_PHYS else CORE_START + (il - N_PHYS) % (CORE_END - CORE_START)


def layer_pass(il):
    """0-based pass of a cache slot, 0 for the layers outside the loop core"""
    return 0 if il < N_PHYS else 1 + (il - N_PHYS) // (CORE_END - CORE_START)


def read_trace_fast(path):
    """like tune.read_trace(), records as (il, n_used, ids ndarray)"""
    n_layer, n_expert_used, groups, _ = read_header(path)
    with open(path, "rb") as f:
        data = f.read()
    pos = header_size(data)
    records = []
    n = len(data)
    while pos + 12 <= n:
        il, n_used, n_ids = struct.unpack_from("<3i", data, pos)
        pos += 12
        if pos + 4 * n_ids > n:
            break  # cut record, the server was killed
        records.append((il, n_used, np.frombuffer(data, dtype="<i4", count=n_ids, offset=pos)))
        pos += 4 * n_ids
    return n_layer, n_expert_used, groups, records


def header_size(data):
    pos = 4
    n_layer, n_expert_used, n_group = struct.unpack_from("<3I", data, pos)
    pos += 12
    for _ in range(n_group):
        _, _, n_layers = struct.unpack_from("<3I", data, pos)
        pos += 12 + 4 * n_layers + 16 + 24
        (n_tensors,) = struct.unpack_from("<I", data, pos)
        pos += 4 + 8 * n_tensors
    return pos


def read_header(path):
    """the header only, through the upstream reader on a copy without records"""
    with open(path, "rb") as f:
        data = f.read(1 << 20)
    if data[:4] != b"MCT1":
        raise SystemExit(f"{path}: not a MoE cache trace")
    tmp = path + ".hdr.tmp"
    with open(tmp, "wb") as f:
        f.write(data[: header_size(data)])
    try:
        return tune.read_trace(tmp)
    finally:
        os.remove(tmp)


def selections(records, n_expert_used):
    """per layer id of the trace: array [n_tokens, n_used] of the selected experts, in token order"""
    # one decode = the records from one lookup of the first layer to the next
    # a decode that misses layers is dropped, so that row t is the same token in every layer:
    # a group with fewer cache slots than a batch needs does not look up (and trace) that batch
    first = min(il for il, _, _ in records)
    layers = sorted({il for il, _, _ in records})
    decodes, cur = [], {}
    for il, n_used, ids in records:
        if il == first and cur:
            decodes.append(cur)
            cur = {}
        cur[il] = ids.reshape(-1, n_used)
    if cur:
        decodes.append(cur)
    # the loop core slots share one cache group, they are all traced or all missing
    core = [il for il in layers if CORE_START <= layer_phys(il) < CORE_END]
    full = [d for d in decodes if all(il in d for il in core) and len({len(d[il]) for il in core}) == 1]
    selections.dropped = len(decodes) - len(full)
    selections.partial = sum(1 for d in full if len(d) != len(layers))
    return {il: np.concatenate([d[il] for d in full if il in d]) for il in layers}


def numbering(sel, n_layer):
    """is the layer id of a record a cache slot (one per pass) or a physical layer?"""
    ils = sorted(sel)
    counts = {il: len(sel[il]) for il in ils}
    base = counts[min(ils)]
    info = {"trace_n_layer": n_layer, "distinct_il": len(ils), "max_il": max(ils), "tokens_layer0": base,
            "decodes_dropped_incomplete": getattr(selections, "dropped", 0),
            "decodes_without_some_outer_layers": getattr(selections, "partial", 0)}
    core_counts = {c for il, c in counts.items() if CORE_START <= layer_phys(il) < CORE_END}
    if max(ils) >= N_PHYS and len(core_counts) == 1:
        info["kind"] = "slot"
    elif max(ils) < N_PHYS and all(counts[il] == base * (3 if CORE_START <= il < CORE_END else 1) for il in ils):
        info["kind"] = "physical"
    else:
        info["kind"] = "unknown"
        info["counts"] = counts
    return info


def usage_counts(sel, n_expert):
    """uses per (slot, expert) and per (physical layer, expert)"""
    slot = np.zeros((max(sel) + 1, n_expert), dtype=np.int64)
    for il, a in sel.items():
        slot[il] = np.bincount(a.ravel(), minlength=n_expert)
    phys = np.zeros((N_PHYS, n_expert), dtype=np.int64)
    for il in sel:
        phys[layer_phys(il)] += slot[il]
    return slot, phys


def concentration(counts, traced_layers, n_expert):
    """share of all expert evaluations covered by the most used experts"""
    v = np.sort(counts[traced_layers].ravel())[::-1].astype(np.float64)
    cum = np.cumsum(v) / v.sum()
    n = len(v)
    out = {"n_experts": n, "n_never_used": int((v == 0).sum())}
    for pct in (5, 10, 20, 30, 50):
        out[f"top_{pct}pct_share"] = round(float(cum[max(1, round(n * pct / 100)) - 1]), 4)
    for share in (0.5, 0.8, 0.9, 0.95, 0.99):
        k = int(np.searchsorted(cum, share) + 1)
        out[f"experts_for_{int(share * 100)}pct"] = k
        out[f"experts_for_{int(share * 100)}pct_frac"] = round(k / n, 4)
    return out, cum


def entropy_stats(row):
    p = row[row > 0] / row.sum()
    h = float(-(p * np.log(p)).sum())
    return h / math.log(len(row)), math.exp(h)  # normalized entropy, effective number of experts


def jaccard_rows(a, b, n_expert):
    """Jaccard of the expert sets of each row of a and b, shape [n, n_used]"""
    ma = np.zeros((len(a), n_expert), dtype=bool)
    mb = np.zeros((len(b), n_expert), dtype=bool)
    np.put_along_axis(ma, a, True, axis=1)
    np.put_along_axis(mb, b, True, axis=1)
    inter = (ma & mb).sum(axis=1)
    union = (ma | mb).sum(axis=1)
    return inter / union, inter


def chance_jaccard(n_expert, k):
    """expected Jaccard of two uniform random k-subsets of n_expert"""
    return sum(math.comb(k, i) * math.comb(n_expert - k, k - i) / math.comb(n_expert, k) * i / (2 * k - i) for i in range(k + 1))


def pass_overlap(sel, n_expert):
    """(c) same token, same physical layer: the experts of pass 1, 2 and 3"""
    rows, all_j = [], {"p1_p2": [], "p2_p3": [], "p1_p3": []}
    all_i = {"p1_p2": [], "p2_p3": [], "p1_p3": []}
    union_sizes = []
    n_core = CORE_END - CORE_START
    for ip in range(CORE_START, CORE_END):
        s = [ip, N_PHYS + (ip - CORE_START), N_PHYS + n_core + (ip - CORE_START)]
        if not all(x in sel for x in s):
            continue
        p = [sel[x] for x in s]
        j12, i12 = jaccard_rows(p[0], p[1], n_expert)
        j23, i23 = jaccard_rows(p[1], p[2], n_expert)
        j13, i13 = jaccard_rows(p[0], p[2], n_expert)
        for name, v in (("p1_p2", i12), ("p2_p3", i23), ("p1_p3", i13)):
            all_i[name].append(v)
        m = np.zeros((len(p[0]), n_expert), dtype=bool)
        for a in p:
            np.put_along_axis(m, a, True, axis=1)
        u = m.sum(axis=1)
        union_sizes.append(u)
        # baseline: the next token in the same pass and layer
        adj = np.mean([jaccard_rows(a[:-1], a[1:], n_expert)[0].mean() for a in p])
        rows.append({"layer": ip, "jaccard_p1_p2": round(float(j12.mean()), 4), "jaccard_p2_p3": round(float(j23.mean()), 4),
                     "jaccard_p1_p3": round(float(j13.mean()), 4), "union_3_passes": round(float(u.mean()), 3),
                     "jaccard_next_token_same_pass": round(float(adj), 4)})
        all_j["p1_p2"].append(j12)
        all_j["p2_p3"].append(j23)
        all_j["p1_p3"].append(j13)
    if not rows:
        return None, []
    k = sel[CORE_START].shape[1]
    summary = {f"jaccard_{name}": round(float(np.concatenate(v).mean()), 4) for name, v in all_j.items()}
    summary.update({f"shared_experts_{name}": round(float(np.concatenate(v).mean()), 2) for name, v in all_i.items()})
    summary["jaccard_mean_of_pairs"] = round(float(np.mean([np.concatenate(v).mean() for v in all_j.values()])), 4)
    summary["union_3_passes_mean"] = round(float(np.concatenate(union_sizes).mean()), 3)
    summary["union_3_passes_min_possible"] = k
    summary["union_3_passes_max_possible"] = 3 * k
    summary["jaccard_next_token_same_pass"] = round(float(np.mean([r["jaccard_next_token_same_pass"] for r in rows])), 4)
    summary["jaccard_chance"] = round(chance_jaccard(n_expert, k), 4)
    return summary, rows


def rankdata(x):
    """average ranks"""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def hit_replay(trace, records, size_mib, merge_passes):
    """(e) hit rate of the upstream cache logic at a cache size, CPU share 0

    merge_passes: what-if, the three passes of a loop layer share one cache entry per expert
    """
    n_layer, n_expert_used, groups, _ = trace
    caches, group_of = [], {}
    for ig, g in enumerate(groups):
        if merge_passes:
            g2 = tune.Group()
            g2.__dict__.update(g.__dict__)
            g2.layers = sorted({layer_phys(il) for il in g.layers})
            g = g2
        n = tune.n_slots_of(g, size_mib, 0, n_expert_used)
        caches.append(tune.ClassLRU(g.layers, g.n_expert, n) if n > 0 else None)
        group_of.update((il, ig) for il in groups[ig].layers)

    first = min(il for il, _, _ in records)
    total_tokens = sum(len(ids) // n_used for il, n_used, ids in records if il == first)
    skip = total_tokens // 2
    seen = 0
    hits = misses = hits_warm = misses_warm = 0
    n_slots = [c.n_slots if c else 0 for c in caches]
    for il, n_used, ids in records:
        n_tokens = len(ids) // n_used
        if il == first:
            seen += n_tokens
        g, c = groups[group_of[il]], caches[group_of[il]]
        if c is None or n_tokens > tune.MAX_BATCH or min(n_tokens * n_used, g.n_expert) > c.n_slots:
            continue
        ids_l = ids.tolist()
        n_uniq = len(set(ids_l))
        n_fill, _ = c.plan(layer_phys(il) if merge_passes else il, ids_l, n_used, 0, lambda n_miss: 0)
        hits += n_uniq - n_fill
        misses += n_fill
        if seen > skip:
            hits_warm += n_uniq - n_fill
            misses_warm += n_fill
    return {
        "size_mib": size_mib, "merge_passes": merge_passes, "n_slots": sum(n_slots),
        "hit_rate_pct": round(100 * hits / max(hits + misses, 1), 2),
        "hit_rate_warm_pct": round(100 * hits_warm / max(hits_warm + misses_warm, 1), 2),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", required=True, help="name=path, repeat per workload")
    ap.add_argument("--sizes", default="", help="cache sizes in MiB for (e), comma separated")
    ap.add_argument("--out", default=os.path.join(WORK, "results"))
    ap.add_argument("--prefix", default="routing")
    ap.add_argument("--scope", choices=["all", "gen"], default="all",
                    help="tokens for (a)-(d): all traced tokens, or only single-token decodes (generation)")
    a = ap.parse_args()

    summary = {"model_loop": {"n_phys": N_PHYS, "core_start": CORE_START, "core_end": CORE_END}, "scope": a.scope, "workloads": {}}
    conc_rows, layer_rows, pass_rows, replay_rows = [], [], [], []
    phys_counts = {}

    for item in a.trace:
        name, path = item.split("=", 1)
        n_layer, n_expert_used, groups, records = read_trace_fast(path)
        trace = (n_layer, n_expert_used, groups, records)
        n_expert = groups[0].n_expert
        # (a)-(d) on the chosen tokens, (e) always replays everything the cache saw
        stat_records = [r for r in records if len(r[2]) // r[1] == 1] if a.scope == "gen" else records
        sel = selections(stat_records, n_expert_used)
        num = numbering(sel, n_layer)
        w = {"trace": os.path.basename(path), "records": len(records), "tokens": num["tokens_layer0"],
             "il_numbering": num, "n_expert": n_expert, "n_expert_used": n_expert_used,
             "batch_sizes": dict(sorted(Counter(len(ids) // n_used for _, n_used, ids in records).items())),
             "groups": [{"layers": g.layers, "n_expert": g.n_expert, "expert_mib": round(g.expert_bytes / 2**20, 3),
                         "host_mib_as_counted": round(g.host_bytes / 2**20, 1)} for g in groups]}

        slot, phys = usage_counts(sel, n_expert)
        phys_counts[name] = phys
        traced_phys = sorted({layer_phys(il) for il in sel})
        traced_slots = sorted(sel)

        # (a)
        c_phys, cum_phys = concentration(phys, traced_phys, n_expert)
        c_slot, cum_slot = concentration(slot, traced_slots, n_expert)
        w["a_concentration_physical"] = c_phys
        w["a_concentration_cache_keys"] = c_slot
        for view, cum in (("physical", cum_phys), ("cache_key", cum_slot)):
            n = len(cum)
            for pct in range(1, 101):
                conc_rows.append({"workload": name, "view": view, "top_pct": pct, "experts": max(1, round(n * pct / 100)),
                                  "share_of_evaluations": round(float(cum[max(1, round(n * pct / 100)) - 1]), 4)})

        # (b)
        loop = [il for il in traced_phys if CORE_START <= il < CORE_END]
        outer = [il for il in traced_phys if not CORE_START <= il < CORE_END]
        if loop and outer:
            m_loop, m_outer = phys[loop].mean(), phys[outer].mean()
            w["b_loop_vs_outer"] = {
                "mean_uses_per_expert_loop": round(float(m_loop), 2), "mean_uses_per_expert_outer": round(float(m_outer), 2),
                "ratio": round(float(m_loop / m_outer), 3),
                "effective_experts_loop_mean": round(float(np.mean([entropy_stats(phys[il])[1] for il in loop])), 1),
                "effective_experts_outer_mean": round(float(np.mean([entropy_stats(phys[il])[1] for il in outer])), 1),
            }
        for il in traced_phys:
            h, eff = entropy_stats(phys[il])
            row = {"workload": name, "layer": il, "in_loop": int(CORE_START <= il < CORE_END), "uses": int(phys[il].sum()),
                   "uses_per_token": round(float(phys[il].sum() / num["tokens_layer0"]), 2), "normalized_entropy": round(h, 4),
                   "effective_experts": round(eff, 1), "experts_never_used": int((phys[il] == 0).sum()),
                   "top8_share": round(float(np.sort(phys[il])[::-1][:8].sum() / phys[il].sum()), 4)}
            layer_rows.append(row)

        # (c)
        if num["kind"] == "slot":
            c_sum, rows = pass_overlap(sel, n_expert)
            w["c_pass_overlap"] = c_sum
            pass_rows += [dict(workload=name, **r) for r in rows]
        else:
            w["c_pass_overlap"] = None  # not measured: the trace does not tell the passes apart

        # (e)
        for size in [int(s) for s in a.sizes.split(",") if s]:
            for merge in (False, True):
                r = hit_replay(trace, records, size, merge)
                replay_rows.append(dict(workload=name, **r))
                print(name, r, flush=True)
        summary["workloads"][name] = w
        print(name, json.dumps({k: v for k, v in w.items() if k[:2] in ("a_", "b_", "c_")}, indent=1), flush=True)

    # (d)
    names = list(phys_counts)
    overlap_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = phys_counts[names[i]].ravel(), phys_counts[names[j]].ravel()
            row = {"workload_a": names[i], "workload_b": names[j], "spearman": round(spearman(x, y), 4)}
            for pct in (5, 10, 20):
                n = round(len(x) * pct / 100)
                ta, tb = set(np.argsort(-x, kind="stable")[:n].tolist()), set(np.argsort(-y, kind="stable")[:n].tolist())
                row[f"top_{pct}pct_overlap"] = round(len(ta & tb) / n, 4)
            overlap_rows.append(row)
    summary["d_workload_overlap"] = overlap_rows
    summary["e_hit_replay"] = replay_rows

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, a.prefix)
    write_csv(p + "_concentration.csv", conc_rows)
    write_csv(p + "_layers.csv", layer_rows)
    write_csv(p + "_pass_jaccard.csv", pass_rows)
    write_csv(p + "_workload_overlap.csv", overlap_rows)
    write_csv(p + "_hit_replay.csv", replay_rows)
    json.dump(summary, open(p + "_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", p + "_summary.json")


if __name__ == "__main__":
    main()
