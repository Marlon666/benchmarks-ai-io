import os, argparse, yaml, statistics as stats
from typing import Dict, Any
from .synthetic_tree import make_tree
from .fs_lister import list_tree
from .outputs import write_csv, write_yaml
from .schema.run_metadata import default_metadata

def _pctl(values, pct):
    if not values: return 0.0
    values = sorted(values)
    k = int(pct * (len(values)-1))
    return values[k]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--run-name", type=str, default="leb-run")
    p.add_argument("--root", type=str, default="./data/emulated_tree")
    p.add_argument("--entries-per-dir", type=int, default=2000)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--warm-cache", type=str, default="false")
    p.add_argument("--outdir", type=str, default="metrics")
    args = p.parse_args()

    # YAML override
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        args.run_name = cfg.get("run", {}).get("name", args.run_name)
        args.root = cfg.get("data", {}).get("root", args.root)
        args.entries_per_dir = cfg.get("data", {}).get("entries_per_dir", args.entries_per_dir)
        args.depth = cfg.get("data", {}).get("depth", args.depth)
        args.concurrency = cfg.get("benchmark", {}).get("concurrency", args.concurrency)
        args.page_size = cfg.get("benchmark", {}).get("page_size", args.page_size)
        args.warm_cache = str(cfg.get("benchmark", {}).get("warm_cache", args.warm_cache)).lower()
        args.outdir = cfg.get("output", {}).get("dir", args.outdir)

    # Prepare output dirs
    run_dir = os.path.join(args.outdir, args.run_name)
    raw_csv = os.path.join(run_dir, "listing_raw.csv")
    summary_yaml = os.path.join(run_dir, "listing_summary.yaml")
    meta_yaml = os.path.join(run_dir, "metadata.yaml")

    # Build synthetic tree if missing
    if not os.path.exists(args.root):
        make_tree(args.root, args.entries_per_dir, args.depth)

    # First page emits TTFB; we approximate by first record duration.
    records = list_tree(args.root, args.page_size, args.concurrency)
    durations = [float(r[1]) - float(r[0]) for r in records]
    total_entries = sum(int(r[2]) for r in records)
    total_time = sum(durations) if durations else 0.0

    # Simple TTFB = first record duration (approximation for the demo).
    ttfb = durations[0] if durations else 0.0
    entries_per_sec = (total_entries / total_time) if total_time > 0 else 0.0

    summary = {
        "entries": total_entries,
        "entries_per_sec": round(entries_per_sec, 2),
        "p50_call_sec": round(stats.median(durations), 6) if durations else 0.0,
        "p95_call_sec": round(_pctl(durations, 0.95), 6) if durations else 0.0,
        "p99_call_sec": round(_pctl(durations, 0.99), 6) if durations else 0.0,
        "ttfb_sec": round(ttfb, 6),
        "amplification": round(len(records) / max(1, (total_entries / max(1, args.page_size))), 4),
        "concurrency": args.concurrency,
        "page_size": args.page_size,
        "warm_cache": args.warm_cache,
    }

    write_csv(raw_csv, records)
    write_yaml(summary_yaml, summary)
    write_yaml(meta_yaml, default_metadata(args.run_name, vars(args), summary))

if __name__ == "__main__":
    main()
