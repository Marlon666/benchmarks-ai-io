import os
import argparse
from typing import Any, Dict

from benchmarks_common.cli import load_yaml_config, parse_bool
from benchmarks_common.metadata import build_metadata
from benchmarks_common.outputs import write_csv, write_yaml
from benchmarks_common.stats import percentile, safe_median

from .synthetic_tree import make_tree
from .fs_lister import list_tree


def main():
    p = argparse.ArgumentParser(
        description="Listing Emulated Benchmark — metadata and enumeration I/O")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--run-name", type=str, default="leb-run")
    p.add_argument("--root", type=str, default="./data/emulated_tree")
    p.add_argument("--entries-per-dir", type=int, default=2000)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--warm-cache", type=str, default="false")
    p.add_argument("--use-stat", type=str, default="true",
                   help="Call os.stat() on every entry for real metadata I/O")
    p.add_argument("--use-scandir", type=str, default="false",
                   help="Use os.scandir() instead of os.walk()")
    p.add_argument("--outdir", type=str, default="metrics")
    args = p.parse_args()

    # YAML override
    if args.config:
        cfg = load_yaml_config(args.config)
        args.run_name = cfg.get("run", {}).get("name", args.run_name)
        args.root = cfg.get("data", {}).get("root", args.root)
        args.entries_per_dir = cfg.get("data", {}).get("entries_per_dir", args.entries_per_dir)
        args.depth = cfg.get("data", {}).get("depth", args.depth)
        args.concurrency = cfg.get("benchmark", {}).get("concurrency", args.concurrency)
        args.page_size = cfg.get("benchmark", {}).get("page_size", args.page_size)
        args.warm_cache = str(cfg.get("benchmark", {}).get("warm_cache", args.warm_cache)).lower()
        args.use_stat = str(cfg.get("benchmark", {}).get("use_stat", args.use_stat))
        args.use_scandir = str(cfg.get("benchmark", {}).get("use_scandir", args.use_scandir))
        args.outdir = cfg.get("output", {}).get("dir", args.outdir)

    use_stat = parse_bool(args.use_stat)
    use_scandir = parse_bool(args.use_scandir)

    # Prepare output dirs
    run_dir = os.path.join(args.outdir, args.run_name)
    raw_csv = os.path.join(run_dir, "listing_raw.csv")
    summary_yaml = os.path.join(run_dir, "listing_summary.yaml")
    meta_yaml = os.path.join(run_dir, "metadata.yaml")

    # Build synthetic tree if missing
    if not os.path.exists(args.root):
        make_tree(args.root, args.entries_per_dir, args.depth)

    records = list_tree(
        args.root, args.page_size, args.concurrency,
        use_stat=use_stat, use_scandir=use_scandir,
    )
    durations = [float(r[1]) - float(r[0]) for r in records]
    total_entries = sum(int(r[2]) for r in records)
    total_time = sum(durations) if durations else 0.0

    ttfb = durations[0] if durations else 0.0
    entries_per_sec = (total_entries / total_time) if total_time > 0 else 0.0

    summary: Dict[str, Any] = {
        "entries": total_entries,
        "entries_per_sec": round(entries_per_sec, 2),
        "p50_call_sec": round(safe_median(durations), 6),
        "p95_call_sec": round(percentile(durations, 0.95), 6),
        "p99_call_sec": round(percentile(durations, 0.99), 6),
        "ttfb_sec": round(ttfb, 6),
        "amplification": round(len(records) / max(1, (total_entries / max(1, args.page_size))), 4),
        "concurrency": args.concurrency,
        "page_size": args.page_size,
        "warm_cache": args.warm_cache,
        "use_stat": use_stat,
        "use_scandir": use_scandir,
    }

    write_csv(raw_csv, records)
    write_yaml(summary_yaml, summary)
    write_yaml(meta_yaml, build_metadata(
        run_name=args.run_name,
        benchmark="listing",
        parameters=vars(args),
        summary=summary,
    ))


if __name__ == "__main__":
    main()
