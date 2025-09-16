import os, time, random, argparse, statistics as stats
from typing import List
from .initialization import init_dist
from .outputs import write_csv_local, write_yaml_local
from .schema.run_metadata import default_metadata

def _p95(values: List[float]) -> float:
    if not values: return 0.0
    k = int(0.95 * (len(values)-1))
    return sorted(values)[k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="demo-run")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--gbs", type=int, default=64, help="global batch size")
    parser.add_argument("--mbs", type=int, default=8, help="micro batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["train","infer"], default="train")
    parser.add_argument("--outdir", type=str, default="metrics")
    args = parser.parse_args()

    dist = init_dist()
    rank = dist["RANK"]

    run_dir = os.path.join(args.outdir, args.run_name)
    per_mb_dir = os.path.join(run_dir, "per_microbatch_data_loading_time")
    per_step_dir = os.path.join(run_dir, "per_step_data_loading_time")

    # Simulate timings (seconds); replace with real measurements later.
    per_mb_rows, per_step_rows = [], []
    step_latencies = []

    microbatches_per_step = max(1, args.gbs // max(1, args.mbs))

    for step in range(args.steps):
        start = time.time()

        # Simulate microbatch I/O times (baseline +/- jitter)
        mb_times = []
        for mb in range(microbatches_per_step):
            t = random.uniform(0.005, 0.015)  # 5–15ms
            mb_times.append(t)
            per_mb_rows.append([step, mb, f"{t:.6f}"])
        # Simulate compute time loosely tied to mode
        compute = 0.01 if args.mode == "infer" else 0.02
        time.sleep(compute)  # lightweight sleep

        # Aggregate step latency
        step_latency = (time.time() - start) + sum(mb_times)
        step_latencies.append(step_latency)
        per_step_rows.append([step, f"{step_latency:.6f}"])

    # Write CSVs
    write_csv_local(os.path.join(per_mb_dir, f"{rank}.csv"), per_mb_rows)
    write_csv_local(os.path.join(per_step_dir, f"{rank}.csv"), per_step_rows)

    # Summaries
    throughput = args.gbs / (stats.mean(step_latencies) if step_latencies else 1.0)
    p95 = _p95(step_latencies)
    summary = {
        "throughput_samples_per_sec": round(throughput, 4),
        "p95_step_latency_sec": round(p95, 6),
        "steps": args.steps,
        "microbatches_per_step": microbatches_per_step,
    }

    meta = default_metadata(args.run_name, args.mode, args.steps, summary)
    write_yaml_local(os.path.join(run_dir, "metadata.yaml"), meta.__dict__)

if __name__ == "__main__":
    main()