"""Generate and save all simulated datasets for model-comparison experiments.

Examples
--------
Smoke test:
    python generate_simulation_datasets.py --smoke-test

Full simulation:
    python generate_simulation_datasets.py --n 8000 --n-replications 200 --n-test 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_dgm import simulate_dataset, add_truth_on_grid, save_npz, THETA_TRUE
from simulation_io import DEFAULT_OUTPUT_DIR, ensure_dir, fixed_test_path, save_json, train_dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FLEXI-Haz paper-style simulation datasets.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--n-replications", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--censoring-mean", type=float, default=30.0)
    parser.add_argument("--eval-step", type=float, default=0.25)
    parser.add_argument("--base-seed", type=int, default=221100)
    parser.add_argument("--theta-true", type=str, default="1,-1",
                        help="true theta_0 as a comma-separated pair; setting 2 uses 1,-1 (the original study used 2,-1)")
    parser.add_argument("--test-seed", type=int, default=221100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.n = 200
        args.n_replications = 3
        args.n_test = 300
        args.output_dir = str(Path(args.output_dir) / "smoke_test")

    theta_true = np.asarray([float(x) for x in args.theta_true.split(",")], dtype=np.float64)
    if theta_true.shape != (2,):
        raise ValueError("--theta-true must have exactly two values")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "datasets")
    ensure_dir(output_dir / "test")

    eval_step = float(args.eval_step)

    eval_times_survival = np.arange(
        eval_step,
        float(args.tau) + 0.5 * eval_step,
        eval_step,
        dtype=float,
    )

    eval_times_metrics = np.arange(
        eval_step,
        float(args.tau),
        eval_step,
        dtype=float,
    )

    eval_times_survival = np.round(eval_times_survival, 10)
    eval_times_metrics = np.round(eval_times_metrics, 10)

    config = {
        "n": args.n,
        "n_replications": args.n_replications,
        "n_test": args.n_test,
        "tau": args.tau,
        "censoring_mean": args.censoring_mean,
        "base_seed": args.base_seed,
        "test_seed": args.test_seed,
        "theta_true": theta_true.tolist(),
        "eval_times_survival": eval_times_survival.tolist(),
        "eval_times_metrics": eval_times_metrics.tolist(),
        "note": "censoring_mean=30 corresponds to exponential censoring rate 1/30.",
    }
    save_json(output_dir / "config.json", config)

    test_path = fixed_test_path(output_dir)
    if args.overwrite or not test_path.exists():
        test_data = simulate_dataset(
            n=args.n_test,
            seed=args.test_seed,
            tau=args.tau,
            censoring_mean=args.censoring_mean,
            theta_true=theta_true,
        )
        test_data["eval_times_survival"] = eval_times_survival
        test_data["eval_times_metrics"] = eval_times_metrics
        test_data["true_survival"] = add_truth_on_grid(test_data, eval_times_survival)["true_survival"]
        save_npz(test_path, test_data)
        save_json(
            output_dir / "test" / "fixed_test_set_manifest.json",
            {
                "path": str(test_path),
                "seed": args.test_seed,
                "n_test": args.n_test,
                "event_rate": test_data["event_rate"],
                "observed_censoring_rate": test_data["observed_censoring_rate"],
                "random_censoring_rate": test_data["random_censoring_rate"],
                "administrative_censoring_rate": test_data["administrative_censoring_rate"],
            },
        )
    else:
        print(f"Fixed test set already exists, skipping: {test_path}")

    manifest_rows = []
    for rep_id in range(args.n_replications):
        seed = args.base_seed + rep_id
        path = train_dataset_path(output_dir, rep_id)
        if path.exists() and not args.overwrite:
            manifest_rows.append({"rep_id": rep_id, "seed": seed, "path": str(path), "status": "exists"})
            continue

        data = simulate_dataset(
            n=args.n,
            seed=seed,
            tau=args.tau,
            censoring_mean=args.censoring_mean,
            theta_true=theta_true,
        )
        save_npz(path, data)
        manifest_rows.append(
            {
                "rep_id": rep_id,
                "seed": seed,
                "path": str(path),
                "status": "generated",
                "event_rate": data["event_rate"],
                "observed_censoring_rate": data["observed_censoring_rate"],
                "random_censoring_rate": data["random_censoring_rate"],
                "administrative_censoring_rate": data["administrative_censoring_rate"],
            }
        )
        print(
            f"rep={rep_id:03d} event_rate={data['event_rate']:.3f} "
            f"censoring={data['observed_censoring_rate']:.3f} -> {path}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    print(f"Saved manifest to {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
