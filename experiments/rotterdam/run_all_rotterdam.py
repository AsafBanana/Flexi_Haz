"""Run the full 5-fold Rotterdam experiment."""

from __future__ import annotations

import argparse
from email import parser
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Rotterdam experiment models.")
    parser.add_argument("--data-path", default=str(ROOT / "rotterdam_breast_cancer.csv"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument(
        "--design",
        choices=["standard", "er_interaction"],
        default="standard",
        help="Rotterdam design: standard or ER-interaction treatment design.",
    )
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=5)
    parser.add_argument("--n-trials-flexihaz", type=int, default=30)
    parser.add_argument("--n-trials-rsf", type=int, default=50)
    parser.add_argument("--tuning-max-epochs", type=int, default=120)
    parser.add_argument("--max-epochs-final", type=int, default=2000)
    parser.add_argument("--projection-max-epochs", type=int, default=2000)
    parser.add_argument("--n-estimators-rsf", type=int, default=500)
    parser.add_argument("--n-jobs-rsf", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-tuning", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--include-zhong-placeholder", action="store_true")
    parser.add_argument("--include-zhong2022", action="store_true")
    parser.add_argument("--n-trials-zhong", type=int, default=100)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def maybe_flags(args: argparse.Namespace) -> list[str]:
    flags = []
    if args.overwrite:
        flags.append("--overwrite")
    if args.overwrite_tuning:
        flags.append("--overwrite-tuning")
    if args.save_predictions:
        flags.append("--save-predictions")
    return flags


def main() -> None:
    args = parse_args()
    base = [
        "--data-path", args.data_path,
        "--results-dir", args.results_dir,
        "--design", args.design,
        "--start-fold", str(args.start_fold),
        "--end-fold", str(args.end_fold),
        "--seed", str(args.seed),
    ]
    flags = maybe_flags(args)

    flexi_common = [
        *base,
        "--n-trials", str(args.n_trials_flexihaz),
        "--tuning-max-epochs", str(args.tuning_max_epochs),
        "--max-epochs", str(args.max_epochs_final),
        "--projection-max-epochs", str(args.projection_max_epochs),
        "--device", args.device,
        *flags,
    ]
    run([sys.executable, "run_flexihaz_pl_rotterdam.py", *flexi_common])
    run([sys.executable, "run_flexihaz_nonlinear_rotterdam.py", *flexi_common])

    run([sys.executable, "run_cox_ph_rotterdam.py", *base, *( ["--overwrite"] if args.overwrite else [] ), *( ["--save-predictions"] if args.save_predictions else [] )])

    rsf_common = [
        *base,
        "--n-trials", str(args.n_trials_rsf),
        "--n-estimators", str(args.n_estimators_rsf),
        "--n-jobs", str(args.n_jobs_rsf),
        *flags,
    ]
    run([sys.executable, "run_rsf_rotterdam.py", *rsf_common])

    if args.include_zhong_placeholder:
        run([sys.executable, "run_zhong_placeholder_rotterdam.py", *base, *( ["--overwrite"] if args.overwrite else [] )])
    
    if args.include_zhong2022:
        zhong_common = [
            *base,
            "--n-trials", str(args.n_trials_zhong),
            "--tuning-max-epochs", str(args.tuning_max_epochs),
            "--max-epochs", str(args.max_epochs_final),
            *flags,
        ]
        run([sys.executable, "run_zhong2022_rotterdam.py", *zhong_common])

    run([sys.executable, "aggregate_rotterdam_results.py", "--results-dir", args.results_dir])


if __name__ == "__main__":
    main()
