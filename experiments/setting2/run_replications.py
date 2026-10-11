"""Run partially linear FLEXI-Haz or the Zhong et al. (2022) PH-DNN on the saved replications.

Both models are fitted with flexihaz.FlexiHazardModel (--model-type flexihaz | zhong) using
the per-model tuned configuration (tune_grid_cv.py for FLEXI-Haz, tune_zhong_optuna.py for Zhong;
tuning/<prefix>_best_params.json).

Examples
--------
Smoke test:
    python run_replications.py --smoke-test

FLEXI-Haz, 200 replications split over machines (3-phase lr, theta refinement, 5 restarts):
    python run_replications.py --model-type flexihaz --n-restarts 5 --start-rep 0 --end-rep 50

Zhong et al. (2022), the comparator's recipe: one initialization, one constant phase at the tuned
learning rate, no theta refinement, raw covariates, Cox warm start on the training subjects
(model folder zhong_plain):
    python run_replications.py --model-type zhong --zhong-plain

Faster learning rate for theta (model folder gets a _thetalr20 suffix):
    python run_replications.py --theta-lr-mult 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_survival import evaluate_survival_predictions, theta_metrics_from_model
from simulation_io import (
    DEFAULT_OUTPUT_DIR,
    completed,
    ensure_dir,
    load_best_params,
    load_fixed_test,
    load_train_dataset,
    rep_auc_path,
    rep_bias_path,
    rep_brier_path,
    rep_log_path,
    rep_predictions_path,
    rep_summary_path,
    rep_theta_path,
    replication_dir,
    save_json,
    write_summary_row,
)

MODEL_NAME = "flexihaz_partial_linear"


def import_flexihaz_model():
    try:
        from flexihaz import FlexiHazardModel
        return FlexiHazardModel
    except ImportError:
        source_dir = os.environ.get("FLEXIHAZ_SOURCE_DIR")
        if source_dir:
            sys.path.insert(0, source_dir)
            from flexihaz import FlexiHazardModel
            return FlexiHazardModel
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run partially linear FLEXI-Haz simulation replications.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--params-path", default=None)
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--end-rep", type=int, default=200)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--projection-max-epochs", type=int, default=2000)
    parser.add_argument("--projection-patience", type=int, default=30)
    parser.add_argument("--projection-n-splits", type=int, default=5)
    parser.add_argument("--projection-batch-size", type=int, default=128, help="minibatch of the projection nets (paper: 128)")
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-restarts", type=int, default=1,
                        help="number of network restarts per replication (1 = original single fit)")
    parser.add_argument("--restart-selection", choices=["val_loss", "efficient_score"], default="val_loss",
                        help="how to choose among restarts: lowest validation loss, or efficient score closest to 0")
    parser.add_argument("--theta-lr-mult", type=float, default=1.0,
                        help="learning-rate multiplier for the linear (theta) branch (1 = same rate as the network)")
    parser.add_argument("--model-type", choices=["flexihaz", "zhong"], default="flexihaz",
                        help="'zhong' fits the proportional-hazards model of Zhong et al. (2022) with the same estimator")
    parser.add_argument("--zhong-plain", action="store_true",
                        help="Zhong's own recipe: one constant-lr phase, no theta refinement, raw (unstandardized) covariates, "
                             "Cox warm start on the training subjects; model folder suffix _plain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.output_dir = str(Path(args.output_dir) / "smoke_test")
        args.start_rep = 0
        args.end_rep = min(args.end_rep, 3)
        args.max_epochs = min(args.max_epochs, 5)
        args.patience = min(args.patience, 2)
        args.projection_max_epochs = min(args.projection_max_epochs, 5)
        args.projection_patience = min(args.projection_patience, 2)
        args.projection_n_splits = 2

    # keep variant runs apart from the single-fit results
    model_name = "zhong" if args.model_type == "zhong" else MODEL_NAME
    if args.zhong_plain:
        model_name = f"{model_name}_plain"
    if args.n_restarts > 1:
        model_name = f"{model_name}_k{args.n_restarts}_{args.restart_selection}"
    if args.theta_lr_mult != 1.0:
        model_name = f"{model_name}_thetalr{args.theta_lr_mult:g}"

    output_dir = Path(args.output_dir)
    ensure_dir(replication_dir(output_dir, model_name))

    default_params = "zhong_best_params.json" if args.model_type == "zhong" else "flexihaz_partial_linear_best_params.json"
    params_path = Path(args.params_path) if args.params_path else output_dir / "tuning" / default_params
    best_params = load_best_params(params_path)
    best_params["grid_size"] = "n"
    best_params['weight_decay'] = 0.0
    best_params['dropout'] = 0.0

    FlexiHazardModel = import_flexihaz_model()
    test = load_fixed_test(output_dir)
    X_test = test["X"]
    Z_test = test["Z"]
    eval_times_survival = test["eval_times_survival"]
    eval_times_metrics = test["eval_times_metrics"]
    S_true = test["true_survival"]

    for rep_id in range(args.start_rep, args.end_rep):
        if args.skip_completed and not args.overwrite and completed(output_dir, model_name, rep_id):
            print(f"rep={rep_id:03d} already completed, skipping")
            continue

        print(f"Running {model_name}, rep={rep_id:03d}")
        try:
            train = load_train_dataset(output_dir, rep_id)
            T, delta, X, Z = train["T"], train["delta"], train["X"], train["Z"]

            lr = best_params.get("learning_rate", None)

            model = FlexiHazardModel(
                **best_params,
                tau=args.tau,
                include_observed_times=True,
                max_epochs=args.max_epochs,
                patience=args.patience,
                compute_theta_variance=True,
                refine_theta=not args.zhong_plain,
                projection_max_epochs=args.projection_max_epochs,
                projection_patience=args.projection_patience,
                projection_n_splits=args.projection_n_splits,
                projection_batch_size=args.projection_batch_size,
                device=args.device,
                seed=args.seed + rep_id,
                verbose=0,
                three_phase_lr=not args.zhong_plain,
                standardize=not args.zhong_plain,
                theta_init_subjects="train" if args.zhong_plain else "all",
                n_restarts=args.n_restarts,
                restart_selection=args.restart_selection,
                theta_lr_multiplier=args.theta_lr_mult,
                model_type=args.model_type,
            )

            t0 = time.perf_counter()
            model.fit(T, delta, X, Z=Z)
            training_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            S_hat = model.predict_survival(X_test, eval_times_survival, Z_new=Z_test)
            prediction_time = time.perf_counter() - t1

            theta_df, theta_summary_extra = theta_metrics_from_model(
                model, model_name=model_name, rep_id=rep_id, theta_true=train["theta_true"]
            )
            theta_df.to_csv(rep_theta_path(output_dir, model_name, rep_id), index=False)

            extra = {
                "n_train": int(T.shape[0]),
                "event_rate": float(train["event_rate"]),
                "observed_censoring_rate": float(train["observed_censoring_rate"]),
                "random_censoring_rate": float(train["random_censoring_rate"]),
                "administrative_censoring_rate": float(train["administrative_censoring_rate"]),
                "params_path": str(params_path),
                "model_type": args.model_type,
                "theta_lr_multiplier": float(args.theta_lr_mult),
                "projection_batch_size": int(args.projection_batch_size),
                "three_phase_lr": not args.zhong_plain,
                "refine_theta": not args.zhong_plain,
                "standardize": not args.zhong_plain,
                "theta_init_subjects": "train" if args.zhong_plain else "all",
                **theta_summary_extra,
            }
            if model.restart_results_ is not None:
                extra["n_restarts"] = args.n_restarts
                extra["restart_selection"] = args.restart_selection
                extra["selected_restart"] = int(model.selected_restart_)
                for r in model.restart_results_:
                    k = r["restart"]
                    extra[f"restart{k}_val_loss"] = r["val_loss"]
                    for j, th in enumerate(r.get("theta", [])):
                        extra[f"restart{k}_theta_{j}"] = float(th)
                    if "score_norm" in r:
                        extra[f"restart{k}_score_norm"] = r["score_norm"]

            summary, bias_df, auc_df, brier_df = evaluate_survival_predictions(
                model_name=model_name,
                rep_id=rep_id,
                train_T=T,
                train_delta=delta,
                test_T=test["T"],
                test_delta=test["delta"],
                S_hat=S_hat,
                S_true=S_true,
                eval_times_survival=eval_times_survival,
                eval_times_metrics=eval_times_metrics,
                training_time_sec=training_time,
                prediction_time_sec=prediction_time,
                extra_summary=extra,
            )
            write_summary_row(rep_summary_path(output_dir, model_name, rep_id), summary)
            bias_df.to_csv(rep_bias_path(output_dir, model_name, rep_id), index=False)
            auc_df.to_csv(rep_auc_path(output_dir, model_name, rep_id), index=False)
            brier_df.to_csv(rep_brier_path(output_dir, model_name, rep_id), index=False)
            if args.save_predictions:
                np.savez_compressed(rep_predictions_path(output_dir, model_name, rep_id), S_hat=S_hat)

            save_json(rep_log_path(output_dir, model_name, rep_id), {"status": "success", "rep_id": rep_id})
            print(f"rep={rep_id:03d} done, ibs={summary['ibs']}, c_index={summary['c_index']}")

        except Exception as exc:  # noqa: BLE001
            error_row = {
                "model": model_name,
                "rep_id": int(rep_id),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            write_summary_row(rep_summary_path(output_dir, model_name, rep_id), error_row)
            save_json(rep_log_path(output_dir, model_name, rep_id), error_row)
            print(f"rep={rep_id:03d} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
