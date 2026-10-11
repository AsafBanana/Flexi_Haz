"""Hyperparameter selection for the Zhong et al. (2022) PH-DNN comparator in setting 2,
reproducing the protocol of the original comparator runs (Supplementary A.3, Setting 2):

* an independent tuning dataset per sample size (seed --tuning-seed);
* 100 Optuna TPE trials over the 27 architectures of Table 7 (categorical) and a learning
  rate sampled log-uniformly from [1e-3, 5e-2];
* each trial: five-fold subject-level CV, folds reshuffled between trials
  (KFold(shuffle=True, random_state=seed + trial)); the held-out fold both drives early
  stopping and gives the fold's validation Cox negative partial log-likelihood; the trial
  value is the mean over folds and all five fits must succeed;
* fits as in the replications: model_type="zhong" (Breslow partial likelihood, full batch),
  one constant-rate Adam phase, no theta refinement, raw covariates, Cox warm start on the
  training subjects, at most 2000 epochs, patience 30.

Writes tuning/zhong_best_params.json (format read by run_replications.py) and
tuning/zhong_trials.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from simulation_dgm import simulate_dataset, save_npz, load_npz
from simulation_io import DEFAULT_OUTPUT_DIR, ensure_dir, load_json, save_json
from tune_grid_cv import ARCHITECTURES

LR_RANGE = (1e-3, 5e-2)


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
    p = argparse.ArgumentParser(description="Optuna tuning of the Zhong (2022) comparator (setting 2 protocol).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--tuning-seed", type=int, default=500000, help="seed of the independent tuning dataset")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=2000)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=100000, help="nominal batch; the partial likelihood is full batch")
    p.add_argument("--seed", type=int, default=20240222, help="Optuna sampler seed; fold reshuffling uses seed + trial")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def tuning_dataset(output_dir: Path, seed: int):
    cfg = load_json(output_dir / "config.json")
    path = output_dir / "datasets" / f"tuning_seed{seed}.npz"
    if path.exists():
        data = load_npz(path)
    else:
        data = simulate_dataset(n=int(cfg["n"]), seed=seed, tau=float(cfg["tau"]), censoring_mean=float(cfg["censoring_mean"]),
                                theta_true=np.asarray(cfg["theta_true"], dtype=np.float64))
        save_npz(path, data)
        print(f"generated tuning dataset n={cfg['n']} seed={seed} -> {path}", flush=True)
    return data, float(cfg["tau"])


def fit_fold(FlexiHazardModel, args, hidden, lr, T, delta, X, Z, tr, va, tau, seed):
    """Train on fold tr with the held-out fold va as the early-stopping set; return its best NLL."""
    order = np.concatenate([tr, va])
    n_tr = tr.size
    model = FlexiHazardModel(
        model_type="zhong", hidden_layers=tuple(hidden), activation="relu", dropout=0.0, weight_decay=0.0,
        learning_rate=lr, three_phase_lr=False, refine_theta=False, compute_theta_variance=False,
        batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience, tau=tau,
        standardize=False, theta_init="cox", theta_init_subjects="train", device=args.device, seed=seed, verbose=0,
    )
    # the held-out fold is the validation set (subject rows keep their order, one row per subject)
    model._train_val_indices = lambda subject_id: (np.arange(n_tr), np.arange(n_tr, order.size))
    model.fit(T[order], delta[order], X[order], Z=Z[order])
    return float(np.min(model.train_history["val_loss"]))


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.n_trials, args.folds, args.max_epochs, args.patience = 2, 2, 3, 2
    import optuna

    FlexiHazardModel = import_flexihaz_model()
    output_dir = Path(args.output_dir)
    tuning_dir = ensure_dir(output_dir / "tuning")
    data, tau = tuning_dataset(output_dir, args.tuning_seed)
    T, delta, X, Z = data["T"], data["delta"], data["X"], data["Z"]
    n = T.shape[0]
    rows = []

    def objective(trial):
        idx = trial.suggest_categorical("hidden_layers_id", list(range(len(ARCHITECTURES))))
        hidden = ARCHITECTURES[idx]
        lr = trial.suggest_float("learning_rate", LR_RANGE[0], LR_RANGE[1], log=True)
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed + trial.number)
        losses, t0 = [], time.perf_counter()
        for fold_id, (tr, va) in enumerate(kf.split(np.arange(n))):
            losses.append(fit_fold(FlexiHazardModel, args, hidden, lr, T, delta, X, Z, tr, va, tau,
                                   seed=args.seed + 1000 * trial.number + fold_id))
            if not np.isfinite(losses[-1]):
                raise RuntimeError(f"fold {fold_id} gave a non-finite validation loss")
        value = float(np.mean(losses))
        rows.append(dict(trial=trial.number, hidden_layers=str(hidden), learning_rate=lr, mean_val_nll=value,
                         fold_val_nll=str([round(v, 6) for v in losses]), time_sec=round(time.perf_counter() - t0, 1)))
        print(f"trial {trial.number}: {hidden} lr={lr:.5g} mean_val_nll={value:.6f} ({rows[-1]['time_sec']}s)", flush=True)
        return value

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))
    pd.DataFrame(rows).to_csv(tuning_dir / "zhong_trials.csv", index=False)
    best = study.best_trial
    hidden = ARCHITECTURES[int(best.params["hidden_layers_id"])]
    payload = {
        "model_type": "zhong", "tuning_seed": args.tuning_seed, "folds": args.folds, "n_trials": args.n_trials,
        "sampler": "optuna TPE", "learning_rate_range": list(LR_RANGE), "selection": "min mean held-out Cox NPLL over 5 reshuffled folds",
        "best_trial_number": best.number, "best_value": float(best.value),
        "best_params": {"hidden_layers": [int(h) for h in hidden], "learning_rate": float(best.params["learning_rate"]),
                        "batch_size": float(args.batch_size), "activation": "relu", "grid_size": "n", "dropout": 0.0, "weight_decay": 0.0},
        "max_epochs_used_for_tuning": args.max_epochs, "patience_used_for_tuning": args.patience,
        "standardize": False, "theta_init_subjects": "train", "three_phase_lr": False, "refine_theta": False,
        "search_space": {"hidden_layers": ARCHITECTURES, "learning_rate": list(LR_RANGE)},
    }
    save_json(tuning_dir / "zhong_best_params.json", payload)
    print(f"\nselected: hidden_layers={hidden} learning_rate={best.params['learning_rate']:.6g} mean_val_nll={best.value:.6f}")
    print(f"wrote {tuning_dir / 'zhong_best_params.json'} and {tuning_dir / 'zhong_trials.csv'}")


if __name__ == "__main__":
    main()
