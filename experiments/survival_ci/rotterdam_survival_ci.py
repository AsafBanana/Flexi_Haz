"""Survival-curve confidence intervals for FLEXI-Haz on the Rotterdam breast-cancer data.

The curve CI needs X with finite support, so the clinical covariates are binned:

    age     <=45 | 46-54 | 55-65 | >65
    nodes   0 | 1-3 | >3
    er      <=3 | 4-9 | 10-49 | >49
    pgr     <=9 | 10-99 | >99
    size    <=20 | 20-50 | >50 (as recorded), meno and grade as recorded

(590 distinct profiles among the 2,982 patients). Z = (hormon, chemo). Endpoint as in
experiments/rotterdam: time to the first of recurrence and death, event = 1 only when that
first time is an event. Time is measured in years.

The script fits the cross-fitted one-step estimator (flexihaz_ci.fit_survival_ci) and writes,
for the most frequent covariate profiles (or profiles given with --profile), the survival
curves with pointwise bands for no treatment, hormonal therapy and chemotherapy, plus a figure
of the first profile.

    python rotterdam_survival_ci.py                     # full run (GPU recommended)
    python rotterdam_survival_ci.py --smoke-test        # 1-minute check
    python rotterdam_survival_ci.py --profile 2,1,1,3,1,2,3   # age,meno,size,grade,nodes,pgr,er bins
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flexihaz_ci import fit_survival_ci

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "rotterdam" / "rotterdam_breast_cancer.csv"
X_COLS = ["age_bin", "meno", "size_bin", "grade", "nodes_bin", "pgr_bin", "er_bin"]
Z_COLS = ["hormon", "chemo"]
TREATMENTS = {"none": (0, 0), "hormon": (1, 0), "chemo": (0, 1)}


def cut(series, bins):
    return (pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=False, include_lowest=True, right=True) + 1).astype(int)


def load_binned_rotterdam(path=DEFAULT_DATA):
    df = pd.read_csv(path)
    rtime, dtime = df["rtime"].astype(float), df["dtime"].astype(float)
    T = np.minimum(rtime, dtime) / 365.25
    delta = (((df["recur"].astype(int) == 1) & (rtime <= dtime)) | ((df["death"].astype(int) == 1) & (dtime <= rtime))).astype(int)
    out = pd.DataFrame({
        "age_bin": cut(df["age"], [-np.inf, 45, 54, 65, np.inf]),
        "meno": df["meno"].astype(int),
        "size_bin": df["size"].map({"<=20": 1, "20-50": 2, ">50": 3}).astype(int),
        "grade": df["grade"].astype(int),
        "nodes_bin": cut(df["nodes"], [-np.inf, 0, 3, np.inf]),
        "pgr_bin": cut(df["pgr"], [-np.inf, 9, 99, np.inf]),
        "er_bin": cut(df["er"], [-np.inf, 3, 9, 49, np.inf]),
        "hormon": df["hormon"].astype(int),
        "chemo": df["chemo"].astype(int),
        "T": T,
        "delta": delta,
    })
    return out


def parse_args():
    p = argparse.ArgumentParser(description="FLEXI-Haz survival-curve CIs on the binned Rotterdam data.")
    p.add_argument("--data-path", default=str(DEFAULT_DATA))
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-nodes", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=10000, help="expanded rows per minibatch")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-profiles", type=int, default=5, help="evaluate the most frequent X profiles")
    p.add_argument("--profile", action="append", default=None,
                   help="comma-separated bins age,meno,size,grade,nodes,pgr,er (repeatable); overrides --n-profiles")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_binned_rotterdam(args.data_path)
    X, Z, T, delta = df[X_COLS].values.astype(float), df[Z_COLS].values.astype(float), df["T"].values, df["delta"].values
    cfg = dict(hidden_layers_nodes=args.hidden_nodes, n_hidden_layers=args.hidden_layers, learning_rate=args.learning_rate,
               batch_size=args.batch_size, patience=args.patience, epochs=args.epochs)
    if args.smoke_test:
        cfg.update(epochs=3, patience=2)
        args.results_dir = str(Path(args.results_dir) / "smoke_test")
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rotterdam: n={len(df)}, events={int(delta.sum())} ({1 - delta.mean():.1%} censoring), "
          f"{df[X_COLS].drop_duplicates().shape[0]} covariate profiles, follow-up {T.max():.1f} years")

    ci = fit_survival_ci(X, Z, T, delta, nn_config=cfg, K=args.folds, random_state=args.seed)

    if args.profile:
        profiles = [np.array([float(v) for v in s.split(",")]) for s in args.profile]
    else:
        counts = df.groupby(X_COLS).size().sort_values(ascending=False)
        profiles = [np.array(k, dtype=float) for k in counts.index[: args.n_profiles]]
    n_by_profile = df.groupby(X_COLS).size()

    curves = []
    for i, x0 in enumerate(profiles):
        for name, z0 in TREATMENTS.items():
            s = ci.survival(x0, np.array(z0, dtype=float), alpha=args.alpha)
            s.insert(0, "treatment", name)
            s.insert(0, "profile_id", i)
            for c, v in zip(X_COLS, x0):
                s[c] = int(v)
            s["n_profile"] = int(n_by_profile.get(tuple(int(v) for v in x0), 0))
            curves.append(s)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out_dir / "rotterdam_survival_curves.csv", index=False)

    def summarize(g):
        g = g.dropna(subset=["S_onestep"])          # the band stops at the profile's last at-risk time
        at = lambda col, t: float(np.interp(t, g["t"], g[col])) if g["t"].min() <= t <= g["t"].max() else np.nan
        return pd.Series({"t_band_max": g["t"].max(), "S_onestep_tmax": g["S_onestep"].iloc[-1],
                          "S_lo_tmax": g["S_lo"].iloc[-1], "S_hi_tmax": g["S_hi"].iloc[-1],
                          "S_onestep_5y": at("S_onestep", 5.0), "S_lo_5y": at("S_lo", 5.0), "S_hi_5y": at("S_hi", 5.0),
                          "S_onestep_10y": at("S_onestep", 10.0), "S_lo_10y": at("S_lo", 10.0), "S_hi_10y": at("S_hi", 10.0)})

    keys = ["profile_id", "treatment"] + X_COLS + ["n_profile"]
    summary = curves.groupby(keys, sort=False)[["t", "S_onestep", "S_lo", "S_hi"]].apply(summarize).reset_index()
    summary.to_csv(out_dir / "rotterdam_survival_summary.csv", index=False)
    json.dump({"n": int(len(df)), "n_events": int(delta.sum()), "folds": args.folds, "seed": args.seed, "nn_config": cfg,
               "t_l_bound_years": ci.t_l_bound, "t_eval_years": ci.t_eval, "grid_points": int(len(ci.t_grid)),
               "theta_by_fold": ci.theta_by_fold.tolist(), "theta_cox_init": ci.beta_init.tolist(),
               "timings_sec": {k: round(v, 1) for k, v in ci.timings.items()}},
              open(out_dir / "rotterdam_run.json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x0 = profiles[0]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, color in zip(TREATMENTS, ["black", "tab:blue", "tab:red"]):
            g = curves[(curves.profile_id == 0) & (curves.treatment == name)]
            ax.step(g["t"], g["S_onestep"], where="post", color=color, label=name)
            ax.fill_between(g["t"], g["S_lo"], g["S_hi"], step="post", color=color, alpha=0.15)
        ax.set_xlabel("years since surgery"); ax.set_ylabel("S(t | x0, z)"); ax.set_ylim(0, 1)
        ax.set_title("FLEXI-Haz survival with pointwise {:.0%} CI, profile ".format(1 - args.alpha)
                     + ",".join(f"{c}={int(v)}" for c, v in zip(X_COLS, x0)), fontsize=9)
        ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "rotterdam_survival_profile0.png", dpi=150)
    except Exception as e:  # matplotlib optional
        print("figure skipped:", repr(e))

    print(summary.round(3).to_string(index=False))
    print(f"\nwrote {out_dir / 'rotterdam_survival_curves.csv'}, rotterdam_survival_summary.csv, rotterdam_run.json")


if __name__ == "__main__":
    main()
