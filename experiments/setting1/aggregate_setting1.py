"""Aggregate the setting-1 replication CSVs into one FLEXI-Haz vs Zhong table.

Reads outputs/flexihaz_<tag>*.csv (run_flexihaz_reps.py) and outputs/zhong_<tag>*.csv
(run_zhong_reps.py), possibly split over several files per setting (--rep-start
batches), and reports per (gamma, rho, n): number of replications, mean theta_hat,
bias in %, empirical SD of theta_hat, mean model SE and 90% / 95% normal-CI
coverage of theta_0 = log 1.5.

FLEXI-Haz rows use theta_hat (selected restart after refinement). Zhong rows use
theta_hat_refined; the unrefined (jointly trained) coefficient is reported as a
separate "zhong_unrefined" row.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import THETA_TRUE

HERE = Path(__file__).resolve().parent
Z90, Z95 = norm.ppf(0.95), norm.ppf(0.975)
PAT = re.compile(r"(flexihaz|zhong)_gamma(?P<gamma>[0-9.]+)_rho(?P<rho>[0-9.]+)_cens(?P<cens>\d+)(?:_k(?P<k>\d))?(?:_n(?P<n>\d+))?")


def summarize(df: pd.DataFrame, est_col: str, se_col: str) -> dict:
    th, se = df[est_col].to_numpy(float), df[se_col].to_numpy(float)
    cov90 = np.mean(np.abs(th - THETA_TRUE) <= Z90 * se) * 100
    cov95 = np.mean(np.abs(th - THETA_TRUE) <= Z95 * se) * 100
    return dict(reps=len(df), theta=th.mean(), bias_pct=100 * (th.mean() - THETA_TRUE) / THETA_TRUE,
                emp_sd=th.std(ddof=1) if len(df) > 1 else np.nan, model_se=se.mean(), cov_90=cov90, cov_95=cov95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default=str(HERE / "outputs"))
    ap.add_argument("--out", default=None, help="summary CSV (default <outputs-dir>/setting1_summary.csv)")
    args = ap.parse_args()
    out_dir = Path(args.outputs_dir)
    groups: dict[tuple, list[Path]] = {}
    for f in sorted(out_dir.glob("*.csv")):
        m = PAT.match(f.name)
        if not m:
            continue
        key = (m.group(1), float(m["gamma"]), float(m["rho"]), int(m["n"] or 0))
        groups.setdefault(key, []).append(f)
    rows = []
    for (model, gamma, rho, n), files in sorted(groups.items()):
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("rep_id")
        if model == "flexihaz":
            rows.append(dict(model="flexihaz", gamma=gamma, rho=rho, n=n, **summarize(df, "theta_hat", "se")))
        else:
            rows.append(dict(model="zhong", gamma=gamma, rho=rho, n=n, **summarize(df, "theta_hat_refined", "se")))
            rows.append(dict(model="zhong_unrefined", gamma=gamma, rho=rho, n=n, **summarize(df, "theta_hat_unrefined", "se")))
    if not rows:
        raise SystemExit(f"no replication CSVs found in {out_dir}")
    summ = pd.DataFrame(rows)
    summ["theta_true"] = THETA_TRUE
    out = Path(args.out) if args.out else out_dir / "setting1_summary.csv"
    summ.to_csv(out, index=False)
    print(summ.round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
