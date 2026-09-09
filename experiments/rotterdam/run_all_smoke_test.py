"""Run a small smoke test for the Rotterdam experiment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "rotterdam_breast_cancer.csv"
RESULTS = ROOT / "results"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    common = ["--data-path", str(DATA), "--results-dir", str(RESULTS), "--smoke-test", "--overwrite", "--overwrite-tuning"]
    run([sys.executable, "run_flexihaz_pl_rotterdam.py", *common])
    run([sys.executable, "run_flexihaz_nonlinear_rotterdam.py", *common])
    run([sys.executable, "run_cox_ph_rotterdam.py", "--data-path", str(DATA), "--results-dir", str(RESULTS), "--smoke-test", "--overwrite"])
    run([sys.executable, "run_rsf_rotterdam.py", *common])
    run([sys.executable, "run_zhong_placeholder_rotterdam.py", "--data-path", str(DATA), "--results-dir", str(RESULTS), "--smoke-test", "--overwrite"])
    run([sys.executable, "aggregate_rotterdam_results.py", "--results-dir", str(RESULTS)])


if __name__ == "__main__":
    main()
