#!/usr/bin/env python3
"""
02_make_splits.py
=================
Idea B: deterioration-pattern-specific sepsis early warning (PhysioNet/CinC 2019).

Freezes ONE patient-level data split that every later script must reuse.

Design
------
* Unit of splitting = patient (never rows or hours), so no patient appears in more
  than one partition.
* Patients with include_main == False in patients.csv (e.g. label already on at
  hour 0) get split = "excluded" and fold = -1. They are kept in the output file
  so nothing disappears silently.
* The remaining patients are split into
      test : a final hold-out (default 20%), touched only for the final report
      dev  : everything else, divided into K stratified folds (default 5)
  Use the dev folds for model development, tuning, thresholds and loss weights.
  Use the test set once, at the end. (Tuning on cross-validation folds and then
  reporting the same folds would give optimistic numbers.)
* Stratification: hospital x septic status (4 strata), so both hospitals and both
  classes are represented in every fold and in the test set.
* Hospital-transfer experiments (train on A, test on B and the reverse) do not use
  these folds; they use the `hospital` column directly (script 12).

Usage
-----
    python 02_make_splits.py --patients outputs/clean/patients.csv --outdir outputs/splits
    python 02_make_splits.py --patients outputs/clean/patients.csv --outdir outputs/splits \
        --n-folds 5 --test-frac 0.2 --seed 42

Outputs (in --outdir)
---------------------
    folds.csv          Patient_ID, hospital, septic, include_main, split, fold
    split_balance.csv  per-partition summary (size, septic %, hospital mix, age, ...)
    splits_meta.json   seed, parameters, library version, counts, file hashes

How later scripts should use it
-------------------------------
    folds = pd.read_csv("outputs/splits/folds.csv")
    hourly = hourly.merge(folds[["Patient_ID", "split", "fold"]], on="Patient_ID")
    train = hourly[(hourly.split == "dev") & (hourly.fold != k)]
    valid = hourly[(hourly.split == "dev") & (hourly.fold == k)]
    test  = hourly[hourly.split == "test"]          # final report only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

REQUIRED = ["Patient_ID", "hospital", "septic", "include_main", "n_hours", "age",
            "first_pos_hour", "unit", "onset_hour0", "age_lt18", "short_stay"]


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def make_splits(pt: pd.DataFrame, n_folds: int, test_frac: float, seed: int) -> pd.DataFrame:
    """Return one row per patient with columns: split, fold. Deterministic."""
    pt = pt.sort_values("Patient_ID").reset_index(drop=True)  # input order must not matter
    pt["stratum"] = pt["hospital"].astype(str) + "_" + np.where(pt["septic"], "sepsis", "none")

    pt["split"] = "excluded"
    pt["fold"] = -1

    elig = pt[pt["include_main"]]
    dev_idx, test_idx = train_test_split(
        elig.index.values, test_size=test_frac, stratify=elig["stratum"], random_state=seed
    )
    pt.loc[test_idx, "split"] = "test"
    pt.loc[dev_idx, "split"] = "dev"

    dev = pt.loc[np.sort(dev_idx)]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(dev, dev["stratum"])):
        pt.loc[dev.index[va], "fold"] = k
    return pt


# --------------------------------------------------------------------------- #
# Checks and reporting
# --------------------------------------------------------------------------- #

def validate(pt: pd.DataFrame, n_folds: int) -> None:
    print("[validate] running integrity checks ...")
    assert pt["Patient_ID"].is_unique, "duplicate Patient_ID"
    assert set(pt["split"]) <= {"dev", "test", "excluded"}
    assert ((pt["split"] == "excluded") == (~pt["include_main"])).all(), "exclusion mismatch"
    assert (pt.loc[pt["split"] != "dev", "fold"] == -1).all(), "non-dev patient has a fold"
    dev = pt[pt["split"] == "dev"]
    assert set(dev["fold"]) == set(range(n_folds)), "dev folds are not 0..K-1"
    assert dev["fold"].notna().all()

    # Each stratum must be spread across all folds, and the test set must contain both
    # hospitals (when present) and some septic patients from each.
    for s, sub in dev.groupby("stratum"):
        counts = sub["fold"].value_counts()
        assert len(counts) == n_folds, f"stratum {s} missing from some fold"
        assert counts.max() - counts.min() <= 1, f"stratum {s} unevenly split across folds"
    test = pt[pt["split"] == "test"]
    for h in pt["hospital"].unique():
        t = test[test["hospital"] == h]
        assert len(t) > 0, f"no test patients from hospital {h}"
        assert t["septic"].sum() > 0, f"no septic test patients from hospital {h}"
    print("[validate] all checks passed")


def smd(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference (|SMD| < 0.1 is the usual 'balanced' rule of thumb)."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def summarise(sub: pd.DataFrame) -> dict:
    sep = sub[sub["septic"]]
    return {
        "patients": len(sub),
        "septic": int(sub["septic"].sum()),
        "septic_pct": round(100 * sub["septic"].mean(), 2),
        "hospital_A_pct": round(100 * (sub["hospital"] == "A").mean(), 2),
        "mean_age": round(sub["age"].mean(), 2),
        "mean_stay_h": round(sub["n_hours"].mean(), 1),
        "median_first_pos_hour": float(sep["first_pos_hour"].median()) if len(sep) else np.nan,
        "unit_unknown_pct": round(100 * (sub["unit"] == 0).mean(), 2),
    }


def balance_table(pt: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    rows = {}
    for name, sub in [("dev_all", pt[pt["split"] == "dev"]),
                      ("test", pt[pt["split"] == "test"]),
                      ("excluded", pt[pt["split"] == "excluded"])]:
        if len(sub):
            rows[name] = summarise(sub)
    dev = pt[pt["split"] == "dev"]
    for k in range(n_folds):
        rows[f"dev_fold{k}"] = summarise(dev[dev["fold"] == k])
    return pd.DataFrame(rows).T


def balance_warnings(pt: pd.DataFrame, n_folds: int) -> list[str]:
    """Flag any fold (or the test set) that differs noticeably from the rest of dev."""
    warns = []
    dev = pt[pt["split"] == "dev"]
    groups = [(f"fold {k}", dev[dev["fold"] == k], dev[dev["fold"] != k]) for k in range(n_folds)]
    groups.append(("test", pt[pt["split"] == "test"], dev))
    for label, a, b in groups:
        for col in ("age", "n_hours"):
            d = smd(a[col], b[col])
            if abs(d) >= 0.1:
                warns.append(f"{label}: SMD for {col} = {d:+.3f} (>= 0.1)")
        d = smd(a.loc[a["septic"], "first_pos_hour"], b.loc[b["septic"], "first_pos_hour"])
        if abs(d) >= 0.2:
            warns.append(f"{label}: SMD for first_pos_hour among septic = {d:+.3f} (>= 0.2)")
    return warns


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", required=True, type=Path, help="patients.csv from 01_clean_data.py")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0 < args.test_frac < 1:
        sys.exit("ERROR: --test-frac must be between 0 and 1")
    if args.n_folds < 2:
        sys.exit("ERROR: --n-folds must be at least 2")

    pt = pd.read_csv(args.patients)
    missing = [c for c in REQUIRED if c not in pt.columns]
    if missing:
        sys.exit(f"ERROR: {args.patients} lacks columns {missing}. Re-run 01_clean_data.py.")
    for c in ("septic", "include_main", "onset_hour0", "age_lt18", "short_stay"):
        pt[c] = pt[c].astype(bool)

    n_ex = int((~pt["include_main"]).sum())
    print(f"[load] {len(pt)} patients; {n_ex} excluded "
          f"(hour-0 onset: {int(pt.loc[~pt['include_main'], 'onset_hour0'].sum())}, "
          f"under-18: {int(pt.loc[~pt['include_main'], 'age_lt18'].sum())}, "
          f"short stay: {int(pt.loc[~pt['include_main'], 'short_stay'].sum())}; flags can overlap)")

    out = make_splits(pt, args.n_folds, args.test_frac, args.seed)

    # Determinism check: the same call on shuffled input must give identical assignments.
    again = make_splits(pt.sample(frac=1.0, random_state=1), args.n_folds, args.test_frac, args.seed)
    assert out[["Patient_ID", "split", "fold"]].equals(again[["Patient_ID", "split", "fold"]]), \
        "split is not deterministic"
    validate(out, args.n_folds)

    args.outdir.mkdir(parents=True, exist_ok=True)
    folds_path = args.outdir / "folds.csv"
    out[["Patient_ID", "hospital", "septic", "include_main", "split", "fold"]].to_csv(folds_path, index=False)

    bal = balance_table(out, args.n_folds)
    bal.to_csv(args.outdir / "split_balance.csv")
    warns = balance_warnings(out, args.n_folds)

    meta = {
        "seed": args.seed, "n_folds": args.n_folds, "test_frac": args.test_frac,
        "stratification": "hospital x septic",
        "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__,
        "patients_csv": str(args.patients), "patients_csv_sha256": sha256(args.patients),
        "folds_csv_sha256": sha256(folds_path),
        "counts": out["split"].value_counts().to_dict(),
        "septic_by_split": out[out["septic"]]["split"].value_counts().to_dict(),
        "balance_warnings": warns,
    }
    with open(args.outdir / "splits_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[write] {folds_path}\n[write] {args.outdir / 'split_balance.csv'}\n"
          f"[write] {args.outdir / 'splits_meta.json'}")
    print("\n=== SPLIT BALANCE ===")
    print(bal.to_string())
    print("\n=== SEPTIC PATIENTS BY HOSPITAL AND SPLIT ===")
    print(pd.crosstab(out.loc[out["septic"], "hospital"], out.loc[out["septic"], "split"]).to_string())
    print("\nBalance warnings:", warns if warns else "none")


if __name__ == "__main__":
    main()