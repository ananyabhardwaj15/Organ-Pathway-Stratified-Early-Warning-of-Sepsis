#!/usr/bin/env python3
"""
03_pattern_features.py
======================
Idea B: deterioration-pattern-specific sepsis early warning (PhysioNet/CinC 2019).

Builds ONE row per septic patient describing how the vital signs changed in the
hours just BEFORE the label switched on. File 04 clusters these rows into
"deterioration patterns".

Window definition
-----------------
Let F = first_pos_hour (first hour where SepsisLabel == 1; see 01_clean_data.py).
The pattern window is the last W hours strictly before F:  Hour in [F - W, F - 1].
Only pre-onset rows (is_pre_onset == True) are used, so no post-label data enters.
If F < W the window is shorter (window_hours = F).

Features per vital v (default HR, Resp, MAP, O2Sat, Temp), all computed from the
observed values inside the window:
    v_n_obs   number of observations
    v_first   mean of the earliest k observations   (k = max(1, min(3, n // 2)))
    v_last    mean of the latest k observations
    v_delta   v_last - v_first
    v_slope   least-squares slope per hour (needs >= 3 observations)
    v_mean, v_std, v_min, v_max
A vital with fewer than --min-obs observations gets NaN for every statistic except
v_n_obs (so a gap in measurement is never silently turned into a number).

Descriptor columns (NOT meant as clustering inputs; used afterwards to describe and
sanity-check clusters, and for the measurement-frequency control):
    desc_<lab>_last / desc_<lab>_n_obs for Creatinine, Platelets, Bilirubin_total,
    Lactate, WBC; plus n_vital_obs_window, n_lab_obs_window, n_lab_hours_window.

Eligibility
-----------
pattern_eligible = window_hours >= --min-window-hours
                   AND at least --min-vitals-ok vitals have >= --min-obs observations.
Ineligible septic patients stay in the file with ineligible_reason set, so the
"unassigned" group can be reported honestly.

Leakage note
------------
These features are defined from data up to the label switch-on. They are for GROUPING
and EVALUATING patients only, never as inputs to the early-warning models. File 04 must
fit the clustering on split == "dev" patients only and assign "test" patients to the
nearest dev cluster.

Usage
-----
    python 03_pattern_features.py \
        --hourly outputs/clean/hourly_clean.parquet \
        --folds  outputs/splits/folds.csv \
        --outdir outputs/patterns
    python 03_pattern_features.py ... --window-hours 12 --min-obs 2

Outputs (in --outdir)
---------------------
    pattern_features.parquet   (or .csv.gz without pyarrow)
    pattern_features_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "Patient_ID"
DEFAULT_VITALS = ["HR", "Resp", "MAP", "O2Sat", "Temp"]
DESC_LABS = ["Creatinine", "Platelets", "Bilirubin_total", "Lactate", "WBC"]
ALL_LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
]
STAT_NAMES = ["n_obs", "first", "last", "delta", "slope", "mean", "std", "min", "max"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_hourly(path: Path, vitals: list[str]) -> pd.DataFrame:
    cols = [ID, "Hour", "first_pos_hour", "septic_patient", "is_pre_onset"] + vitals + ALL_LABS
    cols = list(dict.fromkeys(cols))  # keep order, drop duplicates
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=cols)
    return pd.read_csv(path, usecols=cols)


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #

def series_stats(t: np.ndarray, y: np.ndarray, min_obs: int, k_max: int = 3) -> dict:
    """Summary statistics of one vital inside one window (t = hours relative to F, ascending)."""
    n = len(y)
    out = {s: np.nan for s in STAT_NAMES}
    out["n_obs"] = n
    if n < max(min_obs, 1):
        return out
    out["mean"], out["min"], out["max"] = y.mean(), y.min(), y.max()
    out["last"] = y[-1] if n == 1 else None
    if n >= 2:
        k = max(1, min(k_max, n // 2))
        out["first"] = y[:k].mean()
        out["last"] = y[-k:].mean()
        out["delta"] = out["last"] - out["first"]
        out["std"] = y.std(ddof=1)
    if n >= 3:
        out["slope"] = float(np.polyfit(t, y, 1)[0])
    return out


def build_features(h: pd.DataFrame, cohort: pd.DataFrame, vitals: list[str],
                   window: int, min_obs: int) -> pd.DataFrame:
    # first_pos_hour per patient, taken before any window filtering
    fp = h.loc[h[ID].isin(cohort[ID])].groupby(ID)["first_pos_hour"].first()

    h = h[h[ID].isin(cohort[ID]) & h["is_pre_onset"]].copy()
    h["rel"] = h["Hour"] - h[ID].map(fp)            # -1 = last pre-onset hour
    h = h[h["rel"] >= -window].sort_values([ID, "Hour"])

    rows = []
    for pid, g in h.groupby(ID, sort=False):
        rec = {ID: pid}
        t = g["rel"].to_numpy(dtype=float)
        for v in vitals:
            m = g[v].notna().to_numpy()
            st = series_stats(t[m], g[v].to_numpy(dtype=float)[m], min_obs)
            for s in STAT_NAMES:
                rec[f"{v}_{s}"] = st[s]
        rows.append(rec)
    feat = pd.DataFrame(rows)

    # Make sure every cohort patient has a row (even with no window data)
    feat = cohort[[ID]].merge(feat, on=ID, how="left")
    for v in vitals:
        feat[f"{v}_n_obs"] = feat[f"{v}_n_obs"].fillna(0).astype("int32")

    feat["first_pos_hour"] = feat[ID].map(fp).astype("float32")
    feat["window_hours"] = np.minimum(window, feat["first_pos_hour"]).astype("int32")

    # Descriptor labs and activity counts
    grp = h.groupby(ID)
    desc = grp[DESC_LABS].agg(["last", "count"])
    desc.columns = [f"desc_{lab}_{'last' if stat == 'last' else 'n_obs'}" for lab, stat in desc.columns]
    feat = feat.merge(desc.reset_index(), on=ID, how="left")
    feat["n_vital_obs_window"] = feat[ID].map(h[vitals].notna().sum(axis=1).groupby(h[ID]).sum()).fillna(0).astype("int32")
    labs_present = h[ALL_LABS].notna()
    feat["n_lab_obs_window"] = feat[ID].map(labs_present.sum(axis=1).groupby(h[ID]).sum()).fillna(0).astype("int32")
    feat["n_lab_hours_window"] = feat[ID].map(labs_present.any(axis=1).groupby(h[ID]).sum()).fillna(0).astype("int32")
    for lab in DESC_LABS:
        feat[f"desc_{lab}_n_obs"] = feat[f"desc_{lab}_n_obs"].fillna(0).astype("int32")

    # Attach split info
    feat = feat.merge(cohort[[ID, "hospital", "split", "fold"]], on=ID, how="left")
    return feat


def add_eligibility(feat: pd.DataFrame, vitals: list[str], min_obs: int,
                    min_window_hours: int, min_vitals_ok: int) -> pd.DataFrame:
    ok = pd.concat([(feat[f"{v}_n_obs"] >= min_obs).rename(v) for v in vitals], axis=1)
    feat["n_vitals_ok"] = ok.sum(axis=1).astype("int8")
    short = feat["window_hours"] < min_window_hours
    sparse = feat["n_vitals_ok"] < min_vitals_ok
    feat["pattern_eligible"] = ~(short | sparse)
    feat["ineligible_reason"] = np.select([short, sparse], ["short_window", "sparse_vitals"], default="")
    return feat


# --------------------------------------------------------------------------- #
# Validation and report
# --------------------------------------------------------------------------- #

def validate(feat: pd.DataFrame, cohort: pd.DataFrame, vitals: list[str],
             window: int, min_obs: int) -> None:
    print("[validate] running integrity checks ...")
    assert feat[ID].is_unique and len(feat) == len(cohort), "need exactly one row per septic patient"
    assert (feat["first_pos_hour"] >= 1).all(), "a cohort patient has no pre-onset history"
    assert (feat["window_hours"] >= 1).all() and (feat["window_hours"] <= window).all()
    for v in vitals:
        assert (feat[f"{v}_n_obs"] <= feat["window_hours"]).all(), f"{v}: more observations than hours"
        few = feat[f"{v}_n_obs"] < min_obs
        for s in ("mean", "last", "delta", "slope"):
            assert feat.loc[few, f"{v}_{s}"].isna().all(), f"{v}_{s} filled despite < min_obs observations"
        enough = feat[f"{v}_n_obs"] >= min_obs
        assert feat.loc[enough, f"{v}_mean"].notna().all()
        if min_obs >= 3:
            assert feat.loc[enough, f"{v}_slope"].notna().all()
    el = feat[feat["pattern_eligible"]]
    assert (el["window_hours"] >= 1).all()
    print("[validate] all checks passed")


def window_leak_check(h: pd.DataFrame, feat: pd.DataFrame, window: int) -> None:
    """Independent re-check: no row used for a patient is at/after its label switch-on."""
    sep = h[h[ID].isin(feat[ID]) & h["is_pre_onset"]]
    fp = feat.set_index(ID)["first_pos_hour"]
    assert (sep["Hour"] < sep[ID].map(fp)).all(), "post-onset row in window"
    print("[validate] window leak check passed (all rows strictly before label switch-on)")


def make_report(feat: pd.DataFrame, vitals: list[str], args: argparse.Namespace) -> dict:
    def share(mask: pd.Series) -> float:
        return round(float(mask.mean()), 4) if len(mask) else float("nan")

    rep = {
        "parameters": {"window_hours": args.window_hours, "min_obs": args.min_obs,
                       "min_window_hours": args.min_window_hours,
                       "min_vitals_ok": args.min_vitals_ok, "vitals": vitals},
        "septic_patients": int(len(feat)),
        "eligible": int(feat["pattern_eligible"].sum()),
        "eligible_share": share(feat["pattern_eligible"]),
        "ineligible_reasons": feat["ineligible_reason"].replace("", "eligible").value_counts().to_dict(),
        "eligible_by_hospital": feat.groupby("hospital")["pattern_eligible"].agg(["sum", "size", "mean"]).round(4).to_dict("index"),
        "eligible_by_split": feat.groupby("split")["pattern_eligible"].agg(["sum", "size", "mean"]).round(4).to_dict("index"),
        "window_hours_quantiles": feat["window_hours"].quantile([.05, .25, .5, .75]).to_dict(),
        "vital_coverage_share_with_min_obs": {
            v: {h: share(g[f"{v}_n_obs"] >= args.min_obs) for h, g in feat.groupby("hospital")}
            for v in vitals
        },
        "median_n_obs": {v: float(feat[f"{v}_n_obs"].median()) for v in vitals},
    }
    return rep


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hourly", required=True, type=Path, help="hourly_clean.parquet / .csv.gz from step 01")
    ap.add_argument("--folds", required=True, type=Path, help="folds.csv from step 02")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--window-hours", type=int, default=24, help="W: hours before label switch-on (default 24)")
    ap.add_argument("--min-window-hours", type=int, default=6, help="minimum usable window length (default 6)")
    ap.add_argument("--min-obs", type=int, default=3, help="min observations of a vital to compute its features (default 3)")
    ap.add_argument("--min-vitals-ok", type=int, default=5, help="min number of vitals meeting --min-obs for eligibility")
    ap.add_argument("--vitals", nargs="+", default=DEFAULT_VITALS)
    args = ap.parse_args()

    if args.min_obs < 1 or args.window_hours < 2:
        sys.exit("ERROR: need --min-obs >= 1 and --window-hours >= 2")
    if args.min_vitals_ok > len(args.vitals):
        sys.exit("ERROR: --min-vitals-ok exceeds the number of vitals")

    folds = pd.read_csv(args.folds)
    for c in ("septic", "include_main"):
        folds[c] = folds[c].astype(bool)
    cohort = folds[folds["septic"] & folds["include_main"]].copy()
    print(f"[load] cohort: {len(cohort)} septic patients with include_main "
          f"({cohort['split'].value_counts().to_dict()})")

    h = load_hourly(args.hourly, args.vitals)
    print(f"[load] hourly rows: {len(h)}")

    feat = build_features(h, cohort, args.vitals, args.window_hours, args.min_obs)
    feat = add_eligibility(feat, args.vitals, args.min_obs, args.min_window_hours, args.min_vitals_ok)
    validate(feat, cohort, args.vitals, args.window_hours, args.min_obs)
    window_leak_check(h, feat, args.window_hours)

    args.outdir.mkdir(parents=True, exist_ok=True)
    try:
        out = args.outdir / "pattern_features.parquet"
        feat.to_parquet(out, index=False)
    except ImportError:
        out = args.outdir / "pattern_features.csv.gz"
        feat.to_csv(out, index=False, compression="gzip")
        print("[note] pyarrow not installed: wrote gzip CSV instead of parquet.")
    rep = make_report(feat, args.vitals, args)
    with open(args.outdir / "pattern_features_report.json", "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"[write] {out}\n[write] {args.outdir / 'pattern_features_report.json'}")

    print("\n=== ELIGIBILITY ===")
    print(f"septic patients: {rep['septic_patients']} | eligible: {rep['eligible']} ({rep['eligible_share']*100:.1f}%)")
    print("reasons:", rep["ineligible_reasons"])
    print("by hospital:", {k: (int(v['sum']), int(v['size'])) for k, v in rep["eligible_by_hospital"].items()})
    print("by split:   ", {k: (int(v['sum']), int(v['size'])) for k, v in rep["eligible_by_split"].items()})
    print("window_hours quantiles (5/25/50/75%):", rep["window_hours_quantiles"])
    print("\nShare of septic patients with >= min_obs observations, per vital and hospital:")
    print(pd.DataFrame(rep["vital_coverage_share_with_min_obs"]).T.to_string())


if __name__ == "__main__":
    main()