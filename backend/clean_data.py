#!/usr/bin/env python3
"""
01_clean_data.py
================
Idea B: deterioration-pattern-specific sepsis early warning (PhysioNet/CinC 2019).

What this script does
---------------------
1. Loads the raw hourly CSV (or a zip containing one CSV).
2. Drops redundant columns, sorts by patient and hour, tags the hospital.
3. Fixes invalid values (FiO2 units, impossible vital signs / labs) by setting
   them to NaN. No row is ever deleted by the cleaning rules.
4. Adds flags (inconsistent blood pressure, top-coded age, under-18 age).
5. Adds label-timing columns used by every later step:
       septic_patient, first_pos_hour, hours_to_onset, is_pre_onset
   (rows are NOT removed; later scripts filter on is_pre_onset).
6. Builds a patient-level table with exclusion flags.
7. Validates the result and writes a cleaning report (JSON).

What this script deliberately does NOT do
-----------------------------------------
- No imputation and no missing-value indicators (done in the feature step, so
  that measurement-frequency information is not destroyed).
- No train/test splitting (done in 02_make_splits.py).
- No row deletion. Exclusions are expressed as flags in patients.csv.

Usage
-----
    python 01_clean_data.py --input Dataset_csv.zip --outdir outputs/clean
    python 01_clean_data.py --input Dataset.csv --outdir outputs/clean \
        --min-stay-hours 12 --min-age 18

Outputs (in --outdir)
---------------------
    hourly_clean.parquet   (or hourly_clean.csv.gz if pyarrow is not installed)
    patients.csv
    cleaning_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# In the PhysioNet 2019 training data, hospital A patient IDs are 1..~20,643
# and hospital B IDs are 100,001..120,000. Verified on the supplied file.
HOSPITAL_A_MAX_ID = 50_000

ID_COL = "Patient_ID"
HOUR_COL = "Hour"
LABEL_COL = "SepsisLabel"

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

# Plausibility limits (inclusive). Values outside are set to NaN.
# These are deliberately wide: they remove impossible values, not unusual ones.
LIMITS = {
    # vitals
    "HR": (20, 250),
    "O2Sat": (50, 100),
    "Temp": (30, 43),
    "SBP": (40, 300),
    "MAP": (20, 200),
    "DBP": (10, 200),
    "Resp": (3, 80),
    # labs
    "SaO2": (50, 100),
    "pH": (6.5, 8.0),
    "Glucose": (10, 1500),
    "Potassium": (1, 9),
    "Hct": (5, 70),
    "Hgb": (2, 25),
    "Platelets": (1, 1500),
    "Creatinine": (0.1, 25),
    "Lactate": (0.1, 30),
    "WBC": (0.1, 200),
}

# FiO2 should be a fraction in [0.21, 1.0]. The raw file contains negatives,
# percentages (e.g. 40, 100) and absurd values up to 4000.
FIO2_FRACTION_MIN = 0.20  # 0.20 allows rounding of room air (0.21)
FIO2_PERCENT_MIN = 21.0   # values 21..100 are treated as percentages
FIO2_PERCENT_MAX = 100.0

AGE_TOPCODE = 100  # ages of 90+ appear top-coded to 100 in this dataset
ADULT_AGE = 18

DROP_COLS = ["Unnamed: 0"]  # per-patient row counter, redundant with Hour


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class Report:
    """Collects everything we change so nothing happens silently."""

    def __init__(self) -> None:
        self.data: dict = {"rules": [], "notes": []}

    def rule(self, column: str, rule: str, n_changed: int) -> None:
        self.data["rules"].append(
            {"column": column, "rule": rule, "n_changed": int(n_changed)}
        )

    def note(self, text: str) -> None:
        self.data["notes"].append(text)
        print(f"[note] {text}")

    def set(self, key: str, value) -> None:
        self.data[key] = value


def load_raw(path: Path, report: Report) -> pd.DataFrame:
    """Read the CSV (pandas reads a single-file zip directly)."""
    print(f"[load] reading {path} ...")
    df = pd.read_csv(path, dtype={ID_COL: str})

    # Patient_ID may be written with leading zeros (e.g. '017072').
    # A truncated upload can leave a broken last row with no ID.
    bad = df[ID_COL].isna()
    n_bad = int(bad.sum())
    if n_bad:
        report.note(
            f"{n_bad} row(s) had no Patient_ID (likely a truncated file) and were dropped."
        )
        # If the broken row is the last line, the final patient is cut off mid-record
        # (its later hours are missing), so drop that patient entirely.
        if bad.iloc[-1]:
            last_pid = df.loc[~bad, ID_COL].iloc[-1]
            n_partial = int((df[ID_COL] == last_pid).sum())
            df = df[df[ID_COL] != last_pid]
            report.note(
                f"File ends mid-row: also dropped final patient {last_pid} "
                f"({n_partial} rows, record incomplete)."
            )
        df = df.dropna(subset=[ID_COL])
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="raise").astype("int64")
    return df


def check_required_columns(df: pd.DataFrame) -> None:
    required = [ID_COL, HOUR_COL, LABEL_COL, "ICULOS", "Age", "Gender",
                "Unit1", "Unit2", "HospAdmTime", "FiO2"] + VITALS
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: required columns missing from input: {missing}")


# --------------------------------------------------------------------------- #
# Cleaning steps
# --------------------------------------------------------------------------- #

def clean_fio2(df: pd.DataFrame, report: Report) -> None:
    """Convert FiO2 to a fraction in [0.20, 1.0]; set anything else to NaN."""
    x = df["FiO2"].copy()
    notna = x.notna()

    neg_or_zero = notna & (x <= 0)
    too_big = notna & (x > FIO2_PERCENT_MAX)
    as_percent = notna & (x >= FIO2_PERCENT_MIN) & (x <= FIO2_PERCENT_MAX)
    ambiguous = notna & (x > 1.0) & (x < FIO2_PERCENT_MIN)
    too_small = notna & (x > 0) & (x < FIO2_FRACTION_MIN)

    out = x.copy()
    out[as_percent] = x[as_percent] / 100.0
    out[neg_or_zero | too_big | ambiguous | too_small] = np.nan

    report.rule("FiO2", "<=0 -> NaN", neg_or_zero.sum())
    report.rule("FiO2", f">{FIO2_PERCENT_MAX:g} -> NaN", too_big.sum())
    report.rule("FiO2", f"{FIO2_PERCENT_MIN:g}..{FIO2_PERCENT_MAX:g} treated as percent, /100", as_percent.sum())
    report.rule("FiO2", f"1<x<{FIO2_PERCENT_MIN:g} (ambiguous) -> NaN", ambiguous.sum())
    report.rule("FiO2", f"0<x<{FIO2_FRACTION_MIN} -> NaN", too_small.sum())
    df["FiO2"] = out


def apply_limits(df: pd.DataFrame, report: Report) -> None:
    """Set values outside plausibility limits to NaN, logging each side."""
    for col, (lo, hi) in LIMITS.items():
        if col not in df.columns:
            continue
        x = df[col]
        below = x.notna() & (x < lo)
        above = x.notna() & (x > hi)
        if below.any():
            report.rule(col, f"<{lo} -> NaN", below.sum())
        if above.any():
            report.rule(col, f">{hi} -> NaN", above.sum())
        df.loc[below | above, col] = np.nan


def add_flags(df: pd.DataFrame, report: Report) -> None:
    """Flags that keep suspicious-but-not-impossible rows visible."""
    # SBP < DBP is physiologically inconsistent; MAP should sit between DBP and SBP.
    sbp_lt_dbp = df["SBP"].notna() & df["DBP"].notna() & (df["SBP"] < df["DBP"])
    df["bp_sbp_lt_dbp"] = sbp_lt_dbp
    map_out = (
        df["MAP"].notna() & df["SBP"].notna() & df["DBP"].notna()
        & ((df["MAP"] < df["DBP"] - 1) | (df["MAP"] > df["SBP"] + 1))
    )
    df["bp_map_outside_range"] = map_out
    report.rule("SBP/DBP", "flag bp_sbp_lt_dbp (not removed)", sbp_lt_dbp.sum())
    report.rule("MAP", "flag bp_map_outside_range (not removed)", map_out.sum())

    # Age flags (patient-level constant, stored per row for convenience).
    df["age_topcoded"] = df["Age"] >= AGE_TOPCODE
    df["age_lt18"] = df["Age"] < ADULT_AGE

    # HospAdmTime is extremely skewed (-5367 .. +24 hours): add a signed-log copy.
    df["HospAdmTime_slog"] = np.sign(df["HospAdmTime"]) * np.log1p(df["HospAdmTime"].abs())


def add_unit_and_hospital(df: pd.DataFrame, report: Report) -> None:
    """Merge Unit1/Unit2 into one code; tag hospital from Patient_ID range."""
    # Per the PhysioNet 2019 description Unit1 = MICU, Unit2 = SICU (verify against
    # your copy of the documentation). In this file the two are exact complements
    # whenever present, and both are missing together for ~39% of patients.
    both_present = df["Unit1"].notna() & df["Unit2"].notna()
    inconsistent = both_present & ((df["Unit1"] + df["Unit2"]) != 1)
    if inconsistent.any():
        report.note(f"{int(inconsistent.sum())} rows where Unit1+Unit2 != 1 (unexpected).")
    unit = np.select(
        [df["Unit1"] == 1, df["Unit2"] == 1],
        [1, 2],
        default=0,
    ).astype("int8")
    df["unit"] = unit  # 0 = unknown, 1 = Unit1 (MICU), 2 = Unit2 (SICU)

    df["hospital"] = np.where(df[ID_COL] <= HOSPITAL_A_MAX_ID, "A", "B")
    df["hospital"] = df["hospital"].astype("category")


def add_label_timing(df: pd.DataFrame, report: Report) -> None:
    """
    Label-timing columns. SepsisLabel is non-decreasing within a patient (verified),
    so the first row with label 1 is the label's switch-on hour.

    NOTE: 'first_pos_hour' is the first hour where the *dataset label* is 1. It may be
    shifted relative to the clinical Sepsis-3 onset (check the Challenge documentation).
    We never call it 'clinical onset'.
    """
    first_pos = (
        df.loc[df[LABEL_COL] == 1].groupby(ID_COL)[HOUR_COL].min().rename("first_pos_hour")
    )
    df["first_pos_hour"] = df[ID_COL].map(first_pos).astype("float32")
    df["septic_patient"] = df["first_pos_hour"].notna()
    df["hours_to_onset"] = (df["first_pos_hour"] - df[HOUR_COL]).astype("float32")
    # Non-septic: all rows are usable. Septic: only rows strictly before the label switches on.
    df["is_pre_onset"] = (~df["septic_patient"]) | (df[HOUR_COL] < df["first_pos_hour"])
    report.note(
        f"Rows at/after label switch-on (not pre-onset): {int((~df['is_pre_onset']).sum())} "
        f"of {len(df)}."
    )


# --------------------------------------------------------------------------- #
# Patient table
# --------------------------------------------------------------------------- #

def build_patient_table(df: pd.DataFrame, min_stay: int, min_age: float) -> pd.DataFrame:
    g = df.groupby(ID_COL, sort=True)
    pt = pd.DataFrame(
        {
            "hospital": g["hospital"].first().astype(str),
            "n_hours": g.size(),
            "septic": g["septic_patient"].first(),
            "first_pos_hour": g["first_pos_hour"].first(),
            "age": g["Age"].first(),
            "gender": g["Gender"].first(),
            "unit": g["unit"].first(),
            "age_lt18": g["age_lt18"].first(),
            "age_topcoded": g["age_topcoded"].first(),
        }
    )
    pt["onset_hour0"] = pt["first_pos_hour"] == 0
    pt["onset_lt6h"] = pt["first_pos_hour"] < 6  # label on within the first 6 recorded hours
    pt["short_stay"] = pt["n_hours"] < 12
    # Main-analysis inclusion. Hour-0 positives have no pre-onset history, so they are
    # always excluded. Stay length and age exclusions are optional (CLI flags).
    include = ~pt["onset_hour0"]
    if min_stay > 0:
        include &= pt["n_hours"] >= min_stay
    if min_age > 0:
        include &= pt["age"] >= min_age
    pt["include_main"] = include
    return pt.reset_index()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate(df: pd.DataFrame, n_rows_in: int, n_pat_in: int) -> None:
    print("[validate] running integrity checks ...")
    assert len(df) == n_rows_in, "row count changed (cleaning must not delete rows)"
    assert df[ID_COL].nunique() == n_pat_in, "patient count changed"
    assert not df.duplicated([ID_COL, HOUR_COL]).any(), "duplicate (patient, hour) rows"

    g = df.groupby(ID_COL)
    assert (g[HOUR_COL].diff().dropna() == 1).all(), "hours are not consecutive within patient"
    assert (g[LABEL_COL].diff().dropna() >= 0).all(), "label decreases within a patient"
    assert (g[HOUR_COL].min() == 0).all(), "Hour does not start at 0"

    # Sanity of cleaned ranges
    f = df["FiO2"].dropna()
    assert f.empty or ((f >= FIO2_FRACTION_MIN) & (f <= 1.0)).all(), "FiO2 out of range after cleaning"
    for col, (lo, hi) in LIMITS.items():
        x = df[col].dropna()
        assert x.empty or ((x >= lo) & (x <= hi)).all(), f"{col} out of limits after cleaning"

    # Pre-onset logic
    sep = df[df["septic_patient"]]
    assert (sep.loc[sep["is_pre_onset"], HOUR_COL] < sep.loc[sep["is_pre_onset"], "first_pos_hour"]).all()
    assert (sep.loc[sep["is_pre_onset"], LABEL_COL] == 0).all(), "pre-onset rows contain label 1"
    print("[validate] all checks passed")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_hourly(df: pd.DataFrame, outdir: Path, report: Report) -> str:
    try:
        path = outdir / "hourly_clean.parquet"
        df.to_parquet(path, index=False)
    except ImportError:
        path = outdir / "hourly_clean.csv.gz"
        report.note("pyarrow not installed: wrote gzip CSV instead of parquet "
                    "(pip install pyarrow for faster loading).")
        df.to_csv(path, index=False, compression="gzip")
    print(f"[write] {path}")
    return path.name


def missing_summary(df: pd.DataFrame) -> dict:
    m = (df.isna().mean() * 100).round(2)
    return m[m > 0].sort_values(ascending=False).to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="raw CSV, or a zip containing one CSV")
    ap.add_argument("--outdir", required=True, type=Path, help="output directory (created if needed)")
    ap.add_argument("--min-stay-hours", type=int, default=0,
                    help="optional: exclude stays shorter than this from include_main (0 = no exclusion)")
    ap.add_argument("--min-age", type=float, default=0,
                    help="optional: exclude patients younger than this from include_main (0 = no exclusion)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = Report()

    df = load_raw(args.input, report)
    check_required_columns(df)
    n_rows_in, n_pat_in = len(df), df[ID_COL].nunique()
    report.set("input", {"file": str(args.input), "rows": n_rows_in, "patients": n_pat_in})
    report.set("missing_pct_before", missing_summary(df))

    # --- structure ---
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)
    report.note(f"Dropped columns: {drop}")
    df = df.sort_values([ID_COL, HOUR_COL], kind="stable").reset_index(drop=True)

    # --- value cleaning (cells only, never rows) ---
    clean_fio2(df, report)
    apply_limits(df, report)
    add_flags(df, report)
    add_unit_and_hospital(df, report)
    add_label_timing(df, report)

    # Redundant after merging into 'unit'
    df = df.drop(columns=["Unit2"])
    report.note("Dropped Unit2 (exact complement of Unit1); merged into 'unit'.")

    # --- downcast floats to save memory ---
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    validate(df, n_rows_in, n_pat_in)

    pt = build_patient_table(df, args.min_stay_hours, args.min_age)

    # --- write ---
    hourly_name = write_hourly(df, args.outdir, report)
    pt.to_csv(args.outdir / "patients.csv", index=False)
    print(f"[write] {args.outdir / 'patients.csv'}")

    report.set("output", {"hourly_file": hourly_name, "rows": len(df), "columns": list(df.columns)})
    report.set("missing_pct_after", missing_summary(df))
    sep = pt[pt["septic"]]
    report.set("patients_summary", {
        "patients": int(len(pt)),
        "by_hospital": pt["hospital"].value_counts().to_dict(),
        "septic": int(pt["septic"].sum()),
        "septic_by_hospital": sep["hospital"].value_counts().to_dict(),
        "onset_hour0_excluded": int(pt["onset_hour0"].sum()),
        "onset_lt6h": int(pt["onset_lt6h"].sum()),
        "short_stay_lt12h": int(pt["short_stay"].sum()),
        "age_lt18": int(pt["age_lt18"].sum()),
        "age_topcoded": int(pt["age_topcoded"].sum()),
        "unit_unknown": int((pt["unit"] == 0).sum()),
        "include_main": int(pt["include_main"].sum()),
        "include_main_septic": int((pt["include_main"] & pt["septic"]).sum()),
        "rows_pre_onset": int(df["is_pre_onset"].sum()),
    })
    with open(args.outdir / "cleaning_report.json", "w") as fh:
        json.dump(report.data, fh, indent=2, default=str)
    print(f"[write] {args.outdir / 'cleaning_report.json'}")

    print("\n=== SUMMARY ===")
    for k, v in report.data["patients_summary"].items():
        print(f"{k:24s} {v}")
    print("\nRows changed by rule (non-zero):")
    for r in report.data["rules"]:
        if r["n_changed"]:
            print(f"  {r['column']:10s} {r['rule']:55s} {r['n_changed']:>8d}")


if __name__ == "__main__":
    main()