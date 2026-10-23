#!/usr/bin/env python3
"""
organ_pathways.py
=================
Which organ was failing when sepsis was flagged? Partial SOFA scores, organ groups and the gate check.

    python backend/organ_pathways.py --outputs outputs

What it does
------------
1. Partial SOFA, hour by hour, for every included patient (standard SOFA cut-offs):
       Clotting     platelets   <150 / <100 / <50 / <20 x10^3/uL      -> 1 / 2 / 3 / 4
       Liver        bilirubin   >=1.2 / >=2.0 / >=6.0 / >=12.0 mg/dL  -> 1 / 2 / 3 / 4
       Kidneys      creatinine  >=1.2 / >=2.0 / >=3.5 / >=5.0 mg/dL   -> 1 / 2 / 3 / 4
       Circulation  MAP < 70 mmHg (3-hour rolling median)              -> 1   (no vasopressor data, so 1 is the maximum)
       Lungs        SpO2/FiO2 ratio < 512 / 357 / 214 / 89             -> 1 / 2 / 3 / 4
                    (S/F cut-offs that approximate the P/F SOFA bands, Pandharipande et al. 2009; PaO2 is not in the data.
                    SpO2 is capped at 97% because S/F is unreliable above it. Lungs are scored only in hours with a
                    recorded FiO2; --room-air instead assumes 0.21 when FiO2 is missing.)
   Brain (GCS) is not in the data, so there is no brain component.
   Labs are carried forward at most --lab-ffill hours (default 24), vitals at most 6 hours; older = unknown.
2. Organ group per septic patient, from the window [F - 24 h, F + 6 h] around the label switch-on hour F:
       rise = highest score in the window - first known score in the window (lab values carried forward
              into it), so an organ only counts when the data show it getting worse; one that was already
              abnormal and stayed so does not count as rising
       an organ "rose" when rise >= 1
       exactly one organ rose -> that organ | two or more -> Multi-organ | none -> Unattributable
   Unattributable is split by reason: "labs not measured" (3 or more organs never measured in the window) or
   "no measurable change".
3. Gate check: is the attributable share >= --gate (default 50%) in EACH hospital?
4. Sensitivity: the grouping repeated with other windows (12/3, 48/12 hours) and carry-forward limits (12, 48 h).

LEAKAGE: the organ group uses hours AFTER the label switch-on. It is for describing and grouping patients only
(evaluation strata, choosing expert training rows), never a model input.

Outputs (in <outputs>/organs/)
------------------------------
    organ_scores.parquet       Patient_ID, Hour, sofa_<organ> (NaN = unknown), fio2_assumed
    organ_assignments.csv      one row per septic patient: organ_group, reason, rise per organ, trigger-lab measurement
    organ_gate.json            attributable share overall / per hospital, pass or fail
    organ_sensitivity.csv      group counts and agreement with the primary grouping, per setting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "Patient_ID"
ORGANS = ["Lungs", "Clotting", "Liver", "Circulation", "Kidneys"]
COL = {o: f"sofa_{o.lower()}" for o in ORGANS}
TRIGGER = {"Lungs": "O2Sat", "Clotting": "Platelets", "Liver": "Bilirubin_total", "Circulation": "MAP",
           "Kidneys": "Creatinine"}            # the measurement each organ score depends on
VITAL_FFILL = 6


def ffill(s: pd.Series, by: pd.Series, limit: int) -> pd.Series:
    return s.groupby(by).ffill(limit=limit)


def sofa_scores(h: pd.DataFrame, lab_ffill: int, room_air: bool = False) -> pd.DataFrame:
    """Hourly partial SOFA. h sorted by patient and hour. Unknown = NaN."""
    by = h[ID]
    out = pd.DataFrame({ID: h[ID].to_numpy(), "Hour": h["Hour"].to_numpy()})

    def band(x, cuts, low):            # low=True: lower values are worse (platelets); False: higher worse
        s = sum(((x < c) if low else (x >= c)).astype(float) for c in cuts)
        return s.where(x.notna())

    out[COL["Clotting"]] = band(ffill(h["Platelets"], by, lab_ffill), [150, 100, 50, 20], True).to_numpy()
    out[COL["Liver"]] = band(ffill(h["Bilirubin_total"], by, lab_ffill), [1.2, 2.0, 6.0, 12.0], False).to_numpy()
    out[COL["Kidneys"]] = band(ffill(h["Creatinine"], by, lab_ffill), [1.2, 2.0, 3.5, 5.0], False).to_numpy()
    mp = ffill(h["MAP"], by, VITAL_FFILL)
    mp = mp.groupby(by).rolling(3, min_periods=1).median().reset_index(level=0, drop=True).sort_index()
    out[COL["Circulation"]] = band(mp, [70], True).to_numpy()
    spo2 = ffill(h["O2Sat"], by, VITAL_FFILL).clip(upper=97)
    fio2 = ffill(h["FiO2"], by, lab_ffill)
    out["fio2_assumed"] = fio2.isna().to_numpy()
    # Default: lungs are scored only when FiO2 was recorded (carried forward up to lab_ffill hours). Assuming room air
    # for missing FiO2 makes the first recorded FiO2 look like a sudden lung deterioration.
    sf = spo2 / (fio2.fillna(0.21) if room_air else fio2)
    out[COL["Lungs"]] = band(sf, [512, 357, 214, 89], True).to_numpy()
    return out


def attribute(sc: pd.DataFrame, F: pd.Series, before: int, after: int) -> pd.DataFrame:
    """Organ group per septic patient from the scores in [F - before, F + after]."""
    s = sc[sc[ID].isin(F.index)].copy()
    s["rel"] = s["Hour"] - s[ID].map(F)
    w = s[(s["rel"] >= -before) & (s["rel"] <= after)].sort_values([ID, "Hour"])
    g = w.groupby(ID)
    res = pd.DataFrame(index=F.index)
    known = pd.DataFrame(index=F.index)
    for o in ORGANS:
        c = COL[o]
        peak = g[c].max()
        base = g[c].first()      # first KNOWN score in the window (values are carried forward into it)
        rise = (peak - base).reindex(F.index)                # NaN = never measured in the window
        res[f"rise_{o.lower()}"] = rise
        known[o] = peak.reindex(F.index).notna()
    rose = pd.DataFrame({o: res[f"rise_{o.lower()}"] >= 1 for o in ORGANS})
    n_rose = rose.sum(axis=1)
    grp = np.where(n_rose == 0, "Unattributable",
                   np.where(n_rose >= 2, "Multi-organ", rose.idxmax(axis=1)))
    res["organ_group"] = grp
    n_unknown = (~known).sum(axis=1)
    res["reason"] = np.select([n_rose == 1, n_rose >= 2, n_unknown >= 3],
                              ["one organ rose", "two or more organs rose", "labs not measured"],
                              default="no measurable change")
    res["organs_rose"] = rose.apply(lambda r: "+".join(o for o in ORGANS if r[o]), axis=1)
    res["n_organs_unknown"] = n_unknown
    return res


def trigger_measurement(h: pd.DataFrame, F: pd.Series) -> pd.DataFrame:
    """How often each organ's trigger measurement was taken in the 24 h before F, and how stale it was at F."""
    s = h[h[ID].isin(F.index)].copy()
    s["rel"] = s["Hour"] - s[ID].map(F)
    pre = s[(s["rel"] >= -24) & (s["rel"] < 0)]
    out = pd.DataFrame(index=F.index)
    for o, v in TRIGGER.items():
        out[f"n_obs24_{o.lower()}"] = pre.groupby(ID)[v].count().reindex(F.index).fillna(0).astype(int)
        last = s[(s["rel"] < 0) & s[v].notna()].groupby(ID)["rel"].max()
        out[f"hours_since_{o.lower()}"] = (-last).reindex(F.index)     # NaN = never measured before F
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--before", type=int, default=24, help="hours before label switch-on in the window")
    ap.add_argument("--after", type=int, default=6, help="hours after label switch-on in the window")
    ap.add_argument("--lab-ffill", type=int, default=24, help="max hours a lab value is carried forward")
    ap.add_argument("--gate", type=float, default=0.5, help="minimum attributable share per hospital")
    ap.add_argument("--room-air", action="store_true", help="assume FiO2 = 0.21 when it is not recorded")
    args = ap.parse_args()

    hpath = args.outputs / "clean" / "hourly_clean.parquet"
    folds = pd.read_csv(args.outputs / "splits" / "folds.csv")
    for c in ("septic", "include_main"):
        folds[c] = folds[c].astype(bool)
    inc = folds[folds["include_main"]]
    cols = [ID, "Hour", "first_pos_hour", "Platelets", "Bilirubin_total", "Creatinine", "MAP", "O2Sat", "FiO2"]
    h = pd.read_parquet(hpath, columns=cols)
    h = h[h[ID].isin(inc[ID])].sort_values([ID, "Hour"]).reset_index(drop=True)
    sep = inc[inc["septic"]].set_index(ID)
    F = h.groupby(ID)["first_pos_hour"].first().reindex(sep.index)
    if F.isna().any():
        sys.exit("ERROR: some septic patients have no label switch-on hour; re-run clean_data.py")
    print(f"[load] {h[ID].nunique()} patients, {len(h)} hours, {len(sep)} septic")

    sc = sofa_scores(h, args.lab_ffill, args.room_air)
    # self-check of the cut-offs on hand-made values
    t = pd.DataFrame({ID: [1] * 6, "Hour": range(6), "first_pos_hour": 3.0,
                      "Platelets": [200, 149, 99, 49, 19, np.nan], "Bilirubin_total": [1.1, 1.2, 2.0, 6.0, 12.0, np.nan],
                      "Creatinine": [1.0, 1.2, 2.0, 3.5, 5.0, np.nan], "MAP": [80, 80, 80, 60, 60, 60],
                      "O2Sat": [97, 97, 97, 97, 90, 90], "FiO2": [0.21, 0.21, 0.4, 0.6, 1.0, np.nan]})
    tc = sofa_scores(t, 24, room_air=True)
    assert tc[COL["Clotting"]].tolist()[:5] == [0, 1, 2, 3, 4] and tc[COL["Clotting"]].iloc[5] == 4  # carried forward
    assert tc[COL["Liver"]].tolist()[:5] == [0, 1, 2, 3, 4] and tc[COL["Kidneys"]].tolist()[:5] == [0, 1, 2, 3, 4]
    assert tc[COL["Circulation"]].tolist() == [0, 0, 0, 0, 1, 1]          # 3-hour median needs 2 low hours
    assert tc[COL["Lungs"]].tolist()[:5] == [1, 1, 2, 3, 3]               # 97/.21=462, 97/.4=243, 97/.6=162, 90/1=90
    print("[validate] SOFA cut-offs checked on hand-made values")

    a = attribute(sc, F, args.before, args.after)
    a = a.join(trigger_measurement(h, F))
    a = a.join(sep[["hospital", "split", "fold"]]).reset_index().rename(columns={"index": ID})
    assert len(a) == len(sep) and a["organ_group"].notna().all()

    # gate check
    att = a["organ_group"] != "Unattributable"
    gate = {"threshold": args.gate, "overall_share": float(att.mean()),
            "by_hospital": {hh: float(att[a["hospital"] == hh].mean()) for hh in sorted(a["hospital"].unique())}}
    gate["passes"] = bool(all(v >= args.gate for v in gate["by_hospital"].values()))
    gate["window"] = {"before_h": args.before, "after_h": args.after, "lab_ffill_h": args.lab_ffill,
                      "lungs_room_air_assumed": bool(args.room_air)}
    gate["counts"] = a["organ_group"].value_counts().to_dict()
    gate["unattributable_reasons"] = a.loc[~att, "reason"].value_counts().to_dict()
    gate["fio2_assumed_share_of_hours"] = float(sc["fio2_assumed"].mean())

    # sensitivity: other windows and carry-forward limits
    rows = []
    for lf in sorted({12, args.lab_ffill, 48}):
        s2 = sc if lf == args.lab_ffill else sofa_scores(h, lf, args.room_air)
        for b, af in ((args.before, args.after), (12, 3), (48, 12)):
            g2 = attribute(s2, F, b, af)["organ_group"]
            agree = float((g2.reindex(a[ID]).to_numpy() == a["organ_group"].to_numpy()).mean())
            vc = g2.value_counts()
            rows.append({"lab_ffill_h": lf, "window": f"-{b} h to +{af} h",
                         "primary": lf == args.lab_ffill and (b, af) == (args.before, args.after),
                         "attributable_%": round(100 * float((g2 != "Unattributable").mean()), 1),
                         "agreement_with_primary_%": round(100 * agree, 1),
                         **{f"n_{k}": int(vc.get(k, 0)) for k in ORGANS + ["Multi-organ", "Unattributable"]}})
    sens = pd.DataFrame(rows)

    od = args.outputs / "organs"
    od.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(od / "organ_scores.parquet", index=False)
    a.to_csv(od / "organ_assignments.csv", index=False)
    json.dump(gate, open(od / "organ_gate.json", "w"), indent=2)
    sens.to_csv(od / "organ_sensitivity.csv", index=False)
    print(f"[write] {od}")
    print("\n=== ORGAN GROUPS (septic patients) ===")
    print(pd.crosstab(a["organ_group"], a["hospital"], margins=True).to_string())
    print(f"\nGATE: attributable {100 * gate['overall_share']:.1f}% overall; " +
          ", ".join(f"hospital {k} {100 * v:.1f}%" for k, v in gate["by_hospital"].items()) +
          f" -> {'PASS' if gate['passes'] else 'FAIL'} (needs >= {100 * args.gate:.0f}% in each hospital)")
    print("\nSensitivity of the grouping:")
    print(sens[["lab_ffill_h", "window", "attributable_%", "agreement_with_primary_%"]].to_string(index=False))


if __name__ == "__main__":
    main()
