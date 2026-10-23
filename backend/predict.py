#!/usr/bin/env python3
"""
predict.py
==========
Predict sepsis risk for ONE patient from their hourly data, with the models the pipeline trained.

    python backend/predict.py --outputs outputs --input my_patient.csv            # writes my_patient_predictions.csv
    (the dashboard's Predict tab uses the same functions)

Input: one row per hour, columns as in the PhysioNet 2019 file (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, the labs,
Age, Gender, Unit1, Unit2, HospAdmTime; Hour and ICULOS optional). Missing columns are treated as not measured.

What happens (the SAME code the pipeline used, imported from the other backend scripts):
    1. cleaning       clean_data.clean_fio2 / apply_limits, unit code, HospAdmTime_slog
    2. features       train_models.make_features (causal: hour t only uses hours <= t)
    3. models         LightGBM, logistic regression, GRU: average of the 5 fold models (as for test patients),
                      plus the integrated ensemble (mean percentile, train_models.ensemble_score)
    4. explanation    LightGBM TreeSHAP per hour (pred_contrib), averaged over the fold models
    5. organs         organ_pathways.sofa_scores (partial SOFA per hour)
Research use only. Not a clinical tool.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"sepsis_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clean = _load("clean_data")
tm = _load("train_models")
organ = _load("organ_pathways")

MODEL_COLS = ["p_lgbm", "p_logreg", "p_gru", "p_ensemble"]
INPUT_COLS = tm.VITALS + ["EtCO2"] + tm.LABS + ["Bilirubin_direct", "TroponinI", "Fibrinogen",
                                               "Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=4)
def load_models(outputs: str) -> dict:
    """Everything needed to score a new patient. Cached, so the dashboard loads it once."""
    import joblib
    import lightgbm as lgb
    out = Path(outputs)
    md = out / "models"
    names = json.load(open(out / "features" / "feature_names.json"))
    k = sorted(md.glob("lgbm_fold[0-9].txt"))
    if not k:
        raise FileNotFoundError(f"no LightGBM fold models in {md}; run the pipeline (python run.py) first")
    b = {"names": names,
         "lgbm": [lgb.Booster(model_file=str(f)) for f in k],
         "logreg": [joblib.load(f) for f in sorted(md.glob("logreg_fold[0-9].joblib"))],
         "gru": [], "torch": None}
    try:
        import torch
        b["torch"] = torch
        b["gru"] = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(md.glob("gru_fold[0-9].pt"))]
    except ImportError:
        pass
    ref = md / "ensemble_reference.json"
    b["ensemble"] = json.load(open(ref)) if ref.exists() else None
    th = out / "evaluation" / "thresholds.json"
    b["thresholds"] = json.load(open(th)) if th.exists() else {}
    return b


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #

def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean one patient's raw hourly table exactly as clean_data.py does (cells only, no rows dropped)."""
    h = raw.copy()
    h.columns = [str(c).strip() for c in h.columns]
    if "Hour" not in h.columns:
        h["Hour"] = np.arange(len(h))
    h = h.sort_values("Hour").reset_index(drop=True)
    h["Hour"] = np.arange(len(h))                      # consecutive hours from 0, as the models expect
    for c in INPUT_COLS:
        if c not in h.columns:
            h[c] = np.nan
        h[c] = pd.to_numeric(h[c], errors="coerce")
    if "ICULOS" not in h.columns or h["ICULOS"].isna().all():
        h["ICULOS"] = h["Hour"] + 1
    h["Patient_ID"] = 1
    rep = clean.Report()
    clean.clean_fio2(h, rep)
    clean.apply_limits(h, rep)
    h["unit"] = np.select([h["Unit1"] == 1, h["Unit2"] == 1], [1, 2], default=0).astype("int8")
    adm = h["HospAdmTime"]                              # NaN stays NaN, as in clean_data.add_flags
    h["HospAdmTime_slog"] = np.sign(adm) * np.log1p(adm.abs())
    return h


def features(h: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    F = tm.make_features(h)
    return F[names]


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def _pick(models: list, fold: int | None) -> list:
    """All fold models (new patient: average of the 5) or only the one fold model that did NOT train on a
    development patient (fold = that patient's fold), which reproduces its out-of-fold score."""
    return models if fold is None or fold < 0 else [models[fold]]


def _gru_scores(b: dict, F: pd.DataFrame, fold: int | None = None) -> np.ndarray | None:
    if not b["gru"] or b["torch"] is None:
        return None
    torch = b["torch"]
    SeqNet = tm._seqnet(torch)
    out = []
    for ck in _pick(b["gru"], fold):
        X = F[ck["cols"]].to_numpy(np.float32)
        z = np.nan_to_num(np.clip((X - ck["mu"]) / ck["sd"], -5, 5), nan=0.0)
        xb = np.concatenate([z, np.isnan(X[:, ck["ind_idx"]]).astype(np.float32)], axis=1)[None]
        cfg = ck["config"]
        net = SeqNet(xb.shape[2], cfg["hidden"], cfg["head"], max(1, len(ck["patterns"])), cfg["dropout"])
        net.load_state_dict(ck["state"])
        net.eval()
        with torch.no_grad():
            out.append(torch.sigmoid(net(torch.from_numpy(xb))[0])[0].numpy())
    return np.mean(out, axis=0)


def predict(raw: pd.DataFrame, outputs: str | Path, fold: int | None = None) -> dict:
    """Returns {"hours": per-hour table (scores, alerts, organ scores), "contrib": SHAP per hour, "features": F}.
    fold=None: a new patient, scored by the average of the 5 fold models. fold=k: a development patient of fold k,
    scored only by the fold model that never saw it (its honest, out-of-fold prediction)."""
    b = load_models(str(outputs))
    h = prepare(raw)
    F = features(h, b["names"])
    X = F.to_numpy(np.float32)
    res = pd.DataFrame({"Hour": h["Hour"].to_numpy()})
    res["p_lgbm"] = np.mean([m.predict(X) for m in _pick(b["lgbm"], fold)], axis=0)
    if b["logreg"]:
        res["p_logreg"] = np.mean([m.predict_proba(X)[:, 1] for m in _pick(b["logreg"], fold)], axis=0)
    g = _gru_scores(b, F, fold)
    if g is not None:
        res["p_gru"] = g
    if b["ensemble"] is not None and all(m in res for m in b["ensemble"]["members"]):
        res["p_ensemble"] = tm.ensemble_score({m: res[m].to_numpy() for m in b["ensemble"]["members"]},
                                              b["ensemble"]["reference"])
    for m in [c for c in MODEL_COLS if c in res]:
        for op, tau in b["thresholds"].get(m, {}).items():
            res[f"alert_{m}_{op}"] = res[m] >= tau
    contrib = np.mean([m.predict(X, pred_contrib=True)[:, :-1] for m in _pick(b["lgbm"], fold)], axis=0)
    sc = organ.sofa_scores(h, 24)
    for o in organ.ORGANS:
        res[organ.COL[o]] = sc[organ.COL[o]].to_numpy()
    return {"hours": res, "contrib": pd.DataFrame(contrib, columns=b["names"]), "features": F, "clean": h}


# --------------------------------------------------------------------------- #
# Plain-language reasons
# --------------------------------------------------------------------------- #

NICE = {"HR": "heart rate", "O2Sat": "oxygen saturation", "Temp": "temperature", "SBP": "systolic blood pressure",
        "MAP": "mean blood pressure", "DBP": "diastolic blood pressure", "Resp": "breathing rate", "FiO2": "oxygen given (FiO2)",
        "pH": "blood pH", "PaCO2": "blood CO2", "SaO2": "arterial oxygen", "BaseExcess": "base excess", "HCO3": "bicarbonate",
        "AST": "liver enzyme AST", "BUN": "urea (BUN)", "Alkalinephos": "alkaline phosphatase", "Calcium": "calcium",
        "Chloride": "chloride", "Creatinine": "creatinine", "Glucose": "glucose", "Lactate": "lactate",
        "Magnesium": "magnesium", "Phosphate": "phosphate", "Potassium": "potassium", "Bilirubin_total": "bilirubin",
        "Hct": "haematocrit", "Hgb": "haemoglobin", "PTT": "clotting time (PTT)", "WBC": "white cell count",
        "Platelets": "platelets", "ICULOS": "hours in the ICU", "HospAdmTime_slog": "time since hospital admission",
        "Age": "age", "Gender": "sex", "unit": "ICU type", "n_vital_obs_6h": "number of vital-sign readings in 6 h",
        "n_lab_obs_12h": "number of lab tests in 12 h", "n_lab_hours_12h": "hours with lab tests in 12 h",
        "n_lab_obs_total": "lab tests so far", "lab_age_any": "hours since the last lab test"}
SUFFIX = {"_ffill": "", "_mean6": " (6-hour average)", "_std6": " (6-hour variability)", "_delta6": " (change over 6 h)",
          "_age": " (hours since measured)"}


def nice_name(f: str) -> str:
    if f in NICE:
        return NICE[f]
    for sfx, words in SUFFIX.items():
        if f.endswith(sfx):
            base = f[: -len(sfx)]
            return NICE.get(base, base) + words
    return f


def reasons(contrib_row: pd.Series, feat_row: pd.Series, k: int = 6) -> pd.DataFrame:
    """Top k inputs that moved this hour's LightGBM risk, with their value and direction."""
    top = contrib_row.abs().sort_values(ascending=False).head(k).index
    return pd.DataFrame({"input": [nice_name(f) for f in top],
                         "value": [feat_row[f] for f in top],
                         "effect": ["raises risk" if contrib_row[f] > 0 else "lowers risk" for f in top],
                         "strength": [abs(float(contrib_row[f])) for f in top],
                         "feature": list(top)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--input", type=Path, required=True, help="CSV with one row per hour for one patient")
    args = ap.parse_args()
    raw = pd.read_csv(args.input)
    if "Patient_ID" in raw.columns and raw["Patient_ID"].nunique() > 1:
        sys.exit("ERROR: the file has several patients; give one patient at a time")
    r = predict(raw, args.outputs)["hours"]
    dest = args.input.with_name(args.input.stem + "_predictions.csv")
    r.to_csv(dest, index=False)
    print(r[[c for c in ["Hour"] + MODEL_COLS if c in r]].round(4).tail(12).to_string(index=False))
    print(f"[write] {dest}")


if __name__ == "__main__":
    main()
