#!/usr/bin/env python3
"""
05_train_models.py
==================
Idea B: deterioration-pattern-specific sepsis early warning (PhysioNet/CinC 2019).

One file, run in stages:

    python 05_train_models.py features  --outputs outputs
    python 05_train_models.py baselines --outputs outputs
    python 05_train_models.py ablation  --outputs outputs     # includes the time_only baseline
    python 05_train_models.py gru       --outputs outputs --head single   # p_gru
    python 05_train_models.py gru       --outputs outputs --head mtl      # p_gru_mtl
    python 05_train_models.py gru       --outputs outputs --head moe      # p_gru_moe
    python 05_train_models.py gru       --outputs outputs --exclude TIME  # p_gru_no_time (optional)
    python 05_train_models.py pattern   --outputs outputs     # p_lgbm_expert_<pattern>, p_lgbm_moe_max
    python 05_train_models.py transfer  --outputs outputs     # p_lgbm_within, p_lgbm_transfer
    python 05_train_models.py ensemble  --outputs outputs     # p_ensemble (LightGBM + logistic + GRU)

Every stage adds p_* columns to predictions/predictions.parquet; 06_evaluate.py picks them up.

Prediction task
---------------
For every patient-hour t that is BEFORE the label switches on (is_pre_onset), predict
        y(t) = 1  if the label switches on within the next H hours (default H = 12)
        y(t) = 0  otherwise (non-septic patients, and septic patients further than H hours
                  before switch-on)
Rows at or after the label switch-on are never used, because by then the label is already 1
(there is nothing left to warn about). "Label switch-on" is the dataset label's first hour of 1,
which may differ from the clinical Sepsis-3 onset time.

Features (stage `features`), all CAUSAL (use only hours <= t of the same patient)
-------------------------------------------------------------------------------
per vital (HR, O2Sat, Temp, SBP, MAP, DBP, Resp):
    <v>_ffill   last observed value (carried forward at most 24 h)
    <v>_age     hours since last observation (capped at 72; NaN = never observed)
    <v>_mean6, <v>_std6   mean / std of observed values in the last 6 h
    <v>_delta6  current carried-forward value minus the value 6 h earlier
per lab (23 labs; Bilirubin_direct, Fibrinogen, TroponinI dropped: >98% missing):
    <lab>_ffill, <lab>_age
measurement-activity features (these encode how much clinicians are measuring):
    n_vital_obs_6h, n_lab_obs_12h, n_lab_hours_12h, lab_age_any, n_lab_obs_total
static: Age, Gender, unit, HospAdmTime_slog, ICULOS
The hospital is NOT a feature. Missing values stay NaN (LightGBM handles them; logistic
regression uses median imputation fitted on its training rows only).

Models and evaluation (stage `baselines`)
-----------------------------------------
* The frozen 5 dev folds from 02_make_splits.py are used for out-of-fold (OOF) predictions.
* LightGBM: early stopping (on AUROC) on an inner patient-level 10% holdout taken from the
  training folds, so the validation fold is never used to choose the number of trees.
  Default settings are strongly regularised (7 leaves, min 5000 rows per leaf, L2 = 50, no
  class weighting). With lighter settings (31 leaves, 200 rows per leaf, weight 5) validation
  AUPRC peaked after ~15 trees and then fell: hourly rows of one patient are highly
  correlated, so trees memorise patients. The regularised settings were chosen with one
  coarse probe on the inner holdout of fold 0's training patients only (157 septic patients,
  4 configurations), so treat them as reasonable, not tuned.
* Test-set scores are the average of the 5 fold models. They are SAVED but NOT evaluated here:
  the test set is reserved for the final report (06_evaluate.py).
* Printed metrics are row-level OOF AUROC / AUPRC on dev rows only. Row-level AUPRC is
  dominated by the ~2% positive rate; patient-level lead time and the PhysioNet utility score
  come in 06_evaluate.py.

Ablation (stage `ablation`)
---------------------------
The baseline model leans heavily on ICULOS (time in ICU) and the cumulative lab count. This
stage retrains the SAME LightGBM recipe (same folds, seeds, settings) on feature groups:
    TIME   ICULOS, HospAdmTime_slog
    MEAS   every <x>_age feature + n_vital_obs_6h, n_lab_obs_12h, n_lab_hours_12h,
           n_lab_obs_total, lab_age_any          (how often / how recently things were measured)
    PHYS   <x>_ffill, <v>_mean6, <v>_std6, <v>_delta6   (the measured values themselves)
    DEMO   Age, Gender, unit
variants (new score columns p_lgbm_<variant> in predictions.parquet, picked up by 06_evaluate.py):
    no_time        MEAS + PHYS + DEMO
    no_meas        TIME + PHYS + DEMO
    phys_only      PHYS + DEMO                 (no time, no explicit measurement-behaviour features)
    behavior_only  TIME + MEAS                 (no measured values and no demographics at all)
    time_only      TIME                        (time-in-ICU baseline: ICULOS + HospAdmTime only)
    full           all features, but listed in group order. NOT bit-identical to p_lgbm (the column
                   order changes LightGBM's random feature sampling), so |full - p_lgbm| is a
                   handy noise floor: differences between variants smaller than that are noise
LIMIT: removing MEAS does not remove all measurement-behaviour information. A value feature
is NaN when nothing was measured in the last 24 h, and trees can split on NaN, so phys_only
still carries some. Read phys_only as "much less behaviour", not "no behaviour".
As always the test-set scores are saved but not evaluated here.

GRU (stage `gru`)
-----------------
One GRU over each patient's hourly sequence of the SAME causal features (z-scored with train-fold
statistics, missing -> 0, plus a "not measured in the last 24 h" flag per *_ffill column). Same 5
frozen folds and the same inner 10% early-stopping holdout idea (stratified by septic status, so
it always contains positives); early stopping on holdout AUROC. A causality self-check confirms
that cutting off later hours does not change earlier scores. Heads:
    single  p_gru       sepsis logit only
    mtl     p_gru_mtl   + auxiliary pattern classifier (multi-task)
    moe     p_gru_moe   K expert logits mixed by a softmax gate (mixture of experts); the gate is
                        supervised with the pattern when --aux-weight > 0
The pattern is a TRAINING TARGET only (taught on hours inside the warning window of septic patients
with an assigned pattern; non-septic hours = "none" for mtl). It is never an input.

Pattern experts (stage `pattern`)
---------------------------------
One LightGBM per pattern, trained on all non-septic patients + septic patients of that pattern
(same recipe and folds). p_lgbm_expert_<pattern> scores every row. Uses:
    * oracle (not deployable): score each septic patient with its own pattern's expert
      (06_evaluate.py compare builds this)
    * p_lgbm_moe_max: max over experts, a deployable mixture with no learned gate
Patterns with fewer than --min-patients dev septic patients are skipped.

Hospital transfer (stage `transfer`)
------------------------------------
    p_lgbm_within    5-fold OOF, but each hospital's rows are scored by models trained on that
                     hospital only (the fair within-hospital reference)
    p_lgbm_transfer  one model trained on ALL dev patients of hospital A scores every hospital-B row
                     (dev and test), and the reverse
Thresholds for the transfer evaluation come from the source hospital (06_evaluate.py transfer).

Outputs (under --outputs)
-------------------------
    features/hourly_features.parquet, features/feature_names.json, features/features_report.json
    predictions/predictions.parquet    meta columns + every p_* score column
    models/lgbm_fold{k}.txt, models/logreg_fold{k}.joblib, models/lgbm_feature_importance.csv
    models/baseline_oof_metrics.csv, models/baselines_report.json
    models/ablation_oof_metrics.csv, models/ablation_feature_importance.csv, models/ablation_groups.json
    models/<gru tag>_fold{k}.pt, <gru tag>_training_log.csv, <gru tag>_report.json
    models/lgbm_expert_<pattern>_fold{k}.txt, models/pattern_experts.json
    models/lgbm_transfer_AtoB.txt, lgbm_transfer_BtoA.txt, models/transfer_report.json
With --patient-frac < 1 (quick test; baselines and gru) every output gets a "_quick" suffix instead.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ID = "Patient_ID"
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "Hct", "Hgb", "PTT", "WBC", "Platelets",
]  # 23 labs (Bilirubin_direct, Fibrinogen, TroponinI dropped)
STATIC = ["Age", "Gender", "unit", "HospAdmTime_slog", "ICULOS"]
META = [ID, "Hour", "y", "hours_to_onset", "septic_patient", "split", "fold", "hospital", "pattern"]
FFILL_LIMIT = 24
AGE_CAP = 72


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def get_paths(out: Path, quick: bool) -> dict:
    sfx = "_quick" if quick else ""
    return {
        "hourly": out / "clean" / "hourly_clean.parquet",
        "folds": out / "splits" / "folds.csv",
        "assign": out / "patterns" / "pattern_assignments.csv",
        "features": out / "features" / f"hourly_features{sfx}.parquet",
        "feat_names": out / "features" / f"feature_names{sfx}.json",
        "feat_report": out / "features" / f"features_report{sfx}.json",
        "pred": out / "predictions" / f"predictions{sfx}.parquet",
        "models": out / "models",
        "sfx": sfx,
    }


# --------------------------------------------------------------------------- #
# Stage 1: features
# --------------------------------------------------------------------------- #

def _lag_window_sum(s: pd.Series, by: pd.Series, w: int) -> pd.Series:
    """Sum of the last w rows (including current) per patient via cumulative sums."""
    cs = s.groupby(by).cumsum()
    return cs - cs.groupby(by).shift(w).fillna(0.0)


def make_features(h: pd.DataFrame) -> pd.DataFrame:
    """h must be sorted by (Patient_ID, Hour) with consecutive hours per patient."""
    by = h[ID]
    cols: dict[str, pd.Series] = {}

    # ---- carried-forward value and age for vitals and labs ----
    for v in VITALS + LABS:
        x = h[v]
        obs = x.notna()
        cols[f"{v}_ffill"] = x.groupby(by).ffill(limit=FFILL_LIMIT)
        last_hour = h["Hour"].where(obs).groupby(by).ffill()
        cols[f"{v}_age"] = (h["Hour"] - last_hour).clip(upper=AGE_CAP)

    # ---- rolling 6 h statistics for vitals (observed values only) ----
    n_vital_obs_6h = pd.Series(0.0, index=h.index)
    for v in VITALS:
        x = h[v].astype("float64")
        obs = x.notna().astype("float64")
        x0 = x.fillna(0.0)
        n6 = _lag_window_sum(obs, by, 6)
        s6 = _lag_window_sum(x0, by, 6)
        q6 = _lag_window_sum(x0 ** 2, by, 6)
        n6z = n6.where(n6 > 0)
        mean6 = s6 / n6z
        var6 = (q6 / n6z - mean6 ** 2).clip(lower=0)
        cols[f"{v}_mean6"] = mean6
        cols[f"{v}_std6"] = np.sqrt(var6).where(n6 >= 2)
        ff = cols[f"{v}_ffill"]
        cols[f"{v}_delta6"] = ff - ff.groupby(by).shift(6)
        n_vital_obs_6h += n6

    # ---- measurement-activity features ----
    lab_obs = h[LABS].notna()
    n_lab_now = lab_obs.sum(axis=1).astype("float64")
    any_lab = lab_obs.any(axis=1).astype("float64")
    cols["n_vital_obs_6h"] = n_vital_obs_6h
    cols["n_lab_obs_12h"] = _lag_window_sum(n_lab_now, by, 12)
    cols["n_lab_hours_12h"] = _lag_window_sum(any_lab, by, 12)
    cols["n_lab_obs_total"] = n_lab_now.groupby(by).cumsum()
    last_lab_hour = h["Hour"].where(any_lab.astype(bool)).groupby(by).ffill()
    cols["lab_age_any"] = (h["Hour"] - last_lab_hour).clip(upper=AGE_CAP)

    for c in STATIC:
        cols[c] = h[c]

    F = pd.concat(cols, axis=1).astype("float32")
    return F


def selfcheck_features(h: pd.DataFrame, F: pd.DataFrame, n: int = 6, seed: int = 0) -> None:
    """Recompute a few features by brute force and compare (also proves causality)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(h), size=n, replace=False)
    for i in idx:
        pid, hour = h.iloc[i][ID], h.iloc[i]["Hour"]
        g = h[(h[ID] == pid) & (h["Hour"] <= hour)]          # history only
        for v in ("HR", "MAP", "Lactate"):
            vals = g[v]
            last = vals.dropna()
            ff = np.nan
            if len(last) and (hour - g.loc[last.index[-1], "Hour"]) <= FFILL_LIMIT:
                ff = last.iloc[-1]
            exp_age = min(hour - g.loc[last.index[-1], "Hour"], AGE_CAP) if len(last) else np.nan
            got_ff, got_age = F.iloc[i][f"{v}_ffill"], F.iloc[i][f"{v}_age"]
            assert (np.isnan(ff) and np.isnan(got_ff)) or np.isclose(ff, got_ff, atol=1e-4), f"{v}_ffill mismatch"
            assert (np.isnan(exp_age) and np.isnan(got_age)) or np.isclose(exp_age, got_age), f"{v}_age mismatch"
        w = g[g["Hour"] > hour - 6]["HR"].dropna()
        exp_mean = w.mean() if len(w) else np.nan
        got = F.iloc[i]["HR_mean6"]
        assert (np.isnan(exp_mean) and np.isnan(got)) or np.isclose(exp_mean, got, atol=1e-3), "HR_mean6 mismatch"
        exp_nlab12 = g[g["Hour"] > hour - 12][LABS].notna().sum().sum()
        assert np.isclose(exp_nlab12, F.iloc[i]["n_lab_obs_12h"]), "n_lab_obs_12h mismatch"
    print(f"[validate] brute-force check of {n} random rows passed (features use history only)")


def stage_features(args: argparse.Namespace) -> None:
    P = get_paths(args.outputs, quick=False)
    t0 = time.time()
    folds = pd.read_csv(P["folds"])
    for c in ("septic", "include_main"):
        folds[c] = folds[c].astype(bool)
    keep = folds[folds["include_main"]][[ID, "split", "fold", "hospital"]]

    need = [ID, "Hour", "is_pre_onset", "septic_patient", "hours_to_onset"] + VITALS + LABS + \
           [c for c in STATIC]
    h = pd.read_parquet(P["hourly"], columns=list(dict.fromkeys(need)))
    n_all = len(h)
    h = h[h["is_pre_onset"]].merge(keep, on=ID, how="inner")
    h = h.sort_values([ID, "Hour"]).reset_index(drop=True)
    print(f"[load] {n_all} rows -> {len(h)} pre-onset rows of {h[ID].nunique()} included patients")

    # target
    h["y"] = ((h["hours_to_onset"] <= args.horizon) & h["septic_patient"]).astype("int8")

    # pattern labels (optional)
    if P["assign"].exists():
        a = pd.read_csv(P["assign"])[[ID, "pattern"]]
        h = h.merge(a, on=ID, how="left")
        h["pattern"] = h["pattern"].fillna("none")
        h.loc[~h["septic_patient"], "pattern"] = "none"
    else:
        print("[note] pattern_assignments.csv not found; pattern column set to 'unavailable'")
        h["pattern"] = "unavailable"

    # consecutive-hours sanity
    assert (h.groupby(ID)["Hour"].diff().dropna() == 1).all(), "hours not consecutive"

    F = make_features(h)
    selfcheck_features(h, F)
    assert np.isfinite(F.to_numpy()[~np.isnan(F.to_numpy())]).all(), "infinite feature values"

    meta = h[META].copy()
    meta["hospital"] = meta["hospital"].astype("category")
    meta["pattern"] = meta["pattern"].astype("category")
    out = pd.concat([meta, F], axis=1)
    P["features"].parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(P["features"], index=False)
    names = list(F.columns)
    json.dump(names, open(P["feat_names"], "w"))

    rep = {
        "horizon_hours": args.horizon, "rows": int(len(out)), "patients": int(out[ID].nunique()),
        "n_features": len(names),
        "positives_by_split": out.groupby("split")["y"].sum().to_dict(),
        "positive_rate_by_split": out.groupby("split")["y"].mean().round(5).to_dict(),
        "rows_by_split": out["split"].value_counts().to_dict(),
        "missing_share_top10": F.isna().mean().sort_values(ascending=False).head(10).round(3).to_dict(),
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(rep, open(P["feat_report"], "w"), indent=2, default=str)
    print(f"[write] {P['features']}  ({len(out)} rows x {len(names)} features)")
    print(f"        positives by split: {rep['positives_by_split']} | rate: {rep['positive_rate_by_split']}")
    print(f"        horizon = {args.horizon} h | {rep['seconds']} s")


# --------------------------------------------------------------------------- #
# Stage 2: baselines
# --------------------------------------------------------------------------- #

def subsample_patients(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Quick-test helper: keep a fraction of patients, stratified by hospital x septic."""
    pt = df.groupby(ID).agg(h=("hospital", "first"), s=("septic_patient", "first")).reset_index()
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in pt.groupby(["h", "s"], observed=True):
        n = max(2, int(round(len(g) * frac)))
        keep.extend(rng.choice(g[ID].values, size=min(n, len(g)), replace=False))
    return df[df[ID].isin(keep)].reset_index(drop=True)


def train_idx(mask: np.ndarray, y: np.ndarray, neg_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Row indices of the training mask; negatives subsampled to neg_frac, positives kept."""
    idx = np.flatnonzero(mask)
    if neg_frac >= 1.0:
        return idx
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    neg = rng.choice(neg, size=int(len(neg) * neg_frac), replace=False)
    return np.sort(np.concatenate([pos, neg]))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    if y.sum() == 0 or y.sum() == len(y):
        return {"auroc": np.nan, "auprc": np.nan, "prevalence": float(y.mean()), "n": int(len(y))}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "prevalence": float(y.mean()), "n": int(len(y))}


def stage_baselines(args: argparse.Namespace) -> None:
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    try:
        import lightgbm as lgb
    except ImportError:
        sys.exit("ERROR: lightgbm is not installed. Run: pip install lightgbm")

    quick = args.patient_frac < 1.0
    P = get_paths(args.outputs, quick)
    src = get_paths(args.outputs, quick=False)       # features always read from the full file
    names = json.load(open(src["feat_names"]))
    df = pd.read_parquet(src["features"])
    if quick:
        df = subsample_patients(df, args.patient_frac, args.seed)
        print(f"[quick] using {df[ID].nunique()} patients ({len(df)} rows); outputs get a '_quick' suffix")
    print(f"[load] {len(df)} rows, {len(names)} features")

    X = df[names].to_numpy(np.float32)
    df = df[META].copy()          # drop the wide feature columns to save memory
    y = df["y"].to_numpy(np.int8)
    split = df["split"].to_numpy()
    fold = df["fold"].to_numpy()
    pid = df[ID].to_numpy()
    dev = split == "dev"
    test = split == "test"
    n_folds = int(fold[dev].max()) + 1
    P["models"].mkdir(parents=True, exist_ok=True)
    P["pred"].parent.mkdir(parents=True, exist_ok=True)

    p_lr = np.full(len(df), np.nan, dtype="float64")
    p_gb = np.full(len(df), np.nan, dtype="float64")
    test_lr = np.zeros(int(test.sum()))
    test_gb = np.zeros(int(test.sum()))
    best_iter, importance = [], []
    t_all = time.time()

    for k in range(n_folds):
        t0 = time.time()
        rng = np.random.default_rng(args.seed + k)
        tr_mask = dev & (fold != k)
        va_mask = dev & (fold == k)

        # --- inner patient-level early-stopping holdout (10% of training patients) ---
        tr_pids = np.unique(pid[tr_mask])
        es_pids = rng.choice(tr_pids, size=max(1, int(0.1 * len(tr_pids))), replace=False)
        es_mask = tr_mask & np.isin(pid, es_pids)
        fit_mask = tr_mask & ~es_mask

        # --- LightGBM ---
        ti = train_idx(fit_mask, y, args.lgb_neg_frac, rng)
        ei = np.flatnonzero(es_mask)
        params = dict(objective="binary", metric="auc", learning_rate=0.05,
                      num_leaves=args.num_leaves, min_child_samples=args.min_child_samples,
                      feature_fraction=args.feature_fraction, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=args.lambda_l2, scale_pos_weight=args.pos_weight,
                      verbosity=-1, seed=args.seed + k, num_threads=args.n_jobs)
        dtr = lgb.Dataset(X[ti], y[ti], feature_name=names, free_raw_data=False)
        des = lgb.Dataset(X[ei], y[ei], reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=args.max_trees, valid_sets=[des],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        bi = booster.best_iteration or args.max_trees
        best_iter.append(int(bi))
        p_gb[va_mask] = booster.predict(X[va_mask], num_iteration=bi)
        test_gb += booster.predict(X[test], num_iteration=bi) / n_folds
        booster.save_model(str(P["models"] / f"lgbm_fold{k}{P['sfx']}.txt"), num_iteration=bi)
        importance.append(pd.Series(booster.feature_importance("gain"), index=names))

        # --- Logistic regression (median imputation + scaling fitted on its training rows) ---
        li = train_idx(tr_mask, y, args.lr_neg_frac, rng)
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(C=0.5, max_iter=300))
        lr.fit(X[li], y[li])
        p_lr[va_mask] = lr.predict_proba(X[va_mask])[:, 1]
        test_lr += lr.predict_proba(X[test])[:, 1] / n_folds
        joblib.dump(lr, P["models"] / f"logreg_fold{k}{P['sfx']}.joblib")

        m_gb, m_lr = metrics(y[va_mask], p_gb[va_mask]), metrics(y[va_mask], p_lr[va_mask])
        print(f"[fold {k}] trees={bi:3d} | LightGBM AUROC {m_gb['auroc']:.3f} AUPRC {m_gb['auprc']:.3f} | "
              f"LogReg AUROC {m_lr['auroc']:.3f} AUPRC {m_lr['auprc']:.3f} | {time.time() - t0:.0f}s")

    p_lr[test] = test_lr
    p_gb[test] = test_gb

    # ---- save predictions (meta + scores) ----
    pred = df[META].copy()
    pred["p_logreg"] = p_lr
    pred["p_lgbm"] = p_gb
    pred.to_parquet(P["pred"], index=False)
    print(f"[write] {P['pred']}")

    # ---- OOF metrics (dev only; test set is NOT evaluated here) ----
    rows = []
    for model, col in (("logreg", p_lr), ("lgbm", p_gb)):
        rows.append({"model": model, "group": "dev_all", **metrics(y[dev], col[dev])})
        for k in range(n_folds):
            mk = dev & (fold == k)
            rows.append({"model": model, "group": f"fold{k}", **metrics(y[mk], col[mk])})
        for hosp in sorted(df.loc[dev, "hospital"].astype(str).unique()):
            mh = dev & (df["hospital"].astype(str).to_numpy() == hosp)
            rows.append({"model": model, "group": f"hospital_{hosp}", **metrics(y[mh], col[mh])})
    met = pd.DataFrame(rows)
    met.to_csv(P["models"] / f"baseline_oof_metrics{P['sfx']}.csv", index=False)

    imp = pd.concat(importance, axis=1).mean(axis=1).sort_values(ascending=False)
    (imp / imp.sum()).rename("gain_share").to_csv(P["models"] / f"lgbm_feature_importance{P['sfx']}.csv")
    json.dump({"best_iterations": best_iter, "params": {k: v for k, v in params.items() if k != "seed"},
               "lgb_neg_frac": args.lgb_neg_frac, "lr_neg_frac": args.lr_neg_frac,
               "seconds": round(time.time() - t_all, 1), "quick": quick},
              open(P["models"] / f"baselines_report{P['sfx']}.json", "w"), indent=2)

    print("\n=== OUT-OF-FOLD RESULTS (dev rows only) ===")
    show = met[met["group"].isin(["dev_all", "hospital_A", "hospital_B"])].copy()
    print(show.round(4).to_string(index=False))
    print("\nTop 12 LightGBM features by share of gain:")
    print((imp / imp.sum()).head(12).round(4).to_string())


# --------------------------------------------------------------------------- #
# Stage 3: ablation
# --------------------------------------------------------------------------- #

def feature_groups(names: list[str]) -> dict[str, list[str]]:
    time_f = ["ICULOS", "HospAdmTime_slog"]
    demo = ["Age", "Gender", "unit"]
    extra = ["n_vital_obs_6h", "n_lab_obs_12h", "n_lab_hours_12h", "n_lab_obs_total", "lab_age_any"]
    meas = [n for n in names if n.endswith("_age")] + extra
    phys = [n for n in names if n.endswith(("_ffill", "_mean6", "_std6", "_delta6"))]
    groups = {"TIME": time_f, "MEAS": meas, "PHYS": phys, "DEMO": demo}
    flat = [n for g in groups.values() for n in g]
    assert len(flat) == len(set(flat)), "feature groups overlap"
    assert set(flat) == set(names), f"ungrouped features: {sorted(set(names) - set(flat))}"
    return groups


VARIANTS = {
    "full": ["TIME", "MEAS", "PHYS", "DEMO"],
    "no_time": ["MEAS", "PHYS", "DEMO"],
    "no_meas": ["TIME", "PHYS", "DEMO"],
    "phys_only": ["PHYS", "DEMO"],
    "behavior_only": ["TIME", "MEAS"],
    "time_only": ["TIME"],                     # time-in-ICU baseline: how far does time alone get?
}


def lgbm_params(args: argparse.Namespace, seed: int) -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=args.num_leaves, min_child_samples=args.min_child_samples,
                feature_fraction=args.feature_fraction, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=args.lambda_l2, scale_pos_weight=args.pos_weight,
                verbosity=-1, seed=seed, num_threads=args.n_jobs)


def lgbm_fit(X, y, names, fit_idx, es_idx, args, seed):
    """Same LightGBM recipe as the baselines stage. Returns (booster, best_iteration)."""
    import lightgbm as lgb
    dtr = lgb.Dataset(X[fit_idx], y[fit_idx], feature_name=names, free_raw_data=False)
    des = lgb.Dataset(X[es_idx], y[es_idx], reference=dtr)
    booster = lgb.train(lgbm_params(args, seed), dtr, num_boost_round=args.max_trees, valid_sets=[des],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    return booster, int(booster.best_iteration or args.max_trees)


def es_split(tr_mask, pid, rng, strat=None):
    """Inner 10% patient-level early-stopping holdout taken from the training rows.
    strat (row-level bool, e.g. septic_patient): draw 10% of each class separately, so a small
    training set (one pattern, one hospital) still has positives in the holdout. strat=None keeps
    the original unstratified draw, so baseline/ablation results stay reproducible."""
    if strat is None:
        tr_pids = np.unique(pid[tr_mask])
        es_pids = rng.choice(tr_pids, size=max(1, int(0.1 * len(tr_pids))), replace=False)
    else:
        parts = []
        for s in (False, True):
            pp = np.unique(pid[tr_mask & (strat == s)])
            if len(pp):
                parts.append(rng.choice(pp, size=max(1, int(0.1 * len(pp))), replace=False))
        es_pids = np.concatenate(parts)
    es_mask = tr_mask & np.isin(pid, es_pids)
    return tr_mask & ~es_mask, es_mask


def lgbm_cv(X, y, names, split, fold, pid, args, train_ok=None, strat=None):
    """5-fold out-of-fold LightGBM, identical recipe to the baselines stage.
    train_ok (row mask): only these rows may be used for training (pattern experts, one hospital).
    Returns (scores: dev = OOF, test = average of fold models; best iterations; mean gain)."""
    dev, test = split == "dev", split == "test"
    n_folds = int(fold[dev].max()) + 1
    scores = np.full(len(y), np.nan)
    test_sum = np.zeros(int(test.sum()))
    best_iter, imps, boosters = [], [], []
    for k in range(n_folds):
        rng = np.random.default_rng(args.seed + k)
        tr_mask, va_mask = dev & (fold != k), dev & (fold == k)
        if train_ok is not None:
            tr_mask &= train_ok
        fit_mask, es_mask = es_split(tr_mask, pid, rng, strat)
        ti = train_idx(fit_mask, y, args.lgb_neg_frac, rng)
        booster, bi = lgbm_fit(X, y, names, ti, np.flatnonzero(es_mask), args, args.seed + k)
        best_iter.append(bi)
        scores[va_mask] = booster.predict(X[va_mask], num_iteration=bi)
        test_sum += booster.predict(X[test], num_iteration=bi) / n_folds
        imps.append(pd.Series(booster.feature_importance("gain"), index=names))
        boosters.append(booster)
    scores[test] = test_sum
    lgbm_cv.last_boosters = boosters          # for callers that want to save the fold models
    return scores, best_iter, pd.concat(imps, axis=1).mean(axis=1)


def stage_ablation(args: argparse.Namespace) -> None:
    P = get_paths(args.outputs, quick=False)
    if not P["pred"].exists():
        sys.exit("ERROR: predictions.parquet not found. Run the baselines stage first.")
    names = json.load(open(P["feat_names"]))
    groups = feature_groups(names)
    json.dump({k: v for k, v in groups.items()}, open(P["models"] / "ablation_groups.json", "w"), indent=2)
    print("[groups] " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    # Only the metadata is loaded up front; each variant then reads just its own columns, one at
    # a time, so peak memory stays near one variant's matrix (important on small machines).
    meta = pd.read_parquet(P["features"], columns=META)
    pred = pd.read_parquet(P["pred"])
    assert (pred[ID].to_numpy() == meta[ID].to_numpy()).all() and \
        (pred["Hour"].to_numpy() == meta["Hour"].to_numpy()).all(), "predictions and features are misaligned"
    y = meta["y"].to_numpy(np.int8)
    split, fold, pid = meta["split"].to_numpy(), meta["fold"].to_numpy(), meta[ID].to_numpy()
    dev = split == "dev"
    hosp = meta["hospital"].astype(str).to_numpy()

    met_path = P["models"] / "ablation_oof_metrics.csv"
    imp_path = P["models"] / "ablation_feature_importance.csv"
    met_rows = [] if not met_path.exists() else pd.read_csv(met_path).to_dict("records")
    imp_rows = [] if not imp_path.exists() else pd.read_csv(imp_path).to_dict("records")

    for v in args.variants:
        t0 = time.time()
        cols = [c for g in VARIANTS[v] for c in groups[g]]
        Xv = np.empty((len(meta), len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            Xv[:, j] = pd.read_parquet(P["features"], columns=[c])[c].to_numpy(np.float32)
        scores, bi, imp = lgbm_cv(Xv, y, cols, split, fold, pid, args)
        del Xv
        gc.collect()
        col = f"p_lgbm_{v}"
        pred[col] = scores
        pred.to_parquet(P["pred"], index=False)           # save after every variant
        met_rows = [r for r in met_rows if r["variant"] != v]
        imp_rows = [r for r in imp_rows if r["variant"] != v]
        rec = {"variant": v, "n_features": len(cols), "best_iterations": str(bi)}
        for grp, mask in (("dev_all", dev), ("hospital_A", dev & (hosp == "A")), ("hospital_B", dev & (hosp == "B"))):
            m = metrics(y[mask], scores[mask])
            met_rows.append({**rec, "group": grp, **m})
        share = (imp / imp.sum()).sort_values(ascending=False).head(15)
        imp_rows += [{"variant": v, "feature": f, "gain_share": float(x)} for f, x in share.items()]
        m_all = metrics(y[dev], scores[dev])
        print(f"[{v:13s}] features={len(cols):3d} trees={bi} | OOF AUROC {m_all['auroc']:.4f} "
              f"AUPRC {m_all['auprc']:.4f} | {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(met_rows).to_csv(met_path, index=False)
        pd.DataFrame(imp_rows).to_csv(imp_path, index=False)

    if "full" in args.variants and "p_lgbm" in pred.columns:
        a, b = metrics(y[dev], pred.loc[dev, "p_lgbm_full"].to_numpy()), metrics(y[dev], pred.loc[dev, "p_lgbm"].to_numpy())
        print(f"[noise floor] same model, different column order: AUROC {a['auroc']:.4f} vs {b['auroc']:.4f} "
              f"(diff {abs(a['auroc'] - b['auroc']):.4f}), AUPRC {a['auprc']:.4f} vs {b['auprc']:.4f} "
              f"(diff {abs(a['auprc'] - b['auprc']):.4f}). Treat smaller gaps between variants as noise.")
    print(f"[write] {P['pred']}")


# --------------------------------------------------------------------------- #
# Shared helpers for the stages below
# --------------------------------------------------------------------------- #

def slug(name: str) -> str:
    """'No abnormal vitals' -> 'no_abnormal_vitals' (used in column names)."""
    return "".join(c if c.isalnum() else "_" for c in str(name).lower()).strip("_")


def load_matrix(features_path: Path, cols: list[str], patient_frac: float, seed: int):
    """Meta columns plus a float32 matrix of `cols`, read one column at a time (low peak memory)."""
    meta = pd.read_parquet(features_path, columns=META)
    keep = np.ones(len(meta), bool)
    if patient_frac < 1.0:
        sub = subsample_patients(meta, patient_frac, seed)
        keep = meta[ID].isin(sub[ID].unique()).to_numpy()
        print(f"[quick] using {int(meta.loc[keep, ID].nunique())} patients ({int(keep.sum())} rows); "
              "outputs get a '_quick' suffix")
    X = np.empty((int(keep.sum()), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        X[:, j] = pd.read_parquet(features_path, columns=[c])[c].to_numpy(np.float32)[keep]
    return meta[keep].reset_index(drop=True), X


def save_pred_columns(path: Path, meta: pd.DataFrame, new: dict, quick: bool) -> None:
    """Add score columns to predictions.parquet, refusing to write if rows are misaligned."""
    pred = None
    if path.exists():
        pred = pd.read_parquet(path)
        aligned = len(pred) == len(meta) and (pred[ID].to_numpy() == meta[ID].to_numpy()).all() and \
            (pred["Hour"].to_numpy() == meta["Hour"].to_numpy()).all()
        if not aligned:
            if not quick:
                sys.exit(f"ERROR: {path} rows do not match the feature file. Re-run the baselines stage.")
            print(f"[note] {path.name} is from a different quick subsample; starting it afresh")
            pred = None
    if pred is None:
        pred = meta[META].copy()
    for c, v in new.items():
        assert len(v) == len(pred) and not np.isnan(v).any(), f"{c}: missing scores"
        pred[c] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path, index=False)
    print(f"[write] {path}  (+ {', '.join(new)})")


def assigned_patterns(meta: pd.DataFrame) -> list[str]:
    pat = meta["pattern"].astype(str)
    return sorted(set(pat[meta["septic_patient"].to_numpy(bool)]) - {"none", "unassigned", "unavailable", "Unattributable"})


# --------------------------------------------------------------------------- #
# Stage 4: GRU sequence models (single head, multi-task, mixture of experts)
# --------------------------------------------------------------------------- #

def _seqnet(torch):
    nn = torch.nn

    class SeqNet(nn.Module):
        """GRU over the hourly feature sequence. Causal: the output at hour t sees hours <= t only.
        head = single : one sepsis logit
               mtl    : sepsis logit + auxiliary pattern classifier (classes: none + patterns)
               moe    : K expert logits mixed by a softmax gate; the gate is optionally supervised
                        with the pattern label (aux-weight > 0)"""

        def __init__(self, d_in, hidden, head, n_pat, dropout):
            super().__init__()
            self.head = head
            self.gru = nn.GRU(d_in, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            if head == "moe":
                self.experts = nn.Linear(hidden, n_pat)
                self.gate = nn.Linear(hidden, n_pat)
            else:
                self.out = nn.Linear(hidden, 1)
                self.aux = nn.Linear(hidden, n_pat + 1) if head == "mtl" else None

        def forward(self, x):
            h = self.drop(self.gru(x)[0])
            if self.head == "moe":
                g = self.gate(h)
                return (torch.softmax(g, -1) * self.experts(h)).sum(-1), g
            return self.out(h).squeeze(-1), (self.aux(h) if self.aux is not None else None)

    return SeqNet


def stage_gru(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: torch is not installed. Run: pip install torch  (CPU build is enough)")
    from sklearn.metrics import roc_auc_score
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    quick = args.patient_frac < 1.0
    P, src = get_paths(args.outputs, quick), get_paths(args.outputs, quick=False)
    names = json.load(open(src["feat_names"]))
    groups = feature_groups(names)
    drop = {c for g in args.exclude for c in groups[g]}
    cols = [n for n in names if n not in drop]
    tag = {"single": "gru", "mtl": "gru_mtl", "moe": "gru_moe"}[args.head] + \
        "".join(f"_no_{g.lower()}" for g in sorted(args.exclude))
    meta, X = load_matrix(src["features"], cols, args.patient_frac, args.seed)
    print(f"[load] {len(meta)} rows, {len(cols)} features, head={args.head}, column p_{tag}")

    y = meta["y"].to_numpy(np.float32)
    split, fold, pid = meta["split"].to_numpy(), meta["fold"].to_numpy(), meta[ID].to_numpy()
    sept = meta["septic_patient"].to_numpy(bool)
    dev, test = split == "dev", split == "test"
    n_folds = int(fold[dev].max()) + 1
    starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    lens = np.diff(np.r_[starts, len(meta)])
    p_split, p_fold = split[starts], fold[starts]

    # auxiliary targets (TRAINING TARGETS ONLY, never inputs). Pattern classes are only taught on rows
    # inside the warning horizon (y = 1) of septic patients with an assigned pattern; non-septic rows are
    # class "none" for the multi-task head; everything else is ignored (-100).
    patterns = assigned_patterns(meta)
    if args.head != "single" and len(patterns) < 2:
        sys.exit("ERROR: the mtl / moe heads need pattern_assignments.csv with >= 2 patterns "
                 "(run 04 and then the features stage again).")
    pat = meta["pattern"].astype(str).to_numpy()
    pmap = {p: i for i, p in enumerate(patterns)}
    pos_as = (y == 1) & np.isin(pat, patterns)
    aux = np.full(len(meta), -100, dtype=np.int64)
    if args.head == "mtl":
        aux[~sept] = 0
        aux[pos_as] = 1 + np.array([pmap[p] for p in pat[pos_as]])
    elif args.head == "moe":
        aux[pos_as] = np.array([pmap[p] for p in pat[pos_as]])

    ind_idx = [j for j, c in enumerate(cols) if c.endswith("_ffill")]   # "not measured recently" flags
    d_in = len(cols) + len(ind_idx)
    SeqNet = _seqnet(torch)
    P["models"].mkdir(parents=True, exist_ok=True)

    def make_batch(pats, mu, sd):
        L = lens[pats]
        T = int(L.max())
        xb = np.zeros((len(pats), T, d_in), np.float32)
        yb = np.zeros((len(pats), T), np.float32)
        ab = np.full((len(pats), T), -100, np.int64)
        mb = np.zeros((len(pats), T), bool)
        for i, p in enumerate(pats):
            s, n = starts[p], L[i]
            x = X[s:s + n]
            xb[i, :n, :len(cols)] = np.nan_to_num(np.clip((x - mu) / sd, -5, 5), nan=0.0)
            xb[i, :n, len(cols):] = np.isnan(x[:, ind_idx])
            yb[i, :n], ab[i, :n], mb[i, :n] = y[s:s + n], aux[s:s + n], True
        return torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(ab), torch.from_numpy(mb)

    def batches(pats, rng, bs, shuffle):
        """Length-bucketed batches: patients of similar length together, so little padding."""
        pats = rng.permutation(pats) if shuffle else pats
        out = []
        for i in range(0, len(pats), bs * 20):
            chunk = pats[i:i + bs * 20]
            chunk = chunk[np.argsort(lens[chunk], kind="stable")]
            out += [chunk[j:j + bs] for j in range(0, len(chunk), bs)]
        return [out[i] for i in rng.permutation(len(out))] if shuffle else out

    def predict(model, pats, mu, sd):
        p = np.full(len(meta), np.nan)
        model.eval()
        with torch.no_grad():
            for b in batches(np.asarray(pats), np.random.default_rng(0), 256, shuffle=False):
                xb, _, _, mb = make_batch(b, mu, sd)
                prob = torch.sigmoid(model(xb)[0]).numpy()
                for i, q in enumerate(b):
                    p[starts[q]:starts[q] + lens[q]] = prob[i, :lens[q]]
        return p

    scores = np.full(len(meta), np.nan)
    test_sum = np.zeros(len(meta))
    test_pats = np.flatnonzero(p_split == "test")
    log = []
    t_all = time.time()
    for k in range(n_folds):
        t0 = time.time()
        rng = np.random.default_rng(args.seed + k)
        torch.manual_seed(args.seed + k)
        tr_mask = dev & (fold != k)
        fit_mask, es_mask = es_split(tr_mask, pid, rng, strat=sept)   # stratified: holdout always has positives
        fit_pats = np.flatnonzero(fit_mask[starts])
        es_pats = np.flatnonzero(es_mask[starts])
        va_pats = np.flatnonzero((p_split == "dev") & (p_fold == k))

        # normalisation fitted on the fit rows only
        mu = np.nanmean(X[fit_mask], axis=0)
        sd = np.nanstd(X[fit_mask], axis=0)
        mu, sd = np.nan_to_num(mu), np.where(np.nan_to_num(sd) > 0, np.nan_to_num(sd), 1.0)
        mu, sd = mu.astype(np.float32), sd.astype(np.float32)

        model = SeqNet(d_in, args.hidden, args.head, len(patterns), args.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
        ce = None
        if args.head != "single":
            a_fit = aux[fit_mask]
            cnt = np.bincount(a_fit[a_fit >= 0], minlength=len(patterns) + (args.head == "mtl")).astype(float)
            w = np.where(cnt > 0, cnt.sum() / (len(cnt) * np.maximum(cnt, 1)), 0.0)   # balanced classes
            ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(np.clip(w, 0, 50), dtype=torch.float32))
        best, best_state, wait = -np.inf, None, 0
        for ep in range(args.epochs):
            model.train()
            tot, nb = 0.0, 0
            for b in batches(fit_pats, rng, args.batch, shuffle=True):
                xb, yb, ab, mb = make_batch(b, mu, sd)
                logit, auxo = model(xb)
                loss = bce(logit[mb], yb[mb])
                if auxo is not None and args.aux_weight > 0:
                    am = ab[mb]
                    ok = am >= 0
                    if ok.any():
                        loss = loss + args.aux_weight * ce(auxo[mb][ok], am[ok])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot, nb = tot + float(loss), nb + 1
            pe = predict(model, es_pats, mu, sd)
            auc = roc_auc_score(y[es_mask], pe[es_mask])
            log.append({"fold": k, "epoch": ep, "train_loss": tot / max(nb, 1), "es_auroc": auc})
            improved = auc > best + 1e-4
            if improved:
                best, wait = auc, 0
                best_state = {n: t.clone() for n, t in model.state_dict().items()}
            else:
                wait += 1
            print(f"  [fold {k} epoch {ep:2d}] loss {tot / max(nb, 1):.4f} | early-stop AUROC {auc:.4f}"
                  f"{' *' if improved else ''}", flush=True)
            if wait >= args.patience:
                break
        model.load_state_dict(best_state)
        if k == 0:   # causality self-check: cutting off later hours must not change earlier scores
            q = np.array([es_pats[np.argmax(lens[es_pats])]])
            xb = make_batch(q, mu, sd)[0]
            half = max(1, int(lens[q[0]]) // 2)
            model.eval()
            with torch.no_grad():
                a = torch.sigmoid(model(xb)[0])[0, :half]
                b = torch.sigmoid(model(xb[:, :half])[0])[0]
            assert torch.allclose(a, b, atol=1e-5), "GRU output depends on future hours"
            print(f"[validate] causality check passed (patient with {int(lens[q[0]])} h)")
        va_mask = dev & (fold == k)
        scores[va_mask] = predict(model, va_pats, mu, sd)[va_mask]
        test_sum[test] += predict(model, test_pats, mu, sd)[test] / n_folds
        torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "cols": cols, "ind_idx": ind_idx,
                    "patterns": patterns, "config": {k2: v for k2, v in vars(args).items() if k2 != "outputs"}},
                   P["models"] / f"{tag}_fold{k}{P['sfx']}.pt")
        m = metrics(y[dev & (fold == k)].astype(np.int8), scores[dev & (fold == k)])
        print(f"[fold {k}] best early-stop AUROC {best:.4f} | OOF AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)
    scores[test] = test_sum[test]

    save_pred_columns(P["pred"], meta, {f"p_{tag}": scores}, quick)
    pd.DataFrame(log).to_csv(P["models"] / f"{tag}_training_log{P['sfx']}.csv", index=False)
    m = metrics(y[dev].astype(np.int8), scores[dev])
    rep = {"column": f"p_{tag}", "head": args.head, "excluded_groups": args.exclude, "patterns": patterns,
           "oof_dev": m, "seconds": round(time.time() - t_all, 1),
           "config": {k2: v for k2, v in vars(args).items() if k2 not in ("outputs", "func")}}
    json.dump(rep, open(P["models"] / f"{tag}_report{P['sfx']}.json", "w"), indent=2, default=str)
    print(f"\n=== p_{tag}: OOF dev AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} ({rep['seconds']:.0f}s) ===")


# --------------------------------------------------------------------------- #
# Stage 5: per-pattern LightGBM experts (oracle upper bound) + max-gate mixture
# --------------------------------------------------------------------------- #

def stage_pattern(args: argparse.Namespace) -> None:
    P = get_paths(args.outputs, quick=False)
    names = json.load(open(P["feat_names"]))
    meta, X = load_matrix(P["features"], names, 1.0, args.seed)
    prefix, moe_col, info_file = "p_lgbm_expert_", "p_lgbm_moe_max", "pattern_experts.json"
    if args.labels == "organ":
        # organ groups from organ_pathways.py replace the vital-sign patterns as the expert labels
        apath = args.outputs / "organs" / "organ_assignments.csv"
        if not apath.exists():
            sys.exit("ERROR: organs/organ_assignments.csv not found. Run backend/organ_pathways.py first.")
        og = pd.read_csv(apath).set_index(ID)["organ_group"]
        meta["pattern"] = np.where(meta["septic_patient"].to_numpy(bool),
                                   meta[ID].map(og).fillna("Unattributable").to_numpy(), "none")
        prefix, moe_col, info_file = "p_lgbm_organ_expert_", "p_lgbm_organ_moe_max", "organ_experts.json"
    patterns = assigned_patterns(meta)
    if not patterns:
        sys.exit("ERROR: no group labels found. Run 04 (patterns) or organ_pathways.py (organs) first.")
    y = meta["y"].to_numpy(np.int8)
    split, fold, pid = meta["split"].to_numpy(), meta["fold"].to_numpy(), meta[ID].to_numpy()
    sept = meta["septic_patient"].to_numpy(bool)
    pat = meta["pattern"].astype(str).to_numpy()
    dev = split == "dev"

    new, info = {}, {}
    for g in patterns:
        n_sep = int(meta.loc[dev & sept & (pat == g), ID].nunique())
        if n_sep < args.min_patients:
            print(f"[skip] {g}: only {n_sep} dev septic patients (< --min-patients {args.min_patients})")
            info[g] = {"skipped": True, "n_dev_septic": n_sep}
            continue
        t0 = time.time()
        # training rows: every non-septic patient + septic patients of THIS pattern. The pattern is only a
        # selection rule for training rows (known in hindsight); it is never a model input.
        ok = ~sept | (pat == g)
        sc, bi, _ = lgbm_cv(X, y, names, split, fold, pid, args, train_ok=ok, strat=sept)
        col = f"{prefix}{slug(g)}"
        new[col] = sc
        for k, b in enumerate(lgbm_cv.last_boosters):
            b.save_model(str(P["models"] / f"{prefix[2:]}{slug(g)}_fold{k}.txt"), num_iteration=bi[k])
        own = dev & ((pat == g) | ~sept)
        m = metrics(y[own], sc[own])
        info[g] = {"column": col, "n_dev_septic": n_sep, "best_iterations": bi,
                   "oof_auroc_own_pattern_vs_nonseptic": m["auroc"], "oof_auprc_own_pattern_vs_nonseptic": m["auprc"]}
        print(f"[expert {g:18s}] septic dev patients={n_sep:5d} trees={bi} | OOF (own pattern vs non-septic) "
              f"AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} | {time.time() - t0:.0f}s", flush=True)
    if len(new) >= 2:
        # shortcut: a plain max over experts (no learned gate, no recalibration). Experts trained on more
        # positives score higher, so the max leans to the large patterns; the learned gate is p_gru_moe.
        new[moe_col] = np.max(np.column_stack(list(new.values())), axis=1)
    save_pred_columns(P["pred"], meta, new, quick=False)
    json.dump({"labels": args.labels, "patterns": info, "moe_max": moe_col in new}, open(P["models"] / info_file, "w"),
              indent=2, default=str)


# --------------------------------------------------------------------------- #
# Stage 6: hospital transfer
# --------------------------------------------------------------------------- #

def stage_transfer(args: argparse.Namespace) -> None:
    P = get_paths(args.outputs, quick=False)
    names = json.load(open(P["feat_names"]))
    cols = names if args.variant == "all" else [c for g in VARIANTS[args.variant] for c in feature_groups(names)[g]]
    v = "" if args.variant == "all" else f"_{args.variant}"
    meta, X = load_matrix(P["features"], cols, 1.0, args.seed)
    y = meta["y"].to_numpy(np.int8)
    split, fold, pid = meta["split"].to_numpy(), meta["fold"].to_numpy(), meta[ID].to_numpy()
    sept = meta["septic_patient"].to_numpy(bool)
    hosp = meta["hospital"].astype(str).to_numpy()
    dev = split == "dev"
    hs = sorted(set(hosp))
    if len(hs) != 2:
        sys.exit(f"ERROR: transfer needs exactly two hospitals, found {hs}")

    within = np.full(len(meta), np.nan)
    transfer = np.full(len(meta), np.nan)
    rep = {"variant": args.variant, "n_features": len(cols)}
    for h in hs:
        # within-hospital reference: same folds, trained on this hospital's rows only
        t0 = time.time()
        sc, bi, _ = lgbm_cv(X, y, cols, split, fold, pid, args, train_ok=(hosp == h), strat=sept)
        within[hosp == h] = sc[hosp == h]
        m = metrics(y[dev & (hosp == h)], within[dev & (hosp == h)])
        rep[f"within_{h}"] = {"best_iterations": bi, **m}
        print(f"[within {h}] trees={bi} | OOF AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} | {time.time() - t0:.0f}s",
              flush=True)
    for i, (s, t) in enumerate(((hs[0], hs[1]), (hs[1], hs[0]))):
        # one model on ALL dev patients of the source hospital, applied to every target-hospital row
        t0 = time.time()
        rng = np.random.default_rng(args.seed + 100 + i)
        fit_mask, es_mask = es_split(dev & (hosp == s), pid, rng, strat=sept)
        ti = train_idx(fit_mask, y, args.lgb_neg_frac, rng)
        booster, bi = lgbm_fit(X, y, cols, ti, np.flatnonzero(es_mask), args, args.seed + 100 + i)
        tgt = hosp == t
        transfer[tgt] = booster.predict(X[tgt], num_iteration=bi)
        booster.save_model(str(P["models"] / f"lgbm_transfer_{s}to{t}{v}.txt"), num_iteration=bi)
        m = metrics(y[dev & tgt], transfer[dev & tgt])
        rep[f"transfer_{s}_to_{t}"] = {"best_iterations": bi, **m}
        print(f"[transfer {s}->{t}] trees={bi} | target dev AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} "
              f"(within-{t} reference AUROC {rep[f'within_{t}']['auroc']:.4f}) | {time.time() - t0:.0f}s", flush=True)
    save_pred_columns(P["pred"], meta, {f"p_lgbm_within{v}": within, f"p_lgbm_transfer{v}": transfer}, quick=False)
    json.dump(rep, open(P["models"] / f"transfer_report{v}.json", "w"), indent=2, default=str)
    print("Alert-level transfer results (thresholds from the source hospital): python 06_evaluate.py transfer ...")


# --------------------------------------------------------------------------- #
# Stage 7: integrated ensemble of the three main models
# --------------------------------------------------------------------------- #

ENSEMBLE_MEMBERS = ["p_lgbm", "p_logreg", "p_gru"]
ENSEMBLE_GRID = 1001


def ensemble_reference(p_dev: np.ndarray) -> np.ndarray:
    """Quantiles of a model's dev out-of-fold scores (monotone), used to turn any score into a percentile."""
    return np.maximum.accumulate(np.quantile(p_dev, np.linspace(0, 1, ENSEMBLE_GRID)))


def to_percentile(p: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.interp(p, ref, np.linspace(0, 1, len(ref)))


def ensemble_score(scores: dict, refs: dict) -> np.ndarray:
    """Mean percentile of the members. Percentiles put the models on one scale: the GRU's raw probabilities run high
    (it is trained with extra weight on sepsis hours), so a plain average of probabilities would be dominated by it."""
    return np.mean([to_percentile(np.asarray(scores[m], float), np.asarray(refs[m], float)) for m in refs], axis=0)


def stage_ensemble(args: argparse.Namespace) -> None:
    P = get_paths(args.outputs, quick=False)
    pred = pd.read_parquet(P["pred"])
    members = [m for m in ENSEMBLE_MEMBERS if m in pred.columns]
    if len(members) < 2:
        sys.exit(f"ERROR: the ensemble needs at least two of {ENSEMBLE_MEMBERS}; found {members}.")
    if len(members) < len(ENSEMBLE_MEMBERS):
        print(f"[note] ensemble built from {members} only (missing: {sorted(set(ENSEMBLE_MEMBERS) - set(members))})")
    dev = (pred["split"] == "dev").to_numpy()
    refs = {m: ensemble_reference(pred.loc[dev, m].to_numpy(float)) for m in members}
    ens = ensemble_score({m: pred[m].to_numpy(float) for m in members}, refs)
    assert np.isfinite(ens).all() and ens.min() >= 0 and ens.max() <= 1
    save_pred_columns(P["pred"], pred[META], {"p_ensemble": ens}, quick=False)
    json.dump({"members": members, "method": "mean of dev-percentiles", "grid": ENSEMBLE_GRID,
               "reference": {m: r.tolist() for m, r in refs.items()}},
              open(P["models"] / "ensemble_reference.json", "w"))
    y = pred["y"].to_numpy(np.int8)
    for m in members + ["p_ensemble"]:
        s = ens if m == "p_ensemble" else pred[m].to_numpy()
        r = metrics(y[dev], s[dev])
        print(f"  {m:12s} OOF dev AUROC {r['auroc']:.4f} AUPRC {r['auprc']:.4f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_lgb_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=0, help="LightGBM threads (0 = all)")
    p.add_argument("--max-trees", type=int, default=600)
    p.add_argument("--pos-weight", type=float, default=1.0, help="LightGBM scale_pos_weight (default 1 = none)")
    p.add_argument("--num-leaves", type=int, default=7)
    p.add_argument("--min-child-samples", type=int, default=5000,
                   help="large on purpose: one patient contributes up to ~300 similar hourly rows")
    p.add_argument("--feature-fraction", type=float, default=0.5)
    p.add_argument("--lambda-l2", type=float, default=50.0)
    p.add_argument("--lgb-neg-frac", type=float, default=1.0, help="fraction of negative rows used to train LightGBM")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    f = sub.add_parser("features", help="build the hourly feature matrix")
    f.add_argument("--outputs", type=Path, required=True, help="the outputs/ folder produced by steps 01-04")
    f.add_argument("--horizon", type=int, default=12, help="warning horizon H in hours (default 12)")

    b = sub.add_parser("baselines", help="logistic regression + LightGBM, OOF predictions")
    b.add_argument("--outputs", type=Path, required=True)
    add_lgb_args(b)
    b.add_argument("--lr-neg-frac", type=float, default=0.15, help="fraction of negative rows used to train logistic regression")
    b.add_argument("--patient-frac", type=float, default=1.0, help="quick test: use only this fraction of patients")

    a = sub.add_parser("ablation", help="LightGBM with feature groups removed (adds p_lgbm_<variant> columns)")
    a.add_argument("--outputs", type=Path, required=True)
    add_lgb_args(a)
    a.add_argument("--variants", nargs="+", choices=list(VARIANTS),
                   default=["no_time", "no_meas", "phys_only", "behavior_only", "time_only"])

    g = sub.add_parser("gru", help="GRU sequence model (single / multi-task / mixture-of-experts head)")
    g.add_argument("--outputs", type=Path, required=True)
    g.add_argument("--head", choices=["single", "mtl", "moe"], default="single",
                   help="single = p_gru, mtl = p_gru_mtl (pattern as auxiliary target), moe = p_gru_moe")
    g.add_argument("--exclude", nargs="*", default=[], choices=["TIME", "MEAS", "PHYS", "DEMO"],
                   help="drop feature groups, e.g. --exclude TIME (column gets a _no_time suffix)")
    g.add_argument("--hidden", type=int, default=64)
    g.add_argument("--dropout", type=float, default=0.2)
    g.add_argument("--epochs", type=int, default=12)
    g.add_argument("--patience", type=int, default=3, help="epochs without early-stop AUROC gain before stopping")
    g.add_argument("--batch", type=int, default=64, help="patients per batch")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--pos-weight", type=float, default=5.0, help="BCE weight of positive hours")
    g.add_argument("--aux-weight", type=float, default=0.3, help="weight of the pattern loss (mtl / moe)")
    g.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--patient-frac", type=float, default=1.0, help="quick test: use only this fraction of patients")

    pp = sub.add_parser("pattern", help="per-pattern LightGBM experts (oracle) + max-gate mixture")
    pp.add_argument("--outputs", type=Path, required=True)
    add_lgb_args(pp)
    pp.add_argument("--min-patients", type=int, default=30, help="skip patterns with fewer dev septic patients")
    pp.add_argument("--labels", choices=["pattern", "organ"], default="pattern",
                    help="pattern = vital-sign groups (p_lgbm_expert_*); organ = organ groups (p_lgbm_organ_expert_*)")

    t = sub.add_parser("transfer", help="train on one hospital, test on the other (and within-hospital reference)")
    t.add_argument("--outputs", type=Path, required=True)
    add_lgb_args(t)
    t.add_argument("--variant", choices=["all"] + list(VARIANTS), default="all",
                   help="feature set (default all features; e.g. no_time adds a _no_time suffix)")

    e = sub.add_parser("ensemble", help="integrated model: mean percentile of LightGBM, logistic regression and GRU")
    e.add_argument("--outputs", type=Path, required=True)

    args = ap.parse_args()
    {"ensemble": stage_ensemble, "features": stage_features, "baselines": stage_baselines, "ablation": stage_ablation, "gru": stage_gru,
     "pattern": stage_pattern, "transfer": stage_transfer}[args.stage](args)


if __name__ == "__main__":
    main()