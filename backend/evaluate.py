#!/usr/bin/env python3
"""
06_evaluate.py
==============
Idea B: deterioration-pattern-specific sepsis early warning (PhysioNet/CinC 2019).

Turns the hourly risk scores in predictions.parquet (any column named p_*) into alert-level
results: how many septic patients are warned, how early, at what false-alarm burden, overall,
per hospital and per clinical pattern group. Model-agnostic: new p_* columns added by later
stages (GRU, pattern models) are picked up automatically.

    python 06_evaluate.py performance --outputs outputs                       # dev (out-of-fold)
    python 06_evaluate.py performance --outputs outputs --split test --final  # ONCE, at the end

    python 06_evaluate.py timeprofile --outputs outputs                       # dev only

    python 06_evaluate.py timecontrol --outputs outputs   # time-in-ICU control (+ *_tadj columns)
    python 06_evaluate.py transfer    --outputs outputs   # hospital transfer, thresholds from source
    python 06_evaluate.py frequency   --outputs outputs   # measurement-frequency control (dev)
    python 06_evaluate.py shap        --outputs outputs   # SHAP by pattern group (dev)
    python 06_evaluate.py compare     --outputs outputs   # global vs pattern-aware models, paired
    python 06_evaluate.py plots       --outputs outputs   # data for the Result figures tab (dev)
    python 06_evaluate.py organs      --outputs outputs   # results per organ group (needs organ_pathways.py)

Recommended order: all 05 stages -> timecontrol -> performance -> timeprofile -> transfer ->
frequency -> shap -> compare. At the very end, once: performance / transfer / compare with
--split test --final.

timecontrol: (1) row AUROC/AUPRC within hour-of-stay bins and an hour-stratified AUROC (only pairs of
hours at the same hour of stay are compared, so a pure time score gets ~0.5); writes
within_hour_discrimination.csv. (2) adds p_<model>_tadj = percentile of the score among dev non-septic
hours at the same hour of stay (single hours 0..71, then 72-95, 96-143, 144+). A threshold on a
_tadj score flags the same share of non-septic hours at every hour of stay. Re-run `performance`.
transfer: target-hospital metrics for within-hospital, transfer at the SOURCE threshold (honest), and
transfer at a threshold re-set on target dev (recalibration reference). transfer_metrics.csv.
frequency: per pattern, warning rate vs how often the trigger vitals / labs were measured; warning rate
by measurement-intensity quartile; logistic regression of "warned in time" on pattern, unadjusted and
adjusted for measurement rates, switch-on hour and hospital (ridge C=1, bootstrap CIs).
shap: TreeSHAP from LightGBM (pred_contrib; no shap package), each row explained by the fold model that
did not train on it. shap_by_pattern.csv.
compare: each model at its own dev threshold; per pattern, % warned in time and the PAIRED difference
vs p_lgbm (patient bootstrap); oracle = own-pattern expert. pattern_model_comparison.csv.
timeprofile now also writes nonseptic_far_by_stay: false alarms among non-septic patients whose stay
length falls in the same bins as the septic switch-on hour (results within stay-length strata).

timeprofile: does the model warn EARLY or only LATE? Splits the dev results by when the label
switches on (septic patients: share warned in the last 12 h, with Wilson 95% CIs) and by hour of
stay (non-septic patients: share of patient-hours above the threshold). Writes time_profile.csv.
Why it matters: a model whose risk simply rises with time in ICU looks good on the headline
numbers, because most patient-days are short stays while long stays are almost always flagged.

Definitions
-----------
Cohort        patients with include_main (39,910). Rows = pre-onset hours only.
F             first hour where SepsisLabel == 1 ("label switch-on"; clinical onset is F + 6).
Alert         the model alerts at hour t when p(t) >= tau. Alert EPISODES: a new alert is counted
              only if no hour in the previous --refractory hours (default 12) was above tau.
False alarms  alert episodes in NON-septic patients per 100 patient-days. (Alerts in septic
              patients are not counted as false alarms, so this is a simple, slightly lenient
              burden measure.)
Operating pts tau is chosen on the dev out-of-fold scores, separately for each model:
                far2 / far5 / far10   largest alert rate with <= 2 / 5 / 10 false-alarm episodes
                                      per 100 non-septic patient-days
                util_opt              tau maximising the normalised utility below
              For --split test the SAME dev thresholds are reused (saved in thresholds.json).
sens_any      share of septic patients with any alert before F
sens_timely   share with an alert within the --horizon hours before F (default 12, the
              training horizon); an alert earlier than that does not count here
lead          hours between the FIRST alert and F, median over alerted patients
              (lead_any) or over timely-alerted patients (lead_timely, first alert in window).
              CAUTION: lead_timely is measured from the first above-threshold hour INSIDE the
              window, so it equals the horizon (12) whenever the alert was already on at the
              window start; its median is therefore ~12 by construction. lead_any is the more
              informative number but includes alerts that fired many hours early (for example
              because risk rises with time in ICU), so read it together with sens_timely.
CIs           patient-level bootstrap (default 200 resamples) with tau FIXED; septic and
              non-septic patients are resampled separately. Pattern rows have no FAR (non-septic
              patients have no pattern); they use the same tau as the overall row.

Utility (ADAPTED PhysioNet 2019 score)
--------------------------------------
Uses the official constants and formulas (dt_early -12, dt_optimal -6, dt_late +3,
max_u_tp 1, min_u_fn -2, u_fp -0.05; t_sepsis = F + 6), normalised as
(observed - inaction) / (best - inaction). Adaptation: this project scores only hours before
F, but the official score also covers hours F .. F+9. For those hours the prediction is
assumed to repeat the last pre-F prediction. Hours are truncated at the end of the stay,
as in the official code. The cohort also differs (hour-0 positives are excluded). So the
number is NOT comparable with Challenge leaderboard scores. The implementation is checked
against the official compute_prediction_utility function in the validation step
(--official-check points to the official evaluate_sepsis_score.py, optional).

IMPORTANT caution on pattern results
------------------------------------
The pattern groups (04_pattern_clustering.py) are defined from the vital signs in the last
hours before F, the same signals the models see. "No abnormal vitals" is hard to detect BY
CONSTRUCTION. Differences between groups describe how well the model warns patients who
look a certain way; they do not show the model is better or worse for a disease subtype.

Outputs (under --outputs/evaluation)
------------------------------------
    thresholds.json                 dev thresholds per model and operating point
    alert_metrics{_test}.csv        main results table
    leadtime_curve{_test}.csv       share of septic patients alerted at least h hours before F
    utility_curve{_test}.csv        normalised utility vs threshold (per model)
    time_profile.csv, within_hour_discrimination.csv, transfer_metrics{_test}.csv,
    frequency_by_pattern.csv, frequency_by_quartile.csv, frequency_adjusted.csv, frequency_report.json,
    shap_by_pattern.csv, pattern_model_comparison{_test}.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ID = "Patient_ID"
BIG = 10 ** 6
FAR_TARGETS = (2, 5, 10)

# official utility constants
DT_EARLY, DT_OPT, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP = 1.0, -2.0, -0.05
M1 = MAX_U_TP / (DT_OPT - DT_EARLY)
B1 = -M1 * DT_EARLY
M2 = -MAX_U_TP / (DT_LATE - DT_OPT)
B2 = -M2 * DT_LATE
M3 = MIN_U_FN / (DT_LATE - DT_OPT)
B3 = -M3 * DT_OPT
POST_H = DT_LATE - DT_OPT + 1          # hours F .. F+9 -> 10 hours


def _post_tables() -> tuple[np.ndarray, np.ndarray]:
    """Cumulative utility over hours F+j (j = 0..9), rel = t - t_sepsis = j - 6."""
    u1, u0 = [], []
    for j in range(POST_H):
        rel = j - 6
        if rel <= DT_OPT:
            u1.append(max(M1 * rel + B1, U_FP))
            u0.append(0.0)
        else:
            u1.append(M2 * rel + B2)
            u0.append(M3 * rel + B3)
    return np.concatenate([[0], np.cumsum(u1)]), np.concatenate([[0], np.cumsum(u0)])


POST_A1, POST_A0 = _post_tables()      # index k = number of available post-F hours (0..10)


# --------------------------------------------------------------------------- #
# Cohort with vectorised per-threshold computations
# --------------------------------------------------------------------------- #

class Cohort:
    def __init__(self, df: pd.DataFrame, patients: pd.DataFrame, models: list[str], horizon: int, refractory: int):
        df = df.reset_index(drop=True)
        pid = df[ID].to_numpy()
        self.start = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
        self.m = len(self.start)
        counts = np.diff(np.r_[self.start, len(df)])
        self.pid_row = np.repeat(np.arange(self.m), counts)
        self.pos = np.arange(len(df)) - self.start[self.pid_row]
        assert (self.pos == df["Hour"].to_numpy()).all(), "hours must be consecutive from 0 within each patient"
        self.hour = df["Hour"].to_numpy().astype(np.int64)
        self.horizon, self.refractory = horizon, refractory

        first = df.groupby(ID, sort=False).first()
        self.pids = first.index.to_numpy()
        self.septic = first["septic_patient"].to_numpy().astype(bool)
        self.hospital = first["hospital"].astype(str).to_numpy()
        self.pattern = first["pattern"].astype(str).to_numpy()
        self.n_pre = counts
        nh = patients.set_index(ID)["n_hours"].reindex(self.pids).to_numpy()
        assert not np.isnan(nh).any(), "patients.csv does not cover every evaluated patient"
        self.n_hours = nh.astype(np.int64)
        F = (df["Hour"] + df["hours_to_onset"]).groupby(df[ID], sort=False).first().to_numpy()
        self.F = np.where(self.septic, F, np.nan)
        self.F_int = np.where(self.septic, np.nan_to_num(F, nan=0), 0).astype(np.int64)
        assert (self.n_pre[self.septic] == self.F_int[self.septic]).all(), "septic rows must be exactly hours 0..F-1"
        self.F_row = self.F_int[self.pid_row]
        self.sept_row = self.septic[self.pid_row]
        self.last_row = self.start + counts - 1
        self.scores = {mname: df[mname].to_numpy() for mname in models}

        # utilities that do not depend on the model
        k = np.clip(np.minimum(POST_H, self.n_hours - self.F_int), 0, POST_H)
        self.k_post = np.where(self.septic, k, 0)
        self.U_inact = np.where(self.septic, POST_A0[self.k_post], 0.0)
        hrs = np.arange(len(df))
        rel = self.hour - self.F_row - 6
        best_rows = self.sept_row & (self.hour >= self.F_row - 6)
        pre_best = np.where(best_rows, np.maximum(M1 * rel + B1, U_FP), 0.0)
        self.U_best = np.where(self.septic, np.add.reduceat(pre_best, self.start) + POST_A1[self.k_post], 0.0)
        self.rel_row = rel

    # ---- one threshold --------------------------------------------------- #
    def at(self, p: np.ndarray, tau: float) -> dict:
        above = p >= tau
        a = above.astype(np.int32)
        # alert episodes
        cs = np.cumsum(a)
        base = (cs - a)[self.start]
        cs_p = cs - base[self.pid_row]
        lag = np.zeros(len(a), dtype=np.int64)
        R = self.refractory
        msk = self.pos >= R + 1
        idx = np.flatnonzero(msk)
        lag[idx] = cs_p[idx - R - 1]
        prev = (cs_p - a) - lag
        alert = above & (prev == 0)
        n_alerts = np.add.reduceat(alert.astype(np.int64), self.start)
        n_above = np.add.reduceat(a.astype(np.int64), self.start)
        # first alert and timely first alert
        hh = np.where(above, self.hour, BIG)
        first = np.minimum.reduceat(hh, self.start)
        hw = np.where(above & self.sept_row & (self.hour >= self.F_row - self.horizon), self.hour, BIG)
        first_w = np.minimum.reduceat(hw, self.start)
        det_any = self.septic & (first < BIG)
        det_timely = self.septic & (first_w < BIG)
        lead_any = np.where(det_any, self.F_int - first, np.nan)
        lead_timely = np.where(det_timely, self.F_int - first_w, np.nan)
        # utility (observed)
        pre_u = np.where(above & self.sept_row, np.maximum(M1 * self.rel_row + B1, U_FP), 0.0)
        pre_sum = np.add.reduceat(pre_u, self.start)
        carry = above[self.last_row]
        post = np.where(carry, POST_A1[self.k_post], POST_A0[self.k_post])
        U_obs = np.where(self.septic, pre_sum + post, U_FP * n_above)
        return dict(n_alerts=n_alerts, n_above=n_above, det_any=det_any, det_timely=det_timely,
                    lead_any=lead_any, lead_timely=lead_timely, U_obs=U_obs, first=first)

    def far(self, res: dict, mask: np.ndarray | None = None) -> float:
        ns = ~self.septic if mask is None else (~self.septic & mask)
        days = self.n_pre[ns].sum() / 24.0
        return float(res["n_alerts"][ns].sum() / days * 100) if days > 0 else np.nan

    def util_norm(self, res: dict, mask: np.ndarray | None = None) -> float:
        sel = np.ones(self.m, bool) if mask is None else mask
        den = self.U_best[sel].sum() - self.U_inact[sel].sum()
        return float((res["U_obs"][sel].sum() - self.U_inact[sel].sum()) / den) if den else np.nan


# --------------------------------------------------------------------------- #
# Threshold search
# --------------------------------------------------------------------------- #

def threshold_grid(p: np.ndarray, septic_row: np.ndarray, n: int = 300) -> np.ndarray:
    q = 1 - np.logspace(-4.3, -0.5, n)
    return np.unique(np.quantile(p, q))


def curves_and_thresholds(co: Cohort, name: str, grid_n: int) -> tuple[pd.DataFrame, dict]:
    p = co.scores[name]
    grid = threshold_grid(p, co.sept_row, grid_n)
    rows = []
    for tau in grid:
        r = co.at(p, tau)
        rows.append({"model": name, "tau": float(tau), "far": co.far(r), "util_norm": co.util_norm(r),
                     "sens_any": float(r["det_any"][co.septic].mean()),
                     "sens_timely": float(r["det_timely"][co.septic].mean())})
    cur = pd.DataFrame(rows)
    th = {}
    for t in FAR_TARGETS:
        ok = cur[cur["far"] <= t]
        th[f"far{t}"] = float(ok["tau"].min()) if len(ok) else float(cur["tau"].max())
    th["util_opt"] = float(cur.loc[cur["util_norm"].idxmax(), "tau"])
    return cur, th


# --------------------------------------------------------------------------- #
# Metrics with bootstrap CIs
# --------------------------------------------------------------------------- #

def ci(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)


def group_metrics(co: Cohort, res: dict, mask: np.ndarray, B: int, rng: np.random.Generator,
                  with_far: bool, with_util: bool) -> dict:
    sep_idx = np.flatnonzero(mask & co.septic)
    non_idx = np.flatnonzero(mask & ~co.septic)
    out = {"n_septic": int(len(sep_idx)), "n_nonseptic": int(len(non_idx))}
    if len(sep_idx) == 0:
        return out

    def stats(si, ni):
        d = {
            "sens_any": res["det_any"][si].mean(),
            "sens_timely": res["det_timely"][si].mean(),
            "median_lead_any": np.nanmedian(res["lead_any"][si]) if res["det_any"][si].any() else np.nan,
            "median_lead_timely": np.nanmedian(res["lead_timely"][si]) if res["det_timely"][si].any() else np.nan,
        }
        if with_far and len(ni):
            d["far_per_100pd"] = res["n_alerts"][ni].sum() / (co.n_pre[ni].sum() / 24.0) * 100
            d["share_nonseptic_alerted"] = (res["n_alerts"][ni] > 0).mean()
        if with_util and len(ni):
            allx = np.concatenate([si, ni])
            den = co.U_best[allx].sum() - co.U_inact[allx].sum()
            d["util_norm"] = (res["U_obs"][allx].sum() - co.U_inact[allx].sum()) / den if den else np.nan
        return d

    point = stats(sep_idx, non_idx)
    boots = []
    for _ in range(B):
        si = rng.choice(sep_idx, size=len(sep_idx), replace=True)
        ni = rng.choice(non_idx, size=len(non_idx), replace=True) if len(non_idx) else non_idx
        boots.append(stats(si, ni))
    bdf = pd.DataFrame(boots)
    for k, v in point.items():
        lo, hi = ci(bdf[k].to_numpy(dtype=float))
        out[k], out[f"{k}_lo"], out[f"{k}_hi"] = float(v), lo, hi
    return out


def evaluate(co: Cohort, models: list[str], thresholds: dict, B: int, seed: int, horizon_max: int = 72):
    rng = np.random.default_rng(seed)
    rows, lead_rows = [], []
    groups = [("all", "all", np.ones(co.m, bool))]
    groups += [("hospital", h, co.hospital == h) for h in sorted(set(co.hospital))]
    pats = sorted({p for p in co.pattern[co.septic]})
    groups += [("pattern", p, (co.pattern == p) & co.septic) for p in pats]

    for name in models:
        p = co.scores[name]
        for op, tau in thresholds[name].items():
            res = co.at(p, tau)
            for gtype, g, mask in groups:
                gm = group_metrics(co, res, mask, B, rng, with_far=(gtype != "pattern"),
                                   with_util=(gtype != "pattern"))
                rows.append({"model": name, "operating_point": op, "tau": tau,
                             "group_type": gtype, "group": g, **gm})
                if op in ("far5", "util_opt") and gtype in ("all", "hospital", "pattern"):
                    si = np.flatnonzero(mask & co.septic)
                    if len(si):
                        lead = res["lead_any"][si]
                        for h in range(1, horizon_max + 1):
                            lead_rows.append({"model": name, "operating_point": op, "group_type": gtype,
                                              "group": g, "hours_before_switch_on": h,
                                              "share_alerted_at_least_h_before": float(np.nansum(lead >= h) / len(si)),
                                              "n_septic": int(len(si))})
    return pd.DataFrame(rows), pd.DataFrame(lead_rows)


# --------------------------------------------------------------------------- #
# Validation against the official utility function
# --------------------------------------------------------------------------- #

def official_check(co: Cohort, name: str, tau: float, hourly_path: Path, official_path: Path, n: int = 300) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("official_eval", official_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    h = pd.read_parquet(hourly_path, columns=[ID, "Hour", "SepsisLabel"])
    rng = np.random.default_rng(1)
    res = co.at(co.scores[name], tau)
    pick = rng.choice(co.m, size=min(n, co.m), replace=False)
    max_diff = 0.0
    for i in pick:
        lab = h.loc[h[ID] == co.pids[i], ["Hour", "SepsisLabel"]].sort_values("Hour")["SepsisLabel"].to_numpy()
        n_tot = len(lab)
        assert n_tot == co.n_hours[i], "n_hours does not match the hourly file"
        pred = np.zeros(n_tot, dtype=int)
        rows = slice(co.start[i], co.start[i] + co.n_pre[i])
        pre_pred = (co.scores[name][rows] >= tau).astype(int)
        pred[: len(pre_pred)] = pre_pred
        if co.septic[i]:
            pred[len(pre_pred):] = pre_pred[-1]                      # the stated adaptation
        u_off = mod.compute_prediction_utility(lab, pred, DT_EARLY, DT_OPT, DT_LATE, MAX_U_TP, MIN_U_FN, U_FP, 0)
        max_diff = max(max_diff, abs(u_off - res["U_obs"][i]))
        # best and inaction
        if co.septic[i]:
            t_sep = int(np.argmax(lab)) - DT_OPT
            best = np.zeros(n_tot, dtype=int)
            best[max(0, t_sep + DT_EARLY): min(t_sep + DT_LATE + 1, n_tot)] = 1
            ub = mod.compute_prediction_utility(lab, best, DT_EARLY, DT_OPT, DT_LATE, MAX_U_TP, MIN_U_FN, U_FP, 0)
            ui = mod.compute_prediction_utility(lab, np.zeros(n_tot, dtype=int), DT_EARLY, DT_OPT, DT_LATE, MAX_U_TP, MIN_U_FN, U_FP, 0)
            max_diff = max(max_diff, abs(ub - co.U_best[i]), abs(ui - co.U_inact[i]))
    assert max_diff < 1e-9, f"utility differs from the official function by {max_diff}"
    print(f"[validate] utility matches the official function for {len(pick)} patients (max diff {max_diff:.1e})")


def brute_force_check(co: Cohort, name: str, tau: float, n: int = 200) -> None:
    """Re-derive alerts, first alert and lead for random patients with plain loops."""
    rng = np.random.default_rng(2)
    res = co.at(co.scores[name], tau)
    for i in rng.choice(co.m, size=min(n, co.m), replace=False):
        p = co.scores[name][co.start[i]: co.start[i] + co.n_pre[i]]
        last, ep = -10 ** 9, 0
        first_hr = None
        for t, v in enumerate(p):
            if v >= tau:
                if first_hr is None:
                    first_hr = t
                if t - last > co.refractory:
                    ep += 1
                last = t
        assert ep == res["n_alerts"][i], "alert episode count mismatch"
        assert (first_hr is None) == (res["first"][i] >= BIG)
        if first_hr is not None:
            assert first_hr == res["first"][i]
            if co.septic[i]:
                assert res["lead_any"][i] == co.F_int[i] - first_hr
                timely = any(v >= tau for v in p[max(0, co.F_int[i] - co.horizon):])
                assert timely == res["det_timely"][i]
    print(f"[validate] alert episodes, first alerts, leads checked by brute force on {min(n, co.m)} patients")


# --------------------------------------------------------------------------- #
# Stage: performance
# --------------------------------------------------------------------------- #

def write_merged(new: pd.DataFrame, path: Path, models: list[str] | None, key: str = "model") -> pd.DataFrame:
    """Write a results table. With --models (a partial re-run) the rows of OTHER models already in the file are
    kept, so evaluating one new model never wipes the results of the others."""
    if models and path.exists():
        old = pd.read_csv(path)
        if key in old.columns:
            new = pd.concat([old[~old[key].isin(new[key].unique())], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def stage_performance(args: argparse.Namespace) -> None:
    t0 = time.time()
    out = args.outputs
    pred_path = out / "predictions" / "predictions.parquet"
    evald = out / "evaluation"
    evald.mkdir(parents=True, exist_ok=True)
    sfx = "_test" if args.split == "test" else ""
    if args.split == "test" and not args.final:
        sys.exit("ERROR: the test set is reserved for the final report. Re-run with --final when "
                 "all modelling decisions are frozen.")
    if args.split == "test":
        args.use_saved_thresholds = True
        print("*** FINAL TEST EVALUATION: use these numbers once; do not tune on them. ***")

    patients = pd.read_csv(out / "clean" / "patients.csv")
    df_all = pd.read_parquet(pred_path)
    models = [c for c in df_all.columns if c.startswith("p_")]
    if args.models:
        models = [m for m in models if m in args.models]
    print(f"[load] models found: {models}")

    # thresholds always come from DEV out-of-fold scores
    dev = df_all[df_all["split"] == "dev"]
    co_dev = Cohort(dev, patients, models, args.horizon, args.refractory)
    th_path = evald / "thresholds.json"
    curves = []
    if args.use_saved_thresholds:
        if not th_path.exists():
            sys.exit("ERROR: thresholds.json not found. Run a dev evaluation first.")
        thresholds = json.load(open(th_path))
        miss = [m for m in models if m not in thresholds]
        if miss:
            sys.exit(f"ERROR: no saved thresholds for {miss}; run the dev evaluation again.")
        print(f"[thresholds] loaded {th_path}")
    else:
        thresholds = {}
        for m in models:
            cur, th = curves_and_thresholds(co_dev, m, args.grid)
            curves.append(cur)
            thresholds[m] = th
            print(f"[thresholds] {m}: " + ", ".join(f"{k}={v:.4f}" for k, v in th.items()))
        json.dump({**(json.load(open(th_path)) if th_path.exists() else {}), **thresholds},
                  open(th_path, "w"), indent=2)
        write_merged(pd.concat(curves), evald / "utility_curve.csv", args.models)

    co = co_dev if args.split == "dev" else Cohort(df_all[df_all["split"] == args.split], patients,
                                                    models, args.horizon, args.refractory)
    m0 = models[-1]
    brute_force_check(co, m0, thresholds[m0]["far5"])
    if args.official_check:
        official_check(co, m0, thresholds[m0]["far5"], out / "clean" / "hourly_clean.parquet", args.official_check)

    metrics, lead = evaluate(co, models, thresholds, args.bootstraps, args.seed)
    write_merged(metrics, evald / f"alert_metrics{sfx}.csv", args.models)
    write_merged(lead, evald / f"leadtime_curve{sfx}.csv", args.models)
    print(f"[write] {evald}/alert_metrics{sfx}.csv, leadtime_curve{sfx}.csv  ({time.time() - t0:.0f}s)")

    # ---- console summary ----
    show = metrics[(metrics["operating_point"].isin(["far5", "util_opt"]))]
    cols = ["model", "operating_point", "group_type", "group", "n_septic", "far_per_100pd", "sens_any",
            "sens_timely", "median_lead_any", "util_norm"]
    cols = [c for c in cols if c in show.columns]
    print(f"\n=== {args.split.upper()} RESULTS (95% CI in alert_metrics.csv) ===")
    pd.set_option("display.width", 220)
    print(show[show["group_type"].isin(["all", "hospital"])][cols].round(3).to_string(index=False))
    print("\nBy pattern group (same tau as the overall row; no FAR per pattern):")
    pat = show[(show["group_type"] == "pattern") & (show["operating_point"] == "far5")]
    pc = ["model", "group", "n_septic", "sens_any", "sens_timely", "median_lead_any"]
    print(pat[pc].round(3).to_string(index=False))


# --------------------------------------------------------------------------- #
# Stage: timeprofile
# --------------------------------------------------------------------------- #

SWITCH_BINS = [("1-12", 1, 12), ("13-24", 13, 24), ("25-48", 25, 48), ("49-72", 49, 72), (">72", 73, 10 ** 6)]
HOUR_BINS = [("0-11", 0, 11), ("12-23", 12, 23), ("24-47", 24, 47), ("48-71", 48, 71), (">=72", 72, 10 ** 6)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def stage_timeprofile(args: argparse.Namespace) -> None:
    out = args.outputs
    evald = out / "evaluation"
    th_path = evald / "thresholds.json"
    if not th_path.exists():
        sys.exit("ERROR: thresholds.json not found. Run `performance` on the dev split first.")
    thresholds = json.load(open(th_path))
    patients = pd.read_csv(out / "clean" / "patients.csv")
    df = pd.read_parquet(out / "predictions" / "predictions.parquet")
    df = df[df["split"] == "dev"]
    models = [c for c in df.columns if c.startswith("p_") and c in thresholds]
    if args.models:
        models = [m for m in models if m in args.models]
    co = Cohort(df, patients, models, args.horizon, args.refractory)
    sept = co.septic
    rows = []
    for name in models:
        p = co.scores[name]
        for op, tau in thresholds[name].items():
            res = co.at(p, tau)
            for label, lo, hi in SWITCH_BINS:
                sel = sept & (co.F_int >= lo) & (co.F_int <= hi)
                n = int(sel.sum())
                for kind, key in (("septic_timely_by_switch_on", "det_timely"), ("septic_any_by_switch_on", "det_any")):
                    k = int(res[key][sel].sum())
                    lo_ci, hi_ci = wilson(k, n)
                    rows.append({"model": name, "operating_point": op, "kind": kind, "bin": label, "n": n,
                                 "value": 100 * k / n if n else np.nan, "lo": lo_ci, "hi": hi_ci})
            for label, lo, hi in SWITCH_BINS:
                # stay-length stratum: non-septic patients whose stay length falls in the same bin as the septic
                # switch-on hour, so sensitivity and false alarms can be read side by side within a stratum
                sel = ~sept & (co.n_pre >= lo) & (co.n_pre <= hi)
                rows.append({"model": name, "operating_point": op, "kind": "nonseptic_far_by_stay", "bin": label,
                             "n": int(sel.sum()), "value": co.far(res, sel) if sel.any() else np.nan,
                             "lo": np.nan, "hi": np.nan})
            above = p >= tau
            ns_rows = ~co.sept_row
            total_ns = int(ns_rows.sum())
            for label, lo, hi in HOUR_BINS:
                sel = ns_rows & (co.hour >= lo) & (co.hour <= hi)
                n = int(sel.sum())
                rows.append({"model": name, "operating_point": op, "kind": "nonseptic_alert_state_by_hour", "bin": label,
                             "n": n, "value": 100 * above[sel].sum() / n if n else np.nan, "lo": np.nan, "hi": np.nan})
                rows.append({"model": name, "operating_point": op, "kind": "nonseptic_patient_hours_share", "bin": label,
                             "n": n, "value": 100 * n / total_ns, "lo": np.nan, "hi": np.nan})
    tp = pd.DataFrame(rows)
    tp.to_csv(evald / "time_profile.csv", index=False)
    print(f"[write] {evald / 'time_profile.csv'}")

    pd.set_option("display.width", 200)
    for name in models:
        t = tp[(tp["model"] == name) & (tp["operating_point"] == "far5")]
        a = t[t["kind"] == "septic_timely_by_switch_on"].set_index("bin")
        b = t[t["kind"] == "nonseptic_alert_state_by_hour"].set_index("bin")
        c = t[t["kind"] == "nonseptic_patient_hours_share"].set_index("bin")
        print(f"\n{name} at far5")
        print("  septic: % warned in the last 12 h by label switch-on hour (n):  " +
              " | ".join(f"{k}: {a.loc[k, 'value']:.1f}% (n={int(a.loc[k, 'n'])})" for k in a.index))
        print("  non-septic: % of patient-hours above threshold by hour of stay:  " +
              " | ".join(f"{k}: {b.loc[k, 'value']:.1f}%" for k in b.index))
        f = t[t["kind"] == "nonseptic_far_by_stay"].set_index("bin")
        print("  non-septic: false alarms /100 patient-days by stay length (same bins as above): " +
              " | ".join(f"{k}: {f.loc[k, 'value']:.1f} (n={int(f.loc[k, 'n'])})" for k in f.index))
        print("  share of all non-septic patient-hours in each hour bin:           " +
              " | ".join(f"{k}: {c.loc[k, 'value']:.1f}%" for k in c.index))


# --------------------------------------------------------------------------- #
# Shared helpers for the stages below
# --------------------------------------------------------------------------- #

def slug(name: str) -> str:
    """Same rule as 05_train_models.py: 'No abnormal vitals' -> 'no_abnormal_vitals'."""
    return "".join(c if c.isalnum() else "_" for c in str(name).lower()).strip("_")


def load_split(out: Path, split: str, final: bool) -> pd.DataFrame:
    if split == "test" and not final:
        sys.exit("ERROR: the test set is reserved for the final report. Re-run with --final when all "
                 "modelling decisions are frozen.")
    df = pd.read_parquet(out / "predictions" / "predictions.parquet")
    return df[df["split"] == split].reset_index(drop=True)


def load_thresholds(out: Path) -> dict:
    th = out / "evaluation" / "thresholds.json"
    if not th.exists():
        sys.exit("ERROR: thresholds.json not found. Run `performance` on the dev split first.")
    return json.load(open(th))


def row_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    if y.min() == y.max():
        return np.nan, np.nan
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


# --------------------------------------------------------------------------- #
# Stage: timecontrol (the time-in-ICU fix)
# --------------------------------------------------------------------------- #

def hour_group(hour: np.ndarray) -> np.ndarray:
    """Single hours 0..71, then 72-95, 96-143, 144+ pooled (few long stays)."""
    return np.where(hour < 72, hour, np.where(hour < 96, 72, np.where(hour < 144, 96, 144)))


def time_adjust(p: np.ndarray, hg: np.ndarray, ref: np.ndarray, min_ref: int) -> np.ndarray:
    """Percentile of each score among REFERENCE rows (dev, non-septic) in the same hour group (mid-rank
    for ties). A threshold on this score flags the same share of non-septic hours at every hour of stay,
    so time in ICU alone can no longer raise the alert rate. Sparse late groups borrow earlier hours."""
    out = np.full(len(p), np.nan)
    gs = np.unique(hg)
    for i, g in enumerate(gs):
        j = i
        r = p[ref & (hg == g)]
        while len(r) < min_ref and j > 0:
            j -= 1
            r = p[ref & (hg >= gs[j]) & (hg <= g)]
        if len(r) == 0:
            r = p[ref]
        r = np.sort(r)
        sel = hg == g
        x = p[sel]
        out[sel] = (np.searchsorted(r, x, "left") + np.searchsorted(r, x, "right")) / (2.0 * len(r))
    return out


def stage_timecontrol(args: argparse.Namespace) -> None:
    out = args.outputs
    evald = out / "evaluation"
    evald.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions" / "predictions.parquet"
    df = pd.read_parquet(pred_path)
    models = [c for c in df.columns if c.startswith("p_") and not c.endswith("_tadj")]
    if args.models:
        models = [m for m in models if m in args.models]
    dev = (df["split"] == "dev").to_numpy()
    y = df["y"].to_numpy()
    hour = df["Hour"].to_numpy()
    hg = hour_group(hour)
    print(f"[load] models: {models}")

    # 1. discrimination WITHIN hour of stay (dev only). The hour-stratified AUROC compares positive and
    #    negative hours at the SAME hour of stay, so a score that only tracks time in ICU gets ~0.5.
    rows = []
    for m in models:
        p = df[m].to_numpy()
        a, b = row_auc(y[dev], p[dev])
        rows.append({"model": m, "bin": "all hours (pooled)", "auroc": a, "auprc": b,
                     "n_pos": int(y[dev].sum()), "n_rows": int(dev.sum())})
        num = den = 0.0
        for g in np.unique(hg[dev]):
            s = dev & (hg == g)
            npos = int(y[s].sum())
            nneg = int(s.sum()) - npos
            if npos and nneg:
                num += row_auc(y[s], p[s])[0] * npos * nneg
                den += npos * nneg
        rows.append({"model": m, "bin": "hour-stratified", "auroc": num / den if den else np.nan, "auprc": np.nan,
                     "n_pos": int(y[dev].sum()), "n_rows": int(dev.sum())})
        for label, lo, hi in HOUR_BINS:
            s = dev & (hour >= lo) & (hour <= hi)
            a, b = row_auc(y[s], p[s])
            rows.append({"model": m, "bin": label, "auroc": a, "auprc": b, "n_pos": int(y[s].sum()), "n_rows": int(s.sum())})
    wh = pd.DataFrame(rows)
    write_merged(wh, evald / "within_hour_discrimination.csv", args.models)
    print(f"[write] {evald / 'within_hour_discrimination.csv'}")

    # 2. time-adjusted scores p_<model>_tadj (reference = dev non-septic rows; test reuses it)
    if not args.no_adjust:
        ref = dev & ~df["septic_patient"].to_numpy(bool)
        new = {f"{m}_tadj": time_adjust(df[m].to_numpy(), hg, ref, args.min_ref) for m in models}
        for c, v in new.items():
            df[c] = v
        # self-check: in every single-hour group with plenty of reference rows, a cut at the 95th
        # percentile flags ~5% of the reference rows
        c0 = next(iter(new))
        for g in range(0, 24):
            r = ref & (hg == g)
            if r.sum() >= 2000:
                share = float((df.loc[r, c0] > 0.95).mean())
                assert abs(share - 0.05) < 0.02, f"time adjustment wrong at hour {g}: {share:.3f}"
        df.to_parquet(pred_path, index=False)
        print(f"[write] {pred_path}  (+ {len(new)} *_tadj columns). Run `performance` again to evaluate them.")

    pd.set_option("display.width", 200)
    show = wh[wh["bin"].isin(["all hours (pooled)", "hour-stratified"])].pivot(index="model", columns="bin", values="auroc")
    print("\nRow-level AUROC, pooled vs compared only within the same hour of stay (dev):")
    print(show.round(4).sort_values("hour-stratified", ascending=False).to_string())
    print("A time-only model should drop to about 0.5 in the hour-stratified column; the gap between the two\n"
          "columns shows how much of a model's AUROC comes from time in ICU.")


# --------------------------------------------------------------------------- #
# Stage: transfer (alert-level, thresholds from the source hospital)
# --------------------------------------------------------------------------- #

def stage_transfer(args: argparse.Namespace) -> None:
    out = args.outputs
    evald = out / "evaluation"
    evald.mkdir(parents=True, exist_ok=True)
    v = "" if args.variant == "all" else f"_{args.variant}"
    wcol, tcol = f"p_lgbm_within{v}", f"p_lgbm_transfer{v}"
    patients = pd.read_csv(out / "clean" / "patients.csv")
    dev = load_split(out, "dev", True)
    ev = dev if args.split == "dev" else load_split(out, "test", args.final)
    if wcol not in dev.columns or tcol not in dev.columns:
        sys.exit(f"ERROR: {wcol} / {tcol} not found. Run `05_train_models.py transfer` first.")
    hs = sorted(dev["hospital"].astype(str).unique())
    rng = np.random.default_rng(args.seed)
    rows = []
    for s, t in ((hs[0], hs[1]), (hs[1], hs[0])):
        src_dev = dev[dev["hospital"].astype(str) == s]
        tgt_dev = dev[dev["hospital"].astype(str) == t]
        tgt = ev[ev["hospital"].astype(str) == t]
        # source threshold: from the source hospital's OOF within-hospital scores (the transfer model's
        # own source scores are in-sample, so they would give a biased threshold)
        _, th_src = curves_and_thresholds(Cohort(src_dev, patients, [wcol], args.horizon, args.refractory), wcol, args.grid)
        co_td = Cohort(tgt_dev, patients, [wcol, tcol], args.horizon, args.refractory)
        _, th_t_within = curves_and_thresholds(co_td, wcol, args.grid)
        _, th_t_transfer = curves_and_thresholds(co_td, tcol, args.grid)
        co = Cohort(tgt, patients, [wcol, tcol], args.horizon, args.refractory)
        y = tgt["y"].to_numpy()
        for op in th_src:
            for label, col, tau in (
                    (f"within {t} (threshold from {t} dev)", wcol, th_t_within[op]),
                    (f"transfer {s}->{t} (threshold from {s})", tcol, th_src[op]),
                    (f"transfer {s}->{t} (threshold re-set on {t} dev)", tcol, th_t_transfer[op])):
                res = co.at(co.scores[col], tau)
                gm = group_metrics(co, res, np.ones(co.m, bool), args.bootstraps, rng, True, True)
                a, b = row_auc(y, co.scores[col])
                rows.append({"direction": f"{s}->{t}", "target": t, "setting": label, "model": col,
                             "operating_point": op, "tau": tau, "row_auroc": a, "row_auprc": b,
                             "mean_score": float(co.scores[col].mean()), "observed_rate": float(y.mean()), **gm})
    res = pd.DataFrame(rows)
    sfx = "_test" if args.split == "test" else ""
    res.to_csv(evald / f"transfer_metrics{v}{sfx}.csv", index=False)
    print(f"[write] {evald / f'transfer_metrics{v}{sfx}.csv'}")
    if args.split == "dev":
        print("NOTE: on dev, the 're-set on target' and 'within' thresholds are chosen on the rows they are scored on, "
              "so those two settings are optimistic references; the 'threshold from source' setting is the honest transfer.")
    cols = ["direction", "setting", "row_auroc", "far_per_100pd", "sens_timely", "sens_timely_lo", "sens_timely_hi", "util_norm"]
    pd.set_option("display.width", 220)
    print(res[res["operating_point"] == "far5"][cols].round(3).to_string(index=False))


# --------------------------------------------------------------------------- #
# Stage: frequency (does measurement frequency explain pattern differences?)
# --------------------------------------------------------------------------- #

TRIGGERS = {"Hemodynamic": ["MAP"], "Respiratory": ["O2Sat", "Resp"], "Inflammatory": ["Temp", "HR"]}
ALL5 = ["HR", "Resp", "MAP", "O2Sat", "Temp"]


def _logit_or(X: np.ndarray, y: np.ndarray, names: list[str], B: int, rng) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    def fit(Xb, yb):
        # mild ridge penalty (C = 1 on 0/1 dummies and per-SD covariates): keeps odds ratios finite when a
        # small group is (almost) always warned, at the cost of a slight shrink toward 1
        return LogisticRegression(C=1.0, max_iter=2000).fit(Xb, yb).coef_[0]

    point = fit(X, y)
    boots = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(fit(X[i], y[i]))
    bt = np.array(boots)
    return pd.DataFrame({"term": names, "odds_ratio": np.exp(point),
                         "or_lo": np.exp(np.percentile(bt, 2.5, axis=0)), "or_hi": np.exp(np.percentile(bt, 97.5, axis=0))})


def stage_frequency(args: argparse.Namespace) -> None:
    from scipy.stats import spearmanr
    out = args.outputs
    evald = out / "evaluation"
    thresholds = load_thresholds(out)
    if args.model not in thresholds:
        sys.exit(f"ERROR: no thresholds for {args.model}; run `performance` with it first.")
    tau = thresholds[args.model][args.operating_point]
    patients = pd.read_csv(out / "clean" / "patients.csv")
    df = load_split(out, "dev", True)
    co = Cohort(df, patients, [args.model], args.horizon, args.refractory)
    res = co.at(co.scores[args.model], tau)
    sp = pd.DataFrame({ID: co.pids, "pattern": co.pattern, "hospital": co.hospital, "F": co.F,
                       "det_timely": res["det_timely"], "lead_any": res["lead_any"]})[co.septic]
    pf_path = out / "patterns" / "pattern_features.parquet"
    pf = pd.read_parquet(pf_path) if pf_path.exists() else pd.read_csv(out / "patterns" / "pattern_features.csv.gz")
    d = sp.merge(pf[[ID, "window_hours", "n_vital_obs_window", "n_lab_hours_window"] +
                    [f"{v}_n_obs" for v in ALL5]], on=ID, how="left")
    wh = d["window_hours"].clip(lower=1)
    d["lab_hours_per_h"] = d["n_lab_hours_window"] / wh
    d["vital_obs_per_h"] = d["n_vital_obs_window"] / (len(ALL5) * wh)
    d["temp_obs_per_h"] = d["Temp_n_obs"] / wh
    trig = d["pattern"].map(lambda p: TRIGGERS.get(p, ALL5))
    d["trigger_obs_per_h"] = [sum(r[f"{v}_n_obs"] for v in t) / (len(t) * w)
                              for (_, r), t, w in zip(d.iterrows(), trig, wh)]

    # 1. per pattern
    rows = []
    for p, g in d.groupby("pattern"):
        k, n = int(g["det_timely"].sum()), len(g)
        lo, hi = wilson(k, n)
        det = g[g["lead_any"].notna()]
        rho = spearmanr(det["trigger_obs_per_h"], det["lead_any"]) if len(det) >= 10 else None
        rows.append({"pattern": p, "n_septic": n, "sens_timely_%": 100 * k / n, "lo": lo, "hi": hi,
                     "median_lead_any_h": float(det["lead_any"].median()) if len(det) else np.nan,
                     "trigger_vitals": "+".join(TRIGGERS.get(p, ALL5)),
                     "median_trigger_obs_per_h": float(g["trigger_obs_per_h"].median()),
                     "median_temp_obs_per_h": float(g["temp_obs_per_h"].median()),
                     "median_lab_hours_per_h": float(g["lab_hours_per_h"].median()),
                     "median_label_switch_on_hour": float(g["F"].median()),
                     "spearman_trigger_rate_vs_lead": float(rho.statistic) if rho is not None else np.nan,
                     "spearman_p": float(rho.pvalue) if rho is not None else np.nan})
    byp = pd.DataFrame(rows).sort_values("n_septic", ascending=False)

    # 2. by quartile of measurement intensity (all septic patients)
    rows = []
    for var in ("lab_hours_per_h", "vital_obs_per_h", "trigger_obs_per_h"):
        q = pd.qcut(d[var].rank(method="first"), 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
        for lab, g in d.groupby(q, observed=True):
            k, n = int(g["det_timely"].sum()), len(g)
            lo, hi = wilson(k, n)
            rows.append({"measure": var, "quartile": str(lab), "n_septic": n, "range": f"{g[var].min():.2f}-{g[var].max():.2f}",
                         "sens_timely_%": 100 * k / n, "lo": lo, "hi": hi})
    byq = pd.DataFrame(rows)

    # 3. do pattern differences survive adjustment for measurement frequency and switch-on hour?
    a = d[d["pattern"] != "unassigned"] if args.exclude_unassigned else d
    sizes = a["pattern"].value_counts()
    small = sizes[sizes < args.min_group].index.tolist()
    if small:
        print(f"[note] left out of the regression (< {args.min_group} patients): {small}")
    a = a[~a["pattern"].isin(small)].reset_index(drop=True)
    ref = a["pattern"].value_counts().index[0]
    pats = [p for p in sorted(a["pattern"].unique()) if p != ref]
    dummies = np.column_stack([(a["pattern"] == p).to_numpy(float) for p in pats])
    z = lambda s: ((s - s.mean()) / s.std()).to_numpy(float)
    cov = np.column_stack([z(np.log1p(a["lab_hours_per_h"])), z(a["trigger_obs_per_h"]),
                           z(np.log(a["F"])), (a["hospital"] == "B").to_numpy(float)])
    yv = a["det_timely"].to_numpy(int)
    rng = np.random.default_rng(args.seed)
    t1 = _logit_or(dummies, yv, [f"pattern: {p}" for p in pats], args.bootstraps, rng).assign(model="unadjusted")
    covn = ["lab hours per h (log, per SD)", "trigger-vital obs per h (per SD)", "label switch-on hour (log, per SD)",
            "hospital B"]
    t2 = _logit_or(np.column_stack([dummies, cov]), yv, [f"pattern: {p}" for p in pats] + covn, args.bootstraps,
                   rng).assign(model="adjusted")
    adj = pd.concat([t1, t2]).assign(reference_pattern=ref, outcome="warned in the last 12 h")

    evald.mkdir(parents=True, exist_ok=True)
    byp.to_csv(evald / "frequency_by_pattern.csv", index=False)
    byq.to_csv(evald / "frequency_by_quartile.csv", index=False)
    adj.to_csv(evald / "frequency_adjusted.csv", index=False)
    json.dump({"model": args.model, "operating_point": args.operating_point, "tau": tau, "reference_pattern": ref,
               "n_septic": int(len(d)), "n_regression": int(len(a)), "left_out_small_groups": small,
               "regression": "logistic, ridge C=1, patient bootstrap CIs"}, open(evald / "frequency_report.json", "w"), indent=2)
    print(f"[write] {evald}/frequency_by_pattern.csv, frequency_by_quartile.csv, frequency_adjusted.csv")
    pd.set_option("display.width", 220)
    print(byp.round(3).to_string(index=False))
    print("\nOdds ratios for being warned in time (reference pattern: %s):" % ref)
    print(adj[["model", "term", "odds_ratio", "or_lo", "or_hi"]].round(3).to_string(index=False))
    print("If pattern odds ratios move toward 1 after adjustment, measurement frequency / timing explains part "
          "of the difference.")


# --------------------------------------------------------------------------- #
# Stage: shap (TreeSHAP from LightGBM itself; no shap package needed)
# --------------------------------------------------------------------------- #

def stage_shap(args: argparse.Namespace) -> None:
    import lightgbm as lgb
    out = args.outputs
    evald = out / "evaluation"
    fpath = out / "features" / "hourly_features.parquet"
    names = json.load(open(out / "features" / "feature_names.json"))
    meta = pd.read_parquet(fpath, columns=[ID, "Hour", "y", "split", "fold", "pattern", "septic_patient"])
    rng = np.random.default_rng(args.seed)
    dev = (meta["split"] == "dev").to_numpy()
    y = meta["y"].to_numpy()
    pat = meta["pattern"].astype(str).to_numpy()
    # rows to explain: hours inside the warning window (y = 1), sampled per pattern, + random negative hours
    pick, label = [], []
    for p in sorted(set(pat[dev & (y == 1)])):
        idx = np.flatnonzero(dev & (y == 1) & (pat == p))
        idx = rng.choice(idx, size=min(len(idx), args.max_rows), replace=False)
        pick.append(idx)
        label += [p] * len(idx)
    neg = np.flatnonzero(dev & (y == 0))
    neg = rng.choice(neg, size=min(len(neg), args.max_rows), replace=False)
    pick.append(neg)
    label += ["negative hours (y = 0)"] * len(neg)
    pick = np.concatenate(pick)
    label = np.array(label)
    order = np.argsort(pick)
    pick, label = pick[order], label[order]
    X = np.empty((len(pick), len(names)), dtype=np.float32)
    for j, c in enumerate(names):
        X[:, j] = pd.read_parquet(fpath, columns=[c])[c].to_numpy(np.float32)[pick]
    fold = meta["fold"].to_numpy()[pick]
    contrib = np.zeros_like(X, dtype=np.float64)
    for k in np.unique(fold):
        mpath = out / "models" / f"lgbm_fold{k}.txt"
        if not mpath.exists():
            sys.exit(f"ERROR: {mpath} not found. Run `05_train_models.py baselines` first.")
        b = lgb.Booster(model_file=str(mpath))
        s = fold == k                      # each row is explained by the model that did NOT train on it
        c = b.predict(X[s], pred_contrib=True)
        assert np.allclose(c.sum(axis=1), b.predict(X[s], raw_score=True), atol=1e-6), "SHAP values do not add up"
        contrib[s] = c[:, :-1]
    rows = []
    for g in np.unique(label):
        s = label == g
        ma, ms = np.abs(contrib[s]).mean(axis=0), contrib[s].mean(axis=0)
        rows += [{"group": g, "feature": f, "mean_abs_shap": float(a), "mean_shap": float(m), "n_rows": int(s.sum())}
                 for f, a, m in zip(names, ma, ms)]
    sh = pd.DataFrame(rows).sort_values(["group", "mean_abs_shap"], ascending=[True, False])
    evald.mkdir(parents=True, exist_ok=True)
    sh.to_csv(evald / "shap_by_pattern.csv", index=False)
    print(f"[write] {evald / 'shap_by_pattern.csv'}  ({len(pick)} rows explained, out-of-fold models)")
    for g, t in sh.groupby("group"):
        print(f"  {g:24s} top: " + ", ".join(f"{r.feature} ({r.mean_shap:+.2f})" for r in t.head(5).itertuples()))


# --------------------------------------------------------------------------- #
# Stage: compare (global vs pattern-aware models, paired by patient)
# --------------------------------------------------------------------------- #

PATTERN_AWARE = ["p_gru", "p_gru_mtl", "p_gru_moe", "p_lgbm_moe_max"]


def compare_table(co, res: dict, models: list[str], base: str, prefix: str, B: int, rng) -> pd.DataFrame:
    """Per group: % warned in time per model and the PAIRED difference vs `base` (patient bootstrap). Columns named
    <prefix><group> are per-group experts: each is shown only for its own group, as the oracle."""
    experts = [m for m in models if m.startswith(prefix)]
    pats = sorted(set(co.pattern[co.septic]))
    expert_of = {p: f"{prefix}{slug(p)}" for p in pats}
    assigned = co.septic & np.isin(co.pattern, [p for p in pats if expert_of[p] in experts])

    # oracle = every septic patient scored by the expert of its own (hindsight) pattern, each expert at its
    # own threshold. Not deployable; an upper bound for "if we knew the pattern".
    det_or = np.zeros(co.m, bool)
    for p in pats:
        if expert_of[p] in res:
            s = co.septic & (co.pattern == p)
            det_or[s] = res[expert_of[p]]["det_timely"][s]

    rows = []
    groups = [("all patterns with an expert", assigned)] + [(p, co.septic & (co.pattern == p)) for p in pats]
    for g, mask in groups:
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        cands = [(m, "global reference" if m == base else "pattern-aware", res[m]["det_timely"]) for m in models
                 if not m.startswith(prefix)]
        if g in expert_of and expert_of[g] in res:
            cands.append((expert_of[g], "oracle (own-pattern expert)", res[expert_of[g]]["det_timely"]))
        elif g == "all patterns with an expert" and experts:
            cands.append(("oracle", "oracle (own-pattern expert)", det_or))
        b0 = res[base]["det_timely"]
        for m, role, det in cands:
            diff = det[idx].astype(float) - b0[idx].astype(float)
            bs = [diff[rng.integers(0, len(idx), len(idx))].mean() for _ in range(B)]
            k, n = int(det[idx].sum()), len(idx)
            lo, hi = wilson(k, n)
            rows.append({"group": g, "model": m, "role": role, "n_septic": n, "sens_timely_%": 100 * k / n,
                         "lo": lo, "hi": hi, "diff_vs_reference_pts": 100 * diff.mean(),
                         "diff_lo": 100 * np.percentile(bs, 2.5), "diff_hi": 100 * np.percentile(bs, 97.5),
                         "far_per_100pd": co.far(res[m]) if m in res else np.nan})
    return pd.DataFrame(rows)


def stage_compare(args: argparse.Namespace) -> None:
    out = args.outputs
    evald = out / "evaluation"
    thresholds = load_thresholds(out)
    patients = pd.read_csv(out / "clean" / "patients.csv")
    df = load_split(out, args.split, args.final)
    base = args.reference
    experts = sorted(c for c in df.columns if c.startswith("p_lgbm_expert_"))
    models = [base] + [m for m in PATTERN_AWARE if m in df.columns] + experts
    miss = [m for m in models if m not in thresholds]
    if miss:
        sys.exit(f"ERROR: no thresholds for {miss}. Run `performance` on dev with these models first.")
    co = Cohort(df, patients, models, args.horizon, args.refractory)
    op = args.operating_point
    res = {m: co.at(co.scores[m], thresholds[m][op]) for m in models}
    rng = np.random.default_rng(args.seed)
    cmp_ = compare_table(co, res, models, base, "p_lgbm_expert_", args.bootstraps, rng)
    sfx = "_test" if args.split == "test" else ""
    evald.mkdir(parents=True, exist_ok=True)
    cmp_.to_csv(evald / f"pattern_model_comparison{sfx}.csv", index=False)
    print(f"[write] {evald / f'pattern_model_comparison{sfx}.csv'}  (operating point {op}, reference {base})")
    pd.set_option("display.width", 220)
    print(cmp_[["group", "model", "role", "n_septic", "sens_timely_%", "diff_vs_reference_pts", "diff_lo", "diff_hi",
                "far_per_100pd"]].round(1).to_string(index=False))
    print("diff = paired difference in % warned in the last 12 h vs the reference, same patients, each model at its own "
          "threshold; 95% CI from a patient bootstrap. The oracle row has no single false-alarm rate (several experts).")


# --------------------------------------------------------------------------- #
# Stage: plots (data behind the "Result figures" dashboard tab; dev only)
# --------------------------------------------------------------------------- #

PLOT_MODELS = ["p_logreg", "p_lgbm", "p_gru", "p_ensemble", "p_lgbm_no_time", "p_lgbm_time_only", "p_gru_mtl",
               "p_gru_moe", "p_lgbm_moe_max"]
DIST_VARS = ["HR", "MAP", "SBP", "Resp", "O2Sat", "Temp", "Lactate", "WBC", "Creatinine", "Platelets"]
CORR_VARS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
             "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
             "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct",
             "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
VITAL_SET = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"}


def _thin(a: np.ndarray, b: np.ndarray, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Keep about n points of a curve (for small files and fast plots); first and last point kept."""
    if len(a) <= n:
        return a, b
    i = np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))
    return a[i], b[i]


def kaplan_meier(t: np.ndarray, e: np.ndarray) -> list[tuple[float, float, int]]:
    """Product-limit estimate. Returns (time, survival just after time, number at risk)."""
    rows, s = [(0.0, 1.0, int(len(t)))], 1.0
    for tt in np.unique(t):
        n = int((t >= tt).sum())
        d = int(((t == tt) & (e == 1)).sum())
        if d:
            s *= 1 - d / n
            rows.append((float(tt), s, n))
    return rows


def logrank(t: np.ndarray, e: np.ndarray, g: np.ndarray) -> tuple[float, int, float]:
    """Multi-group log-rank test. Returns (chi-square, degrees of freedom, p-value)."""
    from scipy.stats import chi2
    groups = np.unique(g)
    G = len(groups)
    O, E, V = np.zeros(G), np.zeros(G), np.zeros((G, G))
    for tt in np.unique(t[e == 1]):
        at = t >= tt
        n = at.sum()
        d = ((t == tt) & (e == 1)).sum()
        ng = np.array([(at & (g == k)).sum() for k in groups], float)
        dg = np.array([((t == tt) & (e == 1) & (g == k)).sum() for k in groups], float)
        O += dg
        E += ng * d / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (np.diag(ng / n) - np.outer(ng, ng) / n ** 2)
    diff = (O - E)[:-1]
    stat = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    return stat, G - 1, float(chi2.sf(stat, G - 1))


def stage_plots(args: argparse.Namespace) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    out = args.outputs
    evald = out / "evaluation" / "figures"
    evald.mkdir(parents=True, exist_ok=True)
    df = load_split(out, "dev", True)
    models = args.models or [m for m in PLOT_MODELS if m in df.columns]
    y = df["y"].to_numpy()
    print(f"[load] dev rows: {len(df)} | models: {models}")

    # 1. ROC and precision-recall curves (hour level)
    roc, pr, summ = [], [], []
    for m in models:
        p = df[m].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        fpr, tpr = _thin(fpr, tpr)
        roc += [{"model": m, "fpr": a, "tpr": b} for a, b in zip(fpr, tpr)]
        prec, rec, _ = precision_recall_curve(y, p)
        rec, prec = _thin(rec[::-1], prec[::-1])
        pr += [{"model": m, "recall": a, "precision": b} for a, b in zip(rec, prec)]
        summ.append({"model": m, "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
                     "prevalence": float(y.mean()), "n_hours": int(len(y))})
    pd.DataFrame(roc).to_csv(evald / "roc_curves.csv", index=False)
    pd.DataFrame(pr).to_csv(evald / "pr_curves.csv", index=False)
    pd.DataFrame(summ).to_csv(evald / "curve_summary.csv", index=False)
    print("[1/8] ROC and precision-recall curves")

    # 2. patient-level confusion matrix at each saved operating point
    thresholds = load_thresholds(out)
    patients = pd.read_csv(out / "clean" / "patients.csv")
    tm = [m for m in models if m in thresholds]
    co = Cohort(df, patients, tm, args.horizon, args.refractory)
    cm = []
    for m in tm:
        for op, tau in thresholds[m].items():
            r = co.at(co.scores[m], tau)
            tp = int(r["det_timely"][co.septic].sum())
            fn = int(co.septic.sum()) - tp
            fp = int((r["n_alerts"][~co.septic] > 0).sum())
            tn = int((~co.septic).sum()) - fp
            cm.append({"model": m, "operating_point": op, "tau": tau, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                       "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                       "ppv": tp / (tp + fp) if tp + fp else np.nan})
    pd.DataFrame(cm).to_csv(evald / "confusion_matrix.csv", index=False)
    print("[2/8] patient-level confusion matrices")

    # 3. calibration, overall and per pattern (pattern rows = that pattern's septic patients + ALL non-septic)
    sept_row = df["septic_patient"].to_numpy(bool)
    pat_row = df["pattern"].astype(str).to_numpy()
    groups = [("all", np.ones(len(df), bool))] + [(g, ~sept_row | (pat_row == g))
                                                  for g in sorted(set(pat_row[sept_row]))]
    cal, brier = [], []
    for m in [m for m in models if m != "p_ensemble"]:      # the ensemble score is a percentile, not a probability
        p = df[m].to_numpy()
        for g, mask in groups:
            pm, ym = p[mask], y[mask]
            if ym.sum() == 0:
                continue
            bins = pd.qcut(pd.Series(pm).rank(method="first"), args.calib_bins, labels=False)
            t = pd.DataFrame({"p": pm, "y": ym, "b": bins}).groupby("b").agg(mean_pred=("p", "mean"),
                                                                              observed=("y", "mean"), n=("y", "size"))
            cal += [{"model": m, "group": g, **r} for r in t.reset_index(drop=True).to_dict("records")]
            brier.append({"model": m, "group": g, "brier": float(np.mean((pm - ym) ** 2)),
                          "mean_pred": float(pm.mean()), "observed": float(ym.mean()), "n_hours": int(len(ym))})
    pd.DataFrame(cal).to_csv(evald / "calibration.csv", index=False)
    pd.DataFrame(brier).to_csv(evald / "calibration_summary.csv", index=False)
    print("[3/8] calibration curves")

    # 4. forest plot: drawn by the dashboard from pattern_model_comparison.csv (no new numbers needed)
    print("[4/8] forest plot uses pattern_model_comparison.csv")

    # 5. Kaplan-Meier: time to first alert, counted from the start of the last 24 h before label switch-on
    km_rows, km_tests = [], []
    km_model = args.model if args.model in tm else tm[0]
    for op in ("far5", "util_opt"):
        if op not in thresholds[km_model]:
            continue
        above = co.scores[km_model] >= thresholds[km_model][op]
        ws_row = np.maximum(0, co.F_row - args.km_window)
        hw = np.where(above & co.sept_row & (co.hour >= ws_row), co.hour, BIG)
        first = np.minimum.reduceat(hw, co.start)
        si = np.flatnonzero(co.septic)
        ws = np.maximum(0, co.F_int[si] - args.km_window)
        ev = (first[si] < BIG).astype(int)
        tt = np.where(ev == 1, first[si] - ws, co.F_int[si] - ws).astype(float)   # censored at switch-on
        pat = co.pattern[si]
        for g in ["all"] + sorted(set(pat)):
            s = np.ones(len(si), bool) if g == "all" else pat == g
            km_rows += [{"model": km_model, "operating_point": op, "group": g, "hours_since_window_start": a,
                         "share_not_yet_alerted": b, "n_at_risk": n, "n_patients": int(s.sum())}
                        for a, b, n in kaplan_meier(tt[s], ev[s])]
        big = np.isin(pat, [g for g in set(pat) if (pat == g).sum() >= 10])
        if len(set(pat[big])) >= 2:
            chi, dfree, pv = logrank(tt[big], ev[big], pat[big])
            km_tests.append({"model": km_model, "operating_point": op, "chi2": chi, "df": dfree, "p_value": pv,
                             "groups": ", ".join(sorted(set(pat[big])))})
    pd.DataFrame(km_rows).to_csv(evald / "km_time_to_alert.csv", index=False)
    pd.DataFrame(km_tests).to_csv(evald / "km_logrank.csv", index=False)
    print(f"[5/8] Kaplan-Meier curves ({km_model})")

    # 6-8. EDA on the cleaned hourly data of dev patients (pre-onset hours only)
    hpath = out / "clean" / "hourly_clean.parquet"
    folds = pd.read_csv(out / "splits" / "folds.csv")
    dev_ids = folds.loc[folds["split"] == "dev", ID]
    cols = [ID, "hospital", "septic_patient", "hours_to_onset", "is_pre_onset"] + CORR_VARS
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(hpath).schema.names)
    h = pd.read_parquet(hpath, columns=[c for c in cols if c in have])
    h = h[h[ID].isin(dev_ids) & h["is_pre_onset"]]
    vars_ = [v for v in CORR_VARS if v in h.columns]

    # 6. correlation heatmap (Spearman, pairwise, on a row sample)
    hs = h.sample(n=min(len(h), args.corr_rows), random_state=args.seed)
    corr = hs[vars_].corr(method="spearman", min_periods=args.corr_min_pairs)
    corr.to_csv(evald / "correlation_spearman.csv")
    print(f"[6/8] correlation matrix ({len(hs)} sampled hours)")

    # 7. septic (last 24 h before switch-on) vs non-septic distributions
    sep24 = h["septic_patient"].astype(bool) & (h["hours_to_onset"] <= 24)
    non = ~h["septic_patient"].astype(bool)
    dist = []
    for v in [v for v in DIST_VARS if v in h.columns]:
        for name, msk in (("septic, last 24 h before switch-on", sep24), ("non-septic", non)):
            x = h.loc[msk, v].dropna().to_numpy(float)
            if len(x):
                q = np.percentile(x, [5, 25, 50, 75, 95])
                dist.append({"variable": v, "group": name, "n": int(len(x)), "mean": float(x.mean()),
                             "p5": q[0], "q1": q[1], "median": q[2], "q3": q[3], "p95": q[4]})
    pd.DataFrame(dist).to_csv(evald / "distributions.csv", index=False)
    print("[7/8] septic vs non-septic distributions")

    # 8. missing data: % of hours with no measurement, by variable, hospital and septic status
    miss = []
    sep = h["septic_patient"].astype(bool)
    for v in vars_:
        isna = h[v].isna()
        for gname, msk in (("all", np.ones(len(h), bool)), ("septic", sep), ("non-septic", ~sep),
                           ("hospital A", h["hospital"].astype(str) == "A"), ("hospital B", h["hospital"].astype(str) == "B")):
            if msk.sum():
                miss.append({"variable": v, "kind": "vital" if v in VITAL_SET else "lab", "group": gname,
                             "missing_pct": float(100 * isna[msk].mean())})
    pd.DataFrame(miss).to_csv(evald / "missing_share.csv", index=False)
    print("[8/8] missing-data shares")
    json.dump({"models": models, "km_model": km_model, "km_window_h": args.km_window, "calib_bins": args.calib_bins,
               "corr_rows": int(len(hs)), "split": "dev"}, open(evald / "figures_report.json", "w"), indent=2)
    print(f"[write] {evald}")


# --------------------------------------------------------------------------- #
# Stage: organs (results per organ group, from backend/organ_pathways.py)
# --------------------------------------------------------------------------- #

ORGAN_MAIN = ["p_lgbm", "p_logreg", "p_gru", "p_ensemble"]
ORGAN_TRIGGER = {"Lungs": "lungs", "Clotting": "clotting", "Liver": "liver", "Circulation": "circulation",
                 "Kidneys": "kidneys"}


def stage_organs(args: argparse.Namespace) -> None:
    from scipy.stats import spearmanr
    out = args.outputs
    od = out / "evaluation" / "organs"
    od.mkdir(parents=True, exist_ok=True)
    apath = out / "organs" / "organ_assignments.csv"
    if not apath.exists():
        sys.exit("ERROR: organs/organ_assignments.csv not found. Run backend/organ_pathways.py first.")
    org = pd.read_csv(apath).set_index(ID)
    patients = pd.read_csv(out / "clean" / "patients.csv")
    thresholds = load_thresholds(out)
    dev = load_split(out, "dev", True)
    df = dev if args.split == "dev" else load_split(out, "test", args.final)
    sfx = "_test" if args.split == "test" else ""

    def relabel(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        lab = d[ID].map(org["organ_group"]).fillna("Unattributable")
        d["pattern"] = np.where(d["septic_patient"].to_numpy(bool), lab, "none")
        return d

    df_vital = df.copy()                      # keeps the vital-sign pattern labels for the by-hospital table
    dev, df = relabel(dev), relabel(df)
    experts = sorted(c for c in df.columns if c.startswith("p_lgbm_organ_expert_"))
    extra = [c for c in ["p_lgbm_organ_moe_max"] if c in df.columns]
    main = [m for m in ORGAN_MAIN if m in df.columns]
    models = main + extra + experts

    # thresholds for the organ-expert columns (chosen on dev, like every other model; merged into thresholds.json)
    new = [m for m in extra + experts if m not in thresholds]
    if new:
        if args.split == "test":
            sys.exit(f"ERROR: no dev thresholds for {new}. Run `organs` on dev first.")
        co_dev = Cohort(dev, patients, new, args.horizon, args.refractory)
        for m in new:
            thresholds[m] = curves_and_thresholds(co_dev, m, args.grid)[1]
        th_path = out / "evaluation" / "thresholds.json"
        json.dump({**json.load(open(th_path)), **{m: thresholds[m] for m in new}}, open(th_path, "w"), indent=2)
        print(f"[thresholds] added {len(new)} organ-expert models to thresholds.json")

    co = Cohort(df, patients, models, args.horizon, args.refractory)
    rng = np.random.default_rng(args.seed)
    groups = [g for g in ORGANS_ORDER if g in set(co.pattern[co.septic])]

    # 1. alert results per organ group, main models, every operating point
    def by_group(c_, grps, label):
        out_rows = []
        hs = ["all"] + sorted(set(c_.hospital))
        for m in main:
            for op_, tau in thresholds[m].items():
                res_ = c_.at(c_.scores[m], tau)
                for hh in hs:
                    hm = np.ones(c_.m, bool) if hh == "all" else c_.hospital == hh
                    for g in ["all septic"] + grps:
                        mask = c_.septic & hm & (True if g == "all septic" else c_.pattern == g)
                        if not mask.any():
                            continue
                        gm = group_metrics(c_, res_, mask, args.bootstraps, rng, with_far=False, with_util=False)
                        out_rows.append({"model": m, "operating_point": op_, "hospital": hh, "group": g, **gm})
        print(f"[{label}] {len(grps)} groups x {len(hs)} hospital views x {len(main)} models")
        return out_rows

    rows = by_group(co, groups, "1/5 organ groups")
    pd.DataFrame(rows).to_csv(od / f"organ_alert_metrics{sfx}.csv", index=False)
    co_v = Cohort(df_vital, patients, main, args.horizon, args.refractory)
    vgroups = [g for g in ["Hemodynamic", "Respiratory", "Inflammatory", "Multi-system", "No abnormal vitals", "unassigned"]
               if g in set(co_v.pattern[co_v.septic])]
    pd.DataFrame(by_group(co_v, vgroups, "2/5 vital-sign patterns")).to_csv(
        od / f"pattern_alert_metrics_by_hospital{sfx}.csv", index=False)

    # 2. pattern-aware (organ experts / oracle) vs LightGBM, paired
    if experts:
        op = args.operating_point
        res = {m: co.at(co.scores[m], thresholds[m][op]) for m in models}
        cmp_ = compare_table(co, res, ["p_lgbm"] + extra + experts, "p_lgbm", "p_lgbm_organ_expert_",
                             args.bootstraps_compare, rng)
        cmp_.to_csv(od / f"organ_model_comparison{sfx}.csv", index=False)
        print("[3/5] organ experts and oracle vs LightGBM")

    # 3. measurement frequency of each organ's key test vs warning (single-organ groups, main model, far5)
    m0 = args.model if args.model in main else main[0]
    res = co.at(co.scores[m0], thresholds[m0][args.operating_point])
    sp = pd.DataFrame({ID: co.pids, "group": co.pattern, "det": res["det_timely"], "lead": res["lead_any"]})[co.septic]
    sp = sp.join(org, on=ID)
    fr = []
    for g, key in ORGAN_TRIGGER.items():
        d = sp[sp["group"] == g]
        if not len(d):
            continue
        k, n = int(d["det"].sum()), len(d)
        lo, hi = wilson(k, n)
        det = d[d["lead"].notna() & d[f"hours_since_{key}"].notna()]
        rho = spearmanr(det[f"hours_since_{key}"], det["lead"]) if len(det) >= 10 else None
        fr.append({"organ": g, "key_measurement": {"lungs": "SpO2 / FiO2", "clotting": "platelets", "liver": "bilirubin",
                                                   "circulation": "blood pressure (MAP)", "kidneys": "creatinine"}[key],
                   "n_septic": n, "sens_timely_%": 100 * k / n, "lo": lo, "hi": hi,
                   "median_measurements_24h": float(d[f"n_obs24_{key}"].median()),
                   "median_hours_since_measured": float(d[f"hours_since_{key}"].median()),
                   "median_lead_any_h": float(d["lead"].median()) if d["lead"].notna().any() else np.nan,
                   "spearman_staleness_vs_lead": float(rho.statistic) if rho is not None else np.nan,
                   "spearman_p": float(rho.pvalue) if rho is not None else np.nan})
    pd.DataFrame(fr).to_csv(od / f"organ_frequency{sfx}.csv", index=False)
    print(f"[4/5] measurement frequency per organ ({m0})")

    # 4. time to first alert in the last 24 h before switch-on, per organ group (Kaplan-Meier + log-rank)
    above = co.scores[m0] >= thresholds[m0][args.operating_point]
    ws_row = np.maximum(0, co.F_row - 24)
    first = np.minimum.reduceat(np.where(above & co.sept_row & (co.hour >= ws_row), co.hour, BIG), co.start)
    si = np.flatnonzero(co.septic)
    ws = np.maximum(0, co.F_int[si] - 24)
    ev = (first[si] < BIG).astype(int)
    tt = np.where(ev == 1, first[si] - ws, co.F_int[si] - ws).astype(float)
    pat = co.pattern[si]
    km = []
    for g in ["all septic"] + groups:
        s_ = np.ones(len(si), bool) if g == "all septic" else pat == g
        km += [{"group": g, "hours_since_window_start": a, "share_not_yet_alerted": b, "n_at_risk": n}
               for a, b, n in kaplan_meier(tt[s_], ev[s_])]
    pd.DataFrame(km).to_csv(od / f"organ_km{sfx}.csv", index=False)
    big = np.isin(pat, [g for g in groups if (pat == g).sum() >= 10])
    test = {}
    if len(set(pat[big])) >= 2:
        chi, dfree, pv = logrank(tt[big], ev[big], pat[big])
        test = {"chi2": chi, "df": dfree, "p_value": pv, "groups": sorted(set(pat[big]))}
    json.dump({"model": m0, "operating_point": args.operating_point, "logrank": test, "split": args.split},
              open(od / f"organ_report{sfx}.json", "w"), indent=2)
    print(f"[5/5] time to first alert per organ group" + (f" (log-rank p = {test['p_value']:.2g})" if test else ""))

    t = pd.DataFrame(rows)
    t = t[(t["model"] == m0) & (t["operating_point"] == args.operating_point) & (t["hospital"] == "all")]
    print(t[["group", "n_septic", "sens_timely", "sens_timely_lo", "sens_timely_hi", "median_lead_any"]].round(3)
          .to_string(index=False))


ORGANS_ORDER = ["Lungs", "Clotting", "Liver", "Circulation", "Kidneys", "Multi-organ", "Unattributable"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("performance", help="alert-level performance by model, hospital and pattern")
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--final", action="store_true", help="required for --split test")
    p.add_argument("--models", nargs="+", default=None, help="restrict to these p_* columns")
    p.add_argument("--horizon", type=int, default=12, help="hours before F that count as a timely alert")
    p.add_argument("--refractory", type=int, default=12, help="hours below tau before a new alert episode")
    p.add_argument("--bootstraps", type=int, default=200)
    p.add_argument("--grid", type=int, default=300, help="number of candidate thresholds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-saved-thresholds", action="store_true")
    p.add_argument("--official-check", type=Path, default=None,
                   help="path to the official evaluate_sepsis_score.py (optional extra validation)")
    t = sub.add_parser("timeprofile", help="warning rate by when the label switches on / hour of stay (dev)")
    t.add_argument("--outputs", type=Path, required=True)
    t.add_argument("--models", nargs="+", default=None)
    t.add_argument("--horizon", type=int, default=12)
    t.add_argument("--refractory", type=int, default=12)

    def common(x, split=False):
        x.add_argument("--outputs", type=Path, required=True)
        x.add_argument("--horizon", type=int, default=12)
        x.add_argument("--refractory", type=int, default=12)
        x.add_argument("--seed", type=int, default=42)
        if split:
            x.add_argument("--split", choices=["dev", "test"], default="dev")
            x.add_argument("--final", action="store_true", help="required for --split test")

    tc = sub.add_parser("timecontrol", help="within-hour discrimination + time-adjusted scores (*_tadj)")
    common(tc)
    tc.add_argument("--models", nargs="+", default=None)
    tc.add_argument("--min-ref", type=int, default=500, help="min reference rows per hour group")
    tc.add_argument("--no-adjust", action="store_true", help="only the table; do not add *_tadj columns")

    tr = sub.add_parser("transfer", help="hospital transfer at alert level (thresholds from the source)")
    common(tr, split=True)
    tr.add_argument("--variant", default="all", help="must match the variant used in 05 transfer")
    tr.add_argument("--bootstraps", type=int, default=200)
    tr.add_argument("--grid", type=int, default=300)

    fq = sub.add_parser("frequency", help="does measurement frequency explain pattern differences? (dev)")
    common(fq)
    fq.add_argument("--model", default="p_lgbm")
    fq.add_argument("--operating-point", default="far5")
    fq.add_argument("--bootstraps", type=int, default=200)
    fq.add_argument("--exclude-unassigned", action="store_true", help="drop 'unassigned' from the regression")
    fq.add_argument("--min-group", type=int, default=30, help="patterns with fewer patients are left out of the regression")

    sh = sub.add_parser("shap", help="SHAP values of the LightGBM baseline by pattern group (dev, out-of-fold)")
    sh.add_argument("--outputs", type=Path, required=True)
    sh.add_argument("--max-rows", type=int, default=3000, help="rows explained per group")
    sh.add_argument("--seed", type=int, default=42)

    cp = sub.add_parser("compare", help="global vs pattern-aware models per pattern (paired bootstrap)")
    common(cp, split=True)
    cp.add_argument("--reference", default="p_lgbm")
    cp.add_argument("--operating-point", default="far5")
    cp.add_argument("--bootstraps", type=int, default=1000)

    pl = sub.add_parser("plots", help="data for the Result figures tab: ROC/PR, confusion matrix, calibration, "
                                      "Kaplan-Meier, correlation, distributions, missing data (dev)")
    common(pl)
    pl.add_argument("--models", nargs="+", default=None, help="models for the curves (default: the main models found)")
    pl.add_argument("--model", default="p_lgbm", help="model for the Kaplan-Meier curves")
    pl.add_argument("--calib-bins", type=int, default=10)
    pl.add_argument("--km-window", type=int, default=24, help="hours before switch-on in which time to alert is counted")
    pl.add_argument("--corr-rows", type=int, default=200_000, help="hours sampled for the correlation matrix")
    pl.add_argument("--corr-min-pairs", type=int, default=200, help="min overlapping measurements per correlation")

    og = sub.add_parser("organs", help="results per organ group (needs backend/organ_pathways.py)")
    common(og, split=True)
    og.add_argument("--model", default="p_lgbm", help="model for the frequency and time-to-alert analyses")
    og.add_argument("--operating-point", default="far5")
    og.add_argument("--bootstraps", type=int, default=200)
    og.add_argument("--bootstraps-compare", type=int, default=1000)
    og.add_argument("--grid", type=int, default=300)

    args = ap.parse_args()
    {"organs": stage_organs, "plots": stage_plots, "performance": stage_performance, "timeprofile": stage_timeprofile, "timecontrol": stage_timecontrol,
     "transfer": stage_transfer, "frequency": stage_frequency, "shap": stage_shap,
     "compare": stage_compare}[args.stage](args)


if __name__ == "__main__":
    main()