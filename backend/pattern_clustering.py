#!/usr/bin/env python3
"""
04_pattern_clustering.py
========================
Idea B: deterioration-pattern-specific sepsis early warning (PhysioNet/CinC 2019).

Groups septic patients into "deterioration patterns" from the vital-sign change features
built in 03_pattern_features.py, and tests whether those patterns are real enough to
build the rest of the project on.

Methods (--method)
------------------
rules   (default) deterministic clinical-threshold groups, NO fitting. This is the primary
        stratification used by the rest of the project (see "What the data showed" below).
        Thresholds are common SIRS/qSOFA-style cut-offs, not tuned:
            Hemodynamic   MAP_last < 70
            Respiratory   O2Sat_last < 92 or Resp_last > 22
            Inflammatory  Temp_last > 38 or < 36, or HR_last > 100
        exactly one flag -> that group; two or more -> "Multi-system"; none -> "No abnormal vitals".
gmm     Gaussian mixture on scaled change features (exploratory discovery of patterns)
kmeans  K-means on the same features (a robustness comparison)

What the data showed (PhysioNet 2019, 1,538 dev patients with a usable window)
------------------------------------------------------------------------------
Data-driven clustering did NOT find multiple stable deterioration patterns: only k = 2
was ever stable, k >= 3 collapsed toward the null baseline, silhouette was <= 0.18, and the
k = 2 split flipped between "pass" and "fail" across reasonable settings (k-means, diagonal
covariance, dropping Temp). Change-based clusters were also confounded with window length
(early-onset patients). Report that as a finding; use the rule-based groups, which are
defined in advance, as the stratification, and describe them as clinically defined groups,
not discovered subtypes.

Leakage control
---------------
All fitting (outlier bounds, scaling, clustering) uses split == "dev" patients only.
Test patients are assigned to the nearest dev cluster afterwards.

Is the clustering real? (gmm / kmeans only)
-------------------------------------------
For every k in [k_min, k_max] the script reports:
  * size of the smallest cluster (dev patients)
  * bootstrap stability: refit on resamples, compare with the reference labels
    (adjusted Rand index, ARI; 1 = identical, 0 = chance)
  * hospital replicability: fit on hospital A only and label B (and the reverse), ARI
    against the reference labels
  * a NULL baseline: the same bootstrap stability on synthetic data drawn from ONE
    Gaussian with the same mean and covariance (no cluster structure). If real data are not
    clearly more stable than this, the "clusters" are just slices of a continuum.
  * BIC (gmm) and silhouette
A k is "qualifying" when min cluster size >= --min-cluster-size and median bootstrap ARI
>= --stab-thr. Among qualifying k the lowest BIC (gmm) or best silhouette (kmeans) is
chosen. The overall verdict "feasible" additionally requires hospital ARI >= --hosp-thr,
real bootstrap ARI >= null ARI + --null-margin, silhouette >= --min-silhouette (below about
0.25 the usual rule of thumb says "no substantial structure"), and clusters that are NOT
confounded with window length (max |standardised mean difference| of window_hours between a
cluster and the rest < --max-window-smd). The default thresholds are my rules of thumb, not
published standards; they are printed and saved so you can defend or change them.
If nothing qualifies the script still writes outputs (best-stability k) but sets
feasible = false and says so loudly. Use --method rules, or revisit the features, then.

Usage
-----
    # primary stratification used by later steps (rule-based):
    python 04_pattern_clustering.py --features outputs/patterns/pattern_features.parquet \
        --outdir outputs/patterns
    # exploratory clustering (separate folder so it does not overwrite the primary files):
    python 04_pattern_clustering.py --features outputs/patterns/pattern_features.parquet \
        --outdir outputs/patterns/exploratory_gmm --method gmm
    python 04_pattern_clustering.py ... --method gmm --k 4 --feature-set full

Outputs (in --outdir)
---------------------
    pattern_assignments.csv  one row per septic patient: pattern, pattern_id, pattern_prob, reason
    cluster_selection.csv    metrics for every k tried
    cluster_profiles.csv     per-pattern size, hospital mix, mean features, descriptor labs
    cluster_report.json      parameters, chosen k, feasibility checks, verdict
    cluster_model.joblib     fitted preprocessing + model (gmm / kmeans only)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

ID = "Patient_ID"
DEFAULT_VITALS = ["HR", "Resp", "MAP", "O2Sat", "Temp"]
FEATURE_SETS = {
    "delta_last": ["delta", "last"],              # how it changed + where it ended (default)
    "change": ["delta", "slope"],                 # change only
    "full": ["delta", "last", "slope", "std"],    # everything
}
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Preprocessing and model helpers
# --------------------------------------------------------------------------- #

class Prep:
    """Winsorise at the 1st/99th percentile, then standardise. Fit on dev only."""

    def fit(self, X: np.ndarray) -> "Prep":
        self.lo = np.nanpercentile(X, 1, axis=0)
        self.hi = np.nanpercentile(X, 99, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = Xc.mean(axis=0)
        self.sd = Xc.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.clip(X, self.lo, self.hi) - self.mu) / self.sd


def fit_model(X, k, method, cov, seed, n_init):
    if method == "gmm":
        return GaussianMixture(n_components=k, covariance_type=cov, n_init=n_init,
                               reg_covar=1e-4, random_state=seed).fit(X)
    return KMeans(n_clusters=k, n_init=max(n_init, 5), random_state=seed).fit(X)


def bootstrap_ari(X, k, method, cov, B, seed, ref_labels):
    rng = np.random.default_rng(seed)
    n = len(X)
    out = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        m = fit_model(X[idx], k, method, cov, seed + b + 1, n_init=2)
        out.append(adjusted_rand_score(ref_labels, m.predict(X)))
    return np.array(out)


def hospital_ari(X, hosp, ref_labels, k, method, cov, seed):
    vals = []
    for a, b in (("A", "B"), ("B", "A")):
        ma, mb = hosp == a, hosp == b
        if ma.sum() < k * 30 or mb.sum() < k * 30:
            vals.append(np.nan)
            continue
        m = fit_model(X[ma], k, method, cov, seed, n_init=3)
        vals.append(adjusted_rand_score(ref_labels[mb], m.predict(X[mb])))
    return vals


def evaluate_k(X, hosp, k, args, seed):
    ref = fit_model(X, k, args.method, args.covariance, seed, n_init=5)
    labels = ref.predict(X)
    sizes = np.bincount(labels, minlength=k)
    boot = bootstrap_ari(X, k, args.method, args.covariance, args.bootstraps, seed + 100, labels)

    # Null: one Gaussian, same mean/covariance, no cluster structure
    rng = np.random.default_rng(seed + 7)
    Xnull = rng.multivariate_normal(X.mean(axis=0), np.cov(X.T), size=len(X))
    nref = fit_model(Xnull, k, args.method, args.covariance, seed, n_init=3)
    nlab = nref.predict(Xnull)
    null = bootstrap_ari(Xnull, k, args.method, args.covariance, args.null_bootstraps, seed + 200, nlab)

    h = hospital_ari(X, hosp, labels, k, args.method, args.covariance, seed + 300)
    return {
        "k": k,
        "bic": float(ref.bic(X)) if args.method == "gmm" else np.nan,
        "silhouette": float(silhouette_score(X, labels)),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_share": round(float(sizes.max() / sizes.sum()), 3),
        "sizes": sizes.tolist(),
        "boot_ari_median": float(np.median(boot)),
        "boot_ari_p10": float(np.percentile(boot, 10)),
        "null_boot_ari_median": float(np.median(null)),
        "hospital_ari_A_to_B": h[0],
        "hospital_ari_B_to_A": h[1],
        "hospital_ari_mean": float(np.nanmean(h)) if not np.all(np.isnan(h)) else np.nan,
    }


def choose_k(tbl: pd.DataFrame, args) -> tuple[int, bool]:
    """Return (k, qualifying). Forced --k is honoured but still checked."""
    qual = tbl[(tbl["min_cluster_size"] >= args.min_cluster_size) &
               (tbl["boot_ari_median"] >= args.stab_thr)]
    if args.k is not None:
        return args.k, bool(args.k in qual["k"].values)
    if len(qual):
        if args.method == "gmm":
            return int(qual.sort_values("bic").iloc[0]["k"]), True
        return int(qual.sort_values("silhouette", ascending=False).iloc[0]["k"]), True
    return int(tbl.sort_values("boot_ari_median", ascending=False).iloc[0]["k"]), False


# --------------------------------------------------------------------------- #
# Rule-based patterns
# --------------------------------------------------------------------------- #

def rule_patterns(df: pd.DataFrame) -> pd.Series:
    need = ["MAP_last", "O2Sat_last", "Resp_last", "Temp_last", "HR_last"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"ERROR: --method rules needs columns {miss}; keep all five default vitals.")
    hemo = df["MAP_last"] < 70
    resp = (df["O2Sat_last"] < 92) | (df["Resp_last"] > 22)
    infl = (df["Temp_last"] > 38) | (df["Temp_last"] < 36) | (df["HR_last"] > 100)
    n = hemo.astype(int) + resp.astype(int) + infl.astype(int)
    out = pd.Series("No abnormal vitals", index=df.index)
    out[(n == 1) & hemo] = "Hemodynamic"
    out[(n == 1) & resp] = "Respiratory"
    out[(n == 1) & infl] = "Inflammatory"
    out[n >= 2] = "Multi-system"
    return out


# --------------------------------------------------------------------------- #
# Labelling and profiles
# --------------------------------------------------------------------------- #

def describe_clusters(Xz: np.ndarray, labels: np.ndarray, cols: list[str], k: int) -> dict[int, str]:
    """Short ASCII description from the cluster's mean z-scores (descriptive only)."""
    desc = {}
    for c in range(k):
        z = Xz[labels == c].mean(axis=0)
        top = np.argsort(-np.abs(z))[:2]
        parts = [f"{cols[i]} {'high' if z[i] > 0 else 'low'}" for i in top if abs(z[i]) >= 0.4]
        desc[c] = ", ".join(parts) if parts else "near average"
    return desc


def profile_table(assign: pd.DataFrame, feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = assign.merge(feat[[ID] + [c for c in feat.columns if c not in assign.columns]], on=ID)
    d = d[d["pattern_id"] >= 0]
    rows = []
    for (pid, name), g in d.groupby(["pattern_id", "pattern"]):
        rec = {
            "pattern": name, "pattern_id": pid,
            "description": g["pattern_desc"].iloc[0],
            "n_dev": int((g["split"] == "dev").sum()), "n_test": int((g["split"] == "test").sum()),
            "n_hospital_A": int((g["hospital"] == "A").sum()), "n_hospital_B": int((g["hospital"] == "B").sum()),
            "median_first_pos_hour": float(g["first_pos_hour"].median()),
            "mean_n_lab_hours_window": round(float(g["n_lab_hours_window"].mean()), 2),
        }
        for c in cols:
            rec[f"mean_{c}"] = round(float(g[c].mean()), 3)
        for c in [c for c in feat.columns if c.startswith("desc_") and c.endswith("_last")]:
            rec[f"mean_{c}"] = round(float(g[c].mean()), 3) if g[c].notna().any() else np.nan
            rec[f"share_measured_{c[5:-5]}"] = round(float((g[c.replace('_last', '_n_obs')] >= 1).mean()), 3)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("pattern_id")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, type=Path, help="pattern_features.parquet / .csv.gz from step 03")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--method", choices=["gmm", "kmeans", "rules"], default="rules")
    ap.add_argument("--covariance", choices=["full", "diag", "tied", "spherical"], default="full")
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="delta_last")
    ap.add_argument("--vitals", nargs="+", default=DEFAULT_VITALS)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--k", type=int, default=None, help="force k instead of choosing it")
    ap.add_argument("--bootstraps", type=int, default=50)
    ap.add_argument("--null-bootstraps", type=int, default=20)
    ap.add_argument("--min-cluster-size", type=int, default=150, help="min dev patients in smallest cluster")
    ap.add_argument("--stab-thr", type=float, default=0.6, help="min median bootstrap ARI")
    ap.add_argument("--hosp-thr", type=float, default=0.4, help="min mean hospital cross-ARI")
    ap.add_argument("--null-margin", type=float, default=0.15, help="real ARI must beat null ARI by this much")
    ap.add_argument("--min-silhouette", type=float, default=0.25, help="min silhouette of the chosen k")
    ap.add_argument("--max-window-smd", type=float, default=0.5,
                    help="max |SMD| of window_hours between a cluster and the rest")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feat = pd.read_parquet(args.features) if args.features.suffix == ".parquet" else pd.read_csv(args.features)
    cols = [f"{v}_{s}" for v in args.vitals for s in FEATURE_SETS[args.feature_set]]
    missing = [c for c in cols if c not in feat.columns]
    if missing and args.method != "rules":
        sys.exit(f"ERROR: feature columns missing: {missing}. Re-run 03 with these vitals.")
    print(f"[load] {len(feat)} septic patients; eligible for patterns: {int(feat['pattern_eligible'].sum())}")

    # ---- who can be clustered ----
    feat["assign_reason"] = feat["ineligible_reason"].fillna("")
    use_cols = cols if args.method != "rules" else ["MAP_last", "O2Sat_last", "Resp_last", "Temp_last", "HR_last"]
    nan_rows = feat["pattern_eligible"] & feat[use_cols].isna().any(axis=1)
    feat.loc[nan_rows, "assign_reason"] = "missing_feature"
    can = feat["pattern_eligible"] & ~nan_rows
    if nan_rows.any():
        print(f"[note] {int(nan_rows.sum())} eligible patients lack a required feature value and stay unassigned")
    dev_m = can & (feat["split"] == "dev")
    test_m = can & (feat["split"] == "test")
    print(f"[cohort] dev: {int(dev_m.sum())} | test: {int(test_m.sum())}")

    feat["pattern_id"] = -1
    feat["pattern"] = "unassigned"
    feat["pattern_desc"] = ""
    feat["pattern_prob"] = np.nan
    report: dict = {"parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    sel = pd.DataFrame()
    model_bundle = None

    if args.method == "rules":
        names = rule_patterns(feat.loc[can])
        order = names[dev_m[can]].value_counts().index.tolist()
        order += [n for n in names.unique() if n not in order]
        idmap = {n: i for i, n in enumerate(order)}
        feat.loc[can, "pattern"] = names
        feat.loc[can, "pattern_id"] = names.map(idmap).astype(int)
        feat.loc[can, "pattern_desc"] = names
        report["verdict"] = {"feasible": None, "note": "rules are deterministic; only group sizes are checked"}
        k_final = len(order)
    else:
        prep = Prep().fit(feat.loc[dev_m, cols].to_numpy(float))
        Xdev = prep.transform(feat.loc[dev_m, cols].to_numpy(float))
        hosp = feat.loc[dev_m, "hospital"].to_numpy()

        rows = []
        for k in range(args.k_min, args.k_max + 1):
            r = evaluate_k(Xdev, hosp, k, args, args.seed)
            rows.append(r)
            print(f"[k={k}] sizes={r['sizes']} bootARI={r['boot_ari_median']:.2f} "
                  f"(null {r['null_boot_ari_median']:.2f}) hospARI={r['hospital_ari_mean']:.2f} "
                  f"silh={r['silhouette']:.2f}" + (f" BIC={r['bic']:.0f}" if args.method == "gmm" else ""))
        sel = pd.DataFrame(rows)
        k_final, qualifying = choose_k(sel, args)
        row = sel[sel["k"] == k_final].iloc[0]

        checks = {
            "min_cluster_size_ok": bool(row["min_cluster_size"] >= args.min_cluster_size),
            "stability_ok": bool(row["boot_ari_median"] >= args.stab_thr),
            "hospital_replicable": bool(row["hospital_ari_mean"] >= args.hosp_thr) if not np.isnan(row["hospital_ari_mean"]) else False,
            "beats_null": bool(row["boot_ari_median"] - row["null_boot_ari_median"] >= args.null_margin),
            "silhouette_ok": bool(row["silhouette"] >= args.min_silhouette),
        }
        feasible = all(checks.values())
        report["verdict"] = {"chosen_k": k_final, "qualifying_k_exists": qualifying, "checks": checks, "feasible": feasible}

        # ---- final fit on all dev, assign dev + test ----
        model = fit_model(Xdev, k_final, args.method, args.covariance, args.seed, n_init=20)
        Xall = prep.transform(feat.loc[can, cols].to_numpy(float))
        lab = model.predict(Xall)
        dev_lab = model.predict(Xdev)
        order = np.argsort(-np.bincount(dev_lab, minlength=k_final))   # biggest dev cluster -> P1
        remap = {int(old): new for new, old in enumerate(order)}
        newlab = np.array([remap[int(x)] for x in lab])
        feat.loc[can, "pattern_id"] = newlab
        feat.loc[can, "pattern"] = [f"P{i + 1}" for i in newlab]
        if args.method == "gmm":
            proba = model.predict_proba(Xall)[:, order]
            feat.loc[can, "pattern_prob"] = proba.max(axis=1)
        dev_new = np.array([remap[int(x)] for x in dev_lab])
        wh = feat.loc[dev_m, "window_hours"].to_numpy(float)
        smds = []
        for c in range(k_final):
            a, b = wh[dev_new == c], wh[dev_new != c]
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            smds.append(abs(a.mean() - b.mean()) / pooled if pooled > 0 else 0.0)
        checks["not_confounded_by_window_length"] = bool(max(smds) < args.max_window_smd)
        report["verdict"]["window_hours_max_abs_smd"] = round(float(max(smds)), 3)
        report["verdict"]["checks"] = checks
        report["verdict"]["feasible"] = all(checks.values())
        desc = describe_clusters(Xdev, dev_new, cols, k_final)
        feat.loc[can, "pattern_desc"] = [desc[i] for i in newlab]
        model_bundle = {"prep": prep, "model": model, "order": order.tolist(), "cols": cols, "args": vars(args)}

    # ---- outputs ----
    args.outdir.mkdir(parents=True, exist_ok=True)
    keep = [ID, "hospital", "split", "fold", "first_pos_hour", "window_hours", "pattern_id", "pattern",
            "pattern_desc", "pattern_prob", "assign_reason"]
    feat[keep].to_csv(args.outdir / "pattern_assignments.csv", index=False)
    if len(sel):
        sel.assign(sizes=sel["sizes"].astype(str)).to_csv(args.outdir / "cluster_selection.csv", index=False)
    prof_cols = cols if args.method != "rules" else use_cols
    prof = profile_table(feat[keep], feat, prof_cols)
    prof.to_csv(args.outdir / "cluster_profiles.csv", index=False)
    if model_bundle is not None:
        joblib.dump(model_bundle, args.outdir / "cluster_model.joblib")
    elif args.method == "rules":
        # remove leftovers from an earlier clustering run in the same folder, so stale
        # files cannot be mistaken for results of this run
        for stale in ("cluster_selection.csv", "cluster_model.joblib"):
            if (args.outdir / stale).exists():
                (args.outdir / stale).unlink()
                print(f"[note] removed stale {stale} from an earlier clustering run")

    # ---- validation ----
    assigned = feat["pattern_id"] >= 0
    assert (assigned == can).all(), "assigned set differs from eligible set"
    assert set(feat.loc[assigned, "pattern_id"]) == set(range(k_final)), "pattern ids are not 0..k-1"
    assert (feat.loc[~assigned, "pattern"] == "unassigned").all()
    assert feat.loc[assigned, "split"].isin(["dev", "test"]).all()

    # ---- report ----
    dev_n = feat[assigned & (feat["split"] == "dev")].groupby("pattern").size()
    report["n_patterns"] = int(k_final)
    report["assigned"] = int(assigned.sum())
    report["unassigned"] = {"total": int((~assigned).sum()),
                            "by_reason": feat.loc[~assigned, "assign_reason"].value_counts().to_dict()}
    report["pattern_sizes_dev"] = dev_n.to_dict()
    report["pattern_sizes_test"] = feat[assigned & (feat["split"] == "test")].groupby("pattern").size().to_dict()
    report["smallest_test_pattern"] = int(min(report["pattern_sizes_test"].values())) if report["pattern_sizes_test"] else 0
    if args.method == "gmm":
        te = feat[assigned & (feat["split"] == "test")]
        report["test_low_confidence_share_prob_lt_0.6"] = round(float((te["pattern_prob"] < 0.6).mean()), 3)
        report["dev_low_confidence_share_prob_lt_0.6"] = round(float((feat[assigned & (feat["split"] == "dev")]["pattern_prob"] < 0.6).mean()), 3)
    with open(args.outdir / "cluster_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print("\n=== RESULT ===")
    print(f"patterns: {k_final} | assigned {report['assigned']} | unassigned {report['unassigned']['total']} {report['unassigned']['by_reason']}")
    show = [c for c in ["pattern", "description", "n_dev", "n_test", "n_hospital_A", "n_hospital_B", "median_first_pos_hour"] if c in prof.columns]
    print(prof[show].to_string(index=False))
    if "test_low_confidence_share_prob_lt_0.6" in report:
        print(f"low-confidence assignments (max posterior < 0.6): dev {report['dev_low_confidence_share_prob_lt_0.6']:.1%}, "
              f"test {report['test_low_confidence_share_prob_lt_0.6']:.1%}")
    v = report["verdict"]
    if v.get("feasible") is True:
        print("\nVERDICT: clusters pass all checks:", v["checks"])
    elif v.get("feasible") is False:
        print("\nVERDICT: *** clusters do NOT pass all checks ***", v["checks"])
        print("The patterns should not be presented as real subtypes. Consider --method rules,")
        print("a different --feature-set, or reporting this as a negative finding.")
    else:
        print("\nVERDICT:", v["note"])


if __name__ == "__main__":
    main()