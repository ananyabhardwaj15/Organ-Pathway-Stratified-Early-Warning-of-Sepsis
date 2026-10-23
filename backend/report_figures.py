#!/usr/bin/env python3
"""
report_figures.py
=================
Publication-quality figures (300 dpi PNG + vector PDF) and tables for the report and a paper.

    python backend/report_figures.py --outputs outputs

Reads only result files written by the pipeline; nothing is recomputed or retrained. Development (out-of-fold)
results, plus the final test table when `python run.py --final` has been run.

Outputs (in <outputs>/report/)
------------------------------
    figures/FigNN_<name>.png and .pdf     one file per figure, numbered in reading order
    tables/TableN_<name>.csv and .md      cohort characteristics, model performance, organ groups (+ test set)
    captions.md                           figure and table list with a caption for each
    report_package.zip                    everything above in one file
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, MUTED, LINE = "#16313A", "#5E747B", "#DCE5E7"
MODELS = {"p_lgbm": ("LightGBM", "#0E7C86"), "p_logreg": ("Logistic regression", "#7A6AAE"),
          "p_gru": ("GRU", "#D07A1F"), "p_ensemble": ("Ensemble", "#1F2937")}
ORGANS = {"Lungs": "#2F6FB0", "Clotting": "#8E5BA8", "Liver": "#A0661B", "Circulation": "#B23A48",
          "Kidneys": "#2E8B57", "Multi-organ": "#4A5A61", "Unattributable": "#B8C4C7"}
PATTERNS = {"Hemodynamic": "#B23A48", "Respiratory": "#2F6FB0", "Inflammatory": "#C98A0B", "Multi-system": "#6B4C9A",
            "No abnormal vitals": "#8A9A9E", "unassigned": "#B8C4C7"}
HOSP = {"A": "#0E7C86", "B": "#9FB3B8"}
OP = "far5"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


class Report:
    def __init__(self, out: Path):
        self.out = out
        self.fig_dir = out / "report" / "figures"
        self.tab_dir = out / "report" / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tab_dir.mkdir(parents=True, exist_ok=True)
        self.n_fig, self.n_tab, self.captions, self.files, self.skipped = 0, 0, [], [], []

    def csv(self, *parts):
        p = self.out.joinpath(*parts)
        return pd.read_csv(p) if p.exists() else None

    def js(self, *parts):
        p = self.out.joinpath(*parts)
        return json.load(open(p)) if p.exists() else None

    def save(self, fig, name: str, caption: str) -> None:
        self.n_fig += 1
        stem = f"Fig{self.n_fig:02d}_{name}"
        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"{stem}.{ext}")
            self.files.append(self.fig_dir / f"{stem}.{ext}")
        plt.close(fig)
        self.captions.append(f"**Figure {self.n_fig}.** {caption}  \n`figures/{stem}.png`, `.pdf`")
        print(f"[figure {self.n_fig:2d}] {stem}")

    def table(self, df: pd.DataFrame, name: str, caption: str) -> None:
        self.n_tab += 1
        stem = f"Table{self.n_tab}_{name}"
        df.to_csv(self.tab_dir / f"{stem}.csv", index=False)
        cols = [str(c) for c in df.columns]
        md = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
        md += "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.astype(str).to_numpy())
        (self.tab_dir / f"{stem}.md").write_text(f"**Table {self.n_tab}.** {caption}\n\n{md}")
        self.files += [self.tab_dir / f"{stem}.csv", self.tab_dir / f"{stem}.md"]
        self.captions.append(f"**Table {self.n_tab}.** {caption}  \n`tables/{stem}.csv`, `.md`")
        print(f"[table  {self.n_tab:2d}] {stem}")

    def skip(self, what: str) -> None:
        self.skipped.append(what)
        print(f"[skip] {what}: its result file is missing")


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def iqr(s: pd.Series, digits=0) -> str:
    s = s.dropna()
    if not len(s):
        return "n/a"
    q = s.quantile([0.25, 0.5, 0.75]).round(digits)
    f = (lambda v: f"{v:.{digits}f}")
    return f"{f(q[0.5])} ({f(q[0.25])} to {f(q[0.75])})"


def npct(mask: pd.Series) -> str:
    return f"{int(mask.sum()):,} ({100 * mask.mean():.1f}%)"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def table_cohort(R: Report) -> None:
    pt, fo = R.csv("clean", "patients.csv"), R.csv("splits", "folds.csv")
    if pt is None or fo is None:
        return R.skip("Table: cohort")
    d = pt.merge(fo[["Patient_ID", "split"]], on="Patient_ID")
    d = d[d["include_main"].astype(bool)]
    d["septic"] = d["septic"].astype(bool)
    cols = {"All": d}
    for h in sorted(d["hospital"].unique()):
        cols[f"Hospital {h}"] = d[d["hospital"] == h]
    rows = [("Patients, n", lambda x: f"{len(x):,}"),
            ("Development / test set, n", lambda x: f"{int((x['split'] == 'dev').sum()):,} / {int((x['split'] == 'test').sum()):,}"),
            ("Septic, n (%)", lambda x: npct(x["septic"])),
            ("Age, years, median (IQR)", lambda x: iqr(x["age"])),
            ("Male, n (%)", lambda x: npct(x["gender"] == 1)),
            ("Medical ICU, n (%)", lambda x: npct(x["unit"] == 1)),
            ("Surgical ICU, n (%)", lambda x: npct(x["unit"] == 2)),
            ("ICU unit not recorded, n (%)", lambda x: npct(x["unit"] == 0)),
            ("Recorded ICU stay, hours, median (IQR)", lambda x: iqr(x["n_hours"])),
            ("Hour of sepsis label in septic patients, median (IQR)", lambda x: iqr(x.loc[x["septic"], "first_pos_hour"]))]
    t = pd.DataFrame({"Characteristic": [r[0] for r in rows],
                      **{name: [f(sub) for _, f in rows] for name, sub in cols.items()}})
    excl = pt[~pt["include_main"].astype(bool)]
    R.table(t, "cohort_characteristics",
            f"Characteristics of the included patients, overall and by hospital (PhysioNet/CinC 2019). "
            f"{len(excl):,} patients were excluded because sepsis was already labelled at their first hour.")


def table_performance(R: Report, test: bool = False) -> None:
    sfx = "_test" if test else ""
    am = R.csv("evaluation", f"alert_metrics{sfx}.csv")
    if am is None:
        return R.skip(f"Table: performance{sfx}")
    cs = R.csv("evaluation", "figures", "curve_summary.csv")
    auc = cs.set_index("model") if cs is not None and not test else None
    rows = []
    for m, (name, _) in MODELS.items():
        for scope, gtype, grp in [("All", "all", "all"), ("Hospital A", "hospital", "A"), ("Hospital B", "hospital", "B")]:
            r = am[(am["model"] == m) & (am["operating_point"] == OP) & (am["group_type"] == gtype) & (am["group"] == grp)]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append({"Model": name, "Patients": scope,
                         "AUROC": f"{auc.loc[m, 'auroc']:.3f}" if auc is not None and m in auc.index and scope == "All" else "",
                         "AUPRC": f"{auc.loc[m, 'auprc']:.3f}" if auc is not None and m in auc.index and scope == "All" else "",
                         "Warned in time, % (95% CI)": f"{100 * r['sens_timely']:.1f} ({100 * r['sens_timely_lo']:.1f} to {100 * r['sens_timely_hi']:.1f})",
                         "False alarms per 100 patient-days": f"{r['far_per_100pd']:.2f}",
                         "Non-septic patients alerted, %": f"{100 * r['share_nonseptic_alerted']:.1f}",
                         "Median warning, h": f"{r['median_lead_any']:.0f}",
                         "Normalised utility": f"{r['util_norm']:.3f}" if pd.notna(r.get("util_norm")) else ""})
    if not rows:
        return R.skip(f"Table: performance{sfx}")
    which = "held-out TEST set (final, reported once)" if test else "development patients (out-of-fold)"
    R.table(pd.DataFrame(rows), f"model_performance{sfx}",
            f"Performance of the four models on {which}, at the balanced alert setting (thresholds chosen on development "
            "data for about 5 false alarms per 100 non-septic patient-days). Warned in time = an alert in the 12 hours "
            "before the sepsis label; AUROC and AUPRC are hour-level.")


def table_organs(R: Report) -> None:
    a, om, gate = R.csv("organs", "organ_assignments.csv"), R.csv("evaluation", "organs", "organ_alert_metrics.csv"), R.js("organs", "organ_gate.json")
    if a is None or om is None:
        return R.skip("Table: organ groups")
    rows = []
    for g in ORGANS:
        r = {"Organ group": g}
        for h in ["all"] + sorted(a["hospital"].unique()):
            sub = a if h == "all" else a[a["hospital"] == h]
            k = int((sub["organ_group"] == g).sum())
            r[f"{'All' if h == 'all' else 'Hospital ' + h}, n (%)"] = f"{k} ({100 * k / max(len(sub), 1):.1f}%)"
            x = om[(om["model"] == "p_lgbm") & (om["operating_point"] == OP) & (om["hospital"] == h) & (om["group"] == g)]
            if len(x):
                lo, hi = wilson(round(x.iloc[0]["sens_timely"] * x.iloc[0]["n_septic"]), x.iloc[0]["n_septic"])
                r[f"Warned in time, {'all' if h == 'all' else h}, % (95% CI)"] = (
                    f"{100 * x.iloc[0]['sens_timely']:.1f} ({100 * lo:.1f} to {100 * hi:.1f})")
        rows.append(r)
    t = pd.DataFrame(rows)
    gate_txt = (f" Attributable (not Unattributable): {100 * gate['overall_share']:.1f}% overall; gate (50% in each "
                f"hospital) {'passed' if gate['passes'] else 'not passed'}." if gate else "")
    R.table(t, "organ_groups", "Septic patients by organ group (partial SOFA, window 24 h before to 6 h after the sepsis "
            "label) and the share warned in time by LightGBM at the balanced setting, with Wilson 95% CIs." + gate_txt)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_flow(R: Report) -> None:
    pt, fo = R.csv("clean", "patients.csv"), R.csv("splits", "folds.csv")
    if pt is None or fo is None:
        return R.skip("Figure: patient flow")
    d = pt.merge(fo[["Patient_ID", "split"]], on="Patient_ID")
    inc = d[d["include_main"].astype(bool)]
    sep = lambda x: int(x["septic"].astype(bool).sum())
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    def box(x, y, w, h, text, color="#EAF4F4"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec=MUTED, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))

    box(3, 8.4, 4, 1.2, f"ICU stays in the dataset\nn = {len(d):,} ({sep(d):,} septic)")
    box(7.3, 6.5, 2.6, 1.2, f"Excluded: sepsis label\nat the first hour\nn = {len(d) - len(inc):,}", "#F7EDE8")
    box(3, 6.4, 4, 1.2, f"Included\nn = {len(inc):,} ({sep(inc):,} septic)")
    dev, test = inc[inc["split"] == "dev"], inc[inc["split"] == "test"]
    box(0.4, 3.9, 4.2, 1.5, f"Development set (80%)\nn = {len(dev):,} ({sep(dev):,} septic)\n5 folds: training, tuning,\nthresholds", "#E3F1F2")
    box(5.4, 3.9, 4.2, 1.5, f"Test set (20%)\nn = {len(test):,} ({sep(test):,} septic)\nlocked; used once at the end", "#EEF0F5")
    for h_, x0 in (("A", 0.4), ("B", 2.6)):
        dv = dev[dev["hospital"] == h_]
        box(x0, 1.6, 2.0, 1.3, f"Hospital {h_}\nn = {len(dv):,}\n({sep(dv):,} septic)")
    for h_, x0 in (("A", 5.4), ("B", 7.6)):
        tv = test[test["hospital"] == h_]
        box(x0, 1.6, 2.0, 1.3, f"Hospital {h_}\nn = {len(tv):,}\n({sep(tv):,} septic)")
    arrow(5, 8.4, 5, 7.6)
    arrow(7, 8.8, 8.5, 7.7)
    arrow(4.2, 6.4, 2.5, 5.4)
    arrow(5.8, 6.4, 7.5, 5.4)
    for x in (1.4, 3.6):
        arrow(2.5, 3.9, x, 2.9)
    for x in (6.4, 8.6):
        arrow(7.5, 3.9, x, 2.9)
    ax.set_title("Patient flow", loc="left")
    R.save(fig, "patient_flow", "Patient flow. Split by patient (never by hour), stratified by hospital and sepsis "
           "status; the test set is used once for the final evaluation.")


def fig_roc_pr(R: Report) -> None:
    roc, pr, cs = R.csv("evaluation", "figures", "roc_curves.csv"), R.csv("evaluation", "figures", "pr_curves.csv"), R.csv("evaluation", "figures", "curve_summary.csv")
    if roc is None:
        return R.skip("Figure: ROC / PR")
    auc = cs.set_index("model")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for m, (name, col) in MODELS.items():
        if m in set(roc["model"]):
            r = roc[roc["model"] == m]
            ax[0].plot(r["fpr"], r["tpr"], color=col, lw=1.6, label=f"{name} (AUROC {auc.loc[m, 'auroc']:.3f})",
                       ls="--" if m == "p_ensemble" else "-")
            q = pr[pr["model"] == m]
            ax[1].plot(q["recall"], q["precision"], color=col, lw=1.6, label=f"{name} (AUPRC {auc.loc[m, 'auprc']:.3f})",
                       ls="--" if m == "p_ensemble" else "-")
    ax[0].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax[1].axhline(float(auc["prevalence"].iloc[0]), color=MUTED, lw=0.8, ls=":")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    ax[0].set_title("A  ROC curve", loc="left")
    ax[1].set_title("B  Precision-recall curve", loc="left")
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower left")
    R.save(fig, "roc_pr", f"Hour-level discrimination on development patients. A positive hour lies within 12 h before "
           f"the sepsis label ({100 * auc['prevalence'].iloc[0]:.1f}% of hours). Dotted: chance.")


def fig_confusion(R: Report) -> None:
    cm = R.csv("evaluation", "figures", "confusion_matrix.csv")
    if cm is None:
        return R.skip("Figure: confusion matrices")
    ms = [m for m in MODELS if len(cm[(cm["model"] == m) & (cm["operating_point"] == OP)])]
    fig, axes = plt.subplots(1, len(ms), figsize=(1.95 * len(ms) + 0.4, 2.4))
    axes = np.atleast_1d(axes)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cm", ["#F7FAFA", "#7CC4CA"])
    for n_, (ax, m) in enumerate(zip(axes, ms)):
        c = cm[(cm["model"] == m) & (cm["operating_point"] == OP)].iloc[0]
        z = np.array([[c.tp, c.fn], [c.fp, c.tn]], float)
        share = z / z.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(z):
            ax.text(j, i, f"{int(v):,}\n{100 * share[i, j]:.0f}%", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks([0, 1], ["alert", "no alert"])
        ax.set_yticks([0, 1], ["septic", "not septic"] if n_ == 0 else ["", ""])
        ax.grid(False)
        ax.set_title(MODELS[m][0], loc="left")
    R.save(fig, "confusion_matrices", "Patient-level confusion matrices at the balanced setting. Septic: warned in the "
           "12 h before the label or missed; not septic: at least one false alarm or none. Percentages are per row.")


def fig_tradeoff(R: Report) -> None:
    uc = R.csv("evaluation", "utility_curve.csv")
    if uc is None:
        return R.skip("Figure: sensitivity vs false alarms")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for m, (name, col) in MODELS.items():
        u = uc[(uc["model"] == m) & (uc["far"] <= 30)].sort_values("far")
        if len(u):
            ax.plot(u["far"], 100 * u["sens_timely"], color=col, lw=1.6, label=name, ls="--" if m == "p_ensemble" else "-")
    for x in (2, 5, 10):
        ax.axvline(x, color=MUTED, lw=0.6, ls=":")
    ax.set(xlabel="False alarms per 100 non-septic patient-days", ylabel="Septic patients warned in time (%)",
           xlim=(0, 30), ylim=(0, 101))
    ax.set_title("Warning rate against alarm burden", loc="left")
    ax.legend(loc="lower right")
    R.save(fig, "warning_vs_false_alarms", "Share of septic patients warned in time as the alert threshold varies. "
           "Dotted lines: the strict, balanced and sensitive settings (2, 5 and 10 false alarms per 100 patient-days).")


def fig_by_hospital(R: Report) -> None:
    am = R.csv("evaluation", "alert_metrics.csv")
    if am is None:
        return R.skip("Figure: by hospital")
    d = am[(am["operating_point"] == OP) & (am["group_type"] == "hospital") & am["model"].isin(MODELS)]
    if not len(d):
        return R.skip("Figure: by hospital")
    ms = [m for m in MODELS if m in set(d["model"])]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
    x = np.arange(len(ms))
    for i, h in enumerate(sorted(d["group"].unique())):
        s = d[d["group"] == h].set_index("model").loc[ms]
        ax[0].bar(x + (i - 0.5) * 0.38, 100 * s["sens_timely"], 0.36, color=list(HOSP.values())[i], label=f"Hospital {h}",
                  yerr=[np.clip(100 * (s["sens_timely"] - s["sens_timely_lo"]), 0, None),
                        np.clip(100 * (s["sens_timely_hi"] - s["sens_timely"]), 0, None)],
                  error_kw=dict(lw=0.8, ecolor=INK, capsize=2))
        ax[1].bar(x + (i - 0.5) * 0.38, s["far_per_100pd"], 0.36, color=list(HOSP.values())[i], label=f"Hospital {h}")
    for a, t, yl in ((ax[0], "A  Septic patients warned in time", "%"), (ax[1], "B  False alarms per 100 patient-days", "")):
        a.set_xticks(x, [MODELS[m][0].replace("Logistic regression", "Logistic\nregression") for m in ms])
        a.set_title(t, loc="left")
        a.set_ylabel(yl)
    ax[0].set_ylim(0, 105)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    R.save(fig, "performance_by_hospital", "Results per hospital with one shared threshold per model (balanced "
           "setting). Error bars: 95% patient-bootstrap CIs.")


def fig_calibration(R: Report) -> None:
    cal = R.csv("evaluation", "figures", "calibration.csv")
    if cal is None:
        return R.skip("Figure: calibration")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    lo = 1.0
    for m, (name, col) in MODELS.items():
        c = cal[(cal["model"] == m) & (cal["group"] == "all")]
        if len(c):
            ax.plot(c["mean_pred"], c["observed"], "o-", color=col, lw=1.4, ms=3.5, label=name)
            lo = min(lo, float(c[["mean_pred", "observed"]].replace(0, np.nan).min().min()))
    ax.plot([lo, 1], [lo, 1], color=MUTED, ls=":", lw=0.9)
    ax.set(xscale="log", yscale="log", xlabel="Mean predicted risk", ylabel="Observed share of positive hours")
    ax.set_title("Calibration", loc="left")
    ax.legend(loc="upper left")
    R.save(fig, "calibration", "Calibration in ten equal-size bins of predicted risk (log scales). Dotted: perfect "
           "calibration. The ensemble is omitted because its score is a percentile, not a probability; the GRU is "
           "trained with extra weight on positive hours and over-predicts by design.")


def _ab_groups(d: pd.DataFrame, order: list[str], title: str):
    """One panel: for every group, hospital A and B bars side by side (Wilson 95% CIs), n in the labels."""
    hs = [h for h in ["A", "B"] if h in set(d["hospital"])]
    groups = [g for g in order if g in set(d["group"])]
    fig, ax = plt.subplots(figsize=(5.6, 0.62 * len(groups) + 1.2))
    y = np.arange(len(groups))[::-1]
    w = 0.38
    for i, h in enumerate(hs):
        s_ = d[d["hospital"] == h].set_index("group").reindex(groups)
        n = s_["n_septic"].fillna(0).to_numpy()
        p_ = s_["sens_timely"].to_numpy(float)
        lo, hi = wilson(np.round(p_ * n), n)
        off = (0.5 - i) * w
        ax.barh(y + off, 100 * p_, w * 0.95, color=list(HOSP.values())[i], label=f"Hospital {h}",
                xerr=[np.clip(100 * (p_ - lo), 0, None), np.clip(100 * (hi - p_), 0, None)],
                error_kw=dict(lw=0.8, ecolor=INK, capsize=2))
        for yy, v, nn in zip(y + off, p_, n):
            if np.isfinite(v):
                ax.text(1.5, yy, f"{100 * v:.0f}%  (n={int(nn)})", va="center", fontsize=7,
                        color="white" if i == 0 else INK)
    ax.set_yticks(y, groups)
    ax.set(xlim=(0, 105), xlabel="Septic patients warned in time (%)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12 - 0.3 / len(groups)))
    return fig


def fig_organs(R: Report) -> None:
    a = R.csv("organs", "organ_assignments.csv")
    om = R.csv("evaluation", "organs", "organ_alert_metrics.csv")
    if a is None or om is None:
        return R.skip("Figures: organs")
    gate = R.js("organs", "organ_gate.json")
    # counts by hospital
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    order = [g for g in ORGANS if g in set(a["organ_group"])]
    hs = sorted(a["hospital"].unique())
    left = np.zeros(len(order))
    for h in hs:
        v = np.array([((a["organ_group"] == g) & (a["hospital"] == h)).sum() for g in order])
        ax.barh(np.arange(len(order))[::-1], v, left=left, color=HOSP[h], label=f"Hospital {h}")
        left += v
    ax.set_yticks(np.arange(len(order))[::-1], order)
    ax.set(xlabel="Septic patients")
    ax.set_title("Organ groups of septic patients", loc="left")
    ax.legend(loc="upper right")
    R.save(fig, "organ_groups", "Septic patients by organ group (partial SOFA: lungs from SpO2/FiO2, clotting from "
           "platelets, liver from bilirubin, circulation from MAP, kidneys from creatinine)." +
           (f" Attributable: {100 * gate['overall_share']:.1f}%." if gate else ""))
    # warned in time by organ, A vs B
    d = om[(om["model"] == "p_lgbm") & (om["operating_point"] == OP) & (om["group"] != "all septic")]
    fig = _ab_groups(d, list(ORGANS), "Warned in time by organ group: hospital A vs B")
    R.save(fig, "warned_by_organ_hospital", "Septic patients warned in time by organ group in hospital A and hospital B, "
           "LightGBM at the balanced setting (one shared threshold). Error bars: Wilson 95% CIs.")
    km = R.csv("evaluation", "organs", "organ_km.csv")
    if km is not None:
        rep = R.js("evaluation", "organs", "organ_report.json") or {}
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        for g in ["all septic"] + [g for g in ORGANS if g in set(km["group"])]:
            k = km[km["group"] == g]
            ax.step(k["hours_since_window_start"], k["share_not_yet_alerted"], where="post",
                    color=INK if g == "all septic" else ORGANS[g], lw=2 if g == "all septic" else 1.3,
                    ls="--" if g == "all septic" else "-", label=g)
        ax.set(xlabel="Hours into the last 24 h before the sepsis label", ylabel="Not yet alerted", ylim=(0, 1.02))
        ax.set_title("Time to first alert by organ group", loc="left")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        p = rep.get("logrank", {}).get("p_value")
        R.save(fig, "time_to_alert_by_organ", "Kaplan-Meier curves of time to the first alert in the 24 h before the "
               "sepsis label (LightGBM, balanced), censored at the label." + (f" Log-rank p = {p:.2g}." if p is not None else ""))


def fig_patterns(R: Report) -> None:
    pm = R.csv("evaluation", "organs", "pattern_alert_metrics_by_hospital.csv")
    if pm is None:
        return R.skip("Figure: vital-sign patterns")
    d = pm[(pm["model"] == "p_lgbm") & (pm["operating_point"] == OP) & (pm["group"] != "all septic")]
    fig = _ab_groups(d, list(PATTERNS), "Warned in time by vital-sign pattern: hospital A vs B")
    R.save(fig, "warned_by_vital_pattern_hospital", "Septic patients warned in time by vital-sign pattern in the 24 h "
           "before the label, hospital A vs B, LightGBM at the balanced setting. Error bars: Wilson 95% CIs.")


def fig_time_check(R: Report) -> None:
    wh = R.csv("evaluation", "within_hour_discrimination.csv")
    if wh is None:
        return R.skip("Figure: time check")
    names = {**{m: v[0] for m, v in MODELS.items()}, "p_lgbm_time_only": "Time in ICU\nonly"}
    ms = [m for m in names if m in set(wh["model"])]
    t = wh.pivot(index="model", columns="bin", values="auroc").loc[ms]
    x = np.arange(len(ms))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(x - 0.19, t["all hours (pooled)"], 0.36, color="#9FB3B8", label="All hours compared")
    ax.bar(x + 0.19, t["hour-stratified"], 0.36, color="#0E7C86", label="Same hour of stay only")
    for i, m in enumerate(ms):
        ax.text(i - 0.19, t.loc[m, "all hours (pooled)"] + 0.01, f"{t.loc[m, 'all hours (pooled)']:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.19, t.loc[m, "hour-stratified"] + 0.01, f"{t.loc[m, 'hour-stratified']:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, color=MUTED, ls=":", lw=0.9)
    ax.set_xticks(x, [names[m].replace("Logistic regression", "Logistic\nregression") for m in ms])
    ax.set(ylim=(0.4, 1.02), ylabel="AUROC")
    ax.set_title("Skill beyond time in the ICU", loc="left")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    R.save(fig, "time_in_icu_check", "Hour-level AUROC pooled over all hours and stratified by hour of stay (only "
           "hours at the same hour of stay are compared). A model that only knows the time in the ICU drops to chance.")


def fig_transfer(R: Report) -> None:
    trm = R.csv("evaluation", "transfer_metrics.csv")
    if trm is None:
        return R.skip("Figure: hospital transfer")
    t = trm[(trm["operating_point"] == OP) & ~trm["setting"].str.contains("re-set")].copy()
    t["who"] = np.where(t["setting"].str.startswith("within"), "Trained on the same hospital", "Trained on the other hospital")
    hs = sorted(t["target"].unique())
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, w in enumerate(["Trained on the same hospital", "Trained on the other hospital"]):
        s = t[t["who"] == w].set_index("target").loc[hs]
        ax.bar(np.arange(len(hs)) + (i - 0.5) * 0.38, 100 * s["sens_timely"], 0.36, color=["#9FB3B8", "#0E7C86"][i], label=w,
               yerr=[np.clip(100 * (s["sens_timely"] - s["sens_timely_lo"]), 0, None),
                        np.clip(100 * (s["sens_timely_hi"] - s["sens_timely"]), 0, None)],
               error_kw=dict(lw=0.8, ecolor=INK, capsize=2))
    ax.set_xticks(np.arange(len(hs)), [f"Patients of hospital {h}" for h in hs])
    ax.set(ylim=(0, 105), ylabel="Warned in time (%)")
    ax.set_title("Hospital transfer (LightGBM)", loc="left")
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.12))
    R.save(fig, "hospital_transfer", "LightGBM trained on one hospital and tested on the other, with the threshold the "
           "source hospital would choose, against a model trained on the target hospital itself. 95% bootstrap CIs.")


def fig_forest(R: Report) -> None:
    rows = []
    for f, label in ((("evaluation", "organs", "organ_model_comparison.csv"), "Organ"),
                     (("evaluation", "pattern_model_comparison.csv"), "Vital pattern")):
        c = R.csv(*f)
        if c is None:
            continue
        c = c[(c["group"] == "all patterns with an expert") & (c["role"] != "global reference")]
        nm = {"p_gru_mtl": "Multi-task GRU", "p_gru_moe": "Mixture-of-experts GRU", "p_lgbm_moe_max": "Pattern experts",
              "p_lgbm_organ_moe_max": "Organ experts"}
        for r in c.itertuples():
            name = "Oracle" if r.role.startswith("oracle") else nm.get(r.model)
            if name:
                rows.append((f"{label}: {name}", r.diff_vs_reference_pts, r.diff_lo, r.diff_hi))
    if not rows:
        return R.skip("Figure: forest plot")
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(rows) + 1.0))
    for i, (n, d, lo, hi) in enumerate(rows[::-1]):
        col = "#2E8B57" if lo > 0 else ("#C2410C" if hi < 0 else MUTED)
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ecolor=MUTED, capsize=2, ms=5)
    ax.axvline(0, color=INK, ls="--", lw=0.9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set(xlabel="Difference from LightGBM in % warned in time (percentage points)")
    ax.set_title("Group-aware models against LightGBM", loc="left")
    R.save(fig, "forest_group_aware", "Paired difference from LightGBM in the share of septic patients warned in time "
           "(each model at its own balanced threshold), 95% patient-bootstrap CIs. Oracle: each patient scored by the "
           "expert of its own group (hindsight; an upper bound, not deployable).")


def fig_shap(R: Report) -> None:
    sh = R.csv("evaluation", "shap_by_pattern.csv")
    if sh is None:
        return R.skip("Figure: SHAP")
    pos = sh[~sh["group"].str.startswith("negative")]
    agg = pos.assign(w=pos["mean_abs_shap"] * pos["n_rows"]).groupby("feature")[["w", "n_rows"]].sum()
    agg = (agg["w"] / agg["n_rows"]).sort_values(ascending=False).head(15)[::-1]

    def kind(f):
        if f in ("ICULOS", "HospAdmTime_slog"):
            return "#C2410C"
        if f in ("Age", "Gender", "unit"):
            return "#9FB3B8"
        if f.endswith("_age") or f.startswith(("n_vital_obs", "n_lab_")) or f == "lab_age_any":
            return "#D07A1F"
        return "#0E7C86"
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.barh(range(len(agg)), agg.to_numpy(), color=[kind(f) for f in agg.index])
    ax.set_yticks(range(len(agg)), agg.index, fontsize=8)
    ax.set(xlabel="Mean |SHAP| (log-odds)")
    ax.set_title("Inputs LightGBM relies on most", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in (("#0E7C86", "measured values"), ("#D07A1F", "measurement frequency"),
                                                             ("#C2410C", "time in hospital"), ("#9FB3B8", "patient details"))],
              loc="lower right")
    R.save(fig, "shap_top_features", "Mean absolute SHAP values of the 15 most influential inputs, for hours in the 12 h "
           "before the sepsis label, each hour explained by the fold model that did not train on it.")


def fig_eda(R: Report) -> None:
    G = ("evaluation", "figures")
    cr = R.csv(*G, "correlation_spearman.csv")
    if cr is not None:
        cr = cr.set_index(cr.columns[0])
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        im = ax.imshow(cr.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cr)), cr.columns, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(cr)), cr.index, fontsize=6.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho")
        ax.set_title("Correlation between vital signs and labs", loc="left")
        R.save(fig, "correlation_heatmap", "Pairwise Spearman correlations on a sample of pre-onset hours of development "
               "patients; blank where two variables are rarely measured in the same hour.")
    else:
        R.skip("Figure: correlation")
    ds = R.csv(*G, "distributions.csv")
    if ds is not None:
        vs = list(dict.fromkeys(ds["variable"]))[:10]
        fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.8))
        for ax, v in zip(axes.ravel(), vs):
            stats = []
            for g, col in (("septic, last 24 h before switch-on", "#C2410C"), ("non-septic", "#0E7C86")):
                r = ds[(ds["variable"] == v) & (ds["group"] == g)]
                if len(r):
                    r = r.iloc[0]
                    stats.append({"label": "septic" if g.startswith("septic") else "not septic", "med": r["median"],
                                  "q1": r["q1"], "q3": r["q3"], "whislo": r["p5"], "whishi": r["p95"], "fliers": []})
            b = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.6, medianprops=dict(color=INK, lw=1.2))
            for patch, col in zip(b["boxes"], ("#F2C4B0", "#B9E0E3")):
                patch.set_facecolor(col)
            ax.set_title(v, fontsize=8.5, loc="left")
            ax.tick_params(axis="x", labelsize=7)
        fig.tight_layout()
        R.save(fig, "septic_vs_nonseptic", "Vital signs and labs in septic patients (24 h before the label) and "
               "non-septic patients. Box: interquartile range; whiskers: 5th to 95th percentile.")
    ms = R.csv(*G, "missing_share.csv")
    if ms is not None:
        m = ms[ms["group"] == "all"].sort_values("missing_pct")
        fig, ax = plt.subplots(figsize=(4.8, 6.0))
        ax.barh(range(len(m)), m["missing_pct"], color=np.where(m["kind"] == "vital", "#0E7C86", "#9FB3B8"))
        ax.set_yticks(range(len(m)), m["variable"], fontsize=7)
        ax.set(xlabel="Hours with no measurement (%)", xlim=(0, 100))
        ax.set_title("Missing data", loc="left")
        R.save(fig, "missing_data", "Share of patient-hours without a measurement, per variable (vital signs dark, "
               "labs light); development patients, pre-onset hours.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, required=True)
    args = ap.parse_args()
    R = Report(args.outputs)

    table_cohort(R)
    table_performance(R)
    table_organs(R)
    if (args.outputs / "evaluation" / "alert_metrics_test.csv").exists():
        table_performance(R, test=True)
    fig_flow(R)
    fig_roc_pr(R)
    fig_confusion(R)
    fig_tradeoff(R)
    fig_by_hospital(R)
    fig_calibration(R)
    fig_organs(R)
    fig_patterns(R)
    fig_time_check(R)
    fig_transfer(R)
    fig_forest(R)
    fig_shap(R)
    fig_eda(R)

    cap = R.out / "report" / "captions.md"
    cap.write_text("# Figures and tables\n\nDevelopment (out-of-fold) results unless marked TEST. "
                   "Generated by backend/report_figures.py.\n\n" + "\n\n".join(R.captions) +
                   ("\n\n## Not generated (result file missing)\n\n" + "\n".join(f"- {s}" for s in R.skipped) if R.skipped else ""))
    z = R.out / "report" / "report_package.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in R.files + [cap]:
            zf.write(f, f.relative_to(R.out / "report"))
    print(f"[write] {R.n_fig} figures, {R.n_tab} tables -> {R.out / 'report'}  (all in report_package.zip)")


if __name__ == "__main__":
    main()
