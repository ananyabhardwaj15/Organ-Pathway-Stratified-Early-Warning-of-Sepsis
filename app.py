"""
app.py: sepsis early-warning dashboard (PhysioNet/CinC 2019).

    python run.py --dashboard          (or: streamlit run app.py)

Predict: enter or upload one patient's hourly data and get the hour-by-hour sepsis risk from the trained models, the
alerts, the reasons and the organ scores. The other tabs show how well the models work, overall and for hospital A vs B.
Everything is read from outputs/ (predictions use the saved models). Research use only, not a clinical tool.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Sepsis early warning", page_icon="🩺", layout="wide")
ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Look and feel
# --------------------------------------------------------------------------- #
INK, MUTED, LINE, BG, CARD = "#16313A", "#5E747B", "#DCE5E7", "#F4F8F8", "#FFFFFF"
TEAL, ALERT, GOOD = "#0E7C86", "#C2410C", "#2E8B57"
MODELS = {"p_lgbm": ("LightGBM", "#0E7C86"), "p_logreg": ("Logistic regression", "#7A6AAE"),
          "p_gru": ("GRU", "#D07A1F"), "p_ensemble": ("Ensemble (all three)", "#1F2937")}
SETTINGS = {"far2": "Strict", "far5": "Balanced", "far10": "Sensitive"}
SETTING_HELP = {"far2": "about 2 false alarms per 100 patient-days", "far5": "about 5 false alarms per 100 patient-days",
                "far10": "about 10 false alarms per 100 patient-days"}
HOSP = {"A": "#0E7C86", "B": "#9FB3B8"}
ORGANS = {"Lungs": ("#2F6FB0", "SpO2 / FiO2 ratio", "SpO2/FiO2 below 512 / 357 / 214 / 89"),
          "Clotting": ("#8E5BA8", "platelets", "platelets below 150 / 100 / 50 / 20"),
          "Liver": ("#A0661B", "bilirubin", "bilirubin at least 1.2 / 2.0 / 6.0 / 12.0 mg/dL"),
          "Circulation": ("#B23A48", "blood pressure (MAP)", "MAP below 70 mmHg (no vasopressor data)"),
          "Kidneys": ("#2E8B57", "creatinine", "creatinine at least 1.2 / 2.0 / 3.5 / 5.0 mg/dL"),
          "Multi-organ": ("#4A5A61", "", "two or more organs got worse"),
          "Unattributable": ("#B8C4C7", "", "no organ score got worse, or the tests were not done")}
ORGAN5 = ["Lungs", "Clotting", "Liver", "Circulation", "Kidneys"]
PATTERNS = {"Hemodynamic": ("#B23A48", "low blood pressure (MAP below 70)"),
            "Respiratory": ("#2F6FB0", "low oxygen (below 92%) or fast breathing (above 22)"),
            "Inflammatory": ("#C98A0B", "fever, low temperature, or heart rate above 100"),
            "Multi-system": ("#6B4C9A", "two or more of the above"),
            "No abnormal vitals": ("#8A9A9E", "none of the above"),
            "unassigned": ("#B8C4C7", "not enough data in the 24 h before sepsis")}
AWARE = {"p_gru_mtl": "Multi-task GRU", "p_gru_moe": "Mixture-of-experts GRU", "p_lgbm_moe_max": "Pattern experts",
         "p_lgbm_organ_moe_max": "Organ experts"}
EDIT_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2", "Lactate", "WBC", "Creatinine", "Platelets",
             "Bilirubin_total", "BUN", "Glucose", "pH", "HCO3", "Hgb"]
FONT = "Public Sans, system-ui, -apple-system, Segoe UI, sans-serif"
MD = 'div[data-testid="stMarkdownContainer"]'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stRadio, .stSelectbox, button, input {{ font-family: {FONT}; }}
.stApp {{ background: {BG}; color: {INK}; }}
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {LINE}; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
h1, h2, h3, h4, h5 {{ color: {INK}; letter-spacing: -0.01em; }}
{MD} p.app-title {{ font-size: 2.15rem; font-weight: 700; line-height: 1.15; margin: 0; color: {INK}; }}
{MD} p.app-sub {{ color: {MUTED}; font-size: 1.02rem; margin: .4rem 0 1rem; max-width: 80ch; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-size: 1.02rem; font-weight: 600; padding: .65rem 0; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
{MD} p.lead {{ font-size: 1.07rem; color: {INK}; max-width: 82ch; margin: .3rem 0 1rem; }}
{MD} p.note {{ color: {MUTED}; font-size: .92rem; max-width: 88ch; }}
.stat {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; padding: 1rem 1.1rem; min-height: 9.6rem; }}
.stat .v {{ font-size: 2.05rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }}
.stat .k {{ color: {INK}; font-weight: 600; margin-top: .3rem; }}
.stat .s {{ color: {MUTED}; font-size: .86rem; margin-top: .15rem; }}
.verdict {{ border-left: 4px solid var(--c); background: {CARD}; padding: .8rem 1rem; border-radius: 0 8px 8px 0;
           margin: .3rem 0 1rem; max-width: 88ch; }}
</style>""", unsafe_allow_html=True)

CHART_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
             "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "sepsis_figure"}}


def style(fig, height=420, legend=True):
    fig.update_layout(font=dict(family=FONT, color=INK, size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
                      height=height, margin=dict(l=10, r=10, t=50, b=10), title_font=dict(size=15), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, title=None))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


def show(fig, height=420, legend=True, legend_side=False, name="sepsis_figure", legend_below=False):
    """Every chart keeps its toolbar: camera = download a high-resolution PNG; the expand icon = full screen."""
    fig = style(fig, height, legend)
    if legend_side:
        fig.update_layout(legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top"))
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=-0.18, yanchor="top"),
                          margin=dict(l=10, r=10, t=50, b=40))
    cfg = {**CHART_CFG, "toImageButtonOptions": {**CHART_CFG["toImageButtonOptions"], "filename": name}}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def frame(df, height=None):
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def img(path, width=None):
    try:
        st.image(str(path), width=width or "stretch")
    except Exception:
        st.image(str(path), use_container_width=width is None, width=width)


def lead(t): st.markdown(f'<p class="lead">{t}</p>', unsafe_allow_html=True)
def note(t): st.markdown(f'<p class="note">{t}</p>', unsafe_allow_html=True)
def verdict(t, c=TEAL): st.markdown(f'<div class="verdict" style="--c:{c}">{t}</div>', unsafe_allow_html=True)


def stat(col, value, label, sub="", color=INK):
    col.markdown(f'<div class="stat"><div class="v" style="color:{color}">{value}</div><div class="k">{label}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def wilson_cols(d):
    n, p_ = d["n_septic"].to_numpy(float), d["sens_timely"].to_numpy(float)
    z = 1.96
    den = 1 + z * z / n
    c = (p_ + z * z / (2 * n)) / den
    h = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / den
    return d.assign(w_lo=np.clip(c - h, 0, 1), w_hi=np.clip(c + h, 0, 1))


def gap_sentence(big):
    hi, lo = big.loc[big["sens_timely"].idxmax()], big.loc[big["sens_timely"].idxmin()]
    overlap = lo["w_hi"] >= hi["w_lo"]
    return (f"<b>{hi['group']}</b> patients are warned most often ({100 * hi['sens_timely']:.0f}% of {int(hi['n_septic'])}) "
            f"and <b>{lo['group']}</b> least often ({100 * lo['sens_timely']:.0f}% of {int(lo['n_septic'])}). "
            + ("Their 95% intervals overlap, so the gap could be chance." if overlap
               else "Their 95% intervals do not overlap, so the gap is unlikely to be chance."), lo["group"])


def ab_chart(d, order, colors, title, name):
    """Hospital A and B side by side for every group (Wilson 95% CIs)."""
    d = wilson_cols(d)
    fig = go.Figure()
    for h in [h for h in ["A", "B"] if h in set(d["hospital"])]:
        s = d[d["hospital"] == h].set_index("group").reindex([g for g in order if g in set(d["group"])])
        fig.add_trace(go.Bar(y=s.index, x=s["sens_timely"] * 100, orientation="h", name=f"Hospital {h}",
                             marker_color=HOSP[h],
                             error_x=dict(type="data", symmetric=False, array=np.clip((s["w_hi"] - s["sens_timely"]) * 100, 0, None),
                                          arrayminus=np.clip((s["sens_timely"] - s["w_lo"]) * 100, 0, None), color=INK, thickness=1),
                             text=[f"{v:.0f}% (n={int(n)})" if pd.notna(v) else "" for v, n in
                                   zip(s["sens_timely"] * 100, s["n_septic"].fillna(0))],
                             textposition="inside", insidetextanchor="start",
                             textfont=dict(color="white" if h == "A" else INK, size=12)))
    fig.update_layout(barmode="group", title=title, xaxis_title="% of septic patients warned in time", xaxis_range=[0, 105],
                      yaxis=dict(autorange="reversed"))
    show(fig, 120 + 64 * d["group"].nunique(), name=name)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
OUT = Path(os.environ.get("SEPSIS_OUTPUTS", "outputs"))
if not OUT.exists():
    OUT = ROOT / "outputs"


def F(*parts):
    return OUT.joinpath(*parts)


@st.cache_data(show_spinner=False)
def csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


@st.cache_data(show_spinner=False)
def js(path):
    return json.load(open(path)) if Path(path).exists() else None


def need(df, what):
    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        st.info(f"{what} has not been calculated yet. Run `python run.py`, then reload this page.")
        return False
    return True


@st.cache_resource(show_spinner="Loading the trained models ...")
def predictor():
    spec = importlib.util.spec_from_file_location("sepsis_predict", ROOT / "backend" / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


metrics = csv(str(F("evaluation", "alert_metrics.csv")))
thresholds = js(str(F("evaluation", "thresholds.json"))) or {}
gate = js(str(F("organs", "organ_gate.json")))
assign = csv(str(F("organs", "organ_assignments.csv")))
org_m = csv(str(F("evaluation", "organs", "organ_alert_metrics.csv")))
pat_m = csv(str(F("evaluation", "organs", "pattern_alert_metrics_by_hospital.csv")))

# ---- sidebar ----
st.sidebar.markdown("### Controls")
if metrics is None:
    st.warning(f"No results found in `{OUT}`. Run `python run.py` first, then reload this page.")
    st.stop()
have = [m for m in MODELS if m in set(metrics["model"])]
model = st.sidebar.radio("Model", have, format_func=lambda m: MODELS[m][0],
                         index=have.index("p_lgbm") if "p_lgbm" in have else 0,
                         help="LightGBM, logistic regression and GRU are separate models; the ensemble combines all three.")
ops = [o for o in SETTINGS if o in set(metrics["operating_point"])]
op = st.sidebar.radio("Alert setting", ops, format_func=lambda o: SETTINGS[o], index=ops.index("far5") if "far5" in ops else 0)
st.sidebar.caption(f"**{SETTINGS[op]}:** {SETTING_HELP[op]}. Strict gives fewer false alarms but misses more septic "
                   "patients; Sensitive is the opposite.")
st.sidebar.divider()
st.sidebar.caption("**Warned in time** = an alert in the 12 hours before sepsis is flagged. **False alarm** = an alert for a "
                   "patient who never became septic. Results are for development patients (out-of-fold).")
MNAME, MCOL = MODELS[model]


def row(m, o, gtype="all", group="all", table=None):
    t = metrics if table is None else table
    r = t[(t["model"] == m) & (t["operating_point"] == o) & (t["group_type"] == gtype) & (t["group"] == group)]
    return r.iloc[0] if len(r) else None


st.markdown('<p class="app-title">Sepsis early warning in the ICU</p>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Hour by hour, the models predict whether an ICU patient will develop sepsis in the next 12 '
            'hours and raise an alert when the risk is high. Try it on a patient in <b>Predict</b>; the other tabs show how '
            'well it works, which organs were failing, and whether results hold in both hospitals. PhysioNet 2019 data, two '
            'hospitals; research only, not a clinical tool.</p>', unsafe_allow_html=True)

tabs = st.tabs(["Predict", "Performance", "Organs", "Vital-sign patterns", "Checks & comparisons", "Explainability",
                "Data", "Report figures"])

# --------------------------------------------------------------------------- #
# 1. Predict
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def example_index(out):
    p, f = Path(out, "clean", "patients.csv"), Path(out, "splits", "folds.csv")
    if not (p.exists() and f.exists()):
        return None
    d = pd.read_csv(p).merge(pd.read_csv(f)[["Patient_ID", "split", "fold"]], on="Patient_ID")
    a = Path(out, "organs", "organ_assignments.csv")
    d = d.merge(pd.read_csv(a)[["Patient_ID", "organ_group"]], on="Patient_ID", how="left") if a.exists() else d.assign(organ_group=np.nan)
    d["organ_group"] = d["organ_group"].fillna("not septic")
    return d[d["include_main"] & (d["split"] == "dev")].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def example_raw(out, pid):
    h = pd.read_parquet(Path(out, "clean", "hourly_clean.parquet"), filters=[("Patient_ID", "==", int(pid))]).sort_values("Hour")
    h = h[h["is_pre_onset"]].copy()
    h["Unit2"] = np.where(h["Unit1"].notna(), 1 - h["Unit1"], np.nan)     # the cleaned file stores Unit1 only
    keep = ["Hour", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
            "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
            "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
            "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
    return h[[c for c in keep if c in h.columns]].reset_index(drop=True)


@st.cache_data(show_spinner="Predicting ...")
def run_prediction(out, raw_csv: str, fold):
    pr = predictor()
    r = pr.predict(pd.read_csv(io.StringIO(raw_csv)), out, fold=fold)
    return r["hours"], r["contrib"], r["features"], r["clean"]


with tabs[0]:
    lead("Give the model one patient's hourly data and it predicts, for every hour, the risk that sepsis is flagged in the "
         "next 12 hours, whether it would raise an alert, and why.")
    src = st.radio("Patient data", ["Example patient", "Upload a CSV"], horizontal=True)
    raw, fold, info = None, None, None
    if src == "Example patient":
        idx = example_index(str(OUT))
        if need(idx, "The patient list"):
            c = st.columns([1, 1.2, 1.2])
            kind_ = c[0].radio("Type", ["septic", "not septic"], horizontal=True)
            d = idx[idx["septic"]] if kind_ == "septic" else idx[~idx["septic"]]
            if kind_ == "septic":
                grp = c[1].selectbox("Organ group", ["any"] + [o for o in ORGANS if o in set(d["organ_group"])])
                d = d if grp == "any" else d[d["organ_group"] == grp]
            pid = c[2].selectbox(f"Patient ID ({len(d):,} available)", d.sort_values("Patient_ID")["Patient_ID"].tolist()[:3000])
            info = d[d["Patient_ID"] == pid].iloc[0]
            fold = int(info["fold"])
            base = example_raw(str(OUT), int(pid))
            with st.expander("See or change this patient's data (edit any value, then the prediction updates)"):
                show_cols = ["Hour"] + [c_ for c_ in EDIT_COLS if c_ in base.columns]
                edited = st.data_editor(base[show_cols], hide_index=True, num_rows="fixed", key=f"edit_{pid}",
                                        height=320)
                raw = base.copy()
                raw[show_cols] = edited[show_cols].to_numpy()
                st.download_button("Download this patient as a CSV template", raw.to_csv(index=False).encode(),
                                   file_name=f"patient_{pid}.csv", mime="text/csv")
            note("Example patients come from the development set and are scored by the model trained without them, so "
                 "the prediction is honest. After you edit values, it shows what the model would say for those values.")
    else:
        up = st.file_uploader("CSV with one row per hour for one patient (columns as in the dataset: HR, O2Sat, Temp, "
                              "MAP, Resp, labs, Age, Gender, ...)", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            if "Patient_ID" in raw.columns and raw["Patient_ID"].nunique() > 1:
                first = raw["Patient_ID"].iloc[0]
                st.warning(f"The file has several patients; showing the first one (Patient_ID {first}).")
                raw = raw[raw["Patient_ID"] == first]
        note("Tip: switch to Example patient and use 'Download this patient as a CSV template' for the exact format. "
             "Missing columns are treated as not measured. Uploaded patients are scored by the average of the 5 trained models.")

    if raw is not None and len(raw):
        try:
            hours, contrib, feats, cleaned = run_prediction(str(OUT), raw.to_csv(index=False), fold)
        except Exception as e:
            st.error(f"Could not predict: {e}")
            st.stop()
        cols = [m for m in MODELS if m in hours.columns]
        T = len(hours)
        t = st.slider("Hour to look at", 0, T - 1, T - 1) if T > 1 else 0
        tau = thresholds.get(model, {}).get(op)
        k = st.columns(4)
        p_now = float(hours.loc[t, model]) if model in hours else np.nan
        alert = tau is not None and p_now >= tau
        stat(k[0], f"{p_now:.2f}" if model != "p_ensemble" else f"{100 * p_now:.0f}th", f"{MNAME} risk at hour {t}",
             "0 = low, 1 = high" if model != "p_ensemble" else "percentile of risk among all patient-hours", MCOL)
        stat(k[1], "ALERT" if alert else "no alert", "decision at this hour",
             f"threshold {tau:.3f} ({SETTINGS[op].lower()})" if tau is not None else "", ALERT if alert else GOOD)
        al = hours.loc[: t][hours.loc[: t, model] >= tau] if tau is not None else hours.iloc[0:0]
        stat(k[2], f"hour {int(al['Hour'].min())}" if len(al) else "none", "first alert so far", f"{len(al)} alert hours up to now")
        votes = sum(bool(hours.loc[t, m] >= thresholds[m][op]) for m in cols if m in thresholds and op in thresholds[m])
        stat(k[3], f"{votes} of {len(cols)}", "models alerting now", ", ".join(MODELS[m][0].split(' (')[0] for m in cols
                                                                              if m in thresholds and hours.loc[t, m] >= thresholds[m][op]) or "none")
        reveal = info is not None and st.checkbox("Reveal the recorded outcome (when sepsis was actually flagged)")
        left, right = st.columns([1.5, 1])
        with left:
            fig = go.Figure(go.Scatter(x=hours["Hour"], y=hours[model], mode="lines", name=MNAME,
                                       line=dict(color=MCOL, width=3), fill="tozeroy", fillcolor="rgba(14,124,134,0.07)"))
            if tau is not None:
                above = hours[hours[model] >= tau]
                fig.add_trace(go.Scatter(x=above["Hour"], y=above[model], mode="markers", name="alert hours",
                                         marker=dict(color=ALERT, size=6)))
                fig.add_hline(y=tau, line_dash="dot", line_color=ALERT, annotation_text="alert threshold",
                              annotation_position="top left")
            fig.add_vline(x=t, line_color=INK, line_width=1)
            if reveal and info["septic"]:
                fig.add_vline(x=float(info["first_pos_hour"]), line_dash="dash", line_color=INK,
                              annotation_text="sepsis flagged", annotation_position="top left")
            ylab = "risk percentile (0 to 1)" if model == "p_ensemble" else "predicted risk (0 to 1)"
            fig.update_layout(title=f"{MNAME}: predicted risk hour by hour", xaxis_title="hour in the ICU", yaxis_title=ylab,
                              yaxis_range=[0, 1.02])
            show(fig, 400, legend=False, name="predicted_risk")
            ok_m = [m for m in cols if m in thresholds and op in thresholds[m]]
            if ok_m:
                z = np.vstack([(hours[m] >= thresholds[m][op]).astype(float).to_numpy() for m in ok_m])
                fig = go.Figure(go.Heatmap(z=z, x=hours["Hour"], y=[MODELS[m][0] for m in ok_m], zmin=0, zmax=1, xgap=1, ygap=3,
                                           colorscale=[[0, "#EEF3F4"], [0.5, "#EEF3F4"], [0.5, ALERT], [1, ALERT]], showscale=False,
                                           hovertemplate="%{y}: %{customdata} at hour %{x}<extra></extra>",
                                           customdata=np.where(z > 0, "ALERT", "no alert")))
                fig.add_vline(x=t, line_color=INK, line_width=1)
                if reveal and info["septic"]:
                    fig.add_vline(x=float(info["first_pos_hour"]), line_dash="dash", line_color=INK)
                fig.update_layout(title=f"When each model would alert ({SETTINGS[op].lower()} setting)",
                                  xaxis_title="hour in the ICU", yaxis=dict(autorange="reversed"))
                show(fig, 250, legend=False, name="alert_timeline")
                note("Orange = the model is above its own alert threshold in that hour. Each model has its own scale, so "
                     "this timeline is the fair way to compare them.")
        with right:
            pr = predictor()
            rs = pr.reasons(contrib.iloc[t], feats.iloc[t], k=8)
            rs["signed"] = np.where(rs["effect"] == "raises risk", rs["strength"], -rs["strength"])
            rs["label"] = [f"{a} = {v:.3g}" if pd.notna(v) else f"{a} (not measured)" for a, v in zip(rs["input"], rs["value"])]
            fig = go.Figure(go.Bar(y=rs["label"][::-1], x=rs["signed"][::-1], orientation="h",
                                   marker_color=[ALERT if v > 0 else TEAL for v in rs["signed"][::-1]],
                                   hovertemplate="%{y}<br>SHAP %{x:.3f}<extra></extra>"))
            fig.add_vline(x=0, line_color=INK, line_width=1)
            fig.update_layout(title=f"Why: top reasons at hour {t} (LightGBM)", xaxis_title="SHAP (right = raises risk)")
            show(fig, 420, legend=False, name="prediction_reasons")
            note("Each bar is one input and its value at that hour. Orange pushes the risk up, teal pushes it down "
                 "(SHAP values of the LightGBM model).")
        oc = [f"sofa_{o.lower()}" for o in ORGAN5]
        if all(c_ in hours.columns for c_ in oc):
            z = np.vstack([hours[c_].to_numpy(float) for c_ in oc])
            fig = go.Figure(go.Heatmap(z=z, x=hours["Hour"], y=ORGAN5, zmin=0, zmax=4, xgap=1, ygap=2,
                                       colorscale=[[0, "#EAF4F4"], [0.25, "#F6D9A8"], [0.5, "#EFA66A"], [0.75, "#D9653B"], [1, "#9E2A2B"]],
                                       colorbar=dict(title="SOFA", tickvals=[0, 1, 2, 3, 4]),
                                       hovertemplate="%{y}: %{z} at hour %{x}<extra></extra>"))
            fig.add_vline(x=t, line_color=INK, line_width=1)
            fig.update_layout(title="Organ scores (SOFA, 0 = normal, 4 = worst; blank = not measured)",
                              xaxis_title="hour in the ICU", yaxis=dict(autorange="reversed"))
            show(fig, 280, legend=False, name="organ_scores")
        with st.expander("Vital signs of this patient"):
            vit = [("HR", "heart rate"), ("MAP", "blood pressure (MAP)"), ("Resp", "breathing rate"), ("O2Sat", "oxygen %"),
                   ("Temp", "temperature")]
            fig = make_subplots(rows=len(vit), cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=[v[1] for v in vit])
            for i, (v, _) in enumerate(vit, start=1):
                fig.add_trace(go.Scatter(x=cleaned["Hour"], y=cleaned[v], mode="lines+markers", connectgaps=True,
                                         line=dict(color=MUTED, width=1.3), marker=dict(size=4)), i, 1)
            fig.update_annotations(font=dict(size=12, color=MUTED), x=0, xanchor="left")
            show(fig, 620, legend=False, name="vital_signs")
        out_df = hours.copy()
        st.download_button("Download all hourly predictions (CSV)", out_df.to_csv(index=False).encode(),
                           file_name="predictions.csv", mime="text/csv")
        if reveal:
            if info["septic"]:
                F0 = int(info["first_pos_hour"])
                late = hours[(hours["Hour"] >= F0 - 12) & (hours[model] >= tau)] if tau is not None else hours.iloc[0:0]
                verdict(f"Recorded outcome: sepsis was flagged at hour {F0}; organ group <b>{info['organ_group']}</b>. "
                        + (f"{MNAME} <b>warned in time</b> (alerts in the last 12 hours, from hour {int(late['Hour'].min())})."
                           if len(late) else f"{MNAME} did <b>not</b> warn in the last 12 hours."), MCOL)
            else:
                first = hours.loc[hours[model] >= tau, "Hour"] if tau is not None else pd.Series([], dtype=float)
                verdict("Recorded outcome: this patient never became septic. " +
                        (f"{MNAME} alerted from hour {int(first.min())}, a false alarm." if len(first) else f"{MNAME} never alerted."), MCOL)

# --------------------------------------------------------------------------- #
# 2. Performance
# --------------------------------------------------------------------------- #
with tabs[1]:
    r = row(model, op)
    if r is not None:
        lead(f"With the <b>{SETTINGS[op].lower()}</b> setting, {MNAME} warns <b>{100 * r['sens_timely']:.0f}%</b> of septic "
             f"patients in the 12 hours before sepsis is flagged, with {r['far_per_100pd']:.1f} false alarms per 100 "
             "patient-days.")
        k = st.columns(4)
        stat(k[0], f"{100 * r['sens_timely']:.0f}%", "septic patients warned in time",
             f"95% CI {100 * r['sens_timely_lo']:.0f} to {100 * r['sens_timely_hi']:.0f}%", MCOL)
        stat(k[1], f"{r['far_per_100pd']:.1f}", "false alarms per 100 patient-days", "alerts for patients who never became septic")
        stat(k[2], f"{r['median_lead_any']:.0f} h", "median warning before sepsis", "first alert to sepsis being flagged")
        stat(k[3], f"{r['util_norm']:.2f}", "normalised utility", "PhysioNet 2019 score, adapted (1 = perfect)")
        st.write("")
        st.markdown("#### The four models side by side")
        d = metrics[metrics["model"].isin(have) & (metrics["operating_point"] == op) & (metrics["group_type"] == "all")]
        d = d.set_index("model").loc[have].reset_index()
        cs = csv(str(F("evaluation", "figures", "curve_summary.csv")))
        auc = cs.set_index("model") if cs is not None else None
        frame(pd.DataFrame({"Model": [MODELS[m][0] for m in d["model"]],
                            "AUROC": [round(auc.loc[m, "auroc"], 3) if auc is not None and m in auc.index else None for m in d["model"]],
                            "AUPRC": [round(auc.loc[m, "auprc"], 3) if auc is not None and m in auc.index else None for m in d["model"]],
                            "% warned in time": (100 * d["sens_timely"]).round(1),
                            "95% CI": [f"{100 * a:.1f} to {100 * b:.1f}" for a, b in zip(d["sens_timely_lo"], d["sens_timely_hi"])],
                            "False alarms / 100 days": d["far_per_100pd"].round(2),
                            "Median warning (h)": d["median_lead_any"],
                            "Utility": d["util_norm"].round(3)}))
        st.markdown("#### Hospital A vs hospital B")
        hh = metrics[metrics["model"].isin(have) & (metrics["operating_point"] == op) & (metrics["group_type"] == "hospital")].copy()
        if len(hh):
            hh["name"] = hh["model"].map(lambda m: MODELS[m][0])
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                for h in sorted(hh["group"].unique()):
                    s = hh[hh["group"] == h].set_index("model").loc[have]
                    fig.add_trace(go.Bar(x=[MODELS[m][0] for m in have], y=s["sens_timely"] * 100, name=f"Hospital {h}",
                                         marker_color=HOSP.get(h, MUTED),
                                         error_y=dict(type="data", symmetric=False, array=(s["sens_timely_hi"] - s["sens_timely"]) * 100,
                                                      arrayminus=(s["sens_timely"] - s["sens_timely_lo"]) * 100, color=INK, thickness=1)))
                fig.update_layout(barmode="group", title="Septic patients warned in time", yaxis_title="%", yaxis_range=[0, 105])
                show(fig, 380, name="warned_by_hospital")
            with c2:
                hh["hosp"] = "Hospital " + hh["group"].astype(str)
                fig = px.bar(hh, x="name", y="far_per_100pd", color="hosp", barmode="group",
                             color_discrete_map={f"Hospital {k_}": v for k_, v in HOSP.items()},
                             labels={"name": "", "far_per_100pd": "per 100 patient-days", "hosp": ""},
                             category_orders={"name": [MODELS[m][0] for m in have]})
                fig.update_layout(title="False alarms")
                show(fig, 380, name="false_alarms_by_hospital")
            note("One threshold per model for both hospitals. Different false-alarm rates mean the scores are not on exactly "
                 "the same scale in the two hospitals (see Checks & comparisons, hospital transfer).")
        st.markdown("#### Curves")
        which = st.radio("Curve", ["ROC and precision-recall", "Warning rate vs false alarms", "How early is the first alert?",
                                   "Calibration", "Confusion matrices"], horizontal=True, label_visibility="collapsed")
        G = ("evaluation", "figures")
        cmap = {MODELS[m][0]: MODELS[m][1] for m in MODELS}
        nm = lambda x: x.assign(model=x["model"].map(lambda m: MODELS[m][0]))
        if which == "ROC and precision-recall":
            roc, pr_ = csv(str(F(*G, "roc_curves.csv"))), csv(str(F(*G, "pr_curves.csv")))
            if need(roc, "The ROC curves"):
                l, rr = st.columns(2)
                fig = px.line(nm(roc[roc["model"].isin(have)]), x="fpr", y="tpr", color="model", color_discrete_map=cmap,
                              labels={"fpr": "false positive rate", "tpr": "true positive rate", "model": ""})
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color=MUTED))
                fig.update_layout(title="ROC curve")
                with l:
                    show(fig, 470, name="roc_curve", legend_below=True)
                fig = px.line(nm(pr_[pr_["model"].isin(have)]), x="recall", y="precision", color="model", color_discrete_map=cmap,
                              labels={"model": ""})
                fig.add_hline(y=float(auc["prevalence"].iloc[0]), line_dash="dot", line_color=MUTED)
                fig.update_layout(title="Precision-recall curve")
                with rr:
                    show(fig, 470, name="pr_curve", legend_below=True)
                note(f"Hour by hour; only {100 * auc['prevalence'].iloc[0]:.1f}% of hours are positive, so the precision-recall "
                     "curve is the more honest of the two (dotted line = guessing).")
        elif which == "Warning rate vs false alarms":
            uc = csv(str(F("evaluation", "utility_curve.csv")))
            if need(uc, "The trade-off curve"):
                u = nm(uc[uc["model"].isin(have) & (uc["far"] <= 30)].sort_values("far"))
                fig = px.line(u, x="far", y=u["sens_timely"] * 100, color="model", color_discrete_map=cmap,
                              labels={"far": "false alarms per 100 non-septic patient-days", "y": "% warned in time", "model": ""})
                for x in (2, 5, 10):
                    fig.add_vline(x=x, line_dash="dot", line_color=MUTED)
                fig.update_layout(title="How many septic patients are warned, for a given alarm burden")
                show(fig, 440, name="warning_vs_false_alarms")
                note("The dotted lines are the Strict, Balanced and Sensitive settings.")
        elif which == "How early is the first alert?":
            lt = csv(str(F("evaluation", "leadtime_curve.csv")))
            if need(lt, "The lead-time curve"):
                l_ = nm(lt[lt["model"].isin(have) & (lt["group_type"] == "all") & (lt["operating_point"] == "far5")])
                fig = px.line(l_, x="hours_before_switch_on", y="share_alerted_at_least_h_before", color="model",
                              color_discrete_map=cmap, labels={"hours_before_switch_on": "hours before sepsis is flagged",
                                                               "share_alerted_at_least_h_before": "share alerted at least this early",
                                                               "model": ""})
                fig.update_layout(title="Share of septic patients first alerted at least h hours before (balanced)")
                show(fig, 440, name="lead_time")
                note("Very long lead times often come from long-stay patients whose risk drifts up; read together with the "
                     "time check.")
        elif which == "Calibration":
            cal = csv(str(F(*G, "calibration.csv")))
            if need(cal, "The calibration curve"):
                c = nm(cal[(cal["group"] == "all") & cal["model"].isin(have)])
                fig = px.line(c, x="mean_pred", y="observed", color="model", markers=True, log_x=True, log_y=True,
                              color_discrete_map=cmap, labels={"mean_pred": "predicted risk", "observed": "actual rate", "model": ""})
                lo = float(c[["mean_pred", "observed"]].replace(0, np.nan).min().min())
                fig.add_shape(type="line", x0=lo, y0=lo, x1=1, y1=1, line=dict(dash="dot", color=MUTED))
                fig.update_layout(title="Does a predicted risk of 10% really mean 10%?")
                show(fig, 460, name="calibration")
                note("Points on the dotted line are well calibrated. The ensemble is not shown: its score is a percentile, not "
                     "a probability. The GRU over-predicts by design (extra weight on sepsis hours); its ranking is still valid.")
        else:
            cm = csv(str(F(*G, "confusion_matrix.csv")))
            if need(cm, "The confusion matrices"):
                cc = st.columns(len(have))
                for col, m in zip(cc, have):
                    q = cm[(cm["model"] == m) & (cm["operating_point"] == op)]
                    if not len(q):
                        continue
                    c = q.iloc[0]
                    z = [[c.tp / (c.tp + c.fn), c.fn / (c.tp + c.fn)], [c.fp / (c.fp + c.tn), c.tn / (c.fp + c.tn)]]
                    txt = [[f"warned<br><b>{c.tp:,}</b>", f"missed<br><b>{c.fn:,}</b>"], [f"false alarm<br><b>{c.fp:,}</b>", f"no alert<br><b>{c.tn:,}</b>"]]
                    fig = go.Figure(go.Heatmap(z=z, x=["alert", "no alert"], y=["septic", "not septic"], text=txt,
                                               texttemplate="%{text}", textfont=dict(size=13, color=INK), zmin=0, zmax=1,
                                               colorscale=[[0, "#F7FAFA"], [1, "#9FD3D7"]], showscale=False))
                    fig.update_layout(title=MODELS[m][0], yaxis_autorange="reversed", margin=dict(l=5, r=5, t=40, b=5))
                    with col:
                        show(fig, 300, legend=False, name=f"confusion_{m}")
                note("Counted in patients at the chosen alert setting.")
        test_m = csv(str(F("evaluation", "alert_metrics_test.csv")))
        with st.expander("Final test-set results" + ("" if test_m is not None else " (not run yet)")):
            if test_m is None:
                note("Run `python run.py --final` once, at the very end, when nothing will change any more. These are the "
                     "numbers to report. The development results above are not affected.")
            else:
                t_ = test_m[(test_m["operating_point"] == op) & (test_m["group_type"].isin(["all", "hospital"])) & test_m["model"].isin(have)]
                frame(pd.DataFrame({"Model": t_["model"].map(lambda m: MODELS[m][0]), "Patients": t_["group"].replace({"all": "both hospitals"}),
                                    "% warned in time": (100 * t_["sens_timely"]).round(1),
                                    "95% CI": [f"{100 * a:.1f} to {100 * b:.1f}" for a, b in zip(t_["sens_timely_lo"], t_["sens_timely_hi"])],
                                    "False alarms / 100 days": t_["far_per_100pd"].round(2)}))

# --------------------------------------------------------------------------- #
# 3. Organs
# --------------------------------------------------------------------------- #
with tabs[2]:
    lead("Each septic patient gets an <b>organ group</b>: the organ whose SOFA score got worse in the 24 hours before to 6 "
         "hours after sepsis was flagged (<i>Multi-organ</i> if two or more did, <i>Unattributable</i> if none did). It "
         "describes what was failing, not what caused the sepsis.")
    if need(assign, "The organ grouping"):
        g1, g2 = st.columns([1, 1.3])
        with g1:
            ok = gate["passes"]
            hosp = ", ".join(f"hospital {k_}: {100 * v:.0f}%" for k_, v in gate["by_hospital"].items())
            verdict(f"<b>Gate check {'passed' if ok else 'not passed'}.</b> {100 * gate['overall_share']:.0f}% of septic patients "
                    f"have an organ group ({hosp}). The plan needs at least 50% in each hospital.", GOOD if ok else ALERT)
            note("Brain (GCS) and urine output are not in the data. Lungs are scored only when FiO2 was recorded; circulation "
                 "only reaches 1 point because vasopressor use is not recorded.")
            with st.expander("How each organ is scored (SOFA)"):
                frame(pd.DataFrame({"Organ": ORGAN5, "Scored from": [ORGANS[o][1] for o in ORGAN5],
                                    "SOFA 1 / 2 / 3 / 4 when": [ORGANS[o][2] for o in ORGAN5]}))
        with g2:
            cnt = assign.groupby(["organ_group", "hospital"]).size().reset_index(name="patients")
            fig = px.bar(cnt, y="organ_group", x="patients", color="hospital", orientation="h", barmode="group",
                         category_orders={"organ_group": [o for o in ORGANS if o in set(cnt["organ_group"])]},
                         color_discrete_map=HOSP, labels={"organ_group": "", "patients": "septic patients", "hospital": "hospital"})
            fig.update_layout(title="Septic patients per organ group: hospital A vs B")
            show(fig, 420, name="organ_groups_by_hospital")
        st.markdown("#### Are some organ groups warned later?")
        om = org_m[(org_m["model"] == model) & (org_m["operating_point"] == op)] if org_m is not None else None
        if need(om, "The per-organ result"):
            ab_chart(om[(om["hospital"] != "all") & (om["group"] != "all septic")], list(ORGANS), ORGANS,
                     f"Warned in time by organ group: hospital A vs B ({MNAME}, {SETTINGS[op]})", "warned_by_organ_AB")
            big = wilson_cols(om[(om["hospital"] == "all") & ~om["group"].isin(["all septic", "Unattributable"]) & (om["n_septic"] >= 20)])
            if len(big) >= 2:
                txt, low = gap_sentence(big)
                verdict("Both hospitals together: " + txt, ORGANS.get(low, (TEAL,))[0])
            c1, c2 = st.columns([1.1, 1])
            with c1:
                km = csv(str(F("evaluation", "organs", "organ_km.csv")))
                if km is not None:
                    fig = px.line(km, x="hours_since_window_start", y="share_not_yet_alerted", color="group", line_shape="hv",
                                  color_discrete_map={**{o: v[0] for o, v in ORGANS.items()}, "all septic": INK},
                                  labels={"hours_since_window_start": "hours into the last 24 h before sepsis",
                                          "share_not_yet_alerted": "not yet alerted", "group": ""})
                    fig.update_traces(selector=dict(name="all septic"), line=dict(dash="dash", width=3))
                    rep = js(str(F("evaluation", "organs", "organ_report.json"))) or {}
                    p = rep.get("logrank", {}).get("p_value")
                    fig.update_layout(title="Time to first alert (LightGBM, balanced)" + (f", log-rank p = {p:.2g}" if p is not None else ""),
                                      yaxis_range=[0, 1.02])
                    show(fig, 420, legend_side=True, name="time_to_alert_by_organ")
            with c2:
                fq = csv(str(F("evaluation", "organs", "organ_frequency.csv")))
                if fq is not None:
                    fig = px.scatter(fq, x="median_hours_since_measured", y="sens_timely_%", size="n_septic", color="organ",
                                     text="organ", size_max=40, color_discrete_map={o: v[0] for o, v in ORGANS.items()},
                                     labels={"median_hours_since_measured": "hours since the key test, at sepsis",
                                             "sens_timely_%": "% warned in time"})
                    fig.update_traces(textposition="top center", cliponaxis=False)
                    fig.update_yaxes(range=[max(0, float(fq["sens_timely_%"].min()) - 8), 108])
                    fig.update_layout(title="Is the key test done rarely?")
                    show(fig, 420, legend=False, name="organ_test_frequency")
            if fq is not None:
                frame(fq.rename(columns={"organ": "Organ", "key_measurement": "Key test", "n_septic": "Patients",
                                         "sens_timely_%": "% warned", "median_measurements_24h": "Tests / 24 h",
                                         "median_hours_since_measured": "Hours since test", "median_lead_any_h": "Median warning (h)"})
                      [["Organ", "Key test", "Patients", "% warned", "Tests / 24 h", "Hours since test", "Median warning (h)"]].round(1))
                note("If groups whose key test is older are warned less often, rare testing helps explain the difference.")
        with st.expander("How stable is the grouping? (other windows and carry-forward limits)"):
            sens = csv(str(F("organs", "organ_sensitivity.csv")))
            if sens is not None:
                frame(sens.drop(columns=["primary"]).rename(columns={"lab_ffill_h": "lab carried forward (h)", "window": "window",
                                                                     "attributable_%": "% with an organ group",
                                                                     "agreement_with_primary_%": "% same group as main setting"}))

# --------------------------------------------------------------------------- #
# 4. Vital-sign patterns
# --------------------------------------------------------------------------- #
with tabs[3]:
    lead("A second grouping that uses only the vital signs in the 24 hours before sepsis, so almost every patient gets one.")
    pm = pat_m[(pat_m["model"] == model) & (pat_m["operating_point"] == op)] if pat_m is not None else None
    if need(pm, "The per-pattern result"):
        c1, c2 = st.columns([1.35, 1])
        with c1:
            ab_chart(pm[(pm["hospital"] != "all") & (pm["group"] != "all septic")], list(PATTERNS), PATTERNS,
                     f"Warned in time by vital-sign pattern: hospital A vs B ({MNAME}, {SETTINGS[op]})", "warned_by_pattern_AB")
        with c2:
            frame(pd.DataFrame({"Pattern": list(PATTERNS), "Rule (24 h before sepsis)": [v[1] for v in PATTERNS.values()]}))
            note("The patterns use the same vital signs the model reads, so 'No abnormal vitals' patients are hard to warn "
                 "almost by definition.")
        big = wilson_cols(pm[(pm["hospital"] == "all") & ~pm["group"].isin(["all septic", "unassigned"]) & (pm["n_septic"] >= 20)])
        if len(big) >= 2:
            txt, low = gap_sentence(big)
            verdict("Both hospitals together: " + txt, PATTERNS.get(low, (TEAL,))[0])
        pas = csv(str(F("patterns", "pattern_assignments.csv")))
        if assign is not None and pas is not None:
            x = assign[["Patient_ID", "organ_group"]].merge(pas[["Patient_ID", "pattern"]], on="Patient_ID")
            ct = pd.crosstab(x["pattern"], x["organ_group"])
            ct = ct.reindex(index=[p for p in PATTERNS if p in ct.index], columns=[o for o in ORGANS if o in ct.columns])
            fig = px.imshow(ct, text_auto=True, color_continuous_scale=[[0, "#F7FAFA"], [1, TEAL]], aspect="auto",
                            labels={"x": "organ group", "y": "vital-sign pattern", "color": "patients"})
            fig.update_layout(title="How the vital-sign patterns line up with the organ groups", coloraxis_showscale=False)
            show(fig, 380, legend=False, name="patterns_vs_organs")
        fbp = csv(str(F("evaluation", "frequency_by_pattern.csv")))
        fa = csv(str(F("evaluation", "frequency_adjusted.csv")))
        if fbp is not None:
            with st.expander("Is it measurement frequency? (vital-sign patterns)"):
                frame(fbp[["pattern", "n_septic", "sens_timely_%", "lo", "hi", "trigger_vitals", "median_trigger_obs_per_h",
                           "median_lab_hours_per_h", "median_label_switch_on_hour"]].round(2))
                if fa is not None:
                    st.markdown(f"Odds ratios for being warned in time (reference: **{fa['reference_pattern'].iloc[0]}**), "
                                "before and after adjusting for measurement rates, switch-on hour and hospital:")
                    frame(fa[["model", "term", "odds_ratio", "or_lo", "or_hi"]].round(3))

# --------------------------------------------------------------------------- #
# 5. Checks & comparisons
# --------------------------------------------------------------------------- #
with tabs[4]:
    which = st.radio("Check", ["Time in the ICU", "Hospital transfer", "Group-aware models", "Feature groups (ablation)"],
                     horizontal=True, label_visibility="collapsed")
    if which == "Time in the ICU":
        lead("Long-stay patients get sepsis more often and their risk drifts up, so a model can look good just by alarming "
             "on long stays. Comparing patients only at the same hour of their stay removes that shortcut.")
        wh = csv(str(F("evaluation", "within_hour_discrimination.csv")))
        if need(wh, "The time check"):
            show_m = have + (["p_lgbm_time_only"] if "p_lgbm_time_only" in set(wh["model"]) else [])
            names = {**{m: v[0] for m, v in MODELS.items()}, "p_lgbm_time_only": "Time in ICU only"}
            w = wh[wh["model"].isin(show_m) & wh["bin"].isin(["all hours (pooled)", "hour-stratified"])].copy()
            w["model"] = w["model"].map(names)
            w["bin"] = w["bin"].map({"all hours (pooled)": "all hours compared", "hour-stratified": "same hour of stay only"})
            fig = px.bar(w, x="model", y="auroc", color="bin", barmode="group", text=w["auroc"].round(2),
                         color_discrete_map={"all hours compared": "#9FB3B8", "same hour of stay only": TEAL},
                         category_orders={"model": [names[m] for m in show_m]},
                         labels={"auroc": "AUROC (0.5 = guessing)", "model": "", "bin": ""})
            fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED)
            fig.update_layout(yaxis_range=[0.4, 1], title="Model skill, before and after removing the time effect")
            show(fig, 420, name="time_check")
            t = wh.pivot(index="model", columns="bin", values="auroc")
            if "p_lgbm_time_only" in t.index and model in t.index:
                keep = t.loc[model, "hour-stratified"]
                verdict(f"Time alone: {t.loc['p_lgbm_time_only', 'all hours (pooled)']:.2f} overall, "
                        f"{t.loc['p_lgbm_time_only', 'hour-stratified']:.2f} at the same hour (guessing). {MNAME}: "
                        f"{t.loc[model, 'all hours (pooled)']:.2f} to {keep:.2f}, "
                        + ("so most of its skill is real." if keep >= 0.75 else "so a large part of its skill is time in the ICU."),
                        TEAL if keep >= 0.75 else ALERT)
        tp = csv(str(F("evaluation", "time_profile.csv")))
        if tp is not None:
            a = tp[(tp["model"] == model) & (tp["operating_point"] == op) & tp["kind"].isin(["septic_timely_by_switch_on", "nonseptic_far_by_stay"])]
            if len(a):
                order = ["1-12", "13-24", "25-48", "49-72", ">72"]
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Septic: % warned in time", "Not septic: false alarms per 100 days"))
                s1 = a[a["kind"] == "septic_timely_by_switch_on"].set_index("bin").reindex(order)
                s2 = a[a["kind"] == "nonseptic_far_by_stay"].set_index("bin").reindex(order)
                fig.add_trace(go.Bar(x=order, y=s1["value"], marker_color=MCOL,
                                     text=[f"n={int(n)}" if pd.notna(n) else "" for n in s1["n"]], textposition="outside"), 1, 1)
                fig.add_trace(go.Bar(x=order, y=s2["value"], marker_color=ALERT), 1, 2)
                fig.update_xaxes(title_text="hours in the ICU before sepsis / length of stay")
                show(fig, 360, legend=False, name="time_profile")
                note("If both rise with time, the model is partly reacting to time in the ICU.")
    elif which == "Hospital transfer":
        lead("A model built in one hospital, tested on the other, with the alert threshold the first hospital would choose.")
        trm = csv(str(F("evaluation", "transfer_metrics.csv")))
        if need(trm, "The hospital check"):
            t = trm[trm["operating_point"] == (op if op in set(trm["operating_point"]) else "far5")].copy()
            t = t[~t["setting"].str.contains("re-set")]
            t["who"] = np.where(t["setting"].str.startswith("within"), "built in the same hospital", "built in the other hospital")
            fig = px.bar(t.sort_values(["target", "who"], ascending=[True, False]), x="target", y=t["sens_timely"] * 100,
                         color="who", barmode="group", text=[f"{v:.0f}%" for v in t["sens_timely"] * 100],
                         color_discrete_map={"built in the same hospital": "#9FB3B8", "built in the other hospital": TEAL},
                         category_orders={"target": sorted(t["target"].unique())},
                         labels={"target": "hospital the patients are from", "y": "% warned in time", "who": ""})
            fig.update_layout(yaxis_range=[0, 105], title="LightGBM on each hospital's patients")
            show(fig, 380, name="hospital_transfer")
            frame(t[["direction", "setting", "row_auroc", "far_per_100pd", "sens_timely", "util_norm"]].round(3))
    elif which == "Group-aware models":
        lead("Do models that know the patient groups during training warn more patients than LightGBM? Same patients, each "
             "model at its own balanced threshold.")
        view = st.radio("Group patients by", ["Organ", "Vital-sign pattern"], horizontal=True)
        cmp_ = csv(str(F("evaluation", "organs", "organ_model_comparison.csv") if view == "Organ" else F("evaluation", "pattern_model_comparison.csv")))
        if need(cmp_, "The comparison"):
            known = list(ORGANS) + list(PATTERNS)
            groups = sorted(dict.fromkeys(cmp_["group"]), key=lambda x: -1 if x == "all patterns with an expert" else (known.index(x) if x in known else 99))
            g = st.radio("Patients", groups, horizontal=True, format_func=lambda x: "All groups" if x == "all patterns with an expert" else x)
            c = cmp_[(cmp_["group"] == g) & (cmp_["role"] != "global reference")].copy()
            c = c[c["model"].isin(AWARE) | c["role"].str.startswith("oracle")]
            c["name"] = [("Oracle (knows the group)" if r.role.startswith("oracle") else AWARE[r.model]) for r in c.itertuples()]
            c = c.iloc[::-1]
            if len(c):
                colr = np.where(c["diff_lo"] > 0, GOOD, np.where(c["diff_hi"] < 0, ALERT, MUTED))
                fig = go.Figure(go.Scatter(x=c["diff_vs_reference_pts"], y=c["name"], mode="markers", marker=dict(size=13, color=colr),
                                           error_x=dict(type="data", symmetric=False, array=c["diff_hi"] - c["diff_vs_reference_pts"],
                                                        arrayminus=c["diff_vs_reference_pts"] - c["diff_lo"], color=MUTED, thickness=1.5)))
                fig.add_vline(x=0, line_dash="dash", line_color=INK)
                lo_, hi_ = min(float(c["diff_lo"].min()), 0), max(float(c["diff_hi"].max()), 0)
                pad = max(2.0, 0.12 * (hi_ - lo_))
                fig.update_xaxes(range=[lo_ - pad, hi_ + pad])
                fig.update_layout(title="Difference from LightGBM in % warned in time", xaxis_title="percentage points (right = better)")
                show(fig, 150 + 62 * len(c), legend=False, name="group_aware_forest")
            note("Green = clearly better, orange = clearly worse, grey = no clear difference. The oracle uses each patient's "
                 "group, known only after sepsis: the most that knowing the group could help.")
    else:
        lead("The same LightGBM retrained with groups of inputs removed: how much does each kind of information matter?")
        abl = csv(str(F("models", "ablation_oof_metrics.csv")))
        if need(abl, "The ablation"):
            names = {"no_time": "without time in ICU", "no_meas": "without measurement frequency", "phys_only": "measured values only",
                     "behavior_only": "time + measurement frequency only", "time_only": "time in ICU only", "full": "all inputs"}
            a = abl[abl["group"] == "dev_all"].copy()
            a["variant"] = a["variant"].map(names).fillna(a["variant"])
            fig = px.bar(a.sort_values("auroc"), x="auroc", y="variant", orientation="h", text=a.sort_values("auroc")["auroc"].round(3),
                         color_discrete_sequence=[TEAL], labels={"auroc": "AUROC", "variant": ""})
            fig.update_layout(xaxis_range=[0.5, 1], title="LightGBM with groups of inputs removed (development, out-of-fold)")
            show(fig, 360, legend=False, name="ablation")
            frame(a[["variant", "n_features", "auroc", "auprc"]].round(4))
            note("Differences under about 0.005 AUROC are within noise. 'Measured values only' still carries some "
                 "measurement information, because a value is missing when nothing was measured recently.")

# --------------------------------------------------------------------------- #
# 6. Explainability
# --------------------------------------------------------------------------- #


def kind_of(f):
    if f in ("ICULOS", "HospAdmTime_slog"):
        return "time in hospital"
    if f in ("Age", "Gender", "unit"):
        return "patient details"
    if f.endswith("_age") or f.startswith(("n_vital_obs", "n_lab_")) or f == "lab_age_any":
        return "how often things are measured"
    return "measured values"


KIND_COL = {"measured values": TEAL, "how often things are measured": "#D07A1F", "time in hospital": ALERT, "patient details": "#9FB3B8"}

with tabs[5]:
    lead("Which inputs drive the LightGBM predictions? SHAP values measure how much each input pushed the risk up or down "
         "in the 12 hours before sepsis.")
    sh = csv(str(F("evaluation", "shap_by_pattern.csv")))
    if need(sh, "The SHAP values"):
        pr = predictor()
        groups = ["all septic patients"] + [g for g in sh["group"].unique() if not g.startswith("negative")]
        g = st.radio("Patients", groups, horizontal=True)
        pos = sh[~sh["group"].str.startswith("negative")] if g == "all septic patients" else sh[sh["group"] == g]
        agg = pos.assign(w=pos["mean_abs_shap"] * pos["n_rows"], s=pos["mean_shap"] * pos["n_rows"]).groupby("feature")[["w", "s", "n_rows"]].sum()
        agg = pd.DataFrame({"importance": agg["w"] / agg["n_rows"], "direction": agg["s"] / agg["n_rows"]})
        agg = agg.sort_values("importance", ascending=False).head(15).reset_index()
        agg["input"] = agg["feature"].map(pr.nice_name)
        agg["kind"] = agg["feature"].map(kind_of)
        fig = px.bar(agg.iloc[::-1], x="importance", y="input", color="kind", orientation="h", color_discrete_map=KIND_COL,
                     hover_data=["feature", "direction"], labels={"importance": "average influence (mean |SHAP|)", "input": "", "kind": ""})
        fig.update_layout(title=f"The 15 most influential inputs ({g})")
        show(fig, 540, name="shap_top_features")
        note("Influence is not cause: frequent lab testing is often a sign that clinicians were already worried.")
        imp = csv(str(F("models", "lgbm_feature_importance.csv")))
        if imp is not None:
            imp = imp.rename(columns={imp.columns[0]: "feature"})
            imp["kind"] = imp["feature"].map(kind_of)
            gs = imp.groupby("kind")["gain_share"].sum().reset_index()
            fig = px.pie(gs, names="kind", values="gain_share", color="kind", color_discrete_map=KIND_COL, hole=0.45)
            fig.update_layout(title="Share of the model's decisions by kind of input (split gain)")
            show(fig, 380, name="gain_by_kind")

# --------------------------------------------------------------------------- #
# 7. Data
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Counting measurements ...")
def measured_share(out):
    import pyarrow.parquet as pq
    path = Path(out, "clean", "hourly_clean.parquet")
    if not path.exists():
        return None
    cols = set(pq.ParquetFile(path).schema.names)
    hosp = pd.read_parquet(path, columns=["hospital"])["hospital"].astype(str).to_numpy()
    vit = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
    rows = []
    for v in [c for c in vit + ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
                                "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
                                "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"] if c in cols]:
        s = pd.read_parquet(path, columns=[v])[v].notna()
        share = s.groupby(hosp).mean() * 100
        rows.append({"variable": v, "kind": "vital" if v in vit else "lab", **{f"Hospital {h}": float(x) for h, x in share.items()}})
    return pd.DataFrame(rows)


with tabs[6]:
    t1 = csv(str(F("report", "tables", "Table1_cohort_characteristics.csv")))
    st.markdown("#### Patients, overall and by hospital")
    if t1 is not None:
        frame(t1)
    else:
        note("Table 1 appears after the report_figures step (python run.py).")
    flow = F("report", "figures", "Fig01_patient_flow.png")
    if flow.exists():
        img(flow, width=760)
    which = st.radio("Data view", ["How often things are measured", "Septic vs not septic", "Correlation", "Missing data", "Cleaning"],
                     horizontal=True, label_visibility="collapsed")
    G = ("evaluation", "figures")
    if which == "How often things are measured":
        ms_ = measured_share(str(OUT))
        if need(ms_, "The measurement counts"):
            hc = [c for c in ms_.columns if c.startswith("Hospital")]
            fig = px.imshow(ms_.set_index("variable")[hc], aspect="auto", text_auto=".0f",
                            color_continuous_scale=[[0, "#F7FAFA"], [1, TEAL]], labels={"color": "% of hours"})
            fig.update_layout(title="Share of patient-hours with a measurement: hospital A vs B")
            show(fig, max(560, 18 * len(ms_)), legend=False, name="measurement_rate_by_hospital")
    elif which == "Septic vs not septic":
        ds = csv(str(F(*G, "distributions.csv")))
        if need(ds, "The distributions"):
            vs = list(dict.fromkeys(ds["variable"]))[:10]
            fig = make_subplots(rows=2, cols=5, subplot_titles=vs)
            col = {"septic, last 24 h before switch-on": ALERT, "non-septic": TEAL}
            for i, v in enumerate(vs):
                for r_ in ds[ds["variable"] == v].itertuples():
                    fig.add_trace(go.Box(q1=[r_.q1], median=[r_.median], q3=[r_.q3], lowerfence=[r_.p5], upperfence=[r_.p95],
                                         x=[r_.group.split(",")[0]], marker_color=col.get(r_.group), showlegend=(i == 0),
                                         name="septic (24 h before)" if r_.group.startswith("septic") else "not septic"), i // 5 + 1, i % 5 + 1)
            fig.update_xaxes(showticklabels=False)
            fig.update_layout(title="Vital signs and labs: septic (24 h before sepsis) vs not septic")
            show(fig, 560, name="septic_vs_nonseptic")
    elif which == "Correlation":
        cr = csv(str(F(*G, "correlation_spearman.csv")))
        if need(cr, "The correlation matrix"):
            cr = cr.set_index(cr.columns[0]).rename_axis(None)
            fig = px.imshow(cr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(title="Correlation between vital signs and labs (Spearman)", coloraxis_colorbar=dict(title=""))
            show(fig, 760, legend=False, name="correlation_heatmap")
    elif which == "Missing data":
        ms = csv(str(F(*G, "missing_share.csv")))
        if need(ms, "The missing-data summary"):
            m_ = ms[ms["group"].isin(["hospital A", "hospital B"])]
            order = ms[ms["group"] == "all"].sort_values("missing_pct")["variable"].tolist()
            fig = px.bar(m_, x="missing_pct", y="variable", color="group", barmode="group", orientation="h",
                         color_discrete_map={"hospital A": HOSP["A"], "hospital B": HOSP["B"]},
                         category_orders={"variable": order}, labels={"missing_pct": "% of hours not measured", "variable": "", "group": ""})
            fig.update_layout(title="Missing data: hospital A vs B")
            show(fig, 900, name="missing_data_by_hospital")
    else:
        rep = js(str(F("clean", "cleaning_report.json")))
        if need(rep, "The cleaning report"):
            rules = pd.DataFrame(rep["rules"])
            frame(rules[rules["n_changed"] > 0].sort_values("n_changed", ascending=False))
            for n_ in rep.get("notes", []):
                st.markdown(f"- {n_}")
            note("Impossible values were set to missing; no rows were deleted.")

# --------------------------------------------------------------------------- #
# 8. Report figures
# --------------------------------------------------------------------------- #
with tabs[7]:
    rd = F("report")
    lead("Publication-quality figures (300 dpi PNG and vector PDF) and tables for the report and a paper, numbered in "
         "reading order with captions. Re-made by `python run.py` (step report_figures).")
    z = rd / "report_package.zip"
    if not (rd / "captions.md").exists():
        st.info("The report figures have not been made yet. Run `python run.py`, then reload this page.")
    else:
        st.download_button("Download everything (all figures, tables and captions, .zip)", z.read_bytes(),
                           file_name="sepsis_report_figures.zip", mime="application/zip")
        caps = (rd / "captions.md").read_text()
        cap_of = {}
        for block in caps.split("\n\n"):
            if block.startswith("**Figure") or block.startswith("**Table"):
                key = block.split("`")[1].split("/")[-1].rsplit(".", 1)[0] if "`" in block else ""
                cap_of[key] = block.split("  \n")[0]
        st.markdown("#### Tables")
        for f in sorted((rd / "tables").glob("*.csv")):
            with st.expander(cap_of.get(f.stem, f.stem).replace("**", "")):
                frame(pd.read_csv(f))
                st.download_button(f"Download {f.name}", f.read_bytes(), file_name=f.name, key=f"dl_{f.name}")
        st.markdown("#### Figures")
        figs = sorted((rd / "figures").glob("Fig*.png"))
        for i in range(0, len(figs), 2):
            cc = st.columns(2)
            for col, f in zip(cc, figs[i:i + 2]):
                with col:
                    img(f)
                    note(cap_of.get(f.stem, f.stem).replace("**", ""))
                    b1, b2 = st.columns(2)
                    b1.download_button("PNG (300 dpi)", f.read_bytes(), file_name=f.name, key=f"png_{f.name}")
                    pdf = f.with_suffix(".pdf")
                    if pdf.exists():
                        b2.download_button("PDF (vector)", pdf.read_bytes(), file_name=pdf.name, key=f"pdf_{f.name}")
    with st.expander("Methods and limits (for the report)"):
        st.markdown("""
**Task.** Every hour before sepsis is flagged, predict whether the PhysioNet 2019 sepsis label switches on within 12 hours.
**Models.** LightGBM, logistic regression and a GRU on the same causal hourly features; the ensemble is the mean of the three
models' percentiles. Five patient-level folds give out-of-fold predictions; a stratified 20% test set is used once at the end.
**Alerts.** Thresholds are chosen on development data for about 2, 5 or 10 false alarms per 100 non-septic patient-days.
**Organ groups.** Partial SOFA (lungs, clotting, liver, circulation, kidneys) in the window 24 h before to 6 h after the label.
**Limits.** Retrospective, two hospitals; no GCS, urine output, PaO2 or treatment data; organ and vital-sign groups are known
only in hindsight and are never model inputs; the utility score is adapted and not comparable with the Challenge leaderboard.
**Not a clinical tool.**
""")
