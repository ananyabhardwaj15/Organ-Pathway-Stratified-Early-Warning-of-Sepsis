#!/usr/bin/env python3
"""
run.py
==========
Runs the whole pipeline (every script in backend/) in the right order with one command.

    python run.py                                  # everything: the CSV in data/ -> outputs/
    python run.py --input path/to/Dataset.csv      # a data file somewhere else
    python run.py --dashboard                      # ... and open the dashboard at the end
    python run.py --list                           # show the steps and which are done
    python run.py --from gru                       # re-run from a step onwards
    python run.py --only shap                      # run just these steps
    python run.py --fresh                          # ignore earlier progress, run everything
    python run.py --final                          # ALSO the one-time test-set evaluation

Resuming: finished steps are recorded in <outputs>/run_state.json. If a step fails or you stop the
run (Ctrl+C), just run the same command again: it continues with the first unfinished step. When a
step re-runs, every step after it re-runs too, because later steps read its outputs.
The full console output is appended to <outputs>/run_log.txt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
B = "backend"
DATA_DIR = ROOT / "data"

# name, script, arguments ({I} = input file, {O} = outputs folder)
STEPS = [
    ("clean", "clean_data.py", ["--input", "{I}", "--outdir", "{O}/clean"]),
    ("splits", "make_splits.py", ["--patients", "{O}/clean/patients.csv", "--outdir", "{O}/splits"]),
    ("pattern_features", "pattern_features.py", ["--hourly", "{O}/clean/hourly_clean.parquet",
                                                   "--folds", "{O}/splits/folds.csv", "--outdir", "{O}/patterns"]),
    ("patterns", "pattern_clustering.py", ["--features", "{O}/patterns/pattern_features.parquet",
                                             "--outdir", "{O}/patterns"]),
    ("clustering_gmm", "pattern_clustering.py", ["--features", "{O}/patterns/pattern_features.parquet",
                                                   "--outdir", "{O}/patterns/exploratory_gmm", "--method", "gmm"]),
    ("features", "train_models.py", ["features", "--outputs", "{O}"]),
    ("baselines", "train_models.py", ["baselines", "--outputs", "{O}"]),
    ("ablation", "train_models.py", ["ablation", "--outputs", "{O}"]),
    ("gru", "train_models.py", ["gru", "--outputs", "{O}", "--head", "single"]),
    ("gru_mtl", "train_models.py", ["gru", "--outputs", "{O}", "--head", "mtl"]),
    ("gru_moe", "train_models.py", ["gru", "--outputs", "{O}", "--head", "moe"]),
    ("pattern_experts", "train_models.py", ["pattern", "--outputs", "{O}"]),
    ("transfer_models", "train_models.py", ["transfer", "--outputs", "{O}"]),
    ("timecontrol", "evaluate.py", ["timecontrol", "--outputs", "{O}"]),
    ("performance", "evaluate.py", ["performance", "--outputs", "{O}"]),
    ("timeprofile", "evaluate.py", ["timeprofile", "--outputs", "{O}"]),
    ("transfer_eval", "evaluate.py", ["transfer", "--outputs", "{O}"]),
    ("frequency", "evaluate.py", ["frequency", "--outputs", "{O}"]),
    ("shap", "evaluate.py", ["shap", "--outputs", "{O}"]),
    ("compare", "evaluate.py", ["compare", "--outputs", "{O}"]),
    ("plots", "evaluate.py", ["plots", "--outputs", "{O}"]),
    # organ groups (partial SOFA). Placed last so adding them never re-runs the long training steps.
    ("organ_pathways", "organ_pathways.py", ["--outputs", "{O}"]),
    ("organ_experts", "train_models.py", ["pattern", "--outputs", "{O}", "--labels", "organ"]),
    ("organ_results", "evaluate.py", ["organs", "--outputs", "{O}"]),
    # integrated ensemble (LightGBM + logistic regression + GRU) and the results that include it
    ("ensemble", "train_models.py", ["ensemble", "--outputs", "{O}"]),
    ("ensemble_performance", "evaluate.py", ["performance", "--outputs", "{O}", "--models", "p_ensemble"]),
    ("ensemble_timecontrol", "evaluate.py", ["timecontrol", "--outputs", "{O}", "--models", "p_ensemble", "--no-adjust"]),
    ("refresh_timeprofile", "evaluate.py", ["timeprofile", "--outputs", "{O}"]),
    ("refresh_plots", "evaluate.py", ["plots", "--outputs", "{O}"]),
    ("refresh_organs", "evaluate.py", ["organs", "--outputs", "{O}"]),
    ("report_figures", "report_figures.py", ["--outputs", "{O}"]),
]
FINAL = [  # the ONE-TIME test-set evaluation, only with --final
    ("final_performance", "evaluate.py", ["performance", "--outputs", "{O}", "--split", "test", "--final"]),
    ("final_transfer", "evaluate.py", ["transfer", "--outputs", "{O}", "--split", "test", "--final"]),
    ("final_compare", "evaluate.py", ["compare", "--outputs", "{O}", "--split", "test", "--final"]),
    ("final_organs", "evaluate.py", ["organs", "--outputs", "{O}", "--split", "test", "--final"]),
    ("final_report_figures", "report_figures.py", ["--outputs", "{O}"]),
]
GRU_STEPS = {"gru", "gru_mtl", "gru_moe"}


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def find_input() -> Path:
    """The single .csv / .zip file in data/ (so you only have to drop your file there)."""
    files = sorted(f for f in DATA_DIR.glob("*") if f.suffix.lower() in (".csv", ".zip")) if DATA_DIR.exists() else []
    if len(files) == 1:
        return files[0].resolve()
    if not files:
        sys.exit(f"ERROR: no .csv or .zip file found in {DATA_DIR}\n"
                 "Put your dataset file in the data folder, or pass --input path/to/your_file.csv")
    sys.exit(f"ERROR: several data files in {DATA_DIR}: {[f.name for f in files]}\n"
             "Keep only one there, or choose one with --input data/<name>")


def load_state(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {"done": {}, "input": None}


def save_state(path: Path, state: dict) -> None:
    json.dump(state, open(path, "w"), indent=2)


def run_step(name, script, argv, log) -> bool:
    cmd = [sys.executable, "-u", str(ROOT / B / script)] + argv
    log(f"\n{'=' * 78}\n>>> {name}:  python {B}/{script} {' '.join(argv)}\n{'=' * 78}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        log(line.rstrip("\n"))
    rc = proc.wait()
    mins = (time.time() - t0) / 60
    log(f"<<< {name}: {'OK' if rc == 0 else f'FAILED (exit code {rc})'} after {mins:.1f} min")
    return rc == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None,
                    help="raw CSV (or a zip with one CSV). Default: the one .csv / .zip file in the data/ folder")
    ap.add_argument("--outputs", type=Path, default=Path("outputs"), help="outputs folder (default outputs)")
    ap.add_argument("--from", dest="start", default=None, help="re-run from this step onwards")
    ap.add_argument("--only", nargs="+", default=None, help="run only these steps")
    ap.add_argument("--skip", nargs="+", default=[], help="never run these steps (e.g. --skip gru_mtl gru_moe)")
    ap.add_argument("--fresh", action="store_true", help="forget earlier progress and run everything")
    ap.add_argument("--final", action="store_true", help="also run the one-time final test-set evaluation")
    ap.add_argument("--dashboard", action="store_true", help="open the dashboard when finished")
    ap.add_argument("--list", action="store_true", help="list the steps and their status, then exit")
    args = ap.parse_args()

    missing = sorted({f"{B}/{s[1]}" for s in STEPS + FINAL if not (ROOT / B / s[1]).exists()})
    if not (ROOT / "app.py").exists():
        missing.append("app.py")
    if missing:
        sys.exit(f"ERROR: these files are missing next to run.py: {missing}\n"
                 "Expected layout: run.py, app.py, backend/clean_data.py ... backend/evaluate.py, data/<your CSV>")

    names = [s[0] for s in STEPS + FINAL]
    for n in ([args.start] if args.start else []) + (args.only or []) + args.skip:
        if n not in names:
            sys.exit(f"ERROR: unknown step '{n}'. Steps: {', '.join(names)}")

    out = args.outputs.resolve()
    state_path = out / "run_state.json"
    out.mkdir(parents=True, exist_ok=True)
    state = {"done": {}, "input": None} if args.fresh else load_state(state_path)

    steps = STEPS + (FINAL if args.final or (args.only and set(args.only) & {s[0] for s in FINAL}) else [])
    if args.list:
        print(f"outputs: {out}\n")
        for n, script, _ in STEPS + FINAL:
            mark = "done" if n in state["done"] else "-"
            print(f"  {n:18s} {B}/{script:26s} {mark}  {state['done'].get(n, '')}")
        return

    inp = args.input.resolve() if args.input else find_input()
    log = Tee(out / "run_log.txt")
    log(f"\n##### run.py started {time.strftime('%Y-%m-%d %H:%M:%S')} | outputs: {out}")

    if not inp.exists():
        sys.exit(f"ERROR: data file not found: {inp}\nPut your file there or pass --input path/to/your_file")
    if state["input"] not in (None, str(inp)) and not args.only:
        log(f"[note] input changed ({state['input']} -> {inp}): starting again from the first step")
        state["done"] = {}
    state["input"] = str(inp)

    skip = set(args.skip)
    if importlib.util.find_spec("torch") is None and GRU_STEPS - skip:
        log("[warning] torch is not installed, so the GRU steps are skipped. Install it with\n"
            "          pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "          and run `python run.py --from gru` afterwards.")
        skip |= GRU_STEPS

    start_i = names.index(args.start) if args.start else None
    rerun_rest = False
    t_all = time.time()
    for i, (n, script, argv) in enumerate(steps):
        if n in skip or (args.only and n not in args.only):
            continue
        forced = (start_i is not None and i >= start_i) or bool(args.only)
        if n in state["done"] and not forced and not rerun_rest:
            log(f"[skip] {n} (already done {state['done'][n]})")
            continue
        argv = [a.replace("{I}", str(inp)).replace("{O}", str(out)) for a in argv]
        # un-mark this step (and, unless --only, every later step) BEFORE running it, so a failure or Ctrl+C
        # never leaves an old "done" mark behind
        for later in ([n] if args.only else names[names.index(n):]):
            state["done"].pop(later, None)
        save_state(state_path, state)
        rerun_rest = rerun_rest or not args.only    # later steps read this step's outputs
        try:
            ok = run_step(n, script, argv, log)
        except KeyboardInterrupt:
            log(f"\n[stopped] during {n}. Run the same command again to continue from here.")
            sys.exit(130)
        if not ok:
            log(f"\n[failed] step '{n}'. Fix the error above, then run the same command again to continue "
                f"from this step. Full log: {out / 'run_log.txt'}")
            sys.exit(1)
        state["done"][n] = time.strftime("%Y-%m-%d %H:%M")
        save_state(state_path, state)

    log(f"\n##### all requested steps finished in {(time.time() - t_all) / 60:.1f} min | results in {out}")
    if not args.final:
        log("The final test-set evaluation was NOT run (by design). When everything is frozen: python run.py --final")
    if args.dashboard:
        log("Opening the dashboard (Ctrl+C to stop it) ...")
        env = {**os.environ, "SEPSIS_OUTPUTS": str(out)}
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], cwd=ROOT, env=env)
    else:
        log("Open the dashboard: python run.py --dashboard   (finished steps are skipped, so it opens straight away)")


if __name__ == "__main__":
    main()
