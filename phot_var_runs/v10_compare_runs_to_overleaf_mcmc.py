#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, shutil, sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# ----------------- small utils -----------------
def tex_escape(s: str) -> str:
    if s is None: return ""
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))

def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")

def safe_read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def safe_read_csv_rows(p: Path):
    try:
        return len(pd.read_csv(p))
    except Exception:
        return 0

def fmt_num(v, spec):
    try:
        if v is None: return ""
        f = float(v)
        if f != f:  # NaN
            return ""
        return format(f, spec)
    except Exception:
        return ""

def fmt_int(v):
    try:
        if v is None: return ""
        f = float(v)
        if f != f: return ""
        return str(int(round(f)))
    except Exception:
        return ""

def wrap_label(label: str, maxlen: int = 28) -> str:
    """Insert LaTeX line breaks to avoid overfull columns."""
    words = str(label).split()
    lines, cur = [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if len(nxt) <= maxlen:
            cur = nxt
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return r"\begin{tabular}[t]{@{}l@{}}" + r" \\ ".join(tex_escape(x) for x in lines) + r"\end{tabular}"

# ----------------- data collection -----------------
def collect_cleaning_breakdowns(run_dir: Path) -> dict:
    """Pull useful aggregates from the two cleaning summary CSVs."""
    out = dict(
        n_observers=None, n_sessions=None,
        obs_kept_total=None, obs_removed_total=None,
        obs_median_kept=None, sess_kept_total=None, sess_median_kept=None,
        top_removed=None,
        observers_full=None,
    )

    # by observer
    p_obs = run_dir / "cleaning_summary_by_observer.csv"
    if p_obs.exists():
        try:
            df = pd.read_csv(p_obs)
            out["n_observers"] = len(df)
            if "kept" in df:
                out["obs_kept_total"] = int(df["kept"].fillna(0).sum())
                out["obs_median_kept"] = float(df["kept"].median())
            if "removed" in df:
                out["obs_removed_total"] = int(df["removed"].fillna(0).sum())

            # full table (sorted by removed fraction)
            tmp = df.copy()
            if "observer" not in tmp.columns:
                tmp = tmp.rename(columns={tmp.columns[0]: "observer"})
            tmp["kept"] = tmp.get("kept", 0).fillna(0).astype(float)
            tmp["removed"] = tmp.get("removed", 0).fillna(0).astype(float)
            denom = tmp["kept"] + tmp["removed"]
            tmp["rem_frac"] = tmp["removed"] / denom.where(denom > 0, pd.NA)
            tmp_sorted = tmp.sort_values(["rem_frac", "removed", "kept"], ascending=[False, False, True])
            out["observers_full"] = tmp_sorted[["observer","kept","removed","rem_frac"]].to_dict("records")
            out["top_removed"] = tmp_sorted.head(5)[["observer","kept","removed","rem_frac"]].to_dict("records")
        except Exception:
            pass

    # by session
    p_sess = run_dir / "cleaning_summary_by_session.csv"
    if p_sess.exists():
        try:
            df = pd.read_csv(p_sess)
            out["n_sessions"] = len(df)
            if "kept" in df:
                out["sess_kept_total"] = int(df["kept"].fillna(0).sum())
                out["sess_median_kept"] = float(df["kept"].median())
        except Exception:
            pass

    return out

def collect_run(run_dir: Path, label: str) -> dict:
    r = {
        "label": label,
        "dir": str(run_dir),
        "images": {},   # key -> filename we copied into figs/
        "n_input": None, "n_clean": None, "n_removed": None,
        "P1": None, "P1_err": None, "P2": None, "P2_err": None,
        "tspan": None, "cycles": None,
        "band": None, "minP": None, "maxP": None, "nfreq": None,
        "P1_prior": None, "P1_prior_frac": None,
        # legacy knobs (v3 style)
        "per_observer_sigma": None, "sigma_clip": None, "iters": None,
        # v4 QC knobs
        "group": None, "gap_hours": None,
        "per_night_minN": None, "per_night_sigma": None,
        "global_sigma": None, "global_iters": None,
        "roll_window": None, "roll_sigma": None,
        "thin_maxN": None, "err_max": None, "err_median_factor": None,
        "n_qc_sessions": 0, "n_qc_observers": 0, "n_drop_ranges": 0,
        # cleaning breakdowns
        "cleaning": {},
        # model metrics
        "met_single": {}, "met_twin": {},
        # NEW: MCMC posterior results (None unless the run used --mcmc)
        "mcmc": None,
    }

    sj = run_dir / "summary.json"
    summ = safe_read_json(sj)

    if summ:
        # counts (support v3 and v4)
        r["n_input"] = summ.get("n_points_input")
        n_clean_final = summ.get("n_points_clean_final")
        n_clean = summ.get("n_points_clean")  # v3
        r["n_clean"] = n_clean_final if n_clean_final is not None else n_clean

        per = summ.get("periods", {})
        r["P1"] = per.get("P1");            r["P1_err"] = per.get("P1_err_est")
        r["P2"] = per.get("P2");            r["P2_err"] = per.get("P2_err_est")

        # metrics (with backward-compat)
        met = summ.get("metrics", {})
        r["met_single"] = met.get("single_sine", {}) or {}
        r["met_twin"]   = met.get("twin_sine",   {}) or {}
        for k in ("chi2","dof","rchi2","AIC","BIC"):
            if k in met and k not in r["met_twin"]:
                r["met_twin"][k] = met[k]

        tim = summ.get("timing", {})
        r["tspan"]  = tim.get("Tspan_days")
        r["cycles"] = tim.get("cycles_at_P1_over_span")

        # NEW: MCMC posterior results, if the run was made with --mcmc (else None)
        r["mcmc"] = summ.get("mcmc")

        setg = summ.get("settings", {})
        r["band"] = setg.get("band_include")
        r["minP"] = setg.get("minP"); r["maxP"] = setg.get("maxP"); r["nfreq"] = setg.get("nfreq")
        r["P1_prior"] = setg.get("P1_prior"); r["P1_prior_frac"] = setg.get("P1_prior_frac")

        # legacy (if present)
        r["per_observer_sigma"] = setg.get("per_observer_sigma")
        r["sigma_clip"] = setg.get("sigma_clip")
        r["iters"] = setg.get("iters")

        # v4 QC knobs
        r["group"] = setg.get("group")
        r["gap_hours"] = setg.get("gap_hours")
        r["per_night_minN"] = setg.get("per_night_minN")
        r["per_night_sigma"] = setg.get("per_night_sigma")
        r["global_sigma"] = setg.get("global_sigma")
        r["global_iters"] = setg.get("global_iters")
        r["roll_window"] = setg.get("roll_window")
        r["roll_sigma"] = setg.get("roll_sigma")
        r["thin_maxN"] = setg.get("thin_maxN")
        r["err_max"] = setg.get("err_max")
        r["err_median_factor"] = setg.get("err_median_factor")

    # removed points
    removed_csv = run_dir / "removed_points.csv"
    if removed_csv.exists():
        r["n_removed"] = safe_read_csv_rows(removed_csv)
    elif r["n_input"] is not None and r["n_clean"] is not None:
        try:
            r["n_removed"] = max(0, int(r["n_input"]) - int(r["n_clean"]))
        except Exception:
            r["n_removed"] = None

    # QC counts if present
    sess_qc_csv = run_dir / "observer_session_qc.csv"
    if sess_qc_csv.exists():
        try:
            df = pd.read_csv(sess_qc_csv)
            r["n_qc_sessions"] = int(df.get("is_bad", pd.Series([False]*len(df))).sum())
        except Exception:
            r["n_qc_sessions"] = safe_read_csv_rows(sess_qc_csv)

    obs_qc_csv = run_dir / "observer_qc_summary.csv"
    if obs_qc_csv.exists():
        try:
            df = pd.read_csv(obs_qc_csv)
            if "exclude_observer" in df.columns:
                r["n_qc_observers"] = int(df["exclude_observer"].fillna(False).astype(bool).sum())
            else:
                r["n_qc_observers"] = len(df)
        except Exception:
            r["n_qc_observers"] = safe_read_csv_rows(obs_qc_csv)

    drop_csv = run_dir / "session_drop_ranges.csv"
    if drop_csv.exists():
        r["n_drop_ranges"] = safe_read_csv_rows(drop_csv)

    # cleaning breakdowns
    r["cleaning"] = collect_cleaning_breakdowns(run_dir)
    return r

# ----------------- figure copying -----------------
# Prefer PDF now; fall back to PNG.
FIG_BASES = {
    "06_phase_P1":         "Phase–folded at P1",
    "05_residuals":        "Residuals vs. time",
    "04_model_timeseries": "Model vs. time",
    # NEW: only present if the run was made with --mcmc; _find_best_fig()/copy_and_rename_figs()
    # already print "NOT FOUND" and continue gracefully for older runs, so this is fully backward-compatible.
    "09_mcmc_corner_twin_free": "MCMC corner (twin-sine, free period)",
    "10_mcmc_trace_twin_free":  "MCMC trace (twin-sine, free period)",
}
PREF_EXT = [".pdf", ".PDF", ".png", ".PNG"]  # preference order

def _find_best_fig(run_dir: Path, base: str):
    # try exact at root first (fast path)
    for ext in PREF_EXT:
        p = run_dir / f"{base}{ext}"
        if p.exists():
            return p
    # recursive search, case-insensitive
    candidates = []
    for p in run_dir.rglob("*"):
        if not p.is_file(): continue
        name_lower = p.name.lower()
        if name_lower.startswith(base.lower()) and any(name_lower.endswith(e.lower()) for e in PREF_EXT):
            candidates.append(p)
    if not candidates:
        return None
    # prefer preferred extension then shortest path
    def score(p: Path):
        ext_rank = next((i for i,e in enumerate(PREF_EXT) if p.suffix.lower()==e.lower()), 999)
        return (ext_rank, len(str(p)))
    candidates.sort(key=score)
    return candidates[0]

def copy_and_rename_figs(outdir: Path, run_dir: Path, label: str) -> dict:
    figs_dir = outdir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    prefix = sanitize_for_filename(label)
    out = {}
    print(f"\n[{label}] searching figures in {run_dir} ...")
    for base in FIG_BASES.keys():
        src = _find_best_fig(run_dir, base)
        if src is None:
            print(f"  - {base}: NOT FOUND")
            continue
        dest_name = f"{prefix}__{src.name}"  # keep original extension
        dest = figs_dir / dest_name
        shutil.copy2(src, dest)
        out[f"{base}{src.suffix.lower()}"] = dest_name  # remember with extension
        print(f"  - {base}: {src.name}  ->  figs/{dest_name}")
    return out

# ----------------- LaTeX builder -----------------
def build_tex(outdir: Path, runs: list[dict]):
    tex = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{graphicx,booktabs,geometry,subcaption,array,tabularx,longtable}",
        r"\geometry{margin=1in}",
        r"\title{Multi-run Comparison}",
        rf"\date{{Generated {datetime.now():%Y-%m-%d %H:%M}}}",
        r"\begin{document}\maketitle",
    ]

    # -------- Summary table (no metrics here)
    tex += [
        r"\section*{Run summary}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{4.8cm}rrrrrr@{}}",
        r"\toprule",
        r"Run & $N_{\rm in}$ & $N_{\rm clean}$ & Keep[\%] & $P_1$[d] & $P_2$[d] & Span[d] \\",
        r"\midrule",
    ]
    for r in runs:
        keep = ""
        if r.get("n_input") and r.get("n_clean"):
            try:
                keep = f"{100.0*float(r['n_clean'])/float(r['n_input']):.1f}"
            except Exception:
                keep = ""
        p1s = fmt_num(r.get("P1"), ".3f"); p1e = fmt_num(r.get("P1_err"), ".3f")
        p1  = f"{p1s}±{p1e}" if (p1s and p1e) else (p1s or "—")
        p2s = fmt_num(r.get("P2"), ".3f"); p2e = fmt_num(r.get("P2_err"), ".3f")
        p2  = f"{p2s}±{p2e}" if (p2s and p2e) else (p2s or "—")
        tspan  = fmt_num(r.get("tspan"), ".2f")
        tex.append(
            f"{wrap_label(r['label'])} & {r.get('n_input','')} & {r.get('n_clean','')} & {keep} & "
            f"{p1} & {p2} & {tspan} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabularx}"]

    # -------- NEW: metrics table (single vs twin)
    tex += [
        r"\section*{Model metrics (single vs twin)}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{4.6cm}rr|rrrr|rrrr@{}}",
        r"\toprule",
        r" & & & \multicolumn{4}{c|}{Single @ $P_1$} & \multicolumn{4}{c}{Twin ($P_1{+}P_2$)} \\",
        r"Run & $P_1$[d] & $P_2$[d] & $\chi^2$ & $\chi_r^2$ & AIC & BIC & $\chi^2$ & $\chi_r^2$ & AIC & BIC \\",
        r"\midrule",
    ]
    for r in runs:
        p1s = fmt_num(r.get("P1"), ".3f"); p2s = fmt_num(r.get("P2"), ".3f")
        ms = r.get("met_single", {}) or {}
        mt = r.get("met_twin",   {}) or {}
        row = (
            f"{wrap_label(r['label'])} & {p1s} & {p2s} & "
            f"{fmt_num(ms.get('chi2'),'.1f')} & {fmt_num(ms.get('rchi2'),'.3f')} & {fmt_num(ms.get('AIC'),'.1f')} & {fmt_num(ms.get('BIC'),'.1f')} & "
            f"{fmt_num(mt.get('chi2'),'.1f')} & {fmt_num(mt.get('rchi2'),'.3f')} & {fmt_num(mt.get('AIC'),'.1f')} & {fmt_num(mt.get('BIC'),'.1f')} \\\\"
        )
        tex.append(row)
    tex += [r"\bottomrule", r"\end{tabularx}", r"\normalsize"]

    # -------- NEW: MCMC posterior summary (twin-sine model, fixed & free period modes)
    # Fully additive: if no run used --mcmc, this whole section is skipped (any_mcmc is False).
    # For older runs that lack MCMC data alongside newer ones that have it, individual rows
    # just show "(no MCMC data)" instead of breaking the table.
    any_mcmc = any(r.get("mcmc") for r in runs)
    if any_mcmc:
        tex += [
            r"\section*{MCMC posterior summary (twin-sine model)}",
            r"\scriptsize",
        ]
        for period_mode, mode_label in (("fixed", "periods fixed at periodogram values"),
                                         ("free", "periods free, Gaussian-prior on periodogram values")):
            tex += [
                rf"\subsection*{{Period mode: {mode_label}}}",
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{4.0cm}rrrrrr@{}}",
                r"\toprule",
                r"Run & $P_1$[d] & $P_2$[d] & $A_1$ & $A_2$ & accept.\ frac & converged \\",
                r"\midrule",
            ]
            for r in runs:
                m = (r.get("mcmc") or {}).get(f"twin_{period_mode}")
                if not m:
                    tex.append(f"{wrap_label(r['label'])} & \\multicolumn{{6}}{{c}}{{(no MCMC data)}} \\\\")
                    continue
                s = m.get("summary", {})
                def _pm(name):
                    d = s.get(name)
                    if not d:
                        return "—"
                    return f"${d['median']:.4g}^{{+{d['err_hi']:.2g}}}_{{-{d['err_lo']:.2g}}}$"
                p1cell = _pm("P1") if period_mode == "free" else (fmt_num(r.get("P1"), ".4f") or "—")
                p2cell = _pm("P2") if period_mode == "free" else (fmt_num(r.get("P2"), ".4f") or "—")
                tex.append(
                    f"{wrap_label(r['label'])} & {p1cell} & {p2cell} & {_pm('A1')} & {_pm('A2')} & "
                    f"{fmt_num(m.get('acceptance_fraction'),'.2f')} & {'yes' if m.get('converged') else 'no'} \\\\"
                )
            tex += [r"\bottomrule", r"\end{tabularx}"]
        tex.append(r"\normalsize")
    tex += [
        r"\section*{Cleaning / QC overview}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{4.8cm}rrrrrr@{}}",
        r"\toprule",
        r"Run & Removed & QC sessions & QC observers & Drop ranges & Span[d] & Cycles($P_1$) \\",
        r"\midrule",
    ]
    for r in runs:
        tspan  = fmt_num(r.get("tspan"), ".2f")
        cycles = fmt_num(r.get("cycles"), ".2f")
        removed = "" if r.get("n_removed") is None else str(r["n_removed"])
        tex.append(
            f"{wrap_label(r['label'])} & {removed} & {r.get('n_qc_sessions',0)} & "
            f"{r.get('n_qc_observers',0)} & {r.get('n_drop_ranges',0)} & "
            f"{tspan} & {cycles} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabularx}"]

    # -------- Settings (period search + clipping)  [trimmed to avoid overflow]
    tex += [
        r"\section*{Key settings (period search and legacy clipping)}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{4.8cm}rrrrrrrr@{}}",
        r"\toprule",
        r"Run & Band & $P_{\min}$ & $P_{\max}$ & $N_f$ & $P_1$ prior & prior frac & per-obs $\sigma$ & clip $\sigma$/iters \\",
        r"\midrule",
    ]
    for r in runs:
        # period-search
        minP  = fmt_num(r.get("minP"), ".2f")
        maxP  = fmt_num(r.get("maxP"), ".2f")
        nfreq = str(r.get("nfreq","")) if r.get("nfreq") is not None else ""
        P1pr  = fmt_num(r.get("P1_prior"), ".3f")
        P1pf  = fmt_num(r.get("P1_prior_frac"), ".2f")
        # per-obs sigma: prefer legacy, else per-night sigma
        pos_legacy = fmt_num(r.get("per_observer_sigma"), ".2f")
        pos_v4     = fmt_num(r.get("per_night_sigma"), ".1f")
        pos        = pos_legacy if pos_legacy else pos_v4
        # clip sigma/iters: prefer legacy, else global
        clip_legacy = fmt_num(r.get("sigma_clip"), ".1f")
        it_legacy   = fmt_int(r.get("iters"))
        clip_v4     = fmt_num(r.get("global_sigma"), ".1f")
        it_v4       = fmt_int(r.get("global_iters"))
        clip        = clip_legacy if clip_legacy else clip_v4
        iters       = it_legacy   if it_legacy   else it_v4

        tex.append(
            f"{wrap_label(r['label'])} & {tex_escape(str(r.get('band','')))} & "
            f"{minP} & {maxP} & {nfreq} & {P1pr} & {P1pf} & {pos} & {clip}/{iters} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabularx}"]

    # -------- v4 QC knobs (reduced columns to avoid overflow; err_max removed)
    tex += [
        r"\section*{v4 QC knobs}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{4.8cm}rrrrr@{}}",
        r"\toprule",
        r"Run & gap[h] & night $N_{\min}$ & night $\sigma$ & global $\sigma$/iters & roll [d]/$\sigma$ \\",
        r"\midrule",
    ]
    for r in runs:
        gap = fmt_num(r.get("gap_hours"), ".1f")
        pn_minN = fmt_int(r.get("per_night_minN"))
        pn_sig  = fmt_num(r.get("per_night_sigma"), ".1f")
        gl_sig  = fmt_num(r.get("global_sigma"), ".1f")
        gl_it   = fmt_int(r.get("global_iters"))
        rw      = fmt_num(r.get("roll_window"), ".2f")
        rs      = fmt_num(r.get("roll_sigma"), ".1f")

        tex.append(
            f"{wrap_label(r['label'])} & {gap} & {pn_minN} & {pn_sig} & "
            f"{gl_sig}/{gl_it} & {rw}/{rs} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabularx}"]

    # (REMOVED) Cleaning aggregates section

    # (REMOVED) Top observers mini-tables to avoid duplication

    # -------- All observers longtable by run
    tex += [r"\section*{All observers by run (sorted by removed\%)}"]
    for r in runs:
        obs_full = r.get("cleaning", {}).get("observers_full")
        if not obs_full:
            continue
        tex += [
            rf"\subsection*{{{tex_escape(r['label'])}}}",
            r"\scriptsize",
            r"\begin{longtable}{@{}lrrr@{}}",
            r"\toprule",
            r"Observer & kept & removed & rem.\% \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule Observer & kept & removed & rem.\% \\ \midrule",
            r"\endhead",
        ]
        for row in obs_full:
            name = tex_escape(str(row.get("observer","")))
            kept = fmt_int(row.get("kept"))
            rem  = fmt_int(row.get("removed"))
            frac = row.get("rem_frac")
            pct = ""
            try:
                if frac is not None and float(frac)==float(frac):
                    pct = f"{100*float(frac):.1f}"
            except Exception:
                pct = ""
            tex.append(f"{name} & {kept} & {rem} & {pct} \\\\")
        tex += [r"\bottomrule", r"\end{longtable}", r"\normalsize"]

    # -------- Figure sections (PDF preferred)
    def fig_block(title: str, base: str):
        tex.append(rf"\section*{{{title}}}")
        tex.append(r"\begin{figure}[htbp]\centering")
        col_local = 0
        for r in runs:
            # prefer PDF, fallback PNG
            chosen = None
            for ext in (".pdf", ".png"):
                key = f"{base}{ext}"
                if key in r["images"]:
                    chosen = r["images"][key]
                    break
            if not chosen:
                continue
            tex.append(r"\begin{subfigure}[t]{0.32\linewidth}\centering")
            tex.append(rf"\includegraphics[width=\linewidth]{{figs/{tex_escape(chosen)}}}")
            cap = tex_escape(r["label"])
            if base == "06_phase_P1" and r.get("P1") is not None:
                cap += rf" ($P_1={fmt_num(r['P1'],'.3f')}$ d)"
            tex.append(rf"\caption*{{\small {cap}}}")
            tex.append(r"\end{subfigure}")
            col_local += 1
            if col_local % 3 == 0:
                tex.append(r"\par\medskip")
        tex.append(rf"\caption{{{tex_escape(title)} for each run.}}")
        tex.append(r"\end{figure}")

    fig_block("Phase–folded at $P_1$", "06_phase_P1")
    fig_block("Residuals vs.~time",    "05_residuals")
    fig_block("Model fit vs.~time",    "04_model_timeseries")
    # NEW: only produces figures for runs made with --mcmc; fig_block() already skips
    # any run missing the file (via the "if not chosen: continue" check), so this is
    # safe to call unconditionally even when no run has MCMC data (it just emits an
    # empty figure environment, matching the existing behavior of the other fig_blocks
    # when a figure is missing for some runs).
    if any_mcmc:
        fig_block("MCMC posterior corner plot (twin-sine, free period)", "09_mcmc_corner_twin_free")
        fig_block("MCMC walker trace (twin-sine, free period)",          "10_mcmc_trace_twin_free")

    tex.append(r"\end{document}")
    (outdir / "comparison_report.tex").write_text("\n".join(tex), encoding="utf-8")
    return outdir / "comparison_report.tex"

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Build an Overleaf-ready comparison bundle (PDF preferred, PNG fallback).")
    ap.add_argument("runs", nargs="+", help="Run directories to include (order matters).")
    ap.add_argument("--labels", default=None, help="Comma-separated labels (same count as runs).")
    ap.add_argument("--outdir", default="compare_runs_output", help="Where to write comparison_report.tex and figs/")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figs").mkdir(exist_ok=True)

    run_paths = [Path(r) for r in args.runs]
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(run_paths):
            sys.exit("ERROR: --labels count must match number of runs.")
    else:
        labels = [p.name for p in run_paths]

    collected = []
    for run_dir, lab in zip(run_paths, labels):
        info = collect_run(run_dir, lab)
        info["images"] = copy_and_rename_figs(outdir, run_dir, lab)
        collected.append(info)

    tex_path = build_tex(outdir, collected)
    print("\nComparison bundle created:")
    print("  • LaTeX:", tex_path)
    print("  • Figures copied to:", outdir / "figs")
    print(f"\nUpload the whole folder to Overleaf: {outdir.resolve()}")

if __name__ == "__main__":
    main()
