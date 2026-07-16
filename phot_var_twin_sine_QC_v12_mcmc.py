#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twin-sine (rotation + harmonic) photometric variability model with built-in QC.
- Night-aware clipping (per observer/night)
- Optional rolling-window MAD clipping
- Optional thinning for runtime
- QC tables & plots (per-session, per-observer); suggested night drops
- Phase plot is zero-centered [-0.5, 0.5] and annotates P1

Author:Federico Noguer, Arizona State University, School of earth and Space Exploration 2025
"""

from __future__ import annotations
import argparse, json, os, textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from astropy.timeseries import LombScargle
    ASTROPY_OK = True
except Exception:
    from scipy.signal import lombscargle
    ASTROPY_OK = False

from scipy.optimize import curve_fit

# --- NEW (MCMC): optional dependencies, imported defensively like ASTROPY_OK above.
# Nothing below this block is required unless --mcmc is passed on the command line;
# every existing code path is completely unaffected if these packages are absent. ---
try:
    import emcee
    EMCEE_OK = True
except Exception:
    EMCEE_OK = False

try:
    import corner as corner_pkg
    CORNER_OK = True
except Exception:
    CORNER_OK = False

# ----------------------------- Utilities ----------------------------------
# --- Figure saving helpers: write PNG + PDF with the same basename ----------
def fig_save_all_ext(fig, path_with_ext: str, *, dpi_png: int = 200) -> None:
    """Save a Matplotlib Figure to both .png and .pdf using the same basename."""
    base, _ = os.path.splitext(path_with_ext)
    fig.savefig(base + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(base + ".pdf", dpi=dpi_png, bbox_inches="tight")

def plt_save_all_ext(path_with_ext: str, *, dpi_png: int = 200) -> None:
    """Like fig_save_all_ext but uses the current pyplot figure."""
    base, _ = os.path.splitext(path_with_ext)
    plt.savefig(base + ".png", dpi=dpi_png, bbox_inches="tight")
    plt.savefig(base + ".pdf", dpi=dpi_png, bbox_inches="tight")

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

SCALE_MAD = 1.4826

def mad(x, c=SCALE_MAD):
    x = np.asarray(x, float)
    return c * np.nanmedian(np.abs(x - np.nanmedian(x)))

def weighted_stats(y, w):
    wsum = np.sum(w)
    if wsum == 0:
        return np.nan, np.nan
    mu = np.sum(w * y) / wsum
    var = np.sum(w * (y - mu)**2) / wsum
    return mu, np.sqrt(var)

# ----------------------------- I/O Layer ----------------------------------
DEFAULT_COLS = dict(
    time=["time","jd","bjd","hjd","t","Time","BJD_TDB","BJD","JD","HJD"],
    flux=["flux","f","relative_flux","norm_flux","Flux"],
    flux_err=["flux_err","ferr","sigma","FluxErr","flux_error"],
    mag=["magnitude","mag","Mag","Magnitude"],
    mag_err=["mag_err","mag_sigma","dm","MagErr","Uncertainty","HQuncertainty"],
    observer=["observer","observer_code","obs","Observer","Observer Code","AAVSOID"],
    filt=["filter","band","filt","Filter","Band"],
    flag=["flag","quality_flag","mask","Flag","Validation Flag"],
)

def read_table(path, delimiter=None, comment=None, skiprows=None):
    kw = {}
    if comment not in (None, ""):
        kw["comment"] = comment
    if isinstance(skiprows, int):
        kw["skiprows"] = skiprows
    try_delims = [delimiter] if delimiter else [",","\t",";"]
    for delim in try_delims:
        try:
            df = pd.read_csv(path, delimiter=delim, engine="python", **kw)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    df = pd.read_csv(path, delim_whitespace=True, engine="python", **kw)
    return df

def auto_map_columns(df, user_map=None):
    colmap = {}
    cols_lower = {c.lower(): c for c in df.columns}
    if user_map:
        for k,v in user_map.items():
            if v in df.columns:
                colmap[k] = v
    for key, cands in DEFAULT_COLS.items():
        if key in colmap:
            continue
        for cand in cands:
            if cand in df.columns:
                colmap[key] = cand; break
            if cand.lower() in cols_lower:
                colmap[key] = cols_lower[cand.lower()]; break
    if "time" not in colmap or ("flux" not in colmap and "mag" not in colmap):
        raise ValueError(
            "Could not find required columns for time/(flux or magnitude). "
            f"Found: {df.columns.tolist()} Map: {colmap}"
        )
    return colmap

def prepare_data(df, colmap, normalize=True):
    out = pd.DataFrame()
    out["time"] = df[colmap["time"]].astype(float).values

    if "flux" in colmap:
        out["flux"] = df[colmap["flux"]].astype(float).values
        if "flux_err" in colmap:
            out["flux_err"] = df[colmap["flux_err"]].astype(float).values
        else:
            sig = mad(out["flux"].values); out["flux_err"] = np.full(len(out), sig if np.isfinite(sig) else 1.0)
        out.attrs["flux_mode"] = "flux"

    elif "mag" in colmap:
        mvals = df[colmap["mag"]].astype(float).values
        dm = df[colmap["mag_err"]].astype(float).values if "mag_err" in colmap else np.full(len(mvals), np.nan)
        m_med = np.nanmedian(mvals)
        rel_flux = 10.0**(-0.4 * (mvals - m_med))
        k = 0.4*np.log(10.0)
        rel_flux_err = np.where(np.isfinite(dm), np.abs(k)*rel_flux*dm, np.nan)
        out["flux"] = rel_flux
        if np.all(~np.isfinite(rel_flux_err)):
            sig = mad(rel_flux); rel_flux_err = np.full(len(rel_flux), sig if np.isfinite(sig) else 1.0)
        out["flux_err"] = rel_flux_err
        out.attrs["flux_mode"] = "mag->flux"
    else:
        raise ValueError("Neither flux nor mag columns were found after mapping.")

    out["observer"] = df[colmap.get("observer","observer")].astype(str).values if colmap.get("observer") else "unknown"
    out["filter"] = df[colmap.get("filt","filter")].astype(str).values if colmap.get("filt") else "unknown"
    out["flag"] = df[colmap.get("flag","flag")].values if colmap.get("flag") else 0

    m = np.isfinite(out["time"]) & np.isfinite(out["flux"]) & np.isfinite(out["flux_err"])
    out = out[m].copy()

    if normalize:
        med = np.nanmedian(out["flux"])
        if np.isfinite(med) and med != 0:
            out["flux"] /= med; out["flux_err"] /= med
    return out

# ----------------------------- Sessioning (QC) -----------------------------
def tag_sessions(df: pd.DataFrame, group: str = "gap", gap_hours: float = 6.0) -> pd.DataFrame:
    """Add 'session_id' (e.g., OBS#S3) and 'night_id' columns."""
    d = df.sort_values(["observer", "time"]).copy()
    if group not in {"gap", "calendar"}:
        raise ValueError("--group must be 'gap' or 'calendar'")

    sess_ids = []
    night_ids = []
    for obs, g in d.groupby("observer", sort=False):
        tt = g["time"].to_numpy()
        if group == "gap":
            gap_days = gap_hours / 24.0
            edges = np.r_[True, np.diff(tt) > gap_days]  # new session if big gap
            sess_num = np.cumsum(edges)  # 1..K per observer
            night_id = sess_num
        else:
            night_id = np.floor(tt + 0.5).astype(int)  # astronomical night
            _, sess_num = np.unique(night_id, return_inverse=True)
            sess_num += 1

        sess_ids.extend([f"{obs}#S{n}" for n in sess_num])
        night_ids.extend(night_id)

    d["session_id"] = sess_ids
    d["night_id"] = night_ids
    return d

def per_group_metrics(g: pd.DataFrame) -> dict:
    N = len(g)
    med_err = float(np.nanmedian(g["flux_err"]))
    rmad = float(mad(g["resid"].values))
    mean = float(np.nanmean(g["resid"].values))
    std = float(np.nanstd(g["resid"].values))

    # χ²/dof proxy with robust fallback
    if np.isfinite(g["flux_err"]).any():
        denom = g["flux_err"].to_numpy(copy=True)
        pos = denom[np.isfinite(denom) & (denom > 0)]
        mpos = np.nanmedian(pos) if pos.size else np.nan
        if np.isfinite(mpos):
            denom[~np.isfinite(denom) | (denom <= 0)] = mpos
        chi = float(np.nanmean((g["resid"].to_numpy() / denom) ** 2))
    else:
        chi = np.nan

    medr = float(np.nanmedian(g["resid"]))
    thresh = 4.0 * (rmad if (np.isfinite(rmad) and rmad > 0) else (std if std > 0 else 1.0))
    ofrac = float(np.mean(np.abs(g["resid"].to_numpy() - medr) > thresh))

    return dict(
        N=N, med_err=med_err,
        resid_mean=mean, resid_std=std, resid_MAD=rmad,
        chi2_over_dof=chi, outlier_frac=ofrac,
        t_min=float(np.nanmin(g["time"])), t_max=float(np.nanmax(g["time"]))
    )

def summarize_sessions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (obs, sess), g in df.groupby(["observer", "session_id"], sort=False):
        m = per_group_metrics(g)
        m.update(
            observer=str(obs),
            session_id=str(sess),
            night_id=str(int(np.nanmedian(g["night_id"])) if np.isfinite(np.nanmedian(g["night_id"])) else -1),
        )
        rows.append(m)
    s = pd.DataFrame(rows)
    sort_cols = ["chi2_over_dof", "resid_MAD", "outlier_frac", "N"]
    return s.sort_values(sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)

def rollup_observers(session_df: pd.DataFrame) -> pd.DataFrame:
    g = session_df.groupby("observer", as_index=False)
    obs = g.agg(
        N_total=("N", "sum"),
        n_sess=("session_id", "count"),
        med_MAD=("resid_MAD", "median"),
        med_chi=("chi2_over_dof", "median"),
        med_ofrac=("outlier_frac", "median"),
    )
    return obs

def flag_bad_sessions(sessions: pd.DataFrame, chi_thresh: float, ofrac_thresh: float, minN: int, mad_factor: float) -> pd.Series:
    """ Mark a session as bad if it has enough points AND (χ²/dof high OR outlier_frac high).
        If χ²/dof is unavailable (all NaN), fall back to a robust MAD-based criterion. """
    enough_points = sessions["N"] >= minN
    use_mad = sessions["chi2_over_dof"].isna().all()
    if use_mad:
        ref = np.nanmedian(sessions["resid_MAD"])
        chi_or_mad_bad = sessions["resid_MAD"] > (mad_factor * ref)
    else:
        chi_or_mad_bad = sessions["chi2_over_dof"] >= chi_thresh
    ofrac_bad = sessions["outlier_frac"] >= ofrac_thresh
    return enough_points & (chi_or_mad_bad | ofrac_bad)

def _barh(figfile, labels, values, title, xlabel, height_per=0.35, min_h=6, fontsize=8, xlim=None):
    fig_h = max(min_h, height_per * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, alpha=0.85)
    ax.set_yticks(y, labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=fontsize)
    if xlim is not None:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fig_save_all_ext(fig, figfile, dpi_png=160)
    plt.close(fig)

# ----------------------------- Cleaning (better way) -----------------------
def smart_clean(df: pd.DataFrame, *,
                flag_reject_values=None,
                err_max=None,
                err_median_factor=None,
                group="gap", gap_hours=6.0,
                per_night_minN=20, per_night_sigma=3.0,
                global_sigma=3.0, global_iters=2,
                roll_window=0.5,  # full width [days]; 0 disables
                roll_sigma=4.0,
                thin_maxN=0,
                outdir=None):
    """ Returns: cleaned DataFrame, removed DataFrame (+ reason).

    Steps:
    0. optional: reject flags, gate by error (absolute or relative to median)
    1. session tagging (observer, astronomical night or gap-based)
    2. per-session normalization (divide by session median) + within-session MAD clip
    3. global robust sigma clip (iterations)
    4. optional rolling-window MAD clip
    5. optional thinning (logged as 'thinned_for_runtime')
    """
    work = df.copy()
    removed_records = []

    # 0) flags & error gates
    if flag_reject_values is not None and "flag" in work.columns:
        bad = work["flag"].isin(flag_reject_values)
        if np.any(bad):
            r = work.loc[bad].copy(); r["reason"] = "flag_reject"
            removed_records.append(r); work = work.loc[~bad]
    if err_median_factor is not None:
        cap = err_median_factor * np.nanmedian(work["flux_err"].values)
        err_max = cap if err_max is None else min(err_max, cap)
    if err_max is not None:
        bad = work["flux_err"] > err_max
        if np.any(bad):
            r = work.loc[bad].copy(); r["reason"] = f"flux_err>{err_max}"
            removed_records.append(r); work = work.loc[~bad]

    # 1) sessions
    work = tag_sessions(work, group=group, gap_hours=gap_hours)

    # 2) per-session normalization + within-session clip
    blocks = []
    for (obs, sess), g in work.groupby(["observer", "session_id"], sort=False):
        gg = g.copy()
        if len(gg) >= per_night_minN:
            med = np.nanmedian(gg["flux"])
            if np.isfinite(med) and med > 0:
                gg["flux"] = gg["flux"] / med
                gg["flux_err"] = gg["flux_err"] / med
            m0 = np.nanmedian(gg["flux"])
            s0 = mad(gg["flux"].values)
            if np.isfinite(s0) and s0 > 0:
                bad = np.abs(gg["flux"] - m0) > per_night_sigma * s0
                if np.any(bad):
                    r = gg.loc[bad].copy(); r["reason"] = f"per_night_sigma({obs},{sess})"
                    removed_records.append(r); gg = gg.loc[~bad]
        blocks.append(gg)
    work = pd.concat(blocks, ignore_index=True)

    # 3) global robust sigma-clip (iterated)
    for k in range(int(global_iters)):
        mu = np.nanmedian(work["flux"])
        s = mad(work["flux"].values)
        if not np.isfinite(s) or s == 0:
            break
        bad = np.abs(work["flux"] - mu) > global_sigma * s
        if np.any(bad):
            r = work.loc[bad].copy(); r["reason"] = f"global_sigma_iter{k+1}"
            removed_records.append(r); work = work.loc[~bad]

    # 4) rolling-window MAD clip
    if roll_window and len(work) > 0:
        w = work.sort_values("time").copy()
        half = 0.5 * roll_window
        keeper = np.ones(len(w), bool)
        t = w["time"].to_numpy()
        f = w["flux"].to_numpy()
        for i, ti in enumerate(t):
            m = (t >= ti - half) & (t <= ti + half)
            mu = np.nanmedian(f[m])
            sg = mad(f[m])
            if np.isfinite(sg) and sg > 0 and np.abs(f[i]-mu) > roll_sigma * sg:
                keeper[i] = False
        if not np.all(keeper):
            r = w.loc[~keeper].copy(); r["reason"] = f"rolling_{roll_sigma}sigma"
            removed_records.append(r)
        work = w.loc[keeper].copy()

    # 5) thinning
    if thin_maxN and len(work) > thin_maxN:
        step = max(1, len(work) // thin_maxN)
        thinned_idx = work.index[1::step]  # keep 0, drop every 'step'th
        r = work.loc[thinned_idx].copy(); r["reason"] = "thinned_for_runtime"
        removed_records.append(r)
        work = work.drop(index=thinned_idx).reset_index(drop=True)

    removed = (pd.concat(removed_records, ignore_index=True)
               if removed_records else pd.DataFrame(columns=list(df.columns)+["reason"]))

    # summaries
    if outdir:
        if len(removed) > 0:
            removed.to_csv(os.path.join(outdir, "removed_points.csv"), index=False)
        obs_summary = work.groupby("observer").size().rename("kept").to_frame()
        rem_summary = (removed.groupby("observer").size().rename("removed").to_frame()
                       if len(removed) else pd.DataFrame())
        summary = obs_summary.join(rem_summary, how="outer").fillna(0).astype(int)
        summary.to_csv(os.path.join(outdir, "cleaning_summary_by_observer.csv"))
        # per-session counts (after cleaning)
        sess_summary = work.groupby(["observer","session_id"]).size().rename("kept").reset_index()
        sess_summary.to_csv(os.path.join(outdir, "cleaning_summary_by_session.csv"), index=False)
        with open(os.path.join(outdir,"cleaning_report.txt"),"w") as f:
            f.write("CLEANING REPORT\n")
            f.write(f"Total input points: {len(df)}\n")
            f.write(f"Total kept: {len(work)}\n")
            f.write(f"Total removed: {len(removed)}\n")

    return work.reset_index(drop=True), removed

# ------------------------ Periodogram & Modeling --------------------------
def gls_periodogram(time, flux, flux_err, minP, maxP, nfreq=100000):
    if not (np.isfinite(minP) and np.isfinite(maxP)) or (minP>=maxP):
        return np.array([]), np.array([]), None
    t = np.asarray(time,float); y = np.asarray(flux,float); e = np.asarray(flux_err,float)
    if ASTROPY_OK:
        ls = LombScargle(t,y,e); fmin = 1.0/maxP; fmax = 1.0/minP
        f = np.linspace(fmin, fmax, nfreq); power = ls.power(f); P = 1.0/f
        return P, power, ls
    else:
        y = y - np.average(y, weights=1/np.maximum(e,1e-12)**2)
        f = np.linspace(1.0/maxP, 1.0/minP, nfreq); ang = 2*np.pi*f
        power = lombscargle(t, y, ang); P = 1.0/f
        return P, power, None

def _safe_gls_window(time,y,yerr,lo,hi,nfreq):
    if (lo is None) or (hi is None) or (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo>=hi):
        return np.array([]), np.array([]), None
    return gls_periodogram(time,y,yerr,lo,hi,nfreq)

def apply_exclude_notches(Pgrid, power, exclude_ranges):
    if not exclude_ranges:
        return power
    pow2 = power.copy()
    for (lo,hi) in exclude_ranges:
        mask = (Pgrid>=lo) & (Pgrid<=hi)
        pow2[mask] = -np.inf
    return pow2

def sine_model(t,A,phi,P):
    return A*np.sin(2*np.pi*t/P + phi)

def twin_sine_model(t,A1,phi1,P1,A2,phi2,P2,C):
    return C + sine_model(t,A1,phi1,P1) + sine_model(t,A2,phi2,P2)

def fit_weighted_sine(t,y,yerr,P,y0=None):
    if y0 is None:
        y0=np.median(y)
    w = 1.0/np.maximum(yerr,1e-12)**2
    mu,sig = weighted_stats(y-y0,w); A0 = sig if (np.isfinite(sig) and sig>0) else 0.01
    def f(tt,A,phi,C):
        return C + sine_model(tt,A,phi,P)
    popt,pcov = curve_fit(f,t,y,sigma=yerr,p0=[A0,0.0,y0],absolute_sigma=True,maxfev=20000)
    return popt,pcov

def fit_twin_sine(t,y,yerr,P1,P2,y0=None):
    if y0 is None:
        y0=np.median(y)
    w = 1.0/np.maximum(yerr,1e-12)**2
    mu,sig = weighted_stats(y-y0,w); A0 = sig if (np.isfinite(sig) and sig>0) else 0.01
    def f(tt,A1,phi1,A2,phi2,C):
        return twin_sine_model(tt,A1,phi1,P1,A2,phi2,P2,C)
    popt,pcov = curve_fit(f,t,y,sigma=yerr,p0=[A0,0.0,0.5*A0,0.0,y0],absolute_sigma=True,maxfev=40000)
    return popt,pcov

def gaussian_nll(y, yerr, ymodel):
    """
    Gaussian negative log-likelihood (NLL) for independent points with known errors.
    NLL = 0.5 * sum( (r^2 / var) + ln(2π var) ), where var = yerr^2 (floored).
    """
    resid = np.asarray(y, float) - np.asarray(ymodel, float)
    var = np.maximum(np.asarray(yerr, float), 1e-12)**2
    return 0.5 * np.sum(resid**2 / var + np.log(2.0 * np.pi * var))

def chi2_metrics(y, yerr, ymodel, k_params, mode="chi2"):
    """
    Compute classic χ²-based criteria and exact Gaussian-likelihood criteria.
    - 'mode' controls which values appear under the main keys 'AIC' and 'BIC'.
      'chi2'  -> AIC=χ²+2k,                 BIC=χ² + k ln n  (default; matches your paper)
      'gauss' -> AIC=2k + 2*NLL_gauss,      BIC=k ln n + 2*NLL_gauss
    In all cases we also store:
      - nll:     exact Gaussian negative log-likelihood
      - AIC_ll:  2k + 2*nll
      - BIC_ll:  k ln n + 2*nll
      - AIC_chi: χ² + 2k
      - BIC_chi: χ² + k ln n
    """
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    ymodel = np.asarray(ymodel, float)

    resid = y - ymodel
    var = np.maximum(yerr, 1e-12)**2

    chi2 = np.sum(resid**2 / var)
    n = len(y)
    dof = n - k_params
    rchi2 = chi2 / max(dof, 1)

    # χ²-based ICs (what you had before)
    aic_chi = chi2 + 2 * k_params
    bic_chi = chi2 + k_params * np.log(n)

    # Exact Gaussian log-likelihood ICs
    nll = 0.5 * np.sum(resid**2 / var + np.log(2.0 * np.pi * var))
    aic_ll = 2 * k_params + 2 * nll
    bic_ll = k_params * np.log(n) + 2 * nll

    out = dict(
        chi2=chi2,
        dof=dof,
        rchi2=rchi2,
        # keep your original fields for backward compatibility
        AIC=aic_chi,
        BIC=bic_chi,
        # always include the likelihood-based values explicitly
        nll=nll,
        AIC_ll=aic_ll,
        BIC_ll=bic_ll,
        # and the χ² ones explicitly, too
        AIC_chi=aic_chi,
        BIC_chi=bic_chi,
    )

    # If requested, make the *primary* AIC/BIC reflect the Gaussian likelihood versions
    if mode == "gauss":
        out["AIC"] = aic_ll
        out["BIC"] = bic_ll

    return out

# ============================================================================
# NEW (MCMC): posterior-based error estimation via `emcee`.
# ----------------------------------------------------------------------------
# This entire section is ADDITIVE. It does not alter sine_model(), twin_sine_model(),
# fit_weighted_sine(), fit_twin_sine(), gaussian_nll(), or chi2_metrics() above --
# it only *calls* them from a new likelihood function, so the underlying equations
# and existing curve_fit-based results are completely unchanged.
#
# It replaces nothing: the existing perr = sqrt(diag(cov)) errors and the existing
# estimate_period_err() periodogram-width errors keep being computed exactly as
# before. This just adds a second, more rigorous (posterior-sampling-based) set of
# error bars alongside them, run only if --mcmc is passed.
#
# Four combinations are supported (all requested for this pipeline):
#   (model_type, period_mode) in {"single","twin"} x {"fixed","free"}
#   - "fixed": P1 (and P2) held fixed at the periodogram values, exactly like the
#              existing curve_fit calls in fit_weighted_sine()/fit_twin_sine().
#   - "free" : P1 (and P2) become free MCMC parameters with a Gaussian prior
#              centered on the periodogram value (width set from the existing
#              estimate_period_err() output), so the period itself also gets a
#              genuine posterior-based uncertainty.
# ============================================================================

MCMC_PARAM_SPECS = {
    ("single", "fixed"): [("A", "amp"), ("phi", "phase"), ("C", "const")],
    ("single", "free"):  [("A", "amp"), ("phi", "phase"), ("C", "const"), ("P", "period")],
    ("twin", "fixed"):   [("A1", "amp"), ("phi1", "phase"), ("A2", "amp"), ("phi2", "phase"), ("C", "const")],
    ("twin", "free"):    [("A1", "amp"), ("phi1", "phase"), ("A2", "amp"), ("phi2", "phase"),
                          ("C", "const"), ("P1", "period"), ("P2", "period")],
}

def _mcmc_build_model(theta, spec, fixed, t):
    """
    Assemble the model curve for one MCMC parameter vector `theta`.
    Dispatches to the SAME sine_model()/twin_sine_model() functions used by the
    existing curve_fit path -- no new physics/equations, just a different caller.
    """
    vals = dict(zip([s[0] for s in spec], theta))
    vals.update(fixed)  # fixed periods (only present when period_mode == "fixed")
    is_twin = "A2" in vals
    if is_twin:
        return twin_sine_model(t, vals["A1"], vals["phi1"], vals["P1"],
                                vals["A2"], vals["phi2"], vals["P2"], vals["C"])
    else:
        return vals["C"] + sine_model(t, vals["A"], vals["phi"], vals["P"])

def _wrap_phase(x):
    """Wrap phase(s) to [-pi, pi]. Used only for reporting/plotting -- never affects the sampler,
    since sin(2*pi*t/P + phi) is exactly periodic in phi and needs no wrapping to be evaluated."""
    return (np.asarray(x, float) + np.pi) % (2 * np.pi) - np.pi

def _mcmc_log_prior(theta, spec, prior_cfg):
    """
    Priors used for the MCMC posterior:
      amp    : Uniform(0, amp_max)          -- A >= 0 breaks the (A -> -A, phi -> phi+pi) degeneracy
      const  : Uniform(const_lo, const_hi)  -- wide, data-driven bounds around the baseline flux
      phase  : unbounded / flat             -- the model is exactly 2*pi-periodic in phi already
      period : Gaussian(P0, sigma_P), truncated to [minP, maxP] -- weakly informative prior
               centered on the periodogram peak (so the chain explores around the already-
               identified period/harmonic rather than wandering to an unrelated alias)
    """
    lp = 0.0
    for (name, kind), val in zip(spec, theta):
        if kind == "amp":
            lo, hi = prior_cfg["amp_bounds"]
            if not (lo <= val <= hi):
                return -np.inf
        elif kind == "const":
            lo, hi = prior_cfg["const_bounds"]
            if not (lo <= val <= hi):
                return -np.inf
        elif kind == "period":
            P0, sigmaP, lo, hi = prior_cfg["period_priors"][name]
            if not (lo <= val <= hi):
                return -np.inf
            lp += -0.5 * ((val - P0) / sigmaP) ** 2 - np.log(sigmaP * np.sqrt(2 * np.pi))
        # kind == "phase": flat/unbounded, contributes 0
    return lp

def _mcmc_log_likelihood(theta, spec, fixed, t, y, yerr):
    """
    Log-likelihood = -gaussian_nll(...). Reuses gaussian_nll() defined above verbatim --
    this is the SAME Gaussian negative log-likelihood equation already in the script,
    not a new statistical model.
    """
    ymodel = _mcmc_build_model(theta, spec, fixed, t)
    return -gaussian_nll(y, yerr, ymodel)

def _mcmc_log_posterior(theta, spec, fixed, prior_cfg, t, y, yerr):
    lp = _mcmc_log_prior(theta, spec, prior_cfg)
    if not np.isfinite(lp):
        return -np.inf
    ll = _mcmc_log_likelihood(theta, spec, fixed, t, y, yerr)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_mcmc_fit(t, y, yerr, model_type, period_mode, popt_dict, *,
                  P1=None, P2=None, P1_err_est=None, P2_err_est=None,
                  minP=1.0, maxP=200.0, period_prior_k=3.0,
                  nwalkers=64, nsteps=5000, nburn=1000, thin=5,
                  seed=None, progress=False, moves="de"):
    """
    Run one emcee MCMC fit for a given (model_type, period_mode) combination.

    model_type  : 'single' or 'twin'
    period_mode : 'fixed' -> P1 (and P2) held fixed at the periodogram values
                  'free'  -> P1 (and P2) are free params with Gaussian prior
                             N(P_periodogram, period_prior_k * P_err_est)
                             (falls back to period_prior_k * 1%*P if P_err_est
                             is NaN/unavailable, e.g. when astropy is missing)
    popt_dict   : dict of the EXISTING curve_fit best-fit values, used only to
                  center/initialize the walkers (e.g. from fit_weighted_sine()/
                  fit_twin_sine(), already computed elsewhere in the script).
    moves       : 'de' (default) -> emcee.moves mixture of
                  [DEMove(0.8), DESnookerMove(0.2)], emcee's own recommended
                  strategy for correlated/degenerate posteriors (amplitude,
                  phase, and period are often correlated here) -- typically
                  reduces the autocorrelation time by a large factor vs. the
                  plain stretch move, so fewer steps are needed to converge.
                  'stretch' -> emcee's original default StretchMove (slower
                  mixing on this kind of problem, kept as a fallback option).

    Returns a dict with: posterior summaries (median, 16th/84th pct per param),
    the flat post-burn-in samples + full chain (for plotting), and diagnostics
    (acceptance fraction, autocorrelation time, a simple convergence flag).
    """
    if not EMCEE_OK:
        raise RuntimeError("emcee is not installed; pip install emcee")

    spec = MCMC_PARAM_SPECS[(model_type, period_mode)]
    names = [s[0] for s in spec]
    ndim = len(spec)
    rng = np.random.default_rng(seed)

    # fixed values (periods, only when period_mode == "fixed")
    fixed = {}
    if period_mode == "fixed":
        if model_type == "twin":
            fixed["P1"] = P1; fixed["P2"] = P2
        else:
            fixed["P"] = P1

    # data-driven, weakly-informative prior bounds
    amp_scale = float(np.nanmax(np.abs(y - np.nanmedian(y))))
    if not np.isfinite(amp_scale) or amp_scale <= 0:
        amp_scale = 1.0
    amp_bounds = (0.0, 8.0 * amp_scale)
    const_bounds = (float(np.nanmin(y)) - 2 * amp_scale, float(np.nanmax(y)) + 2 * amp_scale)

    period_priors = {}
    if period_mode == "free":
        def _sigma_for(P0, Perr):
            if Perr is not None and np.isfinite(Perr) and Perr > 0:
                return period_prior_k * Perr
            return max(period_prior_k * 0.01 * P0, 1e-6)  # 1%-of-P fallback if no periodogram width
        if model_type == "single":
            period_priors["P"] = (P1, _sigma_for(P1, P1_err_est), minP, maxP)
        else:
            period_priors["P1"] = (P1, _sigma_for(P1, P1_err_est), minP, maxP)
            period_priors["P2"] = (P2, _sigma_for(P2, P2_err_est), minP, maxP)

    prior_cfg = dict(amp_bounds=amp_bounds, const_bounds=const_bounds, period_priors=period_priors)

    # starting point: the EXISTING curve_fit result (popt_dict), not a new estimate
    theta0 = []
    for name, kind in spec:
        if name in popt_dict:
            v = popt_dict[name]
        elif name in ("P", "P1"):
            v = P1
        elif name == "P2":
            v = P2
        else:
            raise KeyError(f"Missing starting value for MCMC parameter '{name}'")
        if kind == "phase":
            v = float(_wrap_phase(np.array([v]))[0])
        if kind == "amp":
            v = abs(v)
        theta0.append(v)
    theta0 = np.array(theta0, float)

    # initialize walkers in a small ball around theta0, resampling any draw with -inf prior
    step_scale = np.maximum(np.abs(theta0) * 1e-3, 1e-4)
    p0 = []
    tries = 0
    while len(p0) < nwalkers and tries < nwalkers * 200:
        cand = theta0 + step_scale * rng.normal(size=ndim)
        if np.isfinite(_mcmc_log_prior(cand, spec, prior_cfg)):
            p0.append(cand)
        tries += 1
    scale_mult = 10.0
    while len(p0) < nwalkers:  # widen if theta0 sits very close to a bound
        cand = theta0 + scale_mult * step_scale * rng.normal(size=ndim)
        if np.isfinite(_mcmc_log_prior(cand, spec, prior_cfg)):
            p0.append(cand)
        scale_mult *= 1.5
    p0 = np.array(p0[:nwalkers])

    if moves == "de":
        # emcee's own recommended mixture for correlated/degenerate posteriors --
        # see https://emcee.readthedocs.io/en/stable/user/moves/ ("DEMove").
        move_strategy = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    else:
        move_strategy = None  # emcee's default StretchMove

    sampler = emcee.EnsembleSampler(nwalkers, ndim, _mcmc_log_posterior,
                                     args=(spec, fixed, prior_cfg, t, y, yerr),
                                     moves=move_strategy)
    sampler.run_mcmc(p0, nsteps, progress=progress)

    acc_frac = float(np.mean(sampler.acceptance_fraction))
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        tau_max = float(np.nanmax(tau))
        converged = bool(nsteps > 50 * tau_max)
    except Exception:
        tau_max = float("nan")
        converged = False

    nburn_eff = min(nburn, max(nsteps - 1, 0))
    flat = sampler.get_chain(discard=nburn_eff, thin=max(1, thin), flat=True)
    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim) -- includes burn-in, for trace plots

    flat_report = flat.copy()
    for i, (name, kind) in enumerate(spec):
        if kind == "phase":
            flat_report[:, i] = _wrap_phase(flat[:, i])

    summary = {}
    for i, (name, kind) in enumerate(spec):
        p16, p50, p84 = np.percentile(flat_report[:, i], [16, 50, 84])
        summary[name] = dict(median=float(p50), err_lo=float(p50 - p16), err_hi=float(p84 - p50))

    return dict(
        model_type=model_type, period_mode=period_mode, param_names=names, spec=spec,
        summary=summary, flat_samples=flat_report, chain=chain,
        acceptance_fraction=acc_frac, autocorr_time_max=tau_max, converged=converged,
        nwalkers=nwalkers, nsteps=nsteps, nburn=nburn_eff, thin=thin,
    )

def make_mcmc_plots(outdir, tag, result):
    """
    Save a corner (posterior) plot and a walker trace plot for one MCMC run.
    New output files only (03_.. through 08_.. plots from make_plots() are untouched):
      09_mcmc_corner_<tag>.png/.pdf
      10_mcmc_trace_<tag>.png/.pdf
    """
    names = result["param_names"]
    flat = result["flat_samples"]
    chain = result["chain"]

    if CORNER_OK:
        fig = corner_pkg.corner(flat, labels=names, quantiles=[0.16, 0.5, 0.84],
                                 show_titles=True, title_fmt=".4g")
        fig_save_all_ext(fig, os.path.join(outdir, f"09_mcmc_corner_{tag}.png"), dpi_png=150)
        plt.close(fig)

    ndim = len(names)
    fig, axes = plt.subplots(ndim, 1, figsize=(9, 1.8 * ndim), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for i in range(ndim):
        axes[i].plot(chain[:, :, i], alpha=0.4, lw=0.5)
        axes[i].set_ylabel(names[i])
        axes[i].axvline(result["nburn"], color="k", ls="--", lw=0.8)
    axes[-1].set_xlabel("Step (dashed line = end of burn-in)")
    fig.suptitle(f"MCMC trace: {tag}")
    fig.tight_layout()
    fig_save_all_ext(fig, os.path.join(outdir, f"10_mcmc_trace_{tag}.png"), dpi_png=150)
    plt.close(fig)

def run_all_mcmc(outdir, t, y, yerr, params, P1, P2, minP, maxP, *,
                  nwalkers=64, nsteps=5000, nburn=1000, thin=5,
                  period_prior_k=3.0, seed=None, progress=False, moves="de"):
    """
    Orchestrator: runs all four requested combinations --
      (single, fixed), (single, free), (twin, fixed), (twin, free)
    initializing each from the EXISTING curve_fit results already computed by
    prewhiten_twin_sine() (passed in via `params`), and writes corner+trace plots
    for each. Returns a JSON-serializable dict (samples/chains are NOT included,
    only scalar summaries + diagnostics, to keep summary.json small).
    """
    # Refit the single-sine model independently here ONLY to get its starting popt
    # for the walkers -- this calls the EXISTING fit_weighted_sine() function, it
    # does not change how P1/the single-sine model were originally determined.
    (A1_single, phi1_single, C_single), _ = fit_weighted_sine(t, y, yerr, P1)

    out = {}
    combos = [("single", "fixed"), ("single", "free"), ("twin", "fixed"), ("twin", "free")]
    for model_type, period_mode in combos:
        tag = f"{model_type}_{period_mode}"
        if model_type == "single":
            popt_dict = dict(A=A1_single, phi=phi1_single, C=C_single)
        else:
            popt_dict = dict(A1=params["A1"], phi1=params["phi1"],
                              A2=params["A2"], phi2=params["phi2"], C=params["C"])
        res = run_mcmc_fit(
            t, y, yerr, model_type, period_mode, popt_dict,
            P1=P1, P2=P2, P1_err_est=params.get("P1_err_est"), P2_err_est=params.get("P2_err_est"),
            minP=minP, maxP=maxP, period_prior_k=period_prior_k,
            nwalkers=nwalkers, nsteps=nsteps, nburn=nburn, thin=thin,
            seed=seed, progress=progress, moves=moves,
        )
        make_mcmc_plots(outdir, tag, res)
        out[tag] = dict(
            model_type=model_type, period_mode=period_mode, summary=res["summary"],
            acceptance_fraction=res["acceptance_fraction"], autocorr_time_max=res["autocorr_time_max"],
            converged=res["converged"], nwalkers=res["nwalkers"], nsteps=res["nsteps"],
            nburn=res["nburn"], thin=res["thin"],
        )
    return out

def prewhiten_twin_sine(time, flux, flux_err, minP, maxP, nfreq=100000,
                        harmonic_window=0.15,
                        P1_fixed=None,P1_prior=None,P1_prior_frac=0.25,
                        exclude_ranges=None, ic_mode="chi2"):
    t = np.asarray(time,float)

    # prior window
    lo_eff,hi_eff = minP,maxP
    if P1_prior is not None:
        lo_eff = max(minP, P1_prior*(1.0-P1_prior_frac))
        hi_eff = min(maxP, P1_prior*(1.0+P1_prior_frac))
        if lo_eff>=hi_eff:
            lo_eff,hi_eff = minP,maxP

    # choose P1
    if P1_fixed is not None:
        P1=float(P1_fixed); Pgrid=np.array([P1]); power=np.array([1.0])
    else:
        Pgrid,power,_ = gls_periodogram(time,flux,flux_err,lo_eff,hi_eff,nfreq)
        if len(Pgrid)==0:
            raise ValueError("Global GLS returned no frequencies. Check minP/maxP/prior.")
        power_use = apply_exclude_notches(Pgrid,power,exclude_ranges)
        i1 = int(np.nanargmax(power_use)); P1=float(Pgrid[i1])

    # fundamental fit
    (A1_tmp,phi1_tmp,C1_tmp),_ = fit_weighted_sine(time,flux,flux_err,P1)
    model_single = C1_tmp + sine_model(time,A1_tmp,phi1_tmp,P1)
    resid1 = flux - model_single
    metrics_single = chi2_metrics(flux, flux_err, model_single, k_params=3, mode=ic_mode)

    # harmonic search
    P_half = 0.5*P1; P_double = 2.0*P1
    def window(Pc):
        lo=max(minP, Pc*(1.0-harmonic_window)); hi=min(maxP, Pc*(1.0+harmonic_window))
        return lo,hi
    lo_h,hi_h = window(P_half); lo_d,hi_d = window(P_double)
    Ph,pow_h,_ = _safe_gls_window(time,resid1,flux_err,lo_h,hi_h,max(nfreq//4,5000))
    Pd,pow_d,_ = _safe_gls_window(time,resid1,flux_err,lo_d,hi_d,max(nfreq//4,5000))

    P2=None; harmonic_chosen=""
    P2_h = Ph[np.argmax(pow_h)] if len(Ph) else None
    P2_d = Pd[np.argmax(pow_d)] if len(Pd) else None
    if len(Ph) and len(Pd):
        if np.max(pow_h) >= np.max(pow_d):
            P2=P2_h; harmonic_chosen="0.5*Prot"
        else:
            P2=P2_d; harmonic_chosen="2*Prot"
    elif len(Ph):
        P2=P2_h; harmonic_chosen="0.5*Prot"
    elif len(Pd):
        P2=P2_d; harmonic_chosen="2*Prot"
    else:
        Pglob,powglob,_ = gls_periodogram(time,flux,flux_err,minP,maxP,nfreq)
        for idx in np.argsort(powglob)[::-1]:
            if np.abs(Pglob[idx]-P1)/P1 > 0.05:
                P2=Pglob[idx]; harmonic_chosen="fallback_global"; break
        if P2 is None:
            candidate=0.5*P1; candidate = min(max(candidate,minP),maxP)
            P2=candidate; harmonic_chosen="forced_valid_range"

    # twin-sine fit
    (A1,phi1,A2,phi2,C),cov = fit_twin_sine(time,flux,flux_err,P1,P2)
    model_twin = twin_sine_model(time,A1,phi1,P1,A2,phi2,P2,C)
    perr = np.sqrt(np.diag(cov)) if cov is not None else [np.nan]*5

    params = dict(A1=A1,A1_err=perr[0],phi1=phi1,phi1_err=perr[1],P1=P1,
                  A2=A2,A2_err=perr[2],phi2=phi2,phi2_err=perr[3],P2=P2,
                  C=C,C_err=perr[4],harmonic_choice=harmonic_chosen)

# rough period errors
    def estimate_period_err(Pgrid, power, Ppeak):
        """Estimate period uncertainty from periodogram peak width."""
        if len(Pgrid) < 5:
            return np.nan
        
        # Try with adaptive window sizes
        for frac in [0.02, 0.05, 0.10, 0.15]:
            m = (Pgrid > Ppeak * (1 - frac)) & (Pgrid < Ppeak * (1 + frac))
            if not np.any(m) or np.sum(m) < 5:
                continue
            
            Pg, pw = Pgrid[m], power[m]
            
            try:
                # Try parabolic fit
                a, b, c = np.polyfit(Pg, pw, 2)
                
                if a >= 0:
                    # Wrong curvature - fallback to FWHM
                    pmax = np.max(pw)
                    target = pmax / 2.0
                    above_half = pw > target
                    if np.sum(above_half) >= 2:
                        indices = np.where(above_half)[0]
                        fwhm = Pg[indices[-1]] - Pg[indices[0]]
                        return fwhm / 2.355
                    continue
                
                # Good parabola - find 1/e points
                pmax = np.max(pw)
                target = pmax / np.e
                roots = np.roots([a, b, c - target])
                roots = roots[np.isreal(roots)].real
                if len(roots) == 2:
                    return 0.5 * np.abs(roots[1] - roots[0])
                    
            except Exception:
                continue
        
        return np.nan

    if ASTROPY_OK and (P1_fixed is None):
        Pglob,powglob,_ = gls_periodogram(time,flux,flux_err,lo_eff,hi_eff,nfreq)
        sigP1 = estimate_period_err(Pglob,powglob,P1)
        resid_for_P2 = flux - (C + sine_model(time,A1,phi1,P1))
        Ph2,powh2,_ = gls_periodogram(time,resid_for_P2,flux_err,max(minP,0.25*P1),min(maxP,4*P1),max(nfreq//2,5000))
        sigP2 = estimate_period_err(Ph2,powh2,P2)
    else:
        sigP1=np.nan; sigP2=np.nan
    params["P1_err_est"]=sigP1; params["P2_err_est"]=sigP2
    tspan = np.max(t)-np.min(t)
    params["Tspan_days"]=tspan; params["cycles_at_P1_over_span"]=(tspan/P1 if (np.isfinite(P1) and P1>0) else np.nan)

    metrics_twin = chi2_metrics(flux, flux_err, model_twin, k_params=5, mode=ic_mode)

    return (Pgrid,power,P1,resid1,Ph,pow_h,Pd,pow_d,P2,params,
            model_single,metrics_single,model_twin,metrics_twin)

# ------------------------------ Plotting ----------------------------------
def make_plots(outdir, df_raw, df_clean, removed_df,
               P, power, P1, Ph, pow_h, Pd, pow_d, P2,
               time, flux, flux_err, model, param_dict, metrics, target_name="target"):

    def save_all(path_with_ext: str, dpi_png: int = 200) -> None:
        base, _ = os.path.splitext(path_with_ext)
        plt.savefig(base + ".png", dpi=dpi_png, bbox_inches="tight")
        plt.savefig(base + ".pdf", dpi=dpi_png, bbox_inches="tight")

    # Raw & cleaned
    plt.figure(figsize=(11,4))
    if removed_df is not None and len(removed_df)>0:
        plt.scatter(removed_df["time"], removed_df["flux"], s=10, marker="x", alpha=0.6, label="removed")
    plt.errorbar(df_clean["time"], df_clean["flux"], yerr=df_clean["flux_err"], fmt=".", ms=2, alpha=0.8, label="kept")
    plt.xlabel("Time [days]"); plt.ylabel("Relative flux")
    plt.title(f"{target_name}: Raw & cleaned")
    plt.legend(loc="best", fontsize=8); plt.tight_layout()
    save_all(os.path.join(outdir,"01_raw_cleaned.png")); plt.close()

    # Global GLS
    plt.figure(figsize=(11,4))
    if len(P):
        plt.plot(P, power, lw=1); plt.axvline(P1, ls="--", label=f"P1={P1:.3f} d")
        plt.gca().invert_xaxis(); plt.xlabel("Period [days] (x-axis inverted)"); plt.ylabel("Power")
        plt.title(f"{target_name}: GLS (cleaned)"); plt.legend(); plt.tight_layout()
    save_all(os.path.join(outdir,"02_gls_global.png")); plt.close()

    # Harmonics
    plt.figure(figsize=(11,4))
    if len(Ph): plt.plot(Ph, pow_h, lw=1, label="around 0.5*P1")
    if len(Pd): plt.plot(Pd, pow_d, lw=1, label="around 2*P1")
    if P2 is not None: plt.axvline(P2, ls="--", label=f"P2={P2:.3f} d ({param_dict['harmonic_choice']})")
    plt.gca().invert_xaxis(); plt.xlabel("Period [days] (x-axis inverted)"); plt.ylabel("Power")
    plt.title(f"{target_name}: GLS after pre-whitening first sine")
    plt.legend(); plt.tight_layout()
    save_all(os.path.join(outdir,"03_gls_harmonics.png")); plt.close()

    # Model in time (twin-sine)
    idx = np.argsort(time)
    span = time[idx][-1] - time[idx][0]
    samples_per_cycle = 400 if np.isfinite(P1) and P1 > 0 else 400
    n_dense = int(max(1000, samples_per_cycle * (span / max(P1, 1e-6))))
    t_dense = np.linspace(time[idx][0], time[idx][-1], n_dense)
    model_dense = twin_sine_model(
        t_dense, param_dict["A1"], param_dict["phi1"], P1,
        param_dict["A2"], param_dict["phi2"], P2, param_dict["C"]
    )
    plt.figure(figsize=(11,4))
    plt.errorbar(time, flux, yerr=flux_err, fmt=".", ms=2, alpha=0.7, label="data")
    plt.plot(t_dense, model_dense, lw=1.8, label=f"twin-sine model (P1={P1:.3f} d, P2={P2:.3f} d)")
    plt.xlabel("Time [days]"); plt.ylabel("Relative flux")
    plt.title(f"{target_name}: Model fit")
    plt.legend(); plt.tight_layout()
    save_all(os.path.join(outdir, "04_model_timeseries.png")); plt.close()

    # Residuals
    resid = flux - model
    plt.figure(figsize=(11,4)); plt.axhline(0, color="k", lw=0.8)
    plt.errorbar(time, resid, yerr=flux_err, fmt=".", ms=2, alpha=0.7)
    plt.xlabel("Time [days]"); plt.ylabel("Residual [flux]")
    plt.title(f"{target_name}: Residuals")
    plt.tight_layout(); save_all(os.path.join(outdir,"05_residuals.png")); plt.close()

    # Phase-folded at P1 (zero-centered, annotate P1)
    tref = - (param_dict["phi1"] * P1) / (2*np.pi)
    phi = ((time - tref) / P1) % 1.0
    phi[phi > 0.5] -= 1.0

    plt.figure(figsize=(7,5))
    plt.errorbar(phi, flux, yerr=flux_err, fmt=".", ms=2, alpha=0.6, label="data")
    ph_grid = np.linspace(-0.5, 0.5, 1000)
    t_grid = tref + ph_grid * P1
    y_med = np.nanmedian(flux)
    vert_shift = y_med - param_dict["C"]
    m1 = (param_dict["C"] + vert_shift) + sine_model(t_grid, param_dict["A1"], param_dict["phi1"], P1)
    plt.plot(ph_grid, m1, lw=1.6, ls="--", label=f"fundamental only (P1={P1:.3f} d)")
    plt.axvline(0.0, lw=1.0, alpha=0.5)
    p1_err = param_dict.get("P1_err_est", np.nan)
    annot = f"P1 = {P1:.3f} d" if not np.isfinite(p1_err) else f"P1 = {P1:.3f} ± {p1_err:.3f} d"
    plt.text(0.98, 0.02, annot, transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.25", alpha=0.15))
    plt.xlim(-0.5, 0.5); plt.xticks([-0.5,-0.25,0,0.25,0.5])
    plt.xlabel("Phase @ P1 (zero-centered)"); plt.ylabel("Relative flux")
    plt.title(f"{target_name}: Phase-folded P1")
    plt.legend(); plt.tight_layout()
    save_all(os.path.join(outdir,"06_phase_P1.png")); plt.close()

    # Full-model phase plot (06b)
    m_full = vert_shift + twin_sine_model(
        t_grid, param_dict["A1"], param_dict["phi1"], P1,
        param_dict["A2"], param_dict["phi2"], P2, param_dict["C"]
    )
    plt.figure(figsize=(7,5))
    plt.errorbar(phi, flux, yerr=flux_err, fmt=".", ms=2, alpha=0.6, label="data")
    plt.plot(ph_grid, m_full, lw=1.6, label=f"full model (P1={P1:.3f} d, P2={P2:.3f} d)")
    plt.axvline(0.0, lw=1.0, alpha=0.5)
    plt.xlim(-0.5, 0.5); plt.xticks([-0.5,-0.25,0,0.25,0.5])
    plt.xlabel("Phase @ P1 (zero-centered)"); plt.ylabel("Relative flux")
    plt.title(f"{target_name}: Phase-folded P1 (full model)")
    plt.legend(); plt.tight_layout()
    save_all(os.path.join(outdir, "06b_phase_P1_fullmodel.png")); plt.close()

# ------------------------------ Main --------------------------------------
def parse_exclude_periods(s):
    if not s:
        return None
    out = []
    for chunk in str(s).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo,hi = chunk.split("-",1)
            lo=float(lo.strip()); hi=float(hi.strip())
            if hi<lo: lo,hi = hi,lo
            out.append((lo,hi))
    return out if out else None

def main():
    p = argparse.ArgumentParser(
        description="Twin-sine rotational model with built-in QC (night-aware, rolling-window) and paper-quality outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
        python phot_var_twin_sine_QC.py data.txt --delim whitespace \
          --time-col JD --mag-col Magnitude --mag-err-col Uncertainty \
          --observer-col "Observer Code" --filter-col Band --band-include V \
          --group gap --gap-hours 6 --per-night-sigma 3 --roll-window 0.5 --roll-sigma 4 \
          --qc-minN 40 --qc-chi-thresh 3 --qc-ofrac-thresh 0.05 --qc-mad-factor 1.5
        """)
    )

    p.add_argument("input")
    p.add_argument("--delim", choices=["comma","tab","semicolon","whitespace"], default=None)
    p.add_argument("--comment-char", default=None)
    p.add_argument("--skiprows", type=int, default=None)
    # columns
    p.add_argument("--time-col", default=None)
    p.add_argument("--flux-col", default=None)
    p.add_argument("--flux-err-col", default=None)
    p.add_argument("--mag-col", default=None)
    p.add_argument("--mag-err-col", default=None)
    p.add_argument("--observer-col", default=None)
    p.add_argument("--filter-col", default=None)
    p.add_argument("--flag-col", default=None)
    # subsets
    p.add_argument("--band-include", default=None)
    p.add_argument("--band-exclude", default=None)
    p.add_argument("--observer-include", default=None)
    p.add_argument("--observer-exclude", default=None)
    # cleaning (new/better)
    p.add_argument("--group", choices=["gap","calendar"], default="gap")
    p.add_argument("--gap-hours", type=float, default=6.0)
    p.add_argument("--per-night-minN", type=int, default=20)
    p.add_argument("--per-night-sigma", type=float, default=3.0)
    p.add_argument("--global-sigma", type=float, default=3.0)
    p.add_argument("--global-iters", type=int, default=2)
    p.add_argument("--roll-window", type=float, default=0.5, help="Full width in days (0 disables)")
    p.add_argument("--roll-sigma", type=float, default=4.0)
    p.add_argument("--thin-maxN", type=int, default=0, help="0 disables thinning")
    p.add_argument("--err-max", type=float, default=None, help="Absolute cap on flux_err")
    p.add_argument("--err-median-factor", type=float, default=2.0, help="Cap flux_err ≤ factor × median(err)")
    p.add_argument("--flag-reject", default=None)
    # legacy cleaner knobs kept (mapped internally)
    p.add_argument("--sigma-clip", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--iters", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--per-observer-sigma", type=float, default=None, help=argparse.SUPPRESS)
    # QC thresholds (for night drops)
    p.add_argument("--qc-minN", type=int, default=40)
    p.add_argument("--qc-chi-thresh", type=float, default=3.0)
    p.add_argument("--qc-ofrac-thresh", type=float, default=0.05)
    p.add_argument("--qc-mad-factor", type=float, default=1.5)
    p.add_argument("--obs-bad-frac", type=float, default=0.67)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--plot-all-sessions", action="store_true")
    p.add_argument("--plot-all", action="store_true")
    # period search
    p.add_argument("--minP", type=float, default=1.0)
    p.add_argument("--maxP", type=float, default=200.0)
    p.add_argument("--nfreq", type=int, default=100000)
    p.add_argument("--harmonic-window", type=float, default=0.15)
    # priors
    p.add_argument("--P1-prior", type=float, default=None)
    p.add_argument("--P1-prior-frac", type=float, default=0.25)
    p.add_argument("--P1-fixed", type=float, default=None)
    p.add_argument("--exclude-periods", default=None)
    # meta
    p.add_argument("--target", default="target")
    p.add_argument("--outdir", default=None)
    p.add_argument(
        "--ic-mode",
        choices=["chi2","gauss"],
        default="chi2",
        help="Which definition to put under AIC/BIC: "
             "'chi2' (AIC=χ²+2k, BIC=χ²+k ln n; matches paper) or "
             "'gauss' (AIC=2k+2·NLL, BIC=k ln n+2·NLL). "
             "Both sets (…_chi, …_ll) are always saved."
    )

    # --- NEW (MCMC): fully additive, off by default. When --mcmc is not passed,
    # nothing below this point changes any existing behavior or output. ---
    p.add_argument("--mcmc", action="store_true",
                    help="Also compute posterior (MCMC, via emcee) error bars for BOTH the "
                         "single-sine and twin-sine models, in BOTH fixed-period and free-period "
                         "modes (4 runs total). Requires 'emcee' and 'corner' "
                         "(pip install emcee corner); if missing, this step is skipped with a "
                         "warning and everything else runs exactly as before.")
    p.add_argument("--mcmc-walkers", type=int, default=64,
                    help="Number of emcee walkers (default 64; must be >= 2x the number of free params).")
    p.add_argument("--mcmc-steps", type=int, default=5000, help="Total MCMC steps per walker.")
    p.add_argument("--mcmc-burn", type=int, default=1000, help="Burn-in steps discarded before summarizing.")
    p.add_argument("--mcmc-thin", type=int, default=5, help="Thinning applied to the post-burn-in chain.")
    p.add_argument("--mcmc-period-prior-k", type=float, default=3.0,
                    help="Width of the free-period Gaussian prior, in units of the existing "
                         "estimate_period_err() periodogram-peak-width estimate (default 3.0).")
    p.add_argument("--mcmc-seed", type=int, default=None, help="Random seed for walker initialization.")
    p.add_argument("--mcmc-progress", action="store_true", help="Show emcee's console progress bar.")
    p.add_argument("--mcmc-moves", choices=["de", "stretch"], default="de",
                    help="emcee move strategy. 'de' (default) = DEMove+DESnookerMove mixture, "
                         "emcee's recommended choice for correlated/degenerate posteriors -- "
                         "usually mixes much faster (lower autocorrelation time) than 'stretch' "
                         "on real (noisy, multi-observer) photometry. 'stretch' = emcee's "
                         "original default, kept as a fallback.")

    args = p.parse_args()

    # delimiter
    delim = None
    if args.delim == "comma": delim = ","
    elif args.delim == "tab": delim = "\t"
    elif args.delim == "semicolon": delim = ";"
    elif args.delim == "whitespace": delim = None

    df_in = read_table(args.input, delimiter=delim, comment=args.comment_char, skiprows=args.skiprows)

    user_map = {}
    if args.time_col: user_map["time"]=args.time_col
    if args.flux_col: user_map["flux"]=args.flux_col
    if args.flux_err_col: user_map["flux_err"]=args.flux_err_col
    if args.mag_col: user_map["mag"]=args.mag_col
    if args.mag_err_col: user_map["mag_err"]=args.mag_err_col
    if args.observer_col: user_map["observer"]=args.observer_col
    if args.filter_col: user_map["filt"]=args.filter_col
    if args.flag_col: user_map["flag"]=args.flag_col

    colmap = auto_map_columns(df_in, user_map=user_map)
    df = prepare_data(df_in, colmap, normalize=True)

    # subsets
    if args.band_include is not None and "filter" in df.columns:
        df = df.loc[df["filter"].astype(str)==str(args.band_include)].copy()
    if args.band_exclude is not None and "filter" in df.columns:
        df = df.loc[df["filter"].astype(str)!=str(args.band_exclude)].copy()

    if "observer" in df.columns:
        if args.observer_include is not None:
            keep=[s.strip() for s in str(args.observer_include).split(",") if s.strip()]
            df = df.loc[df["observer"].astype(str).isin(keep)].copy()
        if args.observer_exclude is not None:
            drop=[s.strip() for s in str(args.observer_exclude).split(",") if s.strip()]
            df = df.loc[~df["observer"].astype(str).isin(drop)].copy()

    outdir = args.outdir or os.path.join("phot_var_runs", f"{args.target}_{timestamp()}"); ensure_dir(outdir)

    flag_vals = None
    if args.flag_reject:
        try:
            flag_vals = [int(x.strip()) for x in args.flag_reject.split(",")]
        except Exception:
            flag_vals = [x.strip() for x in args.flag_reject.split(",")]

    # ---------- CLEAN (better way) ----------
    df_clean, removed = smart_clean(
        df,
        flag_reject_values=flag_vals,
        err_max=args.err_max,
        err_median_factor=args.err_median_factor,
        group=args.group,
        gap_hours=args.gap_hours,
        per_night_minN=args.per_night_minN,  # guard
        per_night_sigma=args.per_night_sigma,
        global_sigma=args.global_sigma,
        global_iters=args.global_iters,
        roll_window=args.roll_window,
        roll_sigma=args.roll_sigma,
        thin_maxN=args.thin_maxN,
        outdir=outdir
    )

    # Save input & cleaned snapshots
    df.to_csv(os.path.join(outdir,"input_normalized.csv"), index=False)
    df_clean.to_csv(os.path.join(outdir,"cleaned_initial.csv"), index=False)

    # ---------- QC pass (sessions) ----------
    d_qc = tag_sessions(df_clean, group=args.group, gap_hours=args.gap_hours).copy()
    # residuals against global median for QC (pre-fit)
    d_qc["resid"] = d_qc["flux"] - np.nanmedian(d_qc["flux"])
    sessions = summarize_sessions(d_qc)
    bad_mask = flag_bad_sessions(sessions, args.qc_chi_thresh, args.qc_ofrac_thresh, args.qc_minN, args.qc_mad_factor)
    sessions["is_bad"] = bad_mask
    obs = rollup_observers(sessions)
    bad_frac = sessions.groupby("observer")["is_bad"].mean().reindex(obs["observer"]).fillna(0).values
    obs["frac_bad_sessions"] = bad_frac
    obs["exclude_observer"] = obs["frac_bad_sessions"] >= args.obs_bad_frac

    # write QC tables
    sess_csv = os.path.join(outdir, "observer_session_qc.csv")
    sess_js = os.path.join(outdir, "observer_session_qc.json")
    obs_csv = os.path.join(outdir, "observer_qc_summary.csv")
    obs_js = os.path.join(outdir, "observer_qc_summary.json")
    sessions.to_csv(sess_csv, index=False); sessions.to_json(sess_js, orient="records", indent=2)
    obs.to_csv(obs_csv, index=False); obs.to_json(obs_js, orient="records", indent=2)

    # candidate drop ranges
    drops = sessions.loc[sessions["is_bad"],
                         ["observer","session_id","N","resid_MAD","chi2_over_dof","outlier_frac","t_min","t_max"]]
    drops.to_csv(os.path.join(outdir, "session_drop_ranges.csv"), index=False)

    # plots for QC
    metric_col = "chi2_over_dof" if not sessions["chi2_over_dof"].isna().all() else "resid_MAD"
    worst_sessions = sessions.sort_values([metric_col, "outlier_frac", "N"], ascending=[False, False, False])
    sp = sessions.copy() if args.plot_all_sessions else worst_sessions.head(args.top_k or 20)
    lab_sessions = [f"{r.observer} | {r.session_id} | JD {r.t_min:.2f}–{r.t_max:.2f} (N={int(r.N)})" for r in sp.itertuples()]
    vals_sessions = sp[metric_col].fillna(0).values
    _barh(os.path.join(outdir, "07_observer_quality_sessions.png"), lab_sessions, vals_sessions,
          "Session quality (worst first)",
          f"{'χ²/dof' if metric_col=='chi2_over_dof' else 'Residual MAD'} (higher = worse)",
          height_per=0.35, min_h=6, fontsize=7)

    op = obs.copy()
    if not args.plot_all:
        op = op.sort_values(["frac_bad_sessions","med_chi","med_MAD","N_total"],
                            ascending=[False,False,False,False]).head(12)
    lab_obs = [f"{r.observer} (bad_frac={r.frac_bad_sessions:.2f}, n_sess={int(r.n_sess)})" for r in op.itertuples()]
    vals_obs = op["frac_bad_sessions"].values
    _barh(os.path.join(outdir, "08_observer_quality_observers.png"), lab_obs, vals_obs,
          "Observer bad-session fraction", "Fraction of bad sessions (≥ threshold)", xlim=(0,1.0))

    # Apply night drops BEFORE the fit
    bad_sessions = set(sessions.loc[sessions["is_bad"], "session_id"].astype(str))
    if len(bad_sessions):
        df_clean = tag_sessions(df_clean, group=args.group, gap_hours=args.gap_hours)
        keep = ~df_clean["session_id"].astype(str).isin(bad_sessions)
        df_clean2 = df_clean.loc[keep, ["time","flux","flux_err","observer","filter","flag"]].reset_index(drop=True)
        df_clean2.to_csv(os.path.join(outdir,"cleaned.csv"), index=False)
        print(f"Dropped {np.sum(~keep)} points from {len(bad_sessions)} bad sessions before fitting.")
    else:
        df_clean2 = df_clean.copy()
        df_clean2.to_csv(os.path.join(outdir,"cleaned.csv"), index=False)

    # ---------- Period search & model on final cleaned data ----------
    exclude_ranges = parse_exclude_periods(args.exclude_periods)
    (P, power, P1, resid1, Ph, pow_h, Pd, pow_d, P2, params,
     model_single, metrics_single, model_twin, metrics_twin) = prewhiten_twin_sine(
        df_clean2["time"].values, df_clean2["flux"].values, df_clean2["flux_err"].values,
        minP=args.minP, maxP=args.maxP, nfreq=args.nfreq,
        harmonic_window=args.harmonic_window,
        P1_fixed=args.P1_fixed, P1_prior=args.P1_prior, P1_prior_frac=args.P1_prior_frac,
        exclude_ranges=exclude_ranges,
        ic_mode=args.ic_mode
    )

    # Output timeseries with BOTH models
    out = df_clean2.copy()
    out["model_single"] = model_single
    out["resid_single"] = out["flux"] - out["model_single"]
    out["model_twin"] = model_twin
    out["resid_twin"] = out["flux"] - out["model_twin"]
    # Back-compat
    out["model"] = out["model_twin"]
    out["resid"] = out["resid_twin"]
    out.to_csv(os.path.join(outdir,"model_timeseries.csv"), index=False)

    # ---------- NEW (MCMC): opt-in posterior error estimation ----------
    # Fully additive: if --mcmc is not passed, mcmc_results stays None and
    # nothing below (summary.json, fit_report.txt) changes at all.
    mcmc_results = None
    if args.mcmc:
        missing = [pkg for pkg, ok in (("emcee", EMCEE_OK), ("corner", CORNER_OK)) if not ok]
        if missing:
            print(f"WARNING: --mcmc requested but missing package(s): {', '.join(missing)}. "
                  f"Install via: pip install {' '.join(missing)}. Skipping MCMC step "
                  f"(all other outputs are unaffected).")
        else:
            print("Running MCMC error estimation: single-sine & twin-sine models, "
                  "fixed-period & free-period modes (4 runs)...")
            mcmc_results = run_all_mcmc(
                outdir,
                df_clean2["time"].values, df_clean2["flux"].values, df_clean2["flux_err"].values,
                params, P1, P2, args.minP, args.maxP,
                nwalkers=args.mcmc_walkers, nsteps=args.mcmc_steps, nburn=args.mcmc_burn,
                thin=args.mcmc_thin, period_prior_k=args.mcmc_period_prior_k,
                seed=args.mcmc_seed, progress=args.mcmc_progress, moves=args.mcmc_moves,
            )
            print("MCMC done -> 09_mcmc_corner_*.png/.pdf, 10_mcmc_trace_*.png/.pdf written.")

    # Summary JSON includes metrics for BOTH models
    summary = dict(
        target=args.target,
        columns=colmap,
        n_points_input=len(df),
        n_points_clean_initial=len(df_clean),
        n_points_clean_final=len(df_clean2),
        periods=dict(P1=params["P1"], P1_err_est=params["P1_err_est"],
                     P2=params["P2"], P2_err_est=params["P2_err_est"],
                     harmonic_choice=params["harmonic_choice"]),
        params=dict(A1=params["A1"], A1_err=params["A1_err"],
                    phi1=params["phi1"], phi1_err=params["phi1_err"],
                    A2=params["A2"], A2_err=params["A2_err"],
                    phi2=params["phi2"], phi2_err=params["phi2_err"],
                    C=params["C"], C_err=params["C_err"]),
        metrics=dict(
            single_sine=metrics_single,   # k=3
            twin_sine=metrics_twin        # k=5
        ),
        settings=dict(
            minP=args.minP, maxP=args.maxP, nfreq=args.nfreq, harmonic_window=args.harmonic_window,
            group=args.group, gap_hours=args.gap_hours,
            per_night_minN=args.per_night_minN, per_night_sigma=args.per_night_sigma,
            global_sigma=args.global_sigma, global_iters=args.global_iters,
            roll_window=args.roll_window, roll_sigma=args.roll_sigma, thin_maxN=args.thin_maxN,
            err_max=args.err_max, err_median_factor=args.err_median_factor, flag_reject=flag_vals,
            band_include=args.band_include, band_exclude=args.band_exclude,
            observer_include=args.observer_include, observer_exclude=args.observer_exclude,
            P1_prior=args.P1_prior, P1_prior_frac=args.P1_prior_frac, P1_fixed=args.P1_fixed,
            exclude_periods=args.exclude_periods, ic_mode=args.ic_mode
        ),
        software=dict(astropy_timeseries=ASTROPY_OK),
        timing=dict(Tspan_days=params.get("Tspan_days", None),
                    cycles_at_P1_over_span=params.get("cycles_at_P1_over_span", None)),
        qc=dict(qc_minN=args.qc_minN, qc_chi_thresh=args.qc_chi_thresh,
                qc_ofrac_thresh=args.qc_ofrac_thresh, qc_mad_factor=args.qc_mad_factor,
                n_bad_sessions=int(sessions["is_bad"].sum())),
        # NEW: posterior (MCMC) error bars, None unless --mcmc was passed. Does not
        # replace or alter the existing 'params'/'periods' fields above.
        mcmc=mcmc_results
    )
    with open(os.path.join(outdir,"summary.json"),"w") as f:
        json.dump(summary,f,indent=2,sort_keys=True)

    # Fit report with BOTH metrics blocks
    with open(os.path.join(outdir,"fit_report.txt"),"w") as f:
        f.write(f"Target: {args.target}\n")
        f.write(f"Points: input={len(df)}, cleaned_initial={len(df_clean)}, cleaned_final={len(df_clean2)}\n")
        f.write(f"Timespan: {params.get('Tspan_days', np.nan):.3f} d; cycles over span at P1: {params.get('cycles_at_P1_over_span', np.nan):.2f}\n")
        f.write("Periods:\n")
        f.write(f" P1 = {params['P1']:.6f} d (±~{params['P1_err_est'] if params['P1_err_est'] is not None else np.nan:.6f})\n")
        f.write(f" P2 = {params['P2']:.6f} d (±~{params['P2_err_est'] if params['P2_err_est'] is not None else np.nan:.6f}) [{params['harmonic_choice']}]\n")
        f.write("Amplitudes/Phases:\n")
        f.write(f" A1 = {params['A1']:.6g} ± {params['A1_err']:.2g}, phi1={params['phi1']:.4f} ± {params['phi1_err']:.4f}\n")
        f.write(f" A2 = {params['A2']:.6g} ± {params['A2_err']:.2g}, phi2={params['phi2']:.4f} ± {params['phi2_err']:.4f}\n")
        f.write(f" C  = {params['C']:.6g} ± {params['C_err']:.2g}\n\n")

        f.write("Metrics (single-sine @ P1):\n")
        for k,v in metrics_single.items():
            f.write(f"  {k:>6s} = {v:.6f}\n")
        f.write("\nMetrics (twin-sine):\n")
        for k,v in metrics_twin.items():
            f.write(f"  {k:>6s} = {v:.6f}\n")

        try:
            excl_obs = obs.loc[obs["exclude_observer"], "observer"].tolist()
        except Exception:
            excl_obs = []
        if excl_obs:
            f.write("\n\nSuggested --observer-exclude (escalated observers):\n")
            f.write(",".join(excl_obs)+"\n")
        else:
            f.write("\n\nSuggested --observer-exclude: (none — use session_drop_ranges.csv to prune specific nights)\n")

        # --- NEW (MCMC): appended only if --mcmc was used; existing content above is unchanged ---
        if mcmc_results:
            f.write("\n\nMCMC posterior error bars (emcee; see summary.json['mcmc'] for full diagnostics):\n")
            for tag, res in mcmc_results.items():
                f.write(f"\n[{tag}] period_mode={res['period_mode']}, "
                        f"converged={res['converged']}, "
                        f"acceptance_frac={res['acceptance_fraction']:.3f}, "
                        f"autocorr_time_max={res['autocorr_time_max']:.1f}\n")
                for name, s in res["summary"].items():
                    f.write(f"   {name:>5s} = {s['median']:.6g}  (+{s['err_hi']:.3g} / -{s['err_lo']:.3g})\n")

    # plots (final data; use twin model for residuals)
    make_plots(outdir, df, df_clean2, removed if len(removed)>0 else None,
               P, power, P1, Ph, pow_h, Pd, pow_d, P2,
               df_clean2["time"].values, df_clean2["flux"].values, df_clean2["flux_err"].values,
               model_twin, params, metrics_twin, target_name=args.target)

    print(f"Done. Outputs in: {outdir}")
    print("Key files:")
    print(" - fit_report.txt")
    print(" - summary.json")
    print(" - model_timeseries.csv")
    print(" - cleaned_initial.csv, cleaned.csv")
    print(" - removed_points.csv, cleaning_summary_by_observer.csv, cleaning_summary_by_session.csv")
    print(" - observer_session_qc.(csv|json), observer_qc_summary.(csv|json), session_drop_ranges.csv")
    print(" - 01_raw_cleaned.png ... 08_observer_quality_observers.png")
    print(" - 06_phase_P1.png (fundamental only), 06b_phase_P1_fullmodel.png (full model)")
    if mcmc_results:
        print(" - 09_mcmc_corner_{single,twin}_{fixed,free}.png/.pdf (posterior corner plots)")
        print(" - 10_mcmc_trace_{single,twin}_{fixed,free}.png/.pdf (walker trace plots)")
        print(" - MCMC posterior summaries also in summary.json['mcmc'] and fit_report.txt")

if __name__ == "__main__":
    main()
