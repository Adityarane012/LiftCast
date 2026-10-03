"""Deterministic trend-based stall detection for LiftCast.

Implements candidate rule from PROJECT.md §8:
Least-squares linear regression slope of e1RM vs. weeks over a trailing window of W days.
Stalled if normalized slope (slope / window mean e1RM) < theta %/week, given >= min_n sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class StallExperimentResult:
    window_days: int
    theta_pct_week: float
    overall_flag_rate: float
    inside_flag_rate: float
    outside_flag_rate: float
    discrimination_ratio: float
    total_evaluated: int
    inside_count: int
    outside_count: int


def detect_stalls(
    df: pd.DataFrame,
    window_days: int = 56,
    theta_pct_week: float = 0.0,
    min_n: int = 4,
) -> pd.DataFrame:
    """Calculate stall flag for each session in dataframe.
    
    Accepts df with: ['date', 'top_e1rm'] (and optionally 'exercise').
    Returns dataframe with: ['date', 'top_e1rm', 'slope_pct_week', 'n_in_window', 'stalled'].
    
    - window: trailing [curr_date - window_days, curr_date]
    - slope: least-squares slope in kg/week
    - slope_pct_week: (slope / mean_e1rm_in_window) * 100.0
    - stalled: True if n_in_window >= min_n and slope_pct_week < theta_pct_week
    """
    if df.empty:
        return pd.DataFrame(
            columns=["date", "top_e1rm", "slope_pct_week", "n_in_window", "stalled"]
        )

    out = df.copy()
    out["dt"] = pd.to_datetime(out["date"])
    out = out.sort_values("dt").reset_index(drop=True)

    slopes_pct = []
    n_in_windows = []
    stalled_flags = []

    for i in range(len(out)):
        curr_dt = out.loc[i, "dt"]
        cutoff_dt = curr_dt - pd.Timedelta(days=window_days)

        # Trailing window including current session
        window_mask = (out["dt"] >= cutoff_dt) & (out["dt"] <= curr_dt)
        window_df = out.loc[window_mask]

        n_pts = len(window_df)
        n_in_windows.append(n_pts)

        if n_pts < min_n:
            slopes_pct.append(None)
            stalled_flags.append(None)
            continue

        first_dt = window_df["dt"].min()
        xs = (window_df["dt"] - first_dt).dt.days.values / 7.0  # weeks
        ys = window_df["top_e1rm"].values.astype(float)

        mean_y = float(np.mean(ys))
        var_x = float(np.var(xs))

        if var_x < 1e-6 or mean_y <= 0:
            slope_pct = 0.0
        else:
            slope = float(np.cov(xs, ys)[0, 1] / var_x)
            slope_pct = round((slope / mean_y) * 100.0, 3)

        is_stalled = bool(slope_pct < theta_pct_week)

        slopes_pct.append(slope_pct)
        stalled_flags.append(is_stalled)

    out["slope_pct_week"] = slopes_pct
    out["n_in_window"] = n_in_windows
    out["stalled"] = stalled_flags

    res_cols = ["date"]
    if "exercise" in out.columns:
        res_cols.append("exercise")
    res_cols.extend(["top_e1rm", "slope_pct_week", "n_in_window", "stalled"])
    if "is_plateau" in out.columns:
        res_cols.append("is_plateau")

    return out[res_cols]


def run_stall_grid_experiment(
    history_df: pd.DataFrame,
    plateau_start: str = "2026-04-01",
    plateau_end: str = "2026-09-30",
    windows: Sequence[int] = (42, 56, 70, 84),
    thetas: Sequence[float] = (0.0, 0.5),
    min_n: int = 4,
) -> list[StallExperimentResult]:
    """Run pre-registered stall detection grid experiment (§8.3).
    
    Computes overall flag rate and discrimination ratio (inside ÷ outside).
    """
    results: list[StallExperimentResult] = []

    dt_start = pd.to_datetime(plateau_start)
    dt_end = pd.to_datetime(plateau_end)

    for w in windows:
        for th in thetas:
            detected = detect_stalls(history_df, window_days=w, theta_pct_week=th, min_n=min_n)
            valid = detected.dropna(subset=["stalled"]).copy()

            if valid.empty:
                continue

            valid["dt"] = pd.to_datetime(valid["date"])
            total_n = len(valid)

            # Check if ground truth column exists, else use hand-labeled dates
            if "is_plateau" in valid.columns:
                inside_mask = valid["is_plateau"] == True
            else:
                inside_mask = (valid["dt"] >= dt_start) & (valid["dt"] <= dt_end)

            outside_mask = ~inside_mask

            inside_df = valid[inside_mask]
            outside_df = valid[outside_mask]

            overall_rate = float(valid["stalled"].mean())
            inside_rate = float(inside_df["stalled"].mean()) if not inside_df.empty else 0.0
            outside_rate = float(outside_df["stalled"].mean()) if not outside_df.empty else 0.0

            discrim = inside_rate / max(1e-4, outside_rate)

            results.append(
                StallExperimentResult(
                    window_days=w,
                    theta_pct_week=th,
                    overall_flag_rate=round(overall_rate, 3),
                    inside_flag_rate=round(inside_rate, 3),
                    outside_flag_rate=round(outside_rate, 3),
                    discrimination_ratio=round(discrim, 2),
                    total_evaluated=total_n,
                    inside_count=len(inside_df),
                    outside_count=len(outside_df),
                )
            )

    return results


def check_early_stall_warning(
    recent_predictions: Sequence[float],
    recent_rollmeans: Sequence[float],
    threshold_pct: float = 0.01,
) -> bool:
    """Soft early warning using model forecast (§8.5).
    
    If the forecast for the next session is within ±1% of the rolling mean
    for 2 consecutive predictions -> 'possible stall forming'.
    """
    if len(recent_predictions) < 2 or len(recent_rollmeans) < 2:
        return False

    preds = recent_predictions[-2:]
    means = recent_rollmeans[-2:]

    for p, m in zip(preds, means):
        if m <= 0:
            return False
        diff_pct = abs(p - m) / m
        if diff_pct > threshold_pct:
            return False

    return True
