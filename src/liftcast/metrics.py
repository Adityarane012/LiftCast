"""Metrics and feature engineering for LiftCast.

Deterministic, pure functions for:
- Epley 1-Rep Max estimation (e1RM)
- Session top-set extraction (filters warm-ups)
- Leakage-free feature calculation for forecasting models
- Strength tier classification and projection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import Sequence
import numpy as np
import pandas as pd


# Strength tier ratio thresholds (e1RM / bodyweight_kg)
TIER_THRESHOLDS: dict[str, list[tuple[str, float]]] = {
    "Bench Press": [
        ("Beginner", 0.60),
        ("Novice", 0.80),
        ("Intermediate", 1.05),
        ("Advanced", 1.35),
        ("Elite", 1.65),
    ],
    "Deadlift": [
        ("Beginner", 1.00),
        ("Novice", 1.30),
        ("Intermediate", 1.70),
        ("Advanced", 2.10),
        ("Elite", 2.50),
    ],
    "Bent Over Row": [
        ("Beginner", 0.50),
        ("Novice", 0.70),
        ("Intermediate", 0.95),
        ("Advanced", 1.20),
        ("Elite", 1.45),
    ],
    "Lat Pulldown": [
        ("Beginner", 0.50),
        ("Novice", 0.70),
        ("Intermediate", 0.95),
        ("Advanced", 1.20),
        ("Elite", 1.40),
    ],
    "Romanian Deadlift": [
        ("Beginner", 0.80),
        ("Novice", 1.10),
        ("Intermediate", 1.45),
        ("Advanced", 1.80),
        ("Elite", 2.15),
    ],
    "Squat": [
        ("Beginner", 0.80),
        ("Novice", 1.10),
        ("Intermediate", 1.45),
        ("Advanced", 1.85),
        ("Elite", 2.25),
    ],
    "Overhead Press": [
        ("Beginner", 0.40),
        ("Novice", 0.55),
        ("Intermediate", 0.75),
        ("Advanced", 0.95),
        ("Elite", 1.15),
    ],
}


def calculate_epley_e1rm(weight_kg: float, reps: int) -> float:
    """Calculate estimated 1RM using Epley formula: weight * (1 + reps / 30).
    
    Rounds to 2 decimal places.
    """
    if weight_kg <= 0 or reps <= 0:
        return 0.0
    return round(float(weight_kg) * (1.0 + float(reps) / 30.0), 2)


def get_session_top_sets(sets_df: pd.DataFrame) -> pd.DataFrame:
    """Extract top-set e1RM for each lift on each calendar date.
    
    Accepts dataframe with columns: ['date', 'exercise', 'weight_kg', 'reps'].
    Returns dataframe sorted by date with: ['date', 'exercise', 'top_e1rm', 'weight_kg', 'reps'].
    """
    if sets_df.empty:
        return pd.DataFrame(columns=["date", "exercise", "top_e1rm", "weight_kg", "reps"])

    df = sets_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["e1rm"] = [
        calculate_epley_e1rm(w, r) for w, r in zip(df["weight_kg"], df["reps"])
    ]

    # For each date + exercise, pick the set with maximum e1rm
    top_indices = df.groupby(["date", "exercise"])["e1rm"].idxmax()
    top_sets = df.loc[top_indices].copy()
    top_sets = top_sets.rename(columns={"e1rm": "top_e1rm"})
    top_sets = top_sets.sort_values(by=["date", "exercise"]).reset_index(drop=True)
    return top_sets[["date", "exercise", "top_e1rm", "weight_kg", "reps"]]


def build_forecasting_dataset(
    top_sets_df: pd.DataFrame, min_prior_sessions: int = 1
) -> pd.DataFrame:
    """Build leakage-free tabular dataset for training and evaluating forecasters.
    
    For each session, features are computed exclusively using data from strictly
    earlier dates (< session.date).
    
    Features:
    - days_since_start: days since first-ever session across all lifts
    - days_since_prev: days since previous session of this lift
    - prev_ratio: previous e1RM / best_so_far_prior
    - rollmean3_ratio: mean of up to 3 prior e1RMs / best_so_far_prior
    - sessions_14d: count of sessions of this lift in prior 14 days
    - lift: categorical lift name
    
    Targets:
    - target_ratio: actual top_e1rm / best_so_far_prior
    - target_e1rm: actual top_e1rm (for direct evaluation in kg)
    - best_so_far_prior: historical best before this session (for ratio scaling)
    """
    if top_sets_df.empty:
        return pd.DataFrame()

    df = top_sets_df.copy()
    df["dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("dt").reset_index(drop=True)

    first_ever_date = df["dt"].min()
    rows = []

    for idx, row in df.iterrows():
        curr_dt = row["dt"]
        curr_lift = row["exercise"]
        curr_e1rm = float(row["top_e1rm"])

        # Strictly prior sessions for this lift (temporal leakage prevention)
        prior_lift_sessions = df[
            (df["exercise"] == curr_lift) & (df["dt"] < curr_dt)
        ].sort_values("dt")

        n_priors = len(prior_lift_sessions)
        if n_priors < min_prior_sessions:
            continue

        prior_e1rms = prior_lift_sessions["top_e1rm"].values
        prior_dates = prior_lift_sessions["dt"].values

        best_so_far_prior = float(np.max(prior_e1rms))
        prev_e1rm = float(prior_e1rms[-1])
        prev_dt = pd.to_datetime(prior_dates[-1])

        days_since_start = (curr_dt - first_ever_date).days
        days_since_prev = (curr_dt - prev_dt).days

        # Rolling mean of last 3 prior sessions
        last_3 = prior_e1rms[-3:]
        rollmean3 = float(np.mean(last_3))

        # Sessions in prior 14 days [curr_dt - 14d, curr_dt)
        cutoff_14d = curr_dt - pd.Timedelta(days=14)
        sessions_14d = int(
            (prior_lift_sessions["dt"] >= cutoff_14d).sum()
        )

        prev_ratio = prev_e1rm / best_so_far_prior if best_so_far_prior > 0 else 1.0
        rollmean3_ratio = rollmean3 / best_so_far_prior if best_so_far_prior > 0 else 1.0
        target_ratio = curr_e1rm / best_so_far_prior if best_so_far_prior > 0 else 1.0

        rows.append(
            {
                "date": row["date"],
                "lift": curr_lift,
                "days_since_start": days_since_start,
                "days_since_prev": days_since_prev,
                "prev_ratio": prev_ratio,
                "rollmean3_ratio": rollmean3_ratio,
                "sessions_14d": sessions_14d,
                "prev_e1rm": prev_e1rm,
                "best_so_far_prior": best_so_far_prior,
                "target_ratio": target_ratio,
                "target_e1rm": curr_e1rm,
            }
        )

    return pd.DataFrame(rows)


@dataclass
class TierStatus:
    current_tier: str
    next_tier: str | None
    current_ratio: float
    threshold_ratio: float
    next_threshold_ratio: float | None
    current_e1rm: float
    next_threshold_kg: float | None
    projected_weeks_low: float | None
    projected_weeks_high: float | None
    is_projectable: bool
    status_message: str


def get_strength_tier(
    lift: str,
    recent_best_e1rm: float,
    bodyweight_kg: float,
    recent_history: Sequence[tuple[date, float]] | None = None,
) -> TierStatus:
    """Calculate user's strength tier and projected time to next tier."""
    if bodyweight_kg <= 0 or recent_best_e1rm <= 0:
        return TierStatus(
            current_tier="Untrained",
            next_tier="Beginner",
            current_ratio=0.0,
            threshold_ratio=0.0,
            next_threshold_ratio=None,
            current_e1rm=recent_best_e1rm,
            next_threshold_kg=None,
            projected_weeks_low=None,
            projected_weeks_high=None,
            is_projectable=False,
            status_message="Requires positive weight and bodyweight",
        )

    ratio = round(recent_best_e1rm / bodyweight_kg, 2)
    thresholds = TIER_THRESHOLDS.get(
        lift,
        [
            ("Beginner", 0.50),
            ("Novice", 0.75),
            ("Intermediate", 1.00),
            ("Advanced", 1.30),
            ("Elite", 1.60),
        ],
    )

    current_tier = "Untrained"
    threshold_ratio = 0.0
    next_tier = thresholds[0][0]
    next_threshold_ratio = thresholds[0][1]

    for i, (name, thresh) in enumerate(thresholds):
        if ratio >= thresh:
            current_tier = name
            threshold_ratio = thresh
            if i + 1 < len(thresholds):
                next_tier = thresholds[i + 1][0]
                next_threshold_ratio = thresholds[i + 1][1]
            else:
                next_tier = None
                next_threshold_ratio = None

    if next_threshold_ratio is None:
        return TierStatus(
            current_tier=current_tier,
            next_tier=None,
            current_ratio=ratio,
            threshold_ratio=threshold_ratio,
            next_threshold_ratio=None,
            current_e1rm=recent_best_e1rm,
            next_threshold_kg=None,
            projected_weeks_low=None,
            projected_weeks_high=None,
            is_projectable=False,
            status_message=f"At highest tier ({current_tier})!",
        )

    next_threshold_kg = round(next_threshold_ratio * bodyweight_kg, 1)
    kg_needed = next_threshold_kg - recent_best_e1rm

    # Time projection based on recent trend (last 8-12 weeks)
    projected_weeks_low = None
    projected_weeks_high = None
    is_projectable = False
    status_msg = "Trend not projectable right now"

    if recent_history and len(recent_history) >= 4:
        # Sort by date
        sorted_hist = sorted(recent_history, key=lambda x: x[0])
        first_date = sorted_hist[0][0]
        # x in weeks, y in kg
        xs = np.array([(pt[0] - first_date).days / 7.0 for pt in sorted_hist])
        ys = np.array([pt[1] for pt in sorted_hist])

        n = len(xs)
        if np.std(xs) > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            # Standard error of slope
            y_pred = slope * xs + intercept
            residuals = ys - y_pred
            s_err = np.sqrt(np.sum(residuals**2) / max(1, n - 2))
            se_slope = s_err / (np.sqrt(np.sum((xs - np.mean(xs)) ** 2)) + 1e-8)

            if slope > 0.05:  # Positive progression (> 50g/week)
                slope_low = max(0.01, slope - se_slope)
                slope_high = slope + se_slope
                projected_weeks_low = round(kg_needed / slope_high, 1)
                projected_weeks_high = round(kg_needed / slope_low, 1)
                is_projectable = True
                status_msg = f"Projected in {projected_weeks_low:.0f}–{projected_weeks_high:.0f} weeks at current pace"

    return TierStatus(
        current_tier=current_tier,
        next_tier=next_tier,
        current_ratio=ratio,
        threshold_ratio=threshold_ratio,
        next_threshold_ratio=next_threshold_ratio,
        current_e1rm=recent_best_e1rm,
        next_threshold_kg=next_threshold_kg,
        projected_weeks_low=projected_weeks_low,
        projected_weeks_high=projected_weeks_high,
        is_projectable=is_projectable,
        status_message=status_msg,
    )
