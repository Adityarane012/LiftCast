"""Synthetic workout data generator for LiftCast.

Used strictly for unit-testing the stall detector and edge cases.
Always labeled as synthetic. Never mixed with real validation data without clear labeling.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence
import numpy as np
import pandas as pd


def generate_synthetic_lift_history(
    start_date: str = "2025-01-01",
    total_weeks: int = 52,
    sessions_per_week: int = 2,
    base_e1rm: float = 60.0,
    gain: float = 35.0,
    tau: float = 20.0,
    noise_sigma_pct: float = 0.04,
    random_seed: int = 42,
    lift_name: str = "Bench Press",
) -> pd.DataFrame:
    """Generate synthetic progression series with planted events.
    
    Formula: e1rm(t) = base + gain * log(1 + t / tau) + Gaussian noise
    Planted events:
    - Plateau: weeks 20 to 28 (8 weeks flat) -> is_plateau = True
    - Deload: week 35 (-10% for 1 week) -> is_plateau = False (deload != stall)
    - Bad day: week 42 session 1 (-15% single session) -> is_plateau = False
    - 3-week gap: weeks 45 to 48 (no sessions logged)
    """
    rng = np.random.default_rng(random_seed)
    base_dt = date.fromisoformat(start_date)

    records = []
    plateau_start_week = 20
    plateau_end_week = 28
    plateau_frozen_e1rm = None

    deload_week = 35
    bad_day_week = 42
    gap_start_week = 45
    gap_end_week = 48

    for week in range(total_weeks):
        # 3-week gap: skip logging
        if gap_start_week <= week < gap_end_week:
            continue

        for s_idx in range(sessions_per_week):
            session_day_offset = week * 7 + (s_idx * 3)
            sess_date = base_dt + timedelta(days=session_day_offset)

            t_val = float(week)
            # Base log progression
            progression = base_e1rm + gain * math_log(1.0 + t_val / tau)

            is_plateau = False
            is_deload = False
            is_bad_day = False

            # 1. Planted plateau: weeks 20-28
            if plateau_start_week <= week <= plateau_end_week:
                if plateau_frozen_e1rm is None:
                    plateau_frozen_e1rm = progression
                clean_e1rm = plateau_frozen_e1rm
                is_plateau = True
            else:
                if plateau_frozen_e1rm is not None and week > plateau_end_week:
                    # Resume progression smoothly from post-plateau
                    clean_e1rm = base_e1rm + gain * math_log(1.0 + (t_val - (plateau_end_week - plateau_start_week)) / tau)
                else:
                    clean_e1rm = progression

            # 2. Planted deload: week 35 (-10%)
            if week == deload_week:
                clean_e1rm *= 0.90
                is_deload = True

            # 3. Planted single bad day: week 42, first session (-15%)
            if week == bad_day_week and s_idx == 0:
                clean_e1rm *= 0.85
                is_bad_day = True

            # Add Gaussian noise (~4% sigma)
            noise_factor = 1.0 + rng.normal(0, noise_sigma_pct)
            actual_e1rm = max(10.0, round(clean_e1rm * noise_factor, 1))

            records.append(
                {
                    "date": sess_date.strftime("%Y-%m-%d"),
                    "exercise": lift_name,
                    "top_e1rm": actual_e1rm,
                    "is_plateau": is_plateau,
                    "event": (
                        "plateau"
                        if is_plateau
                        else (
                            "deload"
                            if is_deload
                            else ("bad_day" if is_bad_day else "normal")
                        )
                    ),
                    "is_synthetic": True,
                }
            )

    return pd.DataFrame(records)


def math_log(x: float) -> float:
    return float(np.log(max(1e-6, x)))
