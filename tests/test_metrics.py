import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from liftcast.metrics import (
    calculate_epley_e1rm,
    get_session_top_sets,
    build_forecasting_dataset,
    get_strength_tier,
    TIER_THRESHOLDS,
)


def test_epley_formula():
    assert calculate_epley_e1rm(100.0, 10) == 133.33
    assert calculate_epley_e1rm(60.0, 8) == round(60.0 * (1 + 8 / 30), 2)
    assert calculate_epley_e1rm(0, 10) == 0.0
    assert calculate_epley_e1rm(100, 0) == 0.0


def test_get_session_top_sets_filters_warmups():
    data = [
        {"date": "2026-03-01", "exercise": "Bench Press", "weight_kg": 40.0, "reps": 10},  # Warmup
        {"date": "2026-03-01", "exercise": "Bench Press", "weight_kg": 60.0, "reps": 8},   # Working 1
        {"date": "2026-03-01", "exercise": "Bench Press", "weight_kg": 60.0, "reps": 6},   # Working 2
        {"date": "2026-03-01", "exercise": "Deadlift", "weight_kg": 100.0, "reps": 5},     # Other lift
    ]
    df = pd.DataFrame(data)
    top_sets = get_session_top_sets(df)

    assert len(top_sets) == 2
    bench_row = top_sets[top_sets["exercise"] == "Bench Press"].iloc[0]
    # 60 * (1 + 8/30) = 76.0
    assert bench_row["top_e1rm"] == 76.0
    assert bench_row["reps"] == 8


def test_features_no_temporal_leakage():
    # Construct sequence of sessions on distinct dates
    dates = [
        "2026-01-01",
        "2026-01-08",
        "2026-01-15",
        "2026-01-22",
        "2026-01-29",
    ]
    records = []
    for d, w in zip(dates, [60.0, 62.5, 65.0, 67.5, 70.0]):
        records.append({"date": d, "exercise": "Bench Press", "weight_kg": w, "reps": 5})

    df = pd.DataFrame(records)
    top_sets = get_session_top_sets(df)
    features_df = build_forecasting_dataset(top_sets, min_prior_sessions=1)

    assert len(features_df) == 4  # First session has 0 prior sessions

    # Check that for each row, prev_e1rm and best_so_far_prior only reflect strictly earlier sessions
    row1 = features_df.iloc[0]  # Date: 2026-01-08
    assert row1["date"] == "2026-01-08"
    assert row1["prev_e1rm"] == round(60.0 * (1 + 5 / 30), 2)
    assert row1["best_so_far_prior"] == round(60.0 * (1 + 5 / 30), 2)
    assert row1["days_since_prev"] == 7

    row3 = features_df.iloc[2]  # Date: 2026-01-22
    assert row3["date"] == "2026-01-22"
    assert row3["prev_e1rm"] == round(65.0 * (1 + 5 / 30), 2)
    assert row3["best_so_far_prior"] == round(65.0 * (1 + 5 / 30), 2)


def test_strength_tiers_and_projection():
    # Bench Press for 75kg person
    # Intermediate threshold = 1.05 * 75 = 78.75 kg
    status = get_strength_tier("Bench Press", recent_best_e1rm=70.0, bodyweight_kg=75.0)
    assert status.current_tier in ["Beginner", "Novice"]
    assert status.next_tier == "Intermediate"
    assert status.next_threshold_kg == 78.8

    # Test projection with positive slope
    base_date = date(2026, 1, 1)
    history = [
        (base_date + timedelta(weeks=0), 65.0),
        (base_date + timedelta(weeks=2), 66.5),
        (base_date + timedelta(weeks=4), 68.0),
        (base_date + timedelta(weeks=6), 70.0),
    ]
    status_proj = get_strength_tier(
        "Bench Press",
        recent_best_e1rm=70.0,
        bodyweight_kg=75.0,
        recent_history=history,
    )
    assert status_proj.is_projectable is True
    assert status_proj.projected_weeks_low is not None
    assert status_proj.projected_weeks_high is not None
    assert status_proj.projected_weeks_low <= status_proj.projected_weeks_high
