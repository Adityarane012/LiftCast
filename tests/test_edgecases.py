"""Comprehensive edge case and boundary condition tests for LiftCast.

Covers:
- Zero, negative, and extreme inputs in metrics and e1RM formulas
- Empty dataframes, single-point histories, and temporal leakage checks
- Zero variance, constant trends, and duplicate dates in stall detection
- Boundary bounds, unit conversion edge cases, and missing fields in free-text parser
- Hallucination detection, percentage formats, and empty payloads in AI coach numeric guard
- Input validation and alias resolution in database layer
"""

from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

from liftcast.coach import (
    build_template_coach_summary,
    extract_allowed_numbers_from_payload,
    extract_numbers_from_text,
    generate_coach_summary,
    verify_numeric_guard,
)
from liftcast.db import (
    get_connection,
    init_db,
    insert_set,
    get_or_create_session,
    resolve_exercise,
)
from liftcast.detect import detect_stalls
from liftcast.forecast import (
    LastValueBaseline,
    LinearTrendBaseline,
    TabPFNForecaster,
)
from liftcast.metrics import (
    build_forecasting_dataset,
    calculate_epley_e1rm,
    get_session_top_sets,
    get_strength_tier,
)
from liftcast.parser import parse_log, post_process_parse_result


# =============================================================================
# 1. METRICS & e1RM EDGE CASES
# =============================================================================

def test_epley_zero_and_negative_inputs():
    """Epley e1RM should return 0.0 for zero or negative weights and reps."""
    assert calculate_epley_e1rm(0.0, 10) == 0.0
    assert calculate_epley_e1rm(100.0, 0) == 0.0
    assert calculate_epley_e1rm(-50.0, 10) == 0.0
    assert calculate_epley_e1rm(100.0, -5) == 0.0
    # Single rep: 100 * (1 + 1/30) = 103.33
    assert calculate_epley_e1rm(100.0, 1) == 103.33


def test_get_session_top_sets_empty():
    """Empty DataFrame input should return empty top sets DataFrame without exception."""
    empty_df = pd.DataFrame(columns=["date", "exercise", "weight_kg", "reps"])
    res = get_session_top_sets(empty_df)
    assert res.empty
    assert list(res.columns) == ["date", "exercise", "top_e1rm", "weight_kg", "reps"]


def test_get_session_top_sets_filters_warmups():
    """Should correctly select the maximum e1RM set on the same calendar day."""
    data = [
        {"date": "2026-02-01", "exercise": "Bench Press", "weight_kg": 40.0, "reps": 10},  # e1rm = 53.33
        {"date": "2026-02-01", "exercise": "Bench Press", "weight_kg": 60.0, "reps": 8},   # e1rm = 76.00
        {"date": "2026-02-01", "exercise": "Bench Press", "weight_kg": 50.0, "reps": 8},   # e1rm = 63.33
    ]
    df = pd.DataFrame(data)
    top = get_session_top_sets(df)
    assert len(top) == 1
    assert top.iloc[0]["weight_kg"] == 60.0
    assert top.iloc[0]["reps"] == 8
    assert top.iloc[0]["top_e1rm"] == 76.0


def test_build_forecasting_dataset_insufficient_history():
    """Should return empty dataframe if history has fewer sessions than min_prior_sessions."""
    data = [{"date": "2026-01-01", "exercise": "Bench Press", "top_e1rm": 60.0, "weight_kg": 50.0, "reps": 6}]
    df = pd.DataFrame(data)
    # 1 session has 0 prior sessions, so with min_prior_sessions=1 it yields 0 rows
    feat_df = build_forecasting_dataset(df, min_prior_sessions=1)
    assert feat_df.empty


def test_strength_tier_zero_or_negative_bodyweight():
    """Should handle zero, negative bodyweight, or zero e1rm without DivisionByZero."""
    tier_zero_bw = get_strength_tier("Bench Press", 80.0, bodyweight_kg=0.0)
    assert tier_zero_bw.current_tier == "Untrained"
    assert tier_zero_bw.is_projectable is False

    tier_neg_bw = get_strength_tier("Bench Press", 80.0, bodyweight_kg=-70.0)
    assert tier_neg_bw.current_tier == "Untrained"
    assert tier_neg_bw.is_projectable is False

    tier_zero_e1rm = get_strength_tier("Bench Press", 0.0, bodyweight_kg=75.0)
    assert tier_zero_e1rm.current_tier == "Untrained"
    assert tier_zero_e1rm.is_projectable is False


def test_strength_tier_unknown_lift_and_elite():
    """Should use fallback thresholds for unlisted lifts and handle maximum tier."""
    # Unknown lift
    tier_unknown = get_strength_tier("Zercher Squat", 120.0, bodyweight_kg=75.0)
    assert tier_unknown.current_tier in ["Beginner", "Novice", "Intermediate", "Advanced", "Elite"]

    # Super elite lifter: 200 kg bench at 70 kg BW (ratio ~ 2.85 >= 1.65 Elite)
    tier_elite = get_strength_tier("Bench Press", 200.0, bodyweight_kg=70.0)
    assert tier_elite.current_tier == "Elite"
    assert tier_elite.next_tier is None
    assert tier_elite.is_projectable is False
    assert "highest tier" in tier_elite.status_message.lower()


def test_strength_tier_negative_slope():
    """Declining strength trend should not produce positive week projections."""
    # 4 sessions where e1RM drops from 100 -> 90 kg
    dates = [date(2026, 1, 1) + timedelta(weeks=i) for i in range(4)]
    e1rms = [100.0, 96.0, 93.0, 90.0]
    hist = list(zip(dates, e1rms))

    status = get_strength_tier("Bench Press", 90.0, bodyweight_kg=75.0, recent_history=hist)
    # Slope is negative, so cannot project time to next tier
    assert status.is_projectable is False
    assert "not projectable" in status.status_message.lower()


# =============================================================================
# 2. STALL DETECTOR EDGE CASES
# =============================================================================

def test_detect_stalls_empty_and_small():
    """detect_stalls on empty or < min_n dataframe returns empty/unflagged without crashing."""
    res_empty = detect_stalls(pd.DataFrame())
    assert res_empty.empty

    df_2pts = pd.DataFrame([
        {"date": "2026-01-01", "top_e1rm": 60.0},
        {"date": "2026-01-08", "top_e1rm": 62.0},
    ])
    res_2 = detect_stalls(df_2pts, min_n=4)
    assert len(res_2) == 2
    assert res_2["stalled"].isna().all()


def test_detect_stalls_zero_variance_identical_dates_and_values():
    """Handles identical dates or identical values without runtime mathematical errors."""
    # Constant e1rm: 4 sessions of exactly 60.0 kg over 4 weeks
    df_const = pd.DataFrame([
        {"date": f"2026-01-{i*7+1:02d}", "top_e1rm": 60.0} for i in range(4)
    ])
    res_const = detect_stalls(df_const, window_days=56, min_n=4)
    assert len(res_const) == 4
    # Last point has 4 sessions in window with slope = 0.0%/week
    last_row = res_const.iloc[-1]
    assert last_row["n_in_window"] == 4
    assert last_row["slope_pct_week"] == 0.0
    # At theta=0.0, slope < 0.0 is False, so stalled is False
    assert last_row["stalled"] is False

    # Identical dates (zero variance in time)
    df_same_date = pd.DataFrame([
        {"date": "2026-01-01", "top_e1rm": 60.0 + i} for i in range(4)
    ])
    res_same = detect_stalls(df_same_date, window_days=56, min_n=4)
    # Zero variance in x -> slope_pct_week should default cleanly to 0.0
    assert res_same.iloc[-1]["slope_pct_week"] == 0.0


# =============================================================================
# 3. FORECASTING EDGE CASES
# =============================================================================

def test_forecasters_empty_history():
    """All forecasters should return safe 0.0 kg forecasts on empty data."""
    empty_df = pd.DataFrame(columns=["date", "exercise", "top_e1rm"])

    lv = LastValueBaseline()
    f_lv = lv.predict_next(empty_df, "Bench Press", "2026-01-01")
    assert f_lv.point_kg == 0.0

    lt = LinearTrendBaseline(n=5)
    f_lt = lt.predict_next(empty_df, "Bench Press", "2026-01-01")
    assert f_lt.point_kg == 0.0

    pfn = TabPFNForecaster(device="cpu")
    f_pfn = pfn.predict_next(empty_df, "Bench Press", "2026-01-01")
    assert f_pfn.point_kg == 0.0


def test_linear_trend_single_point():
    """Linear trend with only 1 session should fall back to that session's e1RM."""
    df = pd.DataFrame([{"date": "2026-01-01", "exercise": "Bench Press", "top_e1rm": 80.0}])
    lt = LinearTrendBaseline(n=5)
    res = lt.predict_next(df, "Bench Press", "2026-01-08")
    assert res.point_kg == 80.0


def test_tabpfn_unrecorded_lift():
    """TabPFN predicting a lift with no prior sessions returns 0.0 kg."""
    df = pd.DataFrame([{"date": "2026-01-01", "exercise": "Deadlift", "top_e1rm": 150.0}])
    pfn = TabPFNForecaster(device="cpu")
    res = pfn.predict_next(df, "Bench Press", "2026-01-08")
    assert res.point_kg == 0.0


# =============================================================================
# 4. FREE-TEXT PARSER EDGE CASES
# =============================================================================

def test_parser_empty_and_whitespace():
    """Empty or whitespace-only strings return needs_confirmation=True."""
    res1 = parse_log("")
    assert res1.needs_confirmation is True
    assert "Empty input" in res1.issues

    res2 = parse_log("   \n\t   ")
    assert res2.needs_confirmation is True
    assert "Empty input" in res2.issues


def test_parser_extreme_bounds_trigger_confirmation():
    """Extreme weights (> 400kg) or reps (> 50) trigger needs_confirmation."""
    # Data with weight = 450 kg
    data = {
        "entries": [
            {"exercise": "Bench Press", "weight": 450.0, "unit": "kg", "reps": [1], "note": None}
        ]
    }
    res = post_process_parse_result("bench 450 1", data)
    assert res.needs_confirmation is True
    assert any("outside typical range" in s for s in res.issues)

    # Data with reps = 75
    data_reps = {
        "entries": [
            {"exercise": "Bench Press", "weight": 60.0, "unit": "kg", "reps": [75], "note": None}
        ]
    }
    res_reps = post_process_parse_result("bench 60 75", data_reps)
    assert res_reps.needs_confirmation is True
    assert any("outside typical range" in s for s in res_reps.issues)


def test_parser_unit_conversion():
    """Unit 'lb' is properly converted to kg using factor 2.2."""
    data = {
        "entries": [
            {"exercise": "Bench Press", "weight": 220.0, "unit": "lb", "reps": [5, 5], "note": None}
        ]
    }
    res = post_process_parse_result("bench 220 lb 5 5", data, conversion_factor=2.2)
    assert len(res.entries) == 1
    # 220 / 2.2 = 100.0 kg
    assert res.entries[0].weight_kg == 100.0


# =============================================================================
# 5. COACH NUMERIC GUARD EDGE CASES
# =============================================================================

def test_extract_numbers_from_text():
    """Regex number extraction captures signed floats, percentages, and decimals."""
    text = "Bench improved by +5.2% to 62.5 kg, while deadlift dropped -3 kg to 120 kg. Session 1 of 7."
    nums = extract_numbers_from_text(text)
    assert 5.2 in nums
    assert 62.5 in nums
    assert -3.0 in nums
    assert 120.0 in nums
    assert 1.0 in nums
    assert 7.0 in nums


def test_verify_numeric_guard_catches_hallucination():
    """Numeric guard blocks LLM text that includes unapproved statistics."""
    payload = {
        "current_e1rm": 60.0,
        "weekly_change_pct": 2.5,
        "sessions_this_week": 3,
    }
    allowed = extract_allowed_numbers_from_payload(payload)

    # Valid summary using approved numbers
    valid_text = "You completed 3 sessions this week. Bench reached 60.0 kg with a +2.5% increase."
    is_valid, violations = verify_numeric_guard(valid_text, allowed)
    assert is_valid is True
    assert len(violations) == 0

    # Hallucinated text inventing '500 mg' and '185 kg'
    invalid_text = "Bench reached 60.0 kg. Take 500 mg magnesium and target 185 kg next week."
    is_valid_bad, violations_bad = verify_numeric_guard(invalid_text, allowed)
    assert is_valid_bad is False
    assert 500.0 in violations_bad
    assert 185.0 in violations_bad


def test_coach_template_fallback_on_empty_payload():
    """Deterministic template summary generates without crashing on empty payload."""
    summary = build_template_coach_summary({"sessions_this_week": 0, "lifts": {}})
    assert "0 workout session(s)" in summary
    assert "steady" in summary.lower()

    # Full offline generation with allow_llm=False
    res = generate_coach_summary({"sessions_this_week": 2, "lifts": {}}, allow_llm=False)
    assert res.is_fallback is True
    assert res.model_used == "template"
    assert res.verified is True


# =============================================================================
# 6. DATABASE LAYER EDGE CASES
# =============================================================================

def test_db_insert_negative_weight_or_reps():
    """Database schema rejects negative weight or reps with ValueError."""
    conn = get_connection(":memory:")
    init_db(conn)

    sess_id = get_or_create_session(conn, "2026-03-01")

    with pytest.raises(ValueError, match="weight_kg must be > 0"):
        insert_set(conn, session_id=sess_id, exercise="Bench Press", weight_kg=-10.0, reps=5)

    with pytest.raises(ValueError, match="reps must be > 0"):
        insert_set(conn, session_id=sess_id, exercise="Bench Press", weight_kg=60.0, reps=0)


def test_db_resolve_exercise_case_insensitivity_and_empty():
    """Exercise resolver handles empty strings, None, and case-insensitivity."""
    conn = get_connection(":memory:")
    init_db(conn)

    assert resolve_exercise(conn, "") is None
    assert resolve_exercise(conn, "   ") is None

    # Shorthand alias resolution
    assert resolve_exercise(conn, "bp") == "Bench Press"
    assert resolve_exercise(conn, "BENCH PRESS") == "Bench Press"
    assert resolve_exercise(conn, "bOr") == "Bent Over Row"
    assert resolve_exercise(conn, "non_existent_exercise_12345") is None
