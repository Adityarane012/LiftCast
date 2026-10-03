import numpy as np
import pandas as pd
import pytest

from liftcast.forecast import (
    LastValueBaseline,
    LinearTrendBaseline,
    TabPFNForecaster,
    run_rolling_origin_eval,
)


@pytest.fixture
def synthetic_progression():
    # 10 weekly sessions with steady progression: 60, 62, 64, 66, 68, 70, 72, 74, 76, 78
    records = []
    dates = [f"2026-01-{i*3+1:02d}" for i in range(10)]
    for i, d in enumerate(dates):
        records.append(
            {
                "date": d,
                "exercise": "Bench Press",
                "top_e1rm": float(60 + i * 2),
                "weight_kg": float(60 + i * 2),
                "reps": 1,
            }
        )
    return pd.DataFrame(records)


def test_last_value_baseline(synthetic_progression):
    baseline = LastValueBaseline()
    # Predict as of date 2026-01-16 (session index 5, value 70)
    # Most recent session before that is session index 4 (value 68)
    forecast = baseline.predict_next(
        synthetic_progression, lift="Bench Press", as_of="2026-01-16"
    )
    assert forecast.point_kg == 68.0
    assert forecast.method == "last_value"


def test_linear_trend_baseline(synthetic_progression):
    baseline = LinearTrendBaseline(n=5)
    # Predict as of session index 5 (date 2026-01-16)
    forecast = baseline.predict_next(
        synthetic_progression, lift="Bench Press", as_of="2026-01-16"
    )
    # Trend is exactly +2 per session, so it should predict ~70.0
    assert abs(forecast.point_kg - 70.0) < 1.0


def test_tabpfn_forecaster(synthetic_progression):
    forecaster = TabPFNForecaster(device="cpu", random_seed=42)
    forecast = forecaster.predict_next(
        synthetic_progression, lift="Bench Press", as_of="2026-01-22"
    )
    assert forecast.point_kg > 0
    assert forecast.low_kg is not None
    assert forecast.high_kg is not None
    assert forecast.low_kg <= forecast.point_kg <= forecast.high_kg


def test_rolling_origin_eval_runner():
    # Create 15 sessions for Lat Pulldown and Bench Press
    records = []
    for lift in ["Lat Pulldown", "Bench Press"]:
        for i in range(15):
            records.append(
                {
                    "date": f"2026-0{1 + i//10}-{(i%10)*2 + 10:02d}",
                    "exercise": lift,
                    "top_e1rm": float(50 + i * 1.5),
                    "weight_kg": float(50 + i * 1.5),
                    "reps": 1,
                }
            )
    df = pd.DataFrame(records)

    summary_df, detailed_df = run_rolling_origin_eval(
        df, lifts=["Lat Pulldown", "Bench Press"], n_test_sessions=3
    )

    assert not summary_df.empty
    assert not detailed_df.empty
    assert len(detailed_df) == 6  # 2 lifts * 3 test points
    assert "Winner" in summary_df.columns
    assert "Last value MAE (kg)" in summary_df.columns
    assert "Trend-5 MAE (kg)" in summary_df.columns
    assert "TabPFN MAE (kg)" in summary_df.columns
