import sqlite3
import pandas as pd
import pytest
from pathlib import Path

from liftcast.db import get_connection, init_db, get_history_df, get_all_sessions
from liftcast.importer import clean_liftoff_dataframe, import_liftoff_csv, DROP_COLUMNS


@pytest.fixture
def mem_db():
    conn = get_connection(":memory:")
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "sample_liftoff.csv"
    data = [
        # Date, Duration, Workout Name, Exercise Name, Set Order, Weight, Reps, Distance, Seconds, RPE, Notes
        ["2025-03-01 09:00:00", 3600, "Morning Push", "Bench Press", 1, 242.0, 8, 0, 0, "", "Target 8-10"],
        ["2025-03-01 09:30:00", 3600, "Morning Push", "Bench Press", 2, 242.0, 7, 0, 0, "", "Hard set"],
        # Cardio row (should be filtered out)
        ["2025-03-01 10:00:00", 1200, "Cardio", "Treadmill", 1, 0, 0, 2000, 1200, "", ""],
        # Bodyweight row with 0 weight (should be filtered out)
        ["2025-03-01 10:30:00", 300, "Core", "Plank", 1, 0, 1, 0, 60, "", ""],
        # Second session on same calendar date (should be grouped into 2025-03-01)
        ["2025-03-01 18:00:00", 1800, "Evening", "Lat Pulldown", 1, 154.0, 10, 0, 0, "", ""],
        # Another day
        ["2025-03-03 10:00:00", 3600, "Leg Day", "Deadlift", 1, 330.0, 5, 0, 0, "", ""],
    ]
    columns = [
        "Date", "Duration", "Workout Name", "Exercise Name", "Set Order",
        "Weight", "Reps", "Distance", "Seconds", "RPE", "Notes"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_file, index=False)
    return csv_file


def test_clean_liftoff_dataframe_filters_and_privacy():
    data = [
        ["2025-03-01 09:00:00", 3600, "Morning Push", "Bench Press", 1, 242.0, 8, 0, 0, "", "Private Notes"],
        ["2025-03-01 10:00:00", 1200, "Cardio", "Treadmill", 1, 0, 0, 2000, 1200, "", ""],
    ]
    cols = [
        "Date", "Duration", "Workout Name", "Exercise Name", "Set Order",
        "Weight", "Reps", "Distance", "Seconds", "RPE", "Notes"
    ]
    df = pd.DataFrame(data, columns=cols)
    clean = clean_liftoff_dataframe(df, conversion_factor=2.2)

    # Privacy check: private columns dropped
    for col in DROP_COLUMNS:
        assert col not in clean.columns

    # Cardio filtered out
    assert len(clean) == 1

    # Unit conversion: 242.0 / 2.2 = 110.0 kg
    row = clean.iloc[0]
    assert row["exercise"] == "Bench Press"
    assert row["weight_kg"] == 110.0
    assert row["reps"] == 8
    assert row["date"] == "2025-03-01"


def test_import_liftoff_csv_idempotency_and_grouping(mem_db, sample_csv):
    # First import
    res1 = import_liftoff_csv(sample_csv, mem_db, conversion_factor=2.2)
    assert res1["sessions_imported"] == 2  # 2025-03-01 and 2025-03-03
    assert res1["sets_imported"] == 4     # 2 bench, 1 lat pulldown, 1 deadlift

    sessions = get_all_sessions(mem_db)
    assert len(sessions) == 2
    assert [s["date"] for s in sessions] == ["2025-03-01", "2025-03-03"]

    history = get_history_df(mem_db)
    assert len(history) == 4

    # Second import (idempotency check: should skip existing sessions)
    res2 = import_liftoff_csv(sample_csv, mem_db, conversion_factor=2.2)
    assert res2["sessions_imported"] == 0
    assert res2["sets_imported"] == 0
    assert res2["sessions_skipped"] == 2

    # Verification: set count remains 4
    history_after = get_history_df(mem_db)
    assert len(history_after) == 4
