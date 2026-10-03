"""Liftoff CSV Importer for LiftCast.

Imports historical workout data while strictly observing privacy rules:
- Drops 'Workout Name' and 'Notes' (personal health details) before processing.
- Filters out non-lifting rows (bodyweight with 0 weight, cardio with distance/seconds).
- Converts weights from lbs to kg using the defined conversion factor (default: 2.2).
- Groups workouts by calendar day (YYYY-MM-DD).
- Enforces idempotency so re-runs do not duplicate data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

from liftcast.db import get_or_create_session, insert_set, add_alias


DROP_COLUMNS = ["Workout Name", "Notes", "Duration", "RPE"]


def clean_liftoff_dataframe(
    df: pd.DataFrame, conversion_factor: float = 2.2
) -> pd.DataFrame:
    """Filter and sanitize Liftoff export dataframe without leaking private columns."""
    # 1. Immediately drop sensitive/unneeded columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Filter numeric constraints: weight > 0, reps > 0
    df = df[pd.to_numeric(df["Weight"], errors="coerce") > 0]
    df = df[pd.to_numeric(df["Reps"], errors="coerce") > 0]

    # 3. Filter out cardio rows
    if "Distance" in df.columns:
        df = df[
            (df["Distance"].isna())
            | (pd.to_numeric(df["Distance"], errors="coerce") == 0)
        ]
    if "Seconds" in df.columns:
        df = df[
            (df["Seconds"].isna())
            | (pd.to_numeric(df["Seconds"], errors="coerce") == 0)
        ]

    # 4. Clean exercise name
    df["exercise"] = df["Exercise Name"].astype(str).str.strip()

    # 5. Extract ISO calendar date (YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # 6. Convert units (lb -> kg)
    df["weight_kg"] = (
        pd.to_numeric(df["Weight"]) / float(conversion_factor)
    ).round(2)
    df["reps"] = pd.to_numeric(df["Reps"]).astype(int)

    # Preserve set order if present
    if "Set Order" in df.columns:
        df["set_order"] = (
            pd.to_numeric(df["Set Order"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df = df.sort_values(by=["date", "set_order"])
    else:
        df = df.sort_values(by=["date"])

    return df[["date", "exercise", "weight_kg", "reps"]]


def import_liftoff_csv(
    csv_path: str | Path,
    conn: sqlite3.Connection,
    conversion_factor: float = 2.2,
) -> dict[str, int]:
    """Import Liftoff workout history into SQLite database idempotently.
    
    Returns a dictionary of import statistics.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Liftoff CSV not found at: {path}")

    # Read raw CSV
    raw_df = pd.read_csv(path)
    total_raw_rows = len(raw_df)

    # Clean and filter dataframe
    clean_df = clean_liftoff_dataframe(raw_df, conversion_factor=conversion_factor)
    valid_sets_count = len(clean_df)

    sessions_imported = 0
    sets_imported = 0
    sessions_skipped = 0

    # Group by calendar date
    grouped = clean_df.groupby("date", sort=True)
    with conn:
        for session_date, group in grouped:
            cur = conn.execute(
                "SELECT id FROM sessions WHERE date = ? AND source = 'liftoff'",
                (session_date,),
            )
            existing = cur.fetchone()

            if existing:
                sess_id = int(existing["id"])
                count_cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM sets WHERE session_id = ?",
                    (sess_id,),
                )
                cnt = count_cur.fetchone()["cnt"]
                if cnt > 0:
                    sessions_skipped += 1
                    continue
            else:
                sess_id = get_or_create_session(conn, session_date, source="liftoff")
                sessions_imported += 1

            for _, row in group.iterrows():
                exercise_name = str(row["exercise"])
                insert_set(
                    conn,
                    session_id=sess_id,
                    exercise=exercise_name,
                    weight_kg=float(row["weight_kg"]),
                    reps=int(row["reps"]),
                    note=None,
                    raw_text=None,
                )
                add_alias(conn, exercise_name.lower(), exercise_name)
                sets_imported += 1

    return {
        "total_raw_rows": total_raw_rows,
        "valid_sets_count": valid_sets_count,
        "sessions_imported": sessions_imported,
        "sets_imported": sets_imported,
        "sessions_skipped": sessions_skipped,
    }
