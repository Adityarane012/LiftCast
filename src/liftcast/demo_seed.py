"""Demo data seeder for LiftCast.

Populates the local SQLite database with realistic 6-month workout histories
across core lifts (Bench Press, Deadlift, Lat Pulldown, Row, RDL, Squat, OHP)
featuring genuine progressive overload, planted plateaus, and deloads.
Allows reviewers and fresh clones to test all UI and forecasting features immediately.
100% synthetic/anonymized — zero personal or private data.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta
from typing import Any

from liftcast.db import (
    get_connection,
    get_or_create_session,
    init_db,
    insert_set,
)
from liftcast.synthetic import generate_synthetic_lift_history


DEMO_LIFT_CONFIGS = [
    {
        "lift": "Bench Press",
        "base_e1rm": 55.0,
        "gain": 28.0,
        "tau": 20.0,
        "sessions_per_week": 2,
    },
    {
        "lift": "Deadlift",
        "base_e1rm": 95.0,
        "gain": 45.0,
        "tau": 22.0,
        "sessions_per_week": 1,
    },
    {
        "lift": "Lat Pulldown",
        "base_e1rm": 50.0,
        "gain": 22.0,
        "tau": 18.0,
        "sessions_per_week": 2,
    },
    {
        "lift": "Bent Over Row",
        "base_e1rm": 45.0,
        "gain": 25.0,
        "tau": 20.0,
        "sessions_per_week": 2,
    },
    {
        "lift": "Romanian Deadlift",
        "base_e1rm": 65.0,
        "gain": 35.0,
        "tau": 24.0,
        "sessions_per_week": 1,
    },
    {
        "lift": "Squat",
        "base_e1rm": 75.0,
        "gain": 40.0,
        "tau": 22.0,
        "sessions_per_week": 2,
    },
    {
        "lift": "Overhead Press",
        "base_e1rm": 35.0,
        "gain": 18.0,
        "tau": 24.0,
        "sessions_per_week": 1,
    },
]


def seed_demo_database(
    conn: sqlite3.Connection,
    total_weeks: int = 26,
    clear_existing: bool = True,
    bodyweight_kg: float = 75.0,
) -> dict[str, Any]:
    """Seed SQLite database with rich multi-lift workout history."""
    init_db(conn)

    if clear_existing:
        with conn:
            conn.execute("DELETE FROM sets")
            conn.execute("DELETE FROM sessions")

    # Start ~26 weeks ago up to today
    today = date.today()
    start_date = (today - timedelta(weeks=total_weeks)).strftime("%Y-%m-%d")

    sessions_created = 0
    sets_created = 0
    lift_stats: dict[str, int] = {}

    for cfg in DEMO_LIFT_CONFIGS:
        lift = cfg["lift"]
        df_lift = generate_synthetic_lift_history(
            start_date=start_date,
            total_weeks=total_weeks,
            sessions_per_week=cfg["sessions_per_week"],
            base_e1rm=cfg["base_e1rm"],
            gain=cfg["gain"],
            tau=cfg["tau"],
            lift_name=lift,
            random_seed=hash(lift) % 10000,
        )

        lift_sets_count = 0
        for _, row in df_lift.iterrows():
            sess_date = row["date"]
            top_e1rm = float(row["top_e1rm"])

            # Create or get session
            sess_id = get_or_create_session(
                conn,
                date_str=sess_date,
                source="manual",
                bodyweight_kg=bodyweight_kg,
            )

            # 1. Warm-up set (~60% of top e1RM, 10 reps)
            warmup_weight = round(round((top_e1rm * 0.60) / (1.0 + 10.0 / 30.0) / 2.5) * 2.5, 1)
            warmup_weight = max(15.0, warmup_weight)
            insert_set(
                conn,
                session_id=sess_id,
                exercise=lift,
                weight_kg=warmup_weight,
                reps=10,
                note="Warm-up set",
                raw_text=f"warmup {lift} {warmup_weight} 10",
            )
            sets_created += 1

            # 2. Main Top Set (8 reps)
            top_reps = 8
            work_weight = round(round(top_e1rm / (1.0 + float(top_reps) / 30.0) / 2.5) * 2.5, 1)
            work_weight = max(20.0, work_weight)
            insert_set(
                conn,
                session_id=sess_id,
                exercise=lift,
                weight_kg=work_weight,
                reps=top_reps,
                note="Working top set",
                raw_text=f"{lift.lower()} {work_weight} {top_reps}",
            )
            sets_created += 1
            lift_sets_count += 2

            # 3. Back-off volume set (7 reps at ~90% of work weight)
            backoff_weight = round(round((work_weight * 0.90) / 2.5) * 2.5, 1)
            if backoff_weight > 15.0:
                insert_set(
                    conn,
                    session_id=sess_id,
                    exercise=lift,
                    weight_kg=backoff_weight,
                    reps=7,
                    note="Back-off set",
                    raw_text=f"{lift.lower()} {backoff_weight} 7",
                )
                sets_created += 1
                lift_sets_count += 1

        lift_stats[lift] = lift_sets_count

    # Count distinct sessions
    cursor = conn.execute("SELECT COUNT(*) FROM sessions")
    sessions_created = int(cursor.fetchone()[0])

    return {
        "sessions_created": sessions_created,
        "sets_created": sets_created,
        "lifts_seeded": list(lift_stats.keys()),
        "breakdown": lift_stats,
    }
