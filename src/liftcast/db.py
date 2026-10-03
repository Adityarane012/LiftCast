"""Database schema, connection management, and CRUD helpers for LiftCast.

SQLite is used locally; data stays strictly on the user's laptop.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT NOT NULL UNIQUE,          -- ISO date (YYYY-MM-DD calendar day)
  bodyweight_kg REAL,                 -- nullable; entered manually
  source TEXT NOT NULL CHECK (source IN ('liftoff', 'manual'))
);

CREATE TABLE IF NOT EXISTS sets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  exercise TEXT NOT NULL,             -- canonical name
  weight_kg REAL NOT NULL CHECK (weight_kg > 0),
  reps INTEGER NOT NULL CHECK (reps > 0),
  note TEXT,
  raw_text TEXT                       -- original typed line (manual entries)
);

CREATE TABLE IF NOT EXISTS exercise_aliases (
  alias TEXT PRIMARY KEY,             -- lowercase trimmed
  canonical TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sets_session_id ON sets(session_id);
CREATE INDEX IF NOT EXISTS idx_sets_exercise ON sets(exercise);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);
"""

DEFAULT_ALIASES: dict[str, str] = {
    # Bench Press
    "bench": "Bench Press",
    "bp": "Bench Press",
    "flat bench": "Bench Press",
    "bench press": "Bench Press",
    "flat barbell bench press": "Bench Press",
    "bech": "Bench Press",
    "bech press": "Bench Press",

    # Deadlift
    "deadlift": "Deadlift",
    "dl": "Deadlift",
    "conventional deadlift": "Deadlift",
    "deadlifts": "Deadlift",

    # Lat Pulldown
    "lat pulldown": "Lat Pulldown",
    "pulldown": "Lat Pulldown",
    "lat pull": "Lat Pulldown",
    "lat pulldowns": "Lat Pulldown",
    "wide grip lat pulldown": "Lat Pulldown",

    # Bent Over Row
    "bent over row": "Bent Over Row",
    "bor": "Bent Over Row",
    "barbell row": "Bent Over Row",
    "row": "Bent Over Row",
    "bb row": "Bent Over Row",
    "bent over barbell row": "Bent Over Row",

    # Romanian Deadlift
    "rdl": "Romanian Deadlift",
    "romanian deadlift": "Romanian Deadlift",
    "db rdl": "Romanian Deadlift",

    # Squat & OHP
    "squat": "Squat",
    "back squat": "Squat",
    "barbell squat": "Squat",
    "ohp": "Overhead Press",
    "overhead press": "Overhead Press",
    "military press": "Overhead Press",
    "shoulder press": "Overhead Press",
}


def get_connection(db_path: str | Path = "data/liftcast.db") -> sqlite3.Connection:
    """Connect to SQLite database and enforce foreign keys."""
    if isinstance(db_path, str) and db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize database tables and seed default aliases."""
    with conn:
        conn.executescript(SCHEMA_SQL)
    seed_default_aliases(conn)


def seed_default_aliases(conn: sqlite3.Connection) -> None:
    """Seed canonical aliases without overwriting user customizations."""
    with conn:
        for alias, canonical in DEFAULT_ALIASES.items():
            conn.execute(
                "INSERT OR IGNORE INTO exercise_aliases (alias, canonical) VALUES (?, ?)",
                (alias.strip().lower(), canonical.strip()),
            )


def add_alias(conn: sqlite3.Connection, alias: str, canonical: str) -> None:
    """Add or update an exercise alias."""
    alias_clean = alias.strip().lower()
    canonical_clean = canonical.strip()
    with conn:
        conn.execute(
            "INSERT INTO exercise_aliases (alias, canonical) VALUES (?, ?) "
            "ON CONFLICT(alias) DO UPDATE SET canonical = excluded.canonical",
            (alias_clean, canonical_clean),
        )


def resolve_exercise(conn: sqlite3.Connection, exercise_name: str) -> str | None:
    """Resolve an exercise name or shorthand to its canonical name.
    
    Checks exact matches against existing canonical exercises in sets, then alias table.
    """
    if not exercise_name or not exercise_name.strip():
        return None
    raw = exercise_name.strip()
    norm = raw.lower()

    # 1. Alias table match
    row = conn.execute(
        "SELECT canonical FROM exercise_aliases WHERE alias = ?",
        (norm,),
    ).fetchone()
    if row:
        return str(row["canonical"])

    # 2. Case-insensitive match on canonical names already in aliases
    row = conn.execute(
        "SELECT DISTINCT canonical FROM exercise_aliases WHERE LOWER(canonical) = ?",
        (norm,),
    ).fetchone()
    if row:
        return str(row["canonical"])

    # 3. Case-insensitive match on existing sets
    row = conn.execute(
        "SELECT DISTINCT exercise FROM sets WHERE LOWER(exercise) = ?",
        (norm,),
    ).fetchone()
    if row:
        return str(row["exercise"])

    return None


def get_or_create_session(
    conn: sqlite3.Connection,
    date_str: str,
    source: str = "manual",
    bodyweight_kg: float | None = None,
) -> int:
    """Get or create a session by calendar date (YYYY-MM-DD)."""
    date_clean = date_str.strip()[:10]
    cursor = conn.execute("SELECT id, bodyweight_kg FROM sessions WHERE date = ?", (date_clean,))
    row = cursor.fetchone()
    if row:
        sess_id = int(row["id"])
        if bodyweight_kg is not None and row["bodyweight_kg"] is None:
            with conn:
                conn.execute(
                    "UPDATE sessions SET bodyweight_kg = ? WHERE id = ?",
                    (bodyweight_kg, sess_id),
                )
        return sess_id

    with conn:
        cur = conn.execute(
            "INSERT INTO sessions (date, bodyweight_kg, source) VALUES (?, ?, ?)",
            (date_clean, bodyweight_kg, source),
        )
        return int(cur.lastrowid)


def insert_set(
    conn: sqlite3.Connection,
    session_id: int,
    exercise: str,
    weight_kg: float,
    reps: int,
    note: str | None = None,
    raw_text: str | None = None,
) -> int:
    """Insert a single workout set."""
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if reps <= 0:
        raise ValueError(f"reps must be > 0, got {reps}")

    with conn:
        cur = conn.execute(
            "INSERT INTO sets (session_id, exercise, weight_kg, reps, note, raw_text) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, exercise.strip(), float(weight_kg), int(reps), note, raw_text),
        )
        return int(cur.lastrowid)


def get_all_sessions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all sessions sorted by date."""
    cursor = conn.execute(
        "SELECT id, date, bodyweight_kg, source FROM sessions ORDER BY date ASC"
    )
    return [dict(row) for row in cursor.fetchall()]


def get_history_df(
    conn: sqlite3.Connection, lift: str | None = None
) -> pd.DataFrame:
    """Return workout history as a Pandas DataFrame with date, exercise, weight_kg, reps."""
    query = """
    SELECT 
        s.id AS session_id,
        s.date,
        s.bodyweight_kg,
        s.source,
        st.id AS set_id,
        st.exercise,
        st.weight_kg,
        st.reps,
        st.note,
        st.raw_text
    FROM sets st
    JOIN sessions s ON st.session_id = s.id
    """
    params: list[Any] = []
    if lift:
        query += " WHERE LOWER(st.exercise) = LOWER(?)"
        params.append(lift)
    query += " ORDER BY s.date ASC, st.id ASC"

    return pd.read_sql_query(query, conn, params=params)
