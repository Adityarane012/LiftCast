import sqlite3
import pytest
from liftcast.db import (
    get_connection,
    init_db,
    get_or_create_session,
    insert_set,
    resolve_exercise,
    add_alias,
    get_all_sessions,
    get_history_df,
)


@pytest.fixture
def mem_db():
    conn = get_connection(":memory:")
    init_db(conn)
    yield conn
    conn.close()


def test_init_db_creates_tables(mem_db):
    tables = [
        row[0]
        for row in mem_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "sessions" in tables
    assert "sets" in tables
    assert "exercise_aliases" in tables


def test_get_or_create_session_idempotent(mem_db):
    s1 = get_or_create_session(mem_db, "2026-10-01", source="manual", bodyweight_kg=75.0)
    s2 = get_or_create_session(mem_db, "2026-10-01", source="manual")
    assert s1 == s2
    sessions = get_all_sessions(mem_db)
    assert len(sessions) == 1
    assert sessions[0]["bodyweight_kg"] == 75.0


def test_insert_set_validation(mem_db):
    s_id = get_or_create_session(mem_db, "2026-10-01")
    set_id = insert_set(
        mem_db,
        session_id=s_id,
        exercise="Bench Press",
        weight_kg=60.0,
        reps=8,
        note="felt easy",
        raw_text="bench 60 8",
    )
    assert set_id > 0

    with pytest.raises(ValueError):
        insert_set(mem_db, session_id=s_id, exercise="Bench Press", weight_kg=0, reps=8)

    with pytest.raises(ValueError):
        insert_set(mem_db, session_id=s_id, exercise="Bench Press", weight_kg=60, reps=0)


def test_resolve_exercise_and_aliases(mem_db):
    assert resolve_exercise(mem_db, "bench") == "Bench Press"
    assert resolve_exercise(mem_db, "BP") == "Bench Press"
    assert resolve_exercise(mem_db, "dl") == "Deadlift"
    assert resolve_exercise(mem_db, "unknown exercise") is None

    add_alias(mem_db, "incline", "Incline Dumbbell Bench Press")
    assert resolve_exercise(mem_db, "incline") == "Incline Dumbbell Bench Press"


def test_get_history_df(mem_db):
    s_id = get_or_create_session(mem_db, "2026-10-01")
    insert_set(mem_db, s_id, "Bench Press", 60.0, 8)
    insert_set(mem_db, s_id, "Deadlift", 100.0, 5)

    df_all = get_history_df(mem_db)
    assert len(df_all) == 2

    df_bench = get_history_df(mem_db, lift="Bench Press")
    assert len(df_bench) == 1
    assert df_bench.iloc[0]["weight_kg"] == 60.0
