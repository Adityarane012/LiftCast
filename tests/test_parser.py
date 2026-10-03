import json
import sqlite3
from unittest.mock import MagicMock
import pytest

from liftcast.db import get_connection, init_db
from liftcast.parser import parse_log, post_process_parse_result, fallback_heuristic_parse


@pytest.fixture
def mem_db():
    conn = get_connection(":memory:")
    init_db(conn)
    yield conn
    conn.close()


def test_post_process_unit_conversion_and_alias(mem_db):
    data = {
        "entries": [
            {
                "exercise": "bp",
                "weight": 135.0,
                "unit": "lb",
                "reps": [5, 5, 5],
                "note": "solid",
            }
        ]
    }
    result = post_process_parse_result("bp 135 lb 5 5 5", data, conn=mem_db, conversion_factor=2.2)

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.exercise == "Bench Press"
    assert entry.weight_kg == round(135.0 / 2.2, 2)
    assert entry.reps == [5, 5, 5]
    assert entry.set_count == 3
    assert entry.note == "solid"
    assert result.needs_confirmation is False


def test_post_process_out_of_bounds_flags_confirmation(mem_db):
    # Weight 500 kg (over 400kg bound)
    data = {
        "entries": [
            {
                "exercise": "bench",
                "weight": 500.0,
                "unit": "kg",
                "reps": [8],
                "note": None,
            }
        ]
    }
    res = post_process_parse_result("bench 500 8", data, conn=mem_db)
    assert res.needs_confirmation is True
    assert any("outside typical range" in issue for issue in res.issues)


def test_post_process_unknown_exercise_flags_confirmation(mem_db):
    data = {
        "entries": [
            {
                "exercise": "super bizarre extraterrestrial lift",
                "weight": 50.0,
                "unit": "kg",
                "reps": [10],
                "note": None,
            }
        ]
    }
    res = post_process_parse_result("super bizarre extraterrestrial lift 50 10", data, conn=mem_db)
    assert res.needs_confirmation is True
    assert any("Unrecognized exercise" in issue for issue in res.issues)


def test_parser_with_mocked_ollama(mem_db):
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "entries": [
                        {
                            "exercise": "Bench Press",
                            "weight": 60.0,
                            "unit": "kg",
                            "reps": [8, 8, 7],
                            "note": "last set died",
                        }
                    ]
                }
            )
        }
    }

    res = parse_log(
        "bench 60 8 8 7, last set died",
        model="gemma3:1b",
        conn=mem_db,
        client=mock_client,
    )
    assert len(res.entries) == 1
    assert res.entries[0].exercise == "Bench Press"
    assert res.entries[0].weight_kg == 60.0
    assert res.entries[0].reps == [8, 8, 7]
    assert res.entries[0].note == "last set died"
    assert res.needs_confirmation is False


def test_fallback_heuristic_parser():
    res = fallback_heuristic_parse("bench 60 8 8 7, last set died")
    assert "entries" in res
    entry = res["entries"][0]
    assert entry["weight"] == 60.0
    assert entry["reps"] == [8, 8, 7]
    assert entry["note"] == "last set died"
