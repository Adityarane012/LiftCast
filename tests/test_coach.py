from unittest.mock import MagicMock
import pytest

from liftcast.coach import (
    extract_numbers_from_text,
    extract_allowed_numbers_from_payload,
    verify_numeric_guard,
    build_template_coach_summary,
    generate_coach_summary,
)


@pytest.fixture
def sample_payload():
    return {
        "sessions_this_week": 3,
        "total_volume_kg": 2400.0,
        "lifts": {
            "Bench Press": {
                "current_e1rm": 75.0,
                "weekly_change_pct": 2.5,
                "stalled": False,
                "forecast_next_e1rm": 76.5,
            },
            "Lat Pulldown": {
                "current_e1rm": 65.0,
                "weekly_change_pct": 0.0,
                "stalled": True,
                "forecast_next_e1rm": 65.0,
            },
        },
    }


def test_extract_numbers_from_text():
    text = "Bench reached 75.0 kg (+2.5%), while deadlift stalled at 120 kg."
    nums = extract_numbers_from_text(text)
    assert 75.0 in nums
    assert 2.5 in nums
    assert 120.0 in nums


def test_numeric_guard_catches_hallucination(sample_payload):
    allowed = extract_allowed_numbers_from_payload(sample_payload)

    # Valid summary using only payload numbers
    valid_text = "You logged 3 sessions. Bench rose +2.5% to 75 kg with forecast 76.5 kg."
    is_valid, violations = verify_numeric_guard(valid_text, allowed)
    assert is_valid is True
    assert len(violations) == 0

    # Hallucinated number 95.0 kg (not in payload)
    invalid_text = "You logged 3 sessions and hit a huge 95.0 kg bench press!"
    is_valid, violations = verify_numeric_guard(invalid_text, allowed)
    assert is_valid is False
    assert 95.0 in violations


def test_template_coach_summary(sample_payload):
    summary = build_template_coach_summary(sample_payload)
    assert "3 workout session(s)" in summary
    assert "Bench Press: 75.0 kg (+2.5%)" in summary
    assert "Lat Pulldown: 65.0 kg (0.0%)" in summary
    assert "Stall detected on Lat Pulldown" in summary


def test_generate_coach_summary_fallback_on_violation(sample_payload):
    mock_client = MagicMock()
    # Model hallucinates 999 kg
    mock_client.chat.return_value = {
        "message": {"content": "Great work lifting 999 kg this week!"}
    }

    result = generate_coach_summary(
        sample_payload, model="gemma3:1b", client=mock_client
    )
    # Numeric guard should reject 999 and fall back to template summary
    assert result.is_fallback is True
    assert "3 workout session(s)" in result.summary
    assert result.verified is True
