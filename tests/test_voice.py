"""Tests for the ElevenLabs Voice Coach and Audio Briefing module."""

from unittest.mock import MagicMock, patch
import pytest

from liftcast.voice import (
    VoiceResult,
    build_audio_briefing_script,
    synthesize_voice_elevenlabs,
)
from liftcast.coach import extract_allowed_numbers_from_payload, verify_numeric_guard


@pytest.fixture
def sample_stats_payload():
    return {
        "sessions_this_week": 3,
        "lifts": {
            "Bench Press": {
                "current_e1rm": 75.0,
                "weekly_change_pct": 2.5,
                "stalled": False,
                "forecast_next_e1rm": 76.2,
            }
        },
    }


@pytest.fixture
def stalled_stats_payload():
    return {
        "sessions_this_week": 4,
        "lifts": {
            "Lat Pulldown": {
                "current_e1rm": 62.0,
                "weekly_change_pct": -0.5,
                "stalled": True,
                "forecast_next_e1rm": 62.0,
            }
        },
    }


def test_build_audio_briefing_script_normal(sample_stats_payload):
    script = build_audio_briefing_script(sample_stats_payload, user_name="Armaan")
    assert "Armaan" in script
    assert "Bench Press" in script
    assert "75" in script
    # Verify script satisfies numeric guard
    allowed = extract_allowed_numbers_from_payload(sample_stats_payload)
    is_valid, violations = verify_numeric_guard(script, allowed, tolerance=1.0)
    assert is_valid, f"Violations found in voice script: {violations}"


def test_build_audio_briefing_script_stalled(stalled_stats_payload):
    script = build_audio_briefing_script(stalled_stats_payload, user_name="Armaan")
    assert "Lat Pulldown" in script
    assert "62" in script
    assert "deload" in script.lower() or "flattened" in script.lower()


def test_build_audio_briefing_script_empty():
    empty_payload = {"sessions_this_week": 2, "lifts": {}}
    script = build_audio_briefing_script(empty_payload, user_name="Armaan")
    assert "Armaan" in script
    assert "2 session" in script


def test_synthesize_voice_no_api_key():
    result = synthesize_voice_elevenlabs("Great workout Armaan!", api_key=None)
    assert isinstance(result, VoiceResult)
    assert result.provider == "browser_speech"
    assert result.audio_bytes is None
    assert result.success is True
    assert "offline" in result.error_message.lower() or "browser" in result.error_message.lower()


def test_synthesize_voice_elevenlabs_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"ID3_FAKE_MP3_STREAM_DATA"

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = synthesize_voice_elevenlabs(
            script="Great bench press session!",
            api_key="test_xi_key_12345",
            voice_id="test_voice_id",
        )
        assert result.provider == "elevenlabs"
        assert result.success is True
        assert result.audio_bytes == b"ID3_FAKE_MP3_STREAM_DATA"
        assert mock_post.called
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["headers"]["xi-api-key"] == "test_xi_key_12345"


def test_synthesize_voice_elevenlabs_api_error_fallback():
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized: Invalid API Key"

    with patch("requests.post", return_value=mock_response):
        result = synthesize_voice_elevenlabs(
            script="Great bench press session!",
            api_key="bad_key",
        )
        assert result.provider == "browser_speech"
        assert result.success is False
        assert result.audio_bytes is None
        assert "401" in result.error_message


def test_synthesize_voice_elevenlabs_network_timeout():
    with patch("requests.post", side_effect=Exception("Connection timed out")):
        result = synthesize_voice_elevenlabs(
            script="Great bench press session!",
            api_key="test_key",
        )
        assert result.provider == "browser_speech"
        assert result.success is False
        assert result.audio_bytes is None
        assert "timed out" in result.error_message
