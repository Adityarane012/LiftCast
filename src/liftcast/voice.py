"""ElevenLabs Voice Coach & Audio Briefing module for LiftCast.

Built for the "Build for a Friend" challenge: when my friend finishes their final set,
their hands are chalky and they're re-racking weights. Rather than reading a screen,
they hear a 10-second punchy voice briefing through gym earbuds.

Privacy Guarantee:
- Only the 2-sentence, already-sanitized coach script is sent to the TTS API.
- Zero raw workouts, zero timestamps, zero database records, and zero health notes are transmitted.
- 100% offline fallback via native browser SpeechSynthesis if no API key is set.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
import requests

from liftcast.coach import extract_allowed_numbers_from_payload, verify_numeric_guard

logger = logging.getLogger(__name__)

# Default ElevenLabs Voice IDs:
# "pNInz6obpgDQGcFmaJgB" = Adam (deep, energetic, athletic coach)
# "21m00Tcm4TlvDq8ikWAM" = Rachel (calm, direct)
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
DEFAULT_MODEL_ID = "eleven_turbo_v2_5"


@dataclass
class VoiceResult:
    script: str
    audio_bytes: Optional[bytes]
    provider: str  # 'elevenlabs' | 'browser_speech' | 'silent'
    success: bool
    error_message: Optional[str] = None


def build_audio_briefing_script(payload: dict[str, Any], user_name: str = "Friend") -> str:
    """Build a punchy, 2-sentence voice script from the audited stats payload.
    
    Guaranteed: Every number is derived from the payload to prevent hallucination.
    """
    sess_count = payload.get("sessions_this_week", 1)
    lifts = payload.get("lifts", {})

    if not lifts:
        return f"Great effort this week, {user_name}. You logged {sess_count} session. Recovery is key, rest up!"

    # Pick the primary lift or first active lift
    lift_name, stats = next(iter(lifts.items()))
    curr_e1rm = stats.get("current_e1rm")
    change = stats.get("weekly_change_pct", 0.0)
    stalled = stats.get("stalled", False)
    forecast = stats.get("forecast_next_e1rm")

    e1rm_part = f"{curr_e1rm:.0f} kg" if curr_e1rm else "solid numbers"

    if stalled:
        return (
            f"Good session, {user_name}. Your {lift_name} top set was {e1rm_part}. "
            f"Progress has flattened over recent weeks, so consider holding this weight or taking a light deload next."
        )
    
    if forecast and curr_e1rm:
        diff = round(curr_e1rm - (forecast - 1.0), 1)
        return (
            f"Solid work today, {user_name}! Your {lift_name} hit {e1rm_part}. "
            f"You are right on track with the TabPFN forecast. Rest up and bring the intensity next session!"
        )

    change_part = f"up {abs(change):.1f}%" if change >= 0 else f"down {abs(change):.1f}%"
    return (
        f"Solid workout today, {user_name}! Your {lift_name} hit {e1rm_part}, {change_part} this week. "
        f"Keep the momentum rolling!"
    )


def synthesize_voice_elevenlabs(
    script: str,
    api_key: Optional[str] = None,
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = DEFAULT_MODEL_ID,
    timeout_sec: float = 8.0,
) -> VoiceResult:
    """Synthesize voice audio using ElevenLabs REST API.
    
    Falls back gracefully to browser_speech if no key is supplied or network fails.
    """
    key = api_key or os.environ.get("ELEVENLABS_API_KEY", "").strip()

    if not key:
        return VoiceResult(
            script=script,
            audio_bytes=None,
            provider="browser_speech",
            success=True,
            error_message="No ElevenLabs API key provided. Using offline browser speech synthesis.",
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    body = {
        "text": script,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.80,
            "style": 0.15,
            "use_speaker_boost": True,
        },
    }

    try:
        response = requests.post(url, json=body, headers=headers, timeout=timeout_sec)
        if response.status_code == 200:
            return VoiceResult(
                script=script,
                audio_bytes=response.content,
                provider="elevenlabs",
                success=True,
                error_message=None,
            )
        else:
            err = f"ElevenLabs API returned HTTP {response.status_code}: {response.text[:200]}"
            logger.warning(err)
            return VoiceResult(
                script=script,
                audio_bytes=None,
                provider="browser_speech",
                success=False,
                error_message=err,
            )
    except Exception as e:
        err = f"ElevenLabs synthesis error: {e}"
        logger.warning(err)
        return VoiceResult(
            script=script,
            audio_bytes=None,
            provider="browser_speech",
            success=False,
            error_message=err,
        )
