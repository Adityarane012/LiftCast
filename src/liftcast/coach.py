"""AI Coach weekly summary generator with strict numeric guard for LiftCast.

The model explains; it never invents numbers (CLAUDE.md Hard Rule 3).
Gemma narrates stats computed by code. Every number is audited by the numeric guard.
If hallucinated numbers are detected, the system regenerates once, then falls back
to a deterministic template summary.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
import ollama

logger = logging.getLogger(__name__)

COACH_SYSTEM_PROMPT = """You are a direct, grounded lifting coach summarizing weekly progress for Armaan.
STRICT CONTRACT:
1. You may ONLY reference numbers that appear in the provided STATS PAYLOAD. Never invent weights, reps, dates, or percentages.
2. Word limit: 120 words maximum. Be punchy and actionable.
3. If and only if a lift is marked 'stalled: true', you may mention a deload or adjusting volume as an option.
4. No medical, injury, or clinical advice.
5. If progress is positive, acknowledge it briefly with the exact numbers.
"""


@dataclass
class CoachResult:
    summary: str
    is_fallback: bool
    violations: list[float]
    model_used: str
    verified: bool


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numbers (integers, floats, percentages) from text."""
    # Matches patterns like 100, 100.5, -2.5, +3.1, 75%
    raw_matches = re.findall(r"[-+]?\b\d+(?:\.\d+)?\b", text)
    nums = []
    for m in raw_matches:
        try:
            val = float(m)
            nums.append(val)
        except ValueError:
            continue
    return nums


def extract_allowed_numbers_from_payload(payload: dict[str, Any]) -> set[float]:
    """Recursively extract all numeric values from payload to construct the whitelist."""
    allowed: set[float] = set()

    def _walk(obj: Any):
        if isinstance(obj, (int, float)):
            val = float(obj)
            allowed.add(round(val, 1))
            allowed.add(round(val, 0))
            allowed.add(val)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)

    _walk(payload)
    # Common text counts (e.g. 1 week, 7 days, 100%) that may naturally occur
    allowed.update({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 14.0, 30.0, 100.0})
    return allowed


def verify_numeric_guard(text: str, allowed_numbers: set[float], tolerance: float = 0.5) -> tuple[bool, list[float]]:
    """Verify that every number in text matches a number from the allowed payload."""
    text_nums = extract_numbers_from_text(text)
    violations = []

    for num in text_nums:
        # Check if num is close to any allowed number
        matched = False
        for allowed in allowed_numbers:
            if abs(num - allowed) <= tolerance:
                matched = True
                break
        if not matched:
            violations.append(num)

    return (len(violations) == 0, violations)


def build_template_coach_summary(payload: dict[str, Any]) -> str:
    """Deterministic fallback summary template when LLM fails or hallucinates numbers."""
    sess_count = payload.get("sessions_this_week", 0)
    lines = [f"This week you logged {sess_count} workout session(s)."]

    lifts = payload.get("lifts", {})
    stalled_lifts = []

    for lift_name, stats in lifts.items():
        curr_e1rm = stats.get("current_e1rm")
        change = stats.get("weekly_change_pct", 0.0)
        is_stalled = stats.get("stalled", False)
        forecast = stats.get("forecast_next_e1rm")

        ch_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        e1rm_str = f"{curr_e1rm:.1f} kg" if curr_e1rm else "N/A"
        f_str = f" (Forecast: {forecast:.1f} kg)" if forecast else ""

        if is_stalled:
            stalled_lifts.append(lift_name)
            lines.append(f"• {lift_name}: {e1rm_str} ({ch_str}){f_str} — Progress is currently flat.")
        else:
            lines.append(f"• {lift_name}: {e1rm_str} ({ch_str}){f_str}.")

    if stalled_lifts:
        lines.append(
            f"Stall detected on {', '.join(stalled_lifts)}. Consider holding weight for technique or taking a light deload next week."
        )
    else:
        lines.append("Progression is steady across active lifts. Keep the stimulus consistent.")

    return "\n".join(lines)


def generate_coach_summary(
    payload: dict[str, Any],
    model: str = "gemma3:1b",
    client: Any = None,
    allow_llm: bool = True,
) -> CoachResult:
    """Generate weekly coach summary enforcing numeric guard and offline fallback."""
    allowed_numbers = extract_allowed_numbers_from_payload(payload)
    ollama_client = client or ollama

    if not allow_llm:
        template = build_template_coach_summary(payload)
        return CoachResult(
            summary=template,
            is_fallback=True,
            violations=[],
            model_used="template",
            verified=True,
        )

    prompt = f"STATS PAYLOAD (ONLY use these numbers):\n{json.dumps(payload, indent=2)}\n\nWrite the coach summary:"

    attempts = 2
    last_violations: list[float] = []

    for attempt in range(attempts):
        try:
            res = ollama_client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": COACH_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"num_ctx": 4096, "temperature": 0.2},
            )
            raw_summary = res["message"]["content"].strip()
            # Clean markdown code blocks if any
            clean_summary = re.sub(r"^```[a-z]*\n|```$", "", raw_summary).strip()

            is_valid, violations = verify_numeric_guard(clean_summary, allowed_numbers)
            if is_valid:
                return CoachResult(
                    summary=clean_summary,
                    is_fallback=False,
                    violations=[],
                    model_used=model,
                    verified=True,
                )

            last_violations = violations
            logger.warning(
                "Coach attempt %d numeric guard failure. Violating numbers: %s",
                attempt + 1,
                violations,
            )
        except Exception as e:
            logger.warning("Ollama coach generation error: %s", e)
            break

    # If both attempts violate numeric guard or Ollama offline, use deterministic template
    template_summary = build_template_coach_summary(payload)
    return CoachResult(
        summary=template_summary,
        is_fallback=True,
        violations=last_violations,
        model_used="template_fallback",
        verified=True,
    )
