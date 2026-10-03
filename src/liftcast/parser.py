"""Free-text workout log parser using local Gemma via Ollama with JSON schema.

Processes informal, shorthand, typo-ridden, and Hinglish workout logs into structured sets.
Includes deterministic post-processing, alias resolution, sanity bounds, and low-confidence flags.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any
import ollama

from liftcast.db import resolve_exercise

logger = logging.getLogger(__name__)

OLLAMA_PARSER_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exercise": {"type": "string"},
                    "weight": {"type": ["number", "null"]},
                    "unit": {"type": "string", "enum": ["kg", "lb", "unknown"]},
                    "reps": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": ["string", "null"]},
                },
                "required": ["exercise", "weight", "unit", "reps", "note"],
            },
        }
    },
    "required": ["entries"],
}

SYSTEM_PROMPT = """You are a workout log parser. Convert free-text workout notes (including shorthand, typos, Hinglish like 'aaj bench 60 pe 8 reps', 'bench 60 8 8 7, last set died', 'bp 80kg 5x5') into structured JSON.
Rules:
1. Exercise name should capture the exercise described.
2. Weight: float or null if bodyweight.
3. Unit: 'kg', 'lb', or 'unknown' (default to 'unknown' if not specified).
4. Reps: a list of integers, one per set. '8 8 7' -> [8, 8, 7]. '5x5' or '5 reps 5 sets' -> [5, 5, 5, 5, 5].
5. Note: any subjective comment (e.g. 'last set died', 'felt heavy', 'RPE 9') or null.
"""


@dataclass
class ParsedSet:
    exercise_raw: str
    exercise: str | None
    weight: float | None
    unit: str
    weight_kg: float | None
    reps: list[int]
    set_count: int
    note: str | None = None


@dataclass
class ParseResult:
    raw_text: str
    entries: list[ParsedSet] = field(default_factory=list)
    needs_confirmation: bool = False
    issues: list[str] = field(default_factory=list)


def parse_log(
    text: str,
    model: str = "gemma3:1b",
    conn: sqlite3.Connection | None = None,
    client: Any = None,
    conversion_factor: float = 2.2,
) -> ParseResult:
    """Parse informal workout text into structured sets using Ollama or deterministic fallback."""
    raw_text = text.strip()
    if not raw_text:
        return ParseResult(raw_text=raw_text, needs_confirmation=True, issues=["Empty input"])

    ollama_client = client or ollama

    parsed_json = None
    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            format=OLLAMA_PARSER_SCHEMA,
            options={"num_ctx": 8192, "temperature": 0},
        )
        content = response["message"]["content"]
        parsed_json = json.loads(content)
    except Exception as e:
        logger.warning("Ollama query failed or offline (%s). Using fallback heuristic parser.", e)
        parsed_json = fallback_heuristic_parse(raw_text)

    return post_process_parse_result(
        raw_text=raw_text,
        data=parsed_json,
        conn=conn,
        conversion_factor=conversion_factor,
    )


def post_process_parse_result(
    raw_text: str,
    data: dict[str, Any] | None,
    conn: sqlite3.Connection | None = None,
    conversion_factor: float = 2.2,
) -> ParseResult:
    """Deterministic validation and normalization of LLM output (§6.3)."""
    if not data or "entries" not in data or not data["entries"]:
        return ParseResult(
            raw_text=raw_text,
            needs_confirmation=True,
            issues=["Could not detect any exercises or sets in log"],
        )

    parsed_entries: list[ParsedSet] = []
    issues: list[str] = []
    needs_confirmation = False

    for item in data["entries"]:
        raw_ex = str(item.get("exercise", "")).strip()
        weight = item.get("weight")
        unit = str(item.get("unit", "unknown")).lower()
        reps = item.get("reps") or []
        note = item.get("note")

        # 1. Resolve canonical exercise via database
        canonical_ex = None
        if conn and raw_ex:
            canonical_ex = resolve_exercise(conn, raw_ex)

        if not canonical_ex:
            # Check basic name mapping fallback
            canonical_ex = resolve_fallback_name(raw_ex)

        if not canonical_ex:
            needs_confirmation = True
            issues.append(f"Unrecognized exercise: '{raw_ex}' — please confirm canonical name.")
            canonical_ex = raw_ex.title()

        # 2. Normalize units & weight
        weight_kg = None
        if weight is not None:
            try:
                w_val = float(weight)
                if unit == "lb":
                    weight_kg = round(w_val / conversion_factor, 2)
                elif unit == "kg":
                    weight_kg = round(w_val, 2)
                else:
                    # Default unknown to kg, flag in issues (§6.3)
                    weight_kg = round(w_val, 2)
                    issues.append(f"Unit assumed as kg for '{canonical_ex}' ({w_val} kg).")
            except (ValueError, TypeError):
                needs_confirmation = True
                issues.append(f"Invalid weight format: '{weight}'")
        else:
            issues.append(f"No weight provided for '{canonical_ex}' (assumed bodyweight).")

        # 3. Clean reps list
        clean_reps: list[int] = []
        for r in reps:
            try:
                r_int = int(r)
                if r_int > 0:
                    clean_reps.append(r_int)
            except (ValueError, TypeError):
                continue

        if not clean_reps:
            needs_confirmation = True
            issues.append(f"No valid reps found for '{canonical_ex}'.")

        # 4. Sanity bounds (§6.3)
        # 1 <= reps <= 50; 1 <= weight_kg <= 400
        if weight_kg is not None and not (1.0 <= weight_kg <= 400.0):
            needs_confirmation = True
            issues.append(f"Weight {weight_kg} kg is outside typical range [1, 400] kg.")

        for r in clean_reps:
            if not (1 <= r <= 50):
                needs_confirmation = True
                issues.append(f"Reps count {r} is outside typical range [1, 50].")

        parsed_entries.append(
            ParsedSet(
                exercise_raw=raw_ex,
                exercise=canonical_ex,
                weight=weight,
                unit=unit,
                weight_kg=weight_kg,
                reps=clean_reps,
                set_count=len(clean_reps),
                note=note if note else None,
            )
        )

    return ParseResult(
        raw_text=raw_text,
        entries=parsed_entries,
        needs_confirmation=needs_confirmation,
        issues=issues,
    )


def resolve_fallback_name(name: str) -> str | None:
    """Heuristic fallback resolver when database is not connected."""
    n = name.lower().strip()
    if re.search(r"\b(bench|bp|bech)\b", n):
        return "Bench Press"
    if re.search(r"\b(deadlift|dl)\b", n):
        return "Deadlift"
    if re.search(r"\b(pulldown|lat)\b", n):
        return "Lat Pulldown"
    if re.search(r"\b(row|bor)\b", n):
        return "Bent Over Row"
    if re.search(r"\b(rdl|romanian)\b", n):
        return "Romanian Deadlift"
    if re.search(r"\b(squat)\b", n):
        return "Squat"
    if re.search(r"\b(ohp|overhead|military)\b", n):
        return "Overhead Press"
    return None


def fallback_heuristic_parse(text: str) -> dict[str, Any]:
    """Rule-based extractor used if Ollama service is unavailable."""
    entries = []
    # Split on 'and', ';', or newlines
    chunks = re.split(r"\band\b|;|\n", text, flags=re.IGNORECASE)

    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue

        # Extract note if comma present
        note = None
        if "," in c:
            parts = c.split(",", 1)
            c = parts[0].strip()
            note = parts[1].strip()

        # Extract unit
        unit = "unknown"
        if re.search(r"\b(lbs?|pounds?)\b", c, re.IGNORECASE):
            unit = "lb"
            c = re.sub(r"\b(lbs?|pounds?)\b", "", c, flags=re.IGNORECASE)
        elif re.search(r"\b(kgs?|kilos?)\b", c, re.IGNORECASE):
            unit = "kg"
            c = re.sub(r"\b(kgs?|kilos?)\b", "", c, flags=re.IGNORECASE)

        # Look for numbers
        numbers = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", c)]
        # Exercise words (non-digits)
        ex_words = re.sub(r"\b\d+(?:\.\d+)?\b|pe|reps?|set?|x", "", c, flags=re.IGNORECASE).strip()

        if numbers:
            weight = numbers[0]
            reps = [int(n) for n in numbers[1:]] if len(numbers) > 1 else [8]
        else:
            weight = None
            reps = [8]

        entries.append(
            {
                "exercise": ex_words if ex_words else "Unknown Lift",
                "weight": weight,
                "unit": unit,
                "reps": reps,
                "note": note,
            }
        )

    return {"entries": entries}
