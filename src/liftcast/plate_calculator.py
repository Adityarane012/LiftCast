"""Barbell plate loader and gym target calculator for LiftCast.

Computes exact plate combinations for Olympic barbells, snaps weights to realistic
gym plate increments, and inverts Epley e1RM forecasts into actionable working sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


STANDARD_OLYMPIC_PLATES = (20.0, 10.0, 5.0, 2.5, 1.25)


@dataclass
class PlateLoadingResult:
    target_weight_kg: float
    bar_weight_kg: float
    weight_per_side_target: float
    weight_per_side_loaded: float
    total_loaded_kg: float
    plates_per_side: list[float]
    plate_counts: dict[float, int]
    remainder_kg: float
    is_exact: bool
    plates_visual: list[dict[str, Any]] = field(default_factory=list)


def snap_to_plate_increment(weight_kg: float, increment: float = 2.5) -> float:
    """Snap a calculated weight to the nearest gym plate increment (default 2.5 kg)."""
    if weight_kg <= 0:
        return 0.0
    return round(round(weight_kg / increment) * increment, 2)


def calculate_working_weight_from_e1rm(
    e1rm: float,
    target_reps: int = 8,
    increment: float = 2.5,
) -> float:
    """Invert Epley formula to compute recommended working weight from e1RM:
    
    weight = e1rm / (1 + reps / 30)
    Snaps to standard gym plate increments (e.g., 2.5 kg).
    """
    if e1rm <= 0 or target_reps <= 0:
        return 0.0
    raw_weight = e1rm / (1.0 + float(target_reps) / 30.0)
    return snap_to_plate_increment(raw_weight, increment=increment)


def calculate_plates(
    target_weight_kg: float,
    bar_weight_kg: float = 20.0,
    available_plates: Sequence[float] = STANDARD_OLYMPIC_PLATES,
) -> PlateLoadingResult:
    """Calculate exact plate combinations per sleeve on a barbell.
    
    Greedy plate allocation from heaviest to lightest plate denomination.
    """
    if target_weight_kg <= bar_weight_kg:
        return PlateLoadingResult(
            target_weight_kg=round(target_weight_kg, 2),
            bar_weight_kg=round(bar_weight_kg, 2),
            weight_per_side_target=0.0,
            weight_per_side_loaded=0.0,
            total_loaded_kg=round(bar_weight_kg, 2),
            plates_per_side=[],
            plate_counts={},
            remainder_kg=0.0 if target_weight_kg == bar_weight_kg else round(target_weight_kg - bar_weight_kg, 2),
            is_exact=(target_weight_kg == bar_weight_kg),
            plates_visual=[],
        )

    needed_per_side = (target_weight_kg - bar_weight_kg) / 2.0
    sorted_plates = sorted(available_plates, reverse=True)

    remaining = needed_per_side
    plates_used: list[float] = []
    counts: dict[float, int] = {}

    for plate in sorted_plates:
        while remaining >= plate - 1e-4:
            plates_used.append(plate)
            counts[plate] = counts.get(plate, 0) + 1
            remaining = round(remaining - plate, 4)

    loaded_per_side = round(sum(plates_used), 2)
    total_loaded = round(bar_weight_kg + 2.0 * loaded_per_side, 2)
    remainder = round(target_weight_kg - total_loaded, 2)
    is_exact = abs(remainder) < 1e-3

    # Olympic plate colors & visual specifications
    plate_colors = {
        25.0: {"color": "#dc2626", "height_px": 140, "width_px": 28, "label": "25"},
        20.0: {"color": "#0284c7", "height_px": 130, "width_px": 26, "label": "20"},
        15.0: {"color": "#f59e0b", "height_px": 110, "width_px": 22, "label": "15"},
        10.0: {"color": "#10b981", "height_px": 95, "width_px": 18, "label": "10"},
        5.0: {"color": "#f1f5f9", "height_px": 75, "width_px": 16, "label": "5", "text_color": "#0f172a"},
        2.5: {"color": "#f43f5e", "height_px": 60, "width_px": 14, "label": "2.5"},
        1.25: {"color": "#fbbf24", "height_px": 45, "width_px": 12, "label": "1.25", "text_color": "#0f172a"},
        0.5: {"color": "#94a3b8", "height_px": 35, "width_px": 10, "label": "0.5"},
    }

    visual_items = []
    for p in plates_used:
        meta = plate_colors.get(
            p, {"color": "#64748b", "height_px": 60, "width_px": 15, "label": f"{p}"}
        )
        visual_items.append({"weight": p, **meta})

    return PlateLoadingResult(
        target_weight_kg=round(target_weight_kg, 2),
        bar_weight_kg=round(bar_weight_kg, 2),
        weight_per_side_target=round(needed_per_side, 2),
        weight_per_side_loaded=loaded_per_side,
        total_loaded_kg=total_loaded,
        plates_per_side=plates_used,
        plate_counts=counts,
        remainder_kg=remainder,
        is_exact=is_exact,
        plates_visual=visual_items,
    )
