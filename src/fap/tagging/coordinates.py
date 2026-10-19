"""Coordinate engine — the ONLY place screen fractions become football coordinates.

The interactive canvas reports a click as a **fraction** of the pitch (or goal)
interior: ``fx`` in ``[0, 1]`` left→right, ``fy`` in ``[0, 1]`` top→bottom. Because
the input is a fraction, the result is independent of pixel size, DPI, figure size
or Streamlit layout width — the same physical location always yields the same
canonical coordinate.

Canonical spaces (both ``0..100``, matching the repo's canonical event space):

    pitch:  x 0 (left goal) .. 100 (right goal),  y 0 (bottom) .. 100 (top)
    goal:   goal_x 0 (left post) .. 100 (right post),  goal_y 0 (ground) .. 100 (crossbar)

``y`` is flipped from screen space (screen ``fy=0`` is the TOP) so that canonical
``y``/``goal_y`` increase upward, matching how the pitch and goal renderers plot.
All conversions are pure and reversible.
"""
from __future__ import annotations

CANON_MIN = 0.0
CANON_MAX = 100.0


def clamp(value: float, lo: float = CANON_MIN, hi: float = CANON_MAX) -> float:
    return max(lo, min(hi, float(value)))


def in_range(value: float, lo: float = CANON_MIN, hi: float = CANON_MAX) -> bool:
    return lo <= float(value) <= hi


def _round(value: float) -> float:
    return round(clamp(value), 2)


# ------------------------------------------------------------------ pitch
def canonical_from_pitch_fraction(fx: float, fy: float) -> tuple[float, float]:
    """Canvas fraction (fx left→right, fy top→bottom) -> canonical pitch (x, y)."""
    return _round(fx * 100.0), _round((1.0 - fy) * 100.0)


def pitch_fraction_from_canonical(x: float, y: float) -> tuple[float, float]:
    """Inverse of :func:`canonical_from_pitch_fraction` (for placing markers)."""
    return clamp(x) / 100.0, 1.0 - clamp(y) / 100.0


# ------------------------------------------------------------------ goal
def canonical_from_goal_fraction(fx: float, fy: float) -> tuple[float, float]:
    """Canvas fraction within the goal interior -> canonical goal (goal_x, goal_y)."""
    return _round(fx * 100.0), _round((1.0 - fy) * 100.0)


def goal_fraction_from_canonical(goal_x: float, goal_y: float) -> tuple[float, float]:
    """Inverse of :func:`canonical_from_goal_fraction`."""
    return clamp(goal_x) / 100.0, 1.0 - clamp(goal_y) / 100.0


# ------------------------------------------------------------------ metres
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44


def to_metres(value: float | None, size: float) -> float | None:
    """Canonical 0..100 → metres on a dimension of ``size`` metres (pitch length / width, goal width / height)."""
    return None if value is None else round(clamp(value) / 100.0 * float(size), 2)


def from_metres(metres: float, size: float) -> float:
    """Metres on a dimension of ``size`` → canonical 0..100."""
    return _round(float(metres) / float(size) * 100.0) if size else 0.0
