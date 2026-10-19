"""Interactive tagging canvas — the JS interaction layer ONLY.

A tiny Streamlit static component (same no-build pattern as ``tactical_canvas``).
The pitch and the goal are drawn by the PYTHON canonical renderers (PitchFactory
and ``fap.visuals.goal``) and handed to the browser as a background image; the
component adds pointer interaction (click to place a point, click-drag / two-click
to place a start→end line, hover coordinate readout, click a marker to select,
Delete/Escape) and reports the click as an **interior fraction** (0..1 within the
pitch/goal interior) — never pixels. Python converts that fraction to canonical
football coordinates with the pure ``fap.tagging.coordinates`` engine.

The component contains ZERO business logic. ``parse_result`` is the trust boundary:
it validates and clamps whatever the browser sends before it reaches the session,
and is unit-tested without a browser.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_ACTIONS = frozenset({"point", "line", "select", "delete", "cancel"})
_DIR = Path(__file__).resolve().parent / "frontend" / "tagging_canvas"
_impl: Any = None


def _component():
    global _impl
    if _impl is None:
        import streamlit.components.v1 as components
        _impl = components.declare_component("fap_tagging_canvas", path=str(_DIR))
    return _impl


def _frac(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


_BATCH_EVENTS = frozenset({"point", "line"})
_BATCH_OPS = frozenset({"select", "delete", "cancel", "undo", "redo"})


def _parse_batch(value: dict) -> dict[str, Any] | None:
    """``{kind: batch, token, seq, events: [{cid, action, ifx, ify[, ifx2, ify2]}], ops: [{aid, action[, select]}]}``.
    Every event / op carries a unique id so Python applies each exactly once, however often the browser
    re-sends it (nothing is lost when clicks come faster than Streamlit reruns)."""
    seq = value.get("seq")
    if not isinstance(seq, (int, float)):
        return None
    events, ops = [], []
    for e in (value.get("events") or [])[:500]:
        if not isinstance(e, dict) or e.get("action") not in _BATCH_EVENTS or not isinstance(e.get("cid"), str):
            continue
        out = {"cid": e["cid"][:64], "action": e["action"]}
        keys = ("ifx", "ify") + (("ifx2", "ify2") if e["action"] == "line" else ())
        vals = [_frac(e.get(k)) for k in keys]
        if any(v is None for v in vals):
            continue
        out.update(dict(zip(keys, vals)))
        events.append(out)
    for o in (value.get("ops") or [])[:200]:
        if not isinstance(o, dict) or o.get("action") not in _BATCH_OPS or not isinstance(o.get("aid"), str):
            continue
        out = {"aid": o["aid"][:64], "action": o["action"]}
        if isinstance(o.get("select"), str) and o["select"]:
            out["select"] = o["select"][:64]
        ops.append(out)
    return {"kind": "batch", "ts": float(seq), "events": events, "ops": ops}


def parse_result(value: Any) -> dict[str, Any] | None:
    """Normalise the raw component value into a clean intent dict, or ``None``.

    ``ts`` is the browser's monotonic action counter (callers ignore stale reruns).
    Fractions are clamped to ``[0, 1]``. Only allow-listed actions survive; anything
    malformed is dropped. Never raises."""
    if not isinstance(value, dict):
        return None
    if value.get("kind") == "batch":
        return _parse_batch(value)
    ts = value.get("ts")
    if not isinstance(ts, (int, float)):
        return None
    action = value.get("action")
    if action not in _ACTIONS:
        return None
    out: dict[str, Any] = {"ts": float(ts), "action": action}
    if action == "point":
        fx, fy = _frac(value.get("ifx")), _frac(value.get("ify"))
        if fx is None or fy is None:
            return None
        out["ifx"], out["ify"] = fx, fy
    elif action == "line":
        for k in ("ifx", "ify", "ifx2", "ify2"):
            f = _frac(value.get(k))
            if f is None:
                return None
            out[k] = f
    elif action == "select":
        sel = value.get("select")
        if not isinstance(sel, str) or not sel:
            return None
        out["select"] = sel
    # "delete" and "cancel" carry no payload
    return out


def tagging_board(*, space: str, dims: dict[str, float], overlay: list[dict[str, Any]], mode: str,
                  toggles: dict[str, Any], ctx: dict[str, Any], acked: list[str], selected: str | None,
                  key: str, editable: bool = True) -> tuple[bool, dict[str, Any] | None]:
    """The professional board: the browser draws the pitch (``dims`` metres) or the goal itself, keeps a local
    queue of clicks and reports them in idempotent batches; ``acked`` = ids Python has stored."""
    try:
        value = _component()(space=space, dims=dims, overlay=overlay, mode=mode, toggles=toggles, ctx=ctx,
                             acked=list(acked), selected=selected or "", editable=bool(editable),
                             key=key, default=None)
    except Exception:
        return False, None
    return True, parse_result(value)


def tagging_canvas(*, image: str, overlay: list[dict[str, Any]], mode: str,
                   readout: dict[str, Any], colors: dict[str, str], nonce: str,
                   key: str, editable: bool = True) -> tuple[bool, dict[str, Any] | None]:
    """Render the interactive canvas and return ``(rendered, intent)``.

    ``rendered`` is ``True`` when the iframe mounted (the caller then relies on the
    JS canvas for interaction); ``False`` means the component could not be created
    and the page must show its native coordinate-input fallback. ``image`` is the
    Python-rendered pitch/goal PNG (data URI) — the SAME canonical renderers used by
    every other map. ``overlay`` lists markers in interior-fraction coordinates for
    hit-testing/selection. ``readout`` carries the linear map so the browser can show
    a live canonical coordinate under the cursor. Never raises."""
    try:
        value = _component()(image=image, overlay=overlay, mode=mode, readout=readout,
                             colors=colors, nonce=nonce, editable=bool(editable),
                             key=key, default=None)
    except Exception:
        return False, None
    return True, parse_result(value)
