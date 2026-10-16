"""Streamlit static component for the Set Piece Lab designer — a purpose-built HTML5 canvas editor
(players, ball, equipment, runs / passes / shots / dribbles that animate along their curves, zones,
text, a step timeline with notes and in-browser playback).

The browser owns the editing state and posts the whole routine back (debounced) as
``{"kind": "sync", "doc": ..., "cur": step}``; :func:`fap.setpiecelab.designer.clean_doc` is the trust
boundary. Python only pushes a document into the editor when ``load_nonce`` changes (template, data
pattern, library item), so reruns never interrupt editing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fap.setpiecelab.designer import clean_doc

_DIR = Path(__file__).resolve().parent / "frontend" / "setpiece_designer"
_impl: Any = None


def _component():
    global _impl
    if _impl is None:
        import streamlit.components.v1 as components
        _impl = components.declare_component("fap_setpiece_designer", path=str(_DIR))
    return _impl


def parse_result(value: Any) -> dict | None:
    """``{"ts", "doc", "cur", "nonce"}`` from the raw component value, or ``None``. Never raises."""
    if not isinstance(value, dict) or value.get("kind") != "sync" or not isinstance(value.get("ts"), (int, float)):
        return None
    doc = clean_doc(value.get("doc"))
    if doc is None:
        return None
    cur = value.get("cur")
    cur = int(cur) if isinstance(cur, (int, float)) and 0 <= cur < len(doc["steps"]) else 0
    nonce = value.get("nonce")
    return {"ts": float(value["ts"]), "doc": doc, "cur": cur, "nonce": int(nonce) if isinstance(nonce, (int, float)) else None}


def setpiece_designer(*, doc: dict, load_nonce: int, editable: bool, key: str, cur: int = 0) -> tuple[bool, dict | None]:
    """Render the editor → ``(rendered, result)``; ``rendered`` is False only if the component cannot mount."""
    try:
        value = _component()(doc=doc, load_nonce=int(load_nonce), editable=bool(editable), cur=int(cur),
                             key=key, default=None)
    except Exception:
        return False, None
    return True, parse_result(value)
