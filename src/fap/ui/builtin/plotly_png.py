"""Plotly → PNG without kaleido: a tiny Streamlit component renders the figure in the browser with
``Plotly.toImage`` (pixel-identical to what the page shows) and returns the PNG to Python, which then
offers a normal ``st.download_button``.

Flow ("Prepare PNG" → "Download PNG"): the page calls :func:`prepare` with a fresh nonce; the component
renders and posts ``{nonce, images: [{name, png}]}`` back (one rerun); :func:`result` decodes the PNGs.
plotly.js is served next to the component (copied once from the installed ``plotly`` package), with a
CDN fallback.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent / "frontend" / "plotly_png"
_impl: Any = None


def _ensure_plotlyjs() -> None:
    target = _DIR / "plotly.min.js"
    if target.exists():
        return
    try:
        from plotly.offline import get_plotlyjs
        target.write_text(get_plotlyjs(), encoding="utf-8")
    except Exception:  # read-only deploy → the component falls back to the CDN copy
        pass


def _component():
    global _impl
    if _impl is None:
        _ensure_plotlyjs()
        import streamlit.components.v1 as components
        _impl = components.declare_component("fap_plotly_png", path=str(_DIR))
    return _impl


def fig_size(fig, default_w: int = 1300) -> tuple[int, int]:
    lay = fig.layout
    return int(lay.width or default_w), int(lay.height or 520)


def render(figs: list[tuple[str, Any, int, int]], nonce: str, key: str, scale: float = 3.0) -> list[tuple[str, bytes]] | None:
    """Render ``[(name, fig, width, height)]`` → ``[(name, png bytes)]`` once the browser answers, else None."""
    try:
        value = _component()(figs=[{"name": n, "fig": f.to_json(), "w": w, "h": h} for n, f, w, h in figs],
                             nonce=nonce, scale=scale, key=key, default=None)
    except Exception:
        return None
    if not isinstance(value, dict) or value.get("nonce") != nonce:
        return None
    out = []
    for im in value.get("images") or []:
        png = im.get("png") if isinstance(im, dict) else None
        if isinstance(png, str) and png.startswith("data:image/png;base64,"):
            try:
                out.append((str(im.get("name", "chart")), base64.b64decode(png.split(",", 1)[1])))
            except Exception:
                pass
    return out
