"""Server-side rendering of a set-piece routine (see :mod:`fap.setpiecelab.designer`): one step as a
publication PNG, the whole routine as a one-page coaching sheet, and the animation as a GIF.

The look mirrors the browser editor (``frontend/setpiece_designer``): same views, pitch styles, colours
and symbols, so what the coach drew is what gets exported.
"""
from __future__ import annotations

import io
import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from fap.setpiecelab.designer import PITCH_L, PITCH_W, bezier, frame_positions  # noqa: E402

try:
    from fap.matchlab.charts import FONT
except Exception:  # pragma: no cover
    FONT = ["DejaVu Sans"]

# ------------------------------------------------------------------ style (mirrors the JS editor)
STYLE = {
    "grass": dict(field=("#3b8746", "#367e41"), surround="#2f7139", line="#ffffff", line_a=0.9, card="#0f1720",
                  ink="#f8fafc", mut="#a8b3c4", run="#ffffff", pass_="#ffd23f", shot="#ff4d4f", plain="#ffffff",
                  pill="#0b1220"),
    "dark": dict(field=("#142030", "#17253a"), surround="#0f1824", line="#5b6f8c", line_a=0.95, card="#0b0f17",
                 ink="#eef2f8", mut="#8a96ab", run="#e8eef7", pass_="#ffd23f", shot="#ff5d73", plain="#cbd5e1",
                 pill="#0b0f17"),
    "light": dict(field=("#f4f6f9", "#eceff4"), surround="#ffffff", line="#9aa7b8", line_a=1.0, card="#ffffff",
                  ink="#0f172a", mut="#64748b", run="#1e293b", pass_="#d97706", shot="#dc2626", plain="#334155",
                  pill="#ffffff"),
}
VIEW_RECT = {"half": (PITCH_L / 2, PITCH_L), "third": (68.0, PITCH_L), "full": (0.0, PITCH_L)}
SCALE = {"half": 1.0, "third": 0.85, "full": 1.45}
MARGIN = 3.0


def _vertical(view: str) -> bool:
    return view in ("half", "third")


def project(view: str, x: float, y: float) -> tuple[float, float]:
    """World metres → screen (u right, v down)."""
    if _vertical(view):
        return PITCH_W - y, PITCH_L - x
    return x, PITCH_W - y


def extent(view: str) -> tuple[float, float, float, float]:
    """(u0, u1, v0, v1) of the visible area in screen units."""
    x0, x1 = VIEW_RECT[view]
    if _vertical(view):
        return -MARGIN, PITCH_W + MARGIN, -MARGIN, (PITCH_L - x0) + MARGIN
    return x0 - MARGIN, x1 + MARGIN, -MARGIN, PITCH_W + MARGIN


# ------------------------------------------------------------------ pitch
def _pitch_lines() -> list[list[tuple[float, float]]]:
    L, W = PITCH_L, PITCH_W
    out = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]
    circ = lambda cx, cy, r, a0, a1, n=60: [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),  # noqa: E731
                                            cy + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]
    out.append(circ(L / 2, W / 2, 9.15, 0, 360))
    for gx, s in ((0, 1), (L, -1)):
        out.append([(gx, W / 2 - 20.16), (gx + s * 16.5, W / 2 - 20.16), (gx + s * 16.5, W / 2 + 20.16), (gx, W / 2 + 20.16)])
        out.append([(gx, W / 2 - 9.16), (gx + s * 5.5, W / 2 - 9.16), (gx + s * 5.5, W / 2 + 9.16), (gx, W / 2 + 9.16)])
        out.append([(gx, W / 2 - 3.66), (gx - s * 2.0, W / 2 - 3.66), (gx - s * 2.0, W / 2 + 3.66), (gx, W / 2 + 3.66)])
        a = math.degrees(math.acos(5.5 / 9.15))
        px = gx + s * 11
        out.append(circ(px, W / 2, 9.15, -a, a) if s == 1 else circ(px, W / 2, 9.15, 180 - a, 180 + a))
        for cy, a0 in ((0, 0 if s == 1 else 90), (W, 270 if s == 1 else 180)):
            out.append(circ(gx, cy, 1.0, a0, a0 + 90, 12))
    return out


def _spots() -> list[tuple[float, float]]:
    return [(PITCH_L / 2, PITCH_W / 2), (11.0, PITCH_W / 2), (PITCH_L - 11.0, PITCH_W / 2)]


def draw_pitch(ax, view: str, style: str) -> None:
    S = STYLE[style]
    u0, u1, v0, v1 = extent(view)
    ax.set_facecolor(S["surround"])
    ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, color=S["surround"], zorder=0))
    # mowing stripes: bands across the length
    n = 18
    for k in range(n):
        xa, xb = PITCH_L * k / n, PITCH_L * (k + 1) / n
        pa, pb = project(view, xa, 0), project(view, xb, PITCH_W)
        ax.add_patch(Rectangle((min(pa[0], pb[0]), min(pa[1], pb[1])), abs(pb[0] - pa[0]), abs(pb[1] - pa[1]),
                               color=S["field"][k % 2], zorder=0.5, lw=0))
    for pts in _pitch_lines():
        uv = [project(view, x, y) for x, y in pts]
        ax.plot([p[0] for p in uv], [p[1] for p in uv], color=S["line"], alpha=S["line_a"], lw=1.6 * SCALE[view] ** -0.3,
                zorder=1, solid_capstyle="round")
    for x, y in _spots():
        u, v = project(view, x, y)
        ax.add_patch(Circle((u, v), 0.3, color=S["line"], alpha=S["line_a"], zorder=1))
    ax.set_xlim(u0, u1)
    ax.set_ylim(v1, v0)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------------ scene
def _ppm(ax) -> float:
    """Points per metre on this axes."""
    fig = ax.figure
    w_in = ax.get_position().width * fig.get_figwidth()
    u0, u1 = ax.get_xlim()
    return w_in * 72 / (u1 - u0)


def _obj_color(doc: dict, o: dict) -> str:
    if o.get("color"):
        return o["color"]
    c = doc.get("colors", {})
    if o["kind"] == "player":
        key = ("gk_" if o.get("gk") else "") + ("away" if o.get("team") == "away" else "home")
        return c.get(key, "#1f6feb")
    return {"cone": "#f97316", "mannequin": "#facc15", "pole": "#ef4444", "mark": "#9aa5b4"}.get(o["kind"], "#ffffff")


def _ink_on(hexcol: str) -> str:
    h = hexcol.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#111827" if (0.299 * r + 0.587 * g + 0.114 * b) > 165 else "#ffffff"


def _line_pts(view: str, ln: dict, n: int = 40) -> list[tuple[float, float]]:
    a, b, c = ln["a"], ln["b"], ln.get("c")
    pts = [bezier(a, b, c, k / n) for k in range(n + 1)]
    if ln["kind"] == "dribble":
        out = []
        for k in range(n + 1):
            p0 = pts[max(0, k - 1)]
            p1 = pts[min(n, k + 1)]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            d = math.hypot(dx, dy) or 1
            amp = 0.55 * math.sin(k / n * math.pi * 2 * max(2, int(math.dist(a, b) / 2.2)))
            fade = min(1.0, k / 4, (n - k) / 4)
            out.append((pts[k][0] - dy / d * amp * fade, pts[k][1] + dx / d * amp * fade))
        pts = out
    return [project(view, x, y) for x, y in pts]


def _arrow_head(ax, p_prev, p_end, size: float, color: str, alpha: float, z: float) -> None:
    dx, dy = p_end[0] - p_prev[0], p_end[1] - p_prev[1]
    d = math.hypot(dx, dy) or 1
    ux, uy = dx / d, dy / d
    base = (p_end[0] - ux * size, p_end[1] - uy * size)
    left = (base[0] - uy * size * 0.55, base[1] + ux * size * 0.55)
    right = (base[0] + uy * size * 0.55, base[1] - ux * size * 0.55)
    ax.add_patch(Polygon([p_end, left, right], closed=True, color=color, alpha=alpha, zorder=z, lw=0))


def draw_lines(ax, doc: dict, step: dict, alpha: float = 1.0) -> list:
    view, S, k = doc["view"], STYLE[doc["style"]], SCALE[doc["view"]]
    ppm = _ppm(ax)
    artists = []
    for ln in step.get("lines", []):
        col = ln.get("color") or {"run": S["run"], "pass": S["pass_"], "shot": S["shot"], "dribble": S["run"],
                                  "line": S["plain"]}[ln["kind"]]
        uv = _line_pts(view, ln)
        head = ln["kind"] != "line"
        if head:                                   # stop the shaft under the arrow head
            total = sum(math.dist(uv[j], uv[j + 1]) for j in range(len(uv) - 1))
            hs = min(1.5 * k, 0.4 * total)
            while len(uv) > 2 and math.dist(uv[-2], uv[-1]) < 1e-6:
                uv.pop()
            tip = uv[-1]
            cut = [p for p in uv if math.dist(p, tip) > hs * 0.7] or uv[:1]
            shaft = cut + [cut[-1]]
        else:
            shaft = uv
        lw = {"shot": 0.34, "pass": 0.26}.get(ln["kind"], 0.22) * k * ppm
        ls = (0, (2.2, 1.6)) if ln["kind"] == "run" else "-"
        artists += ax.plot([p[0] for p in shaft], [p[1] for p in shaft], color=col, lw=lw, ls=ls, alpha=alpha,
                           zorder=3, solid_capstyle="round", dash_capstyle="round")
        if head:
            _arrow_head(ax, cut[-1], tip, hs * (1.15 if ln["kind"] == "shot" else 1), col, alpha, 3.1)
            artists.append(ax.patches[-1])
    return artists


def draw_zones(ax, doc: dict, step: dict, alpha: float = 1.0) -> list:
    view, S = doc["view"], STYLE[doc["style"]]
    ppm = _ppm(ax)
    arts = []
    for z in step.get("zones", []):
        u, v = project(view, z["x"], z["y"])
        w, h = (z["h"], z["w"]) if _vertical(view) else (z["w"], z["h"])
        col = z.get("color") or "#ffd23f"
        if z["shape"] == "ellipse":
            p = Ellipse((u, v), w, h, facecolor=col, alpha=0.2 * alpha, edgecolor="none", zorder=2)
            e = Ellipse((u, v), w, h, facecolor="none", edgecolor=col, alpha=0.9 * alpha, lw=1.4, ls=(0, (4, 3)), zorder=2)
        else:
            r = min(w, h) * 0.12
            p = FancyBboxPatch((u - w / 2, v - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=col,
                               alpha=0.2 * alpha, edgecolor="none", zorder=2)
            e = FancyBboxPatch((u - w / 2, v - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor="none",
                               edgecolor=col, alpha=0.9 * alpha, lw=1.4, ls=(0, (4, 3)), zorder=2)
        ax.add_patch(p)
        ax.add_patch(e)
        arts += [p, e]
        if z.get("label"):
            arts.append(ax.text(u - w / 2 + 0.5, v - h / 2 + 0.5, z["label"].upper(), ha="left", va="top",
                                fontsize=max(5.0, 1.05 * ppm), fontweight="bold", color=col, alpha=alpha, zorder=2.2))
    return arts


def draw_texts(ax, doc: dict, step: dict, alpha: float = 1.0) -> list:
    view, S = doc["view"], STYLE[doc["style"]]
    ppm = _ppm(ax)
    arts = []
    for t in step.get("texts", []):
        u, v = project(view, t["x"], t["y"])
        arts.append(ax.text(u, v, t["text"], ha="center", va="center", fontsize=t.get("size", 1.8) * ppm * 0.62,
                            fontweight="bold", color=S["ink"], alpha=alpha, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.35", fc=S["pill"], ec="none", alpha=0.78 * alpha)))
    return arts


def draw_objects(ax, doc: dict, positions: dict, alpha_all: float = 1.0) -> list:
    view, S, k = doc["view"], STYLE[doc["style"]], SCALE[doc["view"]]
    ppm = _ppm(ax)
    arts = []
    order = sorted(doc["objects"], key=lambda o: {"mark": 0, "cone": 1, "pole": 1, "mannequin": 2, "player": 3, "ball": 4}[o["kind"]])
    for o in order:
        if o["id"] not in positions:
            continue
        x, y, a = positions[o["id"]]
        a *= alpha_all
        if a <= 0.02:
            continue
        u, v = project(view, x, y)
        col = _obj_color(doc, o)
        kind = o["kind"]
        if kind == "player":
            r = 1.2 * k
            arts.append(ax.add_patch(Circle((u + 0.18 * k, v + 0.28 * k), r, color="#000000", alpha=0.28 * a, zorder=4, lw=0)))
            arts.append(ax.add_patch(Circle((u, v), r, facecolor=col, edgecolor="#ffffff", lw=max(0.8, 0.16 * ppm * k),
                                            alpha=a, zorder=4.1)))
            if o.get("num"):
                arts.append(ax.text(u, v + 0.05 * k, o["num"], ha="center", va="center", color=_ink_on(col),
                                    fontsize=max(4.5, (1.25 if len(o["num"]) < 3 else 0.95) * k * ppm), fontweight="bold",
                                    alpha=a, zorder=4.2))
            if o.get("label"):
                arts.append(ax.text(u, v + r + 0.45 * k, o["label"], ha="center", va="top", color=S["ink"],
                                    fontsize=max(4.0, 0.82 * k * ppm), fontweight="bold", alpha=a, zorder=4.3,
                                    bbox=dict(boxstyle="round,pad=0.22", fc=S["pill"], ec="none", alpha=0.72 * a)))
        elif kind == "ball":
            r = 0.62 * k
            arts.append(ax.add_patch(Circle((u + 0.12, v + 0.18), r, color="#000000", alpha=0.3 * a, zorder=5, lw=0)))
            arts.append(ax.add_patch(Circle((u, v), r, facecolor="#ffffff", edgecolor="#111827", lw=max(0.6, 0.1 * ppm * k),
                                            alpha=a, zorder=5.1)))
            arts.append(ax.add_patch(Polygon([(u + r * 0.42 * math.cos(math.radians(90 + 72 * i)),
                                               v - r * 0.42 * math.sin(math.radians(90 + 72 * i))) for i in range(5)],
                                             closed=True, color="#111827", alpha=a, zorder=5.2, lw=0)))
        elif kind == "cone":
            s = 0.85 * k
            arts.append(ax.add_patch(Polygon([(u, v - s), (u - s * 0.85, v + s * 0.7), (u + s * 0.85, v + s * 0.7)],
                                             closed=True, facecolor=col, edgecolor="#7c2d12", lw=0.6, alpha=a, zorder=3.5)))
        elif kind == "mannequin":
            arts.append(ax.add_patch(FancyBboxPatch((u - 0.45 * k, v - 0.8 * k), 0.9 * k, 1.6 * k,
                                                    boxstyle=f"round,pad=0,rounding_size={0.4 * k}", facecolor=col,
                                                    edgecolor="#713f12", lw=0.7, alpha=a, zorder=3.6)))
        elif kind == "pole":
            arts.append(ax.add_patch(Circle((u, v), 0.42 * k, facecolor=col, edgecolor="#ffffff", lw=0.8, alpha=a, zorder=3.6)))
        elif kind == "mark":
            s = 0.95 * k
            arts.append(ax.add_patch(Polygon([(u, v - s), (u + s, v), (u, v + s), (u - s, v)], closed=True,
                                             facecolor=col, edgecolor="#ffffff", lw=0.7, alpha=0.95 * a, zorder=3.4)))
            if o.get("num"):
                arts.append(ax.text(u, v - s - 0.25, o["num"], ha="center", va="bottom", color=S["ink"],
                                    fontsize=max(4.0, 0.85 * k * ppm), fontweight="bold", alpha=a, zorder=3.45))
    return arts


def draw_step(ax, doc: dict, i: int, positions: dict | None = None, line_alpha: float = 1.0) -> list:
    step = doc["steps"][i]
    pos = positions if positions is not None else {k: (v[0], v[1], 1.0) for k, v in step["pos"].items()}
    return (draw_zones(ax, doc, step) + draw_lines(ax, doc, step, line_alpha) + draw_objects(ax, doc, pos)
            + draw_texts(ax, doc, step))


# ------------------------------------------------------------------ outputs
def _pitch_size(view: str, width_in: float) -> float:
    u0, u1, v0, v1 = extent(view)
    return width_in * (v1 - v0) / (u1 - u0)


def _card(doc: dict, width_in: float, header: float, footer: float):
    S = STYLE[doc["style"]]
    ph = _pitch_size(doc["view"], width_in)
    H = header + ph + footer
    fig = plt.figure(figsize=(width_in, H), facecolor=S["card"])
    ax = fig.add_axes([0, footer / H, 1, ph / H])
    draw_pitch(ax, doc["view"], doc["style"])
    return fig, ax, H


def _meta_line(doc: dict) -> str:
    m = doc.get("meta") or {}
    return "  ·  ".join(x for x in (m.get("category", ""), m.get("phase", "")) if x)


def step_figure(doc: dict, i: int, width_in: float = 8.0):
    S = STYLE[doc["style"]]
    step = doc["steps"][i]
    note = textwrap.wrap(step.get("note", ""), int(width_in * 13)) if step.get("note") else []
    header, footer = 0.95, 0.35 + 0.24 * len(note)
    with plt.rc_context({"font.family": FONT}):
        fig, ax, H = _card(doc, width_in, header, footer)
        fig.text(0.03, 1 - 0.16 / H, doc["name"], ha="left", va="top", fontsize=15, fontweight="bold", color=S["ink"])
        sub = f"STEP {i + 1}/{len(doc['steps'])}  ·  {step.get('name', '').upper()}"
        if _meta_line(doc):
            sub += "      " + _meta_line(doc)
        fig.text(0.03, 1 - 0.58 / H, sub, ha="left", va="top", fontsize=9, color=S["mut"], fontweight="bold")
        for j, line in enumerate(note):
            fig.text(0.03, (footer - 0.18 - 0.24 * j) / H, line, ha="left", va="top", fontsize=9.5, color=S["ink"])
        draw_step(ax, doc, i)
    return fig


def _save(fig, dpi: int) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def step_png(doc: dict, i: int, dpi: int = 300) -> bytes:
    return _save(step_figure(doc, i), dpi)


_DETAIL_LABELS = (("objective", "Objective"), ("trigger", "When to use"), ("taker", "Taker"), ("key_roles", "Key roles"),
                  ("code_word", "Signal"), ("setup", "Setup"), ("instructions", "Rules"),
                  ("coaching_points", "Coaching points"), ("progressions", "Progressions"), ("notes", "Notes"))


def sheet_figure(doc: dict, details: dict | None = None, kind: str = "routine"):
    """One page: title + chips, every step as a numbered panel with its note, coaching details."""
    S = STYLE[doc["style"]]
    details = details or {}
    n = len(doc["steps"])
    cols = 1 if n == 1 else 2 if n in (2, 4) else 3
    rows = math.ceil(n / cols)
    W = 16.0
    gap = 0.35
    pw = (W - 0.8 - gap * (cols - 1)) / cols
    ph = _pitch_size(doc["view"], pw)
    note_h = 0.62
    det = [(lab, str(details.get(k, "")).strip()) for k, lab in _DETAIL_LABELS if str(details.get(k, "")).strip()]
    det_lines = [(lab, textwrap.wrap(val, 110)) for lab, val in det]
    det_h = (0.35 + sum(0.22 * (len(v) + 0.8) for _, v in det_lines)) if det_lines else 0
    header = 1.25
    H = header + rows * (0.4 + ph + note_h) + det_h + 0.35
    with plt.rc_context({"font.family": FONT}):
        fig = plt.figure(figsize=(W, H), facecolor=S["card"])
        title = details.get("title") or doc["name"]
        fig.text(0.4 / W, 1 - 0.25 / H, title, ha="left", va="top", fontsize=22, fontweight="bold", color=S["ink"])
        chips = [details.get(k, "") for k in ("category", "phase", "side", "target", "focus", "players", "duration_min",
                                               "intensity")]
        chips = [c for c in chips if c] or [x for x in _meta_line(doc).split("  ·  ") if x]
        lab = "DRILL" if kind == "drill" else "SET-PIECE ROUTINE"
        fig.text(0.4 / W, 1 - 0.82 / H, f"{lab}   ·   " + "   ·   ".join(str(c) for c in chips) + f"   ·   {n} steps",
                 ha="left", va="top", fontsize=10.5, color=S["mut"], fontweight="bold")
        y = H - header
        for i in range(n):
            r, c = divmod(i, cols)
            top = y - r * (0.4 + ph + note_h)
            left = 0.4 + c * (pw + gap)
            fig.text(left / W, top / H, f"{i + 1}", ha="left", va="top", fontsize=13, fontweight="bold", color=S["pass_"])
            fig.text((left + 0.32) / W, top / H, doc["steps"][i].get("name", "") or f"Step {i + 1}", ha="left", va="top",
                     fontsize=12, fontweight="bold", color=S["ink"])
            ax = fig.add_axes([left / W, (top - 0.4 - ph) / H, pw / W, ph / H])
            draw_pitch(ax, doc["view"], doc["style"])
            draw_step(ax, doc, i)
            note = textwrap.wrap(doc["steps"][i].get("note", ""), int(pw * 15))[:3]
            for j, line in enumerate(note):
                fig.text(left / W, (top - 0.4 - ph - 0.1 - 0.19 * j) / H, line, ha="left", va="top", fontsize=8.8,
                         color=S["mut"])
        yy = H - header - rows * (0.4 + ph + note_h) - 0.15
        for lab_, lines in det_lines:
            fig.text(0.4 / W, yy / H, lab_.upper(), ha="left", va="top", fontsize=9, fontweight="bold", color=S["pass_"])
            yy -= 0.24
            for line in lines:
                fig.text(0.4 / W, yy / H, line, ha="left", va="top", fontsize=10, color=S["ink"])
                yy -= 0.22
            yy -= 0.12
    return fig


def sheet_png(doc: dict, details: dict | None = None, kind: str = "routine", dpi: int = 200) -> bytes:
    return _save(sheet_figure(doc, details, kind), dpi)


def gif_bytes(doc: dict, fps: int = 14, width_px: int = 820, hold: float = 0.8) -> bytes:
    """The routine animated: each step held, then every object travels along its line to the next step."""
    from PIL import Image
    S = STYLE[doc["style"]]
    dpi = 100
    w_in = width_px / dpi
    with plt.rc_context({"font.family": FONT}):
        header = 0.62
        ph = _pitch_size(doc["view"], w_in)
        H = header + ph
        fig = plt.figure(figsize=(w_in, H), facecolor=S["card"], dpi=dpi)
        ax = fig.add_axes([0, 0, 1, ph / H])
        draw_pitch(ax, doc["view"], doc["style"])
        fig.text(0.025, 1 - 0.1 / H, doc["name"], ha="left", va="top", fontsize=11.5, fontweight="bold", color=S["ink"])
        sub = fig.text(0.025, 1 - 0.38 / H, "", ha="left", va="top", fontsize=8, color=S["mut"], fontweight="bold")
        frames: list = []
        durations: list[int] = []

        def grab(ms: int):
            fig.canvas.draw()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
            frames.append(img.convert("RGB").quantize(colors=200, method=Image.Quantize.MEDIANCUT))
            durations.append(ms)

        n = len(doc["steps"])
        for i in range(n):
            st = doc["steps"][i]
            sub.set_text(f"STEP {i + 1}/{n}  ·  {(st.get('name') or '').upper()}")
            arts = draw_step(ax, doc, i)
            grab(int(hold * 1000 if i < n - 1 else 1800))
            for a in arts:
                a.remove()
            if i == n - 1:
                break
            k = max(4, int(st.get("dur", 1.6) * fps))
            for f in range(1, k + 1):
                t = f / k
                pos = frame_positions(doc, i, t)
                arts = (draw_zones(ax, doc, st) + draw_lines(ax, doc, st, alpha=max(0.0, 1 - t * 1.4))
                        + draw_objects(ax, doc, pos) + draw_texts(ax, doc, st, alpha=max(0.0, 1 - t * 2)))
                grab(int(1000 / fps))
                for a in arts:
                    a.remove()
        plt.close(fig)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=False, disposal=2)
    return buf.getvalue()
