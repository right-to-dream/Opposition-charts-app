"""Set-piece routine & drill designer — the board model, templates and library documents.

A *routine* is a plain JSON dict (no tactical-board objects), edited in the browser by the dedicated
``setpiece_designer`` canvas and rendered server-side by :mod:`fap.setpiecelab.render` (PNG / sheet / GIF):

    {"v": 2, "name": str, "view": "half"|"third"|"full", "style": "grass"|"dark"|"light",
     "colors": {"home", "away", "gk_home", "gk_away"},
     "objects": [{"id", "kind": "player"|"ball"|"cone"|"mannequin"|"pole"|"mark",
                  "team": "home"|"away", "gk": bool, "num": str, "label": str, "color": str}],
     "steps":   [{"name", "note", "dur": seconds,
                  "pos":   {object id: [x, y]},            # objects shown in this step (absent = hidden)
                  "lines": [{"id", "kind": "run"|"pass"|"shot"|"dribble"|"line",
                             "a": [x, y], "b": [x, y], "c": [x, y] | None, "obj": id | None}],
                  "zones": [{"id", "shape": "rect"|"ellipse", "x", "y", "w", "h", "color", "label"}],
                  "texts": [{"id", "x", "y", "text", "size"}]}]}

Coordinates are metres on a 105 × 68 pitch, attacking the goal at x = 105, y = 0 = right touchline
(the same frame as Match Lab). A line with ``obj`` belongs to that player / ball: in the animation the
object travels along the line's curve to its position in the next step, and "next step" moves every
attached object to the end of its line — exactly what the browser editor does.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any

PITCH_L, PITCH_W = 105.0, 68.0
VIEWS = ("half", "third", "full")
STYLES = ("grass", "dark", "light")
OBJ_KINDS = ("player", "ball", "cone", "mannequin", "pole", "mark")
LINE_KINDS = ("run", "pass", "shot", "dribble", "line")
DEFAULT_COLORS = {"home": "#1f6feb", "away": "#e5484d", "gk_home": "#f5b301", "gk_away": "#16a34a"}


# ------------------------------------------------------------------ geometry
def bezier(a, b, c, t: float) -> tuple[float, float]:
    """Point on the quadratic curve a→b with control c (straight line when c is None)."""
    if c is None:
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    u = 1 - t
    return (u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1])


def bend_point(a, b, bend: float) -> list[float] | None:
    """Control point for a curve bending ``bend`` × length to the left of a→b (None = straight)."""
    if not bend:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    return [round((a[0] + b[0]) / 2 - dy * bend, 2), round((a[1] + b[1]) / 2 + dx * bend, 2)]


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ model helpers
def new_doc(name: str = "New routine", view: str = "half") -> dict:
    return {"v": 2, "name": name, "view": view, "style": "grass", "colors": dict(DEFAULT_COLORS),
            "objects": [], "steps": [{"name": "Setup", "note": "", "dur": 1.6, "pos": {}, "lines": [], "zones": [], "texts": []}],
            "meta": {}}


def advance(step: dict, name: str = "", note: str = "") -> dict:
    """The next step: same objects, every object with a line moved to the line's end, lines cleared."""
    nxt = {"name": name or "Step", "note": note, "dur": step.get("dur", 1.6), "pos": copy.deepcopy(step["pos"]),
           "lines": [], "zones": copy.deepcopy(step.get("zones", [])), "texts": []}
    for ln in step.get("lines", []):
        if ln.get("obj") in nxt["pos"] and ln["kind"] != "line":
            nxt["pos"][ln["obj"]] = list(ln["b"])
    return nxt


def frame_positions(doc: dict, i: int, t: float) -> dict[str, tuple[float, float, float]]:
    """Positions (x, y, alpha) of every object during the transition step i → i+1 at t ∈ [0, 1]."""
    steps = doc["steps"]
    s0 = steps[i]
    s1 = steps[i + 1] if i + 1 < len(steps) else s0
    te = ease(max(0.0, min(1.0, t)))
    lines = {ln["obj"]: ln for ln in s0.get("lines", []) if ln.get("obj") and ln["kind"] != "line"}
    out = {}
    for oid in set(s0["pos"]) | set(s1["pos"]):
        p0, p1 = s0["pos"].get(oid), s1["pos"].get(oid)
        if p0 and p1:
            ln = lines.get(oid)
            if ln is not None and math.dist(ln["b"], p1) < 0.5:
                x, y = bezier(p0, p1, ln.get("c"), te)
            else:
                x, y = p0[0] + (p1[0] - p0[0]) * te, p0[1] + (p1[1] - p0[1]) * te
            out[oid] = (x, y, 1.0)
        elif p0:
            out[oid] = (p0[0], p0[1], 1 - te)
        else:
            out[oid] = (p1[0], p1[1], te)
    return out


def _num(v: Any, lo: float = -20.0, hi: float = 130.0, d: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return d
    return d if math.isnan(f) else max(lo, min(hi, f))


def _pt(v: Any) -> list[float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return [round(_num(v[0]), 2), round(_num(v[1]), 2)]
    return None


def _str(v: Any, n: int = 120) -> str:
    return str(v)[:n] if isinstance(v, (str, int, float)) else ""


def _color(v: Any, d: str = "") -> str:
    return v if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{3,8}", v) else d


def clean_doc(raw: Any) -> dict | None:
    """Validate a document coming from the browser (the trust boundary). Never raises."""
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list) or not raw["steps"]:
        return None
    doc = {"v": 2, "name": _str(raw.get("name"), 120) or "Routine",
           "view": raw.get("view") if raw.get("view") in VIEWS else "half",
           "style": raw.get("style") if raw.get("style") in STYLES else "grass",
           "colors": {k: _color((raw.get("colors") or {}).get(k), v) for k, v in DEFAULT_COLORS.items()},
           "objects": [], "steps": [], "meta": raw.get("meta") if isinstance(raw.get("meta"), dict) else {}}
    ids = set()
    for o in (raw.get("objects") or [])[:120]:
        if not isinstance(o, dict) or o.get("kind") not in OBJ_KINDS or not _str(o.get("id"), 40):
            continue
        oid = _str(o["id"], 40)
        ids.add(oid)
        doc["objects"].append({"id": oid, "kind": o["kind"], "team": "away" if o.get("team") == "away" else "home",
                               "gk": bool(o.get("gk")), "num": _str(o.get("num"), 4), "label": _str(o.get("label"), 40),
                               "color": _color(o.get("color"))})
    for s in raw["steps"][:30]:
        if not isinstance(s, dict):
            continue
        st = {"name": _str(s.get("name"), 60), "note": _str(s.get("note"), 400),
              "dur": _num(s.get("dur"), 0.4, 6.0, 1.6), "pos": {}, "lines": [], "zones": [], "texts": []}
        for k, v in (s.get("pos") or {}).items():
            p = _pt(v)
            if k in ids and p:
                st["pos"][k] = p
        for ln in (s.get("lines") or [])[:120]:
            if isinstance(ln, dict) and ln.get("kind") in LINE_KINDS and _pt(ln.get("a")) and _pt(ln.get("b")):
                st["lines"].append({"id": _str(ln.get("id"), 40) or f"l{len(st['lines'])}", "kind": ln["kind"],
                                    "a": _pt(ln["a"]), "b": _pt(ln["b"]), "c": _pt(ln.get("c")),
                                    "obj": ln.get("obj") if ln.get("obj") in ids else None,
                                    "color": _color(ln.get("color"))})
        for z in (s.get("zones") or [])[:40]:
            if isinstance(z, dict):
                st["zones"].append({"id": _str(z.get("id"), 40) or f"z{len(st['zones'])}",
                                    "shape": "ellipse" if z.get("shape") == "ellipse" else "rect",
                                    "x": _num(z.get("x")), "y": _num(z.get("y")),
                                    "w": _num(z.get("w"), 0.5, 110, 8), "h": _num(z.get("h"), 0.5, 80, 8),
                                    "color": _color(z.get("color"), "#ffd23f"), "label": _str(z.get("label"), 40)})
        for t in (s.get("texts") or [])[:40]:
            if isinstance(t, dict) and _str(t.get("text"), 120).strip():
                st["texts"].append({"id": _str(t.get("id"), 40) or f"t{len(st['texts'])}", "x": _num(t.get("x")),
                                    "y": _num(t.get("y")), "text": _str(t.get("text"), 120),
                                    "size": _num(t.get("size"), 0.8, 6, 1.8)})
        doc["steps"].append(st)
    return doc if doc["steps"] else None


# ------------------------------------------------------------------ template builder
class _Routine:
    def __init__(self, name: str, view: str = "third", category: str = "", phase: str = "Attacking"):
        self.doc = new_doc(name, view)
        self.doc["steps"] = []
        self.doc["meta"] = {"category": category, "phase": phase}
        self.cur: dict | None = None
        self._n = 0

    def obj(self, kind: str, *, team: str = "home", num: str = "", label: str = "", gk: bool = False,
            color: str = "") -> str:
        self._n += 1
        oid = f"o{self._n}"
        self.doc["objects"].append({"id": oid, "kind": kind, "team": team, "gk": gk, "num": str(num),
                                    "label": label, "color": color})
        return oid

    def player(self, team: str, num, x: float, y: float, label: str = "", gk: bool = False) -> str:
        oid = self.obj("player", team=team, num=num, label=label, gk=gk)
        self.cur["pos"][oid] = [x, y]
        return oid

    def start(self, name: str, note: str = "", dur: float = 1.6) -> "_Routine":
        self.cur = {"name": name, "note": note, "dur": dur, "pos": {}, "lines": [], "zones": [], "texts": []}
        self.doc["steps"].append(self.cur)
        return self

    def ball(self, x: float, y: float) -> str:
        oid = self.obj("ball")
        self.cur["pos"][oid] = [x, y]
        return oid

    def line(self, kind: str, oid: str | None, to, bend: float = 0.0, frm=None) -> None:
        a = list(frm) if frm is not None else list(self.cur["pos"][oid])
        self.cur["lines"].append({"id": f"l{len(self.cur['lines'])}_{len(self.doc['steps'])}", "kind": kind,
                                  "a": a, "b": list(to), "c": bend_point(a, to, bend), "obj": oid, "color": ""})

    def zone(self, x, y, w, h, label: str = "", color: str = "#ffd23f", shape: str = "rect") -> None:
        self.cur["zones"].append({"id": f"z{len(self.cur['zones'])}", "shape": shape, "x": x, "y": y, "w": w, "h": h,
                                  "color": color, "label": label})

    def text(self, x, y, s: str, size: float = 1.8) -> None:
        self.cur["texts"].append({"id": f"t{len(self.cur['texts'])}", "x": x, "y": y, "text": s, "size": size})

    def next(self, name: str, note: str = "", dur: float = 1.6, move: dict | None = None) -> "_Routine":
        nxt = advance(self.cur, name, note)
        nxt["dur"] = dur
        for k, v in (move or {}).items():
            nxt["pos"][k] = list(v)
        self.doc["steps"].append(nxt)
        self.cur = nxt
        return self

    def build(self) -> dict:
        return self.doc


# corner from the right (y = 0): flag at (105, 0)
_CR = (105.9, -1.2)


def _box_defenders(r: _Routine, scheme: str = "mixed") -> dict:
    """Six-to-eight opponents defending a corner from the right."""
    d = {}
    d["gk"] = r.player("away", 1, 104.0, 32.5, gk=True)
    if scheme in ("zonal", "mixed"):
        for i, y in enumerate((28.5, 32.5, 36.5)):
            d[f"z{i}"] = r.player("away", 2 + i, 100.0, y)
    if scheme in ("man", "mixed"):
        for i, (x, y) in enumerate(((95.5, 27.0), (94.5, 33.0), (95.5, 39.0), (93.0, 30.0))):
            d[f"m{i}"] = r.player("away", 5 + i, x, y)
    d["post"] = r.player("away", 9, 103.8, 29.0)
    return d


def _corner_near_post_flick() -> dict:
    r = _Routine("Attacking corner — inswinger, near-post flick-on", category="Corner")
    r.start("Setup", "Stack on the penalty spot; taker #7 on the right. Signal = hand up.")
    t = r.player("home", 7, *_CR, label="Taker")
    a9 = r.player("home", 9, 92.0, 33.0, label="Flick-on")
    a4 = r.player("home", 4, 91.0, 30.0)
    a5 = r.player("home", 5, 91.0, 36.0)
    a10 = r.player("home", 10, 90.0, 39.0, label="Far post")
    a8 = r.player("home", 8, 86.0, 22.0, label="2nd ball")
    _box_defenders(r)
    b = r.ball(104.2, 1.0)
    r.next("Movement", "#9 attacks the near post, #4 and #5 split across the six-yard box, #10 holds the far post.")
    r.line("run", a9, (100.5, 26.5), bend=-0.25)
    r.line("run", a4, (99.0, 32.0), bend=0.1)
    r.line("run", a5, (100.0, 38.0), bend=-0.1)
    r.line("run", a10, (101.0, 43.0), bend=0.15)
    r.line("run", a8, (88.5, 28.0))
    r.zone(98.5, 25.0, 5.0, 5.0, "TARGET")
    r.next("Delivery", "Fast inswinger into the near-post zone at head height.")
    r.line("pass", b, (100.5, 26.5), bend=0.18)
    r.next("Flick-on", "#9 glances it across the six-yard box; #5 / #10 attack the far post.", dur=1.3)
    r.line("pass", b, (101.2, 38.5), bend=-0.08)
    r.line("run", a5, (101.3, 38.8))
    r.next("Finish", "Finish at the far post; #8 holds the edge of the box for the second ball.")
    r.line("shot", b, (105.0, 35.5))
    return r.build()


def _corner_far_post() -> dict:
    r = _Routine("Attacking corner — outswinger to the far post", category="Corner")
    r.start("Setup", "Blockers on the GK, two runners from deep.")
    t = r.player("home", 11, *_CR, label="Taker")
    a6 = r.player("home", 6, 89.0, 42.0, label="Main target")
    a3 = r.player("home", 3, 88.0, 37.0)
    a9 = r.player("home", 9, 102.0, 32.0, label="Screen GK")
    a4 = r.player("home", 4, 93.0, 30.0)
    a8 = r.player("home", 8, 84.0, 30.0, label="Edge")
    _box_defenders(r, "zonal")
    b = r.ball(104.2, 1.0)
    r.next("Movement", "#9 screens the keeper; #4 drags the zonal line to the near post; #6 and #3 curve to the far post.")
    r.line("run", a4, (99.5, 27.0), bend=-0.2)
    r.line("run", a6, (98.0, 44.0), bend=0.25)
    r.line("run", a3, (97.0, 39.5), bend=0.2)
    r.zone(97.5, 44.0, 6.0, 6.0, "TARGET", shape="ellipse")
    r.next("Delivery & finish", "Outswinger to the far post — attack it with momentum, head down into the ground.")
    r.line("pass", b, (98.0, 44.0), bend=0.12)
    r.next("Shot", "")
    r.line("shot", b, (105.0, 31.0))
    return r.build()


def _corner_short() -> dict:
    r = _Routine("Attacking corner — short 2v1 and cut-back", category="Corner")
    r.start("Setup", "#7 takes, #10 shows short; the box is loaded but stays deep.")
    t = r.player("home", 7, *_CR, label="Taker")
    s = r.player("home", 10, 99.0, 9.0, label="Short")
    a9 = r.player("home", 9, 92.0, 30.0)
    a8 = r.player("home", 8, 87.0, 24.0, label="Cut-back")
    a4 = r.player("home", 4, 92.0, 36.0)
    d = {"gk": r.player("away", 1, 104.0, 32.5, gk=True), "p": r.player("away", 2, 99.5, 13.0),
         "z": r.player("away", 3, 100.0, 30.0), "m": r.player("away", 4, 93.0, 33.0)}
    b = r.ball(104.2, 1.0)
    r.next("Short pass", "Short to #10; #7 follows to make the 2v1.")
    r.line("pass", b, (99.0, 9.0))
    r.line("run", t, (101.0, 7.0), bend=0.2)
    r.next("Return & drive", "#10 returns into #7's run along the byline.")
    r.line("pass", b, (101.5, 8.0))
    r.line("run", s, (98.5, 15.5))
    r.next("Cut-back", "#7 drives to the box and cuts back to #8 arriving at the penalty spot.")
    r.line("dribble", b, (102.5, 19.5))
    r.line("run", t, (102.5, 19.5), bend=-0.05)
    r.line("run", a8, (93.5, 27.5), bend=0.15)
    r.next("Finish", "")
    r.line("pass", b, (93.5, 27.5), bend=0.05)
    r.next("Shot", "", move={})
    r.line("shot", b, (105.0, 35.0))
    return r.build()


def _defend_corner(scheme: str) -> dict:
    title = {"zonal": "Defending corner — zonal", "man": "Defending corner — man-to-man",
             "mixed": "Defending corner — mixed (zonal + man)"}[scheme]
    r = _Routine(title, category="Corner", phase="Defending")
    r.start("Shape", {"zonal": "Six zonal players across the six-yard box and penalty spot; two on the edge for second balls.",
                      "man": "Every attacker marked goal-side; one player on the near post, one free for second balls.",
                      "mixed": "Three zonal (near / centre / far six-yard) + markers on the biggest threats."}[scheme])
    r.player("home", 1, 104.0, 32.5, gk=True, label="GK")
    att = [r.player("away", n, x, y) for n, x, y in ((9, 92.0, 30.0), (5, 91.5, 35.0), (4, 93.0, 39.0),
                                                       (6, 90.0, 26.0), (8, 86.0, 33.0))]
    taker = r.player("away", 7, *_CR, label="Taker")
    if scheme in ("zonal", "mixed"):
        for n, y in ((2, 28.0), (3, 32.5), (4, 37.0)):
            r.player("home", n, 100.0, y, label="Zone")
    if scheme == "zonal":
        for n, (x, y) in ((5, (95.0, 26.0)), (6, (94.5, 33.0)), (8, (95.0, 40.0))):
            r.player("home", n, x, y, label="Zone")
    if scheme in ("man", "mixed"):
        for k, (n, (x, y)) in enumerate(((5, (93.0, 30.0)), (6, (92.5, 35.0)), (8, (94.0, 39.0)), (10, (91.0, 26.5)))):
            if scheme == "mixed" and k > 2:
                break
            r.player("home", n, x + 2.4, y + 0.6, label="Mark")
    r.player("home", 11, 104.0, 28.0, label="Near post")
    r.player("home", 7, 85.0, 30.0, label="2nd ball")
    r.player("home", 9, 70.0, 22.0, label="Counter")
    b = r.ball(104.2, 1.0)
    r.zone(100.5, 32.5, 7.0, 18.0, "ZONES" if scheme != "man" else "GK AREA")
    r.next("Their runs", "Attackers start their runs — markers stay goal-side and touch tight; zones attack the ball, not the man.")
    for oid, to in zip(att, ((99.0, 27.0), (98.0, 33.5), (99.0, 39.0), (96.0, 25.0), (91.0, 31.0))):
        r.line("run", oid, to, bend=0.12)
    r.next("Clear & step out", "First contact: clear high and wide. Whole line steps out to the 18 m line; #9 offers for the counter.")
    r.line("pass", b, (99.5, 30.0), bend=0.15)
    r.next("Out", "")
    r.line("pass", b, (78.0, 18.0), bend=0.1)
    return r.build()


def _fk_wide() -> dict:
    r = _Routine("Attacking free kick — wide delivery, blind-side run", category="Free kick")
    r.start("Setup", "Ball 25 m from goal on the right; line of five on the 18 m line.")
    t = r.player("home", 7, 84.0, 8.0, label="Taker")
    line = [r.player("home", n, 88.0, y) for n, y in ((9, 26.0), (4, 30.0), (5, 34.0), (6, 38.0), (10, 42.0))]
    for n, y in ((2, 27.5), (3, 31.5), (4, 35.5), (5, 39.5)):
        r.player("away", n, 89.5, y)
    r.player("away", 1, 104.0, 32.5, gk=True)
    b = r.ball(84.6, 8.6)
    r.next("Movement", "On the run-up: #9 and #4 go near, #5 and #6 hold then spin, #10 blind-side to the far post.")
    for oid, to, bd in zip(line, ((99.0, 27.0), (99.5, 31.0), (97.5, 35.0), (98.0, 39.0), (100.5, 44.0)),
                           (0.1, 0.0, 0.0, -0.1, 0.25)):
        r.line("run", oid, to, bend=bd)
    r.zone(98.5, 31.0, 6.0, 12.0, "DELIVERY ZONE")
    r.next("Delivery", "Whipped delivery between the defensive line and the keeper.")
    r.line("pass", b, (99.0, 34.0), bend=0.15)
    r.next("Finish", "")
    r.line("shot", b, (105.0, 34.5))
    return r.build()


def _fk_direct() -> dict:
    r = _Routine("Attacking free kick — direct shot with dummy", category="Free kick")
    r.start("Setup", "Central, 22 m. #10 and #7 over the ball, wall of four, a player in the wall.")
    a10 = r.player("home", 10, 81.5, 35.5, label="Dummy")
    a7 = r.player("home", 7, 81.5, 32.0, label="Shooter")
    a9 = r.player("home", 9, 92.5, 34.0, label="In the wall")
    for i, y in enumerate((30.0, 31.2, 32.4, 33.6)):
        r.obj("mannequin")
        r.cur["pos"][r.doc["objects"][-1]["id"]] = [92.0, y]
    r.player("away", 1, 104.0, 33.0, gk=True)
    a6 = r.player("home", 6, 91.0, 42.0, label="Rebound")
    b = r.ball(83.0, 34.0)
    r.next("Dummy", "#10 runs over the ball; #9 peels away from the wall to open the gap.")
    r.line("run", a10, (85.0, 37.0))
    r.line("run", a9, (93.0, 37.0))
    r.next("Shot", "#7 shoots through the gap #9 opened; #6 follows in for the rebound.")
    r.line("shot", b, (105.0, 31.0), bend=-0.04)
    r.line("run", a6, (99.0, 38.0))
    return r.build()


def _fk_defend() -> dict:
    r = _Routine("Defending wide free kick — high line", category="Free kick", phase="Defending")
    r.start("Shape", "Line of six on the 16 m line, one blocker on the taker, two for second balls.")
    r.player("home", 1, 103.0, 32.5, gk=True)
    line = [r.player("home", n, 89.0, y) for n, y in ((2, 25.0), (3, 29.0), (4, 33.0), (5, 37.0), (6, 41.0), (8, 45.0))]
    r.player("home", 10, 87.0, 12.0, label="Block")
    r.player("home", 7, 80.0, 30.0, label="2nd ball")
    for n, y in ((9, 26.5), (4, 31.0), (5, 35.0), (6, 40.0)):
        r.player("away", n, 87.0, y)
    r.player("away", 7, 82.5, 6.5, label="Taker")
    b = r.ball(83.0, 7.0)
    r.zone(89.0, 35.0, 1.0, 22.0, "LINE")
    r.next("Step out", "On the taker's run-up the line steps up 2 m together — attackers are left offside.")
    for oid in line:
        p = r.cur["pos"][oid]
        r.line("run", oid, (p[0] - 2.5, p[1]))
    r.next("Attack the ball", "Defend forwards: first contact by the line, clear wide.")
    r.line("pass", b, (90.0, 33.0), bend=0.12)
    return r.build()


def _long_throw() -> dict:
    r = _Routine("Long throw — near-post flick", category="Throw-in")
    r.start("Setup", "Throw from 10 m out on the right; target #9 at the near post.")
    t = r.player("home", 3, 95.0, 0.0, label="Thrower")
    a9 = r.player("home", 9, 94.0, 20.0, label="Flick")
    a5 = r.player("home", 5, 92.0, 30.0)
    a4 = r.player("home", 4, 92.0, 35.0)
    a8 = r.player("home", 8, 86.0, 26.0, label="2nd ball")
    r.player("away", 1, 104.0, 32.5, gk=True)
    for n, (x, y) in ((2, (98.0, 22.0)), (3, (97.0, 30.0)), (4, (97.0, 35.0)), (5, (94.0, 24.0))):
        r.player("away", n, x, y)
    b = r.ball(95.0, 0.6)
    r.next("Runs", "#9 attacks the near post; #5 and #4 run across the six-yard box.")
    r.line("run", a9, (100.0, 21.0), bend=-0.1)
    r.line("run", a5, (100.0, 30.5))
    r.line("run", a4, (100.5, 36.0))
    r.next("Throw & flick", "Flat, fast throw onto #9's head — flick across goal.")
    r.line("pass", b, (100.0, 21.0), bend=0.12)
    r.next("Finish", "")
    r.line("pass", b, (100.5, 33.0))
    r.next("Shot", "")
    r.line("shot", b, (105.0, 33.5))
    return r.build()


def _goal_kick_long() -> dict:
    r = _Routine("Goal kick — long to the target, win the second ball", category="Goal kick", phase="Attacking")
    r.doc["view"] = "full"
    r.start("Setup", "Overload the left side: #9 target, #10 and #8 around him to win the second ball.")
    gk = r.player("home", 1, 4.0, 30.0, gk=True)
    for n, (x, y) in ((4, (16.0, 24.0)), (5, (16.0, 44.0)), (2, (30.0, 6.0)), (3, (30.0, 62.0)), (6, (38.0, 34.0))):
        r.player("home", n, x, y)
    a9 = r.player("home", 9, 58.0, 48.0, label="Target")
    a10 = r.player("home", 10, 54.0, 42.0)
    a8 = r.player("home", 8, 50.0, 52.0)
    a11 = r.player("home", 11, 62.0, 60.0)
    a7 = r.player("home", 7, 60.0, 14.0)
    b = r.ball(5.5, 30.0)
    r.next("Kick", "Long kick into #9's zone; #10 and #8 squeeze around the landing point.")
    r.line("pass", b, (58.0, 48.0), bend=0.12)
    r.line("run", a10, (57.0, 44.5))
    r.line("run", a8, (55.0, 51.0))
    r.line("run", a11, (66.0, 56.0))
    r.zone(58.0, 48.0, 14.0, 12.0, "2ND BALL", shape="ellipse")
    r.next("Knock-down", "#9 heads down into #10's path; #11 runs in behind.")
    r.line("pass", b, (57.0, 44.5))
    r.line("run", a11, (78.0, 58.0), bend=-0.1)
    r.next("Release", "#10 releases #11 in behind the full-back.")
    r.line("pass", b, (78.0, 58.0), bend=0.08)
    return r.build()


def _penalty() -> dict:
    r = _Routine("Penalty — taker routine and rebounds", category="Penalty")
    r.doc["view"] = "third"
    r.start("Setup", "Taker #9; #10 and #8 on the edge of the D for the rebound.")
    t = r.player("home", 9, 92.0, 34.0, label="Taker")
    a10 = r.player("home", 10, 87.0, 27.5, label="Rebound")
    a8 = r.player("home", 8, 87.0, 40.5, label="Rebound")
    r.player("away", 1, 105.0, 34.0, gk=True)
    r.player("away", 4, 87.5, 30.0)
    r.player("away", 5, 87.5, 38.0)
    b = r.ball(94.0, 34.0)
    r.next("Kick", "Pick the side early; strike low and hard. Rebounders go on contact.")
    r.line("shot", b, (105.0, 31.0))
    r.line("run", a10, (97.5, 30.0))
    r.line("run", a8, (97.5, 38.0))
    r.next("Rebound", "If the keeper saves, the first rebounder finishes; the second covers the far side.")
    return r.build()


def _drill_second_ball() -> dict:
    r = _Routine("Drill — second balls on the edge of the box", category="Other")
    r.doc["meta"]["kind"] = "drill"
    r.start("Setup", "Coach serves from the corner flag; 3 v 2 in the box, 2 v 2 on the edge. Cones mark the D.")
    c = r.player("home", "C", *_CR, label="Coach")
    for x, y in ((88.0, 26.0), (88.0, 42.0), (84.0, 34.0)):
        r.obj("cone")
        r.cur["pos"][r.doc["objects"][-1]["id"]] = [x, y]
    h1 = r.player("home", 8, 86.0, 28.0)
    h2 = r.player("home", 6, 86.0, 39.0)
    for n, (x, y) in ((4, (97.0, 30.0)), (5, (97.0, 37.0))):
        r.player("away", n, x, y)
    for n, (x, y) in ((9, (95.0, 33.0)), (10, (95.0, 38.0)), (11, (96.0, 28.0))):
        r.player("home", n, x, y)
    b = r.ball(104.2, 1.0)
    r.next("Serve & clear", "Defenders head the cross out to the edge.")
    r.line("pass", b, (98.0, 32.0), bend=0.15)
    r.next("Win the 2nd ball", "Edge players attack the clearance on the half-volley or recycle wide.")
    r.line("pass", b, (87.0, 31.0), bend=0.1)
    r.line("run", h1, (87.0, 31.0))
    r.next("Finish", "")
    r.line("shot", b, (105.0, 36.0), bend=-0.03)
    return r.build()


TEMPLATES = {
    "Attacking corner · inswinger, near-post flick-on": _corner_near_post_flick,
    "Attacking corner · outswinger to the far post": _corner_far_post,
    "Attacking corner · short 2v1 and cut-back": _corner_short,
    "Defending corner · zonal": lambda: _defend_corner("zonal"),
    "Defending corner · man-to-man": lambda: _defend_corner("man"),
    "Defending corner · mixed": lambda: _defend_corner("mixed"),
    "Attacking free kick · wide delivery": _fk_wide,
    "Attacking free kick · direct shot with dummy": _fk_direct,
    "Defending wide free kick · high line": _fk_defend,
    "Long throw · near-post flick": _long_throw,
    "Goal kick · long + second ball": _goal_kick_long,
    "Penalty · taker and rebounds": _penalty,
    "Drill · second balls on the edge": _drill_second_ball,
}


def template(name: str) -> dict:
    return TEMPLATES[name]()


def blank(view: str = "half", name: str = "New routine") -> dict:
    return new_doc(name, view)


# ------------------------------------------------------------------ data-driven routine
_OUT_COLORS = {"goal": "#e5484d", "shot": "#f97316", "threat": "#f5b301"}


def pattern_doc(sd, row: str) -> dict:
    """The opponent's coded set piece as a routine: every landing point (numbered as in the file, coloured
    by outcome), the hot zone, and — for corners / free kicks — a step with our zonal counter-setup."""
    from fap.setpiecelab import analytics as A
    g = sd.of(row=row)
    pts = g[g["x"].notna()]
    cat = g["category"].iloc[0] if len(g) else ""
    att = (g["phase"].iloc[0] == "Attacking") if len(g) else True
    xs, ys = pts["x"].astype(float).tolist(), pts["y"].astype(float).tolist()
    flip = bool(xs) and sum(xs) / len(xs) < PITCH_L / 2         # shown towards the goal at the top
    if flip:
        xs, ys = [PITCH_L - x for x in xs], [PITCH_W - y for y in ys]
    view = "third" if (not xs or min(xs) >= 70) else "half" if min(xs) >= PITCH_L / 2 - 2 else "full"
    r = _Routine(f"{sd.team} · {row} — their pattern", view=view, category=cat,
                 phase="Defending" if att else "Attacking")
    out_c = A._col(sd, "Outcome")
    n_out = {}
    note = (f"{len(g)} coded, {len(pts)} with a landing point. Colour = outcome "
            f"(red goal · orange shot · yellow threat · grey other).")
    r.start("Their landing points", note)
    for (_, rr), x, y in zip(pts.iterrows(), xs, ys):
        o = (rr[out_c] if out_c else "").lower()
        n_out[o] = n_out.get(o, 0) + 1
        oid = r.obj("mark", num=str(rr["n"]), label=(rr[out_c] if out_c else ""), color=_OUT_COLORS.get(o, "#9aa5b4"))
        r.cur["pos"][oid] = [round(x, 2), round(y, 2)]
    if len(xs) >= 2:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys) / len(ys)) ** 0.5
        r.zone(round(mx, 1), round(my, 1), round(max(5.0, 2.4 * sx), 1), round(max(6.0, 2.4 * sy), 1),
               "THEIR HOT ZONE", color="#e5484d", shape="ellipse")
        if cat in ("Corner", "Free kick") and view in ("half", "third"):
            r.next("Our counter-setup", "Zonal players on their hot zone, GK on his line, one on the near post.")
            for k, (dx, dy) in enumerate(((0, -5), (0, 0), (0, 5), (-4.5, -3), (-4.5, 3)), start=2):
                pid = r.obj("player", team="home", num=str(k), label="Zone")
                r.cur["pos"][pid] = [round(min(103.0, mx + dx), 1), round(max(1.0, min(67.0, my + dy)), 1)]
            gk = r.obj("player", team="home", num="1", gk=True, label="GK")
            r.cur["pos"][gk] = [104.0, 32.5]
    r.doc["meta"].update({"source": "data", "opponent": sd.team, "row": row})
    return r.build()


# ------------------------------------------------------------------ library documents
ROUTINE_FIELDS = ("title", "category", "phase", "side", "target", "code_word", "taker", "key_roles",
                  "coaching_points", "trigger", "video_link", "notes")
DRILL_FIELDS = ("title", "focus", "objective", "players", "area", "duration_min", "intensity", "setup",
                "instructions", "coaching_points", "progressions", "video_link")


def routine_doc(board: dict, details: dict) -> dict:
    return {"kind": "routine", "v": 2, "details": {k: details.get(k, "") for k in ROUTINE_FIELDS}, "board": board}


def drill_doc(board: dict, details: dict) -> dict:
    return {"kind": "drill", "v": 2, "details": {k: details.get(k, "") for k in DRILL_FIELDS}, "board": board}


def board_of(doc: dict) -> dict | None:
    b = doc.get("board")
    return clean_doc(b) if doc.get("v") == 2 else None


def summary_text(doc: dict) -> str:
    d = doc.get("details", {})
    if doc.get("kind") == "drill":
        return " · ".join(x for x in (d.get("focus", ""), f"{d.get('players', '')} players" if d.get("players") else "",
                                      f"{d.get('duration_min', '')} min" if d.get("duration_min") else "", d.get("intensity", "")) if x)
    return " · ".join(x for x in (d.get("category", ""), d.get("phase", ""), d.get("side", ""), d.get("target", "")) if x)
