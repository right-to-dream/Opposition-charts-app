"""Set-piece routine & drill designer — pure, on the Tactical Board model (``fap.tactical``).

* :func:`template` — ready multi-step routines (setup → movement → delivery → finish). Every step is a
  Tactical Board FRAME; players keep their ids across frames, so playing the frames IS the animation
  (and the Tactical Board's GIF export animates it).
* :func:`pattern_board` — a board built from the opponent's coded data: every landing point, the
  hot zone, the key numbers, and a counter-setup frame.
* :func:`routine_doc` / :func:`drill_doc` — the library documents (board + coaching details).

Tactical coordinates are 0–100 on both axes over the full pitch (x → attacked goal at 100,
y = 0 top touchline). Box: x ≥ 84.3, y 20.4–79.6 · six-yard box: x ≥ 94.8, y 36.5–63.5 ·
penalty spot x 89.5 · goal y 44.6–55.4.
"""
from __future__ import annotations

from typing import Any

from fap.tactical.models import Board, Frame, TacticalObject, new_board, new_id

# ------------------------------------------------------------------ low-level helpers
def _player(x, y, n, *, team="home", gk=False, role="") -> TacticalObject:
    return TacticalObject(id=new_id(), type="player", x=x, y=y,
                          props={"number": n, "team": team, "role": role, "goalkeeper": gk, "captain": False, "name": ""})


def _ball(x, y) -> TacticalObject:
    return TacticalObject(id=new_id(), type="ball", x=x, y=y)


def _arrow(x, y, x2, y2, kind="arrow") -> TacticalObject:
    return TacticalObject(id=new_id(), type=kind, x=x, y=y, props={"x2": x2, "y2": y2, "_step": True})


def _text(x, y, s, size=11) -> TacticalObject:
    return TacticalObject(id=new_id(), type="text", x=x, y=y, props={"text": s, "size": size, "_step": True})


def _zone(x, y, w, h, label="") -> TacticalObject:
    """Zone from its TOP-LEFT corner + size (the board draws zones from their centre)."""
    return TacticalObject(id=new_id(), type="zone", x=x + w / 2, y=y + h / 2,
                          props={"w": w, "h": h, "label": label, "opacity": 0.22})


def _wrap(s: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(s, width=width) or [s]


def _caption(x: float, y: float, s: str, size: int = 11, width: int = 52, step: float = 3.6) -> list[TacticalObject]:
    """A caption as one centred text object per wrapped line (the board's text is single-line)."""
    return [_text(x, y + i * step, line, size=size) for i, line in enumerate(_wrap(s, width))]


#: x-centre of the visible window per pitch area (captions are centred text)
_CAPTION_X = {"att_third": 83.0, "att_half": 75.0, "def_half": 25.0, "def_third": 17.0, "full": 50.0}


class _Routine:
    """Tiny builder: define players once, then each step moves some of them and draws that step's
    arrows (runs dashed, ball curved). Arrows / captions of a step don't carry to the next step."""

    def __init__(self, name: str, area: str = "att_half"):
        self.board = new_board(name)
        self.board.pitch.area = area
        self.board.frames = []
        self.objs: dict[str, TacticalObject] = {}

    def add(self, key: str, obj: TacticalObject) -> "_Routine":
        self.objs[key] = obj
        return self

    def step(self, name: str, caption: str = "", moves: dict | None = None, runs: tuple = (),
             ball_to: tuple | None = None, ball_from: str | None = None, extra: tuple = (), hold_ms: int = 1100):
        moves = moves or {}
        arrows: list[TacticalObject] = []
        for key, (x2, y2) in moves.items():
            o = self.objs[key]
            if key in runs:
                arrows.append(_arrow(o.x, o.y, x2, y2, kind="dashed_arrow"))
            o.x, o.y = x2, y2
        if ball_to is not None:
            b = self.objs["ball"]
            arrows.append(_arrow(b.x, b.y, ball_to[0], ball_to[1], kind="curved_arrow"))
            b.x, b.y = ball_to
        fr = Frame(id=new_id("frame"), name=name, duration_ms=hold_ms,
                   objects=[o.clone(fresh_id=False) for o in self.objs.values()])
        fr.objects += arrows + list(extra)
        if caption:
            fr.objects += _caption(_CAPTION_X.get(self.board.pitch.area, 50.0), 4, caption,
                                   width=95 if self.board.pitch.area == "full" else 48)
        self.board.frames.append(fr)
        return self

    def done(self, meta: dict | None = None) -> Board:
        self.board.meta.update(meta or {})
        return self.board


def _defenders_box(r: _Routine, scheme: str = "mixed") -> None:
    """Opponent defending a corner from the TOP-right flag: GK + zonal / man markers."""
    r.add("gk", _player(98.5, 50, 1, team="away", gk=True))
    if scheme in ("zonal", "mixed"):
        for k, (x, y) in enumerate([(96.5, 40), (96.5, 50), (96.5, 60), (91, 42), (91, 52), (91, 62)], start=2):
            r.add(f"z{k}", _player(x, y, k, team="away", role="zonal"))
    if scheme in ("man", "mixed"):
        for k, (x, y) in enumerate([(88, 36), (87, 48), (88, 58), (86, 66)], start=8):
            r.add(f"m{k}", _player(x, y, k, team="away", role="marker"))


# ------------------------------------------------------------------ templates
def _corner_near_post_flick() -> Board:
    r = _Routine("Attacking corner — inswing to near post, flick-on")
    r.add("ball", _ball(99.2, 3)).add("taker", _player(99.5, 5, 7, role="taker"))
    r.add("np", _player(84, 32, 9, role="near-post runner")).add("blk", _player(86, 46, 5, role="blocker"))
    r.add("fp", _player(85, 58, 4, role="far-post attacker")).add("pk", _player(82, 50, 10, role="penalty-spot"))
    r.add("edge", _player(76, 44, 8, role="edge of box")).add("rest", _player(62, 50, 6, role="rest defence"))
    _defenders_box(r, "mixed")
    r.step("1 · Setup", "Setup: stack at the penalty spot, near-post runner wide")
    r.step("2 · Movement", "Runs: 9 attacks near post, 5 blocks the zonal, 4 peels to far post",
           moves={"np": (93, 40), "blk": (91, 44), "fp": (94, 62), "pk": (89, 52)}, runs=("np", "blk", "fp", "pk"))
    r.step("3 · Delivery", "Inswinging ball to the near-post zone", ball_to=(93.5, 40))
    r.step("4 · Finish", "9 flicks on → 4 attacks the far post, 8 covers the edge",
           moves={"fp": (96, 58), "edge": (82, 46)}, runs=("fp", "edge"), ball_to=(96, 58))
    return r.done({"category": "Corner", "phase": "Attacking", "target": "Near post"})


def _corner_far_post() -> Board:
    r = _Routine("Attacking corner — outswing to the penalty spot / far post")
    r.add("ball", _ball(99.2, 3)).add("taker", _player(99.5, 5, 11, role="taker"))
    for k, (x, y) in {"a": (80, 40), "b": (80, 46), "c": (80, 52), "d": (80, 58)}.items():
        r.add(k, _player(x, y, {"a": 9, "b": 5, "c": 4, "d": 10}[k], role="runner"))
    r.add("edge", _player(74, 50, 8, role="edge of box")).add("rest", _player(60, 46, 6, role="rest defence"))
    _defenders_box(r, "zonal")
    r.step("1 · Setup", "Setup: 4 in a line at the edge of the box (the 'train')")
    r.step("2 · Movement", "Train splits: near post, spot, far post, back post",
           moves={"a": (92, 40), "b": (89, 50), "c": (92, 58), "d": (94, 66)}, runs=("a", "b", "c", "d"))
    r.step("3 · Delivery", "Outswinging ball to the penalty spot", ball_to=(89.5, 50))
    r.step("4 · Second ball", "Edge player attacks the cleared ball", moves={"edge": (80, 50)}, runs=("edge",))
    return r.done({"category": "Corner", "phase": "Attacking", "target": "Penalty spot"})


def _corner_short() -> Board:
    r = _Routine("Attacking corner — short corner, cut-back")
    r.add("ball", _ball(99.2, 3)).add("taker", _player(99.5, 5, 7, role="taker"))
    r.add("short", _player(90, 8, 8, role="short option")).add("box1", _player(86, 40, 9))
    r.add("box2", _player(86, 52, 5)).add("box3", _player(86, 60, 4)).add("edge", _player(76, 46, 10, role="cut-back target"))
    _defenders_box(r, "zonal")
    r.step("1 · Setup", "Short option 8 comes to the ball")
    r.step("2 · Short pass", "7 → 8, 7 overlaps on the outside", moves={"taker": (96, 10)}, runs=("taker",), ball_to=(90, 8))
    r.step("3 · Dribble", "8 drives to the box edge, box players pin the line",
           moves={"short": (86, 18), "box1": (92, 42), "box2": (92, 52), "box3": (92, 60)}, runs=("short", "box1", "box2", "box3"),
           ball_to=(86.5, 19))
    r.step("4 · Cut-back", "Cut-back to 10 arriving at the edge", moves={"edge": (82, 40)}, runs=("edge",), ball_to=(82, 40))
    return r.done({"category": "Corner", "phase": "Attacking", "target": "Short / cut-back"})


def _defend_corner(scheme: str) -> Board:
    title = {"zonal": "Defending corner — zonal", "man": "Defending corner — man-to-man",
             "mixed": "Defending corner — mixed (zonal + markers)"}[scheme]
    r = _Routine(title)
    r.add("ball", _ball(99.2, 3)).add("taker", _player(99.5, 5, 7, team="away", role="taker"))
    r.add("gk", _player(98.5, 50, 1, gk=True))
    if scheme in ("zonal", "mixed"):
        for k, (x, y) in enumerate([(96.5, 40), (96.5, 50), (96.5, 60), (91, 42), (91, 52), (91, 62)], start=2):
            r.add(f"z{k}", _player(x, y, k, role="zonal"))
    if scheme in ("man", "mixed"):
        for k, (x, y) in enumerate([(86, 38), (85, 48), (86, 58)], start=8):
            r.add(f"m{k}", _player(x, y, k, role="marker"))
    if scheme == "man":
        for k, (x, y) in enumerate([(94, 42), (92, 52), (94, 60), (88, 66)], start=2):
            r.add(f"x{k}", _player(x, y, k, role="marker"))
    r.add("edge", _player(78, 48, 10, role="edge / 2nd ball")).add("out", _player(70, 30, 11, role="counter outlet"))
    for k, (x, y) in enumerate([(84, 40), (83, 50), (84, 60)], start=9):
        r.add(f"a{k}", _player(x, y, k, team="away"))
    r.step("1 · Setup", f"Shape: {scheme} — zones cover near post, six-yard box and penalty spot",
           extra=(_zone(94.8, 36.5, 5.2, 27, "6-yard zone"),))
    r.step("2 · Delivery", "Ball in: attack it in your zone, markers stay goal-side",
           moves={"a9": (93, 41), "a10": (91, 51), "a11": (93, 60)}, runs=("a9", "a10", "a11"), ball_to=(93, 44))
    r.step("3 · Clear & step", "First contact cleared wide; line steps out, 10 wins the 2nd ball",
           moves={"edge": (80, 30)}, runs=("edge",), ball_to=(80, 28))
    r.step("4 · Counter", "Out-ball to 11 on the counter", ball_to=(62, 26), moves={"out": (60, 24)}, runs=("out",))
    return r.done({"category": "Corner", "phase": "Defending", "target": scheme})


def _fk_wide() -> Board:
    r = _Routine("Attacking free kick — wide delivery behind the line", area="att_half")
    r.add("ball", _ball(78, 14)).add("taker", _player(77, 11, 7, role="taker"))
    line = [(78, 34, 9), (78, 42, 5), (78, 50, 4), (78, 58, 10)]
    for x, y, n in line:
        r.add(f"r{n}", _player(x, y, n, role="runner"))
    for k, (x, y) in enumerate([(81, 36), (81, 44), (81, 52), (81, 60)], start=2):
        r.add(f"d{k}", _player(x, y, k, team="away"))
    r.add("gk", _player(98.5, 50, 1, team="away", gk=True))
    r.step("1 · Setup", "Runners level with the defensive line (onside)")
    r.step("2 · Movement", "Staggered runs into the space between line and GK",
           moves={"r9": (92, 38), "r5": (90, 46), "r4": (92, 54), "r10": (88, 62)}, runs=("r9", "r5", "r4", "r10"))
    r.step("3 · Delivery", "Whipped inswinging ball into the corridor", ball_to=(91, 46))
    return r.done({"category": "Free kick", "phase": "Attacking", "target": "Behind the line"})


def _fk_direct() -> Board:
    r = _Routine("Attacking free kick — direct shot with decoy", area="att_half")
    r.add("ball", _ball(76, 48)).add("taker", _player(73, 44, 10, role="shooter")).add("decoy", _player(73, 52, 7, role="decoy"))
    for k, y in enumerate([44, 47, 50, 53], start=2):
        r.add(f"w{k}", _player(83, y, k, team="away", role="wall"))
    r.add("gk", _player(98.5, 49, 1, team="away", gk=True)).add("reb", _player(84, 60, 9, role="rebound"))
    r.step("1 · Setup", "Wall of 4, decoy and shooter both over the ball")
    r.step("2 · Decoy run", "7 runs over the ball", moves={"decoy": (79, 47)}, runs=("decoy",))
    r.step("3 · Shot", "10 shoots over the wall to the far corner; 9 follows for the rebound",
           ball_to=(99.5, 54), moves={"reb": (95, 56)}, runs=("reb",))
    return r.done({"category": "Free kick", "phase": "Attacking", "target": "Direct"})


def _fk_defend() -> Board:
    r = _Routine("Defending wide free kick — line and wall", area="att_half")
    r.add("ball", _ball(78, 14)).add("taker", _player(77, 11, 7, team="away", role="taker"))
    r.add("wall", _player(81, 18, 11, role="wall")).add("gk", _player(97, 50, 1, gk=True))
    for k, y in enumerate([36, 44, 52, 60, 66], start=2):
        r.add(f"l{k}", _player(84, y, k, role="line"))
    for k, y in enumerate([36, 46, 56], start=8):
        r.add(f"a{k}", _player(81, y, k, team="away"))
    r.step("1 · Setup", "Line at the box edge (x≈84), 1-man wall blocks the low cross")
    r.step("2 · Drop", "On the kick: line drops together, track the runners",
           moves={**{f"l{k}": (90, y) for k, y in zip(range(2, 7), [36, 44, 52, 60, 66])},
                  "a8": (91, 38), "a9": (91, 48), "a10": (91, 57)},
           runs=tuple(f"l{k}" for k in range(2, 7)) + ("a8", "a9", "a10"))
    r.step("3 · Attack the ball", "GK claims inside the six-yard box, defenders head outside it", ball_to=(92, 46))
    return r.done({"category": "Free kick", "phase": "Defending", "target": "Wide"})


def _long_throw() -> Board:
    r = _Routine("Long throw-in — near-post flick")
    r.add("ball", _ball(86, 0.5)).add("taker", _player(86, 1.5, 3, role="thrower"))
    r.add("np", _player(88, 30, 9, role="near-post target")).add("fp1", _player(86, 46, 5)).add("fp2", _player(86, 56, 4))
    r.add("edge", _player(76, 40, 8, role="2nd ball"))
    _defenders_box(r, "zonal")
    r.step("1 · Setup", "Target 9 at the near post edge, 5 and 4 ready to attack the flick")
    r.step("2 · Movement", "9 attacks the near post, 5 and 4 crash the six-yard box",
           moves={"np": (93, 38), "fp1": (94, 48), "fp2": (94, 58)}, runs=("np", "fp1", "fp2"))
    r.step("3 · Throw & flick", "Long throw to 9 → flick across goal", ball_to=(93, 38))
    r.step("4 · Finish", "Flick-on to the far side; 8 picks up any clearance", ball_to=(95, 58),
           moves={"edge": (82, 44)}, runs=("edge",))
    return r.done({"category": "Throw-in", "phase": "Attacking", "target": "Near post"})


def _goal_kick_long() -> Board:
    r = _Routine("Goal kick — long to the target, win the second ball", area="full")
    r.add("ball", _ball(5, 50)).add("gk", _player(4, 50, 1, gk=True))
    r.add("t", _player(60, 46, 9, role="target")).add("s1", _player(54, 38, 8, role="2nd ball")).add("s2", _player(54, 58, 6, role="2nd ball"))
    r.add("w1", _player(62, 22, 11, role="runner")).add("w2", _player(62, 78, 7, role="runner"))
    for k, (x, y) in enumerate([(62, 40), (62, 54), (50, 46)], start=4):
        r.add(f"o{k}", _player(x, y, k, team="away"))
    r.step("1 · Setup", "Target 9 between the CBs, 8 and 6 underneath for the second ball")
    r.step("2 · Delivery", "Long ball onto 9", ball_to=(60, 46))
    r.step("3 · Second ball", "9 flicks into the channel; 11 runs in behind, 8 & 6 close the space",
           moves={"w1": (78, 30), "s1": (60, 40), "s2": (60, 54)}, runs=("w1", "s1", "s2"), ball_to=(76, 30))
    return r.done({"category": "Goal kick", "phase": "Attacking", "target": "Long"})


def _penalty() -> Board:
    r = _Routine("Penalty — taker & rebound positions")
    r.add("ball", _ball(89.5, 50)).add("taker", _player(86, 50, 10, role="taker"))
    r.add("gk", _player(100, 50, 1, team="away", gk=True))
    for k, (x, y) in enumerate([(83, 36), (83, 64), (80, 44)], start=7):
        r.add(f"r{k}", _player(x, y, k, role="rebound"))
    for k, (x, y) in enumerate([(83, 40), (83, 60)], start=4):
        r.add(f"d{k}", _player(x, y, k, team="away"))
    r.step("1 · Setup", "Rebounders on the D's edges, inside the defenders")
    r.step("2 · Kick", "Shot to the chosen corner; rebounders attack the six-yard box",
           ball_to=(100, 45.5), moves={"r7": (94, 42), "r8": (94, 58)}, runs=("r7", "r8"))
    return r.done({"category": "Penalty", "phase": "Attacking", "target": ""})


TEMPLATES = {
    "Attacking corner · inswing near post (flick-on)": _corner_near_post_flick,
    "Attacking corner · outswing to spot (train)": _corner_far_post,
    "Attacking corner · short & cut-back": _corner_short,
    "Defending corner · zonal": lambda: _defend_corner("zonal"),
    "Defending corner · man-to-man": lambda: _defend_corner("man"),
    "Defending corner · mixed": lambda: _defend_corner("mixed"),
    "Attacking free kick · wide delivery": _fk_wide,
    "Attacking free kick · direct with decoy": _fk_direct,
    "Defending free kick · wide (line + wall)": _fk_defend,
    "Long throw-in · near-post flick": _long_throw,
    "Goal kick · long + second ball": _goal_kick_long,
    "Penalty · taker & rebounds": _penalty,
}


def template(name: str) -> Board:
    return TEMPLATES[name]()


# ------------------------------------------------------------------ data-driven board
def pattern_board(sd, row: str) -> Board:
    """The opponent's coded set piece on a board: landing points (numbered as in the file), the hot zone,
    their key numbers, and — for corners / free kicks — a counter-setup frame on that zone."""
    from fap.setpiecelab import analytics as A
    g = sd.of(row=row)
    pts = g[g["x"].notna()]
    cat = g["category"].iloc[0] if len(g) else ""
    b = new_board(f"{sd.team} · {row} — their pattern")
    b.pitch.area = "att_half" if cat in ("Corner", "Free kick") and (pts["x"] > 55).all() else "full"
    cx_cap = _CAPTION_X.get(b.pitch.area, 50.0)
    b.frames = []
    fr = Frame(id=new_id("frame"), name="1 · Their landing points", duration_ms=1400)
    out_c = A._col(sd, "Outcome")
    for _, r in pts.iterrows():
        x, y = r["x"] / 105 * 100, r["y"] / 68 * 100
        o = (r[out_c] if out_c else "").lower()
        col = "#e63946" if o in ("goal", "shot", "threat") else "#f4a261" if o else "#ffffff"
        fr.objects.append(TacticalObject(id=new_id(), type="circle", x=x, y=y,
                                         props={"w": 2.4, "h": 3.7, "filled": True, "opacity": 0.9, "color": col,
                                                "shape": "ellipse", "stroke_color": "#111111", "stroke_width": 1.0}))
        fr.objects.append(TacticalObject(id=new_id(), type="text", x=x, y=y - 3.6,
                                         props={"text": str(r["n"]), "size": 8, "color": "#ffffff"}))
    lines = [f"{sd.team} · {row}: {len(g)} coded · {len(pts)} landing points (red = goal/shot/threat)"]
    for name, vc in list(A.breakdown(sd, row).items())[:4]:
        lines.append(f"{name}: " + ", ".join(f"{k} {v}" for k, v in vc.head(3).items()))
    wrapped = [w for line in lines for w in _wrap(line, 48 if b.pitch.area == "att_half" else 95)]
    for i, line in enumerate(wrapped):
        fr.objects.append(_text(cx_cap, 3 + i * 3.6, line, size=10 if b.pitch.area == "att_half" else 11))
    if len(pts) >= 2:
        xs, ys = pts["x"] / 105 * 100, pts["y"] / 68 * 100
        w = max(4.0, float(xs.std() * 2.4)) if len(pts) > 2 else 6.0
        h = max(8.0, float(ys.std() * 2.4)) if len(pts) > 2 else 10.0
        fr.objects.append(TacticalObject(id=new_id(), type="zone", x=float(xs.mean()), y=float(ys.mean()),
                                         props={"w": w, "h": h, "label": "hot zone", "opacity": 0.18}))
    b.frames.append(fr)
    if len(pts) >= 2 and cat in ("Corner", "Free kick"):
        fr2 = fr.clone(name="2 · Our counter-setup")
        cx, cy = float((pts["x"] / 105 * 100).mean()), float((pts["y"] / 68 * 100).mean())
        for k, (dx, dy) in enumerate([(0, -7), (0, 0), (0, 7), (-5, -4), (-5, 4)], start=2):
            fr2.objects.append(_player(min(99.0, cx + dx), cy + dy, k, role="zonal on their hot zone"))
        fr2.objects.append(_player(98.8, 50, 1, gk=True))
        fr2.objects += _caption(cx_cap, 3 + len(wrapped) * 3.6, "Counter-setup: 5 zonal players on their hot zone",
                                size=10 if b.pitch.area == "att_half" else 11)
        b.frames.append(fr2)
    b.meta.update({"category": cat, "phase": "Defending", "source": "data", "opponent": sd.team, "row": row})
    return b


# ------------------------------------------------------------------ library documents
ROUTINE_FIELDS = ("title", "category", "phase", "side", "target", "code_word", "taker", "key_roles",
                  "coaching_points", "trigger", "video_link", "notes")
DRILL_FIELDS = ("title", "focus", "objective", "players", "area", "duration_min", "intensity", "setup",
                "instructions", "coaching_points", "progressions", "video_link")


def routine_doc(board: Board, details: dict) -> dict:
    return {"kind": "routine", "details": {k: details.get(k, "") for k in ROUTINE_FIELDS}, "board": board.to_dict()}


def drill_doc(board: Board, details: dict) -> dict:
    return {"kind": "drill", "details": {k: details.get(k, "") for k in DRILL_FIELDS}, "board": board.to_dict()}


def board_of(doc: dict) -> Board:
    return Board.from_dict(doc.get("board") or {})


def summary_text(doc: dict) -> str:
    d = doc.get("details", {})
    if doc.get("kind") == "drill":
        return " · ".join(x for x in (d.get("focus", ""), f"{d.get('players', '')} players" if d.get("players") else "",
                                      f"{d.get('duration_min', '')} min" if d.get("duration_min") else "", d.get("intensity", "")) if x)
    return " · ".join(x for x in (d.get("category", ""), d.get("phase", ""), d.get("side", ""), d.get("target", "")) if x)
