"""Deeper set-piece analysis on a :class:`~fap.setpiecelab.loader.SetPieceData` (attacking team's view,
goal at x = 105): delivery zones, the delivery → first contact → shot → goal funnel, where shots come from,
first-contact players, second balls, execution × target, takers, match-by-match, and number-backed insights.

Every figure is a count from the file. Rates are only judged (strength / concern) on a minimum sample;
smaller groups are reported as counts with "small sample".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fap.setpiecelab import analytics as A
from fap.setpiecelab.loader import PITCH_L, PITCH_W, RESULT_COL, SetPieceData, is_yes

MIN_N = 5                    # smallest group whose rate is judged
CY = PITCH_W / 2
SIX_X, BOX_X, EDGE_X = PITCH_L - 5.5, PITCH_L - 16.5, PITCH_L - 26.5
SIX_HW, BOX_HW = 9.16, 20.16
ZONE_ORDER = ("6-yd box · near post", "6-yd box · central", "6-yd box · far post", "Box · near side",
              "Penalty spot", "Box · far side", "Edge of the box", "Wide / deep", "Short (not delivered)")


def _pct(a, b) -> float:
    return a / b * 100 if b else np.nan


def _yes(s: pd.Series) -> tuple[int, int]:
    v = s.map(is_yes)
    return int((v == True).sum()), int(v.notna().sum())  # noqa: E712


def _col(sd, name):
    return A._col(sd, name)


# ------------------------------------------------------------------ geometry
def near_sign(sd: SetPieceData) -> dict[str, float]:
    """Side → +1 if that side's corner flag is at y = 68 (near post = y > 34), else −1 — learnt from the file."""
    d = sd.df[sd.df["oy"].notna() & sd.df["side"].isin(["Left", "Right"])] if "oy" in sd.df else sd.df.iloc[0:0]
    if len(d):
        left = d[d["side"] == "Left"]["oy"]
        right = d[d["side"] == "Right"]["oy"]
        lh = bool(left.mean() > CY) if len(left) else not bool(right.mean() > CY)
    else:
        lh = True
    return {"Left": 1.0 if lh else -1.0, "Right": -1.0 if lh else 1.0}


def _sign_of(r: pd.Series, signs: dict) -> float:
    if pd.notna(r.get("oy", np.nan)) and abs(float(r["oy"]) - CY) > 3:
        return 1.0 if float(r["oy"]) > CY else -1.0
    return signs.get(r.get("side", ""), np.nan)


def zone_of(x: float, y: float, sign: float) -> str:
    if not (pd.notna(x) and pd.notna(y)):
        return ""
    d = (y - CY) * (sign if pd.notna(sign) else 1.0)
    if x >= SIX_X and abs(y - CY) <= SIX_HW:
        if pd.isna(sign):
            return "6-yd box · central"
        return "6-yd box · near post" if d > 3.66 else "6-yd box · far post" if d < -3.66 else "6-yd box · central"
    if x >= BOX_X and abs(y - CY) <= BOX_HW:
        if pd.isna(sign):
            return "Penalty spot"
        return "Box · near side" if d > 5.5 else "Box · far side" if d < -5.5 else "Penalty spot"
    if x >= EDGE_X and abs(y - CY) <= BOX_HW:
        return "Edge of the box"
    return "Wide / deep"


def with_zones(sd: SetPieceData, g: pd.DataFrame) -> pd.DataFrame:
    signs = near_sign(sd)
    g = g.copy()
    exe = _col(sd, "Execution")
    z = []
    for _, r in g.iterrows():
        zz = zone_of(r["x"], r["y"], _sign_of(r, signs))
        if not zz and exe and str(r.get(exe, "")).lower() == "short":
            zz = "Short (not delivered)"
        z.append(zz)
    g["zone"] = z
    return g


def zone_rect(zone: str, sign: float) -> tuple[float, float, float, float] | None:
    """(x0, x1, y0, y1) of a zone in the attacking-team frame (for drawing)."""
    s = sign if pd.notna(sign) else 1.0
    def band(a, b):                              # d-band → y-band
        ys = sorted([CY + a * s, CY + b * s])
        return ys[0], ys[1]
    six = {"6-yd box · near post": (3.66, SIX_HW), "6-yd box · central": (-3.66, 3.66), "6-yd box · far post": (-SIX_HW, -3.66)}
    box = {"Box · near side": (5.5, BOX_HW), "Penalty spot": (-5.5, 5.5), "Box · far side": (-BOX_HW, -5.5)}
    if zone in six:
        return (SIX_X, PITCH_L, *band(*six[zone]))
    if zone in box:
        return (BOX_X, SIX_X, *band(*box[zone]))
    if zone == "Edge of the box":
        return (EDGE_X, BOX_X, CY - BOX_HW, CY + BOX_HW)
    return None


# ------------------------------------------------------------------ tables
def zone_table(sd: SetPieceData, g: pd.DataFrame) -> pd.DataFrame:
    g = with_zones(sd, g)
    c1 = _col(sd, "1st contact")
    rows = []
    for z in ZONE_ORDER:
        d = g[g["zone"] == z]
        if d.empty:
            continue
        y, n = _yes(d[c1]) if c1 else (0, 0)
        dang = int(d[RESULT_COL].str.lower().isin(A.DANGER).sum()) if RESULT_COL in d else 0
        rows.append({"Zone": z, "Deliveries": len(d), "1st contact won": y, "1st contact decided": n,
                     "1st contact won %": _pct(y, n), "Shots": int(d["is_shot"].sum()), "Goals": int(d["is_goal"].sum()),
                     "Dangerous": dang, "Danger %": _pct(dang, len(d))})
    return pd.DataFrame(rows)


def shot_source(sd: SetPieceData, r: pd.Series) -> str:
    """Where the shot of a set piece came from: 1st contact, 2nd ball, direct (free kick / penalty) or later."""
    if not r.get("is_shot", False):
        return ""
    if r["category"] == "Penalty" or r.get("variant") == "Direct":
        return "Direct"
    o1, w1 = A.contact_columns(sd, 1)
    o2, _ = A.contact_columns(sd, 2)
    v1 = str(r.get(o1, "")).lower() if o1 else ""
    v2 = str(r.get(o2, "")).lower() if o2 else ""
    if any(k in v1 for k in ("shot", "goal")):
        return "1st contact"
    if any(k in v2 for k in ("shot", "goal")):
        return "2nd ball"
    return "Later in the move"


def shots_table(sd: SetPieceData, g: pd.DataFrame) -> pd.DataFrame:
    s = g[g["is_shot"]].copy()
    if s.empty:
        return pd.DataFrame(columns=["Source", "Shots", "Goals", "On target"])
    s["source"] = [shot_source(sd, r) for _, r in s.iterrows()]
    on = s["shot_outcome"].str.lower().isin(["goal", "saved", "save", "on target"])
    out = s.assign(on=on).groupby("source").agg(Shots=("is_shot", "size"), Goals=("is_goal", "sum"), **{"On target": ("on", "sum")})
    order = ["1st contact", "2nd ball", "Direct", "Later in the move"]
    return out.reindex([o for o in order if o in out.index]).reset_index().rename(columns={"source": "Source"})


def funnel(sd: SetPieceData, g: pd.DataFrame, attacking: bool) -> list[tuple[str, int]]:
    c1 = _col(sd, "1st contact")
    y, n = _yes(g[c1]) if c1 else (0, 0)
    on = int(g["shot_outcome"].str.lower().isin(["goal", "saved", "save", "on target"]).sum())
    if attacking:
        return [("Taken", len(g)), ("1st contact won", y), ("Shot", int(g["is_shot"].sum())), ("On target", on),
                ("Goal", int(g["is_goal"].sum()))]
    return [("Faced", len(g)), ("Opponent won 1st contact", n - y), ("Shot conceded", int(g["is_shot"].sum())),
            ("On target", on), ("Goal conceded", int(g["is_goal"].sum()))]


def players(sd: SetPieceData, g: pd.DataFrame) -> pd.DataFrame:
    """The team's players at the first contact: won (and what followed) / lost (and what followed)."""
    if "fc_won_by" not in g:
        return pd.DataFrame()
    rows = []
    ids = sorted({p for p in pd.concat([g["fc_won_by"], g["fc_lost_by"]]) if p}, key=lambda v: (len(v), v))
    for p in ids:
        w, l_ = g[g["fc_won_by"] == p], g[g["fc_lost_by"] == p]
        rows.append({"Player": f"#{p}", "player": p, "Won": len(w), "Lost": len(l_),
                     "Shots after his win": int(w["is_shot"].sum()), "Goals after his win": int(w["is_goal"].sum()),
                     "Shots after his loss": int(l_["is_shot"].sum()), "Goals after his loss": int(l_["is_goal"].sum()),
                     "Types": ", ".join(f"{k} {v}" for k, v in pd.concat([w, l_])["row"].value_counts().items())})
    out = pd.DataFrame(rows)
    return out.sort_values(["Won", "Lost"], ascending=[False, True]).reset_index(drop=True) if len(out) else out


def second_balls(sd: SetPieceData, g: pd.DataFrame) -> dict:
    c2 = _col(sd, "2nd ball")
    o2, _ = A.contact_columns(sd, 2)
    if not c2:
        return {}
    y, n = _yes(g[c2])
    v = g[c2].map(is_yes)
    won, lost = g[v == True], g[v == False]  # noqa: E712
    vc = lambda d: d[o2][d[o2] != ""].value_counts().to_dict() if o2 else {}  # noqa: E731
    return {"won": y, "decided": n, "won %": _pct(y, n), "after won": vc(won), "after lost": vc(lost),
            "shots after won": int(won["is_shot"].sum()), "shots after lost": int(lost["is_shot"].sum()),
            "goals after won": int(won["is_goal"].sum()), "goals after lost": int(lost["is_goal"].sum())}


def exec_area(sd: SetPieceData, g: pd.DataFrame) -> pd.DataFrame:
    exe, area = _col(sd, "Execution"), _col(sd, "Area")
    if not exe or not area:
        return pd.DataFrame()
    d = g[(g[exe] != "") & (g[area] != "")]
    if d.empty:
        return pd.DataFrame()
    c1 = _col(sd, "1st contact")
    rows = []
    for (e, a), x in d.groupby([exe, area]):
        y, n = _yes(x[c1]) if c1 else (0, 0)
        dang = int(x[RESULT_COL].str.lower().isin(A.DANGER).sum()) if RESULT_COL in x else 0
        rows.append({"Execution": e, "Target": a, "Count": len(x), "1st contact won": f"{y}/{n}" if n else "–",
                     "Shots": int(x["is_shot"].sum()), "Goals": int(x["is_goal"].sum()), "Danger %": _pct(dang, len(x))})
    return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)


def summary(sd: SetPieceData, g: pd.DataFrame) -> dict:
    c1, c2 = _col(sd, "1st contact"), _col(sd, "2nd ball")
    y1, n1 = _yes(g[c1]) if c1 else (0, 0)
    y2, n2 = _yes(g[c2]) if c2 else (0, 0)
    dang = int(g[RESULT_COL].str.lower().isin(A.DANGER).sum()) if RESULT_COL in g else 0
    return {"Count": len(g), "1st contact won": y1, "1st contact decided": n1, "1st contact won %": _pct(y1, n1),
            "2nd ball won": y2, "2nd ball decided": n2, "2nd ball won %": _pct(y2, n2),
            "Shots": int(g["is_shot"].sum()), "Goals": int(g["is_goal"].sum()), "Dangerous": dang,
            "Danger %": _pct(dang, len(g)), "Shots per set piece": (g["is_shot"].sum() / len(g)) if len(g) else np.nan}


def has_matches(sd: SetPieceData) -> bool:
    return "match" in sd.df and (sd.df["match"] != "").any()


def matches(sd: SetPieceData) -> list[str]:
    return list(dict.fromkeys(m for m in sd.df["match"] if m)) if has_matches(sd) else []


def by_match(sd: SetPieceData) -> pd.DataFrame:
    """One row per match × phase: count, contests, shots, goals."""
    if not has_matches(sd):
        return pd.DataFrame()
    rows = []
    for m in matches(sd):
        for ph in ("Attacking", "Defending"):
            g = sd.df[(sd.df["match"] == m) & (sd.df["phase"] == ph)]
            if len(g):
                rows.append({"Match": m, "Phase": ph, **summary(sd, g)})
    return pd.DataFrame(rows)


def groups(sd: SetPieceData) -> list[tuple[str, str, pd.DataFrame]]:
    """(phase, label, rows) for every coded set-piece type, in report order."""
    order = {c: i for i, c in enumerate(A.CATEGORY_ORDER)}
    out = []
    for ph in ("Attacking", "Defending"):
        rows = [r for r in sd.rows if (sd.of(row=r)["phase"] == ph).all()]
        rows.sort(key=lambda r: (order.get(sd.of(row=r)["category"].iloc[0], 99), r))
        for r in rows:
            out.append((ph, r, sd.of(row=r)))
    return out


# ------------------------------------------------------------------ insights
STRENGTH, CONCERN, NOTE = "strength", "concern", "note"


@dataclass
class Insight:
    section: str
    text: str
    tone: str = NOTE


def _f(v, d=0):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def _rate_txt(y, n):
    return f"{y}/{n} ({_f(_pct(y, n))}%)" if n else "not coded"


def _judge(rate, n, good, bad, higher_better=True):
    if n < MIN_N or rate != rate:
        return NOTE
    if higher_better:
        return STRENGTH if rate >= good else CONCERN if rate <= bad else NOTE
    return CONCERN if rate >= bad else STRENGTH if rate <= good else NOTE


def type_label(row: str) -> str:
    return {"Def Throw in": "Long throw-ins against", "Defensive Corner": "Corners against",
            "Offensive Corner": "Corners", "Offensive Free kick": "Indirect free kicks",
            "Direct Free kick": "Direct free kicks", "Defensive Free kick": "Indirect free kicks against",
            "Defensive Direct Free kick": "Direct free kicks against", "Penalty": "Penalties",
            "Penalty Against": "Penalties against", "Long Throw In": "Long throw-ins", "Long GK": "Long goal kicks"}.get(row, row)


def type_insights(sd: SetPieceData, row: str, g: pd.DataFrame | None = None, section: str | None = None) -> list[Insight]:
    """Findings for one set-piece type (optionally a subset ``g`` of it, e.g. one match)."""
    g = sd.of(row=row) if g is None else g
    if g.empty:
        return []
    team = sd.team
    att = g["phase"].iloc[0] == "Attacking"
    sec = section or type_label(row)
    out: list[Insight] = []
    add = lambda t, tone=NOTE: out.append(Insight(sec, t, tone))  # noqa: E731
    sm = summary(sd, g)
    n = len(g)
    small = " (small sample)" if n < MIN_N else ""
    direct = g["category"].iloc[0] == "Penalty" or g["variant"].iloc[0] == "Direct"

    if direct:                                         # penalties / direct free kicks: shots and goals only
        so = g["shot_outcome"][g["shot_outcome"] != ""].value_counts()
        res = ", ".join(f"{k.lower()} {v}" for k, v in so.items()) or "no shot outcome coded"
        if att:
            add(f"{n} taken{small}: {sm['Goals']} goal{'s' if sm['Goals'] != 1 else ''} — {res}.",
                STRENGTH if n and sm["Goals"] / n >= 0.5 else CONCERN if n >= 3 and sm["Goals"] == 0 else NOTE)
        else:
            add(f"{n} faced{small}: {sm['Goals']} conceded — {res}.",
                CONCERN if sm["Goals"] else STRENGTH if n >= 2 else NOTE)
        tk = g["taker"][g["taker"] != ""].value_counts()
        if att and len(tk):
            add("Taker" + ("s" if len(tk) > 1 else "") + ": " + ", ".join(
                f"#{t} {k} ({int(g[g['taker'] == t]['is_goal'].sum())} goal{'s' if int(g[g['taker'] == t]['is_goal'].sum()) != 1 else ''})"
                for t, k in tk.items()) + ".")
        gp = g[g["goal_x_m"].notna()] if "goal_x_m" in g else g.iloc[0:0]
        if len(gp):
            low = int((gp["goal_y_m"] <= 2.44 / 2).sum())
            add(f"Placement in the goal ({len(gp)} located): "
                + ", ".join(f"{r['goal_x_m']:.1f} m across / {r['goal_y_m']:.1f} m high ({str(r['shot_outcome']).lower() or 'no outcome'})"
                            for _, r in gp.iterrows())
                + f" · {low} low, {len(gp) - low} high (goal 7.32 × 2.44 m).")
        return out

    # volume & how
    exe, area = _col(sd, "Execution"), _col(sd, "Area")
    bits = [f"{n} {'taken' if att else 'faced'}{small}"]
    if (g["side"] != "").any():
        bits.append(" / ".join(f"{s} {int((g['side'] == s).sum())}" for s in ("Left", "Right") if (g["side"] == s).any()))
    if exe and (g[exe] != "").any():
        bits.append("delivery: " + ", ".join(f"{k.lower()} {v}" for k, v in g[exe][g[exe] != ""].value_counts().items()))
    if area and (g[area] != "").any():
        bits.append("target: " + ", ".join(f"{k.lower()} {v}" for k, v in g[area][g[area] != ""].value_counts().items()))
    add(" · ".join(bits) + ".")

    # first contact
    if sm["1st contact decided"]:
        r1 = sm["1st contact won %"]
        if att:
            add(f"{team} won the 1st contact in {_rate_txt(sm['1st contact won'], sm['1st contact decided'])}.",
                _judge(r1, sm["1st contact decided"], 50, 30))
        else:
            add(f"{team} won the 1st contact in {_rate_txt(sm['1st contact won'], sm['1st contact decided'])} — "
                f"the opponent won it {sm['1st contact decided'] - sm['1st contact won']} time(s).",
                _judge(r1, sm["1st contact decided"], 70, 50))
    # outcome / danger
    shots = shots_table(sd, g)
    src = ", ".join(f"{r['Source'].lower()} {int(r['Shots'])}" for _, r in shots.iterrows())
    if att:
        add(f"Danger created in {sm['Dangerous']}/{n} ({_f(sm['Danger %'])}%) · {sm['Shots']} shot{'s' if sm['Shots'] != 1 else ''}"
            + (f" ({src})" if src else "") + f" · {sm['Goals']} goal{'s' if sm['Goals'] != 1 else ''}.",
            _judge(sm["Danger %"], n, 35, 15))
    else:
        add(f"Danger conceded in {sm['Dangerous']}/{n} ({_f(sm['Danger %'])}%) · {sm['Shots']} shot{'s' if sm['Shots'] != 1 else ''} against"
            + (f" ({src})" if src else "") + f" · {sm['Goals']} goal{'s' if sm['Goals'] != 1 else ''} conceded.",
            _judge(sm["Danger %"], n, 15, 30, higher_better=False))
    # zones
    zt = zone_table(sd, g)
    zt = zt[zt["Zone"] != "Short (not delivered)"] if len(zt) else zt
    if len(zt) and zt["Deliveries"].sum() >= MIN_N:
        tot = int(zt["Deliveries"].sum())
        mx = zt["Deliveries"].max()
        tops = zt[zt["Deliveries"] == mx]
        if len(tops) <= 2:
            add("Most deliveries land in the " + " and the ".join(
                f"{r['Zone'].lower()} ({int(r['Deliveries'])}/{tot}"
                + (f", 1st contact won {int(r['1st contact won'])}/{int(r['1st contact decided'])}" if r["1st contact decided"] else "") + ")"
                for _, r in tops.iterrows()) + " — zones from the x / y landing point.")
        else:
            add(f"Deliveries are spread: {len(tops)} zones with {int(mx)} each — "
                + ", ".join(f"{r['Zone'].lower()} {int(r['Deliveries'])}" for _, r in zt.iterrows()) + f" (of {tot} with a landing point).")
        dz = zt[zt["Dangerous"] > 0].sort_values(["Dangerous", "Deliveries"], ascending=False)
        if len(dz):
            add(("Danger came from: " if att else "Danger against came from: ")
                + ", ".join(f"{r['Zone'].lower()} {int(r['Dangerous'])}/{int(r['Deliveries'])}" for _, r in dz.iterrows()) + ".",
                NOTE)
    # execution / target combos (judge only n ≥ MIN_N)
    ea = exec_area(sd, g)
    if len(ea) >= 2:
        big = ea[ea["Count"] >= 4]
        if len(big) >= 2:
            b = big.sort_values(["Danger %", "Count"], ascending=False).iloc[0]
            w = big.sort_values(["Danger %", "Count"], ascending=[True, False]).iloc[0]
            if b["Danger %"] != w["Danger %"]:
                add(f"Most dangerous{' for the opponent' if not att else ''}: {b['Execution'].lower()} to the {b['Target'].lower()} "
                    f"(danger in {_f(b['Danger %'])}% of {int(b['Count'])}) · least: {w['Execution'].lower()} to the "
                    f"{w['Target'].lower()} ({_f(w['Danger %'])}% of {int(w['Count'])}) — patterns with 4+ coded.")
    if exe and att:
        sh = g[g[exe].str.lower() == "short"]
        if len(sh) >= 3:
            add(f"Short {type_label(row).lower()}: {len(sh)} — {int(sh['is_shot'].sum())} shot(s), {int(sh['is_goal'].sum())} goal(s), "
                f"danger {int(sh[RESULT_COL].str.lower().isin(A.DANGER).sum())}/{len(sh)}.")
    # sides
    if (g["side"] == "Left").sum() >= MIN_N and (g["side"] == "Right").sum() >= MIN_N:
        L, R = summary(sd, g[g["side"] == "Left"]), summary(sd, g[g["side"] == "Right"])
        add(f"Left vs right: 1st contact won {_rate_txt(L['1st contact won'], L['1st contact decided'])} vs "
            f"{_rate_txt(R['1st contact won'], R['1st contact decided'])} · danger {L['Dangerous']}/{L['Count']} vs "
            f"{R['Dangerous']}/{R['Count']} · goals {L['Goals']} vs {R['Goals']}.")
    # second balls
    sb = second_balls(sd, g)
    if sb.get("decided"):
        line = f"2nd ball won {_rate_txt(sb['won'], sb['decided'])}"
        if att:
            line += f" — {sb['shots after won']} shot(s) after winning it."
            add(line, _judge(sb["won %"], sb["decided"], 50, 35))
        else:
            line += f" — {sb['shots after lost']} shot(s) and {sb['goals after lost']} goal(s) conceded after losing it."
            add(line, CONCERN if sb["goals after lost"] else _judge(sb["won %"], sb["decided"], 60, 40))
    # players
    pl = players(sd, g)
    if len(pl):
        if att:
            w = pl[pl["Won"] > 0].head(3)
            if len(w):
                add("Main 1st-contact targets: " + ", ".join(
                    f"{r['Player']} won {int(r['Won'])} ({int(r['Shots after his win'])} shot{'s' if r['Shots after his win'] != 1 else ''}"
                    + (f", {int(r['Goals after his win'])} goal" if r["Goals after his win"] else "") + ")" for _, r in w.iterrows()) + ".")
        else:
            w = pl[pl["Won"] > 0].head(3)
            if len(w):
                add("Most 1st contacts won in defence: " + ", ".join(f"{r['Player']} {int(r['Won'])}" for _, r in w.iterrows()) + ".",
                    NOTE)
            lo = pl[pl["Lost"] > 0].sort_values(["Lost", "Goals after his loss"], ascending=False)
            if len(lo):
                add("1st contact lost by: " + ", ".join(
                    f"{r['Player']} {int(r['Lost'])}" + (f" ({int(r['Shots after his loss'])} shot, {int(r['Goals after his loss'])} goal against)"
                                                        if r["Shots after his loss"] else "") for _, r in lo.iterrows()) + ".",
                    CONCERN if lo["Goals after his loss"].sum() else NOTE)
    # takers
    if att:
        tk = A.takers(sd, g)
        if len(tk):
            add("Takers: " + ", ".join(f"#{r['taker']} {int(r['Taken'])} (danger {int(r.get('Dangerous', 0))})" for _, r in tk.head(4).iterrows()) + ".")
    # goals: how each one happened
    for _, r in g[g["is_goal"]].iterrows():
        o1, _w = A.contact_columns(sd, 1)
        o2, _w2 = A.contact_columns(sd, 2)
        c1, c2 = _col(sd, "1st contact"), _col(sd, "2nd ball")
        steps = []
        if c1 and r.get(c1):
            steps.append(f"1st contact {'won' if is_yes(r[c1]) else 'lost'}" + (f" by #{r['fc_won_by'] or r['fc_lost_by']}" if (r.get('fc_won_by') or r.get('fc_lost_by')) else "")
                         + (f" → {str(r[o1]).lower()}" if o1 and r.get(o1) else ""))
        if c2 and r.get(c2):
            steps.append(f"2nd ball {'won' if is_yes(r[c2]) else 'lost'}" + (f" → {str(r[o2]).lower()}" if o2 and r.get(o2) else ""))
        where = []
        if area and r.get(area):
            where.append(str(r[area]).lower())
        if r.get("side"):
            where.append(f"from the {str(r['side']).lower()}")
        add(f"Goal {'scored' if att else 'conceded'} ({r['instance']}{', ' + ' '.join(where) if where else ''}): "
            + (" · ".join(steps) if steps else "no contact detail coded") + ".", STRENGTH if att else CONCERN)
    if (g["notes"] != "").any():
        for _, r in g[g["notes"] != ""].iterrows():
            add(f"Analyst note ({r['instance']}): {r['notes']}")
    return out


def headline_insights(sd: SetPieceData) -> list[Insight]:
    out: list[Insight] = []
    team = sd.team
    a, d = sd.df[sd.df["phase"] == "Attacking"], sd.df[sd.df["phase"] == "Defending"]
    sa, sdf = summary(sd, a), summary(sd, d)
    if len(a):
        out.append(Insight("Headline", f"{team} took {len(a)} set pieces: {sa['Shots']} shots, {sa['Goals']} goal{'s' if sa['Goals'] != 1 else ''} "
                                       f"(1 goal every {len(a) / sa['Goals']:.0f})." if sa["Goals"] else
                           f"{team} took {len(a)} set pieces: {sa['Shots']} shots, no goal.",
                           STRENGTH if sa["Goals"] else CONCERN))
    if len(d):
        out.append(Insight("Headline", f"{team} faced {len(d)} set pieces: {sdf['Shots']} shots and {sdf['Goals']} goal{'s' if sdf['Goals'] != 1 else ''} conceded.",
                           CONCERN if sdf["Goals"] else STRENGTH))
    if sa["1st contact decided"] and sdf["1st contact decided"]:
        out.append(Insight("Headline", f"1st contact: {team} won {_rate_txt(sa['1st contact won'], sa['1st contact decided'])} "
                                       f"when taking, {_rate_txt(sdf['1st contact won'], sdf['1st contact decided'])} when defending."))
    if sa["2nd ball decided"] and sdf["2nd ball decided"]:
        out.append(Insight("Headline", f"2nd ball: {team} won {_rate_txt(sa['2nd ball won'], sa['2nd ball decided'])} "
                                       f"when taking, {_rate_txt(sdf['2nd ball won'], sdf['2nd ball decided'])} when defending."))
    ov = [(r, summary(sd, sd.of(row=r))) for ph, r, _ in groups(sd)]
    att = [(r, s) for r, s in ov if sd.of(row=r)["phase"].iloc[0] == "Attacking" and s["Count"] >= MIN_N]
    de = [(r, s) for r, s in ov if sd.of(row=r)["phase"].iloc[0] == "Defending" and s["Count"] >= MIN_N]
    if att:
        r, s = max(att, key=lambda t: (t[1]["Danger %"], t[1]["Goals"]))
        out.append(Insight("Headline", f"Most dangerous weapon: {type_label(r).lower()} — danger in {s['Dangerous']}/{s['Count']} "
                                       f"({_f(s['Danger %'])}%), {s['Shots']} shots, {s['Goals']} goal(s).", STRENGTH))
    if de:
        r, s = max(de, key=lambda t: (t[1]["Danger %"], t[1]["Goals"]))
        out.append(Insight("Headline", f"Biggest defensive risk: {type_label(r).lower()} — danger conceded in {s['Dangerous']}/{s['Count']} "
                                       f"({_f(s['Danger %'])}%), {s['Shots']} shots, {s['Goals']} goal(s).", CONCERN))
    goals_against = d[d["is_goal"]]
    if len(goals_against):
        out.append(Insight("Headline", "Goals conceded from: " + ", ".join(
            f"{type_label(k).lower()} {v}" for k, v in goals_against["row"].value_counts().items()) + ".", CONCERN))
    if has_matches(sd):
        bm = by_match(sd)
        ag = bm[bm["Phase"] == "Attacking"]
        if len(ag) >= 2:
            b = ag.sort_values(["Goals", "Shots"], ascending=False).iloc[0]
            out.append(Insight("Headline", f"Best attacking set-piece match: {b['Match']} — {b['Shots']} shots, {b['Goals']} goal(s) "
                                           f"from {b['Count']}."))
        dg = bm[bm["Phase"] == "Defending"]
        if len(dg) >= 2:
            w = dg.sort_values(["Goals", "Shots"], ascending=False).iloc[0]
            if w["Shots"]:
                out.append(Insight("Headline", f"Toughest defensive match: {w['Match']} — {w['Shots']} shots and {w['Goals']} goal(s) "
                                               f"conceded from {w['Count']}.", CONCERN if w["Goals"] else NOTE))
    return out


def all_insights(sd: SetPieceData) -> list[Insight]:
    out = headline_insights(sd)
    for ph, r, g in groups(sd):
        out.extend(type_insights(sd, r, g, section=f"{ph} · {type_label(r)}"))
    return out


def match_insights(sd: SetPieceData, match: str) -> list[Insight]:
    """Compact findings for one match: per phase the volume, contests, shots / goals, and how every goal happened."""
    sub = sd.subset(match=match)
    team, out = sd.team, []
    for ph in ("Attacking", "Defending"):
        g = sub.df[sub.df["phase"] == ph]
        if g.empty:
            continue
        att = ph == "Attacking"
        s = summary(sub, g)
        types = ", ".join(f"{type_label(k).lower()} {v}" for k, v in g["row"].value_counts().items())
        bits = [f"{s['Count']} {'taken' if att else 'faced'} ({types})"]
        if s["1st contact decided"]:
            bits.append(f"1st contact won {_rate_txt(s['1st contact won'], s['1st contact decided'])}")
        if s["2nd ball decided"]:
            bits.append(f"2nd ball won {_rate_txt(s['2nd ball won'], s['2nd ball decided'])}")
        bits.append(f"{s['Shots']} shot{'s' if s['Shots'] != 1 else ''} {'' if att else 'against'}".strip()
                    + f" · {s['Goals']} goal{'s' if s['Goals'] != 1 else ''}{'' if att else ' conceded'}")
        tone = (STRENGTH if s["Goals"] else NOTE) if att else (CONCERN if s["Goals"] else NOTE)
        out.append(Insight(f"{match} · {'Taking' if att else 'Defending'}", " · ".join(bits) + ".", tone))
        for _, r in g[g["is_goal"]].iterrows():
            gi = [i for i in type_insights(sub, r["row"], g[g["instance"] == r["instance"]], section="x")
                  if i.text.startswith("Goal ")]
            for i in gi:
                out.append(Insight(f"{match} · {'Taking' if att else 'Defending'}", i.text, i.tone))
    return out
