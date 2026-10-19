"""Set-piece analytics on a :class:`~fap.setpiecelab.loader.SetPieceData` — counts, contest rates,
qualifier breakdowns, cross-tabs, per-category (corners / free kicks / throw-ins …) attacking-vs-defending
comparison, second-ball zones and deterministic insights (numbers only, nothing invented)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fap.setpiecelab.loader import RESULT_COL, SetPieceData, is_yes

DANGER = {"goal", "shot", "threat"}
CONTESTS = ("1st contact", "2nd ball")
# "Yes" in a contest column = the file's team won it — whether it takes or defends the set piece.
CATEGORY_ORDER = ("Corner", "Free kick", "Throw-in", "Goal kick", "Penalty", "Kick-off", "Other")
CATEGORY_PLURAL = {"Corner": "Corners", "Free kick": "Free kicks", "Throw-in": "Throw-ins", "Goal kick": "Goal kicks",
                   "Penalty": "Penalties", "Kick-off": "Kick-offs", "Other": "Other"}
BOX_X, BOX_Y = 105 - 16.5, (34 - 20.16, 34 + 20.16)


def _rate(series: pd.Series) -> tuple[int, int]:
    """(yes, decided) for a Yes/No qualifier column."""
    v = series.map(is_yes)
    return int((v == True).sum()), int(v.notna().sum())  # noqa: E712


def _col(sd: SetPieceData, name: str) -> str | None:
    if name.lower() == "outcome" and RESULT_COL in sd.df:      # the combined result replaces "Outcome"
        return RESULT_COL
    for q in sd.qualifiers:
        if q.strip().lower() == name.lower():
            return q
    return None


def result_col(sd: SetPieceData) -> str | None:
    return RESULT_COL if RESULT_COL in sd.df else None


def contest_label(sd: SetPieceData, v: str) -> str:
    yn = is_yes(v)
    return f"{sd.team} won" if yn is True else "Opponent won" if yn is False else v


def _pct(a: float, b: float) -> float:
    return a / b * 100 if b else np.nan


def _main(g: pd.DataFrame, col: str | None) -> tuple[str, int, int]:
    if not col:
        return "", 0, 0
    vc = g[col][g[col] != ""].value_counts()
    return (vc.index[0], int(vc.iloc[0]), int(vc.sum())) if len(vc) else ("", 0, 0)


def summarise(sd: SetPieceData, g: pd.DataFrame) -> dict:
    """Key numbers for any group of set pieces."""
    rec = {"Count": len(g), "With landing point": int(g["x"].notna().sum()),
           "With second-ball point": int(g["sb_x"].notna().sum()) if "sb_x" in g else 0}
    for c in CONTESTS:
        col = _col(sd, c)
        if col:
            y, n = _rate(g[col])
            rec[f"{c} won"], rec[f"{c} decided"] = y, n
    eff = _col(sd, "Efficiency")
    if eff:
        e = g[eff].str.lower()
        rec["Efficient"] = int((e.str.contains("success") & ~e.str.contains("unsuccess")).sum())
        rec["Efficiency coded"] = int((e != "").sum())
    rc = result_col(sd)
    if rc:
        o = g[rc].str.lower()
        rec["Goals"] = int((o == "goal").sum())
        rec["Shots"] = int((o == "shot").sum())
        rec["Dangerous (goal/shot/threat)"] = int(o.isin(DANGER).sum())
        rec["Result coded"] = int((o != "").sum())
    for name, lab in (("Execution", "Main execution"), ("Area", "Main target"), ("Location", "Main side")):
        v, k, n = _main(g, _col(sd, name))
        rec[lab] = f"{v} ({k}/{n})" if n else "–"
    return rec


def overview(sd: SetPieceData) -> pd.DataFrame:
    rows = []
    for r in sd.rows:
        g = sd.of(row=r)
        rows.append({"Set piece": r, "Category": g["category"].iloc[0], "Phase": g["phase"].iloc[0], **summarise(sd, g)})
    return pd.DataFrame(rows)


def categories(sd: SetPieceData) -> list[str]:
    have = set(sd.df["category"])
    return [c for c in CATEGORY_ORDER if c in have]


def category_table(sd: SetPieceData, cat: str) -> pd.DataFrame:
    """Rows = Attacking / Defending of one category (+ the row types they cover)."""
    rows = []
    for ph in ("Attacking", "Defending"):
        g = sd.of(category=cat, phase=ph)
        if g.empty:
            continue
        rows.append({"Phase": ph, "Coded as": ", ".join(sorted(g["row"].unique())), **summarise(sd, g)})
    return pd.DataFrame(rows)


def breakdown_of(sd: SetPieceData, g: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    contests = {c.lower() for c in CONTESTS}
    for q in sd.qualifiers:
        if q not in g:
            continue
        vals = g[q][g[q] != ""]
        if "outcome" in q.lower():                       # "Threat, Shot" counts as both
            vals = vals.str.split(r"\s*[,;]\s*").explode()
            vals = vals[vals != ""]
        vc = vals.value_counts()
        if len(vc):
            if q.strip().lower() in contests:            # Yes / No → who won it
                vc = vc.rename(lambda v: contest_label(sd, v))
                vc = vc.groupby(level=0).sum().sort_values(ascending=False)
            out[q] = vc
    return out


def breakdown(sd: SetPieceData, row: str) -> dict[str, pd.Series]:
    return breakdown_of(sd, sd.of(row=row))


def phase_compare(sd: SetPieceData, cat: str, quals: list[str] | None = None) -> pd.DataFrame:
    """Long table (qualifier, value, phase, count, share %) — attacking vs defending of one category."""
    rows = []
    for ph in ("Attacking", "Defending"):
        g = sd.of(category=cat, phase=ph)
        for q, vc in breakdown_of(sd, g).items():
            if quals and q not in quals:
                continue
            tot = vc.sum()
            for v, k in vc.items():
                rows.append({"qualifier": q, "value": v, "phase": ph, "count": int(k), "share": k / tot * 100, "of": int(tot)})
    return pd.DataFrame(rows)


def crosstab(sd: SetPieceData, row: str, a: str, b: str) -> pd.DataFrame:
    g = sd.of(row=row)
    g = g[(g[a] != "") & (g[b] != "")]
    return pd.crosstab(g[a], g[b]) if len(g) else pd.DataFrame()


def second_ball_zone(x: float, y: float) -> str:
    if np.isnan(x):
        return ""
    if x >= BOX_X and BOX_Y[0] <= y <= BOX_Y[1]:
        return "Inside the box"
    if x >= BOX_X - 10 and BOX_Y[0] <= y <= BOX_Y[1]:
        return "Edge of the box"
    if x >= BOX_X - 10:
        return "Wide of the box"
    return "Deeper"


def _sentences(sd: SetPieceData, g: pd.DataFrame, att: bool) -> list[str]:
    team, out = sd.team, []
    rc = result_col(sd)
    exe, area, loc = (_main(g, _col(sd, n)) for n in ("Execution", "Area", "Location"))
    if exe[2]:
        out.append(f"delivery mostly {exe[0].lower()} ({exe[1]}/{exe[2]})")
    if area[2]:
        out.append(f"aimed at the {area[0].lower()} ({area[1]}/{area[2]})")
    if loc[2] >= 3:
        out.append(f"from the {loc[0].lower()} {loc[1]}/{loc[2]}")
    for c in CONTESTS:
        col = _col(sd, c)
        if col:
            y, n = _rate(g[col])
            if n:
                out.append(f"{c} won by {team} {y}/{n} ({y / n * 100:.0f}%)")
    if rc:
        o = g[rc][g[rc] != ""]
        if len(o):
            vc = o.value_counts()
            out.append("results: " + ", ".join(f"{k.lower()} {v}" for k, v in vc.items()))
    return out


def insights(sd: SetPieceData) -> list[tuple[str, str]]:
    """(set piece, sentence) — every sentence is a count from the file."""
    team = sd.team
    res = []
    for r in sd.rows:
        g = sd.of(row=r)
        att = g["phase"].iloc[0] == "Attacking"
        who = f"{team}" if att else f"Against {team}"
        res.append((r, f"{who} · " + " · ".join([f"{len(g)} coded"] + _sentences(sd, g, att))))
    ov = overview(sd)
    if "Dangerous (goal/shot/threat)" in ov:
        att = ov[ov["Phase"] == "Attacking"].sort_values(["Dangerous (goal/shot/threat)", "Goals"], ascending=False)
        if len(att) and att["Dangerous (goal/shot/threat)"].iloc[0] > 0:
            r0 = att.iloc[0]
            res.insert(0, ("Headline", f"{team}'s most dangerous weapon: {r0['Set piece']} — "
                                        f"{int(r0['Dangerous (goal/shot/threat)'])} dangerous results from {int(r0['Count'])}"
                                        f" ({int(r0.get('Goals', 0))} goals)."))
        de = ov[ov["Phase"] == "Defending"].sort_values(["Dangerous (goal/shot/threat)", "Goals"], ascending=False)
        if len(de) and de["Dangerous (goal/shot/threat)"].iloc[0] > 0:
            r0 = de.iloc[0]
            res.insert(1, ("Headline", f"Where {team} concede danger: {r0['Set piece']} — "
                                        f"{int(r0['Dangerous (goal/shot/threat)'])} dangerous results against them from "
                                        f"{int(r0['Count'])} ({int(r0.get('Goals', 0))} goals)."))
    return res


def category_insights(sd: SetPieceData, cat: str) -> list[str]:
    """Plain sentences for one set-piece category, attacking vs defending."""
    team, out = sd.team, []
    rc = result_col(sd)
    a, d = sd.of(category=cat, phase="Attacking"), sd.of(category=cat, phase="Defending")
    plural = CATEGORY_PLURAL.get(cat, cat).lower()
    if len(a):
        out.append(f"**{team} taking {plural}** ({len(a)}): " + " · ".join(_sentences(sd, a, True)) + ".")
    if len(d):
        out.append(f"**{team} defending {plural}** ({len(d)}): " + " · ".join(_sentences(sd, d, False)) + ".")
    for c in CONTESTS:
        col = _col(sd, c)
        if col and len(a) and len(d):
            ya, na = _rate(a[col])
            yd, nd = _rate(d[col])
            if na and nd:
                out.append(f"{c}: {team} win it {_pct(ya, na):.0f}% when taking {plural} vs {_pct(yd, nd):.0f}% when "
                           f"defending them.")
    if rc:
        for g, who in ((a, f"{team} created"), (d, f"{team} conceded")):
            o = g[rc].str.lower()
            k = int(o.isin(DANGER).sum())
            if len(g) and (o != "").any():
                goals = int((o == "goal").sum())
                out.append(f"{who} danger from {k}/{len(g)} {plural} ({_pct(k, len(g)):.0f}%)"
                           + (f", {goals} goal{'s' if goals != 1 else ''}" if goals else "") + ".")
        area = _col(sd, "Area")
        if area:
            for g, who in ((a, "created"), (d, "conceded")):
                dg = g[g[rc].str.lower().isin(DANGER) & (g[area] != "")]
                if len(dg) >= 2:
                    vc = dg[area].value_counts()
                    out.append(f"Danger {who} mostly came from deliveries to the {vc.index[0].lower()} "
                               f"({int(vc.iloc[0])}/{len(dg)}).")
    sb = sd.of(category=cat)
    sb = sb[sb["sb_x"].notna()] if "sb_x" in sb else sb.iloc[0:0]
    if len(sb):
        z = pd.Series([second_ball_zone(x, y) for x, y in zip(sb["sb_x"], sb["sb_y"])]).value_counts()
        out.append(f"Second balls (located in {len(sb)}): mostly {z.index[0].lower()} ({int(z.iloc[0])}/{len(sb)}) — "
                   + ", ".join(f"{k.lower()} {v}" for k, v in z.items()) + ".")
    return out


# ------------------------------------------------------------------ contact maps
OUTCOME_COLORS = {"goal": "#e5484d", "shot": "#f97316", "threat": "#f5b301", "cross": "#a855f7", "recycle": "#2ed8a3",
                  "gk punch": "#14b8a6", "gk claim": "#14b8a6", "clearance": "#3ab0ff"}


def contact_columns(sd: SetPieceData, which: int) -> tuple[str | None, str | None]:
    """(outcome column, won Yes/No column) of the 1st contact (which=1) or the 2nd ball / contact (which=2)."""
    tag = ("1st", "first") if which == 1 else ("2nd", "second")
    outc = next((q for q in sd.qualifiers if "outcome" in q.lower() and any(t in q.lower() for t in tag)), None)
    won = _col(sd, CONTESTS[0] if which == 1 else CONTESTS[1])
    return outc, won


def contact_points(sd: SetPieceData, row: str, which: int) -> pd.DataFrame:
    """Instances with a location for that contact: 1st contact = landing point, 2nd = second-ball point."""
    g = sd.of(row=row).copy()
    if which == 1:
        g["cx"], g["cy"] = g["x"], g["y"]
    else:
        if "sb_x" not in g:
            return g.iloc[0:0]
        g["cx"], g["cy"] = g["sb_x"], g["sb_y"]
    return g[g["cx"].notna()]


def contact_rows(sd: SetPieceData, which: int, category: str | None = None) -> list[str]:
    rows = [r for r in sd.rows if category is None or sd.of(row=r)["category"].iloc[0] == category]
    return [r for r in rows if len(contact_points(sd, r, which))]


def row_insights(sd: SetPieceData, row: str) -> list[str]:
    """Findings for one coded set-piece type (e.g. Offensive Corner), every number from the file."""
    team, out = sd.team, []
    g = sd.of(row=row)
    if g.empty:
        return out
    att = g["phase"].iloc[0] == "Attacking"
    rc = result_col(sd)
    out.append(f"{len(g)} {row.lower()}{'s' if len(g) != 1 else ''} coded — {team} {'taking' if att else 'defending'} them.")
    exe, area, loc = (_main(g, _col(sd, n)) for n in ("Execution", "Area", "Location"))
    if exe[2]:
        vc = g[_col(sd, "Execution")][g[_col(sd, "Execution")] != ""].value_counts()
        out.append("Delivery: " + ", ".join(f"{k.lower()} {v}" for k, v in vc.items()) + f" (of {exe[2]}).")
    if area[2]:
        vc = g[_col(sd, "Area")][g[_col(sd, "Area")] != ""].value_counts()
        out.append("Target area: " + ", ".join(f"{k.lower()} {v}" for k, v in vc.items()) + f" (of {area[2]}).")
    if loc[2]:
        vc = g[_col(sd, "Location")][g[_col(sd, "Location")] != ""].value_counts()
        out.append("Side: " + ", ".join(f"{k.lower()} {v}" for k, v in vc.items()) + f" (of {loc[2]}).")
    for which, c in ((1, CONTESTS[0]), (2, CONTESTS[1])):
        col = _col(sd, c)
        outc = contact_columns(sd, which)[0]
        if col:
            y, n = _rate(g[col])
            if n:
                line = f"{c}: {team} won {y}/{n} ({y / n * 100:.0f}%)"
                if outc:
                    vc = g[outc][g[outc] != ""].str.split(r"\s*[,;]\s*").explode().value_counts()
                    if len(vc):
                        line += " — outcome: " + ", ".join(f"{k.lower()} {v}" for k, v in vc.items())
                out.append(line + ".")
    if rc:
        o = g[rc].str.lower()
        k = int(o.isin(DANGER).sum())
        goals = int((o == "goal").sum())
        out.append(f"Danger {'created' if att else 'conceded'}: {k}/{len(g)} ({_pct(k, len(g)):.0f}%)"
                   + (f" · {goals} goal{'s' if goals != 1 else ''}" if goals else "")
                   + (" · overall results: " + ", ".join(f"{a.lower()} {b}" for a, b in g[rc][g[rc] != ""].value_counts().items())
                      if (g[rc] != "").any() else "") + ".")
        ar = _col(sd, "Area")
        if ar:
            dg = g[o.isin(DANGER) & (g[ar] != "")]
            if len(dg):
                vc = dg[ar].value_counts()
                out.append(f"Dangerous ones went to: " + ", ".join(f"{a.lower()} {b}" for a, b in vc.items()) + ".")
    if g["x"].notna().any():
        p = g[g["x"].notna()]
        out.append(f"Landing point coded for {len(p)}/{len(g)} — average x {p['x'].mean():.0f} m, y {p['y'].mean():.0f} m.")
    if "sb_x" in g and g["sb_x"].notna().any():
        sb = g[g["sb_x"].notna()]
        z = pd.Series([second_ball_zone(x, y) for x, y in zip(sb["sb_x"], sb["sb_y"])]).value_counts()
        out.append(f"Second balls located in {len(sb)}: " + ", ".join(f"{a.lower()} {b}" for a, b in z.items()) + ".")
    return out


# ------------------------------------------------------------------ takers, sides, groups
def takers(sd: SetPieceData, g: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per taker: set pieces taken, by type, 1st contact / 2nd ball won by the team, danger, goals."""
    g = sd.df if g is None else g
    g = g[g["taker"].astype(str) != ""] if "taker" in g else g.iloc[0:0]
    rows = []
    rc = result_col(sd)
    c1, c2 = _col(sd, CONTESTS[0]), _col(sd, CONTESTS[1])
    for t, d in g.groupby("taker"):
        rec = {"Taker": f"Player #{t}", "taker": str(t), "Taken": len(d),
               "Types": ", ".join(f"{k} {v}" for k, v in d["row"].value_counts().items())}
        for col, lab in ((c1, "1st contact"), (c2, "2nd ball")):
            if col:
                y, n = _rate(d[col])
                rec[f"{lab} won"], rec[f"{lab} decided"] = y, n
                rec[f"{lab} won %"] = _pct(y, n)
        if rc:
            o = d[rc].str.lower()
            rec["Dangerous"] = int(o.isin(DANGER).sum())
            rec["Goals"] = int((o == "goal").sum())
            rec["Danger %"] = _pct(rec["Dangerous"], len(d))
        for s in ("Left", "Right"):
            rec[s] = int((d["side"] == s).sum()) if "side" in d else 0
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["Taken", "Dangerous"] if "Dangerous" in out else ["Taken"], ascending=False) if len(out) else out


def group_compare(sd: SetPieceData, groups: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Long table (qualifier, value, group, count, share %) for any groups (left vs right, direct vs indirect …)."""
    rows = []
    for label, g in groups:
        for q, vc in breakdown_of(sd, g).items():
            tot = vc.sum()
            for v, k in vc.items():
                rows.append({"qualifier": q, "value": v, "phase": label, "count": int(k), "share": k / tot * 100,
                             "of": int(tot)})
    return pd.DataFrame(rows)


def side_insights(sd: SetPieceData, row: str) -> list[str]:
    """Left vs right for one set-piece type."""
    team, out = sd.team, []
    g = sd.of(row=row)
    if "side" not in g:
        return out
    rc = result_col(sd)
    for s in ("Left", "Right"):
        d = g[g["side"] == s]
        if d.empty:
            continue
        bits = [f"{len(d)} from the {s.lower()}"]
        for c in CONTESTS:
            col = _col(sd, c)
            if col:
                y, n = _rate(d[col])
                if n:
                    bits.append(f"{c} won {y}/{n}")
        if rc:
            k = int(d[rc].str.lower().isin(DANGER).sum())
            bits.append(f"danger {k}/{len(d)}")
        area = _main(d, _col(sd, "Area"))
        if area[2]:
            bits.append(f"mostly to the {area[0].lower()} ({area[1]}/{area[2]})")
        tk = d["taker"][d["taker"] != ""].value_counts() if "taker" in d else pd.Series(dtype=int)
        if len(tk):
            bits.append("taker " + ", ".join(f"#{a} {b}" for a, b in tk.items()))
        out.append(f"**{s}:** " + " · ".join(bits) + ".")
    return out


# ------------------------------------------------------------------ qualifier titles (1st contact vs 2nd ball)
CONTACT_COLORS = {"first": "#3ab0ff", "second": "#b18cff", "result": "#ffc94a", "": None}


def qual_kind(sd: SetPieceData, q: str) -> str:
    """'first' (1st contact + its outcome), 'second' (2nd ball + its outcome), 'result' or ''."""
    k = q.lower()
    if q == RESULT_COL:
        return "result"
    if k.startswith(("1st", "first")):
        return "first"
    if k.startswith(("2nd", "second")):
        return "second"
    return ""


def qual_title(sd: SetPieceData, q: str) -> tuple[str, str]:
    """(headline, detail) for a qualifier panel, so the two contacts are never confused."""
    kind, k = qual_kind(sd, q), q.lower()
    if kind == "result":
        return "OVERALL RESULT", "most dangerous of both contacts"
    if kind in ("first", "second"):
        head = "1ST CONTACT" if kind == "first" else "2ND BALL"
        return head, ("what happened" if "outcome" in k else f"who won it ({sd.team} / opponent)")
    return q.upper(), ""


def ordered_quals(sd: SetPieceData, quals) -> list[str]:
    """Result, then the 1st-contact pair, then the 2nd-ball pair, then everything else (file order)."""
    rank = {"result": 0, "first": 1, "second": 2, "": 3}
    quals = list(quals)
    return sorted(quals, key=lambda q: (rank[qual_kind(sd, q)], "outcome" in q.lower(), quals.index(q)))


def qual_html(sd: SetPieceData, q: str, extra: str = "") -> str:
    head, detail = qual_title(sd, q)
    col = CONTACT_COLORS[qual_kind(sd, q)]
    h = f"<span style='color:{col}'><b>{head}</b></span>" if col else f"<b>{head}</b>"
    return h + (f" · {detail}" if detail else "") + (f" · {extra}" if extra else "")
