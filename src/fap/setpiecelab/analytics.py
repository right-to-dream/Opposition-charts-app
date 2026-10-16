"""Set-piece analytics on a :class:`~fap.setpiecelab.loader.SetPieceData` — counts, contest rates,
qualifier breakdowns, cross-tabs and deterministic insights (numbers only, nothing invented)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fap.setpiecelab.loader import SetPieceData, is_yes

DANGER = {"goal", "shot", "threat"}
CONTESTS = ("1st contact", "2nd ball")
# "Yes" in a contest column = the file's team won it — whether it takes or defends the set piece.


def _rate(series: pd.Series) -> tuple[int, int]:
    """(yes, decided) for a Yes/No qualifier column."""
    v = series.map(is_yes)
    return int((v == True).sum()), int(v.notna().sum())  # noqa: E712


def _col(sd: SetPieceData, name: str) -> str | None:
    for q in sd.qualifiers:
        if q.strip().lower() == name.lower():
            return q
    return None


def contest_label(sd: SetPieceData, v: str) -> str:
    yn = is_yes(v)
    return f"{sd.team} won" if yn is True else "Opponent won" if yn is False else v


def overview(sd: SetPieceData) -> pd.DataFrame:
    rows = []
    out_c, exe_c, area_c, loc_c, eff_c = (_col(sd, n) for n in ("Outcome", "Execution", "Area", "Location", "Efficiency"))
    for r in sd.rows:
        g = sd.of(row=r)
        rec = {"Set piece": r, "Category": g["category"].iloc[0], "Phase": g["phase"].iloc[0], "Count": len(g),
               "With landing point": int(g["x"].notna().sum())}
        for c in CONTESTS:
            col = _col(sd, c)
            if col:
                y, n = _rate(g[col])
                rec[f"{c} won"] = y
                rec[f"{c} decided"] = n
        if eff_c:
            y, n = _rate(g[eff_c])
            rec["Efficient"], rec["Efficiency coded"] = y, n
        if out_c:
            o = g[out_c].str.lower()
            rec["Goals"] = int((o == "goal").sum())
            rec["Shots"] = int((o == "shot").sum())
            rec["Dangerous (goal/shot/threat)"] = int(o.isin(DANGER).sum())
        for c, lab in ((exe_c, "Main execution"), (area_c, "Main target"), (loc_c, "Main side")):
            if c:
                vc = g[c][g[c] != ""].value_counts()
                rec[lab] = f"{vc.index[0]} ({vc.iloc[0]}/{int(vc.sum())})" if len(vc) else "–"
        rows.append(rec)
    return pd.DataFrame(rows)


def breakdown(sd: SetPieceData, row: str) -> dict[str, pd.Series]:
    g = sd.of(row=row)
    out = {}
    contests = {c.lower() for c in CONTESTS}
    for q in sd.qualifiers:
        vc = g[q][g[q] != ""].value_counts()
        if len(vc):
            if q.strip().lower() in contests:            # Yes / No → who won it
                vc = vc.rename(lambda v: contest_label(sd, v))
                vc = vc.groupby(level=0).sum().sort_values(ascending=False)
            out[q] = vc
    return out


def crosstab(sd: SetPieceData, row: str, a: str, b: str) -> pd.DataFrame:
    g = sd.of(row=row)
    g = g[(g[a] != "") & (g[b] != "")]
    return pd.crosstab(g[a], g[b]) if len(g) else pd.DataFrame()


def insights(sd: SetPieceData) -> list[tuple[str, str]]:
    """(set piece, sentence) — every sentence is a count from the file."""
    team = sd.team
    out_c, exe_c, area_c, loc_c = (_col(sd, n) for n in ("Outcome", "Execution", "Area", "Location"))
    res = []
    for r in sd.rows:
        g = sd.of(row=r)
        att = g["phase"].iloc[0] == "Attacking"
        who = f"{team}" if att else f"Against {team}"
        parts = [f"{len(g)} coded"]
        if exe_c:
            vc = g[exe_c][g[exe_c] != ""].value_counts()
            if len(vc):
                parts.append(f"delivery mostly {vc.index[0].lower()} ({vc.iloc[0]}/{int(vc.sum())})")
        if area_c:
            vc = g[area_c][g[area_c] != ""].value_counts()
            if len(vc):
                parts.append(f"aimed at the {vc.index[0].lower()} ({vc.iloc[0]}/{int(vc.sum())})")
        if loc_c:
            vc = g[loc_c][g[loc_c] != ""].value_counts()
            if len(vc) and vc.sum() >= 3:
                parts.append(f"{vc.index[0].lower()} side {vc.iloc[0]}/{int(vc.sum())}")
        for c in CONTESTS:
            col = _col(sd, c)
            if col:
                y, n = _rate(g[col])
                if n:
                    parts.append(f"{c} won by {team} {y}/{n} ({y / n * 100:.0f}%)")
        if out_c:
            o = g[out_c][g[out_c] != ""].str.lower()
            if len(o):
                vc = o.value_counts()
                parts.append("outcomes: " + ", ".join(f"{k} {v}" for k, v in vc.items()))
        res.append((r, f"{who} · " + " · ".join(parts)))
    # headline insights across types
    ov = overview(sd)
    if "Dangerous (goal/shot/threat)" in ov:
        att = ov[ov["Phase"] == "Attacking"].sort_values("Dangerous (goal/shot/threat)", ascending=False)
        if len(att) and att["Dangerous (goal/shot/threat)"].iloc[0] > 0:
            r0 = att.iloc[0]
            res.insert(0, ("Headline", f"{team}'s most dangerous weapon: {r0['Set piece']} — "
                                        f"{int(r0['Dangerous (goal/shot/threat)'])} dangerous outcomes from {int(r0['Count'])}"
                                        f" ({int(r0.get('Goals', 0))} goals)."))
        de = ov[ov["Phase"] == "Defending"].sort_values("Dangerous (goal/shot/threat)", ascending=False)
        if len(de) and de["Dangerous (goal/shot/threat)"].iloc[0] > 0:
            r0 = de.iloc[0]
            res.insert(1, ("Headline", f"Where {team} concede danger: {r0['Set piece']} — "
                                        f"{int(r0['Dangerous (goal/shot/threat)'])} dangerous outcomes against them from "
                                        f"{int(r0['Count'])} ({int(r0.get('Goals', 0))} goals)."))
    return res
