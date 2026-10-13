"""Phases of play measured from pitch location (x, y).

In possession (where the ball is, in the team's attacking frame)
  * Build-up      x < 35 m   (own third — playing out from the back)
  * Progression   35–70 m    (middle third — moving the ball through the lines)
  * Final third   x ≥ 70 m   (chance creation)
  * Attacking transition — any action within ``TRANSITION_S`` seconds of a possession that started
    with a regain (recovery / interception / tackle won), on top of its location phase.

Out of possession (where the defensive action happens, in the defending team's frame)
  * High press    x ≥ 70 m   (in the opponent's defensive third)
  * Mid block     35–70 m
  * Low block     x < 35 m   (own third)
  * Counter-press — a defensive action within ``md.cp_window`` (6–8 s) seconds of losing the ball (or
    tagged "Counter Press").

Thresholds follow ``md.thirds`` so the UI can move the lines.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fap.matchlab.metrics import DEF_ACTIONS, PITCH_L, _pct, in_box, ppda_mask

if TYPE_CHECKING:
    from fap.matchlab.loader import MatchData

IN_PHASES = ("Build-up", "Progression", "Final third")
OUT_PHASES = ("High press", "Mid block", "Low block")
TRANSITION_S = 10.0
COUNTERPRESS_S = 5.0
_ON_BALL = ("pass", "receive", "carry", "dribble", "shot")


def tag_phases(md: "MatchData") -> None:
    ev = md.events
    t1, t2 = md.thirds
    x = ev["x"].to_numpy(float)
    on_ball = ev["etype"].isin(_ON_BALL) & ev["in_possession"]
    ev["phase"] = np.where(on_ball, np.where(x < t1, IN_PHASES[0], np.where(x < t2, IN_PHASES[1], IN_PHASES[2])), "")
    ev.loc[ev["x"].isna(), "phase"] = ""
    is_def = ev["etype"].isin(DEF_ACTIONS) & ev["team"].ne("")
    ev["def_phase"] = np.where(is_def, np.where(x >= t2, OUT_PHASES[0], np.where(x >= t1, OUT_PHASES[1], OUT_PHASES[2])), "")
    ev.loc[ev["x"].isna(), "def_phase"] = ""

    poss = md.possessions
    ev["transition"] = False
    ev["counter_press"] = False
    if poss.empty:
        return
    start_t = ev[ev["poss_id"] >= 0].groupby("poss_id")["t"].min()
    regain = poss.set_index("poss_id")["start_type"].eq("Regain")
    team_of = poss.set_index("poss_id")["team"]
    st = ev["poss_id"].map(start_t)
    ev["transition"] = on_ball & ev["poss_id"].map(regain.to_dict()).eq(True) & ((ev["t"] - st) <= TRANSITION_S)
    prev_team = ev["poss_id"].map(lambda p: team_of.get(p - 1, ""))
    lost_recently = is_def & ev["team"].ne(ev["poss_team"]) & prev_team.eq(ev["team"]) & ((ev["t"] - st) <= md.cp_window)
    ev["counter_press"] = lost_recently | (is_def & ev["lab_counter_press"])


# ------------------------------------------------------------------ in-possession summary
def in_possession_table(md: "MatchData", team: str, half: int | None = None) -> pd.DataFrame:
    ev = _scope(md, half)
    me = ev[(ev["team"] == team) & (ev["phase"] != "")]
    total = len(me)
    rows = []
    for ph in IN_PHASES:
        g = me[me["phase"] == ph]
        p = g[g["etype"] == "pass"]
        comp = p["outcome"] == 1
        losses = int((p["outcome"] == 0).sum() + ((g["etype"] == "dribble") & (g["outcome"] == 0)).sum())
        top = g[g["player"].ne("") & g["player"].ne("0")]["player"].value_counts().head(3)
        rows.append({
            "Phase": ph, "Actions": len(g), "Share %": _pct(len(g), total),
            "Passes": len(p), "Accuracy %": _pct(comp.sum(), len(p)),
            "Forward %": _pct((p["direction"] == "Forward").sum(), len(p)),
            "Progressive": int((p["progressive"] & comp).sum() + ((g["etype"] == "carry") & g["progressive"]).sum()),
            "Line breaks": int((p["line_break"] & comp).sum()),
            "Losses": losses, "Loss %": _pct(losses, len(p) + (g["etype"] == "dribble").sum()),
            "Avg pass (m)": float(p["length"].mean()) if len(p) else np.nan,
            "Under pressure": int(g["under_pressure"].sum()),
            "Key players": ", ".join(f"#{k} ({v})" for k, v in top.items()),
        })
    return pd.DataFrame(rows)


def funnel(md: "MatchData", team: str, half: int | None = None, start: str = "all") -> list[tuple[str, int]]:
    """Possession funnel. ``start='build-up'`` keeps only possessions that START in the own third."""
    poss = _poss(md, team, half)
    if poss.empty:
        return []
    t1, t2 = md.thirds
    if start == "build-up":
        poss = poss[poss["start_x"] < t1]
        stages = [("Started in build-up", len(poss)),
                  ("Reached middle third", int((poss["max_x"] >= t1).sum()))]
    else:
        stages = [("All possessions", len(poss)),
                  ("Reached middle third", int((poss["max_x"] >= t1).sum()))]
    stages += [("Reached final third", int(poss["reached_f3"].sum())),
               ("Reached the box", int(poss["reached_box"].sum())),
               ("Ended in a shot", int(poss["shot"].sum()))]
    return stages


def buildup_outcomes(md: "MatchData", team: str, half: int | None = None) -> dict:
    poss = _poss(md, team, half)
    if poss.empty:
        return {}
    b = poss[poss["start_x"] < md.thirds[0]]
    if b.empty:
        return {"count": 0}
    return {
        "count": len(b), "avg_duration": float(b["duration"].mean()), "avg_passes": float(b["n_passes"].mean()),
        "furthest": b["reach_third"].value_counts().to_dict(),
        "ended": b["end_reason"].value_counts().to_dict(),
        "shots": int(b["shot"].sum()), "goals": int(b["goal"].sum()),
        "lost_in_own_third": int(((b["end_reason"] == "Lost") & (b["end_x"] < md.thirds[0])).sum()),
        "players": pd.Series([p for ps in b["players"] for p in ps]).value_counts().head(10).to_dict(),
    }


def transitions(md: "MatchData", team: str, half: int | None = None) -> dict:
    poss = _poss(md, team, half)
    if poss.empty:
        return {}
    r = poss[poss["start_type"] == "Regain"]
    ev = _scope(md, half)
    me = ev[(ev["team"] == team) & ev["transition"]]
    return {
        "regains": len(r),
        "high_regains": int((r["start_x"] >= md.thirds[1]).sum()),
        "mid_regains": int(((r["start_x"] >= md.thirds[0]) & (r["start_x"] < md.thirds[1])).sum()),
        "low_regains": int((r["start_x"] < md.thirds[0]).sum()),
        "to_final_third": int(r["reached_f3"].sum()), "to_box": int(r["reached_box"].sum()),
        "shots": int(r["shot"].sum()), "goals": int(r["goal"].sum()),
        "transition_actions": len(me),
        "fast_forward_passes": int(((me["etype"] == "pass") & (me["direction"] == "Forward")).sum()),
    }


# ------------------------------------------------------------------ out-of-possession summary
def out_of_possession_table(md: "MatchData", team: str, half: int | None = None) -> pd.DataFrame:
    ev = _scope(md, half)
    me = ev[(ev["team"] == team) & (ev["def_phase"] != "")]
    total = len(me)
    rows = []
    for ph in OUT_PHASES:
        g = me[me["def_phase"] == ph]
        known = g["outcome"].notna() & g["etype"].isin(["tackle", "duel", "press", "interception", "dribble"])
        won = g["etype"].isin(["recovery"]) | (g["etype"].isin(["interception", "tackle"]) & g["labels"].map(lambda l: "won" in l))
        rows.append({
            "Phase": ph, "Actions": len(g), "Share %": _pct(len(g), total),
            "Regains": int(won.sum()),
            "Interceptions": int((g["etype"] == "interception").sum()),
            "Tackles": int((g["etype"] == "tackle").sum()),
            "Recoveries": int((g["etype"] == "recovery").sum()),
            "Duels": int((g["etype"] == "duel").sum()),
            "Fouls": int((g["etype"] == "foul").sum()),
            "Presses": int((g["etype"] == "press").sum()),
            "Success %": _pct(((g["outcome"] == 1) & known).sum(), known.sum()),
            "Counter-press": int(g["counter_press"].sum()),
        })
    return pd.DataFrame(rows)


def defending_kpis(md: "MatchData", team: str, half: int | None = None) -> dict:
    ev = _scope(md, half)
    opp = md.opponent(team)
    me = ev[ev["team"] == team]
    op_pass = ev[(ev["team"] == opp) & (ev["etype"] == "pass")]
    d = me[me["etype"].isin(DEF_ACTIONS)]
    ppda_den = me[ppda_mask(me)]
    ppda_num = op_pass[op_pass["x"] < 0.6 * PITCH_L]
    from fap.matchlab import counterpress
    cp = counterpress.analyse(md, team, md.cp_window, half)
    lost = len(cp)
    regained5 = int(cp["regained"].sum()) if lost else 0
    opp_box = ev[(ev["team"] == opp) & (ev["etype"].isin(["pass", "carry"])) & ev["box_entry"] & (ev["outcome"] == 1)]
    return {
        "PPDA": len(ppda_num) / len(ppda_den) if len(ppda_den) else np.nan,
        "Defensive actions": len(d),
        "Def. action height (m)": float(d["x"].mean()) if len(d) else np.nan,
        "High regains": int(((me["etype"].isin(["recovery"]) | (me["etype"].isin(["interception", "tackle"]) & me["labels"].map(lambda l: "won" in l))) & (me["x"] >= md.thirds[1])).sum()),
        "Counter-press actions": int(me["counter_press"].sum()),
        "Losses": lost, "Regained in window": regained5,
        "Opp. box entries allowed": len(opp_box),
        "Opp. shots allowed": int(((ev["team"] == opp) & (ev["etype"] == "shot")).sum()),
        "Opp. xG allowed": float(ev[(ev["team"] == opp)]["xg"].sum()),
    }


# ------------------------------------------------------------------ helpers
def _scope(md: "MatchData", half: int | None) -> pd.DataFrame:
    return md.events if half is None else md.events[md.events["half"] == half]


def _poss(md: "MatchData", team: str, half: int | None) -> pd.DataFrame:
    poss = md.possessions
    if poss.empty:
        return poss
    poss = poss[poss["team"] == team]
    return poss if half is None else poss[poss["half"] == half]


__all__ = ["IN_PHASES", "OUT_PHASES", "tag_phases", "in_possession_table", "out_of_possession_table",
           "funnel", "buildup_outcomes", "transitions", "defending_kpis", "in_box"]
