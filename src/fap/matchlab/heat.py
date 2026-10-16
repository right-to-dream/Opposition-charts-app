"""Heat-map kinds for Match Lab — which events feed each heat map (shared by the static and the
interactive charts, so both always count the same thing).

Every team's events are in its own analysis frame (attacking x → 105), so a team's map always reads
left → right; an opponent map shows where *the opponent* did it, attacking the same way.
"""
from __future__ import annotations

import pandas as pd

from fap.matchlab.loader import MatchData

KINDS: dict[str, tuple[str, str]] = {
    "activity": ("Activity", "all on-ball & defensive actions"),
    "receptions": ("Receptions", "where the ball is received"),
    "passes": ("Passes made", "where passes are played from"),
    "pass_targets": ("Pass destinations", "where completed passes arrive"),
    "regains": ("Ball regains", "recoveries, interceptions and won tackles"),
    "defensive": ("Defensive actions", "tackles, interceptions, recoveries, blocks, clearances, duels, fouls"),
    "press": ("Pressing", "every press action"),
    "losses": ("Ball losses", "lost passes and dribbles, ball lost, errors"),
}
SHEET = ("activity", "receptions", "passes", "pass_targets", "regains", "defensive", "press", "losses")
_DEF = ("tackle", "interception", "recovery", "block", "clearance", "duel", "foul")


def _scope(md: MatchData, team: str, half=None, player=None) -> pd.DataFrame:
    ev = md.events[md.events["team"] == team]
    if half is not None:
        ev = ev[ev["half"] == half]
    if player:
        ev = ev[ev["player"] == str(player)]
    return ev


def loss_mask(ev: pd.DataFrame) -> pd.Series:
    return (((ev["etype"] == "pass") & (ev["outcome"] == 0)) | ((ev["etype"] == "dribble") & (ev["outcome"] == 0))
            | (ev["etype"] == "ball_lost") | ((ev["etype"] == "error") & ev["labels"].map(lambda l: "ball lost" in l)))


def regain_mask(ev: pd.DataFrame) -> pd.Series:
    return (ev["etype"] == "recovery") | (ev["etype"].isin(["interception", "tackle"]) & ev["labels"].map(lambda l: "won" in l))


def points(md: MatchData, team: str, kind: str, half=None, player=None) -> pd.DataFrame:
    """The events of ``kind`` with their heat-map location in ``x`` / ``y`` (pass destinations use the end point)."""
    ev = _scope(md, team, half, player)
    derived = ev["derived"].astype(bool) if "derived" in ev else pd.Series(False, index=ev.index)
    if kind == "activity":
        out = ev[ev["etype"].isin(["pass", "receive", "carry", "dribble", "shot", "recovery", "interception", "tackle",
                                   "duel"]) & ~derived]
    elif kind == "receptions":
        out = ev[ev["etype"] == "receive"]
        if out["x"].notna().sum() == 0:                       # no receive tags → completed pass end points
            p = ev[(ev["etype"] == "pass") & (ev["outcome"] == 1) & ev["end_x"].notna()].copy()
            p["x"], p["y"] = p["end_x"], p["end_y"]
            out = p
    elif kind == "passes":
        out = ev[ev["etype"] == "pass"]
    elif kind == "pass_targets":
        out = ev[(ev["etype"] == "pass") & (ev["outcome"] == 1) & ev["end_x"].notna()].copy()
        out["x"], out["y"] = out["end_x"], out["end_y"]
    elif kind == "regains":
        out = ev[regain_mask(ev)]
    elif kind == "defensive":
        out = ev[ev["etype"].isin(_DEF)]
    elif kind == "press":
        out = ev[ev["etype"] == "press"]
    elif kind == "losses":
        out = ev[loss_mask(ev)]
    else:
        raise KeyError(kind)
    return out[out["x"].notna() & out["y"].notna()]
