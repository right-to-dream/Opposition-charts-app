"""Player comparison across several matches — minutes, per-90 values, position groups and percentiles.

* :func:`minutes_played` — each player's time on the pitch per match from the ``Sub in`` / ``Sub out`` tags
  (starters from kick-off, substitutes from their entry, to their exit or the end of each half — stoppage
  time included).
* :func:`player_pool` — one row per player over the chosen matches: minutes, matches, starts, summed counts,
  rates recomputed from the summed numerators / denominators, and **per 90** for every count.
* :func:`add_percentiles` — percentile (0–100) of every metric against a pool: the same position group or the
  whole team, among players with at least ``min_minutes``. "Lower is better" metrics (ball losses) are inverted.
* ``METRICS`` — the catalogue (label, column, category, per-90?, higher-is-better) every chart reads.

Positions come from the squad file (GK / DEF / MID / FOR) and can be overridden per player (e.g. CB, FB, CM, W,
ST); the comparison pool follows whatever labels are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fap.matchlab.loader import MatchData

ROLE_GROUP = {"GK": "GK", "GOALKEEPER": "GK", "DEF": "DEF", "D": "DEF", "CB": "DEF", "FB": "DEF", "LB": "DEF",
              "RB": "DEF", "MID": "MID", "M": "MID", "DM": "MID", "CM": "MID", "AM": "MID", "FOR": "FWD",
              "FWD": "FWD", "F": "FWD", "ST": "FWD", "W": "FWD", "LW": "FWD", "RW": "FWD"}
POSITION_CHOICES = ("GK", "CB", "FB", "DM", "CM", "AM", "W", "ST", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class Metric:
    label: str
    column: str            # column in the pool table (per-90 columns end with " p90")
    category: str
    higher_better: bool = True
    fmt: str = "{:.2f}"


# extra counts derived straight from the events (numerator / denominator pairs for rates)
_EXTRA = ("Passes into final third", "Passes into box", "Line-breaking passes", "Long passes", "Long passes completed",
          "Crosses", "Crosses completed", "Forward passes", "Touches in box", "Receptions")


def _extra_counts(md: MatchData, team: str, half=None) -> pd.DataFrame:
    ev = md.events[(md.events["team"] == team) & md.events["player"].ne("") & md.events["player"].ne("0")]
    if half is not None:
        ev = ev[ev["half"] == half]
    from fap.matchlab.metrics import in_box
    rows = []
    allp = md.events[(md.events["team"] == team) & (md.events["etype"] == "pass") & (md.events["outcome"] == 1)]
    if half is not None:
        allp = allp[allp["half"] == half]
    for pl, g in ev.groupby("player"):
        p = g[g["etype"] == "pass"]
        ok = p["outcome"] == 1
        touch = g[g["etype"].isin(["receive", "pass", "carry", "dribble", "shot"]) & g["x"].notna()
                  & ~g["derived"].astype(bool)]
        rows.append({"player": pl,
                     "Passes into final third": int((p["f3_entry"] & ok).sum()),
                     "Passes into box": int((p["box_entry"] & ok).sum()),
                     "Line-breaking passes": int((p["line_break"] & ok).sum()),
                     "Long passes": int((p["length_class"] == "Long").sum()),
                     "Long passes completed": int(((p["length_class"] == "Long") & ok).sum()),
                     "Crosses": int(p["cross"].sum()), "Crosses completed": int((p["cross"] & ok).sum()),
                     "Forward passes": int((p["direction"] == "Forward").sum()),
                     "Touches in box": int(in_box(touch["x"], touch["y"]).sum()),
                     "Receptions": int((allp["receiver"] == pl).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ minutes
def minutes_played(md: MatchData, team: str, half=None) -> pd.DataFrame:
    """player, minutes, started (bool) — from the Sub in / Sub out tags and the half start / end tags."""
    ev = md.events
    bounds = {}
    for h, g in ev.groupby("half"):
        lo = g.loc[g["etype"] == "half_start", "minute"].min()
        hi = g.loc[g["etype"] == "half_end", "minute"].max()
        bounds[int(h)] = (float(lo if pd.notna(lo) else g["minute"].min()), float(hi if pd.notna(hi) else g["minute"].max()))
    me = ev[(ev["team"] == team) & ev["player"].ne("") & ev["player"].ne("0")]
    subs_in = me[me["etype"] == "sub_in"].groupby("player")["minute"].min()
    subs_out = me[me["etype"] == "sub_out"].groupby("player")["minute"].min()
    halves = [h for h in sorted(bounds) if half is None or h == half]
    rows = []
    for pl in sorted(set(me["player"])):
        t_in = float(subs_in.get(pl, -1.0))
        t_out = float(subs_out.get(pl, 1e9))
        started = pl not in subs_in.index
        tot = 0.0
        for h in halves:
            a, b = bounds[h]
            lo, hi = max(a, t_in if not started else a), min(b, t_out)
            tot += max(0.0, hi - lo)
        if tot <= 0 and (me["player"] == pl).any():
            g = me[me["player"] == pl]
            tot = float(g["minute"].max() - g["minute"].min())
        rows.append({"player": pl, "minutes": round(tot, 1), "started": started})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pool
_COUNTS = ("Tackles", "Tackles won", "Interceptions", "Interceptions won", "Duels", "Duels won", "Duels decided",
           "Recoveries", "Presses", "Presses successful", "Presses tagged", "Blocks", "Clearances", "Fouls",
           "Defensive actions", "Progressive passes", "Progressive carries", "Progressive receptions",
           "Progressive actions", "Carries", "Carry distance (m)", "Carries into final third", "Dribbles",
           "Dribbles successful", "Passes", "Passes completed", "Key passes", "Shots", "xG", "Ball losses",
           "Second balls won", "Actions") + _EXTRA
_RATES = {"Pass accuracy %": ("Passes completed", "Passes"), "Long pass accuracy %": ("Long passes completed", "Long passes"),
          "Cross accuracy %": ("Crosses completed", "Crosses"), "Forward pass %": ("Forward passes", "Passes"),
          "Duel success %": ("Duels won", "Duels decided"), "Tackle success %": ("Tackles won", "Tackles"),
          "Dribble success %": ("Dribbles successful", "Dribbles"),
          "Interception won %": ("Interceptions won", "Interceptions")}


def player_pool(items, squad=None, positions: dict | None = None, half=None) -> pd.DataFrame:
    """One row per player of the analysed team over the given MatchItems (``.md``, ``.label``)."""
    from fap.matchlab import player as PL
    parts = []
    for it in items:
        md = it.md
        team = md.our_team
        ps = PL.player_stats(md, team, half)
        if ps.empty:
            continue
        ex = _extra_counts(md, team, half)
        mins = minutes_played(md, team, half)
        d = ps.merge(ex, on="player", how="left").merge(mins, on="player", how="left")
        d["match"] = it.label
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    allm = pd.concat(parts, ignore_index=True)
    for c in _COUNTS:
        if c not in allm:
            allm[c] = 0
    allm["minutes"] = allm["minutes"].fillna(0.0)
    allm["started"] = allm["started"].fillna(False)
    g = allm.groupby("player")
    out = g[list(_COUNTS) + ["minutes"]].sum()
    out["Matches"] = g["match"].nunique()
    out["Starts"] = g["started"].sum().astype(int)
    out = out.reset_index().rename(columns={"minutes": "Minutes"})
    for k, (a, b) in _RATES.items():
        out[k] = np.where(out[b] > 0, out[a] / out[b].where(out[b] > 0, 1) * 100, np.nan)
    m90 = out["Minutes"].where(out["Minutes"] > 0) / 90.0
    for c in _COUNTS:
        out[f"{c} p90"] = out[c] / m90
    out["xG per shot"] = np.where(out["Shots"] > 0, out["xG"] / out["Shots"].where(out["Shots"] > 0, 1), np.nan)
    positions = positions or {}
    names, ids, pos = [], [], []
    for pl in out["player"]:
        pid, nm = squad.lookup(pl) if squad is not None else ("", "")
        role = (squad.role.get(pl, "") if squad is not None and hasattr(squad, "role") else "").upper()
        names.append(nm)
        ids.append(pid)
        pos.append(positions.get(pl) or ROLE_GROUP.get(role, role) or "—")
    out["Name"], out["Player ID"], out["Position"] = names, ids, pos
    out["Player"] = [f"{n} (#{p})" if n else f"#{p}" for n, p in zip(names, out["player"])]
    return out.sort_values("Minutes", ascending=False).reset_index(drop=True)


METRICS: list[Metric] = [
    # Passing
    Metric("Passes", "Passes p90", "Passing", fmt="{:.1f}"),
    Metric("Pass accuracy %", "Pass accuracy %", "Passing", fmt="{:.0f}%"),
    Metric("Forward pass %", "Forward pass %", "Passing", fmt="{:.0f}%"),
    Metric("Long pass accuracy %", "Long pass accuracy %", "Passing", fmt="{:.0f}%"),
    Metric("Key passes", "Key passes p90", "Passing"),
    Metric("Line-breaking passes", "Line-breaking passes p90", "Passing"),
    # Progression
    Metric("Progressive passes", "Progressive passes p90", "Progression"),
    Metric("Progressive carries", "Progressive carries p90", "Progression"),
    Metric("Progressive receptions", "Progressive receptions p90", "Progression"),
    Metric("Passes into final third", "Passes into final third p90", "Progression"),
    Metric("Carries into final third", "Carries into final third p90", "Progression"),
    Metric("Carry distance (m)", "Carry distance (m) p90", "Progression", fmt="{:.0f}"),
    # Attacking
    Metric("Shots", "Shots p90", "Attacking"),
    Metric("xG", "xG p90", "Attacking"),
    Metric("Passes into box", "Passes into box p90", "Attacking"),
    Metric("Touches in box", "Touches in box p90", "Attacking"),
    Metric("Dribbles completed", "Dribbles successful p90", "Attacking"),
    Metric("Dribble success %", "Dribble success %", "Attacking", fmt="{:.0f}%"),
    # Defending
    Metric("Defensive actions", "Defensive actions p90", "Defending", fmt="{:.1f}"),
    Metric("Tackles won", "Tackles won p90", "Defending"),
    Metric("Interceptions won", "Interceptions won p90", "Defending"),
    Metric("Recoveries", "Recoveries p90", "Defending"),
    Metric("Presses", "Presses p90", "Defending"),
    Metric("Second balls won", "Second balls won p90", "Defending"),
    # Duels & security
    Metric("Duels won", "Duels won p90", "Duels & security"),
    Metric("Duel success %", "Duel success %", "Duels & security", fmt="{:.0f}%"),
    Metric("Ball losses", "Ball losses p90", "Duels & security", higher_better=False),
]
METRIC_BY_LABEL = {m.label: m for m in METRICS}
CATEGORIES = list(dict.fromkeys(m.category for m in METRICS))


def add_percentiles(pool: pd.DataFrame, *, by: str = "position", min_minutes: float = 90.0) -> pd.DataFrame:
    """``pct:<label>`` columns: percentile within the same position (``by='position'``) or the whole team,
    among players with at least ``min_minutes``. Players under the threshold get no percentile."""
    out = pool.copy()
    eligible = out["Minutes"] >= min_minutes
    out["Eligible"] = eligible
    groups = out["Position"] if by == "position" else pd.Series("All", index=out.index)
    for m in METRICS:
        col = f"pct:{m.label}"
        out[col] = np.nan
        for _, idx in out[eligible].groupby(groups[eligible]).groups.items():
            vals = out.loc[idx, m.column]
            if vals.notna().sum() == 0:
                continue
            r = vals.rank(pct=True, method="average") * 100
            out.loc[idx, col] = (100 - r + 100 / max(vals.notna().sum(), 1)) if not m.higher_better else r
    out["Pool size"] = groups.map(groups[eligible].value_counts()).fillna(0).astype(int)
    return out


def position_averages(pool: pd.DataFrame, min_minutes: float = 90.0) -> pd.DataFrame:
    """Average of every metric per position (players with ≥ ``min_minutes``)."""
    d = pool[pool["Minutes"] >= min_minutes]
    cols = [m.column for m in METRICS if m.column in d]
    return d.groupby("Position")[cols].mean()


def fmt(m: Metric, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return m.fmt.format(v)
