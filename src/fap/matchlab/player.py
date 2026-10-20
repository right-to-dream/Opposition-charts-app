"""Player profile — everything one player did in the match, plus how it compares with his team-mates.

Sections (all from the same engine columns the team charts use):
  * passing      — volume / accuracy, by type, direction and length, progressive, line breaks, key passes,
                   entries, crosses, forward share, under-pressure accuracy, favourite receivers;
  * receiving    — receptions, progressive receptions, where (third), favourite passers;
  * carrying     — carries, progressive carries, total distance, entries;
  * shooting     — shots, on target, goals, xG (only shown when the player has shots);
  * defending    — actions by type, duels won / lost, tackles & interceptions won, by defensive phase;
  * counter-press — his ball losses that were counter-pressed / regained, and times he was the first presser;
  * losses       — balls lost by type and third;
  * comparison   — key metrics vs the team average and his rank among team-mates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fap.matchlab.metrics import PLAYER_DEF_ACTIONS, THIRD_NAMES, _pct, in_box, player_table
from fap.matchlab.memo import per_match

if TYPE_CHECKING:
    from fap.matchlab.loader import MatchData

# (column in player_table, label, higher is better)
COMPARE = [
    ("Actions", "Actions", True), ("Passes", "Passes", True), ("Acc %", "Pass accuracy %", True),
    ("Prog. passes", "Progressive passes", True), ("Prog. carries", "Progressive carries", True),
    ("Prog. received", "Progressive receptions", True), ("F3 entries", "Final-third entries", True),
    ("Box entries", "Box entries", True), ("Line breaks", "Line-breaking passes", True),
    ("Key passes", "Key passes", True), ("Shots", "Shots", True), ("xG", "xG", True),
    ("Def. actions", "Defensive actions", True), ("Recoveries", "Recoveries", True), ("Losses", "Ball losses", False),
]


def _scope(md: "MatchData", half) -> pd.DataFrame:
    return md.events if half is None else md.events[md.events["half"] == half]


def _losses(g: pd.DataFrame) -> pd.DataFrame:
    from fap.matchlab.loader import loss_mask
    return g[loss_mask(g)]


@per_match
def profile(md: "MatchData", team: str, player: str, half: int | None = None) -> dict:
    player = str(player)
    ev = _scope(md, half)
    me = ev[(ev["team"] == team) & (ev["player"] == player)]
    out: dict = {"player": player, "team": team}
    p = me[me["etype"] == "pass"]
    comp = p["outcome"] == 1

    def cat(col, order=None):
        g = p[p[col] != ""].groupby(col)["outcome"].agg(attempted="size", completed=lambda s: int((s == 1).sum()))
        g = g.reindex([o for o in order if o in g.index]) if order else g.sort_values("attempted", ascending=False)
        g["accuracy"] = g["completed"] / g["attempted"] * 100
        return g

    up = p[p["under_pressure"]]
    out["passing"] = {
        "Passes": len(p), "Completed": int(comp.sum()), "Accuracy %": _pct(comp.sum(), len(p)),
        "Progressive (completed)": int((p["progressive"] & comp).sum()),
        "Line-breaking (completed)": int((p["line_break"] & comp).sum()),
        "Key passes": int(p["key_pass"].sum()), "Assists": int(p["assist"].sum()),
        "Final-third entries": int((p["f3_entry"] & comp).sum()), "Box entries": int((p["box_entry"] & comp).sum()),
        "Crosses (cmp/att)": f"{int((p['cross'] & comp).sum())}/{int(p['cross'].sum())}",
        "Forward %": _pct((p["direction"] == "Forward").sum(), len(p)),
        "Avg length (m)": float(p["length"].mean()) if len(p) else np.nan,
        "Under pressure (cmp/att)": f"{int((up['outcome'] == 1).sum())}/{len(up)}",
    }
    out["by_type"] = cat("pass_type")
    out["by_direction"] = cat("direction", ["Forward", "Left", "Right", "Backward"])
    out["by_length"] = cat("length_class", ["Short", "Medium", "Long"])
    out["to_whom"] = p[comp & p["receiver"].ne("") & p["receiver"].ne("0")]["receiver"].value_counts()

    rec = ev[(ev["team"] == team) & (ev["etype"] == "pass") & (ev["receiver"] == player) & (ev["outcome"] == 1)]
    out["receiving"] = {
        "Receptions": len(rec), "Progressive receptions": int(rec["progressive"].sum()),
        "In final third": int((rec["end_x"] >= md.thirds[1]).sum()),
        "In the box": int(in_box(rec["end_x"], rec["end_y"]).sum()),
    }
    out["from_whom"] = rec[rec["player"].ne("") & rec["player"].ne("0")]["player"].value_counts()
    out["receptions"] = rec

    c = me[(me["etype"] == "carry") & me["end_x"].notna()]
    out["carrying"] = {
        "Carries": len(c), "Progressive carries": int(c["progressive"].sum()),
        "Total distance (m)": float(c["length"].sum()) if len(c) else 0.0,
        "Final-third entries": int(c["f3_entry"].sum()), "Box entries": int(c["box_entry"].sum()),
    }

    s = me[me["etype"] == "shot"]
    out["shooting"] = {
        "Shots": len(s), "On target": int(s["shot_on"].sum()), "Goals": int(s["shot_goal"].sum()),
        "xG": float(s["xg"].sum()), "xG / shot": float(s["xg"].mean()) if len(s) else np.nan,
        "Headers": int((s["body_part"] == "Head").sum()),
    }
    out["shots"] = s

    d = me[me["etype"].isin(PLAYER_DEF_ACTIONS)]
    duels = me[me["etype"] == "duel"]
    won = lambda g: int(g["labels"].map(lambda l: "won" in l).sum())  # noqa: E731
    out["defending"] = {
        "Defensive actions": len(d),
        "Duels won / lost": f"{won(duels)}/{int(duels['labels'].map(lambda l: 'lost' in l).sum())}",
        "Tackles (won ball · cleared)": (f"{int((me['etype'] == 'tackle').sum())} ({won(me[me['etype'] == 'tackle'])} · "
                                         f"{int(me[me['etype'] == 'tackle']['labels'].map(lambda l: 'clear' in l).sum())})"),
        "Dribbled past": int((me["etype"] == "dribbled_past").sum()),
        "Interceptions (won)": f"{int((me['etype'] == 'interception').sum())} ({won(me[me['etype'] == 'interception'])})",
        "Recoveries": int((me["etype"] == "recovery").sum()), "Presses": int((me["etype"] == "press").sum()),
        "Fouls": int((me["etype"] == "foul").sum()),
        "Avg. defensive height (m)": float(d["x"].mean()) if len(d) else np.nan,
    }
    out["def_by_phase"] = d[d["def_phase"] != ""]["def_phase"].value_counts()
    out["defensive_events"] = d

    lo = _losses(me)
    out["losses"] = {"Ball losses": len(lo), **{f"in {t.lower()}": int((lo["third"] == t).sum()) for t in THIRD_NAMES}}
    out["loss_events"] = lo

    from fap.matchlab import counterpress as CP
    cp = CP.analyse(md, team, md.cp_window, half)
    if not cp.empty:
        mine = cp[cp["player"] == player]
        first = cp[cp["first_presser"] == player]
        out["counterpress"] = {
            "His open-play losses": len(mine),
            "…counter-pressed by the team": int(mine["pressed"].sum()),
            "…regained by the team": int(mine["regained"].sum()),
            "First presser after a team loss": len(first),
            "…of which regained": int(first["regained"].sum()),
            "Regains after a loss (by him)": int((cp["regained_by"] == player).sum()),
        }
    else:
        out["counterpress"] = {}

    on = ev[(ev["team"] == team) & (ev["phase"] != "")]
    mine_on = on[on["player"] == player]
    out["phases"] = pd.DataFrame({
        "him": mine_on["phase"].value_counts(normalize=True) * 100,
        "team": on["phase"].value_counts(normalize=True) * 100,
    }).reindex(["Build-up", "Progression", "Final third"]).fillna(0)
    out["comparison"] = comparison(md, team, player, half)
    return out


@per_match
def comparison(md: "MatchData", team: str, player: str, half: int | None = None, min_actions: int = 15) -> pd.DataFrame:
    """Player vs team average (team-mates with ≥ ``min_actions`` actions) and his rank among them."""
    tab = player_table(md, team, half)
    if tab.empty or f"#{player}" not in set(tab["Player"]):
        return pd.DataFrame()
    pool = tab[(tab["Actions"] >= min_actions) | (tab["Player"] == f"#{player}")]
    me = pool[pool["Player"] == f"#{player}"].iloc[0]
    rows = []
    for col, label, higher in COMPARE:
        vals = pd.to_numeric(pool[col], errors="coerce")
        v = float(pd.to_numeric(pd.Series([me[col]]), errors="coerce").iloc[0])
        if np.isnan(v):
            continue
        rank = int((vals > v).sum() + 1) if higher else int((vals < v).sum() + 1)
        rows.append({"Metric": label, "Player": v, "Team average": float(vals.mean()), "Team best": float(vals.max() if higher else vals.min()),
                     "Rank": rank, "Of": int(vals.notna().sum()), "Higher is better": higher})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ leaderboards & multi-player comparison
def _won(g: pd.DataFrame) -> int:
    return int(g["labels"].map(lambda l: "won" in l).sum())


@per_match
def player_stats(md: "MatchData", team: str, half: int | None = None) -> pd.DataFrame:
    """One row per player: every defensive action type with its success, progressive actions,
    carries and dribbles. Success rates only where success is actually tagged."""
    ev = _scope(md, half)
    me = ev[(ev["team"] == team) & ev["player"].ne("") & ev["player"].ne("0")]
    passes = ev[(ev["team"] == team) & (ev["etype"] == "pass") & (ev["outcome"] == 1)]
    rows = []
    from fap.matchlab import secondball as SB
    sb_won = SB.players(md, team, half)
    for pl, g in me.groupby("player"):
        r: dict = {"player": pl}
        tk = g[g["etype"] == "tackle"]
        it = g[g["etype"] == "interception"]
        du = g[g["etype"] == "duel"]
        du_lost = int(du["labels"].map(lambda l: "lost" in l).sum())
        pr = g[g["etype"] == "press"]
        pr_tag = pr[pr["outcome"].notna()]
        r.update({
            "Tackles": len(tk), "Tackles won": _won(tk),
            "Tackles cleared": int(tk["labels"].map(lambda l: "clear" in l and "won" not in l).sum()),
            "Dribbled past": int((g["etype"] == "dribbled_past").sum()),
            "Interceptions": len(it), "Interceptions won": _won(it),
            "Duels": len(du), "Duels won": _won(du), "Duels decided": _won(du) + du_lost,
            "Recoveries": int((g["etype"] == "recovery").sum()),
            "Presses": len(pr), "Presses successful": int((pr_tag["outcome"] == 1).sum()), "Presses tagged": len(pr_tag),
            "Blocks": int((g["etype"] == "block").sum()), "Clearances": int((g["etype"] == "clearance").sum()),
            "Fouls": int((g["etype"] == "foul").sum()),
        })
        r["Defensive actions"] = int(g["etype"].isin(PLAYER_DEF_ACTIONS).sum())
        p = g[g["etype"] == "pass"]
        c = g[(g["etype"] == "carry") & g["end_x"].notna()]
        r["Progressive passes"] = int((p["progressive"] & (p["outcome"] == 1)).sum())
        r["Progressive carries"] = int(c["progressive"].sum())
        r["Progressive receptions"] = int(passes[passes["receiver"] == pl]["progressive"].sum())
        r["Progressive actions"] = r["Progressive passes"] + r["Progressive carries"] + r["Progressive receptions"]
        r["Carries"] = len(c)
        r["Carry distance (m)"] = round(float(c["length"].sum()), 1) if len(c) else 0.0
        r["Carries into final third"] = int(c["f3_entry"].sum())
        dr = g[g["etype"] == "dribble"]
        r["Dribbles"] = len(dr)
        r["Dribbles successful"] = int((dr["outcome"] == 1).sum())
        r["Passes"] = len(p)
        r["Passes completed"] = int((p["outcome"] == 1).sum())
        r["Key passes"] = int(p["key_pass"].sum())
        r["Shots"] = int((g["etype"] == "shot").sum())
        r["xG"] = round(float(g["xg"].sum()), 2)
        r["Ball losses"] = len(_losses(g))
        r["Second balls won"] = int(sb_won.get(pl, 0))
        r["Actions"] = int((~g["derived"].astype(bool)).sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    def pct(a, b):
        return np.where(out[b] > 0, out[a] / out[b].where(out[b] > 0, 1) * 100, np.nan)

    out["Tackles successful"] = out["Tackles won"] + out["Tackles cleared"]
    out["Tackle success %"] = pct("Tackles successful", "Tackles")
    out["Tackle possession %"] = pct("Tackles won", "Tackles")
    out["Interception won %"] = pct("Interceptions won", "Interceptions")
    out["Duel success %"] = pct("Duels won", "Duels decided")
    out["Press success %"] = pct("Presses successful", "Presses tagged")
    out["Dribble success %"] = pct("Dribbles successful", "Dribbles")
    out["Pass accuracy %"] = pct("Passes completed", "Passes")
    return out.sort_values("Actions", ascending=False).reset_index(drop=True)


# leaderboard spec: (title, value column, success numerator, success denominator)
LEADERBOARDS = {
    "Defending": [
        ("Defensive actions", "Defensive actions", None, None),
        ("Tackles (successful = won the ball or cleared it)", "Tackles", "Tackles successful", "Tackles"),
        ("Interceptions (won · rest cleared)", "Interceptions", "Interceptions won", "Interceptions"),
        ("Duels", "Duels", "Duels won", "Duels decided"),
        ("Recoveries", "Recoveries", None, None),
        ("Second balls won", "Second balls won", None, None),
        ("Presses", "Presses", "Presses successful", "Presses tagged"),
        ("Blocks", "Blocks", None, None),
        ("Clearances", "Clearances", None, None),
        ("Fouls committed", "Fouls", None, None),
        ("Dribbled past (lower is better)", "Dribbled past", None, None),
    ],
    "Progression & ball carrying": [
        ("Progressive actions (total)", "Progressive actions", None, None),
        ("Progressive passes", "Progressive passes", None, None),
        ("Progressive carries", "Progressive carries", None, None),
        ("Progressive receptions", "Progressive receptions", None, None),
        ("Carries", "Carries", None, None),
        ("Carry distance (m)", "Carry distance (m)", None, None),
        ("Dribbles", "Dribbles", "Dribbles successful", "Dribbles"),
    ],
}

# comparison radar / table metrics: (label, column, higher is better)
COMPARE_SET = [
    ("Passes", "Passes", True), ("Pass accuracy %", "Pass accuracy %", True), ("Key passes", "Key passes", True),
    ("Progressive passes", "Progressive passes", True), ("Progressive carries", "Progressive carries", True),
    ("Progressive receptions", "Progressive receptions", True), ("Dribbles successful", "Dribbles successful", True),
    ("Shots", "Shots", True), ("xG", "xG", True), ("Defensive actions", "Defensive actions", True),
    ("Tackles won", "Tackles won", True), ("Interceptions won", "Interceptions won", True),
    ("Duels won", "Duels won", True), ("Recoveries", "Recoveries", True), ("Second balls won", "Second balls won", True),
    ("Ball losses", "Ball losses", False),
]


def compare_players(md: "MatchData", team: str, players: list, half: int | None = None,
                    min_actions: int = 15) -> pd.DataFrame:
    """Rows = metrics, columns = the chosen players (+ team best among team-mates with ≥ ``min_actions``
    actions, so a substitute with one pass is never the 'best' pass accuracy)."""
    st_ = player_stats(md, team, half)
    if st_.empty:
        return pd.DataFrame()
    st_ = st_.set_index("player")
    pool = st_[(st_["Actions"] >= min_actions) | st_.index.isin([str(p) for p in players])]
    rows = []
    for label, col, higher in COMPARE_SET:
        if col not in st_.columns:
            continue
        vals = pd.to_numeric(pool[col], errors="coerce")
        row = {"Metric": label, "Higher is better": higher,
               "Team best": float(vals.max() if higher else vals.min()),
               "Team worst": float(vals.min() if higher else vals.max())}
        for p in players:
            row[str(p)] = float(vals.get(str(p), np.nan))
        rows.append(row)
    return pd.DataFrame(rows)
