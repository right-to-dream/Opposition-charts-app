"""Per-position player KPIs from the tidy event table of :mod:`fap.playertag.loader`.

Every metric is a plain count or a success rate computed only over the instances where the outcome was
tagged (an untagged outcome is never counted as a failure). Per-90 values need the minutes played.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from fap.playertag import taxonomy as T

PASSING, CROSSING, ATTACKING, POSSESSION, DEFENDING, GOALKEEPING = (
    "Passing", "Crossing", "Attacking", "Ball retention", "Defending", "Goalkeeping")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    section: str
    kind: str                     # "count" | "pct" | "share"
    fn: Callable                  # df -> count  |  df -> (successes, attempts)
    actions: tuple[str, ...]      # actions it needs (to decide whether it applies)
    lower_better: bool = False
    help: str = ""


@dataclass
class Value:
    metric: Metric
    value: float                  # count, or % for pct/share
    made: int = 0
    attempts: int = 0
    per90: float | None = None

    @property
    def display(self) -> str:
        if self.metric.kind == "count":
            return f"{int(self.value)}"
        if not self.attempts:
            return "–"
        return f"{self.value:.0f}%"

    @property
    def detail(self) -> str:
        if self.metric.kind == "count":
            return "" if self.per90 is None else f"{self.per90:.2f} p90"
        return f"{self.made}/{self.attempts}" if self.attempts else "not tagged"


# ------------------------------------------------------------------ helpers
def _a(d: pd.DataFrame, *actions: str) -> pd.DataFrame:
    return d[d["action"].isin(actions)]


def _count(*actions: str, where: Callable | None = None) -> Callable:
    def f(d):
        s = _a(d, *actions)
        return int(len(s if where is None else s[where(s)]))
    return f


def _rate(*actions: str, where: Callable | None = None) -> Callable:
    def f(d):
        s = _a(d, *actions)
        if where is not None:
            s = s[where(s)]
        s = s[s["success"].notna()]
        return int(s["success"].astype(bool).sum()), int(len(s))
    return f


def _share(action: str, num: Callable, den: Callable | None = None) -> Callable:
    def f(d):
        s = _a(d, action)
        if den is not None:
            s = s[den(s)]
        return int(num(s).sum()), int(len(s))
    return f


def _col(c: str) -> Callable:
    return lambda s: s[c].astype(bool)


def _lab(group: str, label: str) -> Callable:
    return lambda s: s[group].str.contains(label, regex=False)


def _opp_half(s):
    return s["x"] > T.PITCH_L / 2


def _own_half(s):
    return s["x"] <= T.PITCH_L / 2


def _losses(d: pd.DataFrame) -> pd.DataFrame:
    failed = d["action"].isin(("Pass", "Cross", "Dribble", "Carry")) & d["success"].eq(False)
    return d[failed | d["action"].eq("Ball Lost")]


def _saves_rate(d):
    sv, gc = len(_a(d, "Save")), len(_a(d, "Goal Conceded"))
    return int(sv), int(sv + gc)


def _gk_dist(label: str) -> Callable:
    return _rate("Pass", where=_lab("Distribution", label))


METRICS: tuple[Metric, ...] = (
    # passing
    Metric("passes", "Passes", PASSING, "count", _count("Pass"), ("Pass",)),
    Metric("pass_pct", "Pass accuracy", PASSING, "pct", _rate("Pass"), ("Pass",)),
    Metric("prog_passes", "Progressive passes", PASSING, "count", _count("Pass", where=_col("is_progressive")),
           ("Pass",), help="Passes labelled Progressive"),
    Metric("key_passes", "Key passes", PASSING, "count",
           _count("Pass", "Cross", where=lambda s: s["is_key_pass"] | s["is_assist"]), ("Pass",),
           help="Passes / crosses that led to a shot (Key Pass or Assist label)"),
    Metric("assists", "Assists", PASSING, "count", _count("Pass", "Cross", where=_col("is_assist")), ("Pass",)),
    Metric("through_balls", "Through balls", PASSING, "count", _count("Pass", where=_col("is_through_ball")),
           ("Pass",)),
    Metric("line_breaking", "Line-breaking passes", PASSING, "count", _count("Pass", where=_col("is_line_breaking")),
           ("Pass",)),
    Metric("switches", "Switches of play", PASSING, "count", _count("Pass", where=_col("is_switch")), ("Pass",)),
    Metric("f3_entries", "Final-third entries", PASSING, "count",
           _count("Pass", "Carry", where=_col("is_final_third_entry")), ("Pass",)),
    Metric("box_entries", "Box entries", PASSING, "count", _count("Pass", "Carry", "Cross",
                                                                 where=_col("is_box_entry")), ("Pass",)),
    Metric("long_pass_pct", "Long-pass accuracy", PASSING, "pct",
           _rate("Pass", where=lambda s: s["Pass Length"].eq("Long")), ("Pass",)),
    Metric("forward_share", "Forward passes", PASSING, "share",
           _share("Pass", lambda s: s["Direction"].eq("Forward"), lambda s: s["Direction"].ne("")), ("Pass",),
           help="Share of passes with a tagged direction that went forward"),
    Metric("pass_up_pct", "Pass accuracy under pressure", PASSING, "pct", _rate("Pass", where=_col("under_pressure")),
           ("Pass",)),
    Metric("opp_half_pass_pct", "Pass accuracy in opponent half", PASSING, "pct", _rate("Pass", where=_opp_half),
           ("Pass",)),
    # crossing
    Metric("crosses", "Crosses", CROSSING, "count", _count("Cross"), ("Cross",)),
    Metric("cross_pct", "Cross accuracy", CROSSING, "pct", _rate("Cross"), ("Cross",)),
    Metric("cutbacks", "Cutbacks", CROSSING, "count", _count("Cross", where=_lab("Cross Type", "Cutback")), ("Cross",)),
    # attacking
    Metric("shots", "Shots", ATTACKING, "count", _count("Shot"), ("Shot",)),
    Metric("shots_on_target", "Shots on target", ATTACKING, "count",
           _count("Shot", where=lambda s: s["Shot Outcome"].isin(("Goal", "On Target"))), ("Shot",)),
    Metric("goals", "Goals", ATTACKING, "count", _count("Shot", where=_col("is_goal")), ("Shot",)),
    Metric("shot_accuracy", "Shot accuracy", ATTACKING, "pct", _rate("Shot"), ("Shot",),
           help="On target (incl. goals) / shots with a tagged outcome"),
    Metric("conversion", "Shot conversion", ATTACKING, "share", _share("Shot", _col("is_goal")), ("Shot",)),
    Metric("dribbles", "Dribbles", ATTACKING, "count", _count("Dribble"), ("Dribble",)),
    Metric("dribble_pct", "Dribble success", ATTACKING, "pct", _rate("Dribble"), ("Dribble",)),
    Metric("carries", "Carries", ATTACKING, "count", _count("Carry"), ("Carry",)),
    Metric("prog_carries", "Progressive carries", ATTACKING, "count", _count("Carry", where=_col("is_progressive")),
           ("Carry",)),
    Metric("receptions", "Receptions", ATTACKING, "count", _count("Receive"), ("Receive",)),
    Metric("rec_up", "Receptions under pressure", ATTACKING, "count", _count("Receive", where=_col("under_pressure")),
           ("Receive",)),
    Metric("rec_lines", "Receptions between lines", ATTACKING, "count",
           _count("Receive", where=_lab("Reception", "Between Lines")), ("Receive",)),
    Metric("rec_behind", "Receptions in behind", ATTACKING, "count",
           _count("Receive", where=_lab("Reception", "In Behind")), ("Receive",)),
    Metric("rec_final_third", "Receptions in final third", ATTACKING, "count",
           _count("Receive", where=lambda s: s["third"].eq("Final third")), ("Receive",)),
    Metric("fouls_won", "Fouls won", ATTACKING, "count", _count("Foul Won"), ("Foul Won",)),
    Metric("offsides", "Offsides", ATTACKING, "count", _count("Offside"), ("Offside",), lower_better=True),
    # ball retention
    Metric("losses", "Possession losses", POSSESSION, "count", lambda d: int(len(_losses(d))), ("Pass", "Ball Lost"),
           lower_better=True, help="Ball Lost + failed passes, crosses, dribbles and carries"),
    Metric("losses_own_half", "Losses in own half", POSSESSION, "count",
           lambda d: int(_own_half(_losses(d)).sum()), ("Pass", "Ball Lost"), lower_better=True),
    Metric("ball_lost", "Ball lost (touch / dispossessed)", POSSESSION, "count", _count("Ball Lost"), ("Ball Lost",),
           lower_better=True),
    # defending
    Metric("def_actions", "Defensive actions", DEFENDING, "count",
           _count("Tackle", "Interception", "Recovery", "Clearance", "Block", "Ground Duel", "Aerial Duel"),
           ("Tackle", "Interception"), help="Tackles + interceptions + recoveries + clearances + blocks + duels"),
    Metric("duel_pct", "Duels won", DEFENDING, "pct", _rate("Ground Duel", "Aerial Duel"), ("Ground Duel",)),
    Metric("ground_duels", "Ground duels", DEFENDING, "count", _count("Ground Duel"), ("Ground Duel",)),
    Metric("ground_duel_pct", "Ground duels won", DEFENDING, "pct", _rate("Ground Duel"), ("Ground Duel",)),
    Metric("aerials", "Aerial duels", DEFENDING, "count", _count("Aerial Duel"), ("Aerial Duel",)),
    Metric("aerial_pct", "Aerial duels won", DEFENDING, "pct", _rate("Aerial Duel"), ("Aerial Duel",)),
    Metric("tackles", "Tackles", DEFENDING, "count", _count("Tackle"), ("Tackle",)),
    Metric("tackle_pct", "Tackle success", DEFENDING, "pct", _rate("Tackle"), ("Tackle",)),
    Metric("interceptions", "Interceptions", DEFENDING, "count", _count("Interception"), ("Interception",)),
    Metric("recoveries", "Ball recoveries", DEFENDING, "count", _count("Recovery"), ("Recovery",)),
    Metric("recoveries_opp", "Recoveries in opponent half", DEFENDING, "count", _count("Recovery", where=_opp_half),
           ("Recovery",)),
    Metric("clearances", "Clearances", DEFENDING, "count", _count("Clearance"), ("Clearance",)),
    Metric("blocks", "Blocks", DEFENDING, "count", _count("Block"), ("Block",)),
    Metric("presses", "Presses", DEFENDING, "count", _count("Press"), ("Press",)),
    Metric("press_pct", "Press success", DEFENDING, "pct", _rate("Press"), ("Press",),
           help="Regain or forced error / presses with a tagged outcome"),
    Metric("dribbled_past", "Dribbled past", DEFENDING, "count", _count("Dribbled Past"), ("Dribbled Past",),
           lower_better=True),
    Metric("fouls", "Fouls committed", DEFENDING, "count", _count("Foul"), ("Foul",), lower_better=True),
    Metric("cards", "Cards", DEFENDING, "count", _count("Foul", where=lambda s: s["Card"].ne("")), ("Foul",),
           lower_better=True),
    # goalkeeping
    Metric("saves", "Saves", GOALKEEPING, "count", _count("Save"), ("Save",)),
    Metric("goals_conceded", "Goals conceded", GOALKEEPING, "count", _count("Goal Conceded"), ("Save",),
           lower_better=True),
    Metric("save_pct", "Save %", GOALKEEPING, "pct", _saves_rate, ("Save",),
           help="Saves / (saves + goals conceded)"),
    Metric("clean_saves", "Saves held / parried safe", GOALKEEPING, "pct", _rate("Save"), ("Save",),
           help="Saves without a rebound to the opponent"),
    Metric("high_claims", "High claims", GOALKEEPING, "count", _count("High Claim"), ("High Claim",)),
    Metric("claim_pct", "Claim success", GOALKEEPING, "pct", _rate("High Claim"), ("High Claim",)),
    Metric("sweeper", "Sweeper actions", GOALKEEPING, "count", _count("Sweeper Action"), ("Sweeper Action",)),
    Metric("sweeper_pct", "Sweeper success", GOALKEEPING, "pct", _rate("Sweeper Action"), ("Sweeper Action",)),
    Metric("dist_hand_pct", "Hand distribution accuracy", GOALKEEPING, "pct", _gk_dist("Hand"), ("Save",)),
    Metric("dist_gk_pct", "Goal-kick accuracy", GOALKEEPING, "pct", _gk_dist("Goal Kick"), ("Save",)),
    Metric("dist_open_pct", "Open-play kick accuracy", GOALKEEPING, "pct", _gk_dist("Open Play Kick"), ("Save",)),
)
METRIC_BY_KEY = {m.key: m for m in METRICS}

# headline KPIs and section order per position — what a scout looks at first for that role
PROFILES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "GK": (("saves", "save_pct", "goals_conceded", "high_claims", "claim_pct", "sweeper", "pass_pct",
            "long_pass_pct"), (GOALKEEPING, PASSING, POSSESSION, DEFENDING)),
    "CB": (("aerial_pct", "ground_duel_pct", "interceptions", "clearances", "blocks", "prog_passes", "pass_pct",
            "long_pass_pct"), (DEFENDING, PASSING, POSSESSION, ATTACKING)),
    "FB": (("crosses", "cross_pct", "prog_passes", "key_passes", "ground_duel_pct", "tackles", "recoveries",
            "dribbled_past"), (DEFENDING, CROSSING, PASSING, ATTACKING, POSSESSION)),
    "DM": (("passes", "pass_pct", "prog_passes", "line_breaking", "interceptions", "recoveries", "ground_duel_pct",
            "pass_up_pct"), (PASSING, DEFENDING, POSSESSION, ATTACKING)),
    "CM": (("passes", "pass_pct", "prog_passes", "key_passes", "prog_carries", "recoveries", "duel_pct",
            "f3_entries"), (PASSING, ATTACKING, DEFENDING, POSSESSION, CROSSING)),
    "AM": (("key_passes", "assists", "shots", "goals", "dribble_pct", "rec_lines", "box_entries", "losses"),
           (ATTACKING, PASSING, CROSSING, POSSESSION, DEFENDING)),
    "W": (("dribbles", "dribble_pct", "crosses", "cross_pct", "key_passes", "shots", "goals", "prog_carries"),
          (ATTACKING, CROSSING, PASSING, POSSESSION, DEFENDING)),
    "ST": (("shots", "shots_on_target", "goals", "conversion", "aerial_pct", "rec_behind", "key_passes", "presses"),
           (ATTACKING, PASSING, POSSESSION, DEFENDING, CROSSING)),
}
DEFAULT_SECTIONS = (ATTACKING, PASSING, CROSSING, POSSESSION, DEFENDING, GOALKEEPING)


def compute(m: Metric, df: pd.DataFrame, minutes: float | None = None) -> Value:
    r = m.fn(df)
    if m.kind == "count":
        per90 = r * 90.0 / minutes if minutes else None
        return Value(m, float(r), per90=per90)
    made, att = r
    return Value(m, 100.0 * made / att if att else np.nan, int(made), int(att))


def applies(m: Metric, position: str, df: pd.DataFrame) -> bool:
    """Shown when the metric's actions are in the position's window or were tagged anyway."""
    names = {a.name for a in T.actions_for(position)}
    if m.section == GOALKEEPING and position not in ("GK", ""):
        return bool(df["action"].isin(("Save", "Goal Conceded", "High Claim", "Sweeper Action")).any())
    if m.key.startswith("dist_"):
        return position == "GK" or bool(df["Distribution"].ne("").any())
    return bool(set(m.actions) & names) or bool(df["action"].isin(m.actions).any())


def headline(df: pd.DataFrame, position: str, minutes: float | None = None) -> list[Value]:
    keys = PROFILES.get(position, (("passes", "pass_pct", "shots", "goals", "duel_pct", "recoveries",
                                    "key_passes", "losses"), ()))[0]
    return [compute(METRIC_BY_KEY[k], df, minutes) for k in keys]


def sections(df: pd.DataFrame, position: str, minutes: float | None = None) -> list[tuple[str, list[Value]]]:
    order = PROFILES.get(position, ((), DEFAULT_SECTIONS))[1] or DEFAULT_SECTIONS
    order = list(order) + [s for s in DEFAULT_SECTIONS if s not in order]
    out = []
    for sec in order:
        vals = [compute(m, df, minutes) for m in METRICS if m.section == sec and applies(m, position, df)]
        if vals and (sec != GOALKEEPING or position == "GK" or any(v.value for v in vals if v.metric.kind == "count")):
            out.append((sec, vals))
    return out


def stats_table(df: pd.DataFrame, position: str, minutes: float | None = None) -> pd.DataFrame:
    rows = []
    for sec, vals in sections(df, position, minutes):
        for v in vals:
            rows.append({"Section": sec, "Metric": v.metric.label, "Value": v.display,
                         "Made / attempts": f"{v.made}/{v.attempts}" if v.metric.kind != "count" else "",
                         "Per 90": "" if v.per90 is None else round(v.per90, 2),
                         "Number": None if v.value != v.value else round(v.value, 1)})
    return pd.DataFrame(rows)


def per_match(df: pd.DataFrame, position: str, minutes: dict[str, float] | None = None) -> pd.DataFrame:
    """Metric × match, plus an "All matches" column."""
    minutes = minutes or {}
    matches = list(dict.fromkeys(df["match"]))
    cols: dict[str, dict] = {}
    for mname in matches + ["All matches"]:
        d = df if mname == "All matches" else df[df["match"] == mname]
        mins = sum(minutes.get(m, 0) for m in matches) if mname == "All matches" else minutes.get(mname)
        cols[mname] = {(sec, v.metric.label): v.display for sec, vals in sections(d, position, mins or None)
                       for v in vals}
    keys = list(dict.fromkeys(k for c in cols.values() for k in c))
    out = pd.DataFrame({m: [cols[m].get(k, "–") for k in keys] for m in cols},
                       index=pd.MultiIndex.from_tuples(keys, names=["Section", "Metric"]))
    return out.reset_index()


def zone_table(df: pd.DataFrame, actions: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Per zone: actions, tagged outcomes, successes, success %."""
    d = df if not actions else df[df["action"].isin(actions)]
    d = d[d["Zone"].ne("")]
    rows = []
    for i in range(1, T.N_COLS * T.N_LANES + 1):
        z = f"Z{i}"
        s = d[d["Zone"] == z]
        o = s[s["success"].notna()]
        made = int(o["success"].astype(bool).sum())
        rows.append({"Zone": z, "Third": T.zone_third(z), "Lane": T.zone_lane(z), "Actions": int(len(s)),
                     "With outcome": int(len(o)), "Successful": made,
                     "Success %": round(100 * made / len(o), 1) if len(o) else np.nan})
    return pd.DataFrame(rows)


def action_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in T.ACTIONS:
        s = df[df["action"] == a.name]
        if not len(s):
            continue
        o = s[s["success"].notna()]
        made = int(o["success"].astype(bool).sum())
        rows.append({"Action": a.name, "Category": a.category, "Total": int(len(s)), "With outcome": int(len(o)),
                     "Successful": made, "Success %": round(100 * made / len(o), 1) if len(o) else np.nan})
    return pd.DataFrame(rows)


def coverage(df: pd.DataFrame) -> dict[str, float]:
    """How completely the instances were labelled (tagging-quality check)."""
    if not len(df):
        return {}
    need_out = df["action"].isin([a.name for a in T.ACTIONS if "Outcome" in a.groups])
    shots = df["action"].eq("Shot")
    return {"Zone tagged": 100 * df["Zone"].ne("").mean(),
            "Outcome tagged": 100 * df.loc[need_out, "Outcome"].ne("").mean() if need_out.any() else np.nan,
            "Shot outcome tagged": 100 * df.loc[shots, "Shot Outcome"].ne("").mean() if shots.any() else np.nan}
