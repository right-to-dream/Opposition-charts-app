"""Pass / carry taxonomy, possessions, shot xG and team + player tables.

All geometry is in the analysis frame (team attacks → x = 105, y = 0 right touchline).

Definitions (shown in the UI's methodology note)
  * Progressive pass / carry (Wyscout): the move brings the ball closer to the centre of the
    attacked goal by ≥ 30 m when start and end are in the own half, ≥ 15 m when it crosses the
    halfway line, ≥ 10 m when both are in the opponent half.
  * Direction: angle of the move — forward (|θ| ≤ 45°), backward (|θ| ≥ 135°), else lateral
    (left / right from the passer's view).
  * Length: short < 15 m, medium 15–30 m, long ≥ 30 m.
  * Final-third entry: starts before x = 70 and ends at or beyond it. Box entry: ends inside the
    penalty area having started outside it.
  * Channels (width): right flank | right half-space | centre | left half-space | left flank,
    split at the penalty-box and six-yard-box edges.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fap.matchlab.loader import PITCH_L, PITCH_W

if TYPE_CHECKING:
    from fap.matchlab.loader import MatchData

GOAL = (PITCH_L, PITCH_W / 2)
THIRDS = (35.0, 70.0)
BOX_X = PITCH_L - 16.5
BOX_Y = (PITCH_W / 2 - 20.16, PITCH_W / 2 + 20.16)
SIX_Y = (PITCH_W / 2 - 9.16, PITCH_W / 2 + 9.16)
CHANNEL_EDGES = (0.0, BOX_Y[0], SIX_Y[0], SIX_Y[1], BOX_Y[1], PITCH_W)
CHANNELS = ("Right flank", "Right half-space", "Centre", "Left half-space", "Left flank")
THIRD_NAMES = ("Defensive third", "Middle third", "Final third")

SET_PIECE_KINDS = ("corner", "free kick", "throw in", "goal kick", "penalty", "kick off")
DEF_ACTIONS = ("tackle", "interception", "recovery", "block", "clearance", "duel", "foul", "press")
PPDA_ACTIONS = ("tackle", "interception", "foul", "duel")


def ppda_mask(df: pd.DataFrame) -> pd.Series:
    """Defensive actions counted by PPDA: tackles, interceptions (not 'Clear' — those are clearances),
    fouls and duels, in the defending team's attacking 60%."""
    clear = df["labels"].map(lambda l: "clear" in l)
    return df["etype"].isin(PPDA_ACTIONS) & ~(df["etype"].eq("interception") & clear) & (df["x"] > 0.4 * PITCH_L)


# ------------------------------------------------------------------ geometry helpers
def channel_of(y: np.ndarray | pd.Series) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    idx = np.clip(np.searchsorted(CHANNEL_EDGES, yy, side="right") - 1, 0, 4)
    out = np.array(CHANNELS, dtype=object)[idx]
    return np.where(np.isnan(yy), "", out)


def third_of(x, thirds=THIRDS) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    out = np.where(xx < thirds[0], THIRD_NAMES[0], np.where(xx < thirds[1], THIRD_NAMES[1], THIRD_NAMES[2]))
    return np.where(np.isnan(xx), "", out)


def in_box(x, y) -> np.ndarray:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (x >= BOX_X) & (y >= BOX_Y[0]) & (y <= BOX_Y[1])


def dist_to_goal(x, y) -> np.ndarray:
    return np.hypot(GOAL[0] - np.asarray(x, float), GOAL[1] - np.asarray(y, float))


def progressive_mask(x, y, ex, ey) -> np.ndarray:
    x, ex = np.asarray(x, float), np.asarray(ex, float)
    gain = dist_to_goal(x, y) - dist_to_goal(ex, ey)
    half = PITCH_L / 2
    own_own = (x < half) & (ex < half)
    opp_opp = (x >= half) & (ex >= half)
    need = np.where(own_own, 30.0, np.where(opp_opp, 10.0, 15.0))
    ok = (gain >= need) & ~np.isnan(gain)
    return ok


# ------------------------------------------------------------------ enrichment
def _has(labels, *words) -> np.ndarray:
    return np.array([any(w in ls for w in words) for ls in labels], dtype=bool)


def enrich(md: "MatchData") -> None:
    """Add every derived column in place (called by :func:`fap.matchlab.load_match`)."""
    ev = md.events
    labels = ev["labels"]
    dx = ev["end_x"] - ev["x"]
    dy = ev["end_y"] - ev["y"]
    ev["length"] = np.hypot(dx, dy)
    ev["angle"] = np.degrees(np.arctan2(dy, dx))       # 0 = straight forward, + = toward left
    a = ev["angle"].abs()
    ev["direction"] = np.select([ev["length"].isna(), a <= 45, a >= 135, dy > 0],
                                ["", "Forward", "Backward", "Left"], default="Right")
    ev["length_class"] = np.select([ev["length"].isna(), ev["length"] < 15, ev["length"] < 30],
                                   ["", "Short", "Medium"], default="Long")
    ev["progressive"] = progressive_mask(ev["x"], ev["y"], ev["end_x"], ev["end_y"]) & ev["etype"].isin(["pass", "carry"])
    ev["third"] = third_of(ev["x"])
    ev["end_third"] = third_of(ev["end_x"])
    ev["channel"] = channel_of(ev["y"])
    ev["end_channel"] = channel_of(ev["end_y"])
    ev["f3_entry"] = (ev["x"] < THIRDS[1]) & (ev["end_x"] >= THIRDS[1])
    ev["box_entry"] = in_box(ev["end_x"], ev["end_y"]) & ~in_box(ev["x"], ev["y"])
    ev["switch"] = (dy.abs() >= 30) & ev["etype"].eq("pass")

    is_pass = ev["etype"].eq("pass").to_numpy()
    ev["cross"] = _has(labels, "cross") & is_pass
    ev["through_ball"] = _has(labels, "throughball", "through ball") & is_pass
    ev["cutback"] = _has(labels, "cutback") & is_pass
    ev["key_pass"] = _has(labels, "key pass", "assist") & is_pass
    ev["assist"] = _has(labels, "assist") & is_pass
    ev["line_break"] = _has(labels, "line break") & is_pass
    ev["line"] = np.select([_has(labels, "first"), _has(labels, "mid"), _has(labels, "last")],
                           ["First line", "Midfield line", "Last line"], default="")
    ev.loc[~ev["line_break"], "line"] = ""
    ev["height"] = np.select([_has(labels, "ground"), _has(labels, "low"), _has(labels, "high")],
                             ["Ground", "Low", "High"], default="")
    sp = np.array([next((k for k in SET_PIECE_KINDS if k in ls), "") for ls in labels], dtype=object)
    ev["set_piece_kind"] = sp
    ev["set_piece"] = (sp != "") | ev.get("lab_set_piece", False)
    ev["open_play"] = ~ev["set_piece"]
    ev["under_pressure"] = _has(labels, "under pressure")
    ev["pass_type"] = np.select(
        [ev["set_piece_kind"].eq("corner"), ev["set_piece_kind"].eq("free kick"), ev["set_piece_kind"].eq("throw in"),
         ev["set_piece_kind"].eq("goal kick"), ev["cross"], ev["cutback"], ev["through_ball"], ev["switch"],
         ev["line_break"], ev["length_class"].eq("Long")],
        ["Corner", "Free kick", "Throw-in", "Goal kick", "Cross", "Cutback", "Through ball", "Switch",
         "Line-breaking", "Long ball"], default="Regular")
    ev.loc[~ev["etype"].eq("pass"), "pass_type"] = ""

    _shots(md)
    _possessions(md)
    from fap.matchlab import phases
    phases.tag_phases(md)


# ------------------------------------------------------------------ possessions
_CONTROL = {"pass", "carry", "dribble", "shot", "recovery"}
_DEAD = {"ball_out", "foul", "offside", "half_start", "half_end"}


def _is_control(et: str, outcome: float, labels: tuple) -> bool:
    if et in _CONTROL:
        return True
    if et == "receive":
        return outcome != 0.0
    if et in ("interception", "tackle"):
        return "won" in labels
    return False


def _possessions(md: "MatchData") -> None:
    ev = md.events
    n = len(ev)
    pid = np.full(n, -1)
    cur_team, cur_id, dead = None, -1, True
    et, tm, oc, lb = ev["etype"].to_numpy(), ev["team"].to_numpy(), ev["outcome"].to_numpy(float), ev["labels"].to_numpy()
    for i in range(n):
        if et[i] in ("half_start", "half_end"):
            dead = True
            continue
        if tm[i] and _is_control(et[i], oc[i], lb[i]) and (tm[i] != cur_team or dead):
            cur_id += 1
            cur_team, dead = tm[i], False
        pid[i] = cur_id
        if et[i] in _DEAD or et[i] == "shot":
            dead = True
    ev["poss_id"] = pid
    owner = {}
    for i in range(n):
        if pid[i] >= 0 and pid[i] not in owner and tm[i] and _is_control(et[i], oc[i], lb[i]):
            owner[pid[i]] = tm[i]
    ev["poss_team"] = ev["poss_id"].map(owner).fillna("")
    ev["in_possession"] = ev["team"].eq(ev["poss_team"]) & ev["team"].ne("")

    rows = []
    for p, g in ev[ev["poss_id"] >= 0].groupby("poss_id", sort=True):
        team = owner.get(p, "")
        own = g[g["team"] == team]
        if own.empty:
            continue
        first = own.iloc[0]
        last = own.iloc[-1]
        xs = pd.concat([own["x"], own["end_x"]]).dropna()
        passes = own[own["etype"] == "pass"]
        shot = own["etype"].eq("shot").any()
        goal = bool(own["shot_goal"].any()) if "shot_goal" in own else _has(own[own["etype"] == "shot"]["labels"], "goal").any()
        nxt = ev.iloc[g.index[-1] + 1] if g.index[-1] + 1 < n else None
        if shot:
            end_reason = "Shot"
        elif nxt is not None and nxt["etype"] in ("half_end", "half_start"):
            end_reason = "Half end"
        elif g["etype"].isin(["ball_out", "foul", "offside"]).any():
            # the FIRST dead ball ends the possession (later presses etc. are tagged in the same id)
            dead = g[g["etype"].isin(["ball_out", "foul", "offside"])].iloc[0]
            end_reason = "Out of play" if dead["etype"] == "ball_out" else dead["tag"]
        else:
            end_reason = "Lost"
        start_type = ("Regain" if first["etype"] in ("recovery", "interception", "tackle")
                      else first["set_piece_kind"].title() if first["set_piece_kind"] else "Open play")
        if start_type == "Open play" and first["etype"] == "receive":
            start_type = "Open play"
        end_x = last["end_x"] if pd.notna(last.get("end_x")) and last["etype"] in ("pass", "carry") and last["outcome"] != 0 else last["x"]
        rows.append(dict(
            poss_id=p, team=team, half=int(first["half"]), start_min=float(first["minute"]),
            duration=float(max(0.0, own["t"].max() - own["t"].min())),
            start_x=float(first["x"]), start_y=float(first["y"]),
            end_x=float(end_x) if pd.notna(end_x) else np.nan, max_x=float(xs.max()) if len(xs) else np.nan,
            n_passes=int(len(passes)), n_completed=int((passes["outcome"] == 1).sum()),
            n_actions=int(len(own)), players=tuple(dict.fromkeys(p for p in own["player"] if p)),
            start_type=start_type, transition=bool(first["lab_transition"]),
            reached_f3=bool((xs >= THIRDS[1]).any()),
            reached_box=bool(in_box(pd.concat([own["x"], own["end_x"]]), pd.concat([own["y"], own["end_y"]])).any()),
            shot=bool(shot), goal=bool(goal), end_reason=end_reason,
        ))
    poss = pd.DataFrame(rows)
    if not poss.empty:
        poss["start_third"] = third_of(poss["start_x"])
        poss["end_third"] = third_of(poss["end_x"])
        poss["reach_third"] = third_of(poss["max_x"])
    md.possessions = poss


# ------------------------------------------------------------------ shots + xG
def _shots(md: "MatchData") -> None:
    ev = md.events
    s = ev["etype"].eq("shot")
    lab = ev["labels"]
    ev["shot_goal"] = s & _has(lab, "goal")
    ev["shot_blocked"] = s & _has(lab, "blocked")
    ev["shot_off"] = s & _has(lab, "off target")
    nxt_save = ev["etype"].shift(-1).eq("save") & ev["team"].shift(-1).ne(ev["team"])
    ev["shot_on"] = s & (ev["shot_goal"] | _has(lab, "saved", "on target") | (nxt_save & ~ev["shot_blocked"] & ~ev["shot_off"]))
    ev["shot_result"] = np.select([ev["shot_goal"], ev["shot_on"], ev["shot_blocked"], ev["shot_off"]],
                                  ["Goal", "Saved", "Blocked", "Off target"], default="")
    ev.loc[~s, "shot_result"] = ""
    ev["body_part"] = np.select([_has(lab, "head"), _has(lab, "left foot"), _has(lab, "right foot")],
                                ["Head", "Left Foot", "Right Foot"], default="")
    ev["xg"] = np.nan
    if not s.any():
        return
    idx = ev.index[s]
    assist_type, assisted = [], []
    for i in idx:
        prev = ev.loc[max(0, i - 3):i - 1]
        pp = prev[(prev["etype"] == "pass") & (prev["team"] == ev.at[i, "team"]) & (prev["outcome"] != 0)]
        if len(pp):
            q = pp.iloc[-1]
            assisted.append(True)
            assist_type.append("cross" if q["cross"] else "through_ball" if q["through_ball"] else "cutback" if q["cutback"] else "pass")
        else:
            assisted.append(False)
            assist_type.append("")
    shots = pd.DataFrame({
        "event_type": "shot", "x": ev.loc[idx, "x"] / PITCH_L * 100.0, "y": ev.loc[idx, "y"] / PITCH_W * 100.0,
        "body_part": ev.loc[idx, "body_part"].str.lower(),
        "set_piece": ev.loc[idx, "set_piece_kind"].where(ev.loc[idx, "set_piece_kind"].isin(["free kick", "corner", "penalty"]), ""),
        "assisted": assisted, "assist_type": assist_type,
    }, index=idx)
    xg = None
    try:
        from fap.xg.services import xg_service
        xg = xg_service.score_shots(shots, on_invalid="nan")[xg_service.OUTPUT_COLUMN].to_numpy(float)
        md.notes.append("xG: FAP internal xG model.")
    except Exception:  # noqa: BLE001 — fall back to a transparent distance/angle model
        xg = None
    if xg is None or np.all(np.isnan(xg)):
        xg = simple_xg(ev.loc[idx, "x"].to_numpy(float), ev.loc[idx, "y"].to_numpy(float),
                       ev.loc[idx, "body_part"].eq("Head").to_numpy())
        md.notes.append("xG: distance/angle logistic model (internal model unavailable).")
    ev.loc[idx, "xg"] = np.clip(np.nan_to_num(xg, nan=0.03), 0.0, 0.99)


def simple_xg(x, y, header) -> np.ndarray:
    """Transparent logistic xG on distance + goal-mouth angle (headers penalised)."""
    gx, gy = GOAL
    d = np.hypot(gx - x, gy - y)
    a = np.abs(np.arctan2(7.32 * (gx - x), (gx - x) ** 2 + (gy - y) ** 2 - (7.32 / 2) ** 2))
    z = -1.1 - 0.105 * d + 1.6 * a - 0.9 * np.asarray(header, float)
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ tables
def _pct(a: float, b: float) -> float:
    return float(a) / float(b) * 100.0 if b else float("nan")


def team_summary(md: "MatchData", team: str, half: int | None = None) -> dict:
    ev = md.events if half is None else md.events[md.events["half"] == half]
    opp = md.opponent(team)
    me = ev[ev["team"] == team]
    them = ev[ev["team"] == opp]
    p = me[me["etype"] == "pass"]
    op = them[them["etype"] == "pass"]
    comp = p["outcome"] == 1
    shots = me[me["etype"] == "shot"]
    d = me[me["etype"].isin(DEF_ACTIONS)]
    # PPDA: opponent passes in their own 60% / our defensive actions in our attacking 60%
    opp_pass_zone = op[op["x"] < 0.6 * PITCH_L]
    our_def_zone = me[ppda_mask(me)]
    f3_me = (p["x"] >= THIRDS[1]).sum()
    f3_them = (op["x"] >= THIRDS[1]).sum()
    poss = md.possessions
    poss = poss if half is None or poss.empty else poss[poss["half"] == half]
    mine = poss[poss["team"] == team] if not poss.empty else poss
    duels = me[me["etype"] == "duel"]
    return {
        "Passes": len(p), "Pass accuracy %": _pct(comp.sum(), len(p)),
        "Possession % (passes)": _pct(len(p), len(p) + len(op)),
        "Field tilt %": _pct(f3_me, f3_me + f3_them),
        "Progressive passes": int((p["progressive"] & comp).sum()),
        "Progressive carries": int(me[(me["etype"] == "carry") & me["progressive"]].shape[0]),
        "Final-third entries": int((p["f3_entry"] & comp).sum()),
        "Box entries": int((p["box_entry"] & comp).sum() + me[(me["etype"] == "carry") & me["box_entry"]].shape[0]),
        "Line-breaking passes": int((p["line_break"] & comp).sum()),
        "Crosses (completed)": f"{int((p['cross'] & comp).sum())}/{int(p['cross'].sum())}",
        "Long balls %": _pct((p["length_class"] == "Long").sum(), len(p)),
        "Shots": len(shots), "On target": int(shots["shot_on"].sum()), "Goals": int(shots["shot_goal"].sum()),
        "xG": float(shots["xg"].sum()),
        "Possessions": len(mine), "Passes / possession": float(mine["n_passes"].mean()) if len(mine) else float("nan"),
        "PPDA": len(opp_pass_zone) / len(our_def_zone) if len(our_def_zone) else float("nan"),
        "Defensive actions": len(d),
        "Def. action height (m)": float(d["x"].mean()) if len(d) else float("nan"),
        "Recoveries": int((me["etype"] == "recovery").sum()),
        "Interceptions": int((me["etype"] == "interception").sum()),
        "Aerial duels won %": _pct((duels["outcome"] == 1).sum(), duels["outcome"].notna().sum()),
    }


def player_table(md: "MatchData", team: str, half: int | None = None) -> pd.DataFrame:
    ev = md.events if half is None else md.events[md.events["half"] == half]
    me = ev[(ev["team"] == team) & ev["player"].ne("") & ev["player"].ne("0")]
    rows = []
    for pl, g in me.groupby("player"):
        p = g[g["etype"] == "pass"]
        comp = p["outcome"] == 1
        c = g[g["etype"] == "carry"]
        rec = ev[(ev["etype"] == "pass") & (ev["team"] == team) & (ev["receiver"] == pl) & (ev["outcome"] == 1)]
        rows.append({
            "Player": f"#{pl}", "Actions": int((~g["derived"]).sum()),
            "Passes": len(p), "Acc %": round(_pct(comp.sum(), len(p)), 1) if len(p) else np.nan,
            "Prog. passes": int((p["progressive"] & comp).sum()),
            "Prog. carries": int(c["progressive"].sum()),
            "Prog. received": int(rec["progressive"].sum()),
            "F3 entries": int((p["f3_entry"] & comp).sum() + c["f3_entry"].sum()),
            "Box entries": int((p["box_entry"] & comp).sum() + c["box_entry"].sum()),
            "Line breaks": int((p["line_break"] & comp).sum()),
            "Key passes": int(p["key_pass"].sum()),
            "Crosses": f"{int((p['cross'] & comp).sum())}/{int(p['cross'].sum())}",
            "Shots": int((g["etype"] == "shot").sum()), "xG": round(float(g["xg"].sum()), 2),
            "Def. actions": int(g["etype"].isin(DEF_ACTIONS).sum()),
            "Recoveries": int((g["etype"] == "recovery").sum()),
            "Losses": int(((g["etype"] == "pass") & (g["outcome"] == 0)).sum() + (g["etype"] == "ball_lost").sum()
                          + ((g["etype"] == "dribble") & (g["outcome"] == 0)).sum()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Actions", ascending=False).reset_index(drop=True)
    return out


def pass_breakdown(md: "MatchData", team: str, half: int | None = None, player: str | None = None) -> dict[str, pd.DataFrame]:
    """Attempted / completed counts per pass category (type, direction, length, height, line, channel)."""
    p = md.passes(team)
    if half is not None:
        p = p[p["half"] == half]
    if player:
        p = p[p["player"] == player]
    out = {}
    for col, order in (("pass_type", None), ("direction", ["Forward", "Left", "Right", "Backward"]),
                       ("length_class", ["Short", "Medium", "Long"]), ("height", ["Ground", "Low", "High"]),
                       ("line", ["First line", "Midfield line", "Last line"]), ("channel", list(CHANNELS)),
                       ("third", list(THIRD_NAMES))):
        g = p[p[col] != ""].groupby(col)["outcome"].agg(attempted="size", completed=lambda s: int((s == 1).sum()))
        if order:
            g = g.reindex([o for o in order if o in g.index])
        else:
            g = g.sort_values("attempted", ascending=False)
        g["accuracy"] = g["completed"] / g["attempted"] * 100
        out[col] = g
    return out
