"""Counter-press analysis — what happens in the first N seconds (6–8 s) after we lose the ball.

For every open-play ball loss (our possession ends with the opponent taking control):
  * open play only     — a loss followed by a dead ball (ball out / foul / offside) before the
                         opponent's first action, or by an opponent restart, is not counted;
  * loss moment / place = the opponent's first action of their possession (interception, recovery,
    tackle, receive ...) — the instant and spot where they won the ball. Exception: when our own last
    action is an explicit "Ball Lost" or the opponent's first tagged action comes > 3 s after our last
    action (a gap in the tagging), the loss moment is our own last action. ``origin_*`` = our last action;
  * pressure evidence  = our ``Press`` tag, our ``Duel`` / ``Tackle``, or an opponent action labelled
                         "Under Press(ure)" (when the file has no Press tags at all, our interceptions /
                         fouls / blocks count too);
  * pressed?           = pressure evidence inside the window, before any regain; OR a LATE counter-press:
                         pressure in the first ``LATE_EXTRA`` s after the window, while the opponent has
                         not yet completed a pass (they are still reacting, not in a settled possession);
  * regained?          = our NEXT possession starts with an open-play regain (recovery / interception /
                         tackle won) inside the window — or inside the late extension when it follows our
                         counter-press. A restart after the ball goes out is never a regain.
Only each tag's START time is used (clip end times are an automatic fixed length, not real durations).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fap.matchlab.loader import PITCH_L, PITCH_W
from fap.matchlab.metrics import third_of

if TYPE_CHECKING:
    from fap.matchlab.loader import MatchData

DEFAULT_WINDOW = 8.0
LATE_EXTRA = 4.0          # a counter-press may start up to this many seconds after the window
TAG_GAP = 3.0             # opponent's first action this much later than ours = tagging gap
_PRESSURE_FALLBACK = ("tackle", "duel", "interception", "foul", "block")
_PRESSURE_ACTIONS = ("duel", "tackle")
_DEAD = ("ball_out", "offside", "foul")
_RESTARTS = ("throw in", "corner", "goal kick", "free kick", "kick off", "penalty")
_ON_BALL = ("pass", "carry", "dribble", "shot", "receive")
CATEGORIES = ("Pressed & regained", "Pressed, not regained", "Regained without press", "Not pressed, not regained")


def _evidence(ev: pd.DataFrame, team: str, opp: str, h: int, t_from: float, t_to: float, has_press: bool) -> pd.DataFrame:
    """Every sign of our pressure between two moments: (t, kind, player, x, y)."""
    g = ev[(ev["half"] == h) & (ev["t"] > t_from) & (ev["t"] <= t_to)]
    ours = g[g["team"] == team]
    if has_press:
        mine = ours[ours["etype"].isin(("press",) + _PRESSURE_ACTIONS)]
    else:
        mine = ours[ours["etype"].isin(_PRESSURE_FALLBACK)]
    kind = mine["etype"].map({"press": "Press tag", "duel": "Duel", "tackle": "Tackle"}).fillna("Defensive action")
    rows = [pd.DataFrame({"t": mine["t"], "kind": kind, "player": mine["player"], "x": mine["x"], "y": mine["y"]})]
    up = g[(g["team"] == opp) & g["labels"].map(lambda l: "under pressure" in l)]
    if len(up):
        rows.append(pd.DataFrame({"t": up["t"], "kind": "Opponent under pressure", "player": "",
                                  "x": PITCH_L - up["x"], "y": PITCH_W - up["y"]}))
    out = pd.concat(rows) if rows else pd.DataFrame(columns=["t", "kind", "player", "x", "y"])
    return out.sort_values("t", kind="stable")


def analyse(md: "MatchData", team: str, window: float | None = None, half: int | None = None) -> pd.DataFrame:
    window = float(window or getattr(md, "cp_window", DEFAULT_WINDOW))
    ev = md.events
    poss = md.possessions
    if poss.empty:
        return pd.DataFrame()
    poss = poss.sort_values("poss_id").reset_index(drop=True)
    start_t = ev[ev["poss_id"] >= 0].groupby("poss_id")["t"].min()
    has_press = bool(((ev["team"] == team) & (ev["etype"] == "press")).any())
    rows = []
    for i in range(len(poss) - 1):
        p, q = poss.iloc[i], poss.iloc[i + 1]
        if p["team"] != team or p["end_reason"] != "Lost" or q["team"] == team:
            continue
        if half is not None and p["half"] != half:
            continue
        own = ev[(ev["poss_id"] == p["poss_id"]) & (ev["team"] == team) & ev["x"].notna()
                 & ~ev["etype"].isin(["press", "foul", "sub_in", "sub_out", "ball_out", "offside"])]
        their = ev[(ev["poss_id"] == q["poss_id"]) & (ev["team"] == q["team"]) & ev["x"].notna()]
        if own.empty or their.empty:
            continue
        last = own.iloc[-1]
        won = their.iloc[0]
        h = int(won["half"])
        if h != int(last["half"]):
            continue
        # open play only: no dead ball between our last action and their first, no restart
        between = ev.loc[last.name + 1: won.name - 1] if won.name > last.name else ev.iloc[0:0]
        if between["etype"].isin(_DEAD).any() or any(r in won["labels"] for r in _RESTARTS):
            continue
        # loss moment
        explicit = last["etype"] in ("error", "ball_lost") or "ball lost" in last["labels"]
        gap = float(won["t"] - last["t"])
        use_ours = gap > 0 and (explicit or gap > TAG_GAP)
        t0 = float(last["t"]) if use_ours else float(won["t"])
        if use_ours:
            lx, ly = float(last["x"]), float(last["y"])
        else:
            lx, ly = PITCH_L - float(won["x"]), PITCH_W - float(won["y"])   # their frame -> ours
        opp = q["team"]
        # the opponent "settles" with their first completed pass
        op = ev[(ev["half"] == h) & (ev["team"] == opp) & (ev["t"] >= t0) & (ev["etype"] == "pass") & (ev["outcome"] == 1)]
        t_settle = float(op["t"].min()) if len(op) else np.inf
        late_end = t0 + window + LATE_EXTRA
        # our next possession: an open-play regain?
        nxt = poss.iloc[i + 1:]
        nxt = nxt[(nxt["team"] == team) & (nxt["half"] == h)].head(1)
        tr, n = np.nan, None
        if len(nxt) and nxt.iloc[0]["start_type"] == "Regain":
            n = nxt.iloc[0]
            tr = float(start_t.get(n["poss_id"], np.nan))
            if not (tr - t0 <= window + LATE_EXTRA):
                tr, n = np.nan, None
        # pressure before the regain (a regain made BY a tackle / duel is itself pressure)
        stop = tr if not np.isnan(tr) else late_end
        evd = _evidence(ev, team, opp, h, t0, min(stop, late_end), has_press)
        if n is not None:
            first_reg = ev[(ev["poss_id"] == n["poss_id"]) & (ev["team"] == team)].head(1)
            if len(first_reg) and first_reg["etype"].iloc[0] in _PRESSURE_ACTIONS + ("tackle",):
                r0 = first_reg.iloc[0]
                evd = pd.concat([evd, pd.DataFrame([{"t": r0["t"], "kind": r0["etype"].title(),
                                                     "player": r0["player"], "x": r0["x"], "y": r0["y"]}])])
            evd = evd[evd["t"] <= tr]
        in_win = evd[evd["t"] <= t0 + window]
        late = evd[(evd["t"] > t0 + window) & (evd["t"] < t_settle)]
        pressed = len(in_win) > 0 or len(late) > 0
        timing = "in window" if len(in_win) else ("late" if len(late) else "")
        first = in_win.iloc[0] if len(in_win) else (late.iloc[0] if len(late) else None)
        regained = False
        if n is not None:
            regained = (tr - t0 <= window) or (pressed and tr - t0 <= window + LATE_EXTRA)
        rx = ry = np.nan
        reg_shot = reg_f3 = False
        reg_player = ""
        if regained:
            rx, ry = n["start_x"], n["start_y"]
            reg_shot, reg_f3 = bool(n["shot"]), bool(n["reached_f3"])
            fr = ev[(ev["poss_id"] == n["poss_id"]) & (ev["team"] == team)].head(1)
            reg_player = fr["player"].iloc[0] if len(fr) else ""
        n_press = int((evd["kind"] == "Press tag").sum())
        rows.append({
            "loss_event_id": last.get("event_id", np.nan), "half": h, "minute": float(last["minute"]),
            "player": last["player"], "loss_action": _loss_label(last), "x": lx, "y": ly,
            "origin_x": float(last["x"]), "origin_y": float(last["y"]), "won_by_opponent_with": won["etype"],
            "won_event_id": won.get("event_id", np.nan),
            "loss_moment": "our last action" if use_ours else "opponent's first action",
            "third": str(third_of([lx])[0]), "pressed": pressed, "press_timing": timing,
            "press_evidence": first["kind"] if first is not None else "",
            "seconds_to_press": float(first["t"] - t0) if first is not None else np.nan,
            "presses": n_press, "first_presser": first["player"] if first is not None else "",
            "press_x": float(first["x"]) if first is not None and pd.notna(first["x"]) else np.nan,
            "press_y": float(first["y"]) if first is not None and pd.notna(first["y"]) else np.nan,
            "regained": regained, "seconds_to_regain": float(tr - t0) if regained else np.nan,
            "regain_x": rx, "regain_y": ry,
            "regained_by": reg_player, "regain_to_shot": reg_shot, "regain_to_final_third": reg_f3,
            "opp_reached_final_third": bool(q["reached_f3"]), "opp_shot": bool(q["shot"]),
            "category": CATEGORIES[0] if pressed and regained else CATEGORIES[1] if pressed
            else CATEGORIES[2] if regained else CATEGORIES[3],
            "window_s": window, "late_extra_s": LATE_EXTRA,
            "press_source": "Press / Duel / Tackle / opponent Under Press" if has_press else "defensive actions",
        })
    return pd.DataFrame(rows)


def _loss_label(r) -> str:
    if r["etype"] == "pass":
        return "Cross lost" if r.get("cross") else "Pass lost"
    if r["etype"] == "dribble":
        return "Dribble lost"
    if r["etype"] == "shot":
        return "Shot"
    if r["etype"] == "receive":
        return "Lost after receiving"
    if r["etype"] in ("error", "ball_lost"):
        return "Ball lost"
    if r["etype"] in ("recovery", "interception", "tackle", "clearance", "duel"):
        return f"Lost after {r['etype']}"
    return r["etype"].title()


def summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"losses": 0}
    pr = df[df["pressed"]]
    npr = df[~df["pressed"]]
    pct = lambda a, b: float(a) / b * 100 if b else np.nan  # noqa: E731
    return {
        "losses": len(df), "pressed": len(pr), "pressed_pct": pct(len(pr), len(df)),
        "late_presses": int((df["press_timing"] == "late").sum()),
        "avg_seconds_to_press": float(pr["seconds_to_press"].mean()) if len(pr) else np.nan,
        "regained": int(df["regained"].sum()), "regained_pct": pct(df["regained"].sum(), len(df)),
        "regain_rate_when_pressed": pct(pr["regained"].sum(), len(pr)),
        "regain_rate_without_press": pct(npr["regained"].sum(), len(npr)),
        "avg_seconds_to_regain": float(df["seconds_to_regain"].mean()) if df["regained"].any() else np.nan,
        "regain_to_shot": int(df["regain_to_shot"].sum()),
        "opp_reached_final_third": int(df["opp_reached_final_third"].sum()), "opp_shots": int(df["opp_shot"].sum()),
        "window_s": float(df["window_s"].iloc[0]), "late_extra_s": float(df["late_extra_s"].iloc[0]),
        "press_source": df["press_source"].iloc[0],
    }


def by_third(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.groupby("third").agg(losses=("pressed", "size"), pressed=("pressed", "sum"), regained=("regained", "sum"))
    g["pressed %"] = g["pressed"] / g["losses"] * 100
    g["regained %"] = g["regained"] / g["losses"] * 100
    return g.reindex([t for t in ("Defensive third", "Middle third", "Final third") if t in g.index])
