"""Counter-press analysis — what happens in the first N seconds (6–8 s) after we lose the ball.

For every open-play ball loss (our possession ends with the opponent taking control):
  * loss moment / place = the opponent's first action of their possession (interception, recovery,
    tackle, receive ...) — the instant and spot where they won the ball. (Counting from our own last
    pass would add that clip's pre-roll and inflate every time.) ``origin_*`` = our last action;
  * pressed?            = we tag a ``Press`` within the window (when the file has no Press tags,
                          any of our defensive actions — tackle, duel, interception, foul, block — counts);
  * seconds to press    = time from the loss to our first press;
  * regained?           = our NEXT possession starts with an open-play regain (recovery / interception /
                          tackle won) within the window — a restart after the ball goes out is not a regain;
  * seconds to regain, where, and whether that regain led to a shot / the final third.
Times use the event clock of the tagging file, so they inherit its tagging precision (± ~1 s).
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
_PRESSURE_FALLBACK = ("tackle", "duel", "interception", "foul", "block")
_ON_BALL = ("pass", "carry", "dribble", "shot", "receive")
CATEGORIES = ("Pressed & regained", "Pressed, not regained", "Regained without press", "Not pressed, not regained")


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
                 & ~ev["etype"].isin(["press", "foul", "sub_in", "sub_out"])]
        their = ev[(ev["poss_id"] == q["poss_id"]) & (ev["team"] == q["team"]) & ev["x"].notna()]
        if own.empty or their.empty:
            continue
        last = own.iloc[-1]
        won = their.iloc[0]
        t0, h = float(won["t"]), int(won["half"])
        if h != int(last["half"]):
            continue
        lx, ly = PITCH_L - float(won["x"]), PITCH_W - float(won["y"])   # their frame -> ours
        win = ev[(ev["half"] == h) & (ev["t"] > t0) & (ev["t"] <= t0 + window) & (ev["team"] == team)]
        pr = win[win["etype"] == "press"] if has_press else win[win["etype"].isin(_PRESSURE_FALLBACK)]
        # our next possession, and whether it is an open-play regain inside the window
        nxt = poss.iloc[i + 1:]
        nxt = nxt[(nxt["team"] == team) & (nxt["half"] == h)].head(1)
        regained, t_reg, rx, ry, reg_shot, reg_f3, reg_player = False, np.nan, np.nan, np.nan, False, False, ""
        if len(nxt):
            n = nxt.iloc[0]
            tr = float(start_t.get(n["poss_id"], np.nan))
            if n["start_type"] == "Regain" and tr - t0 <= window:
                regained, t_reg, rx, ry = True, tr - t0, n["start_x"], n["start_y"]
                reg_shot, reg_f3 = bool(n["shot"]), bool(n["reached_f3"])
                first = ev[(ev["poss_id"] == n["poss_id"]) & (ev["team"] == team)].head(1)
                reg_player = first["player"].iloc[0] if len(first) else ""
        pressed = len(pr) > 0
        rows.append({
            "loss_event_id": last.get("event_id", np.nan), "half": h, "minute": float(last["minute"]),
            "player": last["player"], "loss_action": _loss_label(last), "x": lx, "y": ly,
            "origin_x": float(last["x"]), "origin_y": float(last["y"]), "won_by_opponent_with": won["etype"],
            "third": str(third_of([lx])[0]), "pressed": pressed,
            "seconds_to_press": float(pr["t"].iloc[0] - t0) if pressed else np.nan,
            "presses": int(len(pr)), "first_presser": pr["player"].iloc[0] if pressed else "",
            "press_x": float(pr["x"].iloc[0]) if pressed and pd.notna(pr["x"].iloc[0]) else np.nan,
            "press_y": float(pr["y"].iloc[0]) if pressed and pd.notna(pr["y"].iloc[0]) else np.nan,
            "regained": regained, "seconds_to_regain": t_reg, "regain_x": rx, "regain_y": ry,
            "regained_by": reg_player, "regain_to_shot": reg_shot, "regain_to_final_third": reg_f3,
            "opp_reached_final_third": bool(q["reached_f3"]), "opp_shot": bool(q["shot"]),
            "category": CATEGORIES[0] if pressed and regained else CATEGORIES[1] if pressed
            else CATEGORIES[2] if regained else CATEGORIES[3],
            "window_s": window, "press_source": "Press tags" if has_press else "defensive actions",
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
        "avg_seconds_to_press": float(pr["seconds_to_press"].mean()) if len(pr) else np.nan,
        "regained": int(df["regained"].sum()), "regained_pct": pct(df["regained"].sum(), len(df)),
        "regain_rate_when_pressed": pct(pr["regained"].sum(), len(pr)),
        "regain_rate_without_press": pct(npr["regained"].sum(), len(npr)),
        "avg_seconds_to_regain": float(df["seconds_to_regain"].mean()) if df["regained"].any() else np.nan,
        "regain_to_shot": int(df["regain_to_shot"].sum()),
        "opp_reached_final_third": int(df["opp_reached_final_third"].sum()), "opp_shots": int(df["opp_shot"].sum()),
        "window_s": float(df["window_s"].iloc[0]), "press_source": df["press_source"].iloc[0],
    }


def by_third(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.groupby("third").agg(losses=("pressed", "size"), pressed=("pressed", "sum"), regained=("regained", "sum"))
    g["pressed %"] = g["pressed"] / g["losses"] * 100
    g["regained %"] = g["regained"] / g["losses"] * 100
    return g.reindex([t for t in ("Defensive third", "Middle third", "Final third") if t in g.index])
