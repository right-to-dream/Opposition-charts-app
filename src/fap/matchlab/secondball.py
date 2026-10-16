"""Second balls — who wins the loose ball after a contested first contact.

The tagging has no "second ball" tag, so it is derived from the event chain:

* **First contact** (the contested ball): a clearance (Clearance, or an Interception / Tackle labelled
  *Clear*), an aerial duel (label *Aerial* / *Head*), or a high / long ball (label *High* or ≥ 30 m) whose
  first touch is contested or nobody controls it. A long ball cleanly received by a team-mate or cleanly
  intercepted by the opponent is a *won first contact*, not a second-ball situation.
* **Second ball**: the first team to *control* the ball (receive, recovery, interception or tackle that
  is not a clearance, pass, carry, dribble, shot, save) within ``WINDOW`` seconds of the first contact.
  Further duels / clearances inside the window keep the ball loose. A stoppage first (out of play, foul,
  offside) or no control in the window → no winner.
* **What next**: did the winner reach the final third or shoot within ``FOLLOW_S`` seconds, before the
  opponent controlled the ball?

Locations are kept in each event's own frame (that team attacking x → 105); :func:`for_team` turns them
into one team's perspective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fap.matchlab.loader import PITCH_L, PITCH_W, MatchData
from fap.matchlab.memo import per_match

WINDOW = 5.0                 # seconds after the first contact
FOLLOW_S = 15.0              # "what happened next" horizon
LONG_M = 30.0
_SKIP = {"press", "sub_in", "sub_out", "half_start", "half_end"}
_STOP = {"ball_out", "offside", "foul", "foul_won"}
_CONTROL = {"receive", "recovery", "pass", "carry", "dribble", "shot", "save"}
KINDS = ("Long / high ball", "Clearance", "Aerial duel")


def _clear(lab) -> bool:
    return "clear" in lab


def _aerial(lab) -> bool:
    return "aerial" in lab or "head" in lab


def _contested(et: str, lab) -> bool:
    """Touches that keep the ball loose."""
    return et in ("duel", "clearance", "block") or (et in ("interception", "tackle") and _clear(lab)) or \
        (et == "pass" and "head" in lab)


def _controls(et: str, lab, outcome) -> bool:
    if et in ("interception", "tackle"):
        return not _clear(lab) and "lost" not in lab
    if et == "receive":
        return outcome != 0
    return et in _CONTROL


@per_match
def contests(md: MatchData) -> pd.DataFrame:
    ev = md.events
    ev = ev[~ev["derived"].astype(bool)] if "derived" in ev else ev
    ev = ev.sort_values("t", kind="stable")
    idx = ev.index.to_numpy()
    et, tm, pl = ev["etype"].to_numpy(), ev["team"].to_numpy(), ev["player"].to_numpy()
    lab, oc, t, half = ev["labels"].to_numpy(), ev["outcome"].to_numpy(float), ev["t"].to_numpy(float), ev["half"].to_numpy()
    x, y = ev["x"].to_numpy(float), ev["y"].to_numpy(float)
    length = ev["length"].to_numpy(float)
    n = len(ev)
    used = np.zeros(n, bool)
    rows = []

    def resolve(k0: int, kind: str, origin: int | None) -> None:
        """Scan from first contact k0 for the first control inside the window."""
        res, win = "No control", None
        for j in range(k0 + 1, n):
            if half[j] != half[k0] or t[j] - t[k0] > WINDOW:
                break
            if et[j] in _SKIP or np.isnan(x[j]) and et[j] not in _STOP:
                continue
            if et[j] in _STOP:
                res = "Out of play / foul"
                break
            if _contested(et[j], lab[j]):
                used[j] = True                         # still loose — part of this contest
                continue
            if _controls(et[j], lab[j], oc[j]):
                res, win = "Won", j
                break
        rec = {"event_id": ev["event_id"].iat[k0], "idx": idx[k0], "minute": ev["minute"].iat[k0], "half": half[k0],
               "t": t[k0], "kind": kind, "contact_team": tm[k0], "contact_player": pl[k0], "contact_etype": et[k0],
               "contact_x": x[k0], "contact_y": y[k0], "origin_team": tm[origin] if origin is not None else tm[k0],
               "result": res, "winner": "", "winner_player": "", "win_etype": "", "win_event_id": np.nan,
               "seconds": np.nan, "x": np.nan, "y": np.nan, "then_final_third": False, "then_shot": False}
        if win is not None:
            rec.update(winner=tm[win], winner_player=pl[win], win_etype=et[win], win_event_id=ev["event_id"].iat[win],
                       seconds=round(t[win] - t[k0], 2), x=x[win], y=y[win])
            f3 = shot = False
            for j in range(win, n):
                if half[j] != half[win] or t[j] - t[win] > FOLLOW_S:
                    break
                if tm[j] != tm[win] and _controls(et[j], lab[j], oc[j]):
                    break
                if tm[j] == tm[win]:
                    f3 = f3 or (not np.isnan(x[j]) and x[j] >= md.thirds[1])
                    shot = shot or et[j] == "shot"
            rec.update(then_final_third=bool(f3), then_shot=bool(shot))
        rows.append(rec)

    for k in range(n):
        if used[k]:
            continue
        e, lb = et[k], lab[k]
        if e == "pass" and ("high" in lb or (not np.isnan(length[k]) and length[k] >= LONG_M)):
            # the long ball's first touch
            first = None
            for j in range(k + 1, n):
                if half[j] != half[k] or t[j] - t[k] > 4.0:
                    break
                if et[j] in _SKIP or (tm[j] == tm[k] and pl[j] == pl[k] and et[j] != "receive"):
                    continue
                first = j
                break
            if first is None:
                used[k] = True
                resolve(k, KINDS[0], k)                # nobody touched it in time — loose ball
            elif et[first] in _STOP:
                continue
            elif _contested(et[first], lab[first]):
                used[first] = True
                kind = KINDS[2] if _aerial(lab[first]) else KINDS[1] if _clear(lab[first]) or et[first] == "clearance" else KINDS[0]
                resolve(first, kind, k)
            # else: clean first contact (received / intercepted) — not a second-ball situation
            continue
        if e == "clearance" or (e in ("interception", "tackle") and _clear(lb)):
            used[k] = True
            resolve(k, KINDS[1], None)
        elif e == "duel" and _aerial(lb):
            used[k] = True
            resolve(k, KINDS[2], None)
    return pd.DataFrame(rows)


def for_team(md: MatchData, team: str, half=None) -> pd.DataFrame:
    """Contests from ``team``'s perspective: won / lost / no winner, locations in its attacking frame."""
    c = contests(md)
    if c.empty:
        return c
    if half is not None:
        c = c[c["half"] == half]
    c = c.copy()
    flip_w = c["winner"].ne(team) & c["winner"].ne("")
    c.loc[flip_w, "x"] = PITCH_L - c.loc[flip_w, "x"]
    c.loc[flip_w, "y"] = PITCH_W - c.loc[flip_w, "y"]
    flip_c = c["contact_team"].ne(team)
    c.loc[flip_c, "contact_x"] = PITCH_L - c.loc[flip_c, "contact_x"]
    c.loc[flip_c, "contact_y"] = PITCH_W - c.loc[flip_c, "contact_y"]
    c["outcome"] = np.select([c["winner"].eq(team), c["winner"].ne("")], ["Won", "Lost"], "No winner")
    t1, t2 = md.thirds
    c["third"] = np.select([c["x"] < t1, c["x"] < t2], ["Defensive third", "Middle third"], "Final third")
    c.loc[c["x"].isna(), "third"] = ""
    return c


def summary(md: MatchData, team: str, half=None) -> dict:
    c = for_team(md, team, half)
    if c.empty:
        return {"contests": 0, "decided": 0, "won": 0, "won %": np.nan}
    d = c[c["outcome"] != "No winner"]
    w = d[d["outcome"] == "Won"]
    out = {"contests": len(c), "decided": len(d), "won": len(w),
           "won %": len(w) / len(d) * 100 if len(d) else np.nan,
           "won → final third": int(w["then_final_third"].sum()), "won → shot": int(w["then_shot"].sum()),
           "lost → opp. final third": int(d[d["outcome"] == "Lost"]["then_final_third"].sum()),
           "lost → opp. shot": int(d[d["outcome"] == "Lost"]["then_shot"].sum()),
           "avg seconds": float(d["seconds"].mean()) if len(d) else np.nan}
    for k in KINDS:
        g = d[d["kind"] == k]
        out[f"{k}: won"] = int((g["outcome"] == "Won").sum())
        out[f"{k}: decided"] = len(g)
    return out


def players(md: MatchData, team: str, half=None) -> pd.Series:
    c = for_team(md, team, half)
    if c.empty:
        return pd.Series(dtype=int)
    w = c[(c["outcome"] == "Won") & c["winner_player"].ne("") & c["winner_player"].ne("0")]
    return w["winner_player"].value_counts()
