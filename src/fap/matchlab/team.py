"""Team deep-dive — detailed team statistics and the three phases with the ball.

* :func:`team_stats`   — every count with its attempts and success rate, grouped by category,
                          for any team (so both teams can be shown side by side).
* :func:`buildup`      — possessions that start in the own third: how they start, how / where the ball
                          leaves the own third, who builds, the goalkeeper's distribution, losses.
* :func:`progression`  — middle third → final third: entries by method / channel / player, lines
                          broken, switches, time and passes needed.
* :func:`final_third`  — channel usage, crosses by flank, box entries, chance creation, shot sources,
                          final-third losses.
All in the team's attacking frame (x → 105, y = 0 right touchline); thresholds follow ``md.thirds``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fap.matchlab.metrics import CHANNELS, PPDA_ACTIONS, PITCH_L, in_box, ppda_mask
from fap.matchlab.memo import per_match

if TYPE_CHECKING:
    from fap.matchlab.loader import MatchData


def _scope(md: "MatchData", team: str, half=None) -> pd.DataFrame:
    ev = md.events[md.events["team"] == team]
    return ev if half is None else ev[ev["half"] == half]


def _has(s: pd.Series, word: str) -> pd.Series:
    return s.map(lambda l: word in l)


def _row(metric: str, value, attempts=None, note: str = "") -> dict:
    v = float(value) if value is not None else np.nan
    a = float(attempts) if attempts is not None else np.nan
    return {"Metric": metric, "Value": v, "Attempts": a,
            "Success %": (v / a * 100) if (attempts is not None and a > 0) else np.nan, "Note": note}


# ------------------------------------------------------------------ team stats
@per_match
def team_stats(md: "MatchData", team: str, half=None) -> pd.DataFrame:
    """Long table: Category, Metric, Value, Attempts, Success %."""
    ev = _scope(md, team, half)
    allev = md.events if half is None else md.events[md.events["half"] == half]
    opp = md.opponent(team)
    t1, t2 = md.thirds
    p = ev[ev["etype"] == "pass"]
    ok = p["outcome"] == 1
    rows: list[tuple[str, dict]] = []
    add = lambda cat, r: rows.append((cat, r))  # noqa: E731

    # possession & sequences
    poss = md.possessions
    mine = poss[(poss["team"] == team) & ((poss["half"] == half) if half is not None else True)] if not poss.empty else poss
    opp_p = allev[(allev["team"] == opp) & (allev["etype"] == "pass")]
    add("Possession", _row("Possession share (passes)", len(p), len(p) + len(opp_p)))
    f3_me = int((p["x"] >= t2).sum())
    f3_op = int((opp_p["x"] >= t2).sum())
    add("Possession", _row("Field tilt (final-third passes)", f3_me, f3_me + f3_op))
    if len(mine):
        add("Possession", _row("Possessions", len(mine)))
        add("Possession", _row("Possessions with 3+ passes", int((mine["n_passes"] >= 3).sum()), len(mine)))
        add("Possession", _row("Possessions with 6+ passes", int((mine["n_passes"] >= 6).sum()), len(mine)))
        add("Possession", _row("Avg passes per possession", round(float(mine["n_passes"].mean()), 1)))
        add("Possession", _row("Avg possession duration (s)", round(float(mine["duration"].mean()), 1)))
        add("Possession", _row("Possessions reaching final third", int(mine["reached_f3"].sum()), len(mine)))
        add("Possession", _row("Possessions ending in a shot", int(mine["shot"].sum()), len(mine)))

    # passing
    def pr(label, mask):
        add("Passing", _row(label, int((mask & ok).sum()), int(mask.sum())))
    pr("Passes", pd.Series(True, index=p.index))
    for lc in ("Short", "Medium", "Long"):
        pr(f"{lc} passes", p["length_class"] == lc)
    for d in ("Forward", "Backward"):
        pr(f"{d} passes", p["direction"] == d)
    pr("Lateral passes", p["direction"].isin(["Left", "Right"]))
    for h in ("Ground", "Low", "High"):
        pr(f"{h} passes", p["height"] == h)
    pr("Progressive passes", p["progressive"])
    pr("Line-breaking passes", p["line_break"])
    for ln in ("First line", "Midfield line", "Last line"):
        pr(f"  · broke the {ln.lower()}", p["line"] == ln)
    pr("Switches of play", p["switch"])
    pr("Through balls", p["through_ball"])
    pr("Passes under pressure", p["under_pressure"])
    pr("Passes in own third", p["x"] < t1)
    pr("Passes in middle third", (p["x"] >= t1) & (p["x"] < t2))
    pr("Passes in final third", p["x"] >= t2)
    add("Passing", _row("Avg pass length (m)", round(float(p["length"].mean()), 1) if len(p) else np.nan))

    # attacking
    s = ev[ev["etype"] == "shot"]
    add("Attacking", _row("Shots", len(s)))
    add("Attacking", _row("Shots on target", int(s["shot_on"].sum()), len(s)))
    add("Attacking", _row("Goals", int(s["shot_goal"].sum()), len(s)))
    add("Attacking", _row("xG", round(float(s["xg"].sum()), 2)))
    add("Attacking", _row("Shots inside the box", int(in_box(s["x"], s["y"]).sum()), len(s)))
    add("Attacking", _row("Headed shots", int((s["body_part"] == "Head").sum()), len(s)))
    mv = ev[ev["etype"].isin(["pass", "carry"]) & ev["end_x"].notna()]
    mv_ok = (mv["outcome"] == 1) | (mv["etype"] == "carry")
    add("Attacking", _row("Final-third entries", int((mv["f3_entry"] & mv_ok).sum()), int(mv["f3_entry"].sum())))
    add("Attacking", _row("Box entries", int((mv["box_entry"] & mv_ok).sum()), int(mv["box_entry"].sum())))
    cr = p[p["cross"]]
    add("Attacking", _row("Crosses", int((cr["outcome"] == 1).sum()), len(cr)))
    add("Attacking", _row("Key passes", int(p["key_pass"].sum())))
    dr = ev[ev["etype"] == "dribble"]
    add("Attacking", _row("Dribbles", int((dr["outcome"] == 1).sum()), len(dr)))
    touch = ev[ev["etype"].isin(["receive", "pass", "carry", "dribble", "shot"]) & ev["x"].notna() & ~ev["derived"].astype(bool)]
    add("Attacking", _row("Touches in the box", int(in_box(touch["x"], touch["y"]).sum())))

    # duels
    du = ev[ev["etype"] == "duel"]
    won, lost = _has(du["labels"], "won"), _has(du["labels"], "lost")
    aer = _has(du["labels"], "aerial")
    add("Duels", _row("All duels — won", int(won.sum()), int((won | lost).sum()), f"{len(du)} duels tagged"))
    add("Duels", _row("Aerial duels — won", int((won & aer).sum()), int(((won | lost) & aer).sum())))
    add("Duels", _row("Ground duels — won", int((won & ~aer).sum()), int(((won | lost) & ~aer).sum())))
    tk = ev[ev["etype"] == "tackle"]
    add("Duels", _row("Tackles — won", int(_has(tk["labels"], "won").sum()), len(tk), "rest = cleared"))
    add("Duels", _row("Dribbles — completed", int((dr["outcome"] == 1).sum()), len(dr)))
    add("Duels", _row("Dribbled past (opponent beat us)", int((ev["etype"] == "dribbled_past").sum())))

    # defending
    it = ev[ev["etype"] == "interception"]
    add("Defending", _row("Interceptions — won the ball", int(_has(it["labels"], "won").sum()), len(it), "rest = cleared"))
    rec = ev[ev["etype"] == "recovery"]
    add("Defending", _row("Recoveries", len(rec)))
    for nm, a, b in (("own third", 0, t1), ("middle third", t1, t2), ("final third", t2, PITCH_L + 1)):
        add("Defending", _row(f"  · recoveries in {nm}", int(((rec["x"] >= a) & (rec["x"] < b)).sum()), len(rec)))
    add("Defending", _row("Clearances (incl. cleared interceptions)",
                          int((ev["etype"] == "clearance").sum() + _has(it["labels"], "clear").sum())))
    add("Defending", _row("Blocks", int((ev["etype"] == "block").sum())))
    add("Defending", _row("Presses tagged", int((ev["etype"] == "press").sum())))
    opp_ev = allev[allev["team"] == opp]
    num = opp_ev[(opp_ev["etype"] == "pass") & (opp_ev["x"] < 0.6 * PITCH_L)]
    den = ev[ppda_mask(ev)]
    add("Defending", _row("PPDA (lower = more intense)", round(len(num) / len(den), 2) if len(den) else np.nan,
                          note=f"{len(num)} opp. passes / {len(den)} actions"))
    d = ev[ev["etype"].isin(["tackle", "interception", "recovery", "block", "clearance", "duel", "foul", "press"])]
    add("Defending", _row("Avg defensive action height (m)", round(float(d["x"].mean()), 1) if len(d) else np.nan))
    from fap.matchlab import counterpress as CP
    cp = CP.analyse(md, team, md.cp_window, half)
    if not cp.empty:
        add("Defending", _row(f"Losses counter-pressed (≤{md.cp_window:.0f}s +late)", int(cp["pressed"].sum()), len(cp)))
        add("Defending", _row("Losses regained in the window", int(cp["regained"].sum()), len(cp)))
    os_ = opp_ev[opp_ev["etype"] == "shot"]
    add("Defending", _row("Shots conceded", len(os_)))
    add("Defending", _row("xG conceded", round(float(os_["xg"].sum()), 2)))

    # discipline & set pieces
    add("Discipline", _row("Fouls committed", int((ev["etype"] == "foul").sum())))
    add("Discipline", _row("Fouls won", int((ev["etype"] == "foul_won").sum())))
    add("Discipline", _row("Yellow cards", int(_has(ev["labels"], "yellow card").sum())))
    add("Discipline", _row("Red cards", int(_has(ev["labels"], "red card").sum())))
    add("Discipline", _row("Offsides", int((ev["etype"] == "offside").sum())))
    for k, nm in (("corner", "Corners"), ("free kick", "Free kicks"), ("throw in", "Throw-ins"), ("goal kick", "Goal kicks")):
        g = p[p["set_piece_kind"] == k]
        add("Set pieces", _row(f"{nm} — ball kept", int((g["outcome"] == 1).sum()), len(g)))

    # ball losses
    lo = ev[((ev["etype"] == "pass") & (ev["outcome"] == 0)) | ((ev["etype"] == "dribble") & (ev["outcome"] == 0))
            | (ev["etype"] == "ball_lost") | ((ev["etype"] == "error") & _has(ev["labels"], "ball lost"))]
    add("Ball losses", _row("Ball losses", len(lo)))
    for nm, a, b in (("own third", 0, t1), ("middle third", t1, t2), ("final third", t2, PITCH_L + 1)):
        add("Ball losses", _row(f"  · in {nm}", int(((lo["x"] >= a) & (lo["x"] < b)).sum()), len(lo)))
    add("Ball losses", _row("Errors leading to a shot", int(_has(ev["labels"], "error led to shot").sum()
                                                            + (ev["tag"].str.lower() == "error led to shot").sum())))
    out = pd.DataFrame([{"Category": c, **r} for c, r in rows])
    return out


def stats_side_by_side(md: "MatchData", half=None) -> pd.DataFrame:
    t = md.teams
    a = team_stats(md, t[0], half).rename(columns={"Value": f"{t[0]}", "Attempts": f"{t[0]} att", "Success %": f"{t[0]} %"})
    if len(t) < 2:
        return a
    b = team_stats(md, t[1], half).rename(columns={"Value": f"{t[1]}", "Attempts": f"{t[1]} att", "Success %": f"{t[1]} %"})
    a["_order"] = range(len(a))
    out = a.merge(b.drop(columns=["Note"]), on=["Category", "Metric"], how="left")   # keep the logical order
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ------------------------------------------------------------------ build-up
def _exit_action(md: "MatchData", poss_id: int, team: str):
    """First own action that takes the ball out of the own third (pass end / carry end / event x ≥ t1)."""
    t1 = md.thirds[0]
    g = md.events[(md.events["poss_id"] == poss_id) & (md.events["team"] == team) & md.events["x"].notna()]
    for _, e in g.iterrows():
        if e["etype"] in ("pass", "carry") and pd.notna(e["end_x"]) and e["x"] < t1 <= e["end_x"]:
            return e
        if e["etype"] in ("dribble", "receive", "shot") and e["x"] >= t1:
            return e
    return None


@per_match
def buildup(md: "MatchData", team: str, half=None) -> dict:
    t1 = md.thirds[0]
    poss = md.possessions
    b = poss[(poss["team"] == team) & (poss["start_x"] < t1)] if not poss.empty else poss
    if half is not None and not b.empty:
        b = b[b["half"] == half]
    out: dict = {"possessions": b}
    if b.empty:
        return out
    exits = []
    for _, s in b.iterrows():
        e = _exit_action(md, int(s["poss_id"]), team)
        if e is None:
            continue
        if e["etype"] == "pass":
            method = "Long pass" if e["length_class"] == "Long" else "Pass"
            succ = e["outcome"] == 1
            chan = e["end_channel"]
        elif e["etype"] == "carry":
            method, succ, chan = "Carry", True, e["end_channel"]
        else:
            method, succ, chan = e["etype"].title(), True, e["channel"]
        exits.append({"poss_id": s["poss_id"], "method": method, "channel": chan, "success": bool(succ),
                      "player": e["player"], "x": e["x"], "y": e["y"],
                      "end_x": e["end_x"] if pd.notna(e.get("end_x")) else e["x"],
                      "end_y": e["end_y"] if pd.notna(e.get("end_y")) else e["y"],
                      "minute": e["minute"], "event_id": e.get("event_id"), "reached_f3": bool(s["reached_f3"]),
                      "shot": bool(s["shot"])})
    out["exits"] = pd.DataFrame(exits)
    ev = _scope(md, team, half)
    own = ev[(ev["x"] < t1) & ev["x"].notna()]
    p = own[own["etype"] == "pass"]
    out["passes"] = p
    pairs = p[(p["outcome"] == 1) & p["receiver"].ne("") & p["receiver"].ne("0") & p["player"].ne("0")]
    out["pairs"] = pairs.groupby(["player", "receiver"]).size().sort_values(ascending=False)
    out["starters"] = b["players"].map(lambda t: t[0] if t else "").value_counts()
    out["losses"] = own[((own["etype"] == "pass") & (own["outcome"] == 0)) | ((own["etype"] == "dribble") & (own["outcome"] == 0))
                        | (own["etype"] == "ball_lost") | ((own["etype"] == "error") & _has(own["labels"], "ball lost"))]
    out["gk"] = goalkeeper_passes(md, team, half)
    return out


@per_match
def goalkeeper(md: "MatchData", team: str) -> str:
    """Our keeper from the squad's role codes, else the player whose actions are deepest."""
    ev = md.events[(md.events["team"] == team) & md.events["player"].ne("") & md.events["player"].ne("0")]
    if ev.empty:
        return ""
    try:
        from fap.matchlab.reference import Reference
        roles = Reference.bundled().squad.role if team == md.our_team else {}
    except Exception:
        roles = {}
    gks = [p for p in ev["player"].unique() if roles.get(p, "").upper() in ("GK", "GOALKEEPER")]
    if gks:
        return max(gks, key=lambda p: int((ev["player"] == p).sum()))
    depth = ev.groupby("player")["x"].mean()
    cnt = ev.groupby("player").size()
    depth = depth[cnt >= 8]
    return str(depth.idxmin()) if len(depth) and depth.min() < 20 else ""


def goalkeeper_passes(md: "MatchData", team: str, half=None) -> pd.DataFrame:
    gk = goalkeeper(md, team)
    ev = _scope(md, team, half)
    return ev[(ev["etype"] == "pass") & (ev["player"] == gk) & ev["end_x"].notna()] if gk else ev.iloc[0:0]


# ------------------------------------------------------------------ progression
@per_match
def progression(md: "MatchData", team: str, half=None) -> dict:
    t1, t2 = md.thirds
    ev = _scope(md, team, half)
    mv = ev[ev["etype"].isin(["pass", "carry"]) & ev["end_x"].notna() & ev["f3_entry"]]
    mv = mv.assign(success=(mv["outcome"] == 1) | (mv["etype"] == "carry"),
                   method=np.where(mv["etype"] == "carry", "Carry",
                                   np.where(mv["cross"], "Cross", np.where(mv["through_ball"], "Through ball",
                                            np.where(mv["length_class"] == "Long", "Long pass", "Pass")))))
    out = {"entries": mv}
    out["by_channel"] = mv.groupby("end_channel")["success"].agg(attempts="size", successful="sum").reindex(CHANNELS[::-1]).fillna(0)
    out["by_method"] = mv.groupby("method")["success"].agg(attempts="size", successful="sum").sort_values("attempts", ascending=False)
    pl = mv[mv["player"].ne("") & mv["player"].ne("0")]
    out["by_player"] = pl.groupby("player")["success"].agg(attempts="size", successful="sum").sort_values("successful", ascending=False)
    mid = ev[(ev["x"] >= t1) & (ev["x"] < t2)]
    p = mid[mid["etype"] == "pass"]
    out["middle_passes"] = p
    out["lines"] = p[p["line_break"]].groupby("line")["outcome"].agg(attempts="size", successful=lambda s: int((s == 1).sum()))
    out["switches"] = ev[(ev["etype"] == "pass") & ev["switch"]]
    poss = md.possessions
    pp = poss[(poss["team"] == team) & poss["reached_f3"] & (poss["start_x"] < t2)] if not poss.empty else poss
    if half is not None and not pp.empty:
        pp = pp[pp["half"] == half]
    times = []
    for _, s in pp.iterrows():
        g = md.events[(md.events["poss_id"] == s["poss_id"]) & (md.events["team"] == team)]
        first = g[(g["x"] >= t2) | (g["end_x"] >= t2)].head(1)
        if len(first):
            before = g.loc[: first.index[0]]
            times.append({"poss_id": s["poss_id"], "seconds": float(first["t"].iloc[0] - g["t"].min()),
                          "passes": int((before["etype"] == "pass").sum()), "start": s["start_type"],
                          "start_third": s["start_third"]})
    out["time_to_f3"] = pd.DataFrame(times)
    out["prog_origins"] = ev[(ev["etype"].isin(["pass", "carry"])) & ev["progressive"] & ev["end_x"].notna()
                             & ((ev["outcome"] == 1) | (ev["etype"] == "carry"))]
    return out


# ------------------------------------------------------------------ final third
@per_match
def final_third(md: "MatchData", team: str, half=None) -> dict:
    t2 = md.thirds[1]
    ev = _scope(md, team, half)
    f3 = ev[(ev["x"] >= t2) & ev["x"].notna() & ~ev["derived"].astype(bool)]
    out: dict = {"events": f3}
    p = f3[f3["etype"] == "pass"]
    rows = []
    for ch in CHANNELS[::-1]:
        g = f3[f3["channel"] == ch]
        gp = p[p["channel"] == ch]
        rows.append({"channel": ch, "actions": len(g), "passes": len(gp), "completed": int((gp["outcome"] == 1).sum()),
                     "key passes": int(gp["key_pass"].sum()), "crosses": int(gp["cross"].sum()),
                     "shots": int((g["etype"] == "shot").sum()),
                     "losses": int(((g["etype"].isin(["pass", "dribble"])) & (g["outcome"] == 0)).sum())})
    out["channels"] = pd.DataFrame(rows).set_index("channel")
    cr = ev[(ev["etype"] == "pass") & ev["cross"]]
    out["crosses"] = cr.assign(flank=np.where(cr["y"] > 34, "Left flank", "Right flank"))
    s = ev[ev["etype"] == "shot"].copy()
    src = []
    for i, r in s.iterrows():
        prev = md.events.loc[max(0, i - 4): i - 1]
        ap = prev[(prev["etype"] == "pass") & (prev["team"] == team) & (prev["outcome"] == 1)]
        if r["set_piece_kind"] == "penalty":
            src.append("Penalty")
        elif r["set_piece"]:
            src.append("Set piece")
        elif bool(r.get("transition")) or r["lab_transition"]:
            src.append("Transition / regain")
        elif len(ap) and ap.iloc[-1]["cross"]:
            src.append("After a cross")
        elif len(ap) and ap.iloc[-1]["through_ball"]:
            src.append("After a through ball")
        elif len(ap):
            src.append("After a pass")
        else:
            src.append("Individual / rebound")
    s["source"] = src
    out["shots"] = s
    kp = ev[(ev["etype"] == "pass") & (ev["key_pass"] | ev["assist"]) & ev["end_x"].notna()]
    out["key_passes"] = kp
    mv = ev[ev["etype"].isin(["pass", "carry"]) & ev["end_x"].notna() & ev["box_entry"]]
    out["box_entries"] = mv.assign(success=(mv["outcome"] == 1) | (mv["etype"] == "carry"))
    out["losses"] = f3[((f3["etype"] == "pass") & (f3["outcome"] == 0)) | ((f3["etype"] == "dribble") & (f3["outcome"] == 0))
                       | (f3["etype"] == "ball_lost") | ((f3["etype"] == "error") & _has(f3["labels"], "ball lost"))]
    out["summary"] = {
        "Actions in final third": len(f3), "Passes (cmp/att)": f"{int((p['outcome'] == 1).sum())}/{len(p)}",
        "Key passes": int(ev[ev["etype"] == "pass"]["key_pass"].sum()),
        "Crosses (cmp/att)": f"{int((cr['outcome'] == 1).sum())}/{len(cr)}",
        "Box entries (succ/att)": f"{int(out['box_entries']['success'].sum())}/{len(out['box_entries'])}",
        "Shots / xG": f"{len(s)} / {s['xg'].sum():.2f}", "Final-third losses": len(out["losses"]),
    }
    return out
