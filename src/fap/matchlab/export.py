"""Enriched event export — the club's 44-column event schema + every Match Lab derived column +
one column per label, with team / match / player IDs.

Sheet "Events" (one row per tagged event, in file order):
  A. The agreed 44 columns (same names and value vocabulary as the converter notebook), but with
     real pass end points (team-mate's Receive / opponent's touch / restart) instead of the
     "next row's start" proxy, and IDs resolved from the reference tables.
  B. Derived analysis columns: attack-normalised coordinates, outcome, receiver, length, angle,
     direction, progressive, entries, pass type, thirds / channels, phases, possession, xG.
  C. ``tag_name`` + ``labels`` exactly as tagged, then ``label_<name>`` = TRUE/FALSE for every label used.
Other sheets: Carries (derived), Possessions, Players, Team summary, Data audit, Info.
"""
from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fap.matchlab import audit, metrics as M
from fap.matchlab.loader import PITCH_L, PITCH_W
from fap.matchlab.reference import Reference

if TYPE_CHECKING:
    from fap.matchlab.loader import MatchData

CORE = ["event_id", "match_id", "team_id", "team", "player_id", "player_name", "player_number", "event_type",
        "outcome", "match_begin_duration", "match_end_duration", "video_begin_duration", "video_end_duration", "period",
        "period_duration", "start_x", "start_y", "end_x", "end_y"]
PASS = ["pass_height", "pass_body_part", "pass_under_pressure", "pass_cross", "pass_cutback", "pass_through_ball",
        "pass_line_break", "pass_line_broken", "pass_restart_context"]
SHOT = ["shot_outcome", "shot_body_part", "shot_type", "shot_under_pressure", "shot_first_time", "shot_deflected"]
OTHER = ["carry_under_pressure", "dribble_outcome", "dribble_under_pressure", "recovery_outcome",
         "recovery_under_pressure", "interception_outcome", "press_counter_press", "press_outcome", "duel_aerial",
         "duel_ground", "duel_outcome"]
SCHEMA_44 = CORE + PASS + SHOT + OTHER
BOOLEAN = {"pass_under_pressure", "pass_cross", "pass_cutback", "pass_through_ball", "pass_line_break",
           "shot_under_pressure", "shot_first_time", "shot_deflected", "carry_under_pressure",
           "dribble_under_pressure", "recovery_under_pressure", "press_counter_press", "duel_aerial", "duel_ground"}
assert len(SCHEMA_44) == 45            # the club's 44 columns + `outcome` after event_type


def _choose(tokens: set, mapping) -> str:
    for out, aliases in mapping:
        if any(a in tokens for a in aliases):
            return out
    return ""


def official_minute(clock_text: str) -> str:
    """'45+03:03.46' → '48'; '12:30.0' → '12' (same rule as the converter notebook)."""
    m = re.fullmatch(r"(?:(\d+)\+)?(\d+):(\d+(?:\.\d+)?)", str(clock_text or "").strip())
    if not m:
        return ""
    return str(int(m.group(1) or 0) + int(m.group(2)) + int(float(m.group(3)) // 60))


def _schema_fields(kind: str, tok: set) -> dict:
    """The notebook's per-event-type qualifier columns from the canonical label set."""
    r: dict = {}
    b = lambda *a: int(any(x in tok for x in a))  # noqa: E731
    if kind == "pass":
        r["pass_height"] = _choose(tok, [("Ground", ["ground"]), ("Low", ["low"]), ("High", ["high"])])
        r["pass_body_part"] = _choose(tok, [("Right Foot", ["right foot"]), ("Left Foot", ["left foot"]),
                                            ("Head", ["head"]), ("Other", ["other"])])
        r["pass_under_pressure"] = b("under pressure")
        r["pass_cross"] = b("cross")
        r["pass_cutback"] = b("cutback")
        r["pass_through_ball"] = b("through ball")
        r["pass_line_break"] = b("line break")
        if r["pass_line_break"]:
            r["pass_line_broken"] = _choose(tok, [("First", ["first"]), ("Midfield", ["mid", "midfield"]), ("Last", ["last"])])
        r["pass_restart_context"] = _choose(tok, [("FK", ["free kick"]), ("Corner", ["corner"]), ("Throw", ["throw in"]),
                                                  ("GK", ["goal kick"]), ("KO", ["kick off"]),
                                                  ("Open play", ["open play", "regular play"])])
    elif kind == "shot":
        r["shot_outcome"] = _choose(tok, [("On Target", ["on target", "goal"]), ("Off Target", ["off target"]),
                                          ("Saved", ["saved"]), ("Blocked", ["blocked"]), ("Post", ["post", "woodwork"])])
        r["shot_body_part"] = _choose(tok, [("Right Foot", ["right foot"]), ("Left Foot", ["left foot"]),
                                            ("Head", ["head"]), ("Other", ["other"])])
        r["shot_type"] = _choose(tok, [("Free Kick", ["free kick"]), ("Penalty", ["penalty"]), ("Corner", ["corner"]),
                                       ("Kick Off", ["kick off"]), ("Open Play", ["open play", "regular play"])])
        r["shot_under_pressure"] = b("under pressure")
        r["shot_first_time"] = b("first time shot")
        r["shot_deflected"] = b("deflected", "deflection")
    elif kind == "carry":
        r["carry_under_pressure"] = b("under pressure")
    elif kind == "dribble":
        r["dribble_outcome"] = _choose(tok, [("Complete", ["complete", "successful"]), ("Incomplete", ["incomplete", "unsuccessful"])])
        r["dribble_under_pressure"] = b("under pressure")
    elif kind == "recovery":
        r["recovery_outcome"] = _choose(tok, [("Successful", ["successful"]), ("Unsuccessful", ["unsuccessful"])])
        r["recovery_under_pressure"] = b("under pressure")
    elif kind == "interception":
        r["interception_outcome"] = _choose(tok, [("Not Won", ["not won"]), ("Won", ["won"])])
    elif kind == "press":
        r["press_counter_press"] = b("counter press")
        r["press_outcome"] = _choose(tok, [("Successful", ["successful"]), ("Unsuccessful", ["unsuccessful"])])
    elif kind == "duel":
        r["duel_aerial"] = b("aerial")
        r["duel_ground"] = b("ground")
        r["duel_outcome"] = _choose(tok, [("Won", ["won"]), ("Lost", ["lost"])])
    return r


def _to_raw(x, y, sign) -> tuple:
    """Analysis-frame point → the file's own orientation (inverse of the loader transform)."""
    x, y, sign = np.asarray(x, float), np.asarray(y, float), np.asarray(sign)
    return np.where(sign > 0, x, PITCH_L - x), np.where(sign > 0, PITCH_W - y, y)


OUTCOME_LABELS = {"successful", "unsuccessful"}


def label_outcome(labels_raw: tuple) -> str:
    """'successful' / 'unsuccessful' from the row's labels ('Successfull' etc. normalised);
    'conflicting' when both are tagged; '' when neither is."""
    has = OUTCOME_LABELS & set(labels_raw)
    if len(has) == 2:
        return "conflicting"
    return next(iter(has), "")


def label_column(name: str) -> str:
    return "label_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_events(md: "MatchData", ref: Reference | None = None, match_id: str = "") -> pd.DataFrame:
    ref = ref or Reference.bundled()
    match_id = match_id or ref.auto_match_id(md)
    ev = md.events
    real = ev[~ev["derived"].astype(bool)].sort_values("event_id").copy()
    res = ref.resolve(md)
    team_ids = {t: r.team_id for t, r in res["teams"].items()}
    team_off = {t: (r.official or t) for t, r in res["teams"].items()}
    our = md.our_team

    out = pd.DataFrame(index=real.index)
    out["event_id"] = real["event_id"].astype(int)
    out["match_id"] = match_id
    out["team_id"] = real["team"].map(team_ids).fillna("")
    out["team"] = real["team"].map(team_off).fillna(real["team"])
    is_our = real["team"] == our
    look = real["player"].map(ref.squad.lookup)
    out["player_id"] = np.where(is_our, look.str[0], "")
    out["player_name"] = np.where(is_our, look.str[1], "")
    out["player_number"] = np.where(is_our & real["player"].ne("0"), real["player"], "")
    out["event_type"] = real["etype"]
    # outcome of this row's action, read from the labels column only
    out["outcome"] = real["labels_raw"].map(label_outcome)
    out["match_begin_duration"] = real["txt_clock"]
    out["match_end_duration"] = real["txt_clock_end"]
    out["video_begin_duration"] = real["txt_video"]
    out["video_end_duration"] = real["txt_video_end"]
    out["period"] = real["half"].astype(int)
    out["period_duration"] = real["txt_clock"].map(official_minute)
    out["start_x"] = real["src_x"].round(2)
    out["start_y"] = real["src_y"].round(2)
    # end points: passes (inferred) + tagged carries (same player's next action) in the file frame
    ex, ey = _to_raw(real["end_x"], real["end_y"], real["attack_sign"])
    carry_end = _tagged_carry_ends(md)
    ex = pd.Series(ex, index=real.index)
    ey = pd.Series(ey, index=real.index)
    for i, (cx, cy) in carry_end.items():
        if i in ex.index:
            ex[i], ey[i] = cx, cy
    out["end_x"] = ex.round(2)
    out["end_y"] = ey.round(2)
    for c in PASS + SHOT + OTHER:
        out[c] = 0 if c in BOOLEAN else ""
    qual = [_schema_fields(k, set(t)) for k, t in zip(real["etype"], real["labels"])]
    q = pd.DataFrame(qual, index=real.index)
    for c in q.columns:
        out[c] = q[c].where(q[c].notna(), out[c])
    for c in BOOLEAN:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(bool)

    # ---- B. derived ------------------------------------------------------------------------------------------
    d = pd.DataFrame(index=real.index)
    d["attack_direction_in_file"] = np.where(real["attack_sign"] > 0, "left_to_right", "right_to_left")
    for a, b in (("x", "x_attack"), ("y", "y_attack"), ("end_x", "end_x_attack"), ("end_y", "end_y_attack")):
        d[b] = real[a].round(2)
    d["end_source"] = real["end_source"].fillna("")
    d["receiver_number"] = np.where(is_our, real["receiver"].fillna(""), "")
    rl = real["receiver"].fillna("").map(ref.squad.lookup)
    d["receiver_player_id"] = np.where(is_our, rl.str[0], "")
    d["receiver_name"] = np.where(is_our, rl.str[1], "")
    d["length_m"] = real["length"].round(2)
    d["angle_deg"] = real["angle"].round(1)
    d["direction"] = real["direction"]
    d["length_class"] = real["length_class"]
    for src, dst in (("progressive", "progressive"), ("f3_entry", "final_third_entry"), ("box_entry", "box_entry"),
                     ("switch", "switch_of_play"), ("key_pass", "key_pass"), ("assist", "assist"),
                     ("set_piece", "set_piece"), ("under_pressure", "under_pressure"), ("transition", "transition"),
                     ("counter_press", "counter_press"), ("in_possession", "in_possession")):
        d[dst] = real[src].fillna(False).astype(bool)
    ispass_carry = real["etype"].isin(["pass", "carry"])
    for c in ("progressive", "final_third_entry", "box_entry"):
        d.loc[~ispass_carry, c] = False
    d["pass_type"] = real["pass_type"]
    d["set_piece_kind"] = real["set_piece_kind"]
    d["start_third"] = real["third"]
    d["start_channel"] = real["channel"]
    d["end_third"] = real["end_third"]
    d["end_channel"] = real["end_channel"]
    d["phase"] = real["phase"]
    d["defensive_phase"] = real["def_phase"]
    d["possession_id"] = real["poss_id"]
    d["possession_team"] = real["poss_team"].map(team_off).fillna(real["poss_team"])
    d["xg"] = real["xg"].round(3)
    d["shot_result"] = real["shot_result"]
    d["shot_goal"] = real["shot_goal"].astype(bool)
    d["minute_decimal"] = real["minute"].round(2)
    # counter-press, on our action that lost the ball (window = md.cp_window)
    from fap.matchlab import counterpress as CP
    cp = CP.analyse(md, our, md.cp_window)
    for col in ("cp_loss", "cp_pressed", "cp_regained"):
        d[col] = False
    for col in ("cp_seconds_to_press", "cp_seconds_to_regain"):
        d[col] = np.nan
    d["cp_first_presser"] = ""
    d["cp_category"] = ""
    d["cp_press_timing"] = ""
    d["cp_press_evidence"] = ""
    d["cp_window_s"] = np.nan
    if not cp.empty:
        cpi = cp.dropna(subset=["loss_event_id"]).set_index(cp["loss_event_id"].dropna().astype(int))
        m = real["event_id"].isin(cpi.index)
        ids = real.loc[m, "event_id"].astype(int)
        d.loc[m, "cp_loss"] = True
        d.loc[m, "cp_pressed"] = ids.map(cpi["pressed"]).astype(bool).to_numpy()
        d.loc[m, "cp_regained"] = ids.map(cpi["regained"]).astype(bool).to_numpy()
        d.loc[m, "cp_seconds_to_press"] = ids.map(cpi["seconds_to_press"]).round(2).to_numpy()
        d.loc[m, "cp_seconds_to_regain"] = ids.map(cpi["seconds_to_regain"]).round(2).to_numpy()
        d.loc[m, "cp_first_presser"] = ids.map(cpi["first_presser"]).fillna("").to_numpy()
        d.loc[m, "cp_category"] = ids.map(cpi["category"]).to_numpy()
        d.loc[m, "cp_press_timing"] = ids.map(cpi["press_timing"]).fillna("").to_numpy()
        d.loc[m, "cp_press_evidence"] = ids.map(cpi["press_evidence"]).fillna("").to_numpy()
        d.loc[m, "cp_window_s"] = md.cp_window

    # ---- C. raw tag + one column per label -------------------------------------------------------------------
    c = pd.DataFrame(index=real.index)
    c["tag_name"] = real["tag"]
    lab_col = next((col for col in md.raw.columns if str(col).strip().lower() in ("labels", "label")), None)
    if lab_col is not None:
        raw_lab = md.raw[lab_col].reset_index(drop=True)
        c["labels"] = real["event_id"].map(lambda i: "" if pd.isna(raw_lab.iloc[int(i) - 1]) else str(raw_lab.iloc[int(i) - 1]))
    else:
        c["labels"] = ""
    counts = pd.Series([lab for t in real["labels_raw"] for lab in t]).value_counts()
    for lab in [x for x in counts.index if x not in OUTCOME_LABELS]:     # those live in `outcome`
        c[label_column(lab)] = real["labels_raw"].map(lambda t, l=lab: l in t)
    return pd.concat([out, d, c], axis=1).reset_index(drop=True)


def _tagged_carry_ends(md: "MatchData") -> dict:
    """End of a TAGGED Carry = the same player's next on-ball action (file frame)."""
    ev = md.events
    res = {}
    idx = list(ev.index)
    for k, i in enumerate(idx):
        r = ev.loc[i]
        if r["etype"] != "carry" or r["derived"] or not r["player"] or r["player"] == "0":
            continue
        for j in idx[k + 1:k + 5]:
            q = ev.loc[j]
            if q["team"] != r["team"] or q["t"] - r["t"] > 15:
                break
            if q["player"] == r["player"] and q["etype"] in ("pass", "shot", "dribble", "receive") and pd.notna(q["src_x"]):
                res[i] = (round(float(q["src_x"]), 2), round(float(q["src_y"]), 2))
                break
    return res


def unmatched_shirts(md: "MatchData", ref: Reference) -> list[str]:
    """Our shirt numbers in the file that have no player_id in the squad (or are ambiguous)."""
    ev = md.events
    nums = ev[(ev["team"] == md.our_team) & ev["player"].ne("") & ev["player"].ne("0") & ~ev["derived"].astype(bool)]["player"]
    return sorted({n for n in nums if not ref.squad.lookup(n)[0]}, key=lambda s: (len(s), s))


def build_workbook(md: "MatchData", ref: Reference | None = None, match_id: str = "") -> bytes:
    ref = ref or Reference.bundled()
    match_id = match_id or ref.auto_match_id(md)
    events = build_events(md, ref, match_id)
    res = ref.resolve(md)
    our = md.our_team
    off = {t: (r.official or t) for t, r in res["teams"].items()}

    carries = md.events[md.events["derived"].astype(bool)].copy()
    carries = pd.DataFrame({
        "match_id": match_id, "team": carries["team"].map(off), "player_number": carries["player"],
        "player_id": carries["player"].map(lambda s: ref.squad.lookup(s)[0]),
        "player_name": carries["player"].map(lambda s: ref.squad.lookup(s)[1]),
        "period": carries["half"], "minute_decimal": carries["minute"].round(2),
        "x_attack": carries["x"].round(2), "y_attack": carries["y"].round(2),
        "end_x_attack": carries["end_x"].round(2), "end_y_attack": carries["end_y"].round(2),
        "length_m": carries["length"].round(2), "direction": carries["direction"],
        "progressive": carries["progressive"].astype(bool), "final_third_entry": carries["f3_entry"].astype(bool),
        "box_entry": carries["box_entry"].astype(bool), "possession_id": carries["poss_id"],
    })
    poss = md.possessions.copy()
    if not poss.empty:
        poss.insert(0, "match_id", match_id)
        poss["team"] = poss["team"].map(off).fillna(poss["team"])
        poss["players"] = poss["players"].map(lambda t: ", ".join(f"#{p}" for p in t))
    players = M.player_table(md, our)
    if not players.empty:
        nums = players["Player"].str.lstrip("#")
        players.insert(1, "player_id", nums.map(lambda s: ref.squad.lookup(s)[0]))
        players.insert(2, "player_name", nums.map(lambda s: ref.squad.lookup(s)[1]))
        players.insert(0, "match_id", match_id)
    team_sum = pd.DataFrame({off.get(t, t): M.team_summary(md, t) for t in md.teams})
    team_sum.index.name = "metric"
    aud = audit.run(md)
    from fap.matchlab import counterpress as CP
    cpt = CP.analyse(md, our, md.cp_window)
    if not cpt.empty:
        cpt.insert(0, "match_id", match_id)
        for col in ("player", "first_presser", "regained_by"):
            cpt[col + "_name"] = cpt[col].map(lambda s_: ref.squad.lookup(s_)[1])
    info = pd.DataFrame([
        ("match_id", match_id), ("source file", md.source_name), ("our team", off.get(our, our)),
        *[(f"team: {t}", f"{r.official or '(not matched)'} | id {r.team_id or '-'} | {r.method}") for t, r in res["teams"].items()],
        ("squad", f"{ref.squad.source}; ambiguous shirts not assigned: {', '.join(ref.squad.ambiguous) or 'none'}"),
        ("shirts without player_id", ", ".join(f"#{n}" for n in unmatched_shirts(md, ref)) or "none"),
        ("coordinates", "start_x/start_y/end_x/end_y = as in the file; *_attack = team attacks left->right, y=0 right touchline"),
        ("pass end points", "next event: team-mate Receive (completed) / opponent touch (lost) / restart or ball-out location"),
        ("progressive", "Wyscout: >=30 m closer to goal (own half), >=15 m (across halves), >=10 m (opponent half)"),
        ("notes", " | ".join(dict.fromkeys(md.notes))),
    ], columns=["item", "value"])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        events.to_excel(xw, sheet_name="Events", index=False)
        carries.to_excel(xw, sheet_name="Carries (derived)", index=False)
        if not poss.empty:
            poss.to_excel(xw, sheet_name="Possessions", index=False)
        if not players.empty:
            players.to_excel(xw, sheet_name="Players", index=False)
        team_sum.to_excel(xw, sheet_name="Team summary")
        if not cpt.empty:
            cpt.to_excel(xw, sheet_name="Counter-press", index=False)
        aud.to_excel(xw, sheet_name="Data audit", index=False)
        info.to_excel(xw, sheet_name="Info", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 9), 48)
    return buf.getvalue()
