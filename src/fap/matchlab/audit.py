"""Data audit — reconcile every engine number against an independent count of the raw file.

Each check re-derives its expected value straight from ``md.raw`` (the file as read) with plain
pandas — not through the engine's own helpers — and compares it with what the engine produced.
Status: PASS (identical), WARN (explained difference / plausibility flag), FAIL (unexplained).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fap.matchlab.loader import PITCH_L, PITCH_W, _resolve

if TYPE_CHECKING:
    from fap.matchlab.loader import MatchData


def _low(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()) if s is not None and s == s else ""


def _labels(v) -> set:
    return {_low(t) for t in str(v).split(",") if t.strip()} if v is not None and v == v else set()


def run(md: "MatchData") -> pd.DataFrame:
    raw = md.raw
    ev = md.events
    real = ev[~ev["derived"].astype(bool)]
    rows: list[dict] = []

    def add(area, check, expected, actual, status=None, detail=""):
        if status is None:
            status = "PASS" if str(expected) == str(actual) else "FAIL"
        rows.append({"Area": area, "Check": check, "Raw file": expected, "Engine": actual,
                     "Status": status, "Detail": detail})

    if raw is None or raw.empty:
        return pd.DataFrame(rows)
    cols = _resolve(raw)
    tag = raw[cols["tag"]].map(_low)
    team = raw[cols["team"]].map(_low) if "team" in cols else pd.Series("", index=raw.index)
    lab = raw[cols["labels"]].map(_labels) if "labels" in cols else pd.Series([set()] * len(raw), index=raw.index)

    # --- completeness -----------------------------------------------------------------
    add("Completeness", "Events (rows)", len(raw), len(real))
    rc = tag.value_counts()
    ec = real["tag"].map(_low).value_counts()
    diff = [(t, int(rc.get(t, 0)), int(ec.get(t, 0))) for t in sorted(set(rc.index) | set(ec.index))
            if rc.get(t, 0) != ec.get(t, 0)]
    add("Completeness", "Count per tag (all tags)", f"{len(rc)} tags", f"{len(ec)} tags",
        "PASS" if not diff else "FAIL", "; ".join(f"{t}: {a}≠{b}" for t, a, b in diff))
    for tm in md.teams:
        add("Completeness", f"Events of {tm}", int((team == _low(tm)).sum()), int((real["team"] == tm).sum()))
    xy = pd.to_numeric(raw[cols["x"]], errors="coerce")
    yy = pd.to_numeric(raw[cols["y"]], errors="coerce")
    outside = int(((xy < 0) | (xy > PITCH_L) | (yy < 0) | (yy > PITCH_W)).sum())
    add("Coordinates", "Events without x/y", int((xy.isna() | yy.isna()).sum()),
        int(real["src_x"].isna().sum()), None, "half markers / subs normally have no location")
    add("Coordinates", "Points outside 105×68 (clamped to the line)", outside, outside,
        "PASS" if outside <= 0.03 * len(raw) else "WARN", "ball-out tags are often just off the pitch")

    # --- passes -------------------------------------------------------------------------
    is_pass_raw = tag.str.contains("pass") | tag.str.contains("cross")
    add("Passes", "Pass events (tags with 'pass'/'cross')", int(is_pass_raw.sum()), int((real["etype"] == "pass").sum()))
    succ_raw = is_pass_raw & (tag.str.startswith("succ") | lab.map(lambda s: bool(s & {"successful", "successfull"})))
    fail_raw = is_pass_raw & (tag.str.startswith("fail") | (tag == "cross saved")
                              | lab.map(lambda s: bool(s & {"unsuccessful", "unsuccessfull", "failure"})))
    both = succ_raw & fail_raw
    if both.any():
        ids = [int(i) + 1 for i in raw.index[both]]
        res = real.set_index("event_id").reindex(ids)["outcome"]
        add("Passes", "Contradictory labels (both successful & unsuccessful)", int(both.sum()), int(both.sum()), "WARN",
            "rows " + ", ".join(f"{i} -> {'completed' if r == 1 else 'lost' if r == 0 else 'unknown'}"
                                 for i, r in res.items()) + " (resolved from the next event; fix in the tagging)")
        succ_raw, fail_raw = succ_raw & ~both, fail_raw & ~both
    p = real[real["etype"] == "pass"]
    lab_ok = p["labels"].map(lambda s: "successful" in s and "unsuccessful" not in s)
    lab_ko = p["labels"].map(lambda s: "unsuccessful" in s and "successful" not in s)
    add("Passes", "Tagged successful", int(succ_raw.sum()), int(((p["outcome"] == 1) & lab_ok).sum()))
    add("Passes", "Tagged unsuccessful", int(fail_raw.sum()), int(((p["outcome"] == 0) & lab_ko).sum()))
    inferred = p[~lab_ok & ~lab_ko]
    add("Passes", "Outcome not tagged → inferred from next event", int((is_pass_raw & ~succ_raw & ~fail_raw).sum()),
        f"{int((inferred['outcome'] == 1).sum())} ok / {int((inferred['outcome'] == 0).sum())} lost / "
        f"{int(inferred['outcome'].isna().sum())} unknown", "WARN" if len(inferred) else "PASS",
        "completed when a team-mate receives next, lost when the opponent touches it next")
    cross_raw = is_pass_raw & (tag.str.contains("cross") | lab.map(lambda s: "cross" in s))
    add("Passes", "Crosses", int(cross_raw.sum()), int(p["cross"].sum()))
    for tm in md.teams:
        m = team == _low(tm)
        add("Passes", f"Passes {tm} (completed / total)", f"{int((succ_raw & m).sum())}+ / {int((is_pass_raw & m).sum())}",
            f"{int(((p['outcome'] == 1) & (p['team'] == tm)).sum())} / {int((p['team'] == tm).sum())}",
            "PASS" if int((is_pass_raw & m).sum()) == int((p["team"] == tm).sum()) else "FAIL",
            "raw shows tagged-successful only; engine adds inferred outcomes")
    ends = p["end_x"].notna()
    add("Passes", "Passes with an end point", f"{len(p)} passes", f"{int(ends.sum())} ({ends.mean() * 100:.1f}%)",
        "PASS" if ends.mean() >= 0.95 else "WARN", "source: " + ", ".join(f"{k} {v}" for k, v in
                                                     p["end_source"].replace("", "none").value_counts().items()))
    comp = p[(p["outcome"] == 1) & ends]
    long_ = int((comp["length"] > 65).sum())
    add("Passes", "Completed passes longer than 65 m (plausibility)", "–", long_,
        "PASS" if long_ <= max(2, 0.01 * len(comp)) else "WARN", "check these in the event table if WARN")
    selfp = int((p["receiver"].ne("") & p["receiver"].ne("0") & (p["receiver"] == p["player"])).sum())
    add("Passes", "Passes 'received' by the passer himself", 0, selfp)

    # --- shots / goals --------------------------------------------------------------------
    is_shot_raw = tag.str.startswith("shot") | (tag == "goal")
    goal_raw = (tag == "goal") | (is_shot_raw & lab.map(lambda s: "goal" in s))
    s = real[real["etype"] == "shot"]
    add("Shots", "Shots (all)", int(is_shot_raw.sum()), len(s))
    add("Shots", "Goals", int(goal_raw.sum()), int(s["shot_goal"].sum()))
    for tm in md.teams:
        m = team == _low(tm)
        add("Shots", f"Shots {tm} (goals)", f"{int((is_shot_raw & m).sum())} ({int((goal_raw & m).sum())})",
            f"{int((s['team'] == tm).sum())} ({int(s[s['team'] == tm]['shot_goal'].sum())})")
    on_raw = is_shot_raw & (tag.str.contains("on target") | (tag == "goal") | lab.map(lambda x: bool(x & {"saved", "on target", "goal"})))
    by_save = int((s["shot_on"] & ~s["labels"].map(lambda x: bool(set(x) & {"saved", "on target", "goal"}))).sum())
    add("Shots", "On target (incl. goals)", int(on_raw.sum()), int(s["shot_on"].sum()),
        "PASS" if int(s["shot_on"].sum()) == int(on_raw.sum()) + by_save else "FAIL",
        f"{by_save} shot(s) without an outcome label counted on target because the keeper's 'Saved' tag follows"
        if by_save else "")

    # --- players ----------------------------------------------------------------------------
    our = md.our_team
    if "player" in cols:
        pl_raw = raw[cols["player"]].astype(str).str.split(",").str[0].str.strip().str.replace(r"\.0$", "", regex=True)
        m = team == _low(our)
        rc = pl_raw[m & ~pl_raw.isin(["0", "nan", ""])].value_counts()
        ecp = real[(real["team"] == our) & real["player"].ne("") & real["player"].ne("0")]["player"].value_counts()
        bad = [k for k in set(rc.index) | set(ecp.index) if rc.get(k, 0) != ecp.get(k, 0)]
        add("Players", f"Events per {our} player", f"{len(rc)} players", f"{len(ecp)} players",
            "PASS" if not bad else "WARN", ("differences (multi-player rows like '0, 20'): " + ", ".join(f"#{b}" for b in bad)) if bad else "")
        for tm in md.teams:
            if tm == our:
                continue
            known = int(real[(real["team"] == tm)]["player"].isin(["", "0"]).mean() * 100)
            add("Players", f"{tm} events without a player number", "–", f"{known}%",
                "WARN" if known > 50 else "PASS", "opponent analysed at team level only" if known > 50 else "")

    # --- orientation / halves -----------------------------------------------------------------
    halves = int((ev["etype"] == "half_start").sum())
    add("Time", "Half-start markers", halves, ev["half"].nunique(), "PASS" if halves in (0, ev["half"].nunique()) else "WARN")
    for (tm, h), d in md.orientation.items():
        g = real[(real["team"] == tm) & (real["half"] == h)]
        sh = g[g["etype"] == "shot"]["src_x"].dropna()
        pp = g[(g["etype"] == "pass") & (g["end_source"] == "receive")]
        dx = (pp["end_raw_x"] - pp["src_x"]).dropna()
        votes = []
        if len(sh):
            votes.append("→" if sh.mean() > PITCH_L / 2 else "←")
        if len(dx) >= 10:
            votes.append("→" if dx.mean() > 0 else "←")
        ok = all(v == d for v in votes)
        add("Orientation", f"{tm} half {h} attacks", " / ".join(votes) or "no evidence", d,
            "PASS" if ok and votes else "WARN",
            f"shots mean raw x {sh.mean():.1f}" if len(sh) else "" + (f"; completed-pass mean dx {dx.mean():+.1f} m" if len(dx) else ""))

    # --- possessions ----------------------------------------------------------------------------
    onb = real[real["etype"].isin(["pass", "receive", "carry", "dribble", "shot"])]
    cov = float((onb["poss_id"] >= 0).mean() * 100) if len(onb) else np.nan
    add("Possessions", "On-ball events inside a possession", "100%", f"{cov:.1f}%", "PASS" if cov >= 98 else "WARN")
    if md.notes:
        add("Notes", "Engine notes", "–", " | ".join(dict.fromkeys(md.notes)), "PASS")
    out = pd.DataFrame(rows)
    for c in ("Raw file", "Engine"):
        out[c] = out[c].astype(str)
    return out


def summary(df: pd.DataFrame) -> dict:
    return df["Status"].value_counts().to_dict() if not df.empty else {}
