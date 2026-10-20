"""Match Lab insights — short, deterministic, number-backed observations for the analyst.

Shown in the app only (Insights tab of a single match and of the multi-match comparison); never written into a
PDF / report. Every insight is a rule over metrics the engine already computes — no new analytics, no guessing:

  * ``match_insights``  — one match: possession, build-up, progression, box entries, chance quality, pressing,
                          counter-press, aerial / second balls, the two halves, and the stand-out players;
  * ``multi_insights``  — several matches: best / weakest all-round match, clear trends, stand-out matches per
                          metric, consistent vs volatile metrics, and the stand-out players per 90.

Thresholds are working guides for these tags (not league benchmarks); each insight carries the numbers behind it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRENGTH, CONCERN, NOTE = "strength", "concern", "note"
SECTIONS = ("Summary", "Possession & build-up", "Progression", "Box entries & chances", "Pressing & transitions",
            "Duels & defending", "Halves", "Players", "Trends", "Stand-out matches", "Consistency")


@dataclass
class Insight:
    section: str
    title: str
    detail: str
    tone: str = NOTE            # strength | concern | note
    weight: float = 0.0         # ordering inside a section (bigger first)


def _ok(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _f(v, d=0) -> str:
    return "–" if not _ok(v) else f"{float(v):.{d}f}"


def _dec(v) -> int:
    v = abs(float(v)) if _ok(v) else 0
    return 2 if v < 3 else 1 if v < 100 else 0


def _name(pl: str, names: dict | None) -> str:
    nm = (names or {}).get(str(pl), "")
    return f"#{pl} {nm}".strip()


# ------------------------------------------------------------------ one match
def match_insights(md, team: str, half: int | None = None, names: dict | None = None) -> list[Insight]:
    from fap.matchlab import multimatch as MM
    m = MM._team_col(md, team, half)
    out: list[Insight] = []
    add = lambda *a, **k: out.append(Insight(*a, **k))  # noqa: E731

    # possession & build-up
    poss, tilt = m.get("Possession % (passes)"), m.get("Field tilt %")
    if _ok(poss):
        if poss >= 60:
            add("Possession & build-up", "Controlled the ball",
                f"{_f(poss)}% possession (passes) and {_f(tilt)}% field tilt.", STRENGTH, poss)
        elif poss <= 40:
            add("Possession & build-up", "Played without much of the ball",
                f"Only {_f(poss)}% possession (passes) · field tilt {_f(tilt)}%.", NOTE, 100 - poss)
        if _ok(tilt) and tilt - poss >= 10:
            add("Possession & build-up", "More territory than possession",
                f"Field tilt {_f(tilt)}% vs possession {_f(poss)}% — the ball was played into the final third quickly.",
                NOTE, tilt - poss)
    ex, lb = m.get("Build-up exit success %"), m.get("Build-up exits by long ball %")
    if _ok(ex):
        if ex >= 75:
            add("Possession & build-up", "Clean build-up exits",
                f"{_f(ex)}% of build-ups left the first line successfully ({_f(m.get('Build-ups'))} build-ups).", STRENGTH, ex)
        elif ex < 60:
            add("Possession & build-up", "Build-up exits often failed",
                f"Only {_f(ex)}% of build-up exits were successful ({_f(m.get('Build-ups'))} build-ups).", CONCERN, 100 - ex)
    if _ok(lb) and lb >= 45:
        add("Possession & build-up", "Direct build-up",
            f"{_f(lb)}% of build-up exits were long balls.", NOTE, lb)
    los = m.get("Losses in own third")
    if _ok(los) and los >= 8:
        from fap.matchlab import team as T
        lo = T.buildup(md, team, half).get("losses", pd.DataFrame())
        who = lo["player"][lo["player"].ne("") & lo["player"].ne("0")].value_counts() if len(lo) else pd.Series(dtype=int)
        top = f" Most by {_name(who.index[0], names)} ({int(who.iloc[0])})." if len(who) else ""
        add("Possession & build-up", "Risky losses in our own third",
            f"{int(los)} balls lost in our own third.{top}", CONCERN, los)
    rf3 = m.get("Build-ups reaching final third %")
    if _ok(rf3):
        add("Possession & build-up", "Build-up → final third",
            f"{_f(rf3)}% of build-ups reached the final third; {_f(m.get('Build-ups ending in a shot %'))}% ended in a shot.",
            STRENGTH if rf3 >= 45 else CONCERN if rf3 < 25 else NOTE, rf3)

    # progression
    sides = {k: m.get(f"F3 entries {k} %") for k in ("left", "centre", "right")}
    sides = {k: v for k, v in sides.items() if _ok(v)}
    f3n = m.get("Final-third entries (pass + carry)")
    if sides and _ok(f3n) and f3n >= 8:
        side, v = max(sides.items(), key=lambda kv: kv[1])
        if v >= 55:
            add("Progression", f"Attacks favoured the {side}",
                f"{_f(v)}% of {int(f3n)} final-third entries went in on the {side} "
                f"(left {_f(sides.get('left'))}% · centre {_f(sides.get('centre'))}% · right {_f(sides.get('right'))}%).",
                NOTE, v)
        if _ok(sides.get("centre")) and sides["centre"] < 10:
            add("Progression", "Little through the centre",
                f"Only {_f(sides['centre'])}% of final-third entries came through the centre.", NOTE, 10 - sides["centre"])
    sr = m.get("Final-third entry success %")
    if _ok(sr):
        add("Progression", "Final-third entries",
            f"{_f(f3n)} successful entries (passes + carries) · {_f(sr)}% of attempts succeeded · "
            f"{_f(m.get('F3 entries by carry %'))}% by carry.",
            STRENGTH if sr >= 75 else CONCERN if sr < 55 else NOTE, sr)
    t3 = m.get("Time to final third (s)")
    if _ok(t3):
        add("Progression", "Speed of progression",
            f"Median {_f(t3, 1)} s from the start of a possession to the final third.", NOTE, 0)

    # box & chances
    bpf = m.get("Box entries per F3 entry %")
    if _ok(bpf) and _ok(f3n) and f3n >= 8:
        if bpf < 30:
            add("Box entries & chances", "Final third → box is the bottleneck",
                f"{int(f3n)} final-third entries but only {_f(m.get('Box entries'))} box entries ({_f(bpf)}%).", CONCERN, 30 - bpf)
        elif bpf >= 45:
            add("Box entries & chances", "Got into the box often",
                f"{_f(m.get('Box entries'))} box entries from {int(f3n)} final-third entries ({_f(bpf)}%).", STRENGTH, bpf)
    bx = m.get("Box entries")
    cr = m.get("Box entries by cross")
    if _ok(bx) and bx >= 5 and _ok(cr) and cr / bx >= 0.4:
        add("Box entries & chances", "Box entries relied on crosses",
            f"{int(cr)} of {int(bx)} box entries were crosses.", NOTE, cr / bx * 100)
    sh, xg, xps, gl = m.get("Shots"), m.get("xG"), m.get("xG per shot"), m.get("Goals")
    if _ok(sh) and sh >= 6 and _ok(xps):
        if xps < 0.08:
            add("Box entries & chances", "Low-quality shots",
                f"{int(sh)} shots for {_f(xg, 2)} xG — {_f(xps, 2)} xG per shot (mostly from distance / under pressure).",
                CONCERN, 0.08 - xps)
        elif xps >= 0.15:
            add("Box entries & chances", "Good chance quality",
                f"{int(sh)} shots for {_f(xg, 2)} xG — {_f(xps, 2)} xG per shot.", STRENGTH, xps)
    if _ok(gl) and _ok(xg) and abs(gl - xg) >= 1:
        add("Box entries & chances", "Finishing " + ("above" if gl > xg else "below") + " the chances",
            f"{int(gl)} goals from {_f(xg, 2)} xG.", STRENGTH if gl > xg else CONCERN, abs(gl - xg))
    xga = m.get("xG against")
    if _ok(xg) and _ok(xga):
        diff = xg - xga
        add("Box entries & chances", "Out-created the opponent" if diff > 0.3 else
            "Opponent created more" if diff < -0.3 else "Chances were level",
            f"xG {_f(xg, 2)} vs {_f(xga, 2)} against · shots {_f(sh)} vs {_f(m.get('Shots against'))}.",
            STRENGTH if diff > 0.3 else CONCERN if diff < -0.3 else NOTE, abs(diff))

    # pressing & transitions
    wb = m.get("Opp. possessions won back in their half %")
    if _ok(wb):
        add("Pressing & transitions", "High press",
            f"{_f(wb)}% of the opponent's possessions were won back in their half · "
            f"{_f(m.get('High regains (final third)'))} regains in the final third (defensive actions only).",
            STRENGTH if wb >= 20 else CONCERN if wb < 10 else NOTE, wb)
    cp, rg = m.get("Counter-pressed %"), m.get("Regained in window %")
    if _ok(cp):
        add("Pressing & transitions", "Reaction to ball losses",
            f"{_f(cp)}% of open-play losses were counter-pressed and {_f(rg)}% were regained within {md.cp_window:.0f} s.",
            STRENGTH if cp >= 40 else CONCERN if cp < 25 else NOTE, cp)

    # duels
    ad, sb = m.get("Aerial duels won %"), m.get("Second balls won %")
    if _ok(ad):
        add("Duels & defending", "Aerial duels", f"{_f(ad)}% of aerial duels won.",
            STRENGTH if ad >= 60 else CONCERN if ad <= 40 else NOTE, abs(ad - 50))
    if _ok(sb):
        add("Duels & defending", "Second balls", f"{_f(sb)}% of second balls won.",
            STRENGTH if sb >= 55 else CONCERN if sb <= 45 else NOTE, abs(sb - 50))
    dh = m.get("Def. action height (m)")
    if _ok(dh):
        add("Duels & defending", "Defensive height",
            f"Average defensive action {_f(dh, 1)} m from our goal ({'high block' if dh >= 45 else 'mid block' if dh >= 35 else 'low block'}).",
            NOTE, 0)

    # halves
    if half is None and set(md.events["half"].dropna().unique()) >= {1, 2}:
        h1, h2 = MM._team_col(md, team, 1), MM._team_col(md, team, 2)
        for k, lab in (("xG", "xG"), ("Final-third entries (pass + carry)", "final-third entries"),
                       ("Box entries", "box entries"), ("Possession % (passes)", "possession %"),
                       ("Counter-pressed %", "counter-pressed %"), ("Opp. possessions won back in their half %",
                                                                    "possessions won back high %")):
            a, b = h1.get(k), h2.get(k)
            if not (_ok(a) and _ok(b)) or max(a, b) == 0:
                continue
            rel = (b - a) / max(abs(a), 1e-9)
            big = abs(b - a) >= (0.4 if k == "xG" else 8 if k.endswith("%") else 4)
            if abs(rel) >= 0.3 and big:
                add("Halves", f"{lab[0].upper() + lab[1:]} {'rose' if b > a else 'dropped'} after the break",
                    f"1st half {_f(a, 2 if k == 'xG' else 0)} → 2nd half {_f(b, 2 if k == 'xG' else 0)}.",
                    NOTE, abs(rel))

    out.extend(_player_insights(md, team, half, names))
    return out


def _player_insights(md, team, half, names) -> list[Insight]:
    from fap.matchlab import player as PL
    ps = PL.player_stats(md, team, half)
    if ps.empty:
        return []
    out = []

    def top(col, title, tone=STRENGTH, min_v=1, extra=None):
        if col not in ps or ps[col].max() < min_v:
            return
        r = ps.sort_values(col, ascending=False).iloc[0]
        v = int(r[col]) if float(r[col]).is_integer() else round(float(r[col]), 2)
        unit = col if col == "xG" else col.lower()
        out.append(Insight("Players", title, f"{_name(r['player'], names)} — {v} {unit}{extra(r) if extra else ''}.",
                           tone, float(r[col])))

    top("Progressive actions", "Main progressor", min_v=5,
        extra=lambda r: f" ({int(r['Progressive passes'])} passes · {int(r['Progressive carries'])} carries · "
                        f"{int(r['Progressive receptions'])} receptions)")
    top("Key passes", "Main creator", min_v=2, extra=lambda r: f" · {r['xG']:.2f} xG of his own" if r["xG"] > 0 else "")
    top("xG", "Biggest goal threat", min_v=0.3,
        extra=lambda r: f" from {int(r['Shots'])} shot{'s' if int(r['Shots']) != 1 else ''}")
    top("Defensive actions", "Most defensive work", min_v=8,
        extra=lambda r: f" ({int(r['Tackles'])} tackles · {int(r['Interceptions'])} interceptions · {int(r['Recoveries'])} recoveries)")
    d = ps[(ps["Duels decided"] >= 5) & (ps["Duel success %"] >= 55)]
    if len(d):
        r = d.sort_values(["Duel success %", "Duels decided"], ascending=False).iloc[0]
        out.append(Insight("Players", "Best in duels", f"{_name(r['player'], names)} — won {int(r['Duels won'])} of "
                                                       f"{int(r['Duels decided'])} duels ({r['Duel success %']:.0f}%).",
                           STRENGTH, float(r["Duel success %"])))
    top("Ball losses", "Most ball losses", CONCERN, min_v=8,
        extra=lambda r: f" from {int(r['Passes'])} passes and {int(r['Dribbles'])} dribbles")
    top("Dribbled past", "Most often dribbled past", CONCERN, min_v=3)
    return out


# ------------------------------------------------------------------ several matches
def multi_insights(table: pd.DataFrame, *, lower_better=None, neutral=None, pool: pd.DataFrame | None = None,
                   min_minutes: float = 90.0) -> list[Insight]:
    """Insights from the comparison table (rows = metrics, a column per match + 'Match average' / 'All matches'),
    plus — if ``pool`` (``playercompare.player_pool``) is given — the stand-out players per 90."""
    from fap.matchlab import multimatch as MM
    lower_better = MM.LOWER_BETTER if lower_better is None else lower_better
    neutral = MM.NEUTRAL if neutral is None else neutral
    matches = [c for c in table.columns if c not in ("Match average", "All matches")]
    out: list[Insight] = []
    if len(matches) < 2:
        return out
    vals = table[matches].apply(pd.to_numeric, errors="coerce")
    judged = [m for m in vals.index if m not in neutral and vals.loc[m].notna().sum() >= 2 and vals.loc[m].std() > 0]

    # z-scores, oriented so + = better
    z = pd.DataFrame({m: (vals.loc[m] - vals.loc[m].mean()) / vals.loc[m].std() * (-1 if m in lower_better else 1)
                      for m in judged}).T
    if len(z):
        score = z.mean(axis=0, skipna=True)
        best, worst = score.idxmax(), score.idxmin()
        def _why(col, n=3, good=True):
            s = z[col].dropna().sort_values(ascending=not good).head(n)
            return ", ".join(s.index)
        out.append(Insight("Summary", f"Best all-round match: {best}",
                           f"Above the average of these matches in the most metrics — strongest in {_why(best)}.",
                           STRENGTH, 3))
        out.append(Insight("Summary", f"Weakest all-round match: {worst}",
                           f"Below the average in the most metrics — weakest in {_why(worst, good=False)}.", CONCERN, 2))
        try:
            groups = {g: [m for m in ms if m in z.index] for g, ms in MM.TEAM_GROUPS.items()}
        except Exception:
            groups = {}
        for g, ms in groups.items():
            if len(ms) >= 2:
                gs = z.loc[ms].mean(axis=0, skipna=True)
                out.append(Insight("Summary", f"{g}: best in {gs.idxmax()}",
                                   f"Weakest {g.lower()} match: {gs.idxmin()} (on {len(ms)} metrics).", NOTE, 1))

    # trends (matches in the order given): rose / fell in every match, by a meaningful amount
    if len(matches) >= 3:
        trends = []
        for m in judged:
            v = vals.loc[m].dropna()
            if len(v) < 3:
                continue
            d = np.diff(v.to_numpy(float))
            change = abs(v.iloc[-1] - v.iloc[0]) / max(abs(v.mean()), 1e-9)
            if ((d > 0).all() or (d < 0).all()) and change >= (0.15 if len(v) == 3 else 0.08):
                up = (d > 0).all()
                good = up != (m in lower_better)
                dec = _dec(v.mean())
                trends.append(Insight("Trends", f"{m} {'rose' if up else 'fell'} in every match",
                                      " → ".join(_f(x, dec) for x in v) + ".", STRENGTH if good else CONCERN, change))
        out.extend(sorted(trends, key=lambda i: -i.weight)[:8])

    # stand-out matches per metric
    if len(z):
        stack = z.stack().rename("z").reset_index()
        stack.columns = ["metric", "match", "z"]
        stack = stack[stack["z"].abs() >= 1.1].assign(a=lambda d: d["z"].abs()).sort_values("a", ascending=False)
        seen = set()
        for _, r in stack.iterrows():
            if r["metric"] in seen:
                continue
            seen.add(r["metric"])
            v, avg = vals.at[r["metric"], r["match"]], vals.loc[r["metric"]].mean()
            good = r["z"] > 0
            out.append(Insight("Stand-out matches", f"{r['metric']}: {'best' if good else 'worst'} in {r['match']}",
                               f"{_f(v, _dec(avg))} vs an average of {_f(avg, _dec(avg))}.",
                               STRENGTH if good else CONCERN, float(r["a"])))
            if len(seen) >= 8:
                break

    # consistency
    if len(matches) >= 3:
        cv = {m: vals.loc[m].std() / abs(vals.loc[m].mean()) for m in judged if abs(vals.loc[m].mean()) > 1e-9}
        stable = sorted((k for k, c in cv.items() if c <= 0.08), key=cv.get)[:5]
        vol = sorted((k for k, c in cv.items() if c >= 0.35), key=cv.get, reverse=True)[:5]
        if stable:
            out.append(Insight("Consistency", "Stable from match to match",
                               ", ".join(f"{k} (±{cv[k] * 100:.0f}%)" for k in stable) + ".", NOTE, 1))
        if vol:
            out.append(Insight("Consistency", "Very different from match to match",
                               ", ".join(f"{k} (±{cv[k] * 100:.0f}%)" for k in vol) + ".", NOTE, 1))

    if pool is not None and len(pool):
        out.extend(_pool_insights(pool, min_minutes))
    return out


def _pool_insights(pool: pd.DataFrame, min_minutes: float) -> list[Insight]:
    p = pool[pool["Minutes"] >= min_minutes]
    if p.empty:
        return []
    out = [Insight("Players", "Most minutes",
                   ", ".join(f"{r['Player']} {r['Minutes']:.0f}′" for _, r in p.head(3).iterrows()) + ".", NOTE, 0)]
    for col, title, tone in (("Progressive passes p90", "Top progressive passers", STRENGTH),
                             ("Progressive carries p90", "Top ball carriers", STRENGTH),
                             ("Key passes p90", "Top creators", STRENGTH),
                             ("xG p90", "Biggest goal threats", STRENGTH),
                             ("Defensive actions p90", "Most defensive work", STRENGTH),
                             ("Ball losses p90", "Most ball losses", CONCERN)):
        if col not in p or not p[col].notna().any() or p[col].max() <= 0:
            continue
        t = p.sort_values(col, ascending=False).head(3)
        out.append(Insight("Players", title + " (per 90)",
                           ", ".join(f"{r['Player']} {r[col]:.{_dec(r[col])}f}" for _, r in t.iterrows()) + ".", tone, 0))
    d = p[p["Duels decided"] >= 8]
    if len(d):
        t = d.sort_values("Duel success %", ascending=False).head(3)
        out.append(Insight("Players", "Best duel record",
                           ", ".join(f"{r['Player']} {r['Duel success %']:.0f}% ({int(r['Duels won'])}/{int(r['Duels decided'])})"
                                     for _, r in t.iterrows()) + ".", STRENGTH, 0))
    return out


def group(insights: list[Insight]) -> dict[str, list[Insight]]:
    out: dict[str, list[Insight]] = {}
    for s in SECTIONS:
        items = [i for i in insights if i.section == s]
        if items:
            out[s] = sorted(items, key=lambda i: -i.weight)
    return out
