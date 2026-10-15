"""Interactive figures for the team deep-dive (Team stats · Build-up · Progression · Final third).

Same conventions as :mod:`.interactive` — Plotly, hover on everything, Dark / Light theme, the team's
attacking frame. Numbers come from :mod:`.team`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fap.matchlab import team as T
from fap.matchlab.charts import DARK, Theme
from fap.matchlab.interactive import (_acc_color, _add_moves, _eid, _empty, _fmt, _heat_color, _mix, _mmss, _pitch,
                                      _point_hover, _rect, _subtitle, _who)
from fap.matchlab.loader import PITCH_L, PITCH_W, MatchData
from fap.matchlab.metrics import CHANNEL_EDGES, CHANNELS

_CFG_LAYOUT = dict(font=dict(family="Segoe UI, Inter, Roboto, Arial, sans-serif"))


def _base(fig: go.Figure, th: Theme, title: str, sub: str = "", height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" + (f"<br><span style='font-size:11px;color:{th.muted}'>{sub}</span>" if sub else ""),
                   x=0.01, font=dict(size=15, color=th.text)),
        height=height, paper_bgcolor=th.bg, plot_bgcolor=th.bg, font=dict(color=th.text, size=11, **_CFG_LAYOUT["font"]),
        margin=dict(l=10, r=20, t=70 if sub else 50, b=10),
        hoverlabel=dict(bgcolor=th.panel, bordercolor=th.grid, font=dict(color=th.text), align="left"),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(color=th.muted)))
    fig.update_xaxes(gridcolor=th.grid, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    return fig


# ------------------------------------------------------------------ team stats (both teams)
def stats_category(side: pd.DataFrame, category: str, teams: list, *, theme: Theme = DARK) -> go.Figure:
    """One column per team (bars all pointing right, so labels never collide). Bar = success rate when
    the metric has attempts, otherwise each team's share of the two teams' total."""
    th = theme
    a, b = teams[0], (teams[1] if len(teams) > 1 else None)
    d = side[side["Category"] == category].iloc[::-1]
    share_rate = ("Possession share", "Field tilt")
    cols = [a] + ([b] if b else [])
    rows = {t: {"len": [], "txt": [], "hov": []} for t in cols}
    ys = list(d["Metric"])
    for _, r in d.iterrows():
        vals = {t: (r.get(t), r.get(f"{t} att"), r.get(f"{t} %")) for t in cols}
        has_att = any(not pd.isna(v[1]) for v in vals.values())
        tot = sum(0 if pd.isna(v[0]) else abs(v[0]) for v in vals.values())
        for t in cols:
            v, att, pct = vals[t]
            if r["Metric"].startswith(share_rate):
                ln, txt = (pct if not pd.isna(pct) else 0), _fmt(pct, 0, "%")
            elif has_att:
                ln = pct if not pd.isna(pct) else 0
                txt = f"{_n(v)}/{_n(att)} · {_fmt(pct, 0, '%')}" if not pd.isna(att) and att > 0 else "–"
            else:
                if r["Metric"].startswith("PPDA") and tot:      # lower = more intense → bar for the lower value
                    other = [vals[o][0] for o in cols if o != t]
                    ln = (other[0] / tot * 100) if other and not pd.isna(other[0]) else 0
                else:
                    ln = (v / tot * 100) if tot and not pd.isna(v) else 0
                txt = _n(v)
            note = f"<br><span style='color:{th.muted}'>{r['Note']}</span>" if isinstance(r.get("Note"), str) and r["Note"] else ""
            rows[t]["len"].append(ln)
            rows[t]["txt"].append(txt)
            rows[t]["hov"].append(f"<b>{t} — {r['Metric']}</b><br>{txt}{note}")
    fig = make_subplots(rows=1, cols=len(cols), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"<b>{t}</b>" for t in cols])
    for i, (t, color) in enumerate(zip(cols, (th.team, th.opp)), 1):
        fig.add_trace(go.Bar(y=ys, x=rows[t]["len"], orientation="h", marker_color=color, text=rows[t]["txt"],
                             textposition="outside", cliponaxis=False, textfont=dict(size=10.5, color=th.text),
                             hovertext=rows[t]["hov"], hoverinfo="text", showlegend=False), 1, i)
        fig.update_xaxes(range=[0, 165], visible=False, row=1, col=i)
    _base(fig, th, category, "bar = success rate (with attempts) or share of the two teams", 95 + 28 * len(ys))
    fig.update_layout(bargap=0.38)
    for ann, color in zip(fig.layout.annotations, (th.team, th.opp)):
        ann.font = dict(color=color, size=12)
    return fig


def _n(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------ build-up
_METHOD_COL = {"Pass": "team", "Long pass": "accent", "Carry": "carry", "Dribble": "good", "Receive": "muted",
               "Cross": "opp", "Through ball": "good", "Shot": "accent"}


def buildup_exits(md: MatchData, team: str, *, half=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    b = T.buildup(md, team, half)
    ex = b.get("exits", pd.DataFrame())
    t1 = md.thirds[0]
    fig = _pitch(th, f"{team} — How we leave our own third",
                 _subtitle(md, team, half, extra=f"{len(b['possessions'])} build-up possessions · {len(ex)} reached the exit"))
    fig.add_shape(type="rect", x0=0, x1=t1, y0=0, y1=PITCH_W, fillcolor=_mix(th.team, th.pitch, 0.85), line=dict(width=0),
                  layer="below")
    fig.add_shape(type="line", x0=t1, x1=t1, y0=0, y1=PITCH_W, line=dict(color=th.accent, width=1.5, dash="dash"))
    if ex.empty:
        _empty(fig, th, "No build-up exits")
        return fig
    for m, g in ex.groupby("method"):
        col = getattr(th, _METHOD_COL.get(m, "team"))
        xs, ys = [], []
        for _, r in g.iterrows():
            xs += [r["x"], r["end_x"], None]
            ys += [r["y"], r["end_y"], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=1.6), opacity=0.5, hoverinfo="skip",
                                 legendgroup=m, showlegend=False))
        txt = [f"<b>{_who(r['player'], names)} — {m}</b> · {_mmss(r['minute'])} · event {_eid(r)}<br>"
               f"Exit through the {r['channel'].lower()} · {'✓ kept the ball' if r['success'] else '✗ lost it'}<br>"
               f"Possession then {'ended in a shot' if r['shot'] else 'reached the final third' if r['reached_f3'] else 'did not reach the final third'}"
               for _, r in g.iterrows()]
        fig.add_trace(go.Scatter(x=g["end_x"], y=g["end_y"], mode="markers", name=f"{m} ({len(g)})", legendgroup=m,
                                 text=txt, hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=9, color=col, symbol=["circle" if s else "x" for s in g["success"]],
                                             line=dict(color=th.bg, width=0.7))))
    # per-channel summary on the exit line
    for k, ch in enumerate(CHANNELS):
        g = ex[ex["channel"] == ch]
        y0, y1 = CHANNEL_EDGES[k], CHANNEL_EDGES[k + 1]
        if not len(g):
            continue
        rate = g["success"].mean()
        fig.add_annotation(x=t1 + 1.5, y=(y0 + y1) / 2, xanchor="left", showarrow=False,
                           text=f"<b>{len(g)}</b> · {rate * 100:.0f}%",
                           hovertext=(f"<b>{ch}</b>: {len(g)} exits, {int(g['success'].sum())} kept ({rate * 100:.0f}%)<br>"
                                      + ", ".join(f"{m} {n}" for m, n in g['method'].value_counts().items())),
                           font=dict(size=11, color=th.text), bgcolor=_mix(th.panel, th.bg, 0.2))
    return fig


def buildup_flow(md: MatchData, team: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    """Sankey: how build-up possessions start → how far they get → how they end."""
    th = theme
    b = T.buildup(md, team, half)["possessions"]
    if b.empty:
        return _base(go.Figure(), th, "Build-up flow", "no build-up possessions")
    far = b["reach_third"].map({"Defensive third": "Stayed in own third", "Middle third": "Reached middle third",
                                "Final third": "Reached final third"}).fillna("–")
    end = np.where(b["shot"], "Shot", b["end_reason"].replace({"Lost": "Ball lost"}))
    a_lab = [f"Start: {s}" for s in b["start_type"]]
    nodes = list(dict.fromkeys(list(a_lab) + list(far) + list(end)))
    idx = {n: i for i, n in enumerate(nodes)}
    links = pd.DataFrame({"s": a_lab, "m": far, "e": end})
    l1 = links.groupby(["s", "m"]).size()
    l2 = links.groupby(["m", "e"]).size()
    src = [idx[s] for s, _ in l1.index] + [idx[m] for m, _ in l2.index]
    tgt = [idx[m] for _, m in l1.index] + [idx[e] for _, e in l2.index]
    val = list(l1.values) + list(l2.values)
    colmap = {"Stayed in own third": th.bad, "Reached middle third": th.accent, "Reached final third": th.good,
              "Shot": th.good, "Ball lost": th.bad}
    ncol = [colmap.get(n, th.team) for n in nodes]
    fig = go.Figure(go.Sankey(
        node=dict(label=[f"{n} ({int(links.isin([n]).any(axis=1).sum())})" for n in nodes], color=ncol, pad=18, thickness=16,
                  line=dict(color=th.bg, width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=[_mix(ncol[t], th.bg, 0.55) for t in tgt],
                  hovertemplate="%{source.label} → %{target.label}<br><b>%{value}</b> possessions<extra></extra>")))
    return _base(fig, th, f"{team} — Build-up: start → how far → how it ended",
                 f"{len(b)} possessions that started in our own third · hover a flow for its count", 420)


def pass_pairs_network(md: MatchData, team: str, passes: pd.DataFrame, title: str, *, theme: Theme = DARK,
                       names: dict | None = None, min_passes: int = 2) -> go.Figure:
    """Network of completed passes inside a zone: node = average position in the zone, link = pair."""
    th, names = theme, names or {}
    p = passes[(passes["outcome"] == 1) & passes["receiver"].ne("") & passes["receiver"].ne("0") & passes["player"].ne("0")]
    fig = _pitch(th, f"{team} — {title}", f"completed passes · links ≥ {min_passes} · hover a player or a link", height=470)
    if p.empty:
        _empty(fig, th)
        return fig
    pts = pd.concat([p[["player", "x", "y"]].rename(columns={"player": "pl"}),
                     p[["receiver", "end_x", "end_y"]].rename(columns={"receiver": "pl", "end_x": "x", "end_y": "y"})])
    pos = pts.groupby("pl")[["x", "y"]].mean()
    made, recv = p.groupby("player").size(), p.groupby("receiver").size()
    pairs = p.assign(a=np.minimum(p["player"], p["receiver"]), b=np.maximum(p["player"], p["receiver"])).groupby(["a", "b"]).size()
    pairs = pairs[pairs >= min_passes]
    mx = pairs.max() if len(pairs) else 1
    mid_x, mid_y, mid_t = [], [], []
    for (a, b), n in pairs.sort_values().items():
        fig.add_trace(go.Scatter(x=[pos.at[a, "x"], pos.at[b, "x"]], y=[pos.at[a, "y"], pos.at[b, "y"]], mode="lines",
                                 showlegend=False, hoverinfo="skip", opacity=0.3 + 0.65 * n / mx,
                                 line=dict(color=th.team, width=1 + 8 * n / mx)))
        mid_x.append((pos.at[a, "x"] + pos.at[b, "x"]) / 2)
        mid_y.append((pos.at[a, "y"] + pos.at[b, "y"]) / 2)
        ab = int(((p["player"] == a) & (p["receiver"] == b)).sum())
        mid_t.append(f"<b>{_who(a, names)} ↔ {_who(b, names)}</b><br>{int(n)} passes · {_who(a, names)} → {ab} · "
                     f"{_who(b, names)} → {int(n) - ab}")
    if mid_x:
        fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", showlegend=False, text=mid_t,
                                 hovertemplate="%{text}<extra></extra>", marker=dict(size=10, color="rgba(0,0,0,0)")))
    vol = made.add(recv, fill_value=0)
    fig.add_trace(go.Scatter(
        x=pos["x"], y=pos["y"], mode="markers+text", text=list(pos.index), showlegend=False,
        textfont=dict(color=th.text, size=11),
        hovertext=[f"<b>{_who(pl, names)}</b><br>passes {int(made.get(pl, 0))} · received {int(recv.get(pl, 0))}"
                   for pl in pos.index], hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(size=[18 + 22 * vol.get(pl, 0) / vol.max() for pl in pos.index], color=th.panel,
                    line=dict(color=th.team, width=2.2))))
    return fig


def gk_distribution(md: MatchData, team: str, *, half=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    g = T.goalkeeper_passes(md, team, half)
    gk = T.goalkeeper(md, team)
    ok = g[g["outcome"] == 1]
    lg = g[g["length_class"] == "Long"]
    sub = (f"{len(ok)}/{len(g)} completed · long {int((lg['outcome'] == 1).sum())}/{len(lg)} · "
           f"short+medium {int(((g['length_class'] != 'Long') & (g['outcome'] == 1)).sum())}/{int((g['length_class'] != 'Long').sum())}"
           if len(g) else "no goalkeeper passes found")
    fig = _pitch(th, f"GK distribution · {_who(gk, names) if gk else 'GK not identified'}", sub, height=470)
    _add_moves(fig, g[g["outcome"] != 1], th.bad, "Unsuccessful", names, th, symbol="x", width=1.4)
    _add_moves(fig, ok[ok["length_class"] == "Long"], th.accent, "Long, completed", names, th, width=2)
    _add_moves(fig, ok[ok["length_class"] != "Long"], th.team, "Short / medium, completed", names, th, width=2)
    if g.empty:
        _empty(fig, th, "No goalkeeper passes")
    return fig


def zone_losses(md: MatchData, team: str, losses: pd.DataFrame, title: str, *, theme: Theme = DARK,
                names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    fig = _pitch(th, f"{team} — {title}", f"{len(losses)} ball losses · hover for who / when / how", height=470)
    if losses.empty:
        _empty(fig, th, "No ball losses here")
        return fig
    lt = {"pass": "Pass lost", "dribble": "Dribble lost", "ball_lost": "Ball lost", "error": "Ball lost (error)"}
    fig.add_trace(go.Scatter(x=losses["x"], y=losses["y"], mode="markers", name=f"Ball loss ({len(losses)})",
                             text=[_point_hover(r, names, lt.get(r["etype"], r["etype"]), ", ".join(r["labels"]))
                                   for _, r in losses.iterrows()], hovertemplate="%{text}<extra></extra>",
                             marker=dict(size=12, color=th.bad, symbol="x", line=dict(color=th.bg, width=1))))
    return fig


# ------------------------------------------------------------------ progression
def f3_entry_routes(md: MatchData, team: str, *, half=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    pr = T.progression(md, team, half)
    mv = pr["entries"]
    t2 = md.thirds[1]
    fig = _pitch(th, f"{team} — How we enter the final third", _subtitle(md, team, half, extra=(
        f"{int(mv['success'].sum())}/{len(mv)} successful entries · colour = method · right side = per-channel totals")))
    fig.add_shape(type="line", x0=t2, x1=t2, y0=0, y1=PITCH_W, line=dict(color=th.accent, width=1.5, dash="dash"))
    for yv in CHANNEL_EDGES[1:-1]:
        fig.add_shape(type="line", x0=t2, x1=PITCH_L, y0=yv, y1=yv, line=dict(color=th.line, width=0.8, dash="dot"), layer="below")
    for m, g in mv.groupby("method"):
        col = getattr(th, _METHOD_COL.get(m, "team"))
        _add_moves(fig, g[g["success"]], col, f"{m}", names, th, width=1.8)
        _add_moves(fig, g[~g["success"]], _mix(col, th.bad, 0.6), f"{m} (failed)", names, th, symbol="x", width=1.1)
    for k, ch in enumerate(CHANNELS):
        g = mv[mv["end_channel"] == ch]
        if not len(g):
            continue
        y0, y1 = CHANNEL_EDGES[k], CHANNEL_EDGES[k + 1]
        fig.add_annotation(x=PITCH_L + 1.5, y=(y0 + y1) / 2, xanchor="left", showarrow=False,
                           text=f"<b>{int(g['success'].sum())}/{len(g)}</b>",
                           hovertext=f"<b>{ch}</b>: {int(g['success'].sum())} of {len(g)} entries successful<br>"
                                     + ", ".join(f"{m} {n}" for m, n in g['method'].value_counts().items()),
                           font=dict(size=11, color=th.text))
    fig.update_xaxes(range=[-3, PITCH_L + 12])
    if mv.empty:
        _empty(fig, th, "No final-third entries")
    return fig


def entry_breakdown(md: MatchData, team: str, *, half=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    pr = T.progression(md, team, half)
    fig = make_subplots(rows=1, cols=3, subplot_titles=["By method", "By channel (where it lands)", "By player"],
                        horizontal_spacing=0.14)
    for i, (key, lab) in enumerate((("by_method", None), ("by_channel", None), ("by_player", "player")), 1):
        g = pr[key]
        if g.empty:
            continue
        g = g.head(8).iloc[::-1]
        ys = [_who(p, names) for p in g.index] if lab else list(g.index)
        rate = np.where(g["attempts"] > 0, g["successful"] / g["attempts"].where(g["attempts"] > 0, 1), np.nan)
        hover = [f"<b>{y}</b><br>{int(s)}/{int(a)} successful ({r * 100:.0f}%)" if a else f"<b>{y}</b><br>none"
                 for y, s, a, r in zip(ys, g["successful"], g["attempts"], rate)]
        fig.add_trace(go.Bar(y=ys, x=g["attempts"], orientation="h", marker_color=_mix(th.team, th.bg, 0.65),
                             hovertext=hover, hoverinfo="text", showlegend=i == 1, name="Attempts"), 1, i)
        fig.add_trace(go.Bar(y=ys, x=g["successful"], orientation="h", marker_color=[_acc_color(th, r) for r in rate],
                             hovertext=hover, hoverinfo="text", showlegend=i == 1, name="Successful (colour = rate)"), 1, i)
    _base(fig, th, f"{team} — Final-third entries", "faint = attempts · solid = successful", 360)
    fig.update_layout(barmode="overlay")
    for a in fig.layout.annotations:
        a.font = dict(color=th.muted, size=12)
    return fig


def progressive_origins(md: MatchData, team: str, *, half=None, theme: Theme = DARK, names: dict | None = None,
                        bins: tuple = (6, 4)) -> go.Figure:
    th, names = theme, names or {}
    pr = T.progression(md, team, half)["prog_origins"]
    fig = _pitch(th, f"{team} — Where our progressive actions start", "shade = count · hover a zone for details", height=470)
    nx, ny = bins
    wx, wy = PITCH_L / nx, PITCH_W / ny
    counts = []
    for i in range(nx):
        for j in range(ny):
            x0, x1, y0, y1 = i * wx, (i + 1) * wx, j * wy, (j + 1) * wy
            counts.append((x0, x1, y0, y1, pr[(pr["x"] >= x0) & (pr["x"] < x1) & (pr["y"] >= y0) & (pr["y"] < y1)]))
    mx = max([len(z) for *_, z in counts] + [1])
    for x0, x1, y0, y1, z in counts:
        if not len(z):
            continue
        top = z[z["player"].ne("0")]["player"].value_counts().head(3)
        txt = (f"<b>{len(z)} progressive actions</b> started here<br>passes {int((z['etype'] == 'pass').sum())} · "
               f"carries {int((z['etype'] == 'carry').sum())} · avg gain {(z['end_x'] - z['x']).mean():.1f} m forward<br>"
               f"Most: " + ", ".join(f"{_who(k, names)} ({v})" for k, v in top.items()))
        _rect(fig, x0, x1, y0, y1, _heat_color(th, len(z) / mx), txt, 0.88, line=dict(color=th.bg, width=1.5))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=str(len(z)), showarrow=False,
                           font=dict(size=12, color=th.text))
    return fig


def time_to_final_third(md: MatchData, team: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    th = theme
    t = T.progression(md, team, half)["time_to_f3"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Seconds from possession start to the final third",
                                                        "Passes before reaching the final third"])
    if not t.empty:
        for col, i, bins in (("seconds", 1, [0, 3, 6, 10, 15, 20, 30, 999]), ("passes", 2, [0, 1, 2, 3, 4, 6, 9, 999])):
            labels = [f"{a}–{b}" if b < 999 else f"{a}+" for a, b in zip(bins[:-1], bins[1:])]
            c = pd.cut(t[col], bins=bins, right=False, labels=labels).value_counts().reindex(labels).fillna(0)
            fig.add_trace(go.Bar(x=labels, y=c.values, marker_color=th.team if i == 1 else th.accent, showlegend=False,
                                 hovertext=[f"{lb}: <b>{int(v)}</b> possessions ({v / len(t) * 100:.0f}%)" for lb, v in zip(labels, c.values)],
                                 hoverinfo="text"), 1, i)
    sub = (f"{len(t)} possessions reached the final third · median {t['seconds'].median():.1f} s and "
           f"{t['passes'].median():.0f} passes" if not t.empty else "none")
    _base(fig, th, f"{team} — Speed of progression", sub, 320)
    for a in fig.layout.annotations:
        a.font = dict(color=th.muted, size=12)
    return fig


def lines_broken(md: MatchData, team: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    th = theme
    ev = md.events[(md.events["team"] == team) & (md.events["etype"] == "pass") & md.events["line_break"]]
    if half is not None:
        ev = ev[ev["half"] == half]
    g = ev.assign(line=ev["line"].replace("", "Line not tagged")).groupby("line")["outcome"].agg(
        attempts="size", successful=lambda s: int((s == 1).sum()))
    order = [x for x in ("First line", "Midfield line", "Last line", "Line not tagged") if x in g.index]
    g = g.reindex(order)
    rate = g["successful"] / g["attempts"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(g.index), y=g["attempts"], name="Attempts", marker_color=_mix(th.team, th.bg, 0.65),
                         hovertext=[f"<b>{i}</b><br>{int(s)}/{int(a)} ({r * 100:.0f}%)" for i, s, a, r in zip(g.index, g['successful'], g['attempts'], rate)],
                         hoverinfo="text"))
    fig.add_trace(go.Bar(x=list(g.index), y=g["successful"], name="Successful", marker_color=[_acc_color(th, r) for r in rate],
                         text=[f"{r * 100:.0f}%" for r in rate], textposition="outside",
                         hovertext=[f"<b>{i}</b><br>{int(s)}/{int(a)} ({r * 100:.0f}%)" for i, s, a, r in zip(g.index, g['successful'], g['attempts'], rate)],
                         hoverinfo="text"))
    _base(fig, th, f"{team} — Lines broken by passes", f"{len(ev)} line-breaking passes", 320)
    fig.update_layout(barmode="overlay")
    return fig


# ------------------------------------------------------------------ final third
def f3_channels(md: MatchData, team: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    th = theme
    c = T.final_third(md, team, half)["channels"]
    fig = go.Figure()
    for col, name, color in (("completed", "Passes completed", th.team), ("key passes", "Key passes", th.accent),
                             ("crosses", "Crosses", th.carry), ("shots", "Shots", th.good), ("losses", "Ball losses", th.bad)):
        fig.add_trace(go.Bar(x=list(c.index), y=c[col], name=name, marker_color=color,
                             hovertext=[f"<b>{ch}</b> · {name}: {int(v)}<br>actions {int(r['actions'])} · passes "
                                        f"{int(r['completed'])}/{int(r['passes'])}" for ch, v, (_, r) in zip(c.index, c[col], c.iterrows())],
                             hoverinfo="text"))
    _base(fig, th, f"{team} — Final third by channel", "where in the final third we play, create and lose the ball", 360)
    fig.update_layout(barmode="group", bargap=0.25)
    return fig


def chance_creation(md: MatchData, team: str, *, half=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    f = T.final_third(md, team, half)
    kp, s = f["key_passes"], f["shots"]
    fig = _pitch(th, f"{team} — Chance creation", _subtitle(md, team, half, extra=(
        f"{len(kp)} key passes / assists · {len(s)} shots · xG {s['xg'].sum():.2f} · shot colour = source")))
    _add_moves(fig, kp, th.accent, "Key pass / assist", names, th, width=2.2)
    src_col = {"After a pass": th.team, "After a cross": th.carry, "After a through ball": th.good,
               "Transition / regain": th.good, "Set piece": th.muted, "Penalty": th.text, "Individual / rebound": th.opp}
    for srcn, g in s.groupby("source"):
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers", name=f"Shot · {srcn} ({len(g)})",
                                 text=[_point_hover(r, names, f"Shot — {r['shot_result'] or 'no outcome'}",
                                                    f"source: {srcn} · xG <b>{r['xg']:.2f}</b> · {r['body_part'] or ''}")
                                       for _, r in g.iterrows()], hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=(g["xg"].fillna(0.03) * 55 + 10).clip(10, 55), color=src_col.get(srcn, th.team),
                                             symbol=["star" if gg else "circle" for gg in g["shot_goal"]],
                                             line=dict(color=th.bg, width=1))))
    fig.update_xaxes(range=[PITCH_L / 2 - 3, PITCH_L + 3])
    if kp.empty and s.empty:
        _empty(fig, th, "No chances created")
    return fig


def shot_sources(md: MatchData, team: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    th = theme
    s = T.final_third(md, team, half)["shots"]
    g = s.groupby("source").agg(shots=("xg", "size"), xg=("xg", "sum"), goals=("shot_goal", "sum"),
                                on_target=("shot_on", "sum")).sort_values("shots")
    fig = go.Figure(go.Bar(y=list(g.index), x=g["shots"], orientation="h", marker_color=th.team,
                           text=[f"{int(a)} · xG {b:.2f}" for a, b in zip(g["shots"], g["xg"])], textposition="outside",
                           cliponaxis=False,
                           hovertext=[f"<b>{i}</b><br>{int(r['shots'])} shots · {int(r['on_target'])} on target · "
                                      f"{int(r['goals'])} goals · xG {r['xg']:.2f}" for i, r in g.iterrows()], hoverinfo="text"))
    _base(fig, th, f"{team} — Where our shots come from", f"{len(s)} shots · xG {s['xg'].sum():.2f}", 110 + 34 * max(len(g), 1))
    fig.update_layout(xaxis=dict(visible=False, range=[0, (g["shots"].max() if len(g) else 1) * 1.6]))
    return fig


def crosses_by_flank(md: MatchData, team: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    th = theme
    cr = T.final_third(md, team, half)["crosses"]
    rows = []
    for fl in ("Left flank", "Right flank"):
        g = cr[cr["flank"] == fl]
        rows.append((fl, len(g), int((g["outcome"] == 1).sum()), int(g["key_pass"].sum()),
                     int((g["height"] == "High").sum()), int((g["height"].isin(["Low", "Ground"])).sum())))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows], name="Crosses", marker_color=_mix(th.carry, th.bg, 0.55),
                         hoverinfo="text", hovertext=[f"<b>{r[0]}</b><br>{r[2]}/{r[1]} completed · key passes {r[3]}<br>"
                                                      f"high {r[4]} · low/ground {r[5]}" for r in rows]))
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[r[2] for r in rows], name="Completed", marker_color=th.carry,
                         text=[f"{r[2]}/{r[1]}" for r in rows], textposition="outside", hoverinfo="skip"))
    _base(fig, th, f"{team} — Crosses by flank", f"{len(cr)} crosses · {int((cr['outcome'] == 1).sum())} completed", 320)
    fig.update_layout(barmode="overlay")
    return fig
