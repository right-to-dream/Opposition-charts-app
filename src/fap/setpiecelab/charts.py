"""Interactive set-piece charts (Plotly, hover on everything) — reuses the Match Lab pitch and theme."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fap.matchlab.charts import DARK, Theme
from fap.matchlab.interactive import _acc_color, _mix, _pitch
from fap.setpiecelab import analytics as A
from fap.setpiecelab.loader import PITCH_L, PITCH_W, SetPieceData, is_yes


def _base(fig: go.Figure, th: Theme, title: str, sub: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" + (f"<br><span style='font-size:11px;color:{th.muted}'>{sub}</span>" if sub else ""),
                   x=0.01, font=dict(size=15, color=th.text)),
        height=height, paper_bgcolor=th.bg, plot_bgcolor=th.bg,
        font=dict(color=th.text, size=11, family="Segoe UI, Inter, Roboto, Arial, sans-serif"),
        margin=dict(l=10, r=20, t=70 if sub else 50, b=10),
        hoverlabel=dict(bgcolor=th.panel, bordercolor=th.grid, font=dict(color=th.text), align="left"),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(color=th.muted)))
    fig.update_xaxes(gridcolor=th.grid, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    return fig


def _palette(th: Theme) -> list[str]:
    return [th.team, th.accent, th.good, th.carry, th.opp, th.bad, th.muted, th.text]


def overview_bars(sd: SetPieceData, *, theme: Theme = DARK) -> go.Figure:
    th = theme
    ov = A.overview(sd).sort_values("Count")
    cols = [th.team if p == "Attacking" else th.opp for p in ov["Phase"]]
    hover = []
    for _, r in ov.iterrows():
        bits = [f"<b>{r['Set piece']}</b> ({r['Phase'].lower()} for {sd.team})", f"{int(r['Count'])} coded"]
        for c in A.CONTESTS:
            if f"{c} decided" in r and r[f"{c} decided"]:
                bits.append(f"{c} = Yes {int(r[f'{c} won'])}/{int(r[f'{c} decided'])}")
        if "Goals" in r:
            bits.append(f"goals {int(r['Goals'])} · shots {int(r['Shots'])} · dangerous {int(r['Dangerous (goal/shot/threat)'])}")
        for k in ("Main execution", "Main target", "Main side"):
            if k in r and r[k] != "–":
                bits.append(f"{k.lower()}: {r[k]}")
        hover.append("<br>".join(bits))
    fig = go.Figure(go.Bar(y=ov["Set piece"], x=ov["Count"], orientation="h", marker_color=cols, text=ov["Count"],
                           textposition="outside", cliponaxis=False, hovertext=hover, hoverinfo="text"))
    _base(fig, th, f"{sd.team} — set pieces coded", f"blue = {sd.team} taking it · red = {sd.team} defending it",
          110 + 34 * len(ov))
    fig.update_layout(xaxis=dict(visible=False, range=[0, ov["Count"].max() * 1.25]))
    return fig


def contest_rates(sd: SetPieceData, *, theme: Theme = DARK) -> go.Figure:
    th = theme
    ov = A.overview(sd)
    fig = go.Figure()
    for c, col in zip(A.CONTESTS, (th.team, th.accent)):
        if f"{c} decided" not in ov:
            continue
        d = ov[ov[f"{c} decided"] > 0]
        rate = d[f"{c} won"] / d[f"{c} decided"] * 100
        fig.add_trace(go.Bar(x=d["Set piece"], y=rate, name=f"{c} = Yes", marker_color=col,
                             text=[f"{int(a)}/{int(b)}" for a, b in zip(d[f"{c} won"], d[f"{c} decided"])],
                             textposition="outside", cliponaxis=False,
                             hovertext=[f"<b>{s}</b><br>{c} coded Yes: {int(a)} of {int(b)} ({p:.0f}%)"
                                        for s, a, b, p in zip(d["Set piece"], d[f"{c} won"], d[f"{c} decided"], rate)],
                             hoverinfo="text"))
    _base(fig, th, f"{sd.team} — 1st contact & 2nd ball", "share of coded instances marked Yes", 380)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 115], ticksuffix="%"), bargap=0.3)
    return fig


def _hover_row(sd: SetPieceData, r: pd.Series) -> str:
    q = " · ".join(f"{k}: <b>{r[k]}</b>" for k in sd.qualifiers if r[k])
    extra = f"<br>extra points: {', '.join(f'({x:.0f}, {y:.0f})' for x, y in r['points'][1:])}" if r["extra_points"] else ""
    note = f"<br>note: {r['notes']}" if r["notes"] else ""
    dur = f" · clip {r['duration']:.1f}s" if pd.notna(r["duration"]) else ""
    return (f"<b>{r['instance']}</b>{dur}<br>landing x {r['x']:.0f}, y {r['y']:.0f} m<br>{q}{extra}{note}")


def landing_map(sd: SetPieceData, row: str, *, color_by: str | None = None, theme: Theme = DARK) -> go.Figure:
    th = theme
    g = sd.of(row=row)
    pts = g[g["x"].notna()]
    color_by = color_by if color_by in sd.qualifiers else None
    att = g["phase"].iloc[0] == "Attacking" if len(g) else True
    fig = _pitch(th, f"{row} — where the ball lands",
                 f"{len(pts)} of {len(g)} with a landing point · {'taken by' if att else 'against'} {sd.team}"
                 f"{' · colour = ' + color_by if color_by else ''} · hover any point", height=520)
    if pts.empty:
        fig.add_annotation(x=PITCH_L / 2, y=PITCH_W / 2, text="No landing points coded for this set piece",
                           showarrow=False, font=dict(color=th.muted, size=15))
        return fig
    # extra points (same set piece, later points)
    for _, r in pts[pts["extra_points"] > 0].iterrows():
        xs = [p[0] for p in r["points"]]
        ys = [p[1] for p in r["points"]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", showlegend=False, hoverinfo="skip",
                                 line=dict(color=th.muted, width=1.2, dash="dot"),
                                 marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(color=th.muted, width=1.5))))
    groups = [("All", pts)] if not color_by else [(v or "not coded", pts[pts[color_by] == v]) for v in pts[color_by].value_counts().index]
    if color_by and (pts[color_by] == "").any() and "not coded" not in [k for k, _ in groups]:
        groups.append(("not coded", pts[pts[color_by] == ""]))
    for (name, d), col in zip(groups, _palette(th)):
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="markers+text", name=f"{name} ({len(d)})",
                                 text=[str(n) for n in d["n"]], textposition="top center", textfont=dict(size=9, color=th.muted),
                                 hovertext=[_hover_row(sd, r) for _, r in d.iterrows()], hoverinfo="text",
                                 marker=dict(size=14, color=th.muted if name == "not coded" else col,
                                             line=dict(color=th.bg, width=1))))
    if pts["x"].min() > PITCH_L / 2 - 5:                      # everything in the attacking half → zoom in
        fig.update_xaxes(range=[PITCH_L / 2 - 3, PITCH_L + 3])
    return fig


def qualifier_grid(sd: SetPieceData, row: str, *, theme: Theme = DARK) -> go.Figure:
    th = theme
    br = A.breakdown(sd, row)
    if not br:
        return _base(go.Figure(), th, f"{row} — details", "no qualifiers coded")
    n = len(br)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(br), horizontal_spacing=0.12, vertical_spacing=0.18)
    total = len(sd.of(row=row))
    for i, (q, vc) in enumerate(br.items()):
        r, c = divmod(i, cols)
        vc = vc.iloc[::-1]
        colors = []
        for k in vc.index:
            yn = is_yes(k)
            kl = k.lower()
            colors.append(th.good if yn is True or kl in ("goal",) else th.bad if yn is False else
                          th.accent if kl in ("shot", "threat") else th.team)
        fig.add_trace(go.Bar(y=list(vc.index), x=vc.values, orientation="h", marker_color=colors, showlegend=False,
                             text=[f"{v} ({v / vc.sum() * 100:.0f}%)" for v in vc.values], textposition="outside",
                             cliponaxis=False,
                             hovertext=[f"<b>{q}: {k}</b><br>{v} of {int(vc.sum())} coded ({v / vc.sum() * 100:.0f}%)<br>"
                                        f"{total - int(vc.sum())} of {total} not coded for {q}" for k, v in vc.items()],
                             hoverinfo="text"), r + 1, c + 1)
        fig.update_xaxes(range=[0, vc.max() * 1.6], visible=False, row=r + 1, col=c + 1)
    _base(fig, th, f"{row} — how it is taken / defended", "share of the coded instances · hover a bar", 140 + 190 * rows)
    fig.update_layout(margin=dict(t=105))
    for a in fig.layout.annotations:
        a.font = dict(color=th.muted, size=12)
    return fig


def cross_heat(sd: SetPieceData, row: str, a: str, b: str, *, theme: Theme = DARK) -> go.Figure:
    th = theme
    ct = A.crosstab(sd, row, a, b)
    fig = go.Figure()
    if ct.empty:
        return _base(fig, th, f"{row} — {a} × {b}", "no instances coded with both")
    z = ct.values
    fig.add_trace(go.Heatmap(z=z, x=list(ct.columns), y=list(ct.index), colorscale=[[i / (len(th.heat) - 1), c] for i, c in enumerate(th.heat)],
                             showscale=False, text=z, texttemplate="%{text}",
                             hovertemplate=f"{a}: %{{y}}<br>{b}: %{{x}}<br><b>%{{z}}</b> set pieces<extra></extra>"))
    _base(fig, th, f"{row} — {a} × {b}", f"{int(z.sum())} instances coded with both", 140 + 40 * len(ct.index))
    return fig


def outcome_by(sd: SetPieceData, row: str, by: str, *, theme: Theme = DARK) -> go.Figure:
    """Stacked outcomes per value of another qualifier (e.g. outcome by execution)."""
    th = theme
    out_c = A._col(sd, "Outcome")
    fig = go.Figure()
    if not out_c or by not in sd.qualifiers:
        return _base(fig, th, f"{row} — outcome by {by}", "no outcome column")
    g = sd.of(row=row)
    g = g[g[by] != ""]
    ct = pd.crosstab(g[by], g[out_c].replace("", "No outcome coded"))
    for (oc, col) in zip(ct.columns, _palette(th)):
        fig.add_trace(go.Bar(x=list(ct.index), y=ct[oc], name=oc, marker_color=th.muted if oc == "No outcome coded" else col,
                             hovertext=[f"<b>{by}: {i}</b><br>{oc}: {int(v)} of {int(ct.loc[i].sum())}" for i, v in ct[oc].items()],
                             hoverinfo="text"))
    _base(fig, th, f"{row} — outcome by {by}", "stacked: every coded instance of each value", 360)
    fig.update_layout(barmode="stack")
    return fig
