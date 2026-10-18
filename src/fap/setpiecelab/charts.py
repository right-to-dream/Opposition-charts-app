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
                bits.append(f"{c} won by {sd.team} {int(r[f'{c} won'])}/{int(r[f'{c} decided'])}")
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
        fig.add_trace(go.Bar(x=d["Set piece"], y=rate, name=f"{c} won by {sd.team}", marker_color=col,
                             text=[f"{int(a)}/{int(b)}" for a, b in zip(d[f"{c} won"], d[f"{c} decided"])],
                             textposition="outside", cliponaxis=False,
                             hovertext=[f"<b>{s}</b><br>{c} won by {sd.team}: {int(a)} of {int(b)} ({p:.0f}%)"
                                        for s, a, b, p in zip(d["Set piece"], d[f"{c} won"], d[f"{c} decided"], rate)],
                             hoverinfo="text"))
    _base(fig, th, f"{sd.team} — 1st contact & 2nd ball", f"share of the contests {sd.team} won — attacking and defending", 380)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 115], ticksuffix="%"), bargap=0.3)
    return fig


def _hover_row(sd: SetPieceData, r: pd.Series) -> str:
    q = " · ".join(f"{k}: <b>{r[k]}</b>" for k in sd.qualifiers if r[k])
    extra = f"<br>extra points: {', '.join(f'({x:.0f}, {y:.0f})' for x, y in r['points'][1:])}" if r["extra_points"] else ""
    note = f"<br>note: {r['notes']}" if r["notes"] else ""
    dur = f" · clip {r['duration']:.1f}s" if pd.notna(r["duration"]) else ""
    sb = (f"<br>second ball x {r['sb_x']:.0f}, y {r['sb_y']:.0f} m ({A.second_ball_zone(r['sb_x'], r['sb_y']).lower()})"
          if "sb_x" in r and pd.notna(r["sb_x"]) else "")
    side = f" · from the {r['landing_side'].lower()}" if r.get("landing_side") else ""
    return (f"<b>{r['instance']}</b>{dur}<br>landing x {r['x']:.0f}, y {r['y']:.0f} m{side}{sb}<br>{q}{extra}{note}")


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
    sb = pts[pts["sb_x"].notna()] if "sb_x" in pts else pts.iloc[0:0]
    if len(sb):                                              # second ball: where the loose ball was won
        xs, ys = [], []
        for _, r in sb.iterrows():
            xs += [r["x"], r["sb_x"], None]
            ys += [r["y"], r["sb_y"], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=th.carry, width=1.2, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=sb["sb_x"], y=sb["sb_y"], mode="markers", name=f"Second ball ({len(sb)})",
                                 hovertext=[_hover_row(sd, r) for _, r in sb.iterrows()], hoverinfo="text",
                                 marker=dict(size=12, symbol="diamond", color=th.carry, line=dict(color=th.bg, width=1))))
    allx = pd.concat([pts["x"], sb["sb_x"]]) if len(sb) else pts["x"]
    if allx.min() > PITCH_L / 2 - 5:                         # everything in the attacking half → zoom in
        fig.update_xaxes(range=[PITCH_L / 2 - 3, PITCH_L + 3])
    return fig


def category_compare(sd: SetPieceData, cat: str, *, theme: Theme = DARK, max_quals: int = 6) -> go.Figure:
    """Attacking vs defending of one category: every coded qualifier as % of that phase's coded instances."""
    th = theme
    pc = A.phase_compare(sd, cat)
    plural = A.CATEGORY_PLURAL.get(cat, cat)
    if pc.empty:
        return _base(go.Figure(), th, f"{sd.team} — {plural}", "nothing coded")
    quals = [q for q in sd.qualifiers if q in set(pc["qualifier"])][:max_quals]
    n = len(quals)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=quals, horizontal_spacing=0.16, vertical_spacing=0.2)
    for i, q in enumerate(quals):
        r, c = divmod(i, cols)
        g = pc[pc["qualifier"] == q]
        order = g.groupby("value")["count"].sum().sort_values().index.tolist()
        for ph, col in (("Attacking", th.team), ("Defending", th.opp)):
            d = g[g["phase"] == ph].set_index("value").reindex(order)
            fig.add_trace(go.Bar(y=order, x=d["share"].fillna(0), orientation="h", name=f"{sd.team} {ph.lower()}",
                                 marker_color=col, showlegend=i == 0, legendgroup=ph,
                                 text=[f"{int(k)}" if k == k and k else "" for k in d["count"]], textposition="outside",
                                 cliponaxis=False,
                                 hovertext=[f"<b>{q}: {v}</b><br>{ph}: {int(k) if k == k else 0} of {int(o) if o == o else 0}"
                                            f" ({sh:.0f}%)" if sh == sh else f"<b>{q}: {v}</b><br>{ph}: 0"
                                            for v, k, o, sh in zip(order, d["count"], d["of"], d["share"])],
                                 hoverinfo="text"), r + 1, c + 1)
        fig.update_xaxes(range=[0, 125], visible=False, row=r + 1, col=c + 1)
    _base(fig, th, f"{sd.team} — {plural}: taking vs defending",
          "% of each phase's coded instances · label = count · hover for details", 150 + 230 * rows)
    fig.update_layout(barmode="group", bargap=0.25, legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    for a in fig.layout.annotations:
        a.font = dict(color=th.muted, size=12)
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
            if kl.endswith(" won"):
                yn = not kl.startswith("opponent")
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


def contact_map(sd: SetPieceData, row: str, which: int, *, theme: Theme = DARK) -> go.Figure:
    """1st-contact (landing point) or 2nd-contact (second-ball point) map, coloured by that contact's outcome;
    circle = the file's team won the contact, ✕ = the opponent won it."""
    th = theme
    outc, won = A.contact_columns(sd, which)
    pts = A.contact_points(sd, row, which)
    g = sd.of(row=row)
    name = "1st contact" if which == 1 else "2nd contact (second ball)"
    where = "where the ball lands" if which == 1 else "where the second ball is played"
    fig = _pitch(th, f"{row} — {name} outcome", f"{len(pts)} of {len(g)} located · {where} · colour = "
                 f"{outc or 'outcome'} · ● {sd.team} won it · ✕ opponent won it · hover any point", height=540)
    if pts.empty:
        fig.add_annotation(x=PITCH_L / 2, y=PITCH_W / 2, text=f"No {name} locations coded for this set piece",
                           showarrow=False, font=dict(color=th.muted, size=15))
        return fig
    vals = pts[outc].replace("", "Not coded") if outc else pd.Series("Not coded", index=pts.index)
    palette = _palette(th)
    for i, v in enumerate(vals.value_counts().index):
        d = pts[vals == v]
        col = th.muted if v == "Not coded" else A.OUTCOME_COLORS.get(v.lower(), palette[i % len(palette)])
        yn = d[won].map(is_yes) if won else pd.Series(None, index=d.index)
        sym = ["circle" if y is True else "x" if y is False else "circle-open" for y in yn]
        fig.add_trace(go.Scatter(x=d["cx"], y=d["cy"], mode="markers+text", name=f"{v} ({len(d)})",
                                 text=[str(n) for n in d["n"]], textposition="top center",
                                 textfont=dict(size=9, color=th.muted),
                                 hovertext=[_hover_row(sd, r) for _, r in d.iterrows()], hoverinfo="text",
                                 marker=dict(size=15, color=col, symbol=sym, line=dict(color=th.bg, width=1))))
    if pts["cx"].min() > PITCH_L / 2 - 5:
        fig.update_xaxes(range=[PITCH_L / 2 - 3, PITCH_L + 3])
    return fig
