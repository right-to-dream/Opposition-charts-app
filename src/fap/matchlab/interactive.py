"""Interactive (hover) versions of Match Lab pitch charts, built with Plotly.

The matplotlib charts in :mod:`.charts` stay the report / PNG versions; these figures are for
analysis inside the app: hover any point for its details (player, time, coordinates, outcome ...),
zoom, and toggle legend groups. Same data, same definitions — they read the same engine outputs.

Coordinates in the hover are given twice: ``x / y`` in the team's attacking frame (attack →,
y = 0 right touchline) and the file's own ``pos_x / pos_y`` so a row can be found in the export.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fap.matchlab import counterpress as CP
from fap.matchlab.charts import DARK, PASS_FILTERS, Theme
from fap.matchlab.loader import PITCH_L, PITCH_W, MatchData


# ------------------------------------------------------------------ pitch
def _pitch(th: Theme, title: str, subtitle: str, height: int = 620) -> go.Figure:
    fig = go.Figure()
    line = dict(color=th.line, width=1.4)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=PITCH_L, y1=PITCH_W),
        dict(type="line", x0=PITCH_L / 2, y0=0, x1=PITCH_L / 2, y1=PITCH_W),
        dict(type="circle", x0=PITCH_L / 2 - 9.15, y0=PITCH_W / 2 - 9.15, x1=PITCH_L / 2 + 9.15, y1=PITCH_W / 2 + 9.15),
    ]
    for x0, x1 in ((0, 16.5), (PITCH_L - 16.5, PITCH_L)):
        shapes.append(dict(type="rect", x0=x0, y0=PITCH_W / 2 - 20.16, x1=x1, y1=PITCH_W / 2 + 20.16))
    for x0, x1 in ((0, 5.5), (PITCH_L - 5.5, PITCH_L)):
        shapes.append(dict(type="rect", x0=x0, y0=PITCH_W / 2 - 9.16, x1=x1, y1=PITCH_W / 2 + 9.16))
    for x0, x1 in ((-1.5, 0), (PITCH_L, PITCH_L + 1.5)):
        shapes.append(dict(type="rect", x0=x0, y0=PITCH_W / 2 - 3.66, x1=x1, y1=PITCH_W / 2 + 3.66))
    for s in shapes:
        s.update(line=line, layer="below")
    fig.update_layout(
        shapes=shapes, height=height, paper_bgcolor=th.bg, plot_bgcolor=th.pitch,
        margin=dict(l=10, r=10, t=80, b=10),
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:{th.muted}'>{subtitle}</span>",
                   x=0.01, xanchor="left", font=dict(size=19, color=th.text)),
        font=dict(family="Segoe UI, Inter, Roboto, Arial, sans-serif", color=th.text),
        xaxis=dict(range=[-3, PITCH_L + 3], visible=False, fixedrange=False),
        yaxis=dict(range=[-6, PITCH_W + 3], visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)",
                    font=dict(color=th.muted, size=12)),
        hoverlabel=dict(bgcolor=th.panel, bordercolor=th.grid, font=dict(color=th.text, size=12), align="left"),
        dragmode="zoom",
    )
    fig.add_annotation(x=PITCH_L * 0.62, y=-3.6, ax=PITCH_L * 0.38, ay=-3.6, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.6, arrowcolor=th.muted, text="")
    fig.add_annotation(x=PITCH_L / 2, y=-5.4, text="ATTACKING DIRECTION", showarrow=False,
                       font=dict(size=10, color=th.muted))
    return fig


def _mmss(minute: float) -> str:
    if minute is None or (isinstance(minute, float) and np.isnan(minute)):
        return "–"
    m = int(minute)
    return f"{m}:{int(round((minute - m) * 60)) % 60:02d}"


def _who(num, names: dict) -> str:
    num = "" if num is None or (isinstance(num, float) and np.isnan(num)) else str(num)
    if not num or num == "0":
        return "unknown player"
    return f"#{num} {names.get(num, '')}".strip()


def _fmt(v, d=1, suffix="") -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}{suffix}"


def _subtitle(md: MatchData, team: str, half, player=None, extra="") -> str:
    parts = [f"{team} vs {md.opponent(team)}", {None: "Full match", 1: "1st half", 2: "2nd half"}.get(half, f"Half {half}")]
    if player:
        parts.append(f"Player #{player}")
    if extra:
        parts.append(extra)
    return "  ·  ".join(parts) + "  ·  hover any point for details"


# ------------------------------------------------------------------ counter-press
def counterpress(md: MatchData, team: str, *, half=None, theme: Theme = DARK, window: float | None = None,
                 names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    window = float(window or md.cp_window)
    d = CP.analyse(md, team, window, half)
    fig = _pitch(th, f"{team} — Counter-press after losing the ball",
                 _subtitle(md, team, half, extra=f"first {window:.0f} s (+{CP.LATE_EXTRA:.0f} s late) after each open-play loss"))
    if d.empty:
        fig.add_annotation(x=PITCH_L / 2, y=PITCH_W / 2, text="No open-play ball losses", showarrow=False,
                           font=dict(color=th.muted, size=15))
        return fig
    ev = md.events.set_index("event_id", drop=False)
    # regain arrows (loss -> regain location)
    rg = d[d["regained"] & d["regain_x"].notna()]
    for _, r in rg.iterrows():
        fig.add_annotation(x=r["regain_x"], y=r["regain_y"], ax=r["x"], ay=r["y"], xref="x", yref="y", axref="x",
                           ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
                           arrowcolor="rgba(140,150,170,0.55)", text="")
    style = {CP.CATEGORIES[0]: (th.good, "circle"), CP.CATEGORIES[1]: (th.accent, "x"),
             CP.CATEGORIES[2]: (th.team, "circle"), CP.CATEGORIES[3]: (th.bad, "x")}
    for cat, (col, sym) in style.items():
        g = d[d["category"] == cat]
        if g.empty:
            continue
        hover = []
        for _, r in g.iterrows():
            lid = r["loss_event_id"]
            at = lid if r.get("loss_moment") == "our last action" else r.get("won_event_id")
            src = ev.loc[int(at)] if pd.notna(at) and int(at) in ev.index else None
            fx = f"{src['src_x']:.1f}" if src is not None and pd.notna(src["src_x"]) else "–"
            fy = f"{src['src_y']:.1f}" if src is not None and pd.notna(src["src_y"]) else "–"
            if r["pressed"]:
                press = (f"<b>Pressed</b> after {_fmt(r['seconds_to_press'])} s ({r['press_timing']}) · "
                         f"{r['press_evidence']}" + (f" · {_who(r['first_presser'], names)}" if r["first_presser"] else ""))
            else:
                press = "<b>No counter-press</b>"
            if r["regained"]:
                reg = f"<b>Regained</b> after {_fmt(r['seconds_to_regain'])} s by {_who(r['regained_by'], names)}"
                if r["regain_to_shot"]:
                    reg += " → shot"
                elif r["regain_to_final_third"]:
                    reg += " → final third"
            else:
                reg = "<b>Not regained</b>"
                if r["opp_shot"]:
                    reg += " · opponent shot"
                elif r["opp_reached_final_third"]:
                    reg += " · opponent reached our final third"
            hover.append(
                f"<b>{cat}</b><br>"
                f"Loss · event {int(lid) if pd.notna(lid) else '–'} · {_mmss(r['minute'])} (half {r['half']})<br>"
                f"{_who(r['player'], names)} · {r['loss_action']} · won by opponent with {r['won_by_opponent_with']}<br>"
                f"Where: x {r['x']:.1f} m · y {r['y']:.1f} m ({r['third']}) · file pos {fx}, {fy}"
                f" (event {int(at) if pd.notna(at) else '–'})<br>"
                f"{press}<br>{reg}")
        fig.add_trace(go.Scatter(
            x=g["x"], y=g["y"], mode="markers", name=f"{cat} ({len(g)})",
            marker=dict(size=13, color=col, symbol=sym, line=dict(color=th.bg, width=1)),
            hovertemplate="%{text}<extra></extra>", text=hover))
    # first-press locations (small diamonds), linked by hover text
    pr = d[d["pressed"] & d["press_x"].notna()]
    if len(pr):
        fig.add_trace(go.Scatter(
            x=pr["press_x"], y=pr["press_y"], mode="markers", name="First pressure", visible="legendonly",
            marker=dict(size=8, color=th.text, symbol="diamond", opacity=0.7),
            text=[f"First pressure · {r['press_evidence']} · {_who(r['first_presser'], names)}<br>"
                  f"{_fmt(r['seconds_to_press'])} s after the loss at {_mmss(r['minute'])}" for _, r in pr.iterrows()],
            hovertemplate="%{text}<extra></extra>"))
    s = CP.summary(d)
    fig.add_annotation(
        x=PITCH_L / 2, y=PITCH_W + 1.2, xref="x", yref="y", xanchor="center", yanchor="bottom", showarrow=False,
        text=(f"Losses <b>{s['losses']}</b>  ·  counter-pressed <b>{s['pressed']}</b> ({s['late_presses']} late)  ·  "
              f"regained <b>{s['regained']}</b>  ·  regain rate pressed / not <b>{_fmt(s['regain_rate_when_pressed'], 0, '%')}</b>"
              f" / <b>{_fmt(s['regain_rate_without_press'], 0, '%')}</b>"),
        font=dict(size=13, color=th.text))
    fig.update_layout(yaxis_range=[-6, PITCH_W + 6])
    return fig


# ------------------------------------------------------------------ pass map
def pass_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, kind: str = "all",
             show_failed: bool = True, open_play_only: bool = False, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    label, fn = PASS_FILTERS.get(kind, PASS_FILTERS["all"])
    p = md.events[(md.events["team"] == team) & (md.events["etype"] == "pass") & md.events["end_x"].notna()]
    if half is not None:
        p = p[p["half"] == half]
    if player:
        p = p[p["player"] == str(player)]
    if open_play_only:
        p = p[p["open_play"]]
    p = p[fn(p).fillna(False).astype(bool)]
    ok, ko = p[p["outcome"] == 1], p[p["outcome"] != 1]
    acc = len(ok) / len(p) * 100 if len(p) else float("nan")
    fig = _pitch(th, f"{team} — {label}", _subtitle(md, team, half, player,
                                                     f"{len(p)} passes · {len(ok)} completed · {_fmt(acc, 0, '%')}"))

    def hover(r) -> str:
        res = "✓ completed" if r["outcome"] == 1 else "✗ unsuccessful" if r["outcome"] == 0 else "? unknown"
        to = f" → {_who(r['receiver'], names)}" if r["receiver"] and r["receiver"] != "0" else ""
        tags = [t for t in ("progressive", "line_break", "f3_entry", "box_entry", "cross", "through_ball", "key_pass",
                            "switch", "under_pressure") if bool(r.get(t))]
        nice = {"f3_entry": "final-third entry", "box_entry": "box entry", "line_break": "line-breaking",
                "through_ball": "through ball", "key_pass": "key pass", "under_pressure": "under pressure"}
        flags = ", ".join(nice.get(t, t) for t in tags)
        return (f"<b>{_who(r['player'], names)}{to}</b><br>"
                f"{res} · {_mmss(r['minute'])} (half {int(r['half'])}) · event {int(r['event_id']) if pd.notna(r['event_id']) else '–'}<br>"
                f"{r['pass_type'] or 'Pass'} · {str(r['direction']).lower()} · {_fmt(r['length'])} m · {r['height'] or 'height n/a'}<br>"
                f"From x {r['x']:.1f}, y {r['y']:.1f} → x {r['end_x']:.1f}, y {r['end_y']:.1f} m"
                f" · file pos {_fmt(r['src_x'])}, {_fmt(r['src_y'])}<br>"
                f"{r['phase'] or ''}{' · ' + flags if flags else ''}"
                f"<br><span style='color:{th.muted}'>end point: {r['end_source'] or 'n/a'}</span>")

    for sub, col, name, width in ((ko, th.bad, "Unsuccessful", 1.6), (ok, th.team, "Completed", 2.2)):
        if sub.empty or (name == "Unsuccessful" and not show_failed):
            continue
        xs, ys = [], []
        for _, r in sub.iterrows():
            xs += [r["x"], r["end_x"], None]
            ys += [r["y"], r["end_y"], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=width), opacity=0.55,
                                 hoverinfo="skip", legendgroup=name, showlegend=False))
        txt = [hover(r) for _, r in sub.iterrows()]
        fig.add_trace(go.Scatter(x=sub["end_x"], y=sub["end_y"], mode="markers", name=f"{name} ({len(sub)})",
                                 legendgroup=name, text=txt, hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=8, color=col, symbol="circle" if name == "Completed" else "x",
                                             line=dict(color=th.bg, width=0.8))))
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["y"], mode="markers", legendgroup=name, showlegend=False,
                                 text=txt, hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=5, color=col, opacity=0.6)))
    if p.empty:
        fig.add_annotation(x=PITCH_L / 2, y=PITCH_W / 2, text="No passes for this selection", showarrow=False,
                           font=dict(color=th.muted, size=15))
    return fig


INTERACTIVE = {"counterpress": counterpress, "pass_map": pass_map}


def build(key: str, md: MatchData, team: str, **kw) -> go.Figure | None:
    fn = INTERACTIVE.get(key)
    if fn is None:
        return None
    import inspect
    allowed = set(inspect.signature(fn).parameters)
    fig = fn(md, team, **{k: v for k, v in kw.items() if k in allowed})
    fig.update_traces(hoverlabel=dict(align="left"))
    return fig
