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
def _pitch_shapes(th: Theme, dx: float = 0.0) -> list:
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
        s["x0"] += dx
        s["x1"] += dx
    return shapes


def _pitch(th: Theme, title: str, subtitle: str, height: int = 620) -> go.Figure:
    fig = go.Figure()
    shapes = _pitch_shapes(th)
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
def _pass_traces(fig, ok, ko, th: Theme, hover, show_failed: bool, dx: float = 0.0, legend: bool = True,
                 suffix: str = "") -> None:
    for sub, col, name, width in ((ko, th.bad, "Unsuccessful", 1.6), (ok, th.team, "Completed", 2.2)):
        if sub.empty or (name == "Unsuccessful" and not show_failed):
            continue
        xs, ys = [], []
        for _, r in sub.iterrows():
            xs += [r["x"] + dx, r["end_x"] + dx, None]
            ys += [r["y"], r["end_y"], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=width), opacity=0.55,
                                 hoverinfo="skip", legendgroup=name + suffix, showlegend=False))
        txt = [hover(r) for _, r in sub.iterrows()]
        fig.add_trace(go.Scatter(x=sub["end_x"] + dx, y=sub["end_y"], mode="markers", name=f"{name}{suffix} ({len(sub)})",
                                 legendgroup=name + suffix, showlegend=legend, text=txt, hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=8, color=col, symbol="circle" if name == "Completed" else "x",
                                             line=dict(color=th.bg, width=0.8))))
        fig.add_trace(go.Scatter(x=sub["x"] + dx, y=sub["y"], mode="markers", legendgroup=name + suffix, showlegend=False,
                                 text=txt, hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=5, color=col, opacity=0.6)))


def _pass_hover_fn(th: Theme, names: dict):
    def hover(r) -> str:
        return _move_hover(r, names, th) + f"<br><span style='color:{th.muted}'>end point: {r['end_source'] or 'n/a'}</span>"
    return hover


def _filter_passes(md: MatchData, team: str, half, player, kind: str, open_play_only: bool) -> pd.DataFrame:
    p = md.events[(md.events["team"] == team) & (md.events["etype"] == "pass") & md.events["end_x"].notna()]
    if half is not None:
        p = p[p["half"] == half]
    if player:
        p = p[p["player"] == str(player)]
    if open_play_only:
        p = p[p["open_play"]]
    _, fn = PASS_FILTERS.get(kind, PASS_FILTERS["all"])
    return p[fn(p).fillna(False).astype(bool)]


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

    _pass_traces(fig, ok, ko, th, hover, show_failed)
    if p.empty:
        fig.add_annotation(x=PITCH_L / 2, y=PITCH_W / 2, text="No passes for this selection", showarrow=False,
                           font=dict(color=th.muted, size=15))
    return fig




# ------------------------------------------------------------------ shared helpers for the remaining charts
def _scope(md: MatchData, team: str, half=None, player=None) -> pd.DataFrame:
    ev = md.events[md.events["team"] == team]
    if half is not None:
        ev = ev[ev["half"] == half]
    if player:
        ev = ev[ev["player"] == str(player)]
    return ev


def _eid(r) -> str:
    v = r.get("event_id")
    return str(int(v)) if v is not None and pd.notna(v) else "derived"


def _move_hover(r, names: dict, th: Theme) -> str:
    """Hover for a pass or a carry."""
    nice = {"f3_entry": "final-third entry", "box_entry": "box entry", "line_break": "line-breaking",
            "through_ball": "through ball", "key_pass": "key pass", "under_pressure": "under pressure"}
    tags = [t for t in ("progressive", "line_break", "f3_entry", "box_entry", "cross", "through_ball", "key_pass",
                        "switch", "under_pressure") if bool(r.get(t))]
    flags = ", ".join(nice.get(t, t) for t in tags)
    if r["etype"] == "carry":
        head = f"<b>{_who(r['player'], names)} — carry</b>"
        kind = f"Carry{' (derived: receive → next action)' if r.get('derived') else ''} · {str(r['direction']).lower()} · {_fmt(r['length'])} m"
        res = ""
    else:
        to = f" → {_who(r['receiver'], names)}" if r["receiver"] and r["receiver"] != "0" else ""
        head = f"<b>{_who(r['player'], names)}{to}</b>"
        res = "✓ completed · " if r["outcome"] == 1 else "✗ unsuccessful · " if r["outcome"] == 0 else "? unknown · "
        kind = f"{r['pass_type'] or 'Pass'} · {str(r['direction']).lower()} · {_fmt(r['length'])} m · {r['height'] or 'height n/a'}"
    return (f"{head}<br>{res}{_mmss(r['minute'])} (half {int(r['half'])}) · event {_eid(r)}<br>{kind}<br>"
            f"From x {r['x']:.1f}, y {r['y']:.1f} → x {r['end_x']:.1f}, y {r['end_y']:.1f} m"
            f" · file pos {_fmt(r['src_x'])}, {_fmt(r['src_y'])}<br>"
            f"{r['phase'] or ''}{' · ' + flags if flags else ''}")


def _add_moves(fig: go.Figure, df: pd.DataFrame, color: str, name: str, names: dict, th: Theme,
               symbol: str = "circle", dash: str | None = None, width: float = 2.0, xa: str = "x", ya: str = "y",
               showlegend: bool = True, ring: pd.Series | None = None) -> None:
    """Lines start → end (no hover) + hoverable end and start markers, one legend group."""
    if df.empty:
        return
    xs, ys = [], []
    for _, r in df.iterrows():
        xs += [r["x"], r["end_x"], None]
        ys += [r["y"], r["end_y"], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), opacity=0.6,
                             hoverinfo="skip", legendgroup=name, showlegend=False, xaxis=xa, yaxis=ya))
    txt = [_move_hover(r, names, th) for _, r in df.iterrows()]
    fig.add_trace(go.Scatter(x=df["end_x"], y=df["end_y"], mode="markers", name=f"{name} ({len(df)})", legendgroup=name,
                             showlegend=showlegend, text=txt, hovertemplate="%{text}<extra></extra>", xaxis=xa, yaxis=ya,
                             marker=dict(size=8, color=color, symbol=symbol, line=dict(color=th.bg, width=0.8))))
    fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="markers", legendgroup=name, showlegend=False, text=txt,
                             hovertemplate="%{text}<extra></extra>", xaxis=xa, yaxis=ya,
                             marker=dict(size=5, color=color, opacity=0.6)))
    if ring is not None and ring.any():
        rr = df[ring.reindex(df.index).fillna(False).astype(bool)]
        fig.add_trace(go.Scatter(x=rr["end_x"], y=rr["end_y"], mode="markers", legendgroup=name, showlegend=False,
                                 hoverinfo="skip", xaxis=xa, yaxis=ya,
                                 marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=th.accent, width=1.8))))


def _point_hover(r, names: dict, title: str, extra: str = "") -> str:
    return (f"<b>{title}</b><br>{_who(r['player'], names)} · {_mmss(r['minute'])} (half {int(r['half'])}) · event {_eid(r)}<br>"
            f"x {r['x']:.1f} m · y {r['y']:.1f} m ({r['third']}, {r['channel']}) · file pos {_fmt(r['src_x'])}, {_fmt(r['src_y'])}"
            f"{'<br>' + extra if extra else ''}")


def _empty(fig: go.Figure, th: Theme, text: str = "No events for this selection") -> None:
    fig.add_annotation(x=PITCH_L / 2, y=PITCH_W / 2, text=text, showarrow=False, font=dict(color=th.muted, size=15))


# ------------------------------------------------------------------ pass map by half (two pitches)
def pass_map_halves(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, kind: str = "all",
                    show_failed: bool = True, open_play_only: bool = False, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    label = PASS_FILTERS.get(kind, PASS_FILTERS["all"])[0]
    gap = 12.0
    dx2 = PITCH_L + gap
    fig = _pitch(th, f"{team} — {label}: 1st half vs 2nd half", _subtitle(md, team, None, player), height=520)
    fig.update_layout(shapes=list(fig.layout.shapes) + _pitch_shapes(th, dx2),
                      xaxis_range=[-3, dx2 + PITCH_L + 3], yaxis_range=[-6, PITCH_W + 9])
    hover = _pass_hover_fn(th, names)
    for h, dx in ((1, 0.0), (2, dx2)):
        p = _filter_passes(md, team, h, player, kind, open_play_only)
        ok, ko = p[p["outcome"] == 1], p[p["outcome"] != 1]
        _pass_traces(fig, ok, ko, th, hover, show_failed, dx=dx, legend=True, suffix=f" · {h}{'st' if h == 1 else 'nd'} half")
        acc = len(ok) / len(p) * 100 if len(p) else float("nan")
        fig.add_annotation(x=dx, y=PITCH_W + 2, xanchor="left", yanchor="bottom", showarrow=False,
                           text=(f"<b>{'1ST' if h == 1 else '2ND'} HALF</b> · {len(p)} passes · {_fmt(acc, 0, '%')} · "
                                 f"{int(ok['progressive'].sum())} prog. · "
                                 f"{_fmt((p['direction'] == 'Forward').mean() * 100 if len(p) else float('nan'), 0, '%')} fwd"),
                           font=dict(size=11.5, color=th.text))
        if p.empty:
            fig.add_annotation(x=dx + PITCH_L / 2, y=PITCH_W / 2, text="No passes", showarrow=False,
                               font=dict(color=th.muted, size=14))
    # attack arrow under each pitch (the default one sits under the first)
    fig.add_annotation(x=dx2 + PITCH_L * 0.62, y=-3.6, ax=dx2 + PITCH_L * 0.38, ay=-3.6, xref="x", yref="y", axref="x",
                       ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.6, arrowcolor=th.muted, text="")
    return fig


# ------------------------------------------------------------------ progression
def progression(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
                names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    ev = _scope(md, team, half, player)
    pp = ev[(ev["etype"] == "pass") & ev["progressive"] & (ev["outcome"] == 1)]
    pf = ev[(ev["etype"] == "pass") & ev["progressive"] & (ev["outcome"] == 0)]
    pc = ev[(ev["etype"] == "carry") & ev["progressive"] & ev["end_x"].notna()]
    fig = _pitch(th, f"{team} — Ball progression", _subtitle(md, team, half, player,
                 f"{len(pp)} progressive passes · {len(pc)} progressive carries (Wyscout)"))
    for x in md.thirds:
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=PITCH_W, line=dict(color=th.line, width=1, dash="dot"), layer="below")
    _add_moves(fig, pf, th.bad, "Unsuccessful progressive pass", names, th, symbol="x", width=1.4)
    _add_moves(fig, pc, th.carry, "Progressive carry", names, th, dash="dot", width=2)
    _add_moves(fig, pp, th.team, "Progressive pass", names, th, width=2.2)
    if pp.empty and pc.empty:
        _empty(fig, th)
    return fig


# ------------------------------------------------------------------ entries
def entries(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, open_play_only: bool = True,
            names: dict | None = None) -> go.Figure:
    from fap.matchlab.metrics import CHANNEL_EDGES
    th, names = theme, names or {}
    ev = _scope(md, team, half, player)
    mv = ev[ev["etype"].isin(["pass", "carry"]) & ev["end_x"].notna() & (ev["f3_entry"] | ev["box_entry"])]
    if open_play_only:
        mv = mv[mv["open_play"]]
    ok = (mv["outcome"] == 1) | (mv["etype"] == "carry")
    fig = _pitch(th, f"{team} — Final-third & box entries", _subtitle(md, team, half, player,
                 f"{int((mv['f3_entry'] & ok).sum())}/{int(mv['f3_entry'].sum())} final-third · "
                 f"{int((mv['box_entry'] & ok).sum())}/{int(mv['box_entry'].sum())} box · ring = box entry"))
    fig.add_shape(type="line", x0=md.thirds[1], x1=md.thirds[1], y0=0, y1=PITCH_W, line=dict(color=th.accent, width=1.2, dash="dash"))
    for yv in CHANNEL_EDGES[1:-1]:
        fig.add_shape(type="line", x0=md.thirds[1], x1=PITCH_L, y0=yv, y1=yv, line=dict(color=th.line, width=0.8, dash="dot"), layer="below")
    _add_moves(fig, mv[(mv["etype"] == "pass") & (mv["outcome"] != 1)], th.bad, "Unsuccessful", names, th, symbol="x",
               width=1.3, ring=mv["box_entry"])
    _add_moves(fig, mv[mv["etype"] == "carry"], th.carry, "Carry", names, th, dash="dot", ring=mv["box_entry"])
    _add_moves(fig, mv[(mv["etype"] == "pass") & (mv["outcome"] == 1)], th.team, "Completed pass", names, th,
               ring=mv["box_entry"])
    if mv.empty:
        _empty(fig, th)
    return fig


# ------------------------------------------------------------------ crosses
def crosses(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    ev = _scope(md, team, half, player)
    c = ev[(ev["etype"] == "pass") & ev["cross"] & ev["end_x"].notna()]
    ok = c[c["outcome"] == 1]
    fig = _pitch(th, f"{team} — Crosses", _subtitle(md, team, half, player,
                 f"{len(ok)}/{len(c)} completed · {int(c['key_pass'].sum())} key passes"))
    _add_moves(fig, c[c["outcome"] != 1], th.bad, "Unsuccessful", names, th, symbol="x", width=1.6)
    _add_moves(fig, ok, th.good, "Completed", names, th, width=2.2, ring=c["key_pass"])
    fig.update_xaxes(range=[PITCH_L / 2 - 3, PITCH_L + 3])
    if c.empty:
        _empty(fig, th, "No crosses")
    return fig


# ------------------------------------------------------------------ shots
def shots(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    from fap.matchlab.metrics import dist_to_goal
    th, names = theme, names or {}
    ev = md.events
    s = _scope(md, team, half, player)
    s = s[(s["etype"] == "shot") & s["x"].notna()]
    fig = _pitch(th, f"{team} — Shot map", _subtitle(md, team, half, player,
                 f"{len(s)} shots · {int(s['shot_on'].sum())} on target · {int(s['shot_goal'].sum())} goals · "
                 f"xG {s['xg'].sum():.2f} · size = xG"))
    style = {"Goal": (th.accent, "star"), "Saved": (th.team, "circle"), "Blocked": (th.muted, "square"),
             "Off target": (th.bad, "circle-open"), "": (th.muted, "circle-open")}
    for res, (col, sym) in style.items():
        g = s[s["shot_result"] == res]
        if g.empty:
            continue
        txt = []
        for i, r in g.iterrows():
            prev = ev.loc[max(0, i - 4):i - 1]
            ap = prev[(prev["etype"] == "pass") & (prev["team"] == team) & (prev["outcome"] == 1)]
            assist = f"Assisted by {_who(ap.iloc[-1]['player'], names)} ({ap.iloc[-1]['pass_type'] or 'pass'})" if len(ap) else "No assisting pass tagged"
            txt.append(_point_hover(r, names, f"Shot — {res or 'no outcome tagged'}",
                                    f"xG <b>{r['xg']:.2f}</b> · {r['body_part'] or 'body part n/a'} · "
                                    f"{_fmt(float(dist_to_goal(r['x'], r['y'])))} m from goal"
                                    f"{' · ' + r['set_piece_kind'] if r['set_piece_kind'] else ''}<br>{assist}"))
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers", name=f"{res or 'Unknown'} ({len(g)})", text=txt,
                                 hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=(g["xg"].fillna(0.03) * 60 + 10).clip(10, 60), color=col, symbol=sym,
                                             line=dict(color=col if "open" in sym else th.bg, width=1.6))))
    fig.update_xaxes(range=[PITCH_L / 2 - 3, PITCH_L + 3])
    if s.empty:
        _empty(fig, th, "No shots")
    return fig


# ------------------------------------------------------------------ defensive actions
_DEF = {"interception": ("Interception", "circle"), "tackle": ("Tackle", "triangle-up"), "recovery": ("Recovery", "diamond"),
        "duel": ("Duel", "square"), "foul": ("Foul committed", "x"), "block": ("Block", "cross"), "clearance": ("Clearance", "triangle-down"),
        "press": ("Press", "circle-open")}


def defensive(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, include_press: bool = False,
              names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    ev = _scope(md, team, half, player)
    kinds = [k for k in _DEF if include_press or k != "press"]
    d = ev[ev["etype"].isin(kinds) & ev["x"].notna()]
    fig = _pitch(th, f"{team} — Defensive actions", _subtitle(md, team, half, player,
                 f"{len(d)} actions · avg height {d['x'].mean():.1f} m" if len(d) else "no actions"))
    for x in md.thirds:
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=PITCH_W, line=dict(color=th.line, width=1, dash="dot"), layer="below")
    if len(d):
        h = d["x"].mean()
        fig.add_shape(type="line", x0=h, x1=h, y0=0, y1=PITCH_W, line=dict(color=th.accent, width=2, dash="dash"))
    palette = [th.team, th.good, th.accent, th.carry, th.bad, th.text, th.muted, th.opp]
    for (k, (nm, sym)), col in zip(((k, _DEF[k]) for k in kinds), palette):
        g = d[d["etype"] == k]
        if g.empty:
            continue
        txt = [_point_hover(r, names, nm, ", ".join(r["labels"]) + (" · " if r["labels"] else "") +
                            f"{r['def_phase'] or ''}{' · counter-press' if r['counter_press'] else ''}")
               for _, r in g.iterrows()]
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers", name=f"{nm} ({len(g)})", text=txt,
                                 hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=10, color=col, symbol=sym, line=dict(color=th.bg, width=0.8))))
    if d.empty:
        _empty(fig, th)
    return fig


# ------------------------------------------------------------------ losses & regains
def losses(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    ev = _scope(md, team, half, player)
    from fap.matchlab.loader import at_loss, loss_mask
    loss = at_loss(ev[loss_mask(ev)])                  # placed where the ball was lost
    loss = loss[loss["x"].notna()]
    reg = ev[(ev["etype"] == "recovery") | (ev["etype"].isin(["interception", "tackle"]) & ev["labels"].map(lambda l: "won" in l))]
    reg = reg[reg["x"].notna()]
    fig = _pitch(th, f"{team} — Ball losses & regains", _subtitle(md, team, half, player,
                 f"{len(loss)} losses ({int((loss['x'] < md.thirds[0]).sum())} in own third) · {len(reg)} regains · ✕ = where the ball was lost"))
    for x in md.thirds:
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=PITCH_W, line=dict(color=th.line, width=1, dash="dot"), layer="below")
    lt = {"pass": "Pass lost", "dribble": "Dribble lost", "ball_lost": "Ball lost", "error": "Ball lost (error)"}
    danger = loss["x"] < md.thirds[0]
    for sub, name, size, edge in ((loss[~danger], "Ball loss", 11, th.bg), (loss[danger], "Loss in own third", 16, th.accent)):
        if sub.empty:
            continue
        txt = [_point_hover(r, names, lt.get(r["etype"], r["etype"]),
                            f"{r['pass_type'] + ' · ' if r['etype'] == 'pass' and r['pass_type'] else ''}"
                            f"{', '.join(r['labels'])}<br>lost at: {r.get('loss_at') or '–'} · played from "
                            f"x {_fmt(r.get('from_x'), 0)}, y {_fmt(r.get('from_y'), 0)}") for _, r in sub.iterrows()]
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["y"], mode="markers", name=f"{name} ({len(sub)})", text=txt,
                                 hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=size, color=th.bad, symbol="x", line=dict(color=edge, width=1.2))))
    if len(reg):
        txt = [_point_hover(r, names, f"Regain — {r['etype']}", f"{', '.join(r['labels'])} · {r['def_phase'] or ''}")
               for _, r in reg.iterrows()]
        fig.add_trace(go.Scatter(x=reg["x"], y=reg["y"], mode="markers", name=f"Regain ({len(reg)})", text=txt,
                                 hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=10, color=th.good, line=dict(color=th.bg, width=0.8))))
    return fig


# ------------------------------------------------------------------ sequence explorer
def sequence(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, poss_id: int | None = None,
             names: tuple | dict = ()) -> go.Figure:
    from fap.matchlab.charts import _action_text, sequence_events
    th = theme
    nm = dict(names) if not isinstance(names, dict) else names
    poss = md.possessions
    if poss_id is None and not poss.empty:
        mine = poss[poss["team"] == team]
        poss_id = int(mine.sort_values(["shot", "n_passes"], ascending=False).iloc[0]["poss_id"]) if len(mine) else None
    row = poss[poss["poss_id"] == poss_id] if poss_id is not None and not poss.empty else poss.iloc[0:0]
    if row.empty:
        fig = _pitch(th, f"{team} — Sequence", "No sequence selected")
        return fig
    s = row.iloc[0]
    seq = sequence_events(md, int(poss_id))
    fig = _pitch(th, f"{team} — Sequence #{poss_id}",
                 f"{s['start_min']:.0f}' · started: {s['start_type']} · {s['n_passes']} passes · {s['duration']:.0f} s · "
                 f"ended: {s['end_reason']} · hover any step for details")
    own = seq[seq["team"] == team]
    n = 0
    for _, e in own.iterrows():
        et = e["etype"]
        if et == "receive":
            fig.add_trace(go.Scatter(x=[e["x"]], y=[e["y"]], mode="markers", showlegend=False,
                                     marker=dict(size=6, color=th.muted), hoverinfo="skip"))
            continue
        n += 1
        txt = (f"<b>Step {n}</b> · {_mmss(e['minute'])}<br>{_action_text(e, nm)}<br>"
               f"x {e['x']:.1f}, y {e['y']:.1f} m · file pos {_fmt(e['src_x'])}, {_fmt(e['src_y'])} · event {_eid(e)}"
               f"{'<br>' + ', '.join(e['labels_raw']) if isinstance(e.get('labels_raw'), tuple) and e['labels_raw'] else ''}")
        if et in ("pass", "carry") and pd.notna(e["end_x"]):
            col = th.carry if et == "carry" else th.team if e["outcome"] == 1 else th.bad
            fig.add_annotation(x=e["end_x"], y=e["end_y"], ax=e["x"], ay=e["y"], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=2.2 if et == "pass" else 1.6,
                               arrowcolor=col, text="")
        sym, col, size = {"shot": ("star", th.accent, 22), "dribble": ("diamond", th.good if e["outcome"] == 1 else th.bad, 14),
                          "recovery": ("square", th.good, 13), "interception": ("square", th.good, 13),
                          "tackle": ("square", th.good, 13)}.get(et, ("circle", th.text, 20))
        fig.add_trace(go.Scatter(x=[e["x"]], y=[e["y"]], mode="markers+text", showlegend=False, text=[str(n)],
                                 textfont=dict(color=th.bg if sym == "circle" else th.text, size=11), textposition="middle center"
                                 if sym == "circle" else "top center",
                                 hovertext=[txt], hovertemplate="%{hovertext}<extra></extra>",
                                 marker=dict(size=size, color=col, symbol=sym, line=dict(color=th.bg, width=1))))
    opp = seq[seq["team"] != team]
    if len(opp):
        o = opp.iloc[0]
        fig.add_trace(go.Scatter(x=[PITCH_L - o["x"]], y=[PITCH_W - o["y"]], mode="markers", showlegend=False,
                                 hovertext=[f"<b>{o['team']} wins the ball</b><br>{o['etype'].replace('_', ' ')} · "
                                            f"{_mmss(o['minute'])} · event {_eid(o)}"],
                                 hovertemplate="%{hovertext}<extra></extra>",
                                 marker=dict(size=16, color=th.bad, symbol="x", line=dict(color=th.bg, width=1))))
    return fig


# ------------------------------------------------------------------ passing network
def network(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, min_passes: int = 3,
            include_subs: bool = False, names: dict | None = None) -> go.Figure:
    from fap.matchlab.charts import _starters
    th, names = theme, names or {}
    ev = _scope(md, team, half, None)
    p = ev[(ev["etype"] == "pass") & (ev["outcome"] == 1) & ev["receiver"].ne("") & ev["player"].ne("") & ev["player"].ne("0")]
    starters = _starters(md, team)
    keep = (lambda s_: s_.isin(starters)) if not include_subs else (lambda s_: s_.ne(""))
    p = p[keep(p["player"]) & keep(p["receiver"])]
    touch = ev[ev["etype"].isin(["pass", "receive", "carry", "dribble", "shot"]) & ev["player"].isin(set(p["player"]) | set(p["receiver"]))]
    pos = touch.groupby("player")[["x", "y"]].mean()
    made = p.groupby("player").size()
    recv = p.groupby("receiver").size()
    pairs = p.assign(a=np.minimum(p["player"].astype(str), p["receiver"].astype(str)),
                     b=np.maximum(p["player"].astype(str), p["receiver"].astype(str))).groupby(["a", "b"]).size()
    pairs = pairs[pairs >= min_passes]
    fig = _pitch(th, f"{team} — Passing network", _subtitle(md, team, half, None,
                 f"{'starting XI' if not include_subs else 'all players'} · links ≥ {min_passes} completed passes"))
    mx = pairs.max() if len(pairs) else 1
    mid_x, mid_y, mid_t = [], [], []
    for (a, b), n in pairs.sort_values().items():
        if a in pos.index and b in pos.index:
            ab = int(((p["player"] == a) & (p["receiver"] == b)).sum())
            ba = int(((p["player"] == b) & (p["receiver"] == a)).sum())
            fig.add_trace(go.Scatter(x=[pos.at[a, "x"], pos.at[b, "x"]], y=[pos.at[a, "y"], pos.at[b, "y"]], mode="lines",
                                     showlegend=False, hoverinfo="skip", opacity=0.25 + 0.7 * n / mx,
                                     line=dict(color=th.team, width=1 + 9 * n / mx)))
            mid_x.append((pos.at[a, "x"] + pos.at[b, "x"]) / 2)
            mid_y.append((pos.at[a, "y"] + pos.at[b, "y"]) / 2)
            mid_t.append(f"<b>{_who(a, names)} ↔ {_who(b, names)}</b><br>{int(n)} completed passes<br>"
                         f"{_who(a, names)} → {_who(b, names)}: {ab}<br>{_who(b, names)} → {_who(a, names)}: {ba}")
    if mid_x:
        fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", showlegend=False, text=mid_t,
                                 hovertemplate="%{text}<extra></extra>", marker=dict(size=10, color="rgba(0,0,0,0)")))
    if len(pos):
        vol = made.add(recv, fill_value=0)
        vmax = vol.max() or 1
        txt = []
        for pl, r in pos.iterrows():
            partners = p[p["player"] == pl]["receiver"].value_counts().head(3)
            fav = ", ".join(f"{_who(k, names)} ({v})" for k, v in partners.items()) or "–"
            txt.append(f"<b>{_who(pl, names)}</b>{'' if pl in starters else ' (sub)'}<br>"
                       f"Passes made {int(made.get(pl, 0))} · received {int(recv.get(pl, 0))}<br>"
                       f"Average position x {r['x']:.1f}, y {r['y']:.1f} m<br>Most passes to: {fav}")
        fig.add_trace(go.Scatter(x=pos["x"], y=pos["y"], mode="markers+text", text=list(pos.index), showlegend=False,
                                 textfont=dict(color=th.text, size=12), hovertext=txt,
                                 hovertemplate="%{hovertext}<extra></extra>",
                                 marker=dict(size=[22 + 26 * vol.get(pl, 0) / vmax for pl in pos.index], color=th.panel,
                                             line=dict(color=[th.team if pl in starters else th.accent for pl in pos.index], width=2.5))))
    else:
        _empty(fig, th)
    return fig


# ------------------------------------------------------------------ colour helpers
def _hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _mix(c1: str, c2: str, t: float) -> str:
    a, b = _hex_rgb(c1), _hex_rgb(c2)
    t = float(min(max(t, 0.0), 1.0))
    return "rgb({},{},{})".format(*[round(a[i] + (b[i] - a[i]) * t) for i in range(3)])


def _acc_color(th: Theme, acc: float) -> str:
    """0 → bad, 0.5 → accent, 1 → good."""
    if acc is None or np.isnan(acc):
        return th.muted
    return _mix(th.bad, th.accent, acc * 2) if acc < 0.5 else _mix(th.accent, th.good, (acc - 0.5) * 2)


def _heat_color(th: Theme, t: float) -> str:
    stops = list(th.heat)
    t = float(min(max(t, 0.0), 1.0)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return _mix(stops[i], stops[i + 1], t - i)


def _zone_name(x0: float, x1: float, y0: float, y1: float) -> str:
    from fap.matchlab.metrics import channel_of, third_of
    return f"{third_of([(x0 + x1) / 2])[0]} · {channel_of([(y0 + y1) / 2])[0]}"


def _dir_name(a: float) -> str:
    if abs(a) <= 15:
        return "straight forward"
    if abs(a) >= 165:
        return "straight back"
    side = "left" if a > 0 else "right"
    if abs(a) < 90:
        return f"forward-{side} ({abs(a):.0f}°)"
    if abs(a) == 90:
        return f"square {side}"
    return f"back-{side} ({abs(a):.0f}°)"


def _poly(fig: go.Figure, xs, ys, color: str, text: str, opacity: float = 0.9, line=None, name=None,
          showlegend=False) -> None:
    fig.add_trace(go.Scatter(x=list(xs) + [xs[0]], y=list(ys) + [ys[0]], fill="toself", fillcolor=color, mode="lines",
                             line=line or dict(width=0, color=color), opacity=opacity, hoveron="fills",
                             text=text, hoverinfo="text", name=name, showlegend=showlegend))


def _rect(fig, x0, x1, y0, y1, color, text, opacity=0.85, line=None):
    _poly(fig, [x0, x1, x1, x0], [y0, y0, y1, y1], color, text, opacity, line)


def _passes(md: MatchData, team: str, half=None, player=None) -> pd.DataFrame:
    p = _scope(md, team, half, player)
    return p[(p["etype"] == "pass") & p["angle"].notna() & p["end_x"].notna()]


# ------------------------------------------------------------------ pass sonars by zone
def sonars(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, names: dict | None = None,
           bins: tuple = (6, 4)) -> go.Figure:
    th = theme
    p = _passes(md, team, half, player)
    fig = _pitch(th, f"{team} — Pass sonars by zone", _subtitle(md, team, half, player,
                 "wedge = direction · length = share of the zone's passes · colour = accuracy"))
    nx, ny = bins
    wx, wy = PITCH_L / nx, PITCH_W / ny
    rmax = min(wx, wy) * 0.46
    edges = np.linspace(-180, 180, 13)
    for i in range(nx):
        for j in range(ny):
            x0, x1, y0, y1 = i * wx, (i + 1) * wx, j * wy, (j + 1) * wy
            z = p[(p["x"] >= x0) & (p["x"] < x1) & (p["y"] >= y0) & (p["y"] < y1)]
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color=th.grid, width=0.6), layer="below")
            if len(z) < 2:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cnt = np.histogram(z["angle"], bins=edges)[0]
            mx = cnt.max()
            for k in range(12):
                if not cnt[k]:
                    continue
                sel = z[(z["angle"] >= edges[k]) & (z["angle"] < edges[k + 1])]
                acc = float((sel["outcome"] == 1).mean())
                r = rmax * (0.25 + 0.75 * cnt[k] / mx)
                th_ = np.radians(np.linspace(edges[k] + 1.5, edges[k + 1] - 1.5, 8))
                xs = [cx] + list(cx + r * np.cos(th_))
                ys = [cy] + list(cy + r * np.sin(th_))
                text = (f"<b>{_zone_name(x0, x1, y0, y1)}</b><br>Direction: {_dir_name((edges[k] + edges[k + 1]) / 2)}<br>"
                        f"{cnt[k]} of {len(z)} passes from this zone ({cnt[k] / len(z) * 100:.0f}%)<br>"
                        f"Completed {int((sel['outcome'] == 1).sum())} · accuracy {acc * 100:.0f}% · avg {sel['length'].mean():.1f} m")
                _poly(fig, xs, ys, _acc_color(th, acc), text, opacity=0.92)
            fig.add_annotation(x=cx, y=y0 + 1.2, text=str(len(z)), showarrow=False, font=dict(size=10, color=th.muted))
    if p.empty:
        _empty(fig, th)
    return fig


# ------------------------------------------------------------------ pass flow
def flow(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, names: dict | None = None,
         bins: tuple = (6, 5)) -> go.Figure:
    th, names = theme, names or {}
    p = _passes(md, team, half, player)
    fig = _pitch(th, f"{team} — Pass flow", _subtitle(md, team, half, player,
                 "shade = passes started in the zone · arrow = average direction & distance"))
    nx, ny = bins
    wx, wy = PITCH_L / nx, PITCH_W / ny
    zones = []
    for i in range(nx):
        for j in range(ny):
            x0, x1, y0, y1 = i * wx, (i + 1) * wx, j * wy, (j + 1) * wy
            zones.append((x0, x1, y0, y1, p[(p["x"] >= x0) & (p["x"] < x1) & (p["y"] >= y0) & (p["y"] < y1)]))
    mx = max([len(z) for *_, z in zones] + [1])
    for x0, x1, y0, y1, z in zones:
        if not len(z):
            _rect(fig, x0, x1, y0, y1, th.pitch, f"<b>{_zone_name(x0, x1, y0, y1)}</b><br>no passes", 0.2)
            continue
        vx, vy = (z["end_x"] - z["x"]).mean(), (z["end_y"] - z["y"]).mean()
        acc = (z["outcome"] == 1).mean()
        top = z[z["player"].ne("") & z["player"].ne("0")]["player"].value_counts().head(2)
        who = ", ".join(f"{_who(k, names)} ({v})" for k, v in top.items()) or "–"
        text = (f"<b>{_zone_name(x0, x1, y0, y1)}</b><br>{len(z)} passes started here · accuracy {acc * 100:.0f}%<br>"
                f"Average pass: {_dir_name(float(np.degrees(np.arctan2(vy, vx))))} · {np.hypot(vx, vy):.1f} m net<br>"
                f"Forward {int((z['direction'] == 'Forward').sum())} · lateral {int(z['direction'].isin(['Left', 'Right']).sum())}"
                f" · back {int((z['direction'] == 'Backward').sum())}<br>Most from: {who}")
        _rect(fig, x0, x1, y0, y1, _heat_color(th, len(z) / mx), text, 0.9, line=dict(color=th.bg, width=1.5))
        n = np.hypot(vx, vy)
        if len(z) >= 2 and n:
            sc = min(1.0, (min(wx, wy) * 0.42) / n)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_annotation(x=cx + vx * sc, y=cy + vy * sc, ax=cx - vx * sc * 0.4, ay=cy - vy * sc * 0.4, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2.2, arrowsize=1,
                               arrowcolor=th.text, opacity=0.35 + 0.6 * acc, text="")
    return fig


# ------------------------------------------------------------------ heat map
def heat(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, style: str = "smooth",
         kind: str = "activity", names: dict | None = None) -> go.Figure:
    from fap.matchlab import heat as HT
    from fap.matchlab.metrics import CHANNEL_EDGES, CHANNELS, THIRD_NAMES
    th, names = theme, names or {}
    label, desc = HT.KINDS[kind]
    ev = HT.points(md, team, kind, half, player)
    fig = _pitch(th, f"{team} — {'Touch' if player and kind == 'activity' else label} heat map",
                 _subtitle(md, team, half, player, f"{len(ev)} events · {desc}"))
    if ev.empty:
        _empty(fig, th)
        return fig
    if style == "zones":
        t1, t2 = md.thirds
        for (a, b), tn in zip(((0, t1), (t1, t2), (t2, PITCH_L)), THIRD_NAMES):
            for k in range(5):
                y0, y1 = CHANNEL_EDGES[k], CHANNEL_EDGES[k + 1]
                m = (ev["x"] >= a) & (ev["x"] < b) & (ev["y"] >= y0) & (ev["y"] < y1)
                share = m.mean()
                top = ev[m]["etype"].value_counts().head(3)
                who = ev[m]["player"][~ev[m]["player"].isin(["", "0"])].value_counts().head(3)
                _rect(fig, a, b, y0, y1, _heat_color(th, share * 6), (
                    f"<b>{tn} · {CHANNELS[k]}</b><br>{int(m.sum())} events · {share * 100:.1f}% of all<br>"
                    + ", ".join(f"{k2} {v}" for k2, v in top.items())
                    + ("<br>most: " + ", ".join(f"{_who(p_, names)} ({v})" for p_, v in who.items()) if len(who) else "")),
                    0.9, line=dict(color=th.bg, width=1.5))
                if m.sum():
                    fig.add_annotation(x=(a + b) / 2, y=(y0 + y1) / 2, text=f"{share * 100:.0f}%", showarrow=False,
                                       font=dict(size=12, color=th.text))
        return fig
    from scipy.ndimage import gaussian_filter
    nx, ny = 42, 28
    H, xe, ye = np.histogram2d(ev["x"], ev["y"], bins=[nx, ny], range=[[0, PITCH_L], [0, PITCH_W]])
    raw = H.T
    smooth = gaussian_filter(raw.astype(float), 1.6)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    text = [[f"x {xe[i]:.0f}–{xe[i + 1]:.0f} m · y {ye[j]:.0f}–{ye[j + 1]:.0f} m<br>{int(raw[j, i])} events in this cell"
             f"<br>density {smooth[j, i] / smooth.max() * 100:.0f}% of the peak" for i in range(nx)] for j in range(ny)]
    fig.add_trace(go.Heatmap(x=xc, y=yc, z=smooth, colorscale=[[i / (len(th.heat) - 1), c] for i, c in enumerate(th.heat)],
                             showscale=False, zsmooth="best", text=text, hoverinfo="text", opacity=0.95))
    return fig


def heat_compare(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, style: str = "smooth",
                 kind: str = "activity", names: dict | None = None) -> list:
    """The same heat map for both teams, side by side (each attacking left → right)."""
    out = []
    for tm in (team, md.opponent(team)):
        f = heat(md, tm, half=half, player=None, theme=theme, style=style, kind=kind, names=names)
        out.append(f)
    return out


def switches(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
             names: dict | None = None) -> go.Figure:
    """Switches of play: >= 30 m across, from one side of the pitch to the other."""
    from fap.matchlab.metrics import CHANNEL_EDGES, SWITCH_MIN_M
    th, names = theme, names or {}
    ev = _scope(md, team, half, player)
    sw = ev[(ev["etype"] == "pass") & ev["switch"] & ev["end_x"].notna()]
    ok, ko = sw[sw["outcome"] == 1], sw[sw["outcome"] != 1]
    r2l = int((sw["end_y"] > sw["y"]).sum())
    fig = _pitch(th, f"{team} — Switches of play", _subtitle(md, team, half, player,
                 f"{len(ok)}/{len(sw)} completed · {r2l} right → left, {len(sw) - r2l} left → right · "
                 f"≥ {SWITCH_MIN_M:.0f} m across, side to side"))
    for yv in (CHANNEL_EDGES[2], CHANNEL_EDGES[3]):
        fig.add_shape(type="line", x0=0, x1=PITCH_L, y0=yv, y1=yv, line=dict(color=th.line, width=0.8, dash="dot"),
                      layer="below")
    _add_moves(fig, ko, th.bad, "Unsuccessful switch", names, th, symbol="x", dash="dash", width=1.6)
    _add_moves(fig, ok, th.team, "Completed switch", names, th, width=2.6)
    if sw.empty:
        _empty(fig, th, "No switches of play")
    return fig


# ------------------------------------------------------------------ phases of play
def _band(fig, th, a, b, color, text):
    _rect(fig, a, b, 0, PITCH_W, color, text, opacity=0.12)


def phases_in(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    from fap.matchlab import phases as PH
    th, names = theme, names or {}
    tab = PH.in_possession_table(md, team, half)
    ev = _scope(md, team, half, None)
    fig = _pitch(th, f"{team} — Phases of play · in possession",
                 _subtitle(md, team, half, extra="hover a band for its numbers · arrows = progressive passes · ✕ = losses"))
    t1, t2 = md.thirds
    cols = [th.team, th.accent, th.good]
    for (a, b), (_, r), col in zip(((0, t1), (t1, t2), (t2, PITCH_L)), tab.iterrows(), cols):
        _band(fig, th, a, b, col, (
            f"<b>{r['Phase']}</b> — {r['Share %']:.0f}% of actions ({int(r['Actions'])})<br>"
            f"Passes {int(r['Passes'])} · accuracy {_fmt(r['Accuracy %'], 0, '%')} · forward {_fmt(r['Forward %'], 0, '%')}<br>"
            f"Progressive {int(r['Progressive'])} · line breaks {int(r['Line breaks'])} · avg pass {_fmt(r['Avg pass (m)'])} m<br>"
            f"Losses {int(r['Losses'])} ({_fmt(r['Loss %'], 0, '%')}) · under pressure {int(r['Under pressure'])}<br>"
            f"Key players: {r['Key players'] or '–'}"))
        fig.add_annotation(x=(a + b) / 2, y=PITCH_W + 1.5, text=f"<b>{r['Phase'].upper()}</b> · {r['Share %']:.0f}%",
                           showarrow=False, yanchor="bottom", font=dict(size=12, color=col))
    prog = ev[(ev["etype"] == "pass") & ev["progressive"] & (ev["outcome"] == 1) & ev["end_x"].notna()]
    _add_moves(fig, prog, th.text, "Progressive pass", names, th, width=1.4)
    from fap.matchlab.loader import at_loss, loss_mask
    loss = at_loss(ev[loss_mask(ev)])                  # placed where the ball was lost
    loss = loss[loss["x"].notna()]
    if len(loss):
        fig.add_trace(go.Scatter(x=loss["x"], y=loss["y"], mode="markers", name=f"Ball loss ({len(loss)})",
                                 text=[_point_hover(r, names, f"Loss — {r['etype']}",
                                                    f"phase: {r['phase'] or '–'}<br>lost at: {r.get('loss_at') or '–'} · "
                                                    f"played from x {_fmt(r.get('from_x'), 0)}, y {_fmt(r.get('from_y'), 0)}")
                                       for _, r in loss.iterrows()], hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=9, color=th.bad, symbol="x")))
    fig.update_layout(yaxis_range=[-6, PITCH_W + 6])
    return fig


def phases_out(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, names: dict | None = None) -> go.Figure:
    from fap.matchlab import phases as PH
    th, names = theme, names or {}
    tab = PH.out_of_possession_table(md, team, half).set_index("Phase")
    ev = _scope(md, team, half, None)
    d = ev[ev["def_phase"] != ""]
    k = PH.defending_kpis(md, team, half)
    fig = _pitch(th, f"{team} — Phases of play · out of possession", _subtitle(md, team, half, extra=(
        f"PPDA {_fmt(k['PPDA'])} · def. height {_fmt(k['Def. action height (m)'])} m · hover a band for its numbers")))
    t1, t2 = md.thirds
    for (a, b), ph, col in zip(((0, t1), (t1, t2), (t2, PITCH_L)), ("Low block", "Mid block", "High press"),
                               (th.bad, th.accent, th.good)):
        if ph not in tab.index:
            continue
        r = tab.loc[ph]
        _band(fig, th, a, b, col, (
            f"<b>{ph}</b> — {r['Share %']:.0f}% of defensive actions ({int(r['Actions'])})<br>"
            f"Regains {int(r['Regains'])} · interceptions {int(r['Interceptions'])} · tackles {int(r['Tackles'])}<br>"
            f"Recoveries {int(r['Recoveries'])} · duels {int(r['Duels'])} · fouls {int(r['Fouls'])} · presses {int(r['Presses'])}<br>"
            f"Success {_fmt(r['Success %'], 0, '%')} · counter-press actions {int(r['Counter-press'])}"))
        fig.add_annotation(x=(a + b) / 2, y=PITCH_W + 1.5, text=f"<b>{ph.upper()}</b> · {r['Share %']:.0f}%",
                           showarrow=False, yanchor="bottom", font=dict(size=12, color=col))
    for cp, name, col, sym in ((False, "Defensive action", th.team, "circle"), (True, "Counter-press action", th.accent, "diamond")):
        g = d[d["counter_press"].astype(bool) == cp]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers", name=f"{name} ({len(g)})",
                                 text=[_point_hover(r, names, r["etype"].title(), f"{r['def_phase']} · {', '.join(r['labels'])}")
                                       for _, r in g.iterrows()], hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=9 if not cp else 11, color=col, symbol=sym, line=dict(color=th.bg, width=0.6))))
    if len(d):
        h = d["x"].mean()
        fig.add_shape(type="line", x0=h, x1=h, y0=0, y1=PITCH_W, line=dict(color=th.text, width=1.5, dash="dash"))
    fig.update_layout(yaxis_range=[-6, PITCH_W + 6])
    return fig


# ------------------------------------------------------------------ build-up sequences (tiled pitches)
def buildup(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, min_passes: int = 3,
            max_panels: int = 12, names: dict | None = None) -> go.Figure:
    from fap.matchlab.charts import _action_text, sequence_events
    th, names = theme, names or {}
    poss = md.possessions
    sel = poss[(poss["team"] == team) & (poss["start_x"] < md.thirds[0]) & (poss["n_passes"] >= min_passes)] if not poss.empty else poss
    if half is not None and not sel.empty:
        sel = sel[sel["half"] == half]
    sel = sel.sort_values(["shot", "reached_f3", "n_passes"], ascending=False).head(max_panels) if not sel.empty else sel
    cols = 4
    rows = max(1, int(np.ceil(len(sel) / cols)))
    gx, gy = PITCH_L + 10, PITCH_W + 16
    fig = _pitch(th, f"{team} — Build-up sequences", _subtitle(md, team, half, extra=(
        f"started in own third, ≥{min_passes} passes · best {len(sel)} · hover any action")), height=230 * rows + 120)
    # one pitch per sequence, tiled with x / y offsets
    allshapes = []
    for k in range(len(sel)):
        r, c = divmod(k, cols)
        for s_ in _pitch_shapes(th, c * gx):
            s_["y0"] -= r * gy
            s_["y1"] -= r * gy
            allshapes.append(s_)
    fig.update_layout(shapes=allshapes, xaxis_range=[-3, cols * gx], yaxis_range=[-(rows - 1) * gy - 4, PITCH_W + 12],
                      annotations=[], showlegend=False)
    for k, (_, s) in enumerate(sel.iterrows()):
        r, c = divmod(k, cols)
        dx, dy = c * gx, -r * gy
        col = th.good if s["shot"] else th.accent if s["reached_f3"] else th.bad
        fig.add_annotation(x=dx, y=PITCH_W + 1.5 + dy, xanchor="left", yanchor="bottom", showarrow=False,
                           text=f"<b>#{s['poss_id']}</b> · {s['start_min']:.0f}' · {s['n_passes']} passes · {s['duration']:.0f}s · {s['end_reason']}",
                           font=dict(size=10.5, color=col))
        seq = sequence_events(md, int(s["poss_id"]))
        own = seq[seq["team"] == team]
        n = 0
        xs, ys, tx = [], [], []
        lx, ly = [], []
        for _, e in own.iterrows():
            if e["etype"] == "receive":
                continue
            n += 1
            if e["etype"] in ("pass", "carry") and pd.notna(e["end_x"]):
                lx += [e["x"] + dx, e["end_x"] + dx, None]
                ly += [e["y"] + dy, e["end_y"] + dy, None]
            xs.append(e["x"] + dx)
            ys.append(e["y"] + dy)
            tx.append(f"<b>Sequence #{s['poss_id']} · step {n}</b> · {_mmss(e['minute'])}<br>{_action_text(e, names)}<br>"
                      f"x {e['x']:.1f}, y {e['y']:.1f} m · event {_eid(e)}")
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=col, width=1.6), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", text=tx, hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=6, color=col, line=dict(color=th.bg, width=0.5))))
        opp = seq[seq["team"] != team]
        if len(opp):
            o = opp.iloc[0]
            fig.add_trace(go.Scatter(x=[PITCH_L - o["x"] + dx], y=[PITCH_W - o["y"] + dy], mode="markers",
                                     hovertext=[f"{o['team']} wins the ball · {o['etype']} · {_mmss(o['minute'])}"],
                                     hovertemplate="%{hovertext}<extra></extra>", marker=dict(size=9, color=th.bad, symbol="x")))
    if sel.empty:
        _empty(fig, th, "No build-up sequences for this selection")
    return fig


# ------------------------------------------------------------------ player dashboard figures
def player_rose(md: MatchData, team: str, player: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    th = theme
    p = _passes(md, team, half, player)
    edges = np.linspace(-180, 180, 17)
    rows = []
    for k in range(16):
        sel = p[(p["angle"] >= edges[k]) & (p["angle"] < edges[k + 1])]
        c = (edges[k] + edges[k + 1]) / 2
        acc = float((sel["outcome"] == 1).mean()) if len(sel) else np.nan
        rows.append((c, len(sel), int((sel["outcome"] == 1).sum()), acc, sel["length"].mean() if len(sel) else np.nan))
    fig = go.Figure(go.Barpolar(
        r=[r[1] for r in rows], theta=[r[0] for r in rows], width=[21] * 16,
        marker=dict(color=[_acc_color(th, r[3]) for r in rows], line=dict(color=th.bg, width=1)),
        text=[f"<b>{_dir_name(r[0])}</b><br>{r[1]} passes · {r[2]} completed"
              + (f" · {r[3] * 100:.0f}%<br>avg length {r[4]:.1f} m" if r[1] else "") for r in rows],
        hoverinfo="text"))
    fig.update_layout(
        title=dict(text=f"<b>Pass directions</b><br><span style='font-size:11px;color:{th.muted}'>bar = passes · colour = accuracy · forward = up</span>",
                   x=0.02, font=dict(size=15, color=th.text)),
        paper_bgcolor=th.bg, font=dict(color=th.text), height=430, margin=dict(l=55, r=55, t=70, b=25),
        polar=dict(bgcolor=th.panel, angularaxis=dict(rotation=90, direction="counterclockwise", tickvals=[0, 90, 180, 270],
                                                       ticktext=["FORWARD", "LEFT", "BACK", "RIGHT"], color=th.muted,
                                                       gridcolor=th.grid),
                   radialaxis=dict(showticklabels=False, gridcolor=th.grid)),
        hoverlabel=dict(bgcolor=th.panel, bordercolor=th.grid, font=dict(color=th.text), align="left"))
    return fig


def player_receptions(md: MatchData, team: str, player: str, *, half=None, theme: Theme = DARK,
                      names: dict | None = None) -> go.Figure:
    th, names = theme, names or {}
    ev = md.events if half is None else md.events[md.events["half"] == half]
    rec = ev[(ev["team"] == team) & (ev["etype"] == "pass") & (ev["receiver"] == str(player)) & (ev["outcome"] == 1)]
    fig = _pitch(th, f"{_who(player, names)} · receptions", _subtitle(md, team, half, player,
                 f"{len(rec)} receptions · {int(rec['progressive'].sum())} progressive"), height=470)
    for prog, name, col in ((False, "Reception", th.team), (True, "Progressive reception", th.accent)):
        g = rec[rec["progressive"].astype(bool) == prog]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g["end_x"], y=g["end_y"], mode="markers", name=f"{name} ({len(g)})",
                                 text=[f"<b>From {_who(r['player'], names)}</b> · {_mmss(r['minute'])} · event {_eid(r)}<br>"
                                       f"{r['pass_type'] or 'Pass'} · {_fmt(r['length'])} m · {str(r['direction']).lower()}<br>"
                                       f"Received at x {r['end_x']:.1f}, y {r['end_y']:.1f} m ({r['end_third']}, {r['end_channel']})"
                                       for _, r in g.iterrows()], hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=11 if prog else 9, color=col, symbol="diamond" if prog else "circle",
                                             line=dict(color=th.bg, width=0.8))))
    if rec.empty:
        _empty(fig, th, "No receptions")
    return fig


def player_defence(md: MatchData, team: str, player: str, *, half=None, theme: Theme = DARK,
                   names: dict | None = None) -> go.Figure:
    from fap.matchlab import counterpress as CP
    from fap.matchlab.player import _losses
    th, names = theme, names or {}
    me = _scope(md, team, half, player)
    d = me[me["etype"].isin(list(_DEF)) & me["x"].notna()]
    from fap.matchlab.loader import at_loss
    lo = at_loss(_losses(me))
    lo = lo[lo["x"].notna()]
    cp = CP.analyse(md, team, md.cp_window, half)
    cp_by_loss = cp.set_index("loss_event_id") if not cp.empty else pd.DataFrame()
    fig = _pitch(th, f"{_who(player, names)} · defending & losses", _subtitle(md, team, half, player,
                 f"{len(d)} defensive actions · {len(lo)} ball losses"), height=470)
    palette = [th.team, th.good, th.accent, th.carry, th.text, th.muted, th.opp, th.good]
    for (k, (nm, sym)), col in zip(_DEF.items(), palette):
        g = d[d["etype"] == k]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers", name=f"{nm} ({len(g)})",
                                 text=[_point_hover(r, names, nm, f"{', '.join(r['labels'])} · {r['def_phase'] or ''}"
                                                    f"{' · counter-press' if r['counter_press'] else ''}") for _, r in g.iterrows()],
                                 hovertemplate="%{text}<extra></extra>",
                                 marker=dict(size=10, color=col, symbol=sym, line=dict(color=th.bg, width=0.7))))
    if len(lo):
        txt = []
        for _, r in lo.iterrows():
            extra = ""
            eid = r.get("event_id")
            if not cp_by_loss.empty and pd.notna(eid) and eid in cp_by_loss.index:
                c = cp_by_loss.loc[eid]
                c = c.iloc[0] if isinstance(c, pd.DataFrame) else c
                extra = (f"Team reaction: <b>{c['category']}</b>"
                         + (f" · press after {_fmt(c['seconds_to_press'])} s ({c['press_timing']})" if c["pressed"] else ""))
            txt.append(_point_hover(r, names, f"Ball loss — {r['etype']}", extra))
        fig.add_trace(go.Scatter(x=lo["x"], y=lo["y"], mode="markers", name=f"Ball loss ({len(lo)})", text=txt,
                                 hovertemplate="%{text}<extra></extra>", marker=dict(size=11, color=th.bad, symbol="x")))
    if d.empty and lo.empty:
        _empty(fig, th)
    return fig


def player_breakdown(md: MatchData, team: str, player: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    from plotly.subplots import make_subplots
    from fap.matchlab.player import profile
    th = theme
    pr = profile(md, team, player, half)
    parts = [("Pass type", pr["by_type"]), ("Direction", pr["by_direction"]), ("Length", pr["by_length"])]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t for t, _ in parts], horizontal_spacing=0.17)
    for i, (t, g) in enumerate(parts, 1):
        if g.empty:
            continue
        labels = list(g.index)[::-1]
        att = g["attempted"].tolist()[::-1]
        cmp_ = g["completed"].tolist()[::-1]
        acc = g["accuracy"].tolist()[::-1]
        hover = [f"<b>{l}</b><br>{c}/{a} completed · {ac:.0f}%" for l, a, c, ac in zip(labels, att, cmp_, acc)]
        fig.add_trace(go.Bar(y=labels, x=att, orientation="h", marker_color=_mix(th.team, th.bg, 0.65), name="Attempted",
                             showlegend=i == 1, text=hover, hoverinfo="text", textposition="none"), 1, i)
        fig.add_trace(go.Bar(y=labels, x=cmp_, orientation="h", marker_color=th.team, name="Completed",
                             showlegend=i == 1, text=hover, hoverinfo="text", textposition="none"), 1, i)
    fig.update_layout(barmode="overlay", height=330, paper_bgcolor=th.bg, plot_bgcolor=th.bg, font=dict(color=th.text),
                      margin=dict(l=105, r=10, t=60, b=10),
                      title=dict(text="<b>Pass details</b>", x=0.01, font=dict(size=15, color=th.text)),
                      legend=dict(orientation="h", y=1.18, x=1, xanchor="right", font=dict(color=th.muted)),
                      hoverlabel=dict(bgcolor=th.panel, bordercolor=th.grid, font=dict(color=th.text), align="left"))
    fig.update_xaxes(gridcolor=th.grid, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True, ticklabelstandoff=4)
    for a in fig.layout.annotations:
        a.font = dict(color=th.muted, size=12)
    return fig


def player_compare(md: MatchData, team: str, player: str, *, half=None, theme: Theme = DARK) -> go.Figure:
    from fap.matchlab.player import comparison
    th = theme
    c = comparison(md, team, str(player), half)
    fig = go.Figure()
    if c.empty:
        fig.update_layout(paper_bgcolor=th.bg, height=200)
        return fig
    c = c.iloc[::-1].rename(columns={"Team average": "avg", "Team best": "best", "Higher is better": "higher"})

    def pct(v, best, higher):
        if higher:
            return v / best * 100 if best else 0.0
        return best / v * 100 if v else 100.0

    me = [pct(r.Player, r.best, r.higher) for r in c.itertuples()]
    avg = [pct(r.avg, r.best, r.higher) for r in c.itertuples()]
    hover = [f"<b>{r.Metric}</b><br>Player {r.Player:.2f}".rstrip("0").rstrip(".") +
             f" · team average {r.avg:.2f} · team best {r.best:.2f}<br>Rank <b>{r.Rank}</b> of {r.Of}"
             f"{'' if r.higher else ' (fewer is better)'}" for r in c.itertuples()]
    cols = [th.good if r.Rank <= max(1, r.Of // 4) else th.accent if r.Rank <= r.Of // 2 else th.team for r in c.itertuples()]
    fig.add_trace(go.Bar(y=c["Metric"], x=me, orientation="h", marker_color=cols, name="Player (% of team best)",
                         text=[f"#{r.Rank}/{r.Of}" for r in c.itertuples()], textposition="outside",
                         textfont=dict(color=th.muted, size=11), hovertext=hover, hoverinfo="text"))
    fig.add_trace(go.Scatter(y=c["Metric"], x=avg, mode="markers", name="Team average",
                             marker=dict(symbol="line-ns", size=18, line=dict(width=3, color=th.text)),
                             hovertext=hover, hoverinfo="text"))
    fig.update_layout(
        title=dict(text=f"<b>Compared with the team</b><br><span style='font-size:11px;color:{th.muted}'>"
                        f"bar = player as % of the team's best · line = team average · label = rank</span>",
                   x=0.01, font=dict(size=15, color=th.text)),
        height=40 + 30 * len(c) + 80, paper_bgcolor=th.bg, plot_bgcolor=th.bg, font=dict(color=th.text),
        margin=dict(l=10, r=40, t=80, b=10), xaxis=dict(range=[0, 118], gridcolor=th.grid, ticksuffix="%"),
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", font=dict(color=th.muted)),
        hoverlabel=dict(bgcolor=th.panel, bordercolor=th.grid, font=dict(color=th.text), align="left"))
    return fig


# ------------------------------------------------------------------ leaderboards & player comparison
def leaderboard(stats: pd.DataFrame, title: str, col: str, num: str | None = None, den: str | None = None, *,
                theme: Theme = DARK, names: dict | None = None, top: int = 8) -> go.Figure:
    """Top players for one metric; bar colour = success rate when the action has a tagged outcome."""
    th, names = theme, names or {}
    d = stats[stats[col] > 0].sort_values([col] + ([num] if num else []), ascending=False).head(top)
    d = d.iloc[::-1]
    labels = [_who(p, names) for p in d["player"]]
    if num and den:
        rate = np.where(d[den] > 0, d[num] / d[den].where(d[den] > 0, 1), np.nan)
        cols = [_acc_color(th, r) if not np.isnan(r) else th.team for r in rate]
        txt = [f"{v:g}  ·  {int(a)}/{int(b)} ({r * 100:.0f}%)" if b > 0 else f"{v:g}  ·  no outcome"
               for v, a, b, r in zip(d[col], d[num], d[den], rate)]
        hover = [f"<b>{lb}</b><br>{title}: {v:g}<br>Successful {int(a)} of {int(b)} with an outcome"
                 + (f" · <b>{r * 100:.0f}%</b>" if b > 0 else "") for lb, v, a, b, r in zip(labels, d[col], d[num], d[den], rate)]
    else:
        cols = [th.team] * len(d)
        txt = [f"{v:g}" for v in d[col]]
        hover = [f"<b>{lb}</b><br>{title}: {v:g}" for lb, v in zip(labels, d[col])]
    fig = go.Figure(go.Bar(y=labels, x=d[col], orientation="h", marker_color=cols, text=txt, textposition="outside",
                           textfont=dict(color=th.muted, size=11), cliponaxis=False, hovertext=hover, hoverinfo="text"))
    sub = "colour = success rate" if num else ""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" + (f"<br><span style='font-size:11px;color:{th.muted}'>{sub}</span>" if sub else ""),
                   x=0.02, font=dict(size=14, color=th.text)),
        height=110 + 30 * max(len(d), 1), paper_bgcolor=th.bg, plot_bgcolor=th.bg, font=dict(color=th.text, size=11),
        margin=dict(l=10, r=70, t=60 if sub else 45, b=10),
        xaxis=dict(visible=False, range=[0, (d[col].max() if len(d) else 1) * 1.45]),
        yaxis=dict(automargin=True), bargap=0.35,
        hoverlabel=dict(bgcolor=th.panel, bordercolor=th.grid, font=dict(color=th.text), align="left"))
    if d.empty:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="none in this match", showarrow=False,
                           font=dict(color=th.muted))
    return fig


def _num(v, metric: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if metric == "xG":
        return f"{v:.2f}"
    if "%" in metric:
        return f"{v:.0f}%"
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"


def compare_radar(md: MatchData, team: str, players: list, *, half=None, theme: Theme = DARK,
                  names: dict | None = None) -> go.Figure:
    """2–4 players on one radar: each axis = % of the team's best (ball losses inverted)."""
    from fap.matchlab.player import compare_players
    th, names = theme, names or {}
    c = compare_players(md, team, players, half)
    fig = go.Figure()
    if c.empty or not players:
        return fig
    palette = [th.team, th.accent, th.good, th.carry]
    cats = list(c["Metric"])
    for p, col in zip(players, palette):
        p = str(p)
        vals, hov = [], []
        for _, r in c.iterrows():
            v, best, hi = r[p], r["Team best"], r["Higher is better"]
            if np.isnan(v):
                pct = 0.0
            elif hi:
                pct = v / best * 100 if best else 0.0
            else:                                            # fewer is better: worst → 0, best → 100
                worst = r["Team worst"]
                pct = 100.0 if worst == best else (worst - v) / (worst - best) * 100
            vals.append(min(pct, 100.0))
            m = r["Metric"]
            others = " · ".join(f"{_who(q, names)} {_num(r[str(q)], m)}" for q in players if str(q) != p and not np.isnan(r[str(q)]))
            hov.append(f"<b>{_who(p, names)}</b> — {m}: <b>{_num(v, m)}</b><br>team best {_num(best, m)}"
                       f"{'' if hi else ' (fewer is better)'}<br>{others}")
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=_who(p, names), fill="toself",
                                      line=dict(color=col, width=2), opacity=0.55, hovertext=hov + hov[:1], hoverinfo="text"))
    fig.update_layout(
        title=dict(text=f"<b>Player comparison</b><br><span style='font-size:11px;color:{th.muted}'>"
                        f"each axis = % of the team's best · ball losses: team's most = 0, fewest = 100</span>",
                   x=0.01, font=dict(size=15, color=th.text)),
        height=560, paper_bgcolor=th.bg, font=dict(color=th.text, size=11), margin=dict(l=70, r=70, t=80, b=40),
        polar=dict(bgcolor=th.panel, radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor=th.grid),
                   angularaxis=dict(gridcolor=th.grid, color=th.text)),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center", font=dict(color=th.text)),
        hoverlabel=dict(bgcolor=th.panel, bordercolor=th.grid, font=dict(color=th.text), align="left"))
    return fig


INTERACTIVE = {"counterpress": counterpress, "pass_map": pass_map, "pass_map_halves": pass_map_halves,
               "progression": progression, "entries": entries, "crosses": crosses, "shots": shots,
               "defensive": defensive, "losses": losses, "sequence": sequence, "pass_network": network,
               "pass_sonars": sonars, "pass_flow": flow, "heatmap": heat, "heat_compare": heat_compare,
               "switches": switches, "phases_in": phases_in,
               "phases_out": phases_out, "buildup": buildup}


def build(key: str, md: MatchData, team: str, **kw):
    """Interactive figure for a chart key — a Figure, a list of Figures (side by side), or None when
    the chart is static-only."""
    fn = INTERACTIVE.get(key)
    if fn is None:
        return None
    import inspect
    allowed = set(inspect.signature(fn).parameters)
    out = fn(md, team, **{k: v for k, v in kw.items() if k in allowed})
    for f in (out if isinstance(out, list) else [out]):
        f.update_traces(hoverlabel=dict(align="left"))
    return out
