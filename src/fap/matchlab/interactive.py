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
        "duel": ("Duel", "square"), "foul": ("Foul", "x"), "block": ("Block", "cross"), "clearance": ("Clearance", "triangle-down"),
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
    loss = ev[((ev["etype"] == "pass") & (ev["outcome"] == 0)) | ((ev["etype"] == "dribble") & (ev["outcome"] == 0))
              | (ev["etype"] == "ball_lost") | ((ev["etype"] == "error") & ev["labels"].map(lambda l: "ball lost" in l))]
    loss = loss[loss["x"].notna()]
    reg = ev[(ev["etype"] == "recovery") | (ev["etype"].isin(["interception", "tackle"]) & ev["labels"].map(lambda l: "won" in l))]
    reg = reg[reg["x"].notna()]
    fig = _pitch(th, f"{team} — Ball losses & regains", _subtitle(md, team, half, player,
                 f"{len(loss)} losses ({int((loss['x'] < md.thirds[0]).sum())} in own third) · {len(reg)} regains"))
    for x in md.thirds:
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=PITCH_W, line=dict(color=th.line, width=1, dash="dot"), layer="below")
    lt = {"pass": "Pass lost", "dribble": "Dribble lost", "ball_lost": "Ball lost", "error": "Ball lost (error)"}
    danger = loss["x"] < md.thirds[0]
    for sub, name, size, edge in ((loss[~danger], "Ball loss", 11, th.bg), (loss[danger], "Loss in own third", 16, th.accent)):
        if sub.empty:
            continue
        txt = [_point_hover(r, names, lt.get(r["etype"], r["etype"]),
                            f"{r['pass_type'] + ' · ' if r['etype'] == 'pass' and r['pass_type'] else ''}"
                            f"{', '.join(r['labels'])}") for _, r in sub.iterrows()]
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


INTERACTIVE = {"counterpress": counterpress, "pass_map": pass_map, "pass_map_halves": pass_map_halves,
               "progression": progression, "entries": entries, "crosses": crosses, "shots": shots,
               "defensive": defensive, "losses": losses, "sequence": sequence, "pass_network": network}


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
