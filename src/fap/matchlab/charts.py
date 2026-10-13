"""Publication-grade match charts (matplotlib + mplsoccer) on a 105 x 68 m pitch.

Every public chart has the signature ``chart(md, team, *, half=None, player=None, theme=DARK, **opts)``
and returns a ``matplotlib.figure.Figure``: a titled, sub-titled, footed card with the attack
direction marked. ``CHARTS`` is the catalogue the UI renders from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge  # noqa: E402
from mplsoccer import Pitch, VerticalPitch  # noqa: E402

from fap.matchlab import metrics as M  # noqa: E402
from fap.matchlab import phases as P  # noqa: E402
from fap.matchlab.loader import PITCH_L, PITCH_W, MatchData  # noqa: E402

def _available_fonts(prefs: list[str]) -> list[str]:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in prefs if f in have] or ["DejaVu Sans"]


FONT = _available_fonts(["Segoe UI", "Inter", "Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"])


# ======================================================================== themes
@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    pitch: str
    line: str
    text: str
    muted: str
    grid: str
    team: str
    opp: str
    good: str
    bad: str
    accent: str
    carry: str
    heat: tuple

    def cmap(self, name: str = "heat") -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(f"{self.name}_{name}", list(self.heat))


DARK = Theme("dark", bg="#0b0f17", panel="#131a26", pitch="#0b0f17", line="#33415a", text="#eef2f8",
             muted="#8a96ab", grid="#1e2736", team="#3ab0ff", opp="#ff5d73", good="#2ed8a3", bad="#ff5d73",
             accent="#ffc94a", carry="#b18cff",
             heat=("#0b0f17", "#10263f", "#14507a", "#1f8fc4", "#7fe0ff", "#fff6c9"))
LIGHT = Theme("light", bg="#ffffff", panel="#f4f6fa", pitch="#ffffff", line="#b9c2d0", text="#0f172a",
              muted="#64748b", grid="#e7ebf1", team="#1d5fd6", opp="#d6334f", good="#0f9f6e", bad="#e0344f",
              accent="#c98a00", carry="#7c4dff",
              heat=("#ffffff", "#e6f0ff", "#b6d1fb", "#5c95ee", "#1d4fb3", "#0b2463"))
THEMES = {"Dark": DARK, "Light": LIGHT}


# ======================================================================== scaffolding
def _rc(th: Theme) -> dict:
    return {"font.family": FONT, "text.color": th.text, "axes.labelcolor": th.muted, "xtick.color": th.muted,
            "ytick.color": th.muted, "axes.edgecolor": th.grid, "axes.facecolor": th.bg, "figure.facecolor": th.bg,
            "savefig.facecolor": th.bg, "axes.titlecolor": th.text}


def _pitch(th: Theme, vertical: bool = False, half: bool = False, **kw):
    cls = VerticalPitch if vertical else Pitch
    kw.setdefault("line_zorder", 1)
    return cls(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W, pitch_color=th.pitch,
               line_color=th.line, linewidth=1.1, goal_type="box", corner_arcs=True, half=half,
               pad_top=1.5, pad_bottom=1.5, pad_left=1.5, pad_right=1.5, **kw)


def _figure(th: Theme, size: tuple, title: str, subtitle: str, note: str = "") -> plt.Figure:
    fig = plt.figure(figsize=size, facecolor=th.bg)
    fig._ml_title, fig._ml_subtitle, fig._ml_text, fig._ml_muted = title, subtitle, th.text, th.muted
    fig.text(0.025, 0.965, title, ha="left", va="top", fontsize=19, fontweight="bold", color=th.text)
    fig.text(0.025, 0.915, subtitle, ha="left", va="top", fontsize=10.5, color=th.muted)
    fig.add_artist(plt.Line2D([0.025, 0.975], [0.885, 0.885], color=th.grid, lw=1, transform=fig.transFigure))
    fig.text(0.025, 0.018, "MATCH LAB  ·  pitch 105 × 68 m", ha="left", va="bottom", fontsize=7.5,
             color=th.muted, fontweight="bold", alpha=0.8)
    if note:
        fig.text(0.975, 0.018, note, ha="right", va="bottom", fontsize=7.5, color=th.muted)
    return fig


def _subtitle(md: MatchData, team: str, half, player, extra: str = "") -> str:
    parts = [f"{team} vs {md.opponent(team)}" if md.opponent(team) else team,
             {None: "Full match", 1: "1st half", 2: "2nd half"}.get(half, f"Half {half}")]
    if player:
        parts.append(f"Player #{player}")
    if extra:
        parts.append(extra)
    return "   ·   ".join(parts)


def _attack_arrow(ax, th: Theme, vertical: bool = False) -> None:
    if vertical:
        ax.annotate("", xy=(-3.2, 70), xytext=(-3.2, 35), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=th.muted, lw=1.4))
        ax.text(-4.6, 52, "ATTACK", rotation=90, color=th.muted, fontsize=7.5, ha="center", va="center",
                fontweight="bold")
        return
    ax.annotate("", xy=(70, -3.4), xytext=(35, -3.4), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=th.muted, lw=1.4))
    ax.text(52.5, -5.2, "ATTACKING DIRECTION", color=th.muted, fontsize=7, ha="center", va="top",
            fontweight="bold")


def _legend(ax, th: Theme, items: list[tuple[str, str, str]], y: float = -0.075, ncol: int | None = None) -> None:
    """items = [(label, color, kind)] kind ∈ line|dash|dot|x|bar."""
    from matplotlib.lines import Line2D
    handles = []
    for label, color, kind in items:
        if kind == "line":
            h = Line2D([0], [0], color=color, lw=2.6, label=label)
        elif kind == "dash":
            h = Line2D([0], [0], color=color, lw=2, ls=(0, (3, 2)), label=label)
        elif kind == "x":
            h = Line2D([0], [0], color=color, marker="X", lw=0, markersize=8, label=label)
        elif kind == "ring":
            h = Line2D([0], [0], markerfacecolor="none", markeredgecolor=color, marker="o", lw=0, markersize=9, label=label)
        else:
            h = Line2D([0], [0], color=color, marker="o", lw=0, markersize=8, label=label)
        handles.append(h)
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(items),
                    frameon=False, fontsize=8.5, labelcolor=th.muted, handlelength=1.8, columnspacing=1.6)
    return leg


def _kpis(fig, th: Theme, rect, items: list[tuple[str, str]], title: str = "", cols: int = 1,
          accent: str | None = None) -> None:
    """A stacked KPI panel (big value, small caption)."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03", fc=th.panel, ec=th.grid,
                                lw=1, transform=ax.transAxes))
    top = 0.95
    if title:
        ax.text(0.07, 0.955, title.upper(), fontsize=8, color=th.muted, fontweight="bold", va="top")
        top = 0.88
    rows = int(np.ceil(len(items) / cols))
    h = top / max(rows, 1)
    for i, (label, value) in enumerate(items):
        r, c = divmod(i, cols)
        x0 = 0.07 + c * (0.93 / cols)
        yv = top - r * h - h * 0.12
        ax.text(x0, yv, value, fontsize=17 if cols == 1 else 15, fontweight="bold",
                color=accent or th.text, va="top")
        ax.text(x0, yv - h * 0.5, label, fontsize=8, color=th.muted, va="top")
    return ax


def _fmt(v, d: int = 0, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, str):
        return v
    s = f"{v:.{d}f}"
    return s + ("%" if pct else "")


def _scope(md: MatchData, team: str, half, player) -> pd.DataFrame:
    ev = md.events
    ev = ev[ev["team"] == team]
    if half is not None:
        ev = ev[ev["half"] == half]
    if player:
        ev = ev[ev["player"] == str(player)]
    return ev


def _all_scope(md: MatchData, half) -> pd.DataFrame:
    return md.events if half is None else md.events[md.events["half"] == half]


def _bands(ax, th: Theme, md: MatchData, labels=("BUILD-UP", "PROGRESSION", "FINAL THIRD"), alpha=0.06,
           colors=None) -> None:
    t1, t2 = md.thirds
    xs = [(0, t1), (t1, t2), (t2, PITCH_L)]
    colors = colors or [th.team, th.accent, th.good]
    for (a, b), lab, col in zip(xs, labels, colors):
        ax.add_patch(Rectangle((a, 0), b - a, PITCH_W, fc=col, alpha=alpha, ec="none", zorder=0.5))
        ax.text((a + b) / 2, PITCH_W + 2.4, lab, ha="center", va="bottom", fontsize=8, color=th.muted,
                fontweight="bold")
    for xv in (t1, t2):
        ax.plot([xv, xv], [0, PITCH_W], color=th.line, lw=1, ls=(0, (4, 3)), zorder=1.2)


def _hbar(ax, th: Theme, labels, att, comp, color, title: str = "", pct: bool = True) -> None:
    """Attempted (faint) vs completed (solid) horizontal bars with % accuracy."""
    y = np.arange(len(labels))[::-1]
    att = np.asarray(att, float)
    comp = np.asarray(comp, float)
    ax.barh(y, att, color=to_rgba(color, 0.22), height=0.62, edgecolor="none")
    ax.barh(y, comp, color=color, height=0.62, edgecolor="none")
    m = max(att.max() if len(att) else 1, 1)
    for yi, a, c in zip(y, att, comp):
        txt = f"{int(c)}/{int(a)}" + (f"  {c / a * 100:.0f}%" if pct and a else "")
        ax.text(a + m * 0.02, yi, txt, va="center", fontsize=8.5, color=th.text)
    ax.set_yticks(y, labels, fontsize=9, color=th.text)
    ax.set_xlim(0, m * 1.45)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(th.grid)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title.upper(), loc="left", fontsize=9, color=th.muted, fontweight="bold", pad=6)


def _empty(ax, th: Theme, msg: str = "No events for this selection") -> None:
    ax.text(52.5, 34, msg, ha="center", va="center", color=th.muted, fontsize=12, zorder=5)


# ======================================================================== PASSING
PASS_FILTERS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "all": ("All passes", lambda p: pd.Series(True, index=p.index)),
    "progressive": ("Progressive passes", lambda p: p["progressive"]),
    "forward": ("Forward passes", lambda p: p["direction"] == "Forward"),
    "backward": ("Backward passes", lambda p: p["direction"] == "Backward"),
    "lateral": ("Lateral passes", lambda p: p["direction"].isin(["Left", "Right"])),
    "line_break": ("Line-breaking passes", lambda p: p["line_break"]),
    "f3_entry": ("Final-third entries", lambda p: p["f3_entry"]),
    "box_entry": ("Box entries", lambda p: p["box_entry"]),
    "long": ("Long passes (≥30 m)", lambda p: p["length_class"] == "Long"),
    "switch": ("Switches of play", lambda p: p["switch"]),
    "through_ball": ("Through balls", lambda p: p["through_ball"]),
    "cross": ("Crosses", lambda p: p["cross"]),
    "key_pass": ("Key passes & assists", lambda p: p["key_pass"]),
    "under_pressure": ("Passes under pressure", lambda p: p["under_pressure"]),
    "set_piece": ("Set-piece deliveries", lambda p: p["set_piece"]),
}


def pass_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, kind: str = "all",
             show_failed: bool = True, open_play_only: bool = False) -> plt.Figure:
    label, fn = PASS_FILTERS.get(kind, PASS_FILTERS["all"])
    th = theme
    with plt.rc_context(_rc(th)):
        p = _scope(md, team, half, player)
        p = p[(p["etype"] == "pass") & p["end_x"].notna()]
        if open_play_only:
            p = p[p["open_play"]]
        p = p[fn(p).fillna(False).astype(bool)]
        ok = p[p["outcome"] == 1]
        ko = p[p["outcome"] != 1]
        fig = _figure(th, (14, 8), f"{team} — {label}", _subtitle(md, team, half, player),
                      "End points inferred from the next tagged event")
        ax = fig.add_axes([0.02, 0.1, 0.72, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        if show_failed and len(ko):
            pitch.lines(ko["x"], ko["y"], ko["end_x"], ko["end_y"], comet=True, transparent=True,
                        color=th.bad, lw=2.2, alpha_start=0.02, alpha_end=0.55, ax=ax, zorder=2)
            pitch.scatter(ko["end_x"], ko["end_y"], s=26, marker="X", color=th.bad, alpha=0.75, ax=ax, zorder=3,
                          linewidths=0)
        if len(ok):
            pitch.lines(ok["x"], ok["y"], ok["end_x"], ok["end_y"], comet=True, transparent=True,
                        color=th.team, lw=2.8, alpha_start=0.03, alpha_end=0.95, ax=ax, zorder=4)
            pitch.scatter(ok["end_x"], ok["end_y"], s=34, color=th.team, edgecolors=th.bg, linewidths=0.8,
                          ax=ax, zorder=5)
        if p.empty:
            _empty(ax, th)
        _attack_arrow(ax, th)
        _legend(ax, th, [("Completed", th.team, "line"), ("Unsuccessful", th.bad, "x")], y=-0.06)
        comp = len(ok)
        att = len(p)
        fwd = (p["direction"] == "Forward").sum()
        items = [("Attempted", _fmt(att)), ("Completed", _fmt(comp)), ("Accuracy", _fmt(M._pct(comp, att), 1, True)),
                 ("Progressive (cmp)", _fmt(int((ok["progressive"]).sum()))),
                 ("Forward share", _fmt(M._pct(fwd, att), 0, True)),
                 ("Avg length", _fmt(p["length"].mean(), 1) + " m" if att else "–")]
        _kpis(fig, th, [0.765, 0.36, 0.21, 0.5], items, title="Summary")
        # top passers
        axb = fig.add_axes([0.80, 0.1, 0.17, 0.2])
        if not player and att:
            top = p.groupby("player")["outcome"].agg(att="size", comp=lambda s: int((s == 1).sum()))
            top = top[top.index.isin([x for x in top.index if x not in ("", "0")])].sort_values("att", ascending=False).head(5)
            if len(top):
                _hbar(axb, th, [f"#{i}" for i in top.index], top["att"], top["comp"], th.team, "Top players", pct=False)
            else:
                axb.set_axis_off()
        else:
            axb.set_axis_off()
    return fig


def progression_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, player)
        pp = ev[(ev["etype"] == "pass") & ev["progressive"] & (ev["outcome"] == 1)]
        pf = ev[(ev["etype"] == "pass") & ev["progressive"] & (ev["outcome"] == 0)]
        pc = ev[(ev["etype"] == "carry") & ev["progressive"]]
        fig = _figure(th, (14, 8), f"{team} — Ball progression",
                      _subtitle(md, team, half, player, "progressive passes & carries (Wyscout definition)"),
                      "Progressive = ≥30 m closer to goal in own half · ≥15 m across halves · ≥10 m in opp. half")
        ax = fig.add_axes([0.02, 0.1, 0.72, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        _bands(ax, th, md, alpha=0.035)
        if len(pf):
            pitch.lines(pf["x"], pf["y"], pf["end_x"], pf["end_y"], comet=True, transparent=True, color=th.bad,
                        lw=2, alpha_end=0.45, ax=ax, zorder=2)
        if len(pc):
            pitch.lines(pc["x"], pc["y"], pc["end_x"], pc["end_y"], color=th.carry, lw=2, ls=(0, (2, 2)),
                        ax=ax, zorder=3, alpha=0.9)
            pitch.scatter(pc["end_x"], pc["end_y"], s=22, color=th.carry, ax=ax, zorder=4)
        if len(pp):
            pitch.arrows(pp["x"], pp["y"], pp["end_x"], pp["end_y"], color=th.team, width=1.8, headwidth=4.5,
                         headlength=4.5, ax=ax, zorder=5, alpha=0.95)
        _attack_arrow(ax, th)
        _legend(ax, th, [("Progressive pass", th.team, "line"), ("Progressive carry", th.carry, "dash"),
                         ("Unsuccessful prog. pass", th.bad, "line")], y=-0.06)
        # counts by origin third
        t1, t2 = md.thirds
        for a, b in ((0, t1), (t1, t2), (t2, PITCH_L)):
            n_p = ((pp["x"] >= a) & (pp["x"] < b)).sum()
            n_c = ((pc["x"] >= a) & (pc["x"] < b)).sum()
            ax.text((a + b) / 2, 1.2, f"{n_p} passes · {n_c} carries", ha="center", va="bottom", fontsize=8.5,
                    color=th.text, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc=th.panel, ec=th.grid, alpha=0.9))
        items = [("Progressive passes", _fmt(len(pp))), ("Prog. pass accuracy", _fmt(M._pct(len(pp), len(pp) + len(pf)), 0, True)),
                 ("Progressive carries", _fmt(len(pc))), ("Final-third entries", _fmt(int(((ev["etype"].isin(["pass", "carry"])) & ev["f3_entry"] & (ev["outcome"] == 1)).sum()))),
                 ("Box entries", _fmt(int(((ev["etype"].isin(["pass", "carry"])) & ev["box_entry"] & (ev["outcome"] == 1)).sum())))]
        _kpis(fig, th, [0.765, 0.42, 0.21, 0.44], items, title="Progression")
        axb = fig.add_axes([0.80, 0.1, 0.17, 0.26])
        if not player:
            both = pd.concat([pp.assign(k="p"), pc.assign(k="c")])
            both = both[both["player"].ne("") & both["player"].ne("0")]
            if len(both):
                top = both.groupby(["player", "k"]).size().unstack(fill_value=0)
                for col in ("p", "c"):
                    if col not in top:
                        top[col] = 0
                top["t"] = top["p"] + top["c"]
                top = top.sort_values("t", ascending=False).head(6)
                y = np.arange(len(top))[::-1]
                axb.barh(y, top["p"], color=th.team, height=0.6)
                axb.barh(y, top["c"], left=top["p"], color=th.carry, height=0.6)
                for yi, t in zip(y, top["t"]):
                    axb.text(t + 0.2, yi, str(int(t)), va="center", fontsize=8.5, color=th.text)
                axb.set_yticks(y, [f"#{i}" for i in top.index], fontsize=9, color=th.text)
                axb.set_xticks([])
                for s in ("top", "right", "bottom"):
                    axb.spines[s].set_visible(False)
                axb.spines["left"].set_color(th.grid)
                axb.tick_params(length=0)
                axb.set_title("TOP PROGRESSORS", loc="left", fontsize=9, color=th.muted, fontweight="bold")
            else:
                axb.set_axis_off()
        else:
            axb.set_axis_off()
    return fig


def pass_profile(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    """Pass types, directions, lengths, heights, lines broken + a direction rose."""
    th = theme
    with plt.rc_context(_rc(th)):
        bd = M.pass_breakdown(md, team, half, player)
        p = _scope(md, team, half, player)
        p = p[(p["etype"] == "pass") & p["angle"].notna()]
        fig = _figure(th, (15, 8.4), f"{team} — Passing profile", _subtitle(md, team, half, player),
                      "Faint bar = attempted · solid = completed")
        # rose
        axr = fig.add_axes([0.03, 0.12, 0.28, 0.68], projection="polar")
        axr.set_facecolor(th.panel)
        axr.set_theta_zero_location("N")
        axr.set_theta_direction(1)
        bins = np.linspace(-180, 180, 17)
        ang = p["angle"].to_numpy()
        cnt, _ = np.histogram(ang, bins=bins)
        okc, _ = np.histogram(p.loc[p["outcome"] == 1, "angle"], bins=bins)
        acc = np.divide(okc, cnt, out=np.zeros_like(okc, dtype=float), where=cnt > 0)
        centers = np.radians((bins[:-1] + bins[1:]) / 2)
        cmap = LinearSegmentedColormap.from_list("acc", [th.bad, th.accent, th.good])
        axr.bar(centers, cnt, width=np.radians(22.5) * 0.92, color=[cmap(a) for a in acc], edgecolor=th.bg,
                linewidth=0.8, alpha=0.95)
        axr.set_xticks(np.radians([0, 90, 180, 270]), ["FORWARD", "LEFT", "BACK", "RIGHT"], fontsize=8,
                       color=th.muted, fontweight="bold")
        axr.set_yticklabels([])
        axr.grid(color=th.grid, lw=0.8)
        axr.spines["polar"].set_color(th.grid)
        axr.set_title("DIRECTION ROSE  (colour = accuracy)", fontsize=9, color=th.muted, fontweight="bold", pad=14)
        # bars
        specs = [("pass_type", "Pass type", [0.40, 0.47, 0.25, 0.36]),
                 ("direction", "Direction", [0.73, 0.66, 0.24, 0.17]),
                 ("length_class", "Length", [0.73, 0.43, 0.24, 0.15]),
                 ("height", "Height", [0.40, 0.11, 0.25, 0.22]),
                 ("line", "Line broken", [0.73, 0.11, 0.24, 0.2])]
        for key, title, rect in specs:
            ax = fig.add_axes(rect)
            ax.set_facecolor(th.bg)
            g = bd[key]
            if g.empty:
                ax.set_axis_off()
                ax.text(0.0, 0.5, f"{title}: no data", color=th.muted, fontsize=9, transform=ax.transAxes)
                continue
            _hbar(ax, th, list(g.index), g["attempted"], g["completed"], th.team, title)
    return fig


def pass_sonars(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
                bins: tuple = (6, 4)) -> plt.Figure:
    """Zone sonars: from each pitch zone, where do passes go (wedge = direction, length = volume,
    colour = accuracy)."""
    th = theme
    with plt.rc_context(_rc(th)):
        p = _scope(md, team, half, player)
        p = p[(p["etype"] == "pass") & p["angle"].notna()]
        fig = _figure(th, (14, 8.6), f"{team} — Pass sonars by zone", _subtitle(md, team, half, player),
                      "Wedge = direction · length = share of the zone's passes · colour = accuracy")
        ax = fig.add_axes([0.03, 0.1, 0.8, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        nx, ny = bins
        wx, wy = PITCH_L / nx, PITCH_W / ny
        cmap = LinearSegmentedColormap.from_list("acc", [th.bad, th.accent, th.good])
        rmax = min(wx, wy) * 0.46
        for i in range(nx):
            for j in range(ny):
                z = p[(p["x"] >= i * wx) & (p["x"] < (i + 1) * wx) & (p["y"] >= j * wy) & (p["y"] < (j + 1) * wy)]
                cx, cy = (i + 0.5) * wx, (j + 0.5) * wy
                ax.add_patch(Rectangle((i * wx, j * wy), wx, wy, fill=False, ec=th.grid, lw=0.6, zorder=1))
                if len(z) < 2:
                    continue
                edges = np.linspace(-180, 180, 13)
                cnt, _ = np.histogram(z["angle"], bins=edges)
                okc, _ = np.histogram(z.loc[z["outcome"] == 1, "angle"], bins=edges)
                mx = cnt.max()
                for k in range(12):
                    if cnt[k] == 0:
                        continue
                    r = rmax * (0.25 + 0.75 * cnt[k] / mx)
                    ax.add_patch(Wedge((cx, cy), r, edges[k] + 1.5, edges[k + 1] - 1.5,
                                       fc=cmap(okc[k] / cnt[k]), ec="none", alpha=0.92, zorder=3))
                ax.text(cx, j * wy + 1.0, f"{len(z)}", ha="center", va="bottom", fontsize=7.5, color=th.muted, zorder=4)
        _attack_arrow(ax, th)
        cax = fig.add_axes([0.86, 0.3, 0.015, 0.4])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("Accuracy %", color=th.muted, fontsize=9)
        cb.outline.set_edgecolor(th.grid)
        cb.ax.tick_params(colors=th.muted, labelsize=8)
    return fig


def pass_flow(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
              bins: tuple = (6, 5)) -> plt.Figure:
    """Zone-to-zone flow: zone shade = passes started there, arrow = average pass vector."""
    th = theme
    with plt.rc_context(_rc(th)):
        p = _scope(md, team, half, player)
        p = p[(p["etype"] == "pass") & p["end_x"].notna()]
        fig = _figure(th, (14, 8.6), f"{team} — Pass flow", _subtitle(md, team, half, player),
                      "Shade = passes started in the zone · arrow = average direction & distance")
        ax = fig.add_axes([0.03, 0.1, 0.8, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        if len(p):
            st = pitch.bin_statistic(p["x"], p["y"], statistic="count", bins=bins)
            pitch.heatmap(st, ax=ax, cmap=th.cmap(), edgecolors=th.bg, lw=1.5, alpha=0.9, zorder=0.8)
            nx, ny = bins
            wx, wy = PITCH_L / nx, PITCH_W / ny
            for i in range(nx):
                for j in range(ny):
                    z = p[(p["x"] >= i * wx) & (p["x"] < (i + 1) * wx) & (p["y"] >= j * wy) & (p["y"] < (j + 1) * wy)]
                    if len(z) < 2:
                        continue
                    cx, cy = (i + 0.5) * wx, (j + 0.5) * wy
                    vx, vy = (z["end_x"] - z["x"]).mean(), (z["end_y"] - z["y"]).mean()
                    n = np.hypot(vx, vy)
                    scale = min(1.0, (min(wx, wy) * 0.42) / n) if n else 0
                    acc = (z["outcome"] == 1).mean()
                    ax.annotate("", xy=(cx + vx * scale, cy + vy * scale), xytext=(cx - vx * scale * 0.4, cy - vy * scale * 0.4),
                                arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.8", color=th.text,
                                                lw=2.2, alpha=0.35 + 0.6 * acc), zorder=4)
                    ax.text(cx, j * wy + 1.0, f"{len(z)} · {acc * 100:.0f}%", ha="center", va="bottom",
                            fontsize=7.5, color=th.text, alpha=0.85, zorder=5)
        else:
            _empty(ax, th)
        _attack_arrow(ax, th)
    return fig


def _starters(md: MatchData, team: str) -> set:
    ev = md.events[md.events["team"] == team]
    subs = set(ev[ev["etype"] == "sub_in"]["player"])
    return {p for p in ev["player"].unique() if p and p != "0" and p not in subs}


def pass_network(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
                 min_passes: int = 3, include_subs: bool = False) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, None)
        p = ev[(ev["etype"] == "pass") & (ev["outcome"] == 1) & ev["receiver"].ne("") & ev["player"].ne("")
               & ev["player"].ne("0")]
        starters = _starters(md, team)
        keep = (lambda s: s.isin(starters)) if not include_subs else (lambda s: s.ne(""))
        p = p[keep(p["player"]) & keep(p["receiver"])]
        touch = ev[ev["etype"].isin(["pass", "receive", "carry", "dribble", "shot"]) & ev["player"].isin(set(p["player"]) | set(p["receiver"]))]
        pos = touch.groupby("player")[["x", "y"]].mean()
        vol = p.groupby("player").size().add(p.groupby("receiver").size(), fill_value=0)
        fig = _figure(th, (14, 8.6), f"{team} — Passing network",
                      _subtitle(md, team, half, None, "starting XI" if not include_subs else "all players"),
                      f"Node = average touch position, size = passes made + received · links ≥ {min_passes} passes")
        ax = fig.add_axes([0.02, 0.1, 0.72, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        pairs = p.assign(a=np.minimum(p["player"].astype(str), p["receiver"].astype(str)),
                         b=np.maximum(p["player"].astype(str), p["receiver"].astype(str))).groupby(["a", "b"]).size()
        pairs = pairs[pairs >= min_passes].sort_values()
        mx = pairs.max() if len(pairs) else 1
        for (a, b), n in pairs.items():
            if a in pos.index and b in pos.index:
                lw = 1 + 7 * n / mx
                ax.plot([pos.at[a, "x"], pos.at[b, "x"]], [pos.at[a, "y"], pos.at[b, "y"]], color=th.team,
                        lw=lw, alpha=0.18 + 0.7 * n / mx, solid_capstyle="round", zorder=2)
        if len(pos):
            vmax = vol.max()
            for pl, r in pos.iterrows():
                s = 380 + 1500 * (vol.get(pl, 0) / vmax)
                sub = pl not in starters
                ax.scatter(r["x"], r["y"], s=s, color=th.panel, edgecolors=th.accent if sub else th.team,
                           linewidths=2.4, zorder=4)
                ax.text(r["x"], r["y"], pl, ha="center", va="center", fontsize=11, fontweight="bold",
                        color=th.text, zorder=5)
        else:
            _empty(ax, th)
        _attack_arrow(ax, th)
        top = pairs.sort_values(ascending=False).head(8)
        items = [(f"#{a} ↔ #{b}", str(int(n))) for (a, b), n in top.items()]
        _kpis(fig, th, [0.765, 0.1, 0.21, 0.76], items or [("No links", "–")], title="Top combinations")
    return fig


def passing_matrix(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, None)
        p = ev[(ev["etype"] == "pass") & (ev["outcome"] == 1) & ev["receiver"].ne("") & ev["player"].ne("")
               & ev["player"].ne("0")]
        players = sorted(set(p["player"]) | set(p["receiver"]), key=lambda s: (len(s), s))
        mat = pd.crosstab(p["player"], p["receiver"]).reindex(index=players, columns=players, fill_value=0)
        fig = _figure(th, (12.5, 10.5), f"{team} — Passing combination matrix", _subtitle(md, team, half, None),
                      "Completed passes, passer (row) → receiver (column)")
        ax = fig.add_axes([0.1, 0.09, 0.78, 0.76])
        if mat.size:
            im = ax.imshow(mat.to_numpy(), cmap=th.cmap(), aspect="auto")
            vmax = mat.to_numpy().max() or 1
            for i in range(len(players)):
                for j in range(len(players)):
                    v = mat.iat[i, j]
                    if v:
                        ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                                color=th.bg if v > vmax * 0.6 else th.text, fontweight="bold")
            ax.set_xticks(range(len(players)), [f"#{x}" for x in players], fontsize=9)
            ax.set_yticks(range(len(players)), [f"#{x}" for x in players], fontsize=9)
            ax.set_xlabel("Receiver", color=th.muted)
            ax.set_ylabel("Passer", color=th.muted)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            cax = fig.add_axes([0.9, 0.25, 0.015, 0.45])
            cb = fig.colorbar(im, cax=cax)
            cb.outline.set_edgecolor(th.grid)
            cb.ax.tick_params(colors=th.muted, labelsize=8)
        else:
            ax.set_axis_off()
    return fig


# ======================================================================== TERRITORY
def touch_heatmap(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
                  style: str = "smooth") -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, player)
        ev = ev[ev["etype"].isin(["pass", "receive", "carry", "dribble", "shot", "recovery", "interception", "tackle", "duel"])
                & ev["x"].notna() & ~ev["derived"]]
        fig = _figure(th, (14, 8.6), f"{team} — {'Touch' if player else 'Activity'} heat map",
                      _subtitle(md, team, half, player, f"{len(ev)} on-ball & defensive actions"),
                      "Positional zones: 5 channels × thirds/halves" if style == "zones" else "Gaussian-smoothed density")
        ax = fig.add_axes([0.03, 0.1, 0.8, 0.76])
        pitch = _pitch(th, line_zorder=3)
        pitch.draw(ax=ax)
        if len(ev):
            if style == "zones":
                st = pitch.bin_statistic_positional(ev["x"], ev["y"], statistic="count", positional="full", normalize=True)
                pitch.heatmap_positional(st, ax=ax, cmap=th.cmap(), edgecolors=th.bg, lw=1.5)
                pitch.label_heatmap(st, ax=ax, str_format="{:.0%}", color=th.text, fontsize=10, fontweight="bold",
                                    ha="center", va="center", exclude_zeros=True,
                                    path_effects=[])
            else:
                from scipy.ndimage import gaussian_filter
                st = pitch.bin_statistic(ev["x"], ev["y"], statistic="count", bins=(42, 28), normalize=True)
                st["statistic"] = gaussian_filter(st["statistic"], 1.6)
                pitch.heatmap(st, ax=ax, cmap=th.cmap(), edgecolors="none", zorder=0.6)
        else:
            _empty(ax, th)
        _attack_arrow(ax, th)
        # third / channel shares
        if len(ev):
            thirds = pd.Series(M.third_of(ev["x"])).value_counts(normalize=True).reindex(M.THIRD_NAMES).fillna(0)
            chans = pd.Series(M.channel_of(ev["y"])).value_counts(normalize=True).reindex(M.CHANNELS[::-1]).fillna(0)
            items = [(f"{k}", f"{v * 100:.0f}%") for k, v in thirds.items()] + \
                    [(f"{k}", f"{v * 100:.0f}%") for k, v in chans.items()]
            _kpis(fig, th, [0.845, 0.1, 0.13, 0.76], items, title="Share")
    return fig


def entries_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
                open_play_only: bool = True) -> plt.Figure:
    """Final-third & box entries by the 5 attacking channels."""
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, player)
        mv = ev[ev["etype"].isin(["pass", "carry"]) & ev["end_x"].notna() & (ev["f3_entry"] | ev["box_entry"])]
        if open_play_only:
            mv = mv[mv["open_play"]]
        fig = _figure(th, (15, 8.4), f"{team} — Final-third & box entries",
                      _subtitle(md, team, half, player, "open play" if open_play_only else "incl. set pieces"),
                      "Channel = where the entry lands (5-lane model)")
        ax = fig.add_axes([0.02, 0.1, 0.62, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        for yv in M.CHANNEL_EDGES[1:-1]:
            ax.plot([M.THIRDS[1], PITCH_L], [yv, yv], color=th.line, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.plot([M.THIRDS[1]] * 2, [0, PITCH_W], color=th.accent, lw=1.2, ls=(0, (5, 3)), zorder=1, alpha=0.7)
        for kind, sub in (("fail", mv[(mv["etype"] == "pass") & (mv["outcome"] != 1)]),
                          ("pass", mv[(mv["etype"] == "pass") & (mv["outcome"] == 1)]),
                          ("carry", mv[mv["etype"] == "carry"])):
            if sub.empty:
                continue
            col = {"fail": th.bad, "pass": th.team, "carry": th.carry}[kind]
            box = sub["box_entry"]
            pitch.arrows(sub["x"], sub["y"], sub["end_x"], sub["end_y"], color=col, width=1.5 if kind != "fail" else 1.1,
                         headwidth=4.5, headlength=4.5, alpha=0.85 if kind != "fail" else 0.45, ax=ax, zorder=3)
            if box.any():
                pitch.scatter(sub.loc[box, "end_x"], sub.loc[box, "end_y"], s=40, facecolor="none",
                              edgecolors=th.accent, linewidths=1.4, ax=ax, zorder=4)
        _attack_arrow(ax, th)
        _legend(ax, th, [("Completed pass", th.team, "line"), ("Carry", th.carry, "line"),
                         ("Unsuccessful", th.bad, "line"), ("Box entry", th.accent, "ring")], y=-0.06)
        # channel table
        axc = fig.add_axes([0.70, 0.5, 0.27, 0.34])
        ch = pd.Series(M.channel_of(mv["end_y"]), index=mv.index)
        ok = (mv["outcome"] == 1) | (mv["etype"] == "carry")
        rows = []
        for c in M.CHANNELS[::-1]:
            m = ch == c
            rows.append((c, int(m.sum()), int((m & ok).sum())))
        _hbar(axc, th, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], th.team, "Entries by channel")
        f3 = mv[mv["f3_entry"]]
        bx = mv[mv["box_entry"]]
        f3ok = ((f3["outcome"] == 1) | (f3["etype"] == "carry")).sum()
        bxok = ((bx["outcome"] == 1) | (bx["etype"] == "carry")).sum()
        items = [("Final-third entries", f"{f3ok}/{len(f3)}"), ("F3 success", _fmt(M._pct(f3ok, len(f3)), 0, True)),
                 ("Box entries", f"{bxok}/{len(bx)}"), ("Box success", _fmt(M._pct(bxok, len(bx)), 0, True))]
        _kpis(fig, th, [0.70, 0.1, 0.27, 0.32], items, title="Totals", cols=2)
    return fig


def crosses_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, player)
        c = ev[(ev["etype"] == "pass") & ev["cross"] & ev["end_x"].notna()]
        fig = _figure(th, (14, 9), f"{team} — Crosses", _subtitle(md, team, half, player),
                      "Delivery lines + landing-zone density")
        ax = fig.add_axes([0.03, 0.08, 0.6, 0.8])
        pitch = _pitch(th, vertical=True, half=True)
        pitch.draw(ax=ax)
        if len(c):
            from scipy.ndimage import gaussian_filter
            land = c[c["end_x"] >= 60]
            st = pitch.bin_statistic(land["end_x"], land["end_y"], statistic="count", bins=(30, 20))
            st["statistic"] = gaussian_filter(st["statistic"].astype(float), 1.3)
            pitch.heatmap(st, ax=ax, cmap=LinearSegmentedColormap.from_list("x", [to_rgba(th.pitch, 0), to_rgba(th.accent, 0.75)]),
                          edgecolors="none", zorder=0.6)
            ok = c[c["outcome"] == 1]
            ko = c[c["outcome"] != 1]
            kp = c[c["key_pass"]]
            pitch.lines(ko["x"], ko["y"], ko["end_x"], ko["end_y"], comet=True, transparent=True, color=th.bad,
                        lw=2, alpha_end=0.6, ax=ax, zorder=2)
            pitch.lines(ok["x"], ok["y"], ok["end_x"], ok["end_y"], comet=True, transparent=True, color=th.good,
                        lw=2.6, alpha_end=0.95, ax=ax, zorder=3)
            pitch.scatter(c["end_x"], c["end_y"], s=26, color=np.where(c["outcome"] == 1, th.good, th.bad), ax=ax, zorder=4)
            if len(kp):
                pitch.scatter(kp["end_x"], kp["end_y"], s=150, marker="*", color=th.accent, edgecolors=th.bg, ax=ax, zorder=5)
        else:
            ax.text(34, 80, "No crosses", ha="center", color=th.muted, fontsize=12)
        _legend(ax, th, [("Completed", th.good, "line"), ("Unsuccessful", th.bad, "line"), ("Key pass", th.accent, "dot")], y=0.0)
        left = c[c["y"] > PITCH_W / 2]
        right = c[c["y"] <= PITCH_W / 2]
        items = [("Crosses", _fmt(len(c))), ("Completed", _fmt(int((c["outcome"] == 1).sum()))),
                 ("Accuracy", _fmt(M._pct((c["outcome"] == 1).sum(), len(c)), 0, True)),
                 ("Key passes", _fmt(int(c["key_pass"].sum()))),
                 ("From the left", f"{int((left['outcome'] == 1).sum())}/{len(left)}"),
                 ("From the right", f"{int((right['outcome'] == 1).sum())}/{len(right)}")]
        _kpis(fig, th, [0.67, 0.46, 0.3, 0.4], items, title="Crossing", cols=2)
        axb = fig.add_axes([0.72, 0.08, 0.24, 0.3])
        if not player and len(c):
            g = c[c["player"].ne("") & c["player"].ne("0")].groupby("player")["outcome"].agg(att="size", comp=lambda s: int((s == 1).sum()))
            g = g.sort_values("att", ascending=False).head(6)
            _hbar(axb, th, [f"#{i}" for i in g.index], g["att"], g["comp"], th.good, "By player")
        else:
            axb.set_axis_off()
    return fig


def shot_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, player)
        s = ev[(ev["etype"] == "shot") & ev["x"].notna()]
        fig = _figure(th, (13, 9), f"{team} — Shot map", _subtitle(md, team, half, player),
                      "Marker size = xG")
        ax = fig.add_axes([0.03, 0.08, 0.6, 0.8])
        pitch = _pitch(th, vertical=True, half=True)
        pitch.draw(ax=ax)
        style = {"Goal": (th.accent, th.accent, "*", 1.7), "Saved": (th.team, th.team, "o", 1.0),
                 "Blocked": (th.muted, th.muted, "o", 1.0), "Off target": ("none", th.bad, "o", 1.0),
                 "": ("none", th.muted, "o", 1.0)}
        for res, (fc, ec, mk, k) in style.items():
            g = s[s["shot_result"] == res]
            if len(g):
                pitch.scatter(g["x"], g["y"], s=(g["xg"].fillna(0.03) * 1600 + 70) * k, marker=mk, facecolor=fc,
                              edgecolors=ec, linewidths=1.8, alpha=0.9, ax=ax, zorder=4)
        for _, r in s.iterrows():
            if r["player"] and r["player"] != "0":
                ax.text(r["y"], r["x"] - 2.2, f"#{r['player']}", ha="center", va="top", fontsize=7, color=th.muted, zorder=5)
        _legend(ax, th, [("Goal", th.accent, "dot"), ("Saved", th.team, "dot"), ("Blocked", th.muted, "dot"),
                         ("Off target", th.bad, "ring")], y=0.0)
        xg = s["xg"].sum()
        items = [("Shots", _fmt(len(s))), ("On target", _fmt(int(s["shot_on"].sum()))), ("Goals", _fmt(int(s["shot_goal"].sum()))),
                 ("xG", _fmt(xg, 2)), ("xG / shot", _fmt(xg / len(s), 2) if len(s) else "–"),
                 ("Avg distance", _fmt(M.dist_to_goal(s["x"], s["y"]).mean(), 1) + " m" if len(s) else "–"),
                 ("Headers", _fmt(int((s["body_part"] == "Head").sum()))),
                 ("From set pieces", _fmt(int(s["set_piece"].sum())))]
        _kpis(fig, th, [0.67, 0.12, 0.3, 0.74], items, title="Shooting", cols=2)
    return fig


# ======================================================================== DEFENDING
_DEF_STYLE = {"interception": ("Interception", "o"), "tackle": ("Tackle", "^"), "recovery": ("Recovery", "D"),
              "duel": ("Duel", "s"), "foul": ("Foul", "X"), "block": ("Block", "P"), "clearance": ("Clearance", "v"),
              "press": ("Press", "o")}


def defensive_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
                  include_press: bool = False) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, player)
        kinds = [k for k in _DEF_STYLE if include_press or k != "press"]
        d = ev[ev["etype"].isin(kinds) & ev["x"].notna()]
        fig = _figure(th, (15, 8.6), f"{team} — Defensive actions", _subtitle(md, team, half, player),
                      "6-zone breakdown: 3 thirds × 2 halves of the width")
        ax = fig.add_axes([0.02, 0.1, 0.66, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        t1, t2 = md.thirds
        zx = [(0, t1), (t1, t2), (t2, PITCH_L)]
        for zi, (a, b) in enumerate(zx):
            for yi, (c0, c1) in enumerate(((0, PITCH_W / 2), (PITCH_W / 2, PITCH_W))):
                m = (d["x"] >= a) & (d["x"] < b) & (d["y"] >= c0) & (d["y"] < c1)
                share = m.sum() / len(d) if len(d) else 0
                ax.add_patch(Rectangle((a, c0), b - a, c1 - c0, fc=th.team, alpha=0.04 + 0.32 * share, ec=th.grid, lw=0.8, zorder=0.5))
                ax.text((a + b) / 2, c1 - 2, f"{m.sum()}  ·  {share * 100:.0f}%", ha="center", va="top", fontsize=9,
                        color=th.text, fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25", fc=th.panel, ec="none", alpha=0.85))
        palette = [th.team, th.good, th.accent, th.carry, th.bad, th.text, th.muted, th.opp]
        items_leg = []
        for k, col in zip(kinds, palette):
            g = d[d["etype"] == k]
            if g.empty:
                continue
            name, mk = _DEF_STYLE[k]
            won = g["outcome"] != 0
            pitch.scatter(g.loc[won, "x"], g.loc[won, "y"], s=60, marker=mk, color=col, edgecolors=th.bg, linewidths=0.6, ax=ax, zorder=4)
            pitch.scatter(g.loc[~won, "x"], g.loc[~won, "y"], s=60, marker=mk, facecolor="none", edgecolors=col, linewidths=1.2, ax=ax, zorder=4)
            items_leg.append((f"{name} ({len(g)})", col, "dot"))
        if len(d):
            h = d["x"].mean()
            ax.plot([h, h], [0, PITCH_W], color=th.accent, lw=2, ls=(0, (6, 3)), zorder=5)
            ax.text(h, PITCH_W + 1.2, f"Avg. defensive height {h:.1f} m", ha="center", va="bottom", fontsize=8.5,
                    color=th.accent, fontweight="bold")
        _attack_arrow(ax, th)
        if items_leg:
            _legend(ax, th, items_leg, y=-0.06, ncol=min(len(items_leg), 4))
        k = P.defending_kpis(md, team, half)
        items = [("PPDA", _fmt(k["PPDA"], 1)), ("Defensive actions", _fmt(len(d))),
                 ("High regains", _fmt(k["High regains"])), ("Counter-press actions", _fmt(k["Counter-press actions"])),
                 (f"Regained ≤{md.cp_window:.0f} s after loss", f"{k['Regained in window']}/{k['Losses']}"),
                 ("Opp. shots / xG allowed", f"{k['Opp. shots allowed']} / {k['Opp. xG allowed']:.2f}")]
        _kpis(fig, th, [0.71, 0.1, 0.26, 0.76], items, title="Defending")
    return fig


def losses_recoveries(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _scope(md, team, half, player)
        loss = ev[((ev["etype"] == "pass") & (ev["outcome"] == 0)) | ((ev["etype"] == "dribble") & (ev["outcome"] == 0))
                  | (ev["etype"] == "ball_lost") | ((ev["etype"] == "error") & ev["labels"].map(lambda l: "ball lost" in l))]
        loss = loss[loss["x"].notna()]
        reg = ev[(ev["etype"] == "recovery") | (ev["etype"].isin(["interception", "tackle"]) & ev["labels"].map(lambda l: "won" in l))]
        reg = reg[reg["x"].notna()]
        fig = _figure(th, (14, 8.6), f"{team} — Ball losses & regains", _subtitle(md, team, half, player),
                      "Loss = unsuccessful pass / dribble / tagged ball lost")
        ax = fig.add_axes([0.02, 0.1, 0.72, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        _bands(ax, th, md, alpha=0.035)
        pitch.scatter(reg["x"], reg["y"], s=70, color=th.good, edgecolors=th.bg, linewidths=0.6, alpha=0.85, ax=ax, zorder=3)
        danger = loss["x"] < md.thirds[0]
        pitch.scatter(loss.loc[~danger, "x"], loss.loc[~danger, "y"], s=70, marker="X", color=th.bad, alpha=0.75, ax=ax, zorder=4)
        pitch.scatter(loss.loc[danger, "x"], loss.loc[danger, "y"], s=150, marker="X", color=th.bad, edgecolors=th.accent,
                      linewidths=1.2, ax=ax, zorder=5)
        _attack_arrow(ax, th)
        _legend(ax, th, [("Regain", th.good, "dot"), ("Ball loss", th.bad, "x"), ("Loss in own third", th.accent, "ring")], y=-0.06)
        t1, t2 = md.thirds
        items = []
        for name, a, b in (("own third", 0, t1), ("middle third", t1, t2), ("final third", t2, PITCH_L + 1)):
            items.append((f"Losses · {name}", _fmt(int(((loss["x"] >= a) & (loss["x"] < b)).sum()))))
        for name, a, b in (("own third", 0, t1), ("middle third", t1, t2), ("final third", t2, PITCH_L + 1)):
            items.append((f"Regains · {name}", _fmt(int(((reg["x"] >= a) & (reg["x"] < b)).sum()))))
        _kpis(fig, th, [0.765, 0.1, 0.21, 0.76], items, title="By zone")
    return fig


# ======================================================================== PHASES
def phases_in_possession(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        tab = P.in_possession_table(md, team, half)
        ev = _scope(md, team, half, None)
        fig = _figure(th, (16, 9.6), f"{team} — Phases of play · in possession",
                      _subtitle(md, team, half, None, "phase = where the ball is (x)"),
                      f"Arrows = progressive passes · ✕ = ball losses   |   Build-up < {md.thirds[0]:.0f} m ≤ Progression < {md.thirds[1]:.0f} m ≤ Final third")
        ax = fig.add_axes([0.02, 0.37, 0.62, 0.5])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        cols = [th.team, th.accent, th.good]
        _bands(ax, th, md, alpha=0.09, colors=cols)
        prog = ev[(ev["etype"] == "pass") & ev["progressive"] & (ev["outcome"] == 1)]
        pitch.arrows(prog["x"], prog["y"], prog["end_x"], prog["end_y"], color=th.text, alpha=0.35, width=1.2,
                     headwidth=4, headlength=4, ax=ax, zorder=2)
        loss = ev[(ev["etype"].isin(["pass", "dribble"])) & (ev["outcome"] == 0) & ev["x"].notna()]
        pitch.scatter(loss["x"], loss["y"], s=36, marker="X", color=th.bad, alpha=0.8, ax=ax, zorder=3)
        t1, t2 = md.thirds
        for (a, b), (_, r), col in zip(((0, t1), (t1, t2), (t2, PITCH_L)), tab.iterrows(), cols):
            ax.text((a + b) / 2, PITCH_W / 2,
                    f"{r['Share %']:.0f}%\n", ha="center", va="center", fontsize=26, fontweight="bold", color=col, zorder=6,
                    alpha=0.95)
            ax.text((a + b) / 2, PITCH_W / 2 - 6, "of actions", ha="center", va="center", fontsize=8.5, color=th.muted, zorder=6)
        _attack_arrow(ax, th)
        # phase table cards
        for i, (_, r) in enumerate(tab.iterrows()):
            items = [("Pass accuracy", _fmt(r["Accuracy %"], 0, True)), ("Forward passes", _fmt(r["Forward %"], 0, True)),
                     ("Progressive", _fmt(r["Progressive"])), ("Line breaks", _fmt(r["Line breaks"])),
                     ("Losses", f"{r['Losses']} ({_fmt(r['Loss %'], 0, True)})"), ("Avg pass", _fmt(r["Avg pass (m)"], 1) + " m")]
            _kpis(fig, th, [0.66 + (i % 3) * 0.11, 0.37, 0.105, 0.5], items, title=r["Phase"], accent=cols[i])
        # funnel
        axf = fig.add_axes([0.05, 0.07, 0.4, 0.22])
        fn = P.funnel(md, team, half)
        if fn:
            labels = [a for a, _ in fn]
            vals = np.array([b for _, b in fn], float)
            y = np.arange(len(vals))[::-1]
            axf.barh(y, vals, color=[to_rgba(th.team, 0.35 + 0.65 * i / max(len(vals) - 1, 1)) for i in range(len(vals))], height=0.7)
            for yi, v in zip(y, vals):
                axf.text(v + vals.max() * 0.01, yi, f"{int(v)}  ({v / vals[0] * 100:.0f}%)" if vals[0] else str(int(v)),
                         va="center", fontsize=9, color=th.text)
            axf.set_yticks(y, labels, fontsize=9, color=th.text)
            axf.set_xlim(0, vals.max() * 1.3 if vals.max() else 1)
        axf.set_xticks([])
        for s_ in ("top", "right", "bottom"):
            axf.spines[s_].set_visible(False)
        axf.spines["left"].set_color(th.grid)
        axf.tick_params(length=0)
        axf.set_title("POSSESSION FUNNEL — ALL POSSESSIONS", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        # build-up outcome
        axb = fig.add_axes([0.56, 0.07, 0.4, 0.22])
        bo = P.buildup_outcomes(md, team, half)
        if bo.get("count"):
            fur = bo["furthest"]
            labels = list(M.THIRD_NAMES)
            vals = [fur.get(l, 0) for l in labels]
            axb.bar(["Stayed in own third", "Reached middle third", "Reached final third"], vals,
                    color=[th.bad, th.accent, th.good], width=0.6)
            for i, v in enumerate(vals):
                axb.text(i, v + 0.3, f"{v}  ({v / bo['count'] * 100:.0f}%)", ha="center", va="bottom", fontsize=9, color=th.text)
            axb.set_ylim(0, max(vals) * 1.3 + 1)
            axb.set_yticks([])
            axb.tick_params(axis="x", labelsize=9, colors=th.text, length=0)
            for s_ in ("top", "right", "left"):
                axb.spines[s_].set_visible(False)
            axb.spines["bottom"].set_color(th.grid)
            axb.set_title(f"BUILD-UP POSSESSIONS ({bo['count']}) — HOW FAR DID THEY GET?  ·  avg {bo['avg_passes']:.1f} passes, "
                          f"{bo['avg_duration']:.0f}s", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        else:
            axb.set_axis_off()
    return fig


def phases_out_of_possession(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        tab = P.out_of_possession_table(md, team, half)
        ev = _scope(md, team, half, None)
        d = ev[ev["def_phase"] != ""]
        fig = _figure(th, (16, 9.6), f"{team} — Phases of play · out of possession",
                      _subtitle(md, team, half, None, "phase = where the defensive action happens (x)"),
                      f"Dots = defensive actions · ◆ = counter-press (≤{md.cp_window:.0f} s after loss)   |   Low block < {md.thirds[0]:.0f} m ≤ Mid block < {md.thirds[1]:.0f} m ≤ High press")
        ax = fig.add_axes([0.02, 0.37, 0.62, 0.5])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        cols = [th.bad, th.accent, th.good]     # low, mid, high
        _bands(ax, th, md, labels=("LOW BLOCK", "MID BLOCK", "HIGH PRESS"), alpha=0.09, colors=cols)
        cp = d["counter_press"]
        pitch.scatter(d.loc[~cp, "x"], d.loc[~cp, "y"], s=40, color=th.team, alpha=0.7, edgecolors=th.bg, linewidths=0.4, ax=ax, zorder=3)
        pitch.scatter(d.loc[cp, "x"], d.loc[cp, "y"], s=80, color=th.accent, marker="D", edgecolors=th.bg, linewidths=0.6, ax=ax, zorder=4)
        t1, t2 = md.thirds
        order = {"Low block": 0, "Mid block": 1, "High press": 2}
        spans = ((0, t1), (t1, t2), (t2, PITCH_L))
        for _, r in tab.iterrows():
            i = order[r["Phase"]]
            a, b = spans[i]
            ax.text((a + b) / 2, PITCH_W / 2 + 2, f"{r['Share %']:.0f}%", ha="center", va="center", fontsize=26,
                    fontweight="bold", color=cols[i], zorder=6)
            ax.text((a + b) / 2, PITCH_W / 2 - 6, "of def. actions", ha="center", va="center", fontsize=8.5, color=th.muted, zorder=6)
        if len(d):
            h = d["x"].mean()
            ax.plot([h, h], [0, PITCH_W], color=th.text, lw=1.6, ls=(0, (6, 3)), zorder=5, alpha=0.8)
            ax.text(h + 0.8, 1.2, f"avg height {h:.1f} m", fontsize=8, color=th.text, zorder=6)
        _attack_arrow(ax, th)
        for _, r in tab.iterrows():
            i = order[r["Phase"]]
            items = [("Regains", _fmt(r["Regains"])), ("Interceptions", _fmt(r["Interceptions"])),
                     ("Tackles", _fmt(r["Tackles"])), ("Duels", _fmt(r["Duels"])), ("Fouls", _fmt(r["Fouls"])),
                     ("Counter-press", _fmt(r["Counter-press"]))]
            _kpis(fig, th, [0.66 + i * 0.11, 0.37, 0.105, 0.5], items, title=r["Phase"], accent=cols[i])
        k = P.defending_kpis(md, team, half)
        tr = P.transitions(md, team, half)
        items = [("PPDA", _fmt(k["PPDA"], 1)), ("Def. action height", _fmt(k["Def. action height (m)"], 1) + " m"),
                 (f"Regained ≤{md.cp_window:.0f} s after loss", f"{k['Regained in window']}/{k['Losses']}"),
                 ("Opp. box entries allowed", _fmt(k["Opp. box entries allowed"])),
                 ("Opp. shots allowed", _fmt(k["Opp. shots allowed"])), ("Opp. xG allowed", _fmt(k["Opp. xG allowed"], 2))]
        _kpis(fig, th, [0.04, 0.06, 0.44, 0.24], items, title="Defensive control", cols=3)
        if tr:
            items = [("Regains → possessions", _fmt(tr["regains"])),
                     ("High / mid / low regains", f"{tr['high_regains']} / {tr['mid_regains']} / {tr['low_regains']}"),
                     ("Regain → final third", _fmt(tr["to_final_third"])), ("Regain → box", _fmt(tr["to_box"])),
                     ("Regain → shot", _fmt(tr["shots"])), ("Regain → goal", _fmt(tr["goals"]))]
            _kpis(fig, th, [0.52, 0.06, 0.44, 0.24], items, title="Attacking transition (after regain)", cols=3,
                  accent=th.good)
    return fig


def phase_flow(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    """Where possessions start (by phase) vs the furthest phase they reach — a phase transition matrix."""
    th = theme
    with plt.rc_context(_rc(th)):
        poss = md.possessions
        poss = poss[poss["team"] == team] if not poss.empty else poss
        if half is not None and not poss.empty:
            poss = poss[poss["half"] == half]
        fig = _figure(th, (14, 8.2), f"{team} — Possession phase flow", _subtitle(md, team, half, None),
                      "Rows: phase where the possession started · columns: furthest phase reached / outcome")
        ax = fig.add_axes([0.08, 0.12, 0.52, 0.66])
        names = ["Build-up", "Progression", "Final third"]
        if not poss.empty:
            start = pd.Series(M.third_of(poss["start_x"], md.thirds)).map(dict(zip(M.THIRD_NAMES, names))).to_numpy()
            reach = pd.Series(M.third_of(poss["max_x"], md.thirds)).map(dict(zip(M.THIRD_NAMES, names))).to_numpy()
            cols = names + ["Box", "Shot"]
            mat = np.zeros((3, 5))
            for i, s_ in enumerate(names):
                m = start == s_
                for j, r_ in enumerate(names):
                    mat[i, j] = ((reach == r_) & m).sum()
                mat[i, 3] = (poss["reached_box"].to_numpy() & m).sum()
                mat[i, 4] = (poss["shot"].to_numpy() & m).sum()
            rowsum = np.array([(start == s_).sum() for s_ in names], float)
            share = np.divide(mat, rowsum[:, None], out=np.zeros_like(mat), where=rowsum[:, None] > 0)
            ax.imshow(share, cmap=th.cmap(), vmin=0, vmax=1, aspect="auto")
            for i in range(3):
                for j in range(5):
                    ax.text(j, i, f"{int(mat[i, j])}\n{share[i, j] * 100:.0f}%", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=th.bg if share[i, j] > 0.55 else th.text)
            ax.set_xticks(range(5), ["Furthest: build-up", "Furthest: progression", "Furthest: final third", "Reached box", "Shot"],
                          fontsize=9, color=th.text)
            ax.set_yticks(range(3), [f"Started in {n.lower()}\n({int(rowsum[i])} poss.)" for i, n in enumerate(names)],
                          fontsize=9.5, color=th.text)
            ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            # start type / end reason panels
            axs = fig.add_axes([0.69, 0.5, 0.28, 0.3])
            st = poss["start_type"].value_counts()
            _hbar(axs, th, list(st.index), st.values, st.values, th.team, "How possessions started", pct=False)
            axe = fig.add_axes([0.69, 0.12, 0.28, 0.28])
            er = poss["end_reason"].value_counts()
            _hbar(axe, th, list(er.index), er.values, er.values, th.opp, "How possessions ended", pct=False)
        else:
            ax.set_axis_off()
    return fig


# ======================================================================== RHYTHM & SEQUENCES
def momentum(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, window: float = 5.0) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        ev = _all_scope(md, half)
        opp = md.opponent(team)
        fig = _figure(th, (16, 9), f"{team} vs {opp} — Match momentum", _subtitle(md, team, half, None),
                      f"Bars: final-third actions difference per 3 min · lines: passes per minute ({window:.0f}-min rolling)")
        ax1 = fig.add_axes([0.06, 0.5, 0.9, 0.34])
        ax2 = fig.add_axes([0.06, 0.1, 0.9, 0.3])
        mmax = float(np.ceil(ev["minute"].max())) if len(ev) else 90
        edges = np.arange(0, mmax + 3, 3)
        on = ev["etype"].isin(["pass", "carry", "dribble", "shot", "receive"]) & (ev["x"] >= md.thirds[1])
        a, _ = np.histogram(ev.loc[on & (ev["team"] == team), "minute"], bins=edges)
        b, _ = np.histogram(ev.loc[on & (ev["team"] == opp), "minute"], bins=edges)
        diff = a - b
        ax1.bar(edges[:-1] + 1.5, np.where(diff > 0, diff, 0), width=2.6, color=th.team)
        ax1.bar(edges[:-1] + 1.5, np.where(diff < 0, diff, 0), width=2.6, color=th.opp)
        ax1.axhline(0, color=th.line, lw=1)
        lim = max(abs(diff).max() if len(diff) else 1, 1) * 1.35
        ax1.set_ylim(-lim, lim)
        ax1.text(0.005, 0.97, f"▲ {team} on top", transform=ax1.transAxes, color=th.team, fontsize=9, fontweight="bold", va="top")
        ax1.text(0.005, 0.03, f"▼ {opp} on top", transform=ax1.transAxes, color=th.opp, fontsize=9, fontweight="bold", va="bottom")
        shots = ev[ev["etype"] == "shot"]
        for _, s in shots.iterrows():
            up = s["team"] == team
            y = lim * 0.8 * (1 if up else -1)
            mk = "*" if s["shot_goal"] else "o"
            ax1.scatter(s["minute"], y, s=220 if s["shot_goal"] else 38, marker=mk,
                        color=th.accent if s["shot_goal"] else (th.team if up else th.opp), edgecolors=th.bg, zorder=4)
        # passes per minute
        grid = np.arange(0, mmax + 1, 0.5)
        for tm, col in ((team, th.team), (opp, th.opp)):
            t = ev[(ev["team"] == tm) & (ev["etype"] == "pass")]["minute"].to_numpy()
            rate = np.array([((t > g - window / 2) & (t <= g + window / 2)).sum() / window for g in grid])
            ax2.plot(grid, rate, color=col, lw=2.2, label=f"{tm} passes / min")
            ax2.fill_between(grid, rate, color=col, alpha=0.12)
        ax2.legend(frameon=False, fontsize=9, labelcolor=th.muted, loc="upper right")
        for axx in (ax1, ax2):
            axx.set_xlim(0, mmax)
            for s_ in ("top", "right"):
                axx.spines[s_].set_visible(False)
            axx.spines["left"].set_color(th.grid)
            axx.spines["bottom"].set_color(th.grid)
            axx.tick_params(labelsize=8.5)
            axx.grid(axis="x", color=th.grid, lw=0.6)
            if half is None:
                axx.axvline(45, color=th.muted, ls=(0, (4, 3)), lw=1)
        ax1.set_title("FINAL-THIRD PRESSURE  (★ goal · ● shot)", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        ax2.set_title("PASSING RHYTHM", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        ax2.set_xlabel("Minute", color=th.muted)
    return fig


def sequences(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        poss = md.possessions
        poss = poss[poss["team"] == team] if not poss.empty else poss
        if half is not None and not poss.empty:
            poss = poss[poss["half"] == half]
        fig = _figure(th, (16, 9), f"{team} — Possession sequences", _subtitle(md, team, half, None),
                      "A sequence ends on a turnover, a dead ball or a shot")
        if poss.empty:
            return fig
        ax1 = fig.add_axes([0.05, 0.5, 0.28, 0.33])
        bins = [0, 1, 3, 6, 10, 100]
        lab = ["0", "1–2", "3–5", "6–9", "10+"]
        cnt = pd.cut(poss["n_passes"], bins=[-1] + bins[1:], labels=lab, right=False).value_counts().reindex(lab).fillna(0)
        ax1.bar(lab, cnt.values, color=th.team, width=0.65)
        for i, v in enumerate(cnt.values):
            ax1.text(i, v + 0.5, str(int(v)), ha="center", fontsize=9, color=th.text)
        ax1.set_title("PASSES PER SEQUENCE", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        ax2 = fig.add_axes([0.39, 0.5, 0.28, 0.33])
        third_cnt = poss["reach_third"].value_counts().reindex(M.THIRD_NAMES).fillna(0)
        ax2.bar(["Defensive", "Middle", "Final"], third_cnt.values, color=[th.bad, th.accent, th.good], width=0.6)
        for i, v in enumerate(third_cnt.values):
            ax2.text(i, v + 0.5, str(int(v)), ha="center", fontsize=9, color=th.text)
        ax2.set_title("FURTHEST THIRD REACHED", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        for axx in (ax1, ax2):
            for s_ in ("top", "right", "left"):
                axx.spines[s_].set_visible(False)
            axx.spines["bottom"].set_color(th.grid)
            axx.set_yticks([])
            axx.tick_params(labelsize=9, colors=th.text, length=0)
        ax3 = fig.add_axes([0.74, 0.5, 0.22, 0.33])
        pl = pd.Series([p for ps in poss["players"] for p in ps if p != "0"]).value_counts().head(8)
        _hbar(ax3, th, [f"#{i}" for i in pl.index], pl.values, pl.values, th.team, "Most involved", pct=False)
        long = poss[poss["n_passes"] >= 3]
        items = [("Sequences", _fmt(len(poss))), ("3+ pass sequences", _fmt(len(long))),
                 ("Avg passes", _fmt(poss["n_passes"].mean(), 1)), ("Avg duration", _fmt(poss["duration"].mean(), 1) + " s"),
                 ("Longest (passes)", _fmt(poss["n_passes"].max())), ("Reached final third", _fmt(int(poss["reached_f3"].sum()))),
                 ("Reached box", _fmt(int(poss["reached_box"].sum()))), ("Ended in shot", _fmt(int(poss["shot"].sum())))]
        _kpis(fig, th, [0.05, 0.08, 0.91, 0.3], items, title="Sequence summary", cols=4)
    return fig


def team_comparison(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        opp = md.opponent(team)
        a = M.team_summary(md, team, half)
        b = M.team_summary(md, opp, half) if opp else {}
        keys = ["Goals", "xG", "Shots", "On target", "Possession % (passes)", "Field tilt %", "Passes", "Pass accuracy %",
                "Progressive passes", "Progressive carries", "Final-third entries", "Box entries", "Line-breaking passes",
                "PPDA", "Def. action height (m)", "Recoveries", "Interceptions"]
        fig = _figure(th, (12, 11), f"{team}  vs  {opp}", _subtitle(md, team, half, None, "match statistics"))
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.78])
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.5, len(keys) - 0.3)
        ax.set_axis_off()
        ax.text(-1.0, len(keys) - 0.4, team, ha="left", fontsize=14, fontweight="bold", color=th.team)
        ax.text(1.0, len(keys) - 0.4, opp, ha="right", fontsize=14, fontweight="bold", color=th.opp)
        for i, k in enumerate(keys):
            y = len(keys) - 1 - i
            va, vb = a.get(k, np.nan), b.get(k, np.nan)
            fa = float(va) if not isinstance(va, str) else np.nan
            fb = float(vb) if not isinstance(vb, str) else np.nan
            tot = (abs(fa) if not np.isnan(fa) else 0) + (abs(fb) if not np.isnan(fb) else 0)
            sa = (fa / tot) if tot else 0
            sb = (fb / tot) if tot else 0
            if k == "PPDA" and tot:       # lower PPDA = more intense: invert the bar share
                sa, sb = sb, sa
            ax.barh(y, -sa * 0.72, left=-0.2, height=0.42, color=th.team, alpha=0.9)
            ax.barh(y, sb * 0.72, left=0.2, height=0.42, color=th.opp, alpha=0.9)
            ax.barh(y, -0.72, left=-0.2, height=0.42, color=th.grid, alpha=0.35, zorder=0)
            ax.barh(y, 0.72, left=0.2, height=0.42, color=th.grid, alpha=0.35, zorder=0)
            dec = 2 if k == "xG" else 1 if ("%" in k or k in ("PPDA", "Def. action height (m)")) else 0
            ax.text(-0.2 - 0.76, y, _fmt(va, dec), ha="right", va="center", fontsize=10.5, fontweight="bold", color=th.text)
            ax.text(0.2 + 0.76, y, _fmt(vb, dec), ha="left", va="center", fontsize=10.5, fontweight="bold", color=th.text)
            ax.text(0, y, k.replace(" (passes)", ""), ha="center", va="center", fontsize=9, color=th.muted)
    return fig


def player_dashboard(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK) -> plt.Figure:
    th = theme
    with plt.rc_context(_rc(th)):
        if not player:
            fig = _figure(th, (14, 8), f"{team} — Player dashboard", "Pick a player to see their dashboard")
            return fig
        ev = _scope(md, team, half, player)
        p = ev[(ev["etype"] == "pass") & ev["end_x"].notna()]
        fig = _figure(th, (16, 9.4), f"#{player} — Player dashboard", _subtitle(md, team, half, player))
        pitch = _pitch(th, line_zorder=3)
        ax1 = fig.add_axes([0.02, 0.42, 0.46, 0.42])
        pitch.draw(ax=ax1)
        ok, ko = p[p["outcome"] == 1], p[p["outcome"] != 1]
        if len(ko):
            pitch.lines(ko["x"], ko["y"], ko["end_x"], ko["end_y"], comet=True, transparent=True, color=th.bad, lw=1.8, alpha_end=0.55, ax=ax1)
        if len(ok):
            pitch.lines(ok["x"], ok["y"], ok["end_x"], ok["end_y"], comet=True, transparent=True,
                        color=th.team, lw=2.2, ax=ax1)
            pitch.scatter(ok["end_x"], ok["end_y"], s=18, color=th.team, ax=ax1, zorder=4)
        ax1.set_title("PASSES", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        ax2 = fig.add_axes([0.5, 0.42, 0.46, 0.42])
        pitch.draw(ax=ax2)
        t = ev[ev["etype"].isin(["pass", "receive", "carry", "dribble", "shot", "recovery", "interception", "tackle", "duel"])
               & ev["x"].notna() & ~ev["derived"]]
        if len(t) >= 3:
            from scipy.ndimage import gaussian_filter
            st = pitch.bin_statistic(t["x"], t["y"], statistic="count", bins=(30, 20), normalize=True)
            st["statistic"] = gaussian_filter(st["statistic"], 1.4)
            pitch.heatmap(st, ax=ax2, cmap=th.cmap(), edgecolors="none", zorder=0.6)
        d = ev[ev["etype"].isin(["interception", "tackle", "recovery", "duel", "block", "clearance"]) & ev["x"].notna()]
        pitch.scatter(d["x"], d["y"], s=34, marker="D", color=th.accent, edgecolors=th.bg, ax=ax2, zorder=5)
        ax2.set_title("TOUCHES  (◆ defensive actions)", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        tab = M.player_table(md, team, half)
        row = tab[tab["Player"] == f"#{player}"]
        r = row.iloc[0] if len(row) else None
        keys = ["Actions", "Passes", "Acc %", "Prog. passes", "Prog. carries", "Prog. received", "F3 entries", "Box entries",
                "Line breaks", "Key passes", "Crosses", "Shots", "xG", "Def. actions", "Recoveries", "Losses"]
        items = [(k, _fmt(r[k], 1 if k in ("Acc %",) else 2 if k == "xG" else 0) if r is not None else "–") for k in keys]
        _kpis(fig, th, [0.03, 0.06, 0.94, 0.3], items, cols=8)
    return fig


# ======================================================================== sequences (action by action)
_SEQ_ACTIONS = ("pass", "carry", "dribble", "shot", "recovery", "interception", "tackle", "receive", "duel", "foul")


def sequence_events(md: MatchData, poss_id: int) -> pd.DataFrame:
    """The possession's own actions (plus the opponent action that ended it), in order."""
    ev = md.events
    g = ev[(ev["poss_id"] == poss_id) & ev["x"].notna()]
    if g.empty:
        return g
    team = g["poss_team"].iloc[0]
    own = g[(g["team"] == team) & g["etype"].isin(_SEQ_ACTIONS)]
    nxt = ev[(ev["poss_id"] == poss_id + 1) & (ev["team"] != team) & ev["x"].notna()].head(1)
    return pd.concat([own, nxt])


def _action_text(e, names: dict | None = None) -> str:
    who = f"#{e['player']}" if e["player"] and e["player"] != "0" else ""
    if names and e["player"] in names:
        who = f"#{e['player']} {names[e['player']]}"
    et = e["etype"]
    if et == "pass":
        to = f" → #{e['receiver']}" if e.get("receiver") and e["receiver"] not in ("", "0") else ""
        kind = e["pass_type"] if e["pass_type"] and e["pass_type"] != "Regular" else "Pass"
        ok = "✓" if e["outcome"] == 1 else "✗" if e["outcome"] == 0 else "?"
        ln = f" {e['length']:.0f} m" if pd.notna(e.get("length")) else ""
        prog = " · progressive" if e.get("progressive") else ""
        return f"{who} {kind}{to} {ok} {str(e['direction']).lower()}{ln}{prog}".strip()
    if et == "carry":
        return f"{who} Carry {e['length']:.0f} m{' · progressive' if e.get('progressive') else ''}".strip()
    if et == "shot":
        return f"{who} SHOT — {e['shot_result'] or 'no outcome tagged'} (xG {e['xg']:.2f})".strip()
    if et == "dribble":
        return f"{who} Dribble {'✓' if e['outcome'] == 1 else '✗' if e['outcome'] == 0 else ''}".strip()
    if et in ("recovery", "interception", "tackle"):
        return f"{who} {et.title()} (regain)".strip()
    return f"{who} {et.replace('_', ' ').title()}".strip()


def _draw_sequence(ax, pitch, th: Theme, seq: pd.DataFrame, team: str, numbered: bool = True, small: bool = False) -> list:
    """Draw every action of a sequence. Returns the numbered actions (for the action list)."""
    steps = []
    lw = 1.3 if small else 2.4
    hw = 3.5 if small else 5
    own = seq[seq["team"] == team]
    for _, e in own.iterrows():
        et = e["etype"]
        if et == "receive":
            pitch.scatter(e["x"], e["y"], s=10 if small else 26, color=th.text, alpha=0.6, ax=ax, zorder=4)
            continue
        steps.append(e)
        n = len(steps)
        if et == "pass" and pd.notna(e["end_x"]):
            col = th.team if e["outcome"] == 1 else th.bad
            pitch.arrows(e["x"], e["y"], e["end_x"], e["end_y"], color=col, width=lw, headwidth=hw, headlength=hw,
                         ax=ax, zorder=5, alpha=0.95)
        elif et == "carry" and pd.notna(e["end_x"]):
            pitch.lines(e["x"], e["y"], e["end_x"], e["end_y"], color=th.carry, lw=lw, ls=(0, (2, 2)), ax=ax, zorder=5)
        elif et == "shot":
            pitch.scatter(e["x"], e["y"], s=90 if small else 380, marker="*", color=th.accent, edgecolors=th.bg, ax=ax, zorder=7)
        elif et == "dribble":
            pitch.scatter(e["x"], e["y"], s=24 if small else 110, marker="D", color=th.good if e["outcome"] == 1 else th.bad,
                          edgecolors=th.bg, ax=ax, zorder=6)
        elif et in ("recovery", "interception", "tackle"):
            pitch.scatter(e["x"], e["y"], s=24 if small else 120, marker="s", color=th.good, edgecolors=th.bg, ax=ax, zorder=6)
        if numbered and not small:
            ax.annotate(str(n), (e["x"], e["y"]), xytext=(-9, 9), textcoords="offset points", fontsize=8.5,
                        fontweight="bold", color=th.bg, ha="center", va="center", zorder=9,
                        bbox=dict(boxstyle="circle,pad=0.25", fc=th.text, ec="none"))
            if e["player"] and e["player"] != "0":
                ax.annotate(f"#{e['player']}", (e["x"], e["y"]), xytext=(10, -10), textcoords="offset points",
                            fontsize=7.5, color=th.muted, zorder=9)
    opp = seq[seq["team"] != team]
    if len(opp):
        o = opp.iloc[0]
        ox, oy = PITCH_L - o["x"], PITCH_W - o["y"]                     # their frame → ours
        pitch.scatter(ox, oy, s=40 if small else 160, marker="X", color=th.bad, edgecolors=th.bg, ax=ax, zorder=8)
    return steps


def buildup_sequences(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
                      min_passes: int = 3, max_panels: int = 12) -> plt.Figure:
    """Small multiples of build-up sequences (start in own third, ≥ min_passes), every action drawn."""
    th = theme
    with plt.rc_context(_rc(th)):
        poss = md.possessions
        sel = poss[(poss["team"] == team) & (poss["start_x"] < md.thirds[0]) & (poss["n_passes"] >= min_passes)] if not poss.empty else poss
        if half is not None and not sel.empty:
            sel = sel[sel["half"] == half]
        sel = sel.sort_values(["shot", "reached_f3", "n_passes"], ascending=False).head(max_panels) if not sel.empty else sel
        fig = _figure(th, (16, 10.4), f"{team} — Build-up sequences",
                      _subtitle(md, team, half, None, f"started in own third, ≥{min_passes} passes · best {len(sel)} shown"),
                      "Arrow = pass (blue ok / red lost) · dashed = carry · ■ regain · ◆ dribble · ★ shot · ✕ where the opponent won it")
        cols = 4
        rows = max(1, int(np.ceil(len(sel) / cols)))
        pitch = _pitch(th)
        w, h = 0.92 / cols, 0.8 / rows
        for k, (_, s) in enumerate(sel.iterrows()):
            r, c = divmod(k, cols)
            ax = fig.add_axes([0.04 + c * w, 0.86 - (r + 1) * h, w * 0.94, h * 0.84])
            pitch.draw(ax=ax)
            seq = sequence_events(md, s["poss_id"])
            _draw_sequence(ax, pitch, th, seq, team, numbered=False, small=True)
            col = th.good if s["shot"] else th.accent if s["reached_f3"] else th.bad
            ax.set_title(f"#{s['poss_id']}  ·  {s['start_min']:.0f}'  ·  {s['n_passes']} passes  ·  {s['duration']:.0f}s  ·  {s['end_reason']}",
                         fontsize=8, color=col, pad=2, fontweight="bold")
            _set_panel(fig, f"Sequence #{s['poss_id']}", ax)
        if sel.empty:
            fig.text(0.5, 0.5, "No build-up sequences for this selection", ha="center", color=th.muted, fontsize=12)
    return fig


def sequence_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, poss_id: int | None = None,
                 names: tuple = ()) -> plt.Figure:
    """One possession, action by action: numbered steps on the pitch + the action list."""
    th = theme
    nm = dict(names)
    with plt.rc_context(_rc(th)):
        poss = md.possessions
        mine = poss[poss["team"] == team] if not poss.empty else poss
        if poss_id is None and not mine.empty:
            poss_id = int(mine.sort_values(["shot", "n_passes"], ascending=False).iloc[0]["poss_id"])
        row = poss[poss["poss_id"] == poss_id] if not poss.empty else poss
        if row.empty:
            return _figure(th, (16, 9), f"{team} — Sequence", "No sequence selected")
        s = row.iloc[0]
        seq = sequence_events(md, int(poss_id))
        fig = _figure(th, (17, 9.2), f"{team} — Sequence #{poss_id}",
                      f"{s['start_min']:.0f}'  ·  started: {s['start_type']}  ·  {s['n_passes']} passes  ·  "
                      f"{s['duration']:.0f} s  ·  ended: {s['end_reason']}  ·  furthest: {s['reach_third']}",
                      "Numbers = order of actions · arrow = pass · dashed = carry · ■ regain · ◆ dribble · ★ shot · ✕ opponent wins it")
        ax = fig.add_axes([0.02, 0.1, 0.64, 0.76])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        steps = _draw_sequence(ax, pitch, th, seq, team, numbered=True)
        _attack_arrow(ax, th)
        _set_panel(fig, "Pitch", ax)
        axl = fig.add_axes([0.68, 0.1, 0.3, 0.76])
        axl.set_axis_off()
        axl.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", fc=th.panel, ec=th.grid,
                                     transform=axl.transAxes))
        axl.text(0.05, 0.97, "ACTIONS", fontsize=8.5, color=th.muted, fontweight="bold", va="top", transform=axl.transAxes)
        lines = [(f"{i}", f"{int(e['minute'])}'{int(round((e['minute'] % 1) * 60)):02d}", _action_text(e, nm))
                 for i, e in enumerate(steps, 1)]
        opp = seq[seq["team"] != team]
        if len(opp):
            o = opp.iloc[0]
            lines.append(("✕", f"{int(o['minute'])}'{int(round((o['minute'] % 1) * 60)):02d}",
                          f"{o['team']} {o['etype'].replace('_', ' ')} — possession lost"))
        step = min(0.06, 0.9 / max(len(lines), 1))
        fs = 9 if step >= 0.045 else 7.5
        for k, (n, tm, txt) in enumerate(lines):
            y = 0.92 - k * step
            axl.text(0.05, y, n, fontsize=fs, fontweight="bold", color=th.accent, va="top", transform=axl.transAxes)
            axl.text(0.12, y, tm, fontsize=fs, color=th.muted, va="top", transform=axl.transAxes)
            axl.text(0.27, y, txt[:52], fontsize=fs, color=th.text, va="top", transform=axl.transAxes)
        _set_panel(fig, "Action list", axl)
    return fig


def sequence_table(md: MatchData, poss_id: int, names: dict | None = None) -> pd.DataFrame:
    seq = sequence_events(md, int(poss_id))
    if seq.empty:
        return pd.DataFrame()
    team = seq["poss_team"].iloc[0]
    rows, n = [], 0
    for _, e in seq.iterrows():
        if e["team"] == team and e["etype"] == "receive":
            continue
        n += 1 if e["team"] == team else 0
        rows.append({"#": str(n) if e["team"] == team else "✕", "Minute": f"{e['minute']:.2f}", "Team": e["team"],
                     "Player": (f"#{e['player']} {(names or {}).get(e['player'], '')}".strip()
                                if e["player"] and e["player"] != "0" else ""),
                     "Action": _action_text(e, None) if e["team"] == team else f"{e['etype'].replace('_', ' ')} (wins the ball)",
                     "Phase": e["phase"], "x": round(float(e["x"]), 1), "y": round(float(e["y"]), 1),
                     "Labels": ", ".join(e["labels_raw"]) if isinstance(e.get("labels_raw"), tuple) else ""})
    return pd.DataFrame(rows)


# ======================================================================== pass maps by half
def pass_map_halves(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK, kind: str = "all",
                    show_failed: bool = True, open_play_only: bool = False) -> plt.Figure:
    label, fn = PASS_FILTERS.get(kind, PASS_FILTERS["all"])
    th = theme
    with plt.rc_context(_rc(th)):
        fig = _figure(th, (17, 7.8), f"{team} — {label}: 1st half vs 2nd half", _subtitle(md, team, None, player),
                      "End points inferred from the next tagged event")
        pitch = _pitch(th)
        for i, h in enumerate((1, 2)):
            ax = fig.add_axes([0.015 + i * 0.495, 0.1, 0.475, 0.66])
            pitch.draw(ax=ax)
            p = _scope(md, team, h, player)
            p = p[(p["etype"] == "pass") & p["end_x"].notna()]
            if open_play_only:
                p = p[p["open_play"]]
            p = p[fn(p).fillna(False).astype(bool)]
            ok, ko = p[p["outcome"] == 1], p[p["outcome"] != 1]
            if show_failed and len(ko):
                pitch.lines(ko["x"], ko["y"], ko["end_x"], ko["end_y"], comet=True, transparent=True, color=th.bad,
                            lw=2, alpha_start=0.02, alpha_end=0.55, ax=ax, zorder=2)
                pitch.scatter(ko["end_x"], ko["end_y"], s=20, marker="X", color=th.bad, alpha=0.75, ax=ax, zorder=3, linewidths=0)
            if len(ok):
                pitch.lines(ok["x"], ok["y"], ok["end_x"], ok["end_y"], comet=True, transparent=True, color=th.team,
                            lw=2.5, alpha_start=0.03, alpha_end=0.95, ax=ax, zorder=4)
                pitch.scatter(ok["end_x"], ok["end_y"], s=26, color=th.team, edgecolors=th.bg, linewidths=0.6, ax=ax, zorder=5)
            if p.empty:
                _empty(ax, th)
            _attack_arrow(ax, th)
            acc = M._pct(len(ok), len(p))
            prog = int(ok["progressive"].sum())
            fwd = M._pct((p["direction"] == "Forward").sum(), len(p))
            ax.set_title(f"{'1ST' if h == 1 else '2ND'} HALF", loc="left", fontsize=12, fontweight="bold", color=th.text, pad=26)
            ax.text(0, PITCH_W + 3.2, f"{len(p)} passes  ·  {_fmt(acc, 0, True)} accurate  ·  {prog} progressive  ·  "
                    f"{_fmt(fwd, 0, True)} forward", fontsize=9.5, color=th.muted, va="bottom")
            _set_panel(fig, f"{'1st' if h == 1 else '2nd'} half", ax)
    return fig


# ======================================================================== counter-press
def counterpress_map(md: MatchData, team: str, *, half=None, player=None, theme: Theme = DARK,
                     window: float | None = None) -> plt.Figure:
    from fap.matchlab import counterpress as CP
    th = theme
    window = float(window or md.cp_window)
    with plt.rc_context(_rc(th)):
        d = CP.analyse(md, team, window, half)
        sm = CP.summary(d)
        fig = _figure(th, (16.5, 9.6), f"{team} — Counter-press after losing the ball",
                      _subtitle(md, team, half, None, f"first {window:.0f} s after each open-play loss"),
                      f"Loss = where the opponent won the ball · pressed = {sm.get('press_source', 'Press tags')} within {window:.0f} s · "
                      f"regained = open-play regain within {window:.0f} s")
        ax = fig.add_axes([0.02, 0.36, 0.6, 0.5])
        pitch = _pitch(th)
        pitch.draw(ax=ax)
        _bands(ax, th, md, alpha=0.03)
        cols = {CP.CATEGORIES[0]: th.good, CP.CATEGORIES[1]: th.accent, CP.CATEGORIES[2]: th.team, CP.CATEGORIES[3]: th.bad}
        if not d.empty:
            rg = d[d["regained"] & d["regain_x"].notna()]
            for _, r in rg.iterrows():
                ax.annotate("", xy=(r["regain_x"], r["regain_y"]), xytext=(r["x"], r["y"]),
                            arrowprops=dict(arrowstyle="-|>", color=th.text, lw=1, alpha=0.45), zorder=3)
            for cat, col in cols.items():
                g = d[d["category"] == cat]
                if len(g):
                    pitch.scatter(g["x"], g["y"], s=110, color=col, edgecolors=th.bg, linewidths=0.8, ax=ax, zorder=5,
                                  marker="o" if "regained" in cat.lower() and "not" not in cat.lower() else "X")
        else:
            _empty(ax, th, "No open-play ball losses")
        _attack_arrow(ax, th)
        _legend(ax, th, [(f"{c} ({int((d['category'] == c).sum()) if not d.empty else 0})", col, "dot")
                         for c, col in cols.items()], y=-0.07, ncol=2)
        _set_panel(fig, "Pitch", ax)
        items = [("Ball losses (open play)", _fmt(sm.get("losses", 0))),
                 (f"Pressed within {window:.0f} s", f"{sm.get('pressed', 0)}  ({_fmt(sm.get('pressed_pct'), 0, True)})"),
                 ("Avg. time to first press", _fmt(sm.get("avg_seconds_to_press"), 1) + " s"),
                 (f"Regained within {window:.0f} s", f"{sm.get('regained', 0)}  ({_fmt(sm.get('regained_pct'), 0, True)})"),
                 ("Regain rate · pressed / not", f"{_fmt(sm.get('regain_rate_when_pressed'), 0, True)} / {_fmt(sm.get('regain_rate_without_press'), 0, True)}"),
                 ("Regains → shot", _fmt(sm.get("regain_to_shot", 0))),
                 ("Opp. reached our final third", _fmt(sm.get("opp_reached_final_third", 0))),
                 ("Opp. shots after our loss", _fmt(sm.get("opp_shots", 0)))]
        kax = _kpis(fig, th, [0.65, 0.36, 0.32, 0.5], items, title="Counter-press", cols=2)
        _set_panel(fig, "Summary", kax)
        axh = fig.add_axes([0.05, 0.07, 0.4, 0.2])
        if not d.empty and d["pressed"].any():
            bins = np.arange(0, window + 1, 1)
            pr = d[d["pressed"]]
            a, _ = np.histogram(pr.loc[pr["regained"], "seconds_to_press"], bins=bins)
            b, _ = np.histogram(pr.loc[~pr["regained"], "seconds_to_press"], bins=bins)
            xs = bins[:-1] + 0.5
            axh.bar(xs, a, width=0.8, color=th.good, label="then regained")
            axh.bar(xs, b, width=0.8, bottom=a, color=th.accent, label="not regained")
            axh.set_xticks(bins)
            axh.legend(frameon=False, fontsize=8, labelcolor=th.muted)
        axh.set_xlabel("seconds after the loss", color=th.muted, fontsize=8.5)
        axh.set_title("TIME TO FIRST PRESS", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        for s_ in ("top", "right"):
            axh.spines[s_].set_visible(False)
        axh.tick_params(labelsize=8)
        _set_panel(fig, "Time to press", axh)
        axt = fig.add_axes([0.55, 0.07, 0.4, 0.2])
        bt = CP.by_third(d)
        if not bt.empty:
            x = np.arange(len(bt))
            axt.bar(x - 0.2, bt["pressed %"], width=0.38, color=th.accent, label="pressed %")
            axt.bar(x + 0.2, bt["regained %"], width=0.38, color=th.good, label="regained %")
            for xi, (_, r) in zip(x, bt.iterrows()):
                axt.text(xi, max(r["pressed %"], r["regained %"]) + 3, f"{int(r['losses'])} losses", ha="center",
                         fontsize=8, color=th.muted)
            axt.set_xticks(x, [t.replace(" third", "") for t in bt.index], fontsize=9)
            axt.set_ylim(0, 110)
            axt.legend(frameon=False, fontsize=8, labelcolor=th.muted, loc="upper right")
        axt.set_title("BY ZONE OF THE LOSS", loc="left", fontsize=9, color=th.muted, fontweight="bold")
        for s_ in ("top", "right"):
            axt.spines[s_].set_visible(False)
        axt.tick_params(labelsize=8)
        _set_panel(fig, "By zone", axt)
    return fig


# ======================================================================== panels (download one part of a chart)
def _set_panel(fig, name: str, *axes) -> None:
    panels = getattr(fig, "_ml_panels", None)
    if panels is None:
        panels = {}
        fig._ml_panels = panels
    panels.setdefault(name, []).extend(a for a in axes if a is not None)


# Charts built before panels existed: name their axes by creation order.
_PANEL_INDEX = {
    "pass_map": {"Pitch": [0], "Summary": [1], "Top players": [2]},
    "progression": {"Pitch": [0], "Summary": [1], "Top progressors": [2]},
    "pass_profile": {"Direction rose": [0], "Pass type": [1], "Direction": [2], "Length": [3], "Height": [4], "Line broken": [5]},
    "pass_sonars": {"Pitch": [0, 1]},
    "pass_network": {"Pitch": [0], "Top combinations": [1]},
    "heatmap": {"Pitch": [0], "Share": [1]},
    "entries": {"Pitch": [0], "Entries by channel": [1], "Totals": [2]},
    "crosses": {"Pitch": [0], "Summary": [1], "By player": [2]},
    "shots": {"Pitch": [0], "Summary": [1]},
    "defensive": {"Pitch": [0], "Summary": [1]},
    "losses": {"Pitch": [0], "By zone": [1]},
    "phases_in": {"Pitch": [0], "Phase cards": [1, 2, 3], "Possession funnel": [4], "Build-up outcomes": [5]},
    "phases_out": {"Pitch": [0], "Phase cards": [1, 2, 3], "Defensive control": [4], "Attacking transition": [5]},
    "phase_flow": {"Phase matrix": [0], "How possessions started": [1], "How possessions ended": [2]},
    "momentum": {"Final-third pressure": [0], "Passing rhythm": [1]},
    "sequences": {"Passes per sequence": [0], "Furthest third": [1], "Most involved": [2], "Summary": [3]},
    "player": {"Passes": [0], "Touches": [1], "Stats": [2]},
}


def panels_of(fig) -> list[str]:
    return list(getattr(fig, "_ml_panels", {}) or {})


def panel_png(fig, name: str, dpi: int = 300) -> bytes:
    """PNG of ONE part of a dashboard chart, with its own title — the rest of the figure hidden."""
    import io
    from matplotlib.transforms import Bbox
    axes = fig._ml_panels[name]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    keep = set(axes)
    for a in fig.axes:
        a.set_visible(a in keep)
    for art in list(fig.texts) + list(fig.artists) + list(fig.lines) + list(fig.patches):
        art.set_visible(False)
    bb = Bbox.union([a.get_tightbbox(r) for a in axes])
    W, H = fig.bbox.width, fig.bbox.height
    title = getattr(fig, "_ml_title", "")
    t1 = fig.text(bb.x0 / W, (bb.y1 + 0.42 * fig.dpi) / H, f"{title}  —  {name}" if title else name, ha="left", va="bottom",
                  fontsize=14, fontweight="bold", color=getattr(fig, "_ml_text", "#000"))
    t2 = fig.text(bb.x0 / W, (bb.y1 + 0.14 * fig.dpi) / H, getattr(fig, "_ml_subtitle", ""), ha="left", va="bottom",
                  fontsize=9, color=getattr(fig, "_ml_muted", "#666"))
    fig.canvas.draw()
    full = Bbox.union([bb, t1.get_window_extent(r), t2.get_window_extent(r)])
    inches = full.transformed(fig.dpi_scale_trans.inverted()).padded(0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches=inches)
    plt.close(fig)
    return buf.getvalue()


# ======================================================================== catalogue
@dataclass(frozen=True)
class ChartSpec:
    key: str
    label: str
    group: str
    fn: Callable
    needs_player: bool = False


CHARTS: list[ChartSpec] = [
    ChartSpec("comparison", "Match statistics", "Overview", team_comparison),
    ChartSpec("momentum", "Match momentum", "Overview", momentum),
    ChartSpec("pass_map", "Pass map", "Passing", pass_map),
    ChartSpec("pass_map_halves", "Pass map · 1st vs 2nd half", "Passing", pass_map_halves),
    ChartSpec("pass_profile", "Pass types & directions", "Passing", pass_profile),
    ChartSpec("pass_sonars", "Pass sonars by zone", "Passing", pass_sonars),
    ChartSpec("pass_flow", "Pass flow", "Passing", pass_flow),
    ChartSpec("pass_network", "Passing network", "Passing", pass_network),
    ChartSpec("passing_matrix", "Passing matrix", "Passing", passing_matrix),
    ChartSpec("progression", "Ball progression", "Progression", progression_map),
    ChartSpec("entries", "Final-third & box entries", "Progression", entries_map),
    ChartSpec("crosses", "Crosses", "Progression", crosses_map),
    ChartSpec("shots", "Shot map", "Progression", shot_map),
    ChartSpec("phases_in", "Phases · in possession", "Phases", phases_in_possession),
    ChartSpec("phases_out", "Phases · out of possession", "Phases", phases_out_of_possession),
    ChartSpec("phase_flow", "Possession phase flow", "Phases", phase_flow),
    ChartSpec("buildup", "Build-up sequences", "Phases", buildup_sequences),
    ChartSpec("sequence", "Sequence explorer", "Phases", sequence_map),
    ChartSpec("sequences", "Possession sequences", "Phases", sequences),
    ChartSpec("counterpress", "Counter-press", "Defending", counterpress_map),
    ChartSpec("defensive", "Defensive actions", "Defending", defensive_map),
    ChartSpec("losses", "Ball losses & regains", "Defending", losses_recoveries),
    ChartSpec("heatmap", "Activity heat map", "Territory", touch_heatmap),
    ChartSpec("player", "Player dashboard", "Players", player_dashboard, needs_player=True),
]
CHART_BY_KEY = {c.key: c for c in CHARTS}


def render(key: str, md: MatchData, team: str, **kw) -> plt.Figure:
    fig = CHART_BY_KEY[key].fn(md, team, **kw)
    if not getattr(fig, "_ml_panels", None):
        for name, idx in _PANEL_INDEX.get(key, {}).items():
            axes = [fig.axes[i] for i in idx if i < len(fig.axes) and fig.axes[i].get_visible()]
            if axes and any(a.has_data() or a.texts or a.patches or a.collections for a in axes):
                _set_panel(fig, name, *axes)
    return fig


def to_png(fig: plt.Figure, dpi: int = 150, panel: str | None = None) -> bytes:
    import io
    if panel:
        return panel_png(fig, panel, dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return buf.getvalue()
