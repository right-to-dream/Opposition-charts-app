"""Professional player-comparison charts (matplotlib, Match Lab themes): percentile profile, radar, side-by-side
per-90 bars, scatter, ranking and player-vs-position. Every chart is a titled card that exports as a PNG."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from fap.matchlab.charts import DARK, FONT, Theme  # noqa: E402
from fap.matchlab.playercompare import CATEGORIES, METRIC_BY_LABEL, METRICS, Metric, fmt  # noqa: E402

PLAYER_COLORS = ("#3ab0ff", "#ff5d73", "#ffc94a", "#2ed8a3")


def _rc(th: Theme):
    return plt.rc_context({"font.family": FONT, "text.color": th.text, "axes.labelcolor": th.muted,
                           "xtick.color": th.muted, "ytick.color": th.muted})


def _card(th: Theme, size, title: str, subtitle: str, foot: str = ""):
    fig = plt.figure(figsize=size, facecolor=th.bg)
    fig.text(0.03, 1 - 0.28 / size[1], title, ha="left", va="top", fontsize=18, fontweight="bold", color=th.text)
    fig.text(0.03, 1 - 0.72 / size[1], subtitle, ha="left", va="top", fontsize=10, color=th.muted)
    fig.add_artist(plt.Line2D([0.03, 0.97], [1 - 1.0 / size[1]] * 2, color=th.grid, lw=1, transform=fig.transFigure))
    fig.text(0.03, 0.15 / size[1], "MATCH LAB · player comparison" + (f"   ·   {foot}" if foot else ""), ha="left",
             va="bottom", fontsize=7.5, color=th.muted, fontweight="bold")
    return fig


def tier_color(p: float, th: Theme) -> str:
    if p != p:
        return th.grid
    return th.bad if p < 33 else th.accent if p < 66 else th.good


def _row(pp: pd.DataFrame, key: str) -> pd.Series:
    return pp[pp["player"] == str(key)].iloc[0]


def _who(r: pd.Series) -> str:
    return r["Player"]


# ------------------------------------------------------------------ percentile profile (scouting-report style)
def percentile_profile(pp: pd.DataFrame, key: str, *, pool_label: str, theme: Theme = DARK,
                       metrics: list[str] | None = None):
    th = theme
    r = _row(pp, key)
    ms = [m for m in METRICS if (metrics is None or m.label in metrics)]
    cats = [c for c in CATEGORIES if any(m.category == c for m in ms)]
    n_rows = len(ms) + len(cats)
    H = 1.6 + 0.34 * n_rows + 0.4
    with _rc(th):
        fig = _card(th, (11, H), f"{_who(r)} — percentile profile",
                    f"{r['Position']}  ·  {int(r['Matches'])} matches ({int(r['Starts'])} starts)  ·  {r['Minutes']:.0f} min  ·  "
                    f"percentile vs {pool_label}" + ("" if r.get("Eligible", True) else "  ·  below the minutes threshold — no percentiles"),
                    "bars = percentile 0–100 · value = per 90 (or %) · red < 33 · amber 33–66 · green > 66")
        ax = fig.add_axes([0.32, 0.4 / H, 0.5, 1 - 1.55 / H])
        y = 0
        ticks, labels = [], []
        for c in cats:
            ax.text(-2, y, c.upper(), ha="right", va="center", fontsize=9.5, fontweight="bold", color=th.accent,
                    transform=ax.transData, clip_on=False)
            ax.plot([-60, 100], [y - 0.5, y - 0.5], color=th.grid, lw=0.6, clip_on=False)
            y -= 1
            for m in [m for m in ms if m.category == c]:
                p = r.get(f"pct:{m.label}", np.nan)
                ax.barh(y, 100, color=th.panel, height=0.62, zorder=1)
                if p == p:
                    ax.barh(y, p, color=tier_color(p, th), height=0.62, zorder=2)
                    ax.text(min(p, 100) + 1.5, y, f"{p:.0f}", va="center", fontsize=9, color=th.text, fontweight="bold")
                ax.text(-2, y, m.label, ha="right", va="center", fontsize=9.5, color=th.text, clip_on=False)
                ax.text(118, y, fmt(m, r.get(m.column)), ha="right", va="center", fontsize=9.5, color=th.text,
                        clip_on=False, fontweight="bold")
                y -= 1
        ax.text(118, 0.9, "PER 90", ha="right", va="center", fontsize=8, color=th.muted, fontweight="bold", clip_on=False)
        ax.set_xlim(0, 100)
        ax.set_ylim(y + 0.4, 0.6)
        ax.axis("off")
        for xv in (33, 66):
            ax.axvline(xv, color=th.grid, lw=0.8, ls=(0, (2, 3)), zorder=0)
    return fig


# ------------------------------------------------------------------ radar
def compare_radar(pp: pd.DataFrame, keys: list[str], metrics: list[str], *, pool_label: str, theme: Theme = DARK):
    th = theme
    ms = [METRIC_BY_LABEL[k] for k in metrics if k in METRIC_BY_LABEL]
    n = len(ms)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = [_row(pp, k) for k in keys]
    with _rc(th):
        fig = _card(th, (11, 10.2), "Player comparison — percentile radar",
                    "   ·   ".join(f"{r['Player']} ({r['Position']}, {r['Minutes']:.0f} min)" for r in rows),
                    f"percentile vs {pool_label} · outer ring = 100th")
        ax = fig.add_axes([0.14, 0.1, 0.72, 0.72], polar=True)
        ax.set_facecolor(th.bg)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for ring in (25, 50, 75, 100):
            ax.plot(np.r_[ang, ang[:1]], [ring] * (n + 1), color=th.grid, lw=0.8)
        for a in ang:
            ax.plot([a, a], [0, 100], color=th.grid, lw=0.6)
        for i, r in enumerate(rows):
            vals = np.array([r.get(f"pct:{m.label}", np.nan) for m in ms], float)
            v = np.nan_to_num(vals)
            col = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            ax.fill(np.r_[ang, ang[:1]], np.r_[v, v[:1]], color=col, alpha=0.18)
            ax.plot(np.r_[ang, ang[:1]], np.r_[v, v[:1]], color=col, lw=2.4, label=r["Player"])
            ax.scatter(ang, v, color=col, s=26, zorder=3)
        ax.set_xticks(ang, [textwrap.fill(m.label, 14) for m in ms], fontsize=9, color=th.text)
        ax.set_yticks([25, 50, 75], ["25", "50", "75"], fontsize=7, color=th.muted)
        ax.set_ylim(0, 100)
        ax.spines["polar"].set_visible(False)
        ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=min(4, len(rows)), frameon=False,
                  labelcolor=th.text, fontsize=10)
    return fig


# ------------------------------------------------------------------ side-by-side per-90 bars
def compare_bars(pp: pd.DataFrame, keys: list[str], metrics: list[str], *, pool_label: str, theme: Theme = DARK):
    th = theme
    ms = [METRIC_BY_LABEL[k] for k in metrics if k in METRIC_BY_LABEL]
    rows = [_row(pp, k) for k in keys]
    nb = len(rows)
    H = 1.7 + len(ms) * (0.32 * nb + 0.35)
    with _rc(th):
        fig = _card(th, (12, H), "Player comparison — per 90",
                    "   ·   ".join(f"{r['Player']}  {r['Minutes']:.0f} min" for r in rows),
                    f"bar = value relative to the best of the selected players · label = per 90 (percentile vs {pool_label})")
        ax = fig.add_axes([0.24, 0.45 / H, 0.62, 1 - 1.6 / H])
        y = 0
        for m in ms:
            vals = [r.get(m.column, np.nan) for r in rows]
            best = np.nanmax([v for v in vals if v == v] or [1]) or 1
            ax.text(-2, y - (nb - 1) * 0.42 / 2, m.label, ha="right", va="center", fontsize=10, color=th.text,
                    clip_on=False, fontweight="bold")
            for i, (r, v) in enumerate(zip(rows, vals)):
                yy = y - i * 0.42
                share = (v / best * 100) if (v == v and best) else 0
                if not m.higher_better and v == v:
                    share = (min([x for x in vals if x == x]) / v * 100) if v else 100
                ax.barh(yy, max(share, 0), height=0.36, color=PLAYER_COLORS[i % len(PLAYER_COLORS)])
                p = r.get(f"pct:{m.label}", np.nan)
                ax.text(max(share, 0) + 1.5, yy, fmt(m, v) + (f"  ({p:.0f})" if p == p else ""), va="center",
                        fontsize=9, color=th.text)
            y -= nb * 0.42 + 0.45
        ax.set_xlim(0, 125)
        ax.set_ylim(y + 0.2, 0.4)
        ax.axis("off")
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=PLAYER_COLORS[i % 4], label=r["Player"]) for i, r in enumerate(rows)],
                   loc="upper right", bbox_to_anchor=(0.97, 1 - 1.05 / H), frameon=False, labelcolor=th.text, ncol=nb,
                   fontsize=9)
    return fig


# ------------------------------------------------------------------ scatter
def scatter(pp: pd.DataFrame, mx: str, my: str, *, highlight: list[str] | None = None, min_minutes: float = 90.0,
            theme: Theme = DARK):
    th = theme
    a, b = METRIC_BY_LABEL[mx], METRIC_BY_LABEL[my]
    d = pp[(pp["Minutes"] >= min_minutes) & pp[a.column].notna() & pp[b.column].notna()]
    highlight = set(highlight or [])
    pos_cols = {}
    pal = [th.team, th.opp, th.accent, th.good, th.carry, th.muted]
    with _rc(th):
        fig = _card(th, (11, 8.4), f"{my} vs {mx}", f"players with ≥ {min_minutes:.0f} min · colour = position · dashed = median",
                    "per 90 unless %")
        ax = fig.add_axes([0.09, 0.1, 0.86, 0.75])
        ax.set_facecolor(th.bg)
        if d.empty:
            ax.text(0.5, 0.5, "No players over the minutes threshold", ha="center", transform=ax.transAxes, color=th.muted)
            return fig
        ax.axvline(d[a.column].median(), color=th.grid, ls=(0, (4, 3)), lw=1)
        ax.axhline(d[b.column].median(), color=th.grid, ls=(0, (4, 3)), lw=1)
        for i, p in enumerate(sorted(d["Position"].unique())):
            pos_cols[p] = pal[i % len(pal)]
        for _, r in d.iterrows():
            col = pos_cols[r["Position"]]
            hl = r["player"] in highlight
            ax.scatter(r[a.column], r[b.column], s=230 if hl else 120, color=col, edgecolors=th.text if hl else th.bg,
                       linewidths=2 if hl else 1, zorder=3, alpha=1 if (hl or not highlight) else 0.55)
            ax.annotate(r["Name"].split(" ")[-1] if r["Name"] else f"#{r['player']}", (r[a.column], r[b.column]),
                        xytext=(7, 5), textcoords="offset points", fontsize=9 if hl else 8,
                        color=th.text if (hl or not highlight) else th.muted, fontweight="bold" if hl else "normal")
        ax.set_xlabel(mx + (" per 90" if a.column.endswith("p90") else ""), color=th.muted)
        ax.set_ylabel(my + (" per 90" if b.column.endswith("p90") else ""), color=th.muted)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(th.grid)
        ax.grid(color=th.grid, lw=0.4, alpha=0.6)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], marker="o", lw=0, color=c, markersize=9, label=p) for p, c in pos_cols.items()],
                  loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=len(pos_cols), frameon=False, labelcolor=th.text)
    return fig


# ------------------------------------------------------------------ ranking
def ranking(pp: pd.DataFrame, metric: str, *, position: str | None = None, min_minutes: float = 90.0,
            highlight: list[str] | None = None, theme: Theme = DARK):
    th = theme
    m = METRIC_BY_LABEL[metric]
    d = pp[(pp["Minutes"] >= min_minutes) & pp[m.column].notna()]
    if position:
        d = d[d["Position"] == position]
    d = d.sort_values(m.column, ascending=not m.higher_better)
    highlight = set(highlight or [])
    H = 1.7 + 0.42 * max(1, len(d))
    with _rc(th):
        fig = _card(th, (11, H), f"Ranking — {metric}" + (" per 90" if m.column.endswith("p90") else ""),
                    f"{position or 'all positions'} · players with ≥ {min_minutes:.0f} min · team average dashed",
                    "lower is better" if not m.higher_better else "")
        ax = fig.add_axes([0.3, 0.4 / H, 0.6, 1 - 1.55 / H])
        if d.empty:
            ax.axis("off")
            return fig
        d = d.iloc[::-1]
        y = np.arange(len(d))
        cols = [th.accent if p in highlight else th.team for p in d["player"]]
        ax.barh(y, d[m.column], color=cols, height=0.62)
        for yi, v in zip(y, d[m.column]):
            ax.text(v + d[m.column].max() * 0.015, yi, fmt(m, v), va="center", fontsize=9, color=th.text)
        ax.axvline(d[m.column].mean(), color=th.muted, ls=(0, (4, 3)), lw=1)
        ax.set_yticks(y, [f"{p}  ·  {pos}" for p, pos in zip(d["Player"], d["Position"])], fontsize=9.5, color=th.text)
        ax.set_xlim(0, d[m.column].max() * 1.18 or 1)
        ax.set_xticks([])
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(th.grid)
        ax.tick_params(length=0)
    return fig


# ------------------------------------------------------------------ player vs position average
def vs_position(pp: pd.DataFrame, key: str, averages: pd.DataFrame, *, theme: Theme = DARK,
                metrics: list[str] | None = None):
    th = theme
    r = _row(pp, key)
    pos = r["Position"]
    ms = [m for m in METRICS if (metrics is None or m.label in metrics)]
    avg = averages.loc[pos] if pos in averages.index else pd.Series(dtype=float)
    H = 1.7 + 0.42 * len(ms)
    with _rc(th):
        fig = _card(th, (11, H), f"{r['Player']} vs the {pos} average",
                    f"{r['Minutes']:.0f} min · difference to the average of {pos} players (per 90 / %)",
                    "green = better than the position average · red = worse")
        ax = fig.add_axes([0.3, 0.4 / H, 0.5, 1 - 1.55 / H])
        y = np.arange(len(ms))[::-1]
        for yi, m in zip(y, ms):
            v, a = r.get(m.column, np.nan), avg.get(m.column, np.nan)
            if v != v or a != a or not a:
                diff = np.nan
            else:
                diff = (v - a) / abs(a) * 100 * (1 if m.higher_better else -1)
            d = np.clip(np.nan_to_num(diff), -100, 98)
            ax.barh(yi, d, color=th.good if d >= 0 else th.bad, height=0.6)
            ax.text(103, yi, f"{fmt(m, v)}  vs  {fmt(m, a)}", va="center", fontsize=9, color=th.text, clip_on=False)
            if diff == diff:                           # % label on the other side of the zero line — never overlaps
                ax.text(-3 if d >= 0 else 3, yi, f"{diff:+.0f}%", va="center", ha="right" if d >= 0 else "left",
                        fontsize=8.5, color=th.good if d >= 0 else th.bad, fontweight="bold")
        ax.axvline(0, color=th.muted, lw=1)
        ax.set_facecolor(th.bg)
        ax.set_yticks(y, [m.label for m in ms], fontsize=9.5, color=th.text)
        ax.set_xlim(-105, 100)
        ax.set_xticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    return fig
