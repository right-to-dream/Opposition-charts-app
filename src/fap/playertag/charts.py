"""Static matplotlib charts for the player tag panel (every download is a 300-dpi PNG).

Uses the Match Lab themes (Dark / Light) and pitch so the look matches the rest of the analysis pages.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from fap.matchlab.charts import FONT, THEMES, Theme, _pitch  # noqa: E402
from fap.playertag import stats as S  # noqa: E402
from fap.playertag import taxonomy as T  # noqa: E402
from fap.playertag.window import Window  # noqa: E402

DEFAULT = THEMES["Dark"]


def png(fig, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return buf.getvalue()


def _rc(th: Theme):
    return plt.rc_context({"font.family": FONT, "text.color": th.text, "axes.labelcolor": th.muted,
                           "xtick.color": th.muted, "ytick.color": th.muted, "axes.edgecolor": th.grid,
                           "axes.facecolor": th.bg, "figure.facecolor": th.bg, "savefig.facecolor": th.bg})


# ------------------------------------------------------------------ code window mock-up
def code_window(win: Window, th: Theme = DEFAULT):
    W, H = 1600, win.height
    with _rc(th):
        fig = plt.figure(figsize=(16, 16 * H / W))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), W, H, color=th.bg, zorder=0))
        ax.text(20, 36, f"PLAYER CODE WINDOW  ·  {win.title.upper()}", fontsize=17, fontweight="bold",
                color=th.text, va="center")
        ax.text(W - 20, 36, "Hudl Sportscode  ·  code → zone → outcome → extra labels", fontsize=10,
                color=th.muted, va="center", ha="right")
        zx, zy, zw, zh = win.zone_box
        ax.add_patch(Rectangle((zx - 6, zy - 6), zw + 8, zh + 8, facecolor="#1f4d2a", edgecolor="#9fd3a8",
                               lw=1.2, zorder=1))
        ax.plot([zx + zw / 2 - 2, zx + zw / 2 - 2], [zy - 6, zy + zh + 2], color="#9fd3a8", lw=1, zorder=1.5)
        for i, lane in enumerate(T.LANES):
            ax.text(zx - 12, zy + (i + .5) * zh / 3, lane, rotation=90, ha="center", va="center",
                    fontsize=8, color=th.muted)
        for h in win.headers:
            ax.text(h.x, h.y + 12, h.text, fontsize=8.5, fontweight="bold", color=th.muted, va="center")
        for b in win.buttons:
            code = b.kind == "code"
            fill = b.colour if (code or b.group in ("Outcome", "Zone")) else th.panel
            edge = b.colour
            ax.add_patch(FancyBboxPatch((b.x, b.y), b.w, b.h, boxstyle="round,pad=0,rounding_size=7",
                                        facecolor=fill, edgecolor=edge, lw=1.6, zorder=2))
            txt_c = "#ffffff" if fill != th.panel else th.text
            ax.text(b.x + b.w / 2, b.y + b.h / 2, b.name, ha="center", va="center", zorder=3, color=txt_c,
                    fontsize=10.5 if code else (11 if b.group == "Zone" else 9), fontweight="bold" if code else None)
            if b.hotkey:
                ax.add_patch(FancyBboxPatch((b.x + b.w - 20, b.y + 4), 16, 15,
                                            boxstyle="round,pad=0,rounding_size=3", facecolor="#00000055",
                                            edgecolor="none", zorder=3))
                ax.text(b.x + b.w - 12, b.y + 11.5, b.hotkey, ha="center", va="center", fontsize=7.5,
                        color="#ffffff", zorder=4, fontweight="bold")
        ax.text(20, H - 12, "Colour = code type  ·  green pitch = zone labels  ·  small badge = hotkey",
                fontsize=8, color=th.muted)
    return fig


def _ink(face, th: Theme) -> str:
    """Readable text colour on a zone fill."""
    r, g, b, _ = matplotlib.colors.to_rgba(face)
    return "#0b0f17" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


# ------------------------------------------------------------------ zone map
def zone_map(df: pd.DataFrame, title: str, th: Theme = DEFAULT, *, actions: tuple[str, ...] | None = None,
             mode: str = "count", ax=None):
    """mode "count" (heat by volume) or "success" (colour by success %, needs tagged outcomes)."""
    zt = S.zone_table(df, actions)
    own = ax is None
    with _rc(th):
        pitch = _pitch(th)
        if own:
            fig, ax = pitch.draw(figsize=(10, 7))
        else:
            fig = ax.figure
            pitch.draw(ax=ax)
        total = int(zt["Actions"].sum())
        vmax = max(1, int(zt["Actions"].max()))
        cmap = th.cmap() if mode == "count" else plt.get_cmap("RdYlGn")
        for _, r in zt.iterrows():
            x0, x1, y0, y1 = T.zone_bounds(r["Zone"])
            if mode == "count":
                val = r["Actions"] / vmax
                face = cmap(0.15 + 0.85 * val) if r["Actions"] else th.pitch
                txt = f"{r['Actions']}" if r["Actions"] else ""
                sub = f"{100 * r['Actions'] / total:.0f}%" if total and r["Actions"] else ""
            else:
                pct = r["Success %"]
                face = cmap(pct / 100) if pct == pct else th.pitch
                txt = f"{pct:.0f}%" if pct == pct else ""
                sub = f"{r['Successful']}/{r['With outcome']}" if r["With outcome"] else ""
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=face, alpha=.85, edgecolor=th.line,
                                   lw=.8, zorder=0.5))
            col = _ink(face, th)
            if txt:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 2.6, txt, ha="center", va="center", fontsize=15,
                        fontweight="bold", color=col, zorder=3)
                ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 3.6, sub, ha="center", va="center", fontsize=8.5,
                        color=col, zorder=3)
            ax.text(x0 + 1.2, y1 - 1.6, r["Zone"], fontsize=6.5, color=th.muted, va="top", zorder=3)
        ax.annotate("", xy=(60, -2.2), xytext=(45, -2.2), xycoords="data", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=th.muted, lw=1.2))
        ax.text(62, -2.2, "Attack", ha="left", va="center", fontsize=7.5, color=th.muted)
        ax.set_title(title, color=th.text, fontsize=13, fontweight="bold", loc="left", pad=10)
    return fig


# ------------------------------------------------------------------ success by action
def action_bars(df: pd.DataFrame, title: str, th: Theme = DEFAULT, ax=None):
    s = S.action_summary(df)
    own = ax is None
    with _rc(th):
        if own:
            fig, ax = plt.subplots(figsize=(9, max(3.2, .42 * len(s) + 1.2)))
        else:
            fig = ax.figure
        if not len(s):
            ax.text(.5, .5, "No tagged actions", ha="center", va="center", color=th.muted, transform=ax.transAxes)
            ax.axis("off")
            return fig
        s = s.iloc[::-1]
        ypos = np.arange(len(s))
        made = s["Successful"].to_numpy()
        tagged = s["With outcome"].to_numpy()
        ax.barh(ypos, s["Total"], color=th.grid, height=.62, label="Total")
        ax.barh(ypos, tagged, color=th.bad, alpha=.75, height=.62, label="Unsuccessful")
        ax.barh(ypos, made, color=th.good, height=.62, label="Successful")
        for i, (_, r) in enumerate(s.iterrows()):
            lab = f"{r['Total']}" + (f"   {r['Success %']:.0f}%" if r["With outcome"] and
                                     r["Action"] not in ("Interception", "Recovery", "Clearance", "Block")
                                     else "")
            ax.text(r["Total"] + max(s["Total"]) * .015, i, lab, va="center", fontsize=9, color=th.text)
        ax.set_yticks(ypos, s["Action"])
        ax.set_xlim(0, max(s["Total"]) * 1.18)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=th.muted)
        ax.set_title(title, color=th.text, fontsize=13, fontweight="bold", loc="left")
    return fig


# ------------------------------------------------------------------ one-page player sheet
def player_sheet(df: pd.DataFrame, player: str, position: str, th: Theme = DEFAULT, *, minutes: float | None = None,
                 subtitle: str = ""):
    kpis = S.headline(df, position, minutes)
    with _rc(th):
        fig = plt.figure(figsize=(16, 11))
        fig.text(.03, .955, player or "Player", fontsize=24, fontweight="bold", color=th.text)
        fig.text(.03, .925, f"{T.POSITIONS.get(position, position)}  ·  {subtitle}", fontsize=12, color=th.muted)
        n = len(kpis)
        for i, v in enumerate(kpis):
            x = .03 + i * (.94 / n)
            ax = fig.add_axes([x, .79, .94 / n - .01, .1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=.08", facecolor=th.panel,
                                        edgecolor=th.grid, transform=ax.transAxes))
            ax.text(.5, .6, v.display, ha="center", va="center", fontsize=22, fontweight="bold",
                    color=th.accent if not v.metric.lower_better else th.opp)
            ax.text(.5, .25, v.metric.label, ha="center", va="center", fontsize=9, color=th.muted, wrap=True)
            if v.detail:
                ax.text(.5, .08, v.detail, ha="center", va="center", fontsize=7.5, color=th.muted)
        poss = tuple(a.name for a in T.ACTIONS if a.category == T.IN_POSS)
        out = tuple(a.name for a in T.ACTIONS if a.category != T.IN_POSS)
        zone_map(df, "In possession · actions by zone", th, actions=poss, ax=fig.add_axes([.03, .38, .45, .36]))
        zone_map(df, "Out of possession · actions by zone", th, actions=out, ax=fig.add_axes([.52, .38, .45, .36]))
        action_bars(df, "Actions & success rate", th, ax=fig.add_axes([.12, .04, .36, .29]))
        zone_map(df, "Success % by zone (tagged outcomes)", th, mode="success", ax=fig.add_axes([.52, .02, .45, .32]))
    return fig


def per_match_trend(table: pd.DataFrame, metric: str, th: Theme = DEFAULT):
    row = table[table["Metric"] == metric]
    with _rc(th):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        if not len(row):
            ax.axis("off")
            return fig
        r = row.iloc[0]
        ms = [c for c in table.columns if c not in ("Section", "Metric", "All matches")]
        vals = [float(str(r[m]).rstrip("%")) if str(r[m]) not in ("–", "") else np.nan for m in ms]
        ax.plot(range(len(ms)), vals, marker="o", color=th.team, lw=2)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f" {r[ms[i]]}", fontsize=9, color=th.text, va="bottom")
        ax.set_xticks(range(len(ms)), ms, rotation=20, ha="right", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{metric} · match by match", color=th.text, fontsize=12, fontweight="bold", loc="left")
    return fig
