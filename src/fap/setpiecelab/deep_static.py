"""Static (matplotlib / mplsoccer) charts for :mod:`fap.setpiecelab.deep` — every one takes the rows ``g`` to
draw (a type, one side, one match …) so the same chart works for the aggregated and the detailed views.

* :func:`delivery_map`  — where it was taken → where it landed, ● 1st contact won / ✕ lost, colour = result,
                          ◆ second ball;
* :func:`zone_map`      — the box split into delivery zones (near / central / far, 6-yd box, penalty area, edge):
                          deliveries, 1st contact won and danger per zone;
* :func:`players_chart` — the team's players at the first contact (won / lost and what followed);
* :func:`funnel_chart`  — taken → 1st contact won → shot → on target → goal (or the defending version);
* :func:`shots_chart`   — where the shots came from and how they ended;
* :func:`match_chart`   — match by match (when the file has a match / opponent column);
* :func:`summary_table` — one row per set-piece type with the key numbers.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mplsoccer import VerticalPitch

from fap.matchlab.charts import DARK, Theme
from fap.setpiecelab import analytics as A
from fap.setpiecelab import deep as D
from fap.setpiecelab.loader import PITCH_L, PITCH_W, RESULT_COL, SetPieceData, is_yes
from fap.setpiecelab.static import _figure, _rc, _style_ax

RESULT_COLORS = {"goal": "#e5484d", "shot": "#f97316", "threat": "#f5b301", "cross": "#a855f7", "recycle": "#2ed8a3",
                 "counter": "#22c55e", "gk punch": "#14b8a6", "gk claim": "#14b8a6", "clearance": "#3ab0ff",
                 "out of bounds": "#94a3b8"}


def _half_pitch(th: Theme):
    return VerticalPitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W, half=True,
                         pitch_color=th.pitch, line_color=th.line, linewidth=1.2, goal_type="box",
                         corner_arcs=True, pad_top=3, pad_bottom=2, pad_left=2, pad_right=2)


def _res_color(v: str, th: Theme) -> str:
    return RESULT_COLORS.get(str(v).lower(), th.muted)


# ------------------------------------------------------------------ delivery map
def delivery_map(sd: SetPieceData, g: pd.DataFrame, title: str, subtitle: str = "", *, theme: Theme = DARK,
                 show_origin: bool = True, show_second: bool = True):
    th = theme
    c1 = A._col(sd, "1st contact")
    pts = g[g["x"].notna()]
    sb = g[g["sb_x"].notna()] if show_second and "sb_x" in g else g.iloc[0:0]
    lo = min([pts["x"].min() if len(pts) else 90, sb["sb_x"].min() if len(sb) else 90,
              pts["ox"].min() if show_origin and "ox" in pts and pts["ox"].notna().any() else 90])
    y_lo = max(PITCH_L / 2, min(lo - 4, 76))
    ratio = (PITCH_L + 3 - y_lo) / (PITCH_W + 8)
    with _rc(th):
        fig = _figure(th, (10, 2.4 + 9.4 * ratio), title, subtitle or (
            f"{len(pts)} of {len(g)} with a landing point   ·   colour = result   ·   "
            f"● {sd.team} won the 1st contact   ✕ opponent won it   ◆ second ball"))
        H = 2.4 + 9.4 * ratio
        ax = fig.add_axes([0.03, 1.3 / H, 0.94, 1 - 2.25 / H])
        pitch = _half_pitch(th)
        pitch.draw(ax=ax)
        ax.set_ylim(y_lo, PITCH_L + 3)
        if pts.empty:
            ax.text(PITCH_W / 2, PITCH_L - 12, "No landing points coded", ha="center", va="center", color=th.muted, fontsize=13)
            return fig
        for _, r in pts.iterrows():
            col = _res_color(r.get(RESULT_COL, ""), th)
            if show_origin and pd.notna(r.get("ox")) and pd.notna(r.get("oy")):
                pitch.lines(r["ox"], r["oy"], r["x"], r["y"], ax=ax, color=col, lw=1.2, alpha=0.32, comet=True,
                            transparent=True, zorder=2)
            yn = is_yes(r[c1]) if c1 else None
            mk = "o" if yn is not False else "X"
            pitch.scatter(r["x"], r["y"], ax=ax, s=240, marker=mk, color=col if yn is not None else "none",
                          edgecolors=th.bg if yn is not None else col, linewidth=1.3 if yn is not None else 2, zorder=4)
            pitch.annotate(str(r["n"]), (r["x"] + 1.5, r["y"] - 1.2), ax=ax, ha="center", va="center", fontsize=7,
                           color=th.muted, zorder=5)
        for _, r in sb.iterrows():
            if pd.notna(r["x"]):
                pitch.lines(r["x"], r["y"], r["sb_x"], r["sb_y"], ax=ax, color=th.carry, lw=1, ls=(0, (2, 2)), zorder=3)
            pitch.scatter(r["sb_x"], r["sb_y"], ax=ax, s=110, marker="D", color=th.carry, edgecolors=th.bg, lw=1, zorder=4)
        res = pts[RESULT_COL].replace("", "Not coded").value_counts() if RESULT_COL in pts else pd.Series(dtype=int)
        handles = [Line2D([0], [0], marker="o", lw=0, markersize=9, color=_res_color(k, th), label=f"{k} ({v})")
                   for k, v in res.items()]
        handles += [Line2D([0], [0], marker="o", lw=0, markersize=9, color=th.text, label=f"{sd.team} won 1st contact"),
                    Line2D([0], [0], marker="X", lw=0, markersize=9, color=th.text, label="Opponent won it")]
        if len(sb):
            handles.append(Line2D([0], [0], marker="D", lw=0, markersize=8, color=th.carry, label=f"Second ball ({len(sb)})"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=8.5,
                  labelcolor=th.text, handletextpad=0.3, columnspacing=1.0)
    return fig


# ------------------------------------------------------------------ zone map
def zone_map(sd: SetPieceData, g: pd.DataFrame, title: str, subtitle: str = "", *, theme: Theme = DARK):
    """Zones drawn with the NEAR side (the taker's side) on the right."""
    th = theme
    zt = D.zone_table(sd, g)
    att = (g["phase"].iloc[0] == "Attacking") if len(g) else True
    with _rc(th):
        fig = _figure(th, (10, 6.6), title, subtitle or (
            f"{int(zt['Deliveries'].sum()) if len(zt) else 0} deliveries located   ·   near side = the taker's side (right)   ·   "
            f"label = deliveries · 1st contact won by {sd.team} · danger {'created' if att else 'conceded'}"))
        ax = fig.add_axes([0.03, 0.12, 0.94, 0.72])
        pitch = _half_pitch(th)
        pitch.draw(ax=ax)
        ax.set_ylim(74, PITCH_L + 3)
        if zt.empty:
            ax.text(PITCH_W / 2, PITCH_L - 12, "No landing points coded", ha="center", color=th.muted, fontsize=13)
            return fig
        mx = max(zt["Deliveries"].max(), 1)
        cmap = th.cmap()
        for _, r in zt.iterrows():
            rect = D.zone_rect(r["Zone"], -1.0)      # near side = small y = drawn on the right of a vertical pitch
            if rect is None:
                continue
            x0, x1, y0, y1 = rect
            px, py = y0, x0                          # vertical pitch axes: horizontal = y, vertical = x
            ax.add_patch(Rectangle((px, py), y1 - y0, x1 - x0, color=cmap(0.15 + 0.75 * r["Deliveries"] / mx), alpha=0.75,
                                   zorder=1.5, lw=0))
            ax.add_patch(Rectangle((px, py), y1 - y0, x1 - x0, fill=False, edgecolor=th.line, lw=0.6, zorder=1.6))
            cx, cy = px + (y1 - y0) / 2, py + (x1 - x0) / 2
            c1 = f"{int(r['1st contact won'])}/{int(r['1st contact decided'])}" if r["1st contact decided"] else "–"
            txt = f"{int(r['Deliveries'])}\n1st {c1}\n⚠ {int(r['Dangerous'])}"
            ax.text(cx, cy, txt, ha="center", va="center", fontsize=8.5 if (x1 - x0) < 8 else 10, color=th.text,
                    fontweight="bold", zorder=3, linespacing=1.15)
        other = zt[zt["Zone"].isin(["Wide / deep", "Short (not delivered)"])]
        if len(other):
            fig.text(0.5, 0.075, "   ·   ".join(f"{r['Zone']}: {int(r['Deliveries'])} (danger {int(r['Dangerous'])})"
                                                for _, r in other.iterrows()), ha="center", fontsize=9.5, color=th.muted)
        ax.text(2, PITCH_L + 1.6, "near side (taker) →", ha="right", va="center", fontsize=8.5, color=th.muted)
        ax.text(PITCH_W - 2, PITCH_L + 1.6, "← far side", ha="left", va="center", fontsize=8.5, color=th.muted)
    return fig


# ------------------------------------------------------------------ players
def players_chart(sd: SetPieceData, g: pd.DataFrame, title: str, subtitle: str = "", *, theme: Theme = DARK):
    th = theme
    pl = D.players(sd, g)
    with _rc(th):
        h = 1.9 + 0.46 * max(1, len(pl))
        fig = _figure(th, (11, h), title, subtitle or f"{sd.team}'s player at the 1st contact   ·   green = won it   ·   "
                                                      "red = lost it   ·   label = what followed")
        ax = fig.add_axes([0.12, 0.1, 0.6, 1 - 1.5 / h])
        if pl.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No first-contact players coded", ha="center", color=th.muted, fontsize=12)
            return fig
        pl = pl.iloc[::-1]
        y = np.arange(len(pl))
        ax.barh(y, pl["Won"], color=th.good, height=0.6)
        ax.barh(y, pl["Lost"], left=pl["Won"], color=th.bad, height=0.6)
        top = (pl["Won"] + pl["Lost"]).max()
        for yi, (_, r) in zip(y, pl.iterrows()):
            bits = [f"won {int(r['Won'])}"] + ([f"lost {int(r['Lost'])}"] if r["Lost"] else [])
            if r["Shots after his win"]:
                bits.append(f"{int(r['Shots after his win'])} shot after a win" + (f" ({int(r['Goals after his win'])} goal)" if r["Goals after his win"] else ""))
            if r["Shots after his loss"]:
                bits.append(f"{int(r['Shots after his loss'])} shot after a loss" + (f" ({int(r['Goals after his loss'])} goal)" if r["Goals after his loss"] else ""))
            ax.text(r["Won"] + r["Lost"] + top * 0.02, yi, "  ·  ".join(bits), va="center", fontsize=9, color=th.text)
        ax.set_yticks(y, pl["Player"], fontsize=11, color=th.text, fontweight="bold")
        ax.set_xlim(0, top * 1.02)
        ax.set_xticks([])
        _style_ax(ax, th)
        ax.spines["bottom"].set_visible(False)
    return fig


# ------------------------------------------------------------------ funnel & shots
def funnel_chart(sd: SetPieceData, g: pd.DataFrame, title: str, attacking: bool, *, theme: Theme = DARK):
    th = theme
    steps = D.funnel(sd, g, attacking)
    with _rc(th):
        fig = _figure(th, (10, 4.6), title, "every step counted from the file   ·   % of all " + ("taken" if attacking else "faced"))
        ax = fig.add_axes([0.2, 0.1, 0.75, 0.68])
        top = max(steps[0][1], 1)
        cols = [th.team, th.accent, th.carry, th.good if attacking else th.bad, th.good if attacking else th.bad]
        y = np.arange(len(steps))[::-1]
        for yi, (lab, v), col in zip(y, steps, cols):
            w = v / top
            ax.barh(yi, w, left=(1 - w) / 2, color=col, height=0.7)
            ax.text(0.5, yi, f"{v}  ({v / top * 100:.0f}%)" if lab not in ("Taken", "Faced") else str(v), ha="center",
                    va="center", fontsize=11, fontweight="bold", color="#ffffff")
        ax.set_yticks(y, [s[0] for s in steps], fontsize=10.5, color=th.text)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        _style_ax(ax, th)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
    return fig


def shots_chart(sd: SetPieceData, g: pd.DataFrame, title: str, *, theme: Theme = DARK):
    th = theme
    st = D.shots_table(sd, g)
    so = g["shot_outcome"][g["shot_outcome"] != ""].value_counts()
    with _rc(th):
        fig = _figure(th, (11, 4.4), title, f"{int(g['is_shot'].sum())} shots   ·   left = where the shot came from   ·   "
                                            "right = how it ended")
        a1 = fig.add_axes([0.16, 0.12, 0.32, 0.64])
        a2 = fig.add_axes([0.64, 0.12, 0.32, 0.64])
        for ax, data, colors in ((a1, st.set_index("Source")["Shots"] if len(st) else pd.Series(dtype=int), None),
                                 (a2, so, "out")):
            if data is None or len(data) == 0:
                ax.axis("off")
                ax.text(0.5, 0.5, "none coded", ha="center", color=th.muted, transform=ax.transAxes)
                continue
            data = data.iloc[::-1]
            cols = [RESULT_COLORS.get("goal") if str(k).lower() == "goal" else th.team for k in data.index] if colors \
                else [th.accent] * len(data)
            ax.barh(range(len(data)), data.values, color=cols, height=0.6)
            for yi, v in enumerate(data.values):
                ax.text(v + data.max() * 0.04, yi, str(int(v)), va="center", fontsize=10, color=th.text)
            ax.set_yticks(range(len(data)), data.index, fontsize=10, color=th.text)
            ax.set_xlim(0, data.max() * 1.3)
            ax.set_xticks([])
            _style_ax(ax, th)
            ax.spines["bottom"].set_visible(False)
    return fig


# ------------------------------------------------------------------ matches
def match_chart(sd: SetPieceData, title: str, *, theme: Theme = DARK):
    th = theme
    bm = D.by_match(sd)
    ms = D.matches(sd)
    with _rc(th):
        h = 2.2 + 0.55 * max(1, len(ms))
        fig = _figure(th, (13, h), title, "per match   ·   left = taking (shots / goals)   ·   right = defending "
                                          "(shots / goals conceded)   ·   1st contact won %")
        if bm.empty:
            fig.text(0.5, 0.5, "No match / opponent column in the file", ha="center", color=th.muted, fontsize=12)
            return fig
        for k, (ph, col, x0) in enumerate((("Attacking", th.team, 0.14), ("Defending", th.opp, 0.58))):
            ax = fig.add_axes([x0, 0.1, 0.36, 1 - 1.5 / h])
            d = bm[bm["Phase"] == ph].set_index("Match").reindex(ms)
            y = np.arange(len(ms))
            ax.barh(y + 0.18, d["Shots"].fillna(0), height=0.34, color=col, label="Shots")
            ax.barh(y - 0.18, d["Goals"].fillna(0), height=0.34, color=RESULT_COLORS["goal"], label="Goals")
            mx = max(d["Shots"].max() if d["Shots"].notna().any() else 1, 1)
            for yi, (_, r) in zip(y, d.iterrows()):
                if pd.notna(r["Count"]):
                    ax.text(mx * 1.05, yi, f"{int(r['Count'])} {'taken' if ph == 'Attacking' else 'faced'} · 1st "
                            + (f"{int(r['1st contact won'])}/{int(r['1st contact decided'])}" if r["1st contact decided"] else "–"),
                            va="center", fontsize=8.5, color=th.muted)
            ax.set_yticks(y, ms if k == 0 else [""] * len(ms), fontsize=10, color=th.text)
            ax.set_xlim(0, mx * 2.1)
            ax.invert_yaxis()
            ax.set_title(f"{sd.team} {'taking' if ph == 'Attacking' else 'defending'}", loc="left", color=col,
                         fontsize=11, fontweight="bold")
            ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=th.text)
            _style_ax(ax, th)
    return fig


# ------------------------------------------------------------------ summary table
def summary_table(sd: SetPieceData, title: str, *, theme: Theme = DARK):
    th = theme
    rows = []
    for ph, r, g in D.groups(sd):
        s = D.summary(sd, g)
        rows.append([ph, D.type_label(r), str(s["Count"]),
                     f"{s['1st contact won']}/{s['1st contact decided']}" if s["1st contact decided"] else "–",
                     f"{s['2nd ball won']}/{s['2nd ball decided']}" if s["2nd ball decided"] else "–",
                     str(s["Shots"]), str(s["Goals"]), f"{s['Danger %']:.0f}%" if s["Count"] else "–"])
    cols = ["Phase", "Set piece", "Count", f"1st contact won", "2nd ball won", "Shots", "Goals", "Danger %"]
    with _rc(th):
        h = 1.6 + 0.42 * (len(rows) + 1)
        fig = _figure(th, (13, h), title, f"Taking = {sd.team} takes it (shots / goals scored)   ·   Defending = against "
                                          f"{sd.team} (shots / goals conceded)   ·   contests = won by {sd.team}")
        ax = fig.add_axes([0.03, 0.06, 0.94, 1 - 1.3 / h])
        ax.axis("off")
        n = len(rows) + 1
        widths = [0.11, 0.27, 0.08, 0.14, 0.14, 0.08, 0.08, 0.1]
        xs = np.cumsum([0] + widths[:-1])
        for j, c in enumerate(cols):
            ax.text(xs[j] + 0.005, 1 - 0.5 / n, c, ha="left", va="center", fontsize=10, fontweight="bold", color=th.accent)
        for i, row in enumerate(rows, start=1):
            yc = 1 - (i + 0.5) / n
            if i % 2:
                ax.add_patch(Rectangle((0, 1 - (i + 1) / n), 1, 1 / n, color=th.panel, lw=0, zorder=0))
            for j, v in enumerate(row):
                col = (th.team if v == "Attacking" else th.opp) if j == 0 else th.text
                ax.text(xs[j] + 0.005, yc, v, ha="left", va="center", fontsize=10, color=col,
                        fontweight="bold" if j in (0, 1) else "normal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return fig


def insights_page(items: list, title: str, subtitle: str = "", *, theme: Theme = DARK, width: int = 120):
    """A page of insights (``deep.Insight``) grouped by section — coloured by strength / concern / note."""
    th = theme
    tone_col = {"strength": th.good, "concern": th.bad, "note": th.team}
    lines = []
    last = None
    for it in items:
        if it.section != last:
            lines.append(("h", it.section, None))
            last = it.section
        for k, w in enumerate(textwrap.wrap(it.text, width)):
            lines.append(("t", ("•  " if k == 0 else "   ") + w, tone_col.get(it.tone, th.team)))
    H = 1.4 + 0.27 * len(lines) + 0.12 * sum(1 for l_ in lines if l_[0] == "h")
    with _rc(th):
        fig = _figure(th, (13, H), title, subtitle or "green = strength   ·   red = point to watch   ·   blue = note   ·   "
                                                     "every number counted from the file")
        y = 1 - 1.15 / H
        for kind, text, col in lines:
            if kind == "h":
                y -= 0.12 / H
                fig.text(0.03, y, text.upper(), ha="left", va="top", fontsize=10, fontweight="bold", color=th.accent)
            else:
                fig.text(0.045, y, text, ha="left", va="top", fontsize=10, color=th.text)
                fig.text(0.033, y, "▍", ha="left", va="top", fontsize=10, color=col)
            y -= 0.27 / H
    return fig
