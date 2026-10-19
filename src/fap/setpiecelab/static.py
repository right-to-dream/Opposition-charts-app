"""Static, publication-grade PNG versions of the Set Piece Lab charts (matplotlib + mplsoccer, Match Lab
themes). The page shows the interactive Plotly charts; every download is one of these PNGs."""
from __future__ import annotations

import io
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from mplsoccer import Pitch, VerticalPitch  # noqa: E402

from fap.matchlab.charts import DARK, FONT, Theme  # noqa: E402
from fap.setpiecelab import analytics as A  # noqa: E402
from fap.setpiecelab.loader import PITCH_L, PITCH_W, SetPieceData, is_yes  # noqa: E402

OUT_COLORS = {"goal": "#e5484d", "shot": "#f97316", "threat": "#f5b301"}


def _figure(th: Theme, size, title: str, subtitle: str):
    fig = plt.figure(figsize=size, facecolor=th.bg)
    fig.text(0.03, 0.975, title, ha="left", va="top", fontsize=17, fontweight="bold", color=th.text)
    fig.text(0.03, 0.975 - 0.42 / size[1], subtitle, ha="left", va="top", fontsize=10, color=th.muted)
    fig.text(0.03, 0.012, "SET PIECE LAB  ·  pitch 105 × 68 m", ha="left", va="bottom", fontsize=7.5, color=th.muted,
             fontweight="bold", alpha=0.85)
    return fig


def _style_ax(ax, th: Theme) -> None:
    ax.set_facecolor(th.bg)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th.grid)
    ax.tick_params(colors=th.muted, length=0)


def png(fig, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return buf.getvalue()


def _rc(th: Theme):
    return plt.rc_context({"font.family": FONT, "text.color": th.text, "axes.labelcolor": th.muted,
                           "xtick.color": th.muted, "ytick.color": th.muted})


# ------------------------------------------------------------------ overview
def overview(sd: SetPieceData, *, theme: Theme = DARK):
    th = theme
    ov = A.overview(sd).sort_values("Count")
    with _rc(th):
        h = 1.6 + 0.48 * len(ov)
        fig = _figure(th, (10, h), f"{sd.team} — set pieces coded",
                      f"blue = {sd.team} taking it   ·   red = {sd.team} defending it   ·   bar label = count · dangerous (goals)")
        ax = fig.add_axes([0.24, 0.08, 0.7, 1 - 1.25 / h])
        y = np.arange(len(ov))
        cols = [th.team if p == "Attacking" else th.opp for p in ov["Phase"]]
        ax.barh(y, ov["Count"], color=cols, height=0.62)
        has_d = "Dangerous (goal/shot/threat)" in ov
        for yi, (_, r) in zip(y, ov.iterrows()):
            extra = f"   {int(r['Dangerous (goal/shot/threat)'])} dangerous ({int(r.get('Goals', 0))} goals)" if has_d else ""
            ax.text(r["Count"] + ov["Count"].max() * 0.015, yi, f"{int(r['Count'])}{extra}", va="center", fontsize=9.5,
                    color=th.text)
        ax.set_yticks(y, ov["Set piece"], fontsize=10.5, color=th.text)
        ax.set_xlim(0, ov["Count"].max() * 1.6)
        ax.set_xticks([])
        _style_ax(ax, th)
        ax.spines["bottom"].set_visible(False)
    return fig


def contests(sd: SetPieceData, *, theme: Theme = DARK):
    th = theme
    ov = A.overview(sd)
    keys = [c for c in A.CONTESTS if f"{c} decided" in ov]
    ov = ov[sum(ov[f"{c} decided"] for c in keys) > 0] if keys else ov.iloc[0:0]
    with _rc(th):
        h = 1.8 + 0.62 * max(1, len(ov))
        fig = _figure(th, (10, h), f"{sd.team} — 1st contact & 2nd ball",
                      f"share of the contests {sd.team} won (attacking and defending)   ·   label = won / coded")
        ax = fig.add_axes([0.24, 0.1, 0.7, 1 - 1.4 / h])
        y = np.arange(len(ov))
        bh = 0.36
        for j, (c, col) in enumerate(zip(keys, (th.team, th.accent))):
            dec = ov[f"{c} decided"].to_numpy(float)
            won = ov[f"{c} won"].to_numpy(float)
            rate = np.where(dec > 0, won / np.maximum(dec, 1) * 100, np.nan)
            yy = y + (j - (len(keys) - 1) / 2) * bh
            ax.barh(yy, np.nan_to_num(rate), height=bh * 0.9, color=col, label=f"{c} won by {sd.team}")
            for yi, r_, w_, d_ in zip(yy, rate, won, dec):
                if d_ > 0:
                    ax.text(r_ + 1.5, yi, f"{r_:.0f}%  ({int(w_)}/{int(d_)})", va="center", fontsize=8.5, color=th.text)
        ax.set_yticks(y, ov["Set piece"], fontsize=10.5, color=th.text)
        ax.set_xlim(0, 135)
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=8.5)
        ax.invert_yaxis()
        _style_ax(ax, th)
        ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=th.text)
    return fig


# ------------------------------------------------------------------ landing map
def landing(sd: SetPieceData, row: str, *, color_by: str | None = None, theme: Theme = DARK):
    th = theme
    g = sd.of(row=row)
    pts = g[g["x"].notna()].copy()
    color_by = color_by if color_by in sd.qualifiers else None
    att = (g["phase"].iloc[0] == "Attacking") if len(g) else True
    flip = bool(len(pts)) and bool(pts["x"].mean() < PITCH_L / 2)
    if flip:                                        # their own goal: show it at the top like the others
        pts["x"], pts["y"] = PITCH_L - pts["x"], PITCH_W - pts["y"]
        pts["points"] = pts["points"].map(lambda p: [(PITCH_L - a, PITCH_W - b) for a, b in p])
        if "sb_x" in pts:
            pts["sb_x"], pts["sb_y"] = PITCH_L - pts["sb_x"], PITCH_W - pts["sb_y"]
    sbp = pts[pts["sb_x"].notna()] if "sb_x" in pts else pts.iloc[0:0]
    half = bool(len(pts) == 0 or pts["x"].min() >= PITCH_L / 2 - 2)
    with _rc(th):
        pitch = VerticalPitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W, half=half,
                              pitch_color=th.pitch, line_color=th.line, linewidth=1.2, goal_type="box",
                              corner_arcs=True, pad_top=2, pad_bottom=2, pad_left=2, pad_right=2) if half else \
            Pitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W, pitch_color=th.pitch,
                  line_color=th.line, linewidth=1.2, goal_type="box", corner_arcs=True)
        size = (8.6, 8.4) if half else (11, 8.6)
        fig = _figure(th, size, f"{sd.team} · {row} — where the ball lands",
                      f"{len(pts)} of {len(g)} with a landing point   ·   {'taken by' if att else 'against'} {sd.team}"
                      + (f"   ·   colour = {color_by}" if color_by else "")
                      + ("   ·   shown towards the top goal" if flip else ""))
        ax = fig.add_axes([0.03, 0.1, 0.94, 0.76])
        pitch.draw(ax=ax)
        lo_x = min(pts["x"].min(), sbp["sb_x"].min()) if len(sbp) else (pts["x"].min() if len(pts) else 0)
        if half and len(pts) and bool(lo_x >= 72):                     # zoom onto the final third
            ax.set_ylim(68, PITCH_L + 3)
        if pts.empty:
            ax.text(PITCH_W / 2 if half else PITCH_L / 2, (PITCH_L * 0.75) if half else PITCH_W / 2,
                    "No landing points coded", ha="center", va="center", color=th.muted, fontsize=13)
            return fig
        for _, r in pts[pts["extra_points"] > 0].iterrows():
            xs, ys = [p[0] for p in r["points"]], [p[1] for p in r["points"]]
            pitch.plot(xs, ys, ax=ax, color=th.muted, lw=1.1, ls=(0, (2, 2)), zorder=2)
            pitch.scatter(xs[1:], ys[1:], ax=ax, s=55, facecolor="none", edgecolor=th.muted, lw=1.2, zorder=2)
        palette = [th.team, th.accent, th.good, th.carry, th.opp, th.bad, th.muted]
        if color_by:
            vals = [v for v in pts[color_by].value_counts().index if v] + (["" ] if (pts[color_by] == "").any() else [])
            groups = []
            for i, v in enumerate(vals):
                col = (OUT_COLORS.get(v.lower(), palette[i % len(palette)]) if color_by.lower() in ("outcome", "result")
                       else palette[i % len(palette)])
                groups.append((v or "not coded", pts[pts[color_by] == v], th.muted if not v else col))
        else:
            groups = [("All", pts, th.team)]
        for name, d, col in groups:
            pitch.scatter(d["x"], d["y"], ax=ax, s=230, color=col, edgecolors=th.bg, linewidth=1.4, zorder=3,
                          label=f"{name} ({len(d)})")
            for _, r in d.iterrows():
                pitch.annotate(str(r["n"]), (r["x"], r["y"]), ax=ax, ha="center", va="center", fontsize=7.5,
                               fontweight="bold", color="#ffffff" if col not in (th.accent,) else "#111111", zorder=4)
        if len(sbp):                                   # second ball: where the loose ball was won
            for _, r in sbp.iterrows():
                pitch.plot([r["x"], r["sb_x"]], [r["y"], r["sb_y"]], ax=ax, color=th.carry, lw=1.1, ls=(0, (2, 2)),
                           zorder=2.5)
            pitch.scatter(sbp["sb_x"], sbp["sb_y"], ax=ax, s=150, marker="D", color=th.carry, edgecolors=th.bg,
                          linewidth=1, zorder=3.2, label=f"Second ball ({len(sbp)})")
            groups = groups + [("Second ball", sbp, th.carry)]
        if len(pts) >= 3:
            mx, my = pts["x"].mean(), pts["y"].mean()
            pitch.scatter([mx], [my], ax=ax, s=1400, facecolor=to_rgba(th.accent, 0.08), edgecolor=th.accent,
                          lw=1.2, ls=(0, (3, 2)), zorder=1.5)
            pitch.annotate("average", (mx, my), xytext=(mx - 3.2, my), ax=ax, ha="center", va="center", fontsize=7.5,
                           color=th.accent, zorder=1.6)
        leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=min(5, len(groups)), frameon=False,
                        fontsize=9, labelcolor=th.text, handletextpad=0.3, columnspacing=1.2)
    return fig


# ------------------------------------------------------------------ qualifiers
def _qual_title(ax, sd: SetPieceData, q: str, th: Theme, extra: str = "") -> None:
    """Panel title: coloured headline (1st contact = blue, 2nd ball = purple, overall result = gold) + detail."""
    head, detail = A.qual_title(sd, q)
    col = A.CONTACT_COLORS[A.qual_kind(sd, q)] or th.muted
    ax.set_title(head, loc="left", fontsize=10, color=col, fontweight="bold", pad=16)
    tail = "  ·  ".join(x for x in (detail, extra) if x)
    if tail:
        ax.text(0, 1.02, tail, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=th.muted)
def _bar_color(th: Theme, k: str) -> str:
    yn = is_yes(k)
    kl = k.lower()
    if kl.endswith(" won"):
        yn = not kl.startswith("opponent")
    return th.good if yn is True else th.bad if yn is False else OUT_COLORS.get(kl, th.team)


def qualifiers(sd: SetPieceData, row: str, *, theme: Theme = DARK):
    th = theme
    br = A.breakdown(sd, row)
    total = len(sd.of(row=row))
    n = max(1, len(br))
    cols = min(n, 3)
    rows_ = int(np.ceil(n / cols))
    heights = [max(len(v) for v in list(br.values())[r * cols:(r + 1) * cols]) for r in range(rows_)] if br else [1]
    H = 1.5 + sum(0.75 + 0.36 * h for h in heights)
    with _rc(th):
        fig = _figure(th, (12, H), f"{sd.team} · {row} — how it is taken / defended",
                      f"{total} coded   ·   every qualifier: count and share of the coded instances")
        if not br:
            fig.text(0.5, 0.5, "No qualifiers coded", ha="center", color=th.muted, fontsize=13)
            return fig
        top = 1 - 1.25 / H
        items = [(q, br[q]) for q in A.ordered_quals(sd, br)]
        for r in range(rows_):
            hgt = (0.36 * heights[r]) / H
            for c in range(cols):
                i = r * cols + c
                if i >= len(items):
                    continue
                q, vc = items[i]
                ax = fig.add_axes([0.1 + c * 0.315, top - hgt - 0.35 / H, 0.2, hgt])
                vc = vc.iloc[::-1]
                ax.barh(range(len(vc)), vc.values, color=[_bar_color(th, k) for k in vc.index], height=0.66)
                for yi, v in enumerate(vc.values):
                    ax.text(v + vc.max() * 0.04, yi, f"{v}  ({v / vc.sum() * 100:.0f}%)", va="center", fontsize=8.5,
                            color=th.text)
                ax.set_yticks(range(len(vc)), vc.index, fontsize=9, color=th.text)
                ax.set_xlim(0, vc.max() * 1.75)
                ax.set_ylim(-0.6, max(len(vc), 2) - 0.4)
                ax.set_xticks([])
                _style_ax(ax, th)
                ax.spines["bottom"].set_visible(False)
                _qual_title(ax, sd, q, th, f"{int(vc.sum())}/{total} coded")
            top -= hgt + 0.75 / H
    return fig


def cross(sd: SetPieceData, row: str, a: str, b: str, *, theme: Theme = DARK):
    th = theme
    ct = A.crosstab(sd, row, a, b)
    with _rc(th):
        h = 2.2 + 0.55 * max(1, len(ct.index))
        w = max(7.5, 2.6 + 1.25 * max(1, len(ct.columns)))
        fig = _figure(th, (w, h), f"{sd.team} · {row} — {a} × {b}",
                      f"{int(ct.values.sum()) if not ct.empty else 0} instances coded with both")
        if ct.empty:
            fig.text(0.5, 0.45, "No instances coded with both", ha="center", color=th.muted, fontsize=12)
            return fig
        ax = fig.add_axes([0.25, 0.1, 0.7, 1 - 1.6 / h])
        ax.imshow(ct.values, cmap=th.cmap(), aspect="auto")
        for (i, j), v in np.ndenumerate(ct.values):
            light = v > ct.values.max() * 0.55
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11, fontweight="bold",
                    color=th.bg if light else th.text)
        ax.set_xticks(range(len(ct.columns)), ct.columns, fontsize=9.5, color=th.text)
        ax.set_yticks(range(len(ct.index)), ct.index, fontsize=9.5, color=th.text)
        ax.xaxis.tick_top()
        ax.set_xlabel(b, color=th.muted, fontsize=9)
        ax.xaxis.set_label_position("top")
        ax.set_ylabel(a, color=th.muted, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    return fig


def outcome_by(sd: SetPieceData, row: str, by: str, *, theme: Theme = DARK):
    th = theme
    out_c = A._col(sd, "Outcome")
    g = sd.of(row=row)
    with _rc(th):
        if not out_c or by not in sd.qualifiers:
            fig = _figure(th, (9, 3), f"{row} — outcome by {by}", "no outcome column")
            return fig
        g = g[g[by] != ""]
        ct = pd.crosstab(g[by], g[out_c].replace("", "No outcome coded"))
        h = 2.0 + 0.6 * max(1, len(ct.index))
        fig = _figure(th, (10, h), f"{sd.team} · {row} — outcome by {by}", "every coded instance, stacked by outcome")
        ax = fig.add_axes([0.22, 0.16, 0.72, 1 - 1.6 / h])
        left = np.zeros(len(ct.index))
        palette = [th.team, th.carry, th.good, th.accent, th.opp]
        for i, oc in enumerate(ct.columns):
            col = th.muted if oc == "No outcome coded" else OUT_COLORS.get(oc.lower(), palette[i % len(palette)])
            vals = ct[oc].to_numpy(float)
            ax.barh(range(len(ct.index)), vals, left=left, color=col, height=0.6, label=oc, edgecolor=th.bg, lw=1)
            for yi, (l_, v) in enumerate(zip(left, vals)):
                if v:
                    ax.text(l_ + v / 2, yi, str(int(v)), ha="center", va="center", fontsize=9, fontweight="bold",
                            color="#ffffff")
            left += vals
        ax.set_yticks(range(len(ct.index)), ct.index, fontsize=10, color=th.text)
        ax.set_xticks([])
        ax.invert_yaxis()
        _style_ax(ax, th)
        ax.spines["bottom"].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.02), ncol=min(5, len(ct.columns)), frameon=False,
                  fontsize=9, labelcolor=th.text)
    return fig


def report_card(sd: SetPieceData, row: str, *, theme: Theme = DARK):
    """One page per set piece: landing map + the key numbers + every qualifier, ready for an opposition report."""
    th = theme
    g = sd.of(row=row)
    ins = dict(A.insights(sd)).get(row, "")
    lmap = landing(sd, row, color_by=A._col(sd, "Outcome"), theme=th)
    qual = qualifiers(sd, row, theme=th)
    a, b = png(lmap, 200), png(qual, 200)
    from PIL import Image
    ia, ib = Image.open(io.BytesIO(a)), Image.open(io.BytesIO(b))
    with _rc(th):
        W = 16.0
        ha = W * 0.48 * ia.height / ia.width
        hb = W * 0.5 * ib.height / ib.width
        text = textwrap.wrap(ins, 150)
        H = 1.3 + 0.25 * len(text) + max(ha, hb) + 0.3
        fig = _figure(th, (W, H), f"{sd.team} · {row}", f"{len(g)} coded   ·   {g['phase'].iloc[0] if len(g) else ''} for {sd.team}")
        for j, line in enumerate(text):
            fig.text(0.03, 1 - (1.0 + 0.25 * j) / H, line, ha="left", va="top", fontsize=10, color=th.text)
        top = 1 - (1.2 + 0.25 * len(text)) / H
        ax1 = fig.add_axes([0.01, top - ha / H, 0.48, ha / H])
        ax1.imshow(ia)
        ax1.axis("off")
        ax2 = fig.add_axes([0.5, top - hb / H, 0.49, hb / H])
        ax2.imshow(ib)
        ax2.axis("off")
    return fig


def category_compare(sd: SetPieceData, cat: str, *, theme: Theme = DARK, max_quals: int = 12):
    """Attacking vs defending of one category, every qualifier as % of that phase — with the key findings."""
    import re as _re
    th = theme
    pc = A.phase_compare(sd, cat)
    plural = A.CATEGORY_PLURAL.get(cat, cat)
    quals = A.ordered_quals(sd, [q for q in sd.qualifiers if not pc.empty and q in set(pc["qualifier"])])[:max_quals]
    ins = [_re.sub(r"\*\*", "", t) for t in A.category_insights(sd, cat)]
    lines = [w for t in ins for w in textwrap.wrap(t, 150)]
    cols = min(3, max(1, len(quals)))
    rows_ = int(np.ceil(len(quals) / cols)) if quals else 0
    heights = []
    for r in range(rows_):
        grp = quals[r * cols:(r + 1) * cols]
        heights.append(max(pc[pc["qualifier"] == q]["value"].nunique() for q in grp))
    head = 1.3 + 0.24 * len(lines)
    H = head + sum(0.8 + 0.42 * h for h in heights) + 0.3
    with _rc(th):
        fig = _figure(th, (16, H), f"{sd.team} — {plural}: taking vs defending",
                      f"blue = {sd.team} taking {plural.lower()}   ·   red = {sd.team} defending them   ·   "
                      "bars = % of that phase's coded instances, label = count")
        for j, line in enumerate(lines):
            fig.text(0.03, 1 - (1.05 + 0.24 * j) / H, line, ha="left", va="top", fontsize=10, color=th.text)
        top = 1 - head / H
        for r in range(rows_):
            hgt = 0.42 * heights[r] / H
            for c in range(cols):
                i = r * cols + c
                if i >= len(quals):
                    continue
                q = quals[i]
                g = pc[pc["qualifier"] == q]
                order = g.groupby("value")["count"].sum().sort_values().index.tolist()
                ax = fig.add_axes([0.1 + c * 0.315, top - hgt - 0.45 / H, 0.2, hgt])
                yy = np.arange(len(order))
                for k, (ph, col, off) in enumerate((("Attacking", th.team, 0.2), ("Defending", th.opp, -0.2))):
                    d = g[g["phase"] == ph].set_index("value").reindex(order)
                    sh, cn = d["share"].fillna(0).to_numpy(), d["count"].fillna(0).to_numpy()
                    ax.barh(yy + off, sh, height=0.38, color=col)
                    for yi, v, n_ in zip(yy + off, sh, cn):
                        if n_:
                            ax.text(v + 2, yi, f"{int(n_)}", va="center", fontsize=8, color=th.text)
                ax.set_yticks(yy, order, fontsize=9, color=th.text)
                ax.set_xlim(0, 125)
                ax.set_ylim(-0.7, max(len(order), 2) - 0.3)
                ax.set_xticks([])
                _style_ax(ax, th)
                ax.spines["bottom"].set_visible(False)
                _qual_title(ax, sd, q, th)
            top -= hgt + 0.8 / H
    return fig


def contact_map(sd: SetPieceData, row: str, which: int, *, theme: Theme = DARK):
    """Static 1st-contact / 2nd-contact outcome map (● team won the contact, ✕ opponent won it)."""
    th = theme
    outc, won = A.contact_columns(sd, which)
    pts = A.contact_points(sd, row, which).copy()
    g = sd.of(row=row)
    name = "1st contact" if which == 1 else "2nd contact (second ball)"
    flip = bool(len(pts)) and bool(pts["cx"].mean() < PITCH_L / 2)
    if flip:
        pts["cx"], pts["cy"] = PITCH_L - pts["cx"], PITCH_W - pts["cy"]
    with _rc(th):
        pitch = VerticalPitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W, half=True,
                              pitch_color=th.pitch, line_color=th.line, linewidth=1.2, goal_type="box",
                              corner_arcs=True, pad_top=2, pad_bottom=2, pad_left=2, pad_right=2)
        fig = _figure(th, (8.6, 8.6), f"{sd.team} · {row} — {name} outcome",
                      f"{len(pts)} of {len(g)} located   ·   colour = {outc or 'outcome'}   ·   "
                      f"● {sd.team} won it   ✕ opponent won it")
        ax = fig.add_axes([0.03, 0.12, 0.94, 0.74])
        pitch.draw(ax=ax)
        if len(pts) and bool(pts["cx"].min() >= 72):
            ax.set_ylim(68, PITCH_L + 3)
        if pts.empty:
            ax.text(PITCH_W / 2, PITCH_L * 0.75, f"No {name} locations coded", ha="center", va="center",
                    color=th.muted, fontsize=13)
            return fig
        vals = pts[outc].replace("", "Not coded") if outc else pd.Series("Not coded", index=pts.index)
        palette = [th.team, th.accent, th.good, th.carry, th.opp, th.bad]
        from matplotlib.lines import Line2D
        handles = []
        for i, v in enumerate(vals.value_counts().index):
            d = pts[vals == v]
            col = th.muted if v == "Not coded" else A.OUTCOME_COLORS.get(v.lower(), palette[i % len(palette)])
            yn = d[won].map(is_yes) if won else pd.Series(None, index=d.index)
            for marker, mask in (("o", yn.map(lambda z: z is True)), ("X", yn.map(lambda z: z is False)),
                                 ("o", yn.map(lambda z: z is None))):
                dd = d[mask.astype(bool)]
                if dd.empty:
                    continue
                hollow = marker == "o" and mask is not None and dd.index.isin(d[yn.map(lambda z: z is None)].index).all() \
                    and not dd.index.isin(d[yn.map(lambda z: z is True)].index).any()
                pitch.scatter(dd["cx"], dd["cy"], ax=ax, s=260, marker=marker,
                              color="none" if hollow else col, edgecolors=col if hollow else th.bg,
                              linewidth=2 if hollow else 1.2, zorder=3)
                for _, r in dd.iterrows():
                    pitch.annotate(str(r["n"]), (r["cx"] + 1.6, r["cy"]), ax=ax, ha="center", va="center", fontsize=7.5,
                                   color=th.muted, zorder=4)
            handles.append(Line2D([0], [0], marker="o", lw=0, markersize=9, color=col, label=f"{v} ({len(d)})"))
        handles += [Line2D([0], [0], marker="o", lw=0, markersize=9, color=th.text, label=f"{sd.team} won"),
                    Line2D([0], [0], marker="X", lw=0, markersize=9, color=th.text, label="Opponent won")]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False,
                  fontsize=8.5, labelcolor=th.text, handletextpad=0.3, columnspacing=1.0)
    return fig


def compare_groups(sd: SetPieceData, groups: list, title: str, subtitle: str = "", lines: list[str] | None = None, *,
                   theme: Theme = DARK, max_quals: int = 12):
    """Static version of charts.compare_groups (+ optional findings above the bars)."""
    import re as _re
    th = theme
    groups = [(lab, g) for lab, g in groups if len(g)]
    pc = A.group_compare(sd, groups)
    quals = A.ordered_quals(sd, [q for q in sd.qualifiers if not pc.empty and q in set(pc["qualifier"])])[:max_quals]
    text = [w for t in (lines or []) for w in textwrap.wrap(_re.sub(r"\*\*", "", t), 150)]
    cols = min(3, max(1, len(quals)))
    rows_ = int(np.ceil(len(quals) / cols)) if quals else 0
    heights = [max(pc[pc["qualifier"] == q]["value"].nunique() for q in quals[r * cols:(r + 1) * cols]) for r in range(rows_)]
    head = 1.3 + 0.24 * len(text)
    H = head + sum(0.8 + 0.42 * h for h in heights) + 0.3
    colors = [th.team, th.accent, th.good, th.carry]
    with _rc(th):
        fig = _figure(th, (16, H), title, subtitle or "   ·   ".join(
            f"{lab} ({len(g)})" for lab, g in groups) + "   ·   bars = % of the group's coded instances, label = count")
        for j, line in enumerate(text):
            fig.text(0.03, 1 - (1.05 + 0.24 * j) / H, line, ha="left", va="top", fontsize=10, color=th.text)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=colors[k % len(colors)], label=lab) for k, (lab, _) in enumerate(groups)],
                   loc="upper right", bbox_to_anchor=(0.98, 1 - 0.2 / H), frameon=False, labelcolor=th.text, ncol=len(groups))
        top = 1 - head / H
        nb = len(groups)
        for r in range(rows_):
            hgt = 0.42 * heights[r] / H
            for c in range(cols):
                i = r * cols + c
                if i >= len(quals):
                    continue
                q = quals[i]
                g = pc[pc["qualifier"] == q]
                order = g.groupby("value")["count"].sum().sort_values().index.tolist()
                ax = fig.add_axes([0.1 + c * 0.315, top - hgt - 0.45 / H, 0.2, hgt])
                yy = np.arange(len(order))
                bw = 0.8 / nb
                for k, (lab, _) in enumerate(groups):
                    d = g[g["phase"] == lab].set_index("value").reindex(order)
                    off = (nb - 1) / 2 * bw - k * bw
                    sh, cn = d["share"].fillna(0).to_numpy(), d["count"].fillna(0).to_numpy()
                    ax.barh(yy + off, sh, height=bw * 0.95, color=colors[k % len(colors)])
                    for yi, v, n_ in zip(yy + off, sh, cn):
                        if n_:
                            ax.text(v + 2, yi, f"{int(n_)}", va="center", fontsize=8, color=th.text)
                ax.set_yticks(yy, order, fontsize=9, color=th.text)
                ax.set_xlim(0, 125)
                ax.set_ylim(-0.7, max(len(order), 2) - 0.3)
                ax.set_xticks([])
                _style_ax(ax, th)
                ax.spines["bottom"].set_visible(False)
                _qual_title(ax, sd, q, th)
            top -= hgt + 0.8 / H
    return fig


def takers_chart(sd: SetPieceData, g, title: str, *, theme: Theme = DARK):
    th = theme
    t = A.takers(sd, g)
    with _rc(th):
        h = 1.8 + 0.5 * max(1, len(t))
        fig = _figure(th, (11, h), title, f"{sd.team} takers by shirt number   ·   bar = set pieces taken   ·   "
                      "red = led to a goal / shot / threat")
        ax = fig.add_axes([0.17, 0.12, 0.57, 1 - 1.5 / h])
        if t.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No taker coded", ha="center", color=th.muted, fontsize=12)
            return fig
        t = t.iloc[::-1]
        y = np.arange(len(t))
        dang = t["Dangerous"].to_numpy() if "Dangerous" in t else np.zeros(len(t))
        ax.barh(y, dang, color=th.bad, height=0.6)
        ax.barh(y, t["Taken"] - dang, left=dang, color=th.team, height=0.6)
        for yi, (_, r) in zip(y, t.iterrows()):
            bits = [f"{int(r['Taken'])} taken"]
            if r.get("1st contact decided"):
                bits.append(f"1st contact {int(r['1st contact won'])}/{int(r['1st contact decided'])}")
            if "Dangerous" in r:
                bits.append(f"danger {int(r['Dangerous'])}")
            if r.get("Goals"):
                bits.append(f"{int(r['Goals'])} goal(s)")
            bits.append(f"L {r['Left']} · R {r['Right']}")
            ax.text(r["Taken"] + t["Taken"].max() * 0.02, yi, "   ·   ".join(bits), va="center", fontsize=9, color=th.text)
        ax.set_yticks(y, t["Taker"], fontsize=11, color=th.text, fontweight="bold")
        ax.set_ylabel("Taker (shirt number)", color=th.muted, fontsize=10, labelpad=10)
        ax.set_xlim(0, t["Taken"].max() * 1.05)
        ax.set_xticks([])
        _style_ax(ax, th)
        ax.spines["bottom"].set_visible(False)
        ax.set_clip_on(False)
    return fig


def goal_placement(sd: SetPieceData, row: str, *, theme: Theme = DARK):
    from matplotlib.patches import Rectangle
    th = theme
    W_, H_ = 7.32, 2.44
    g = sd.of(row=row)
    pts = g[g["goal_x_m"].notna()] if "goal_x_m" in g else g.iloc[0:0]
    rc = A.result_col(sd)
    with _rc(th):
        fig = _figure(th, (10, 5.2), f"{sd.team} · {row} — placement in the goal",
                      f"{len(pts)} of {len(g)} located   ·   as seen from the taker   ·   metres from the left post / ground")
        ax = fig.add_axes([0.05, 0.12, 0.9, 0.66])
        ax.add_patch(Rectangle((-1.2, -0.4), W_ + 2.4, 0.4, color="#3b8746"))
        for k in (1, 2):
            ax.plot([W_ * k / 3] * 2, [0, H_], color=th.grid, lw=1, ls=(0, (3, 3)))
            ax.plot([0, W_], [H_ * k / 3] * 2, color=th.grid, lw=1, ls=(0, (3, 3)))
        ax.plot([0, 0, W_, W_], [0, H_, H_, 0], color=th.text, lw=6, solid_capstyle="butt")
        palette = [th.team, th.accent, th.good, th.carry]
        if len(pts):
            vals = pts[rc].replace("", "Not coded") if rc else pd.Series("Coded", index=pts.index)
            for i, v in enumerate(vals.value_counts().index):
                d = pts[vals == v]
                col = A.OUTCOME_COLORS.get(v.lower(), palette[i % len(palette)])
                ax.scatter(d["goal_x_m"], d["goal_y_m"], s=420, color=col, edgecolors=th.bg, lw=1.5, zorder=3,
                           label=f"{v} ({len(d)})")
                for _, r in d.iterrows():
                    if r["taker"]:
                        ax.text(r["goal_x_m"] + 0.32, r["goal_y_m"], f"#{r['taker']}", ha="left", va="center",
                                fontsize=10, color=th.text, fontweight="bold", zorder=4)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, labelcolor=th.text)
        else:
            ax.text(W_ / 2, H_ / 2, "No goal placement coded", ha="center", color=th.muted, fontsize=13)
        ax.set_xlim(-1.2, W_ + 1.2)
        ax.set_ylim(-0.4, H_ + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
    return fig
