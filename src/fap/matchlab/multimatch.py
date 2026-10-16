"""Multi-match comparison — the same Match Lab charts and numbers across several matches.

* :func:`combine` merges several loaded matches into ONE :class:`MatchData` (our team keeps its name,
  every opponent becomes "Opponents"; times / ids / possession ids are offset so nothing chains across
  matches), so every existing chart and analysis runs unchanged on "all matches together".
* :func:`player_options` — our players across the matches, labelled with the squad's player id + name.
* :func:`comparison_table` — one column per match + "All matches" for the team or one player.
* :func:`trend_figure`, :func:`grid_png`, :func:`pdf_report` — static outputs (PNG / PDF).
"""
from __future__ import annotations

import io
import math
import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fap.matchlab.loader import MatchData

OPP = "Opponents"
_T_OFFSET = 100_000.0


@dataclass
class MatchItem:
    label: str
    md: MatchData
    sig: str


def default_label(md: MatchData, i: int, name: str = "") -> str:
    opp = md.opponent(md.our_team)
    return f"{i + 1}. vs {opp}" if opp else (name.rsplit(".", 1)[0] or f"Match {i + 1}")


# ------------------------------------------------------------------ combine
def combine(items: list[MatchItem]) -> MatchData:
    """All matches as one MatchData (our team vs "Opponents")."""
    our = items[0].md.our_team
    evs, poss = [], []
    pid_off = 0
    for i, it in enumerate(items):
        md = it.md
        me = md.our_team
        e = md.events.copy()
        e["match"] = it.label
        e["match_idx"] = i
        e["team"] = np.where(e["team"].eq(me), our, np.where(e["team"].ne(""), OPP, ""))
        e["poss_team"] = np.where(e["poss_team"].eq(me), our, np.where(e["poss_team"].ne(""), OPP, ""))
        e["t"] = e["t"] + i * _T_OFFSET
        if "event_id" in e:
            e["event_id"] = pd.to_numeric(e["event_id"], errors="coerce") + i * 1_000_000
        valid = e["poss_id"] >= 0
        e.loc[valid, "poss_id"] = e.loc[valid, "poss_id"] + pid_off
        p = md.possessions.copy()
        if not p.empty:
            p["poss_id"] = p["poss_id"] + pid_off
            p["team"] = np.where(p["team"].eq(me), our, OPP)
            p["match"] = it.label
            pid_off = int(p["poss_id"].max()) + 1
        elif valid.any():
            pid_off = int(e.loc[valid, "poss_id"].max()) + 1
        evs.append(e)
        poss.append(p)
    ev = pd.concat(evs, ignore_index=True)
    ps = pd.concat([p for p in poss if not p.empty], ignore_index=True) if any(not p.empty for p in poss) else pd.DataFrame()
    first = items[0].md
    return MatchData(events=ev, teams=[our, OPP], source_name=f"{len(items)} matches", possessions=ps,
                     thirds=first.thirds, cp_window=first.cp_window,
                     notes=[f"Combined: {', '.join(it.label for it in items)}"])


# ------------------------------------------------------------------ players
def player_options(items: list[MatchItem], squad=None) -> list[dict]:
    """Our players across the matches: {key (shirt), label, id, name, matches: [match indexes]}.
    The squad maps shirt → (player_id, name), so the same player is recognised in every match."""
    seen: dict[str, set] = {}
    for i, it in enumerate(items):
        ev = it.md.team_events(it.md.our_team)
        for p in ev["player"].unique():
            if p and p != "0":
                seen.setdefault(str(p), set()).add(i)
    out = []
    for shirt, ms in seen.items():
        pid, name = squad.lookup(shirt) if squad is not None else ("", "")
        label = f"{name} (#{shirt})" if name else f"#{shirt}"
        if pid:
            label += f" · ID {pid}"
        out.append({"key": shirt, "label": label, "id": pid, "name": name, "matches": sorted(ms)})
    return sorted(out, key=lambda d: (-len(d["matches"]), len(d["key"]), d["key"]))


# ------------------------------------------------------------------ comparison table
TEAM_METRICS = ("Possession % (passes)", "Field tilt %", "Passes", "Pass accuracy %", "Progressive passes",
                "Progressive carries", "Final-third entries", "Box entries", "Line-breaking passes", "Long balls %",
                "Shots", "On target", "Goals", "xG", "PPDA", "Def. action height (m)", "Recoveries")
PLAYER_METRICS = ("Actions", "Passes", "Pass accuracy %", "Progressive passes", "Progressive carries",
                  "Progressive receptions", "Key passes", "Carries", "Carry distance (m)", "Carries into final third",
                  "Dribbles", "Dribble success %", "Shots", "xG", "Defensive actions", "Tackles won",
                  "Interceptions won", "Duels won", "Duel success %", "Recoveries", "Second balls won", "Ball losses")


def _team_col(md: MatchData, team: str, half) -> dict:
    from fap.matchlab import counterpress as CP
    from fap.matchlab import metrics as M
    from fap.matchlab import secondball as SB
    from fap.matchlab import team as T
    a = M.team_summary(md, team, half)
    out = {k: a.get(k) for k in TEAM_METRICS}
    opp = md.events[(md.events["team"] == md.opponent(team)) & (md.events["etype"] == "pass")]
    if half is not None:
        opp = opp[opp["half"] == half]
    if opp.empty:                      # opponent not tagged → shares / PPDA cannot be measured
        for k in ("Possession % (passes)", "Field tilt %", "PPDA"):
            out[k] = np.nan
    b = T.buildup(md, team, half)
    bp = b.get("possessions", pd.DataFrame())
    if len(bp):
        out["Build-ups"] = len(bp)
        out["Build-ups leaving own third %"] = float((bp["max_x"] >= md.thirds[0]).mean() * 100)
        out["Build-ups reaching final third %"] = float(bp["reached_f3"].mean() * 100)
        out["Losses in own third"] = len(b.get("losses", []))
    sb = SB.summary(md, team, half)
    out["Second balls won %"] = sb.get("won %")
    cps = CP.summary(CP.analyse(md, team, md.cp_window, half))
    out["Counter-pressed %"] = cps.get("pressed_pct")
    out["Regained in window %"] = cps.get("regained_pct")
    return out


def _player_col(md: MatchData, team: str, player: str, half) -> dict:
    from fap.matchlab import player as PL
    ps = PL.player_stats(md, team, half)
    r = ps[ps["player"] == str(player)] if len(ps) else ps
    if r.empty:
        return {k: np.nan for k in PLAYER_METRICS}
    r = r.iloc[0]
    return {k: r.get(k, np.nan) for k in PLAYER_METRICS}


def comparison_table(items: list[MatchItem], combined: MatchData | None, *, player: str | None = None,
                     half=None) -> pd.DataFrame:
    """Rows = metrics, columns = each match (+ "All matches" computed on the combined data)."""
    cols = {}
    for it in items:
        md = it.md
        cols[it.label] = (_player_col(md, md.our_team, player, half) if player else _team_col(md, md.our_team, half))
    if combined is not None:
        our = combined.teams[0]
        cols["All matches"] = _player_col(combined, our, player, half) if player else _team_col(combined, our, half)
    df = pd.DataFrame(cols)
    return df.apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------------ static outputs
def _theme(theme):
    from fap.matchlab.charts import DARK
    return theme or DARK


def trend_figure(table: pd.DataFrame, metrics: list[str], title: str, subtitle: str = "", theme=None):
    """Small multiples: one line per metric across the matches (dashed = average)."""
    import matplotlib.pyplot as plt
    from fap.matchlab.charts import FONT
    th = _theme(theme)
    t = table.drop(columns=["All matches"], errors="ignore")
    metrics = [m for m in metrics if m in t.index] or list(t.index[:6])
    n = len(metrics)
    cols = 3 if n > 4 else 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    with plt.rc_context({"font.family": FONT}):
        fig = plt.figure(figsize=(5.2 * cols, 2.9 * rows + 1.2), facecolor=th.bg)
        fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=17, fontweight="bold", color=th.text)
        if subtitle:
            fig.text(0.02, 0.985 - 0.45 / (2.9 * rows + 1.2), subtitle, ha="left", va="top", fontsize=10, color=th.muted)
        top = 1 - 1.0 / (2.9 * rows + 1.2)
        gs = fig.add_gridspec(rows, cols, left=0.05, right=0.98, bottom=0.08, top=top - 0.04, hspace=0.75, wspace=0.25)
        x = np.arange(len(t.columns))
        for i, m in enumerate(metrics):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            ax.set_facecolor(th.bg)
            y = t.loc[m].to_numpy(float)
            ax.plot(x, y, color=th.team, lw=2.4, marker="o", ms=7, mec=th.bg, zorder=3)
            if np.isfinite(y).any():
                avg = np.nanmean(y)
                ax.axhline(avg, color=th.muted, lw=1, ls=(0, (4, 3)), zorder=1)
                for xi, yi in zip(x, y):
                    if np.isfinite(yi):
                        ax.annotate(f"{yi:.0f}" if abs(yi) >= 10 else f"{yi:.1f}", (xi, yi), xytext=(0, 7),
                                    textcoords="offset points", ha="center", fontsize=8.5, color=th.text)
                lo, hi = np.nanmin(y), np.nanmax(y)
                pad = (hi - lo) * 0.25 or max(abs(hi), 1) * 0.2
                ax.set_ylim(lo - pad, hi + pad * 1.6)
            ax.set_title(f"{m}   (avg {np.nanmean(y):.1f})" if np.isfinite(y).any() else m, loc="left",
                         fontsize=10, color=th.text, fontweight="bold")
            ax.set_xticks(x, [textwrap.shorten(c, 16, placeholder="…") for c in t.columns], fontsize=8, color=th.muted,
                          rotation=0)
            ax.tick_params(axis="y", colors=th.muted, labelsize=8, length=0)
            ax.tick_params(axis="x", length=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(th.grid)
            ax.grid(axis="y", color=th.grid, lw=0.6)
    return fig


def table_figure(table: pd.DataFrame, title: str, subtitle: str = "", theme=None):
    import matplotlib.pyplot as plt
    from fap.matchlab.charts import FONT
    th = _theme(theme)
    t = table.copy()
    cells = [[("–" if not np.isfinite(v) else f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}") for v in row]
             for row in t.to_numpy(float)]
    h = 1.3 + 0.32 * (len(t) + 1)
    with plt.rc_context({"font.family": FONT}):
        fig = plt.figure(figsize=(max(9.0, 3.2 + 1.45 * len(t.columns)), h), facecolor=th.bg)
        fig.text(0.02, 1 - 0.25 / h, title, ha="left", va="top", fontsize=16, fontweight="bold", color=th.text)
        if subtitle:
            fig.text(0.02, 1 - 0.62 / h, subtitle, ha="left", va="top", fontsize=9.5, color=th.muted)
        ax = fig.add_axes([0.01, 0.02, 0.98, 1 - 1.05 / h])
        ax.axis("off")
        tb = ax.table(cellText=cells, rowLabels=list(t.index), colLabels=list(t.columns), loc="upper center",
                      cellLoc="center", rowLoc="left")
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 1.35)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor(th.grid)
            cell.set_text_props(color=th.text)
            last = c == len(t.columns) - 1 and "All matches" in t.columns
            cell.set_facecolor(th.panel if r == 0 or c == -1 else (th.grid if last else th.bg))
            if r == 0 or c == -1:
                cell.set_text_props(color=th.text, fontweight="bold")
    return fig


def fig_png(fig, dpi: int = 200) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return buf.getvalue()


def grid_png(pngs: list[tuple[str, bytes]], title: str, subtitle: str = "", theme=None, cols: int | None = None,
             width: int = 3000) -> bytes:
    """Tile one chart per match into a single image with match captions."""
    from PIL import Image, ImageDraw, ImageFont
    th = _theme(theme)
    ims = [(lab, Image.open(io.BytesIO(b)).convert("RGB")) for lab, b in pngs]
    if not ims:
        return b""
    cols = cols or (1 if len(ims) == 1 else 2 if len(ims) in (2, 4) else 3)
    rows = math.ceil(len(ims) / cols)
    pad, cap, head = 30, 70, 150
    cw = (width - pad * (cols + 1)) // cols
    scaled = []
    for lab, im in ims:
        h = int(im.height * cw / im.width)
        scaled.append((lab, im.resize((cw, h), Image.LANCZOS)))
    rh = max(im.height for _, im in scaled) + cap
    H = head + rows * (rh + pad) + pad
    canvas = Image.new("RGB", (width, H), th.bg)
    d = ImageDraw.Draw(canvas)

    def font(sz, bold=False):
        for f in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(f, sz)
            except Exception:
                continue
        return ImageFont.load_default()
    d.text((pad, 30), title, fill=th.text, font=font(54, True))
    if subtitle:
        d.text((pad, 98), subtitle, fill=th.muted, font=font(30))
    for k, (lab, im) in enumerate(scaled):
        r, c = divmod(k, cols)
        x0 = pad + c * (cw + pad)
        y0 = head + r * (rh + pad)
        d.text((x0, y0 + 12), lab, fill=th.accent, font=font(38, True))
        canvas.paste(im, (x0, y0 + cap))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def pdf_report(title: str, subtitle: str, matches: list[str], pages: list[tuple[str, bytes]], theme=None) -> bytes:
    """A4-landscape PDF: cover + one page per (heading, PNG)."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    from fap.matchlab.charts import FONT
    th = _theme(theme)
    W, H = 11.69, 8.27
    buf = io.BytesIO()
    with plt.rc_context({"font.family": FONT}), PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(W, H), facecolor=th.bg)
        fig.text(0.07, 0.68, title, fontsize=30, fontweight="bold", color=th.text, va="top")
        fig.text(0.07, 0.585, subtitle, fontsize=14, color=th.muted, va="top")
        fig.add_artist(plt.Line2D([0.07, 0.93], [0.54, 0.54], color=th.accent, lw=2, transform=fig.transFigure))
        for i, m in enumerate(matches):
            fig.text(0.07, 0.49 - i * 0.04, m, fontsize=13, color=th.text, va="top")
        fig.text(0.07, 0.06, "MATCH LAB · multi-match comparison · pitch 105 × 68 m", fontsize=9, color=th.muted,
                 fontweight="bold")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
        for heading, png in pages:
            im = Image.open(io.BytesIO(png)).convert("RGB")
            fig = plt.figure(figsize=(W, H), facecolor=th.bg)
            fig.text(0.04, 0.965, heading, fontsize=13, fontweight="bold", color=th.muted, va="top")
            box_w, box_h = 0.92, 0.86
            ar = im.width / im.height
            if ar > (box_w * W) / (box_h * H):
                w, h = box_w, box_w * W / ar / H
            else:
                h, w = box_h, box_h * H * ar / W
            ax = fig.add_axes([(1 - w) / 2, 0.04 + (box_h - h) / 2, w, h])
            ax.imshow(np.asarray(im))
            ax.axis("off")
            pdf.savefig(fig, facecolor=fig.get_facecolor(), dpi=200)
            plt.close(fig)
    return buf.getvalue()
