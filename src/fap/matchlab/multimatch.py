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
                "Shots", "On target", "Goals", "xG", "PPDA", "Def. action height (m)", "Recoveries", "Possessions",
                "Passes / possession", "Interceptions", "Defensive actions", "Aerial duels won %", "PPDA (possessions)",
                "Seconds per def. action", "High regains", "High regains (final third)",
                "Opp. possessions won back in their half %")
# rows of the team comparison, grouped — and the metrics where LOWER is better (colouring / radar)
TEAM_GROUPS = {
    "Possession": ("Possession % (passes)", "Field tilt %", "Passes", "Pass accuracy %", "Possessions",
                   "Passes / possession", "Long balls %"),
    "Build-up": ("Build-ups", "Build-up passes / possession", "Build-up exit success %", "Build-up exits by long ball %",
                 "Build-ups leaving own third %", "Build-ups reaching final third %", "Build-ups ending in a shot %",
                 "Losses in own third"),
    "Progression": ("Progressive passes", "Progressive carries", "Line-breaking passes", "Switches of play",
                    "Final-third entries", "Final-third entries (pass + carry)", "Final-third entry success %",
                    "F3 entries by carry %", "F3 entries left %", "F3 entries centre %", "F3 entries right %",
                    "Time to final third (s)"),
    "Box entries": ("Box entries", "Box entry attempts", "Box entry success %", "Box entries by pass",
                    "Box entries by carry", "Box entries by cross", "Box entries per F3 entry %", "Touches in box"),
    "Attacking": ("Shots", "On target", "Goals", "xG", "xG per shot"),
    "Pressing": ("PPDA", "PPDA (possessions)", "Seconds per def. action", "High regains", "High regains (final third)",
                 "Opp. possessions won back in their half %"),
    "Defending": ("Shots against", "xG against", "Goals against", "Def. action height (m)", "Defensive actions",
                  "Recoveries", "Interceptions", "Aerial duels won %", "Second balls won %"),
    "Transitions": ("Counter-pressed %", "Regained in window %"),
}
LOWER_BETTER = {"PPDA", "PPDA (possessions)", "Seconds per def. action", "Shots against", "xG against", "Goals against", "Losses in own third", "Ball losses"}
NEUTRAL = {"Long balls %", "Possessions", "Build-ups", "Actions", "Carries", "Build-up passes / possession",
           "Build-up exits by long ball %", "F3 entries by carry %", "F3 entries left %", "F3 entries centre %",
           "F3 entries right %", "Time to final third (s)", "Box entries by pass", "Box entries by carry",
           "Box entries by cross"}   # style, not better / worse


def metric_group(name: str) -> str:
    return next((g for g, ms in TEAM_GROUPS.items() if name in ms), "Other")
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
    if opp.empty:                      # opponent not tagged → shares cannot be measured (PPDA falls back to possessions)
        for k in ("Possession % (passes)", "Field tilt %"):
            out[k] = np.nan
    b = T.buildup(md, team, half)
    bp = b.get("possessions", pd.DataFrame())
    if len(bp):
        out["Build-ups"] = len(bp)
        out["Build-ups leaving own third %"] = float((bp["max_x"] >= md.thirds[0]).mean() * 100)
        out["Build-ups reaching final third %"] = float(bp["reached_f3"].mean() * 100)
        out["Losses in own third"] = len(b.get("losses", []))
        out["Build-up passes / possession"] = float(bp["n_passes"].mean())
        out["Build-ups ending in a shot %"] = float(bp["shot"].mean() * 100)
        ex = b.get("exits", pd.DataFrame())
        if len(ex):
            out["Build-up exit success %"] = float(ex["success"].mean() * 100)
            out["Build-up exits by long ball %"] = float((ex["method"] == "Long pass").mean() * 100)
    out.update(_progression_rows(md, team, half))
    opp_team = md.opponent(team)
    opp_ev = md.events[md.events["team"] == opp_team]
    if half is not None:
        opp_ev = opp_ev[opp_ev["half"] == half]
    if len(opp_ev[opp_ev["etype"] == "shot"]) or len(opp):        # opponent tagged → what we conceded
        o = M.team_summary(md, opp_team, half)
        out["Shots against"], out["xG against"], out["Goals against"] = o.get("Shots"), o.get("xG"), o.get("Goals")
    else:
        out["Shots against"] = out["xG against"] = out["Goals against"] = np.nan
    out["xG per shot"] = (out["xG"] / out["Shots"]) if out.get("Shots") else np.nan
    sb = SB.summary(md, team, half)
    out["Second balls won %"] = sb.get("won %")
    cps = CP.summary(CP.analyse(md, team, md.cp_window, half))
    out["Counter-pressed %"] = cps.get("pressed_pct")
    out["Regained in window %"] = cps.get("regained_pct")
    return out


def _progression_rows(md: MatchData, team: str, half) -> dict:
    """Progression into the final third and entries into the box (passes + carries) for the team comparison."""
    from fap.matchlab import team as T
    from fap.matchlab.metrics import in_box
    out: dict = {}
    pr = T.progression(md, team, half)
    en = pr.get("entries", pd.DataFrame())
    out["Switches of play"] = int((pr.get("switches", pd.DataFrame()).get("outcome", pd.Series(dtype=float)) == 1).sum())
    if len(en):
        ok = en[en["success"]]
        out["Final-third entries (pass + carry)"] = len(ok)
        out["Final-third entry success %"] = float(en["success"].mean() * 100)
        if len(ok):
            out["F3 entries by carry %"] = float((ok["method"] == "Carry").mean() * 100)
            ch = ok["end_channel"].astype(str)
            out["F3 entries left %"] = float(ch.str.startswith("Left").mean() * 100)
            out["F3 entries centre %"] = float((ch == "Centre").mean() * 100)
            out["F3 entries right %"] = float(ch.str.startswith("Right").mean() * 100)
    else:
        out["Final-third entries (pass + carry)"] = 0
    tt = pr.get("time_to_f3", pd.DataFrame())
    if len(tt):
        out["Time to final third (s)"] = float(tt["seconds"].median())
    ev = md.events if half is None else md.events[md.events["half"] == half]
    me = ev[ev["team"] == team]
    bx = me[me["etype"].isin(["pass", "carry"]) & me["end_x"].notna() & me["box_entry"]]
    succ = (bx["outcome"] == 1) | (bx["etype"] == "carry")
    out["Box entry attempts"] = len(bx)
    out["Box entry success %"] = float(succ.mean() * 100) if len(bx) else np.nan
    out["Box entries by pass"] = int((succ & (bx["etype"] == "pass") & ~bx["cross"].astype(bool)).sum())
    out["Box entries by carry"] = int((bx["etype"] == "carry").sum())
    out["Box entries by cross"] = int((succ & (bx["etype"] == "pass") & bx["cross"].astype(bool)).sum())
    f3 = out.get("Final-third entries (pass + carry)") or 0
    out["Box entries per F3 entry %"] = (int(succ.sum()) / f3 * 100) if f3 else np.nan
    touch = me[me["etype"].isin(["receive", "pass", "carry", "dribble", "shot"]) & me["x"].notna() & ~me["derived"].astype(bool)]
    out["Touches in box"] = int(in_box(touch["x"], touch["y"]).sum())
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
    df = pd.DataFrame(cols).apply(pd.to_numeric, errors="coerce")
    if not player:                                     # grouped, in a fixed professional order
        order = [m for ms in TEAM_GROUPS.values() for m in ms if m in df.index]
        df = df.reindex(order + [m for m in df.index if m not in order])
    matches = [it.label for it in items]
    df.insert(len(matches), "Match average", df[matches].mean(axis=1))
    if not player and "All matches" in df:             # shares need the opponent tagged — only those matches count
        for k in ("Possession % (passes)", "Field tilt %"):
            if k in df.index and df.loc[k, matches].isna().any():
                df.at[k, "All matches"] = df.loc[k, matches].mean()
        if "PPDA" in df.index:                         # matches may mix the two PPDA methods → average, not pooled
            df.at["PPDA", "All matches"] = df.loc["PPDA", matches].mean()
    return df


# ------------------------------------------------------------------ static outputs
def _theme(theme):
    from fap.matchlab.charts import DARK
    return theme or DARK


def trend_figure(table: pd.DataFrame, metrics: list[str], title: str, subtitle: str = "", theme=None):
    """Small multiples: one line per metric across the matches (dashed = average)."""
    import matplotlib.pyplot as plt
    from fap.matchlab.charts import FONT
    th = _theme(theme)
    t = table.drop(columns=["All matches", "Match average"], errors="ignore")
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


def scorecard_figure(table: pd.DataFrame, title: str, subtitle: str = "", theme=None, lower_better=LOWER_BETTER):
    """Match-by-match scorecard: one column per match, cells coloured better (green) / worse (red) than the
    average of the matches for that metric, grouped by phase of play."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    from fap.matchlab.charts import FONT
    th = _theme(theme)
    matches = [c for c in table.columns if c not in ("All matches", "Match average")]
    cols = matches + [c for c in ("Match average", "All matches") if c in table.columns]
    groups: list[tuple[str, list[str]]] = []
    for m in table.index:
        g = metric_group(m)
        if not groups or groups[-1][0] != g:
            groups.append((g, []))
        groups[-1][1].append(m)
    n_rows = len(table.index) + len(groups)
    cw, rh, lw = 1.35, 0.34, 3.2
    W = lw + cw * len(cols) + 0.5
    H = 1.5 + rh * (n_rows + 1) + 0.4
    cmap = LinearSegmentedColormap.from_list("sc", [th.bad, th.panel, th.good])
    with plt.rc_context({"font.family": FONT}):
        fig = plt.figure(figsize=(W, H), facecolor=th.bg)
        fig.text(0.3 / W, 1 - 0.25 / H, title, ha="left", va="top", fontsize=17, fontweight="bold", color=th.text)
        fig.text(0.3 / W, 1 - 0.68 / H, subtitle or "green = better than the average of these matches · red = worse",
                 ha="left", va="top", fontsize=9.5, color=th.muted)
        ax = fig.add_axes([0, 0.4 / H, 1, 1 - 1.5 / H])
        ax.set_xlim(0, W)
        ax.set_ylim(-(n_rows + 1) * rh, 0)
        ax.axis("off")
        for j, c in enumerate(cols):
            ax.text(lw + (j + 0.5) * cw, -rh / 2, textwrap.fill(c, 14), ha="center", va="center", fontsize=8.5,
                    color=th.accent if c in ("Match average", "All matches") else th.text, fontweight="bold")
        y = -rh
        for g, ms in groups:
            ax.text(0.3, y - rh / 2, g.upper(), ha="left", va="center", fontsize=8.5, fontweight="bold", color=th.accent)
            ax.plot([0.3, W - 0.3], [y - rh, y - rh], color=th.grid, lw=0.6)
            y -= rh
            for m in ms:
                vals = table.loc[m, matches].to_numpy(float)
                mu, sd = np.nanmean(vals) if np.isfinite(vals).any() else np.nan, np.nanstd(vals) if np.isfinite(vals).any() else np.nan
                ax.text(0.3, y - rh / 2, m, ha="left", va="center", fontsize=9, color=th.text)
                for j, c in enumerate(cols):
                    v = table.at[m, c]
                    if c in matches and np.isfinite(v) and sd and np.isfinite(sd) and m not in NEUTRAL:
                        z = (v - mu) / sd * (-1 if m in lower_better else 1)
                        ax.add_patch(Rectangle((lw + j * cw + 0.04, y - rh + 0.03), cw - 0.08, rh - 0.06,
                                               color=cmap(0.5 + np.clip(z, -2, 2) / 4), lw=0))
                    txt = "–" if not np.isfinite(v) else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}" if abs(v) < 3 and v != int(v) else f"{v:.1f}")
                    ax.text(lw + (j + 0.5) * cw, y - rh / 2, txt, ha="center", va="center", fontsize=9,
                            color=th.text, fontweight="bold" if c in ("Match average", "All matches") else "normal")
                y -= rh
    return fig


def match_radar(table: pd.DataFrame, metrics: list[str], title: str, subtitle: str = "", theme=None,
                lower_better=LOWER_BETTER):
    """Each match as a shape: every metric scaled 0 (worst of these matches) → 100 (best)."""
    import matplotlib.pyplot as plt
    from fap.matchlab.charts import FONT
    th = _theme(theme)
    matches = [c for c in table.columns if c not in ("All matches", "Match average")]
    ms = [m for m in metrics if m in table.index and np.isfinite(table.loc[m, matches].to_numpy(float)).sum() >= 2]
    n = max(len(ms), 3)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    colors = ["#3ab0ff", "#ff5d73", "#ffc94a", "#2ed8a3", "#b18cff", "#f97316", "#e5e7eb"]
    with plt.rc_context({"font.family": FONT}):
        fig = plt.figure(figsize=(10.5, 10), facecolor=th.bg)
        fig.text(0.03, 0.97, title, ha="left", va="top", fontsize=17, fontweight="bold", color=th.text)
        fig.text(0.03, 0.93, subtitle or "each metric scaled across these matches: centre = worst, edge = best",
                 ha="left", va="top", fontsize=9.5, color=th.muted)
        ax = fig.add_axes([0.13, 0.1, 0.74, 0.74], polar=True)
        ax.set_facecolor(th.bg)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for ring in (25, 50, 75, 100):
            ax.plot(np.r_[ang, ang[:1]], [ring] * (n + 1), color=th.grid, lw=0.8)
        for a in ang:
            ax.plot([a, a], [0, 100], color=th.grid, lw=0.6)
        for i, c in enumerate(matches):
            vals = []
            for m in ms:
                row = table.loc[m, matches].to_numpy(float)
                lo, hi = np.nanmin(row), np.nanmax(row)
                v = table.at[m, c]
                sc = 50.0 if not np.isfinite(v) or hi == lo else (v - lo) / (hi - lo) * 100
                vals.append(100 - sc if m in lower_better else sc)
            vals = np.array(vals + [50.0] * (n - len(vals)))
            col = colors[i % len(colors)]
            ax.fill(np.r_[ang, ang[:1]], np.r_[vals, vals[:1]], color=col, alpha=0.12)
            ax.plot(np.r_[ang, ang[:1]], np.r_[vals, vals[:1]], color=col, lw=2.2, label=c)
        ax.set_xticks(ang[:len(ms)], [textwrap.fill(m, 14) for m in ms], fontsize=9, color=th.text)
        ax.set_yticks([])
        ax.set_ylim(0, 100)
        ax.spines["polar"].set_visible(False)
        ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=min(4, len(matches)), frameon=False,
                  labelcolor=th.text, fontsize=10)
    return fig
