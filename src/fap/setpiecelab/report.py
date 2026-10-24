"""Set-piece PDF report — compact, every set-piece type on its own pages, no overall result.

Order (A4 landscape):
  1. cover — the general numbers (taken / faced, shots, goals, 1st contact & 2nd ball);
  2. summary table — one row per set-piece type;
  3. (key insights — blue = attacking, red = defending);
  4. (match by match — one chart + one table, when the file has a match / opponent column);
  5. per set-piece type, each on its own (corners, corners against, long throws against …):
       (insights) → deliveries (1st contact outcome) + delivery zones → 2nd ball outcome + 1st-contact players
       → how it is taken / defended (every contact on its own) → funnel + shots (Shot Outcome) + takers;
     direct free kicks & penalties have no contacts: (insights) → placement in the goal + shots + takers.

Charts of one page are packed together (side by side, or stacked when they are wide) so the report stays short.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

from fap.matchlab.charts import THEMES  # noqa: E402
from fap.setpiecelab import deep as D  # noqa: E402
from fap.setpiecelab import deep_static as X  # noqa: E402
from fap.setpiecelab import static as S  # noqa: E402
from fap.setpiecelab.loader import SetPieceData  # noqa: E402

PAGE = (11.69, 8.27)
SECTIONS = ("maps", "zones", "second balls", "players", "details", "funnel", "shots", "takers", "goal", "sides")
DEFAULT_SECTIONS = tuple(s for s in SECTIONS if s != "sides")


@dataclass
class ReportOptions:
    title: str = ""
    subtitle: str = ""
    footer: str = "SET PIECE LAB"
    theme: str = "Dark"
    include_insights: bool = True
    phases: tuple = ("Attacking", "Defending")
    per_match: bool = True                    # match-by-match page when the file has a match column
    match_insights: bool = False              # short insights per match
    sections: tuple = DEFAULT_SECTIONS
    types: tuple = ()                         # restrict to these coded types (empty = all)
    dpi: int = 150
    log: list = field(default_factory=list)


@dataclass
class _Block:
    png: bytes
    page: int                                 # blocks with the same page id share one page
    section: str = ""


class _Doc:
    def __init__(self, sd: SetPieceData, opts: ReportOptions):
        self.sd, self.o = sd, opts
        self.th = THEMES.get(opts.theme, THEMES["Dark"])
        self.blocks: list[_Block] = []
        self._page = 0

    def new_page(self) -> None:
        self._page += 1

    def add(self, section: str, make) -> None:
        """Render a chart onto the current page (a failing chart is skipped and logged, never breaks the report)."""
        try:
            self.blocks.append(_Block(S.png(make(), self.o.dpi), self._page, section))
        except Exception as exc:
            self.o.log.append(f"{section}: {exc}")
            plt.close("all")

    def insights(self, items, section: str, title: str, per_page: int = 26) -> None:
        chunk, lines = [], 0
        for it in items:
            n = 1 + len(it.text) // 120 + (1 if not chunk or chunk[-1].section != it.section else 0)
            if chunk and lines + n > per_page:
                self.new_page()
                self.add(section, lambda c=list(chunk), t=title: X.insights_page(c, t, theme=self.th))
                chunk, lines = [], 0
                title = title if title.endswith("(cont.)") else title + " (cont.)"
            chunk.append(it)
            lines += n
        if chunk:
            self.new_page()
            self.add(section, lambda c=list(chunk), t=title: X.insights_page(c, t, theme=self.th))


def _cover(pdf, sd: SetPieceData, o: ReportOptions, th) -> None:
    fig = plt.figure(figsize=PAGE, facecolor=th.bg)
    a = sd.df[sd.df["phase"] == "Attacking"]
    d = sd.df[sd.df["phase"] == "Defending"]
    sa, sdd = D.summary(sd, a), D.summary(sd, d)
    fig.text(0.06, 0.8, (o.title or f"{sd.team} — set pieces").upper(), fontsize=30, fontweight="bold", color=th.text)
    fig.text(0.06, 0.73, o.subtitle or (f"{len(sd.df)} set pieces coded"
                                       + (f" · {len(D.matches(sd))} matches" if D.has_matches(sd) else "")),
             fontsize=15, color=th.muted)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.68, 0.68], color=th.accent, lw=2))
    for k, (ph, s_, col, verb) in enumerate((("ATTACKING", sa, th.team, "taken"), ("DEFENDING", sdd, th.opp, "faced"))):
        y0 = 0.58 - k * 0.27
        fig.text(0.06, y0, ph, fontsize=13, fontweight="bold", color=col)
        cards = [(verb.title(), str(s_["Count"])), ("Shots" + (" against" if k else ""), f"{s_['Shots']}"),
                 ("On target", str(s_["On target"])), ("Goals" + (" conceded" if k else ""), str(s_["Goals"])),
                 ("1st contact won", f"{s_['1st contact won']}/{s_['1st contact decided']}"),
                 ("2nd ball won", f"{s_['2nd ball won']}/{s_['2nd ball decided']}")]
        for i, (lab, val) in enumerate(cards):
            x = 0.06 + i * 0.15
            fig.text(x, y0 - 0.06, lab.upper(), fontsize=9.5, color=th.muted, fontweight="bold")
            fig.text(x, y0 - 0.14, val, fontsize=26, fontweight="bold", color=col)
    fig.text(0.06, 0.07, f"{sd.team}'s point of view   ·   attacking = {sd.team} takes it   ·   defending = against "
                         f"{sd.team}   ·   contests = won by {sd.team}   ·   every number counted from the coded file",
             fontsize=9, color=th.muted)
    if o.footer:
        fig.text(0.06, 0.03, o.footer, fontsize=8.5, color=th.muted, fontweight="bold")
    pdf.savefig(fig, facecolor=th.bg)
    plt.close(fig)


def _place(fig, imgs: list) -> None:
    """Side by side, or stacked when every chart on the page is wide."""
    asp = [im.width / im.height for im in imgs]
    top, bottom, left, right = 0.935, 0.06, 0.025, 0.975
    if len(imgs) > 1 and min(asp) > 1.55:                         # wide charts → one under the other
        H = top - bottom
        tot = sum(1 / a for a in asp)
        y = top
        for im, a in zip(imgs, asp):
            h = H * (1 / a) / tot
            ax = fig.add_axes([left, y - h, right - left, h - 0.008])
            ax.imshow(im, interpolation="lanczos")
            ax.set_anchor("N")
            ax.axis("off")
            y -= h
        return
    w = (right - left) / len(imgs)
    for k, im in enumerate(imgs):
        ax = fig.add_axes([left + k * w, bottom, w - (0.008 if len(imgs) > 1 else 0), top - bottom])
        ax.imshow(im, interpolation="lanczos")
        ax.set_anchor("N")
        ax.axis("off")


def _write(pdf, doc: _Doc, header: str) -> None:
    th = doc.th
    pages: dict[int, list[_Block]] = {}
    for b in doc.blocks:
        pages.setdefault(b.page, []).append(b)
    for no, (_, blocks) in enumerate(sorted(pages.items()), start=2):
        fig = plt.figure(figsize=PAGE, facecolor=th.bg)
        fig.text(0.025, 0.965, header, fontsize=9, color=th.muted, fontweight="bold")
        fig.text(0.975, 0.965, blocks[0].section, fontsize=9, color=th.accent, ha="right", fontweight="bold")
        if doc.o.footer:
            fig.text(0.025, 0.018, doc.o.footer, fontsize=8, color=th.muted, fontweight="bold")
        fig.text(0.975, 0.018, str(no), fontsize=8, color=th.muted, ha="right")
        _place(fig, [Image.open(io.BytesIO(b.png)) for b in blocks])
        pdf.savefig(fig, facecolor=th.bg)
        plt.close(fig)


def build_report(sd: SetPieceData, opts: ReportOptions | None = None) -> bytes:
    o = opts or ReportOptions()
    doc = _Doc(sd, o)
    th = doc.th
    team = sd.team
    sec = set(o.sections)
    old_footer = S.FOOTER
    S.FOOTER = ""                                     # charts carry no footer — the page has the report's own
    try:
        doc.new_page()
        doc.add("Summary", lambda: X.summary_table(sd, f"{team} — set pieces at a glance", theme=th))
        if o.include_insights:
            doc.insights(D.headline_insights(sd), "Key insights", f"{team} — key insights")
        if o.per_match and D.has_matches(sd):
            doc.new_page()
            doc.add("Match by match", lambda: X.match_chart(sd, f"{team} — set pieces match by match", theme=th))
            doc.new_page()
            doc.add("Match by match", lambda: X.match_table(sd, f"{team} — every match", theme=th))
            if o.include_insights and o.match_insights:
                items = [i for m in D.matches(sd) for i in D.match_insights(sd, m)]
                doc.insights(items, "Match by match", f"{team} — match by match insights")

        for ph, row, g in D.groups(sd):
            if ph not in o.phases or (o.types and row not in o.types):
                continue
            att = ph == "Attacking"
            lab = D.type_label(row)
            s_ = f"{ph} · {lab}"
            title = f"{team} · {lab}"
            direct = g["category"].iloc[0] == "Penalty" or g["variant"].iloc[0] == "Direct"
            items = D.type_insights(sd, row, g, section=lab) if o.include_insights else []
            if direct:                                # no contacts: the shot only — insights on the same page
                doc.new_page()
                if items:
                    doc.add(s_, lambda items=items, title=title: X.insights_page(items, f"{title} — insights", theme=th))
                if "goal" in sec and g["goal_x_m"].notna().any():
                    doc.add(s_, lambda row=row: S.goal_placement(sd, row, theme=th))
                if "shots" in sec and g["is_shot"].any():
                    doc.add(s_, lambda g=g, title=title: X.shots_chart(sd, g, f"{title} — shots", theme=th))
                if att and "takers" in sec and (g["taker"] != "").any():
                    doc.add(s_, lambda g=g, title=title: S.takers_chart(sd, g, f"{title} — takers", theme=th))
                continue
            if items:
                doc.insights(items, s_, f"{title} — insights")
            if g["x"].notna().any() and ({"maps", "zones"} & sec):
                doc.new_page()
                if "maps" in sec:
                    doc.add(s_, lambda g=g, title=title: X.delivery_map(sd, g, f"{title} — deliveries & 1st contact", theme=th))
                if "zones" in sec:
                    doc.add(s_, lambda g=g, title=title: X.zone_map(sd, g, f"{title} — delivery zones", theme=th))
            has_sb = g["sb_x"].notna().any() and "second balls" in sec
            has_pl = "players" in sec and ((g["fc_won_by"] != "").any() or (g["fc_lost_by"] != "").any())
            if has_sb or has_pl:
                doc.new_page()
                if has_sb:
                    doc.add(s_, lambda row=row: S.contact_map(sd, row, 2, theme=th))
                if has_pl:
                    doc.add(s_, lambda g=g, title=title: X.players_chart(sd, g, f"{title} — 1st-contact players", theme=th))
            if "details" in sec:
                doc.new_page()
                doc.add(s_, lambda row=row: S.qualifiers(sd, row, theme=th))
            if {"funnel", "shots", "takers"} & sec:
                doc.new_page()
                if "funnel" in sec:
                    doc.add(s_, lambda g=g, title=title, att=att: X.funnel_chart(sd, g, f"{title} — from delivery to goal", att,
                                                                               theme=th))
                if "shots" in sec and g["is_shot"].any():
                    doc.add(s_, lambda g=g, title=title: X.shots_chart(sd, g, f"{title} — shots", theme=th))
                if att and "takers" in sec and (g["taker"] != "").any():
                    doc.add(s_, lambda g=g, title=title: S.takers_chart(sd, g, f"{title} — takers", theme=th))
            if "sides" in sec and (g["side"] == "Left").sum() >= 3 and (g["side"] == "Right").sum() >= 3 and g["x"].notna().any():
                doc.new_page()
                for side in ("Left", "Right"):
                    gs = g[g["side"] == side]
                    doc.add(s_, lambda gs=gs, side=side, title=title: X.delivery_map(sd, gs, f"{title} — from the {side.lower()}",
                                                                                 theme=th))
        buf = io.BytesIO()
        header = (o.title or f"{team} — set pieces").upper()
        with PdfPages(buf) as pdf:
            _cover(pdf, sd, o, th)
            _write(pdf, doc, header)
            pdf.infodict()["Title"] = header
        return buf.getvalue()
    finally:
        S.FOOTER = old_footer
