"""Set-piece PDF report — every chart and map of the Set Piece Lab, aggregated and in detail, with optional
insights. A4 landscape pages; each chart is the same static PNG the page downloads (``static`` /
``deep_static``), two pitch maps per page where they fit.

Order: cover → summary table → (insights: headline) → per phase: overview, 1st contact & 2nd ball, funnel,
shots, delivery zones, first-contact players, takers → taking vs defending per category → (match by match)
→ per set-piece type: (insights), delivery map, delivery zones, how it is taken / defended, funnel, shots,
players, takers, left vs right, second balls, goal placement.
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
from fap.setpiecelab import analytics as A  # noqa: E402
from fap.setpiecelab import deep as D  # noqa: E402
from fap.setpiecelab import deep_static as X  # noqa: E402
from fap.setpiecelab import static as S  # noqa: E402
from fap.setpiecelab.loader import SetPieceData  # noqa: E402

PAGE = (11.69, 8.27)
SECTIONS = ("maps", "zones", "details", "funnel", "shots", "players", "takers", "sides", "second balls", "goal")


@dataclass
class ReportOptions:
    title: str = ""
    subtitle: str = ""
    theme: str = "Dark"
    include_insights: bool = True
    detailed: bool = True                     # per set-piece type pages (False = aggregated only)
    phases: tuple = ("Attacking", "Defending")
    per_match: bool = True                    # match-by-match pages when the file has a match column
    match_insights: bool = False              # insights per match (long)
    sections: tuple = SECTIONS
    types: tuple = ()                         # restrict the detailed part to these coded types ("" = all)
    dpi: int = 150
    log: list = field(default_factory=list)


@dataclass
class _Block:
    png: bytes
    half: bool = False
    section: str = ""


def _png(fig, dpi) -> bytes:
    return S.png(fig, dpi)


class _Doc:
    def __init__(self, sd: SetPieceData, opts: ReportOptions):
        self.sd, self.o = sd, opts
        self.th = THEMES.get(opts.theme, THEMES["Dark"])
        self.blocks: list[_Block] = []

    def add(self, fig, section: str, half: bool = False) -> None:
        try:
            self.blocks.append(_Block(_png(fig, self.o.dpi), half, section))
        except Exception as exc:                              # one chart must never break the report
            self.o.log.append(f"{section}: {exc}")
            plt.close("all")

    def safe(self, section: str, make, half: bool = False) -> None:
        try:
            self.add(make(), section, half)
        except Exception as exc:
            self.o.log.append(f"{section}: {exc}")
            plt.close("all")

    def insights(self, items, section: str, title: str, per_page: int = 24) -> None:
        if not items:
            return
        chunk, lines = [], 0
        for it in items:
            n = 1 + len(it.text) // 120
            if chunk and lines + n > per_page:
                self.add(X.insights_page(chunk, title, theme=self.th), section)
                chunk, lines = [], 0
                title = title + " (cont.)" if not title.endswith("(cont.)") else title
            chunk.append(it)
            lines += n + (1 if not chunk[:-1] or chunk[-2].section != it.section else 0)
        if chunk:
            self.add(X.insights_page(chunk, title, theme=self.th), section)


def _cover(pdf, sd: SetPieceData, o: ReportOptions, th) -> None:
    fig = plt.figure(figsize=PAGE, facecolor=th.bg)
    a = sd.df[sd.df["phase"] == "Attacking"]
    d = sd.df[sd.df["phase"] == "Defending"]
    sa, sdd = D.summary(sd, a), D.summary(sd, d)
    fig.text(0.06, 0.78, (o.title or f"{sd.team} — set pieces").upper(), fontsize=30, fontweight="bold", color=th.text)
    fig.text(0.06, 0.71, o.subtitle or f"{len(sd.df)} set pieces coded"
             + (f" · {len(D.matches(sd))} matches" if D.has_matches(sd) else ""), fontsize=15, color=th.muted)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.66, 0.66], color=th.accent, lw=2))
    cards = [("Taken", f"{sa['Count']}", f"{sa['Shots']} shots · {sa['Goals']} goals"),
             ("Faced", f"{sdd['Count']}", f"{sdd['Shots']} shots · {sdd['Goals']} goals conceded"),
             ("1st contact won", f"{sa['1st contact won']}/{sa['1st contact decided']}", "when taking"),
             ("1st contact won", f"{sdd['1st contact won']}/{sdd['1st contact decided']}", "when defending"),
             ("2nd ball won", f"{sa['2nd ball won']}/{sa['2nd ball decided']}", "when taking"),
             ("2nd ball won", f"{sdd['2nd ball won']}/{sdd['2nd ball decided']}", "when defending")]
    for i, (lab, val, hint) in enumerate(cards):
        x = 0.06 + (i % 3) * 0.3
        y = 0.47 - (i // 3) * 0.2
        fig.text(x, y + 0.08, lab.upper(), fontsize=11, color=th.muted, fontweight="bold")
        fig.text(x, y + 0.02, val, fontsize=28, fontweight="bold", color=th.team if i % 2 == 0 else th.opp)
        fig.text(x, y - 0.02, hint, fontsize=10.5, color=th.muted)
    fig.text(0.06, 0.06, f"{sd.team}'s point of view · taking = {sd.team} takes it · defending = against {sd.team} · "
                         "pitch 105 × 68 m · every number counted from the coded file", fontsize=9, color=th.muted)
    pdf.savefig(fig, facecolor=th.bg)
    plt.close(fig)


def _write(pdf, doc: _Doc, header: str) -> None:
    th = doc.th
    i, page = 0, 1
    blocks = doc.blocks
    while i < len(blocks):
        b = blocks[i]
        pair = [b]
        if b.half and i + 1 < len(blocks) and blocks[i + 1].half and blocks[i + 1].section == b.section:
            pair.append(blocks[i + 1])
        fig = plt.figure(figsize=PAGE, facecolor=th.bg)
        fig.text(0.03, 0.965, header, fontsize=9, color=th.muted, fontweight="bold")
        fig.text(0.97, 0.965, b.section, fontsize=9, color=th.accent, ha="right", fontweight="bold")
        fig.text(0.97, 0.02, str(page + 1), fontsize=8, color=th.muted, ha="right")
        w = 0.94 / len(pair)
        for k, blk in enumerate(pair):
            img = Image.open(io.BytesIO(blk.png))
            ax = fig.add_axes([0.03 + k * w, 0.05, w - (0.01 if len(pair) > 1 else 0), 0.89])
            ax.imshow(img, interpolation="lanczos")
            ax.set_anchor("N")
            ax.axis("off")
        pdf.savefig(fig, facecolor=th.bg)
        plt.close(fig)
        i += len(pair)
        page += 1


def build_report(sd: SetPieceData, opts: ReportOptions | None = None) -> bytes:
    o = opts or ReportOptions()
    doc = _Doc(sd, o)
    th = doc.th
    team = sd.team
    sec = set(o.sections)

    doc.safe("Summary", lambda: X.summary_table(sd, f"{team} — set pieces at a glance", theme=th))
    if o.include_insights:
        doc.insights(D.headline_insights(sd), "Key insights", f"{team} — key insights")

    # ---- aggregated, per phase
    for ph in o.phases:
        sub = sd.subset(phase=ph)
        g = sub.df
        if g.empty:
            continue
        att = ph == "Attacking"
        name = f"{team} {'taking' if att else 'defending'} set pieces"
        s_ = f"Aggregated · {ph}"
        doc.safe(s_, lambda: S.overview(sub, theme=th))
        doc.safe(s_, lambda: S.contests(sub, theme=th))
        if "funnel" in sec:
            doc.safe(s_, lambda: X.funnel_chart(sub, g, f"{name} — from delivery to goal", att, theme=th))
        if "shots" in sec and g["is_shot"].any():
            doc.safe(s_, lambda: X.shots_chart(sub, g, f"{name} — shots {'taken' if att else 'conceded'}", theme=th))
        if "zones" in sec:
            dl = g[g["category"].isin(["Corner", "Free kick", "Throw-in"]) & (g["variant"] != "Direct")]
            if dl["x"].notna().any():
                doc.safe(s_, lambda: X.zone_map(sub, dl, f"{name} — delivery zones (corners, free kicks, throw-ins)", theme=th), half=True)
                doc.safe(s_, lambda: X.delivery_map(sub, dl, f"{name} — every delivery", theme=th), half=True)
        if "players" in sec and ((g["fc_won_by"] != "").any() or (g["fc_lost_by"] != "").any()):
            doc.safe(s_, lambda: X.players_chart(sub, g, f"{name} — {team}'s players at the 1st contact", theme=th))
        if att and "takers" in sec and (g["taker"] != "").any():
            doc.safe(s_, lambda: S.takers_chart(sub, g, f"{name} — takers", theme=th))

    for cat in A.categories(sd):
        if len(sd.of(category=cat, phase="Attacking")) and len(sd.of(category=cat, phase="Defending")) and "details" in sec:
            doc.safe("Taking vs defending", lambda cat=cat: S.category_compare(sd, cat, theme=th))

    # ---- match by match
    if o.per_match and D.has_matches(sd):
        doc.safe("Match by match", lambda: X.match_chart(sd, f"{team} — set pieces match by match", theme=th))
        for m in D.matches(sd):
            sm = sd.subset(match=m)
            doc.safe("Match by match", lambda sm=sm, m=m: X.summary_table(sm, f"{team} vs {m} — set pieces", theme=th))
            if o.include_insights and o.match_insights:
                doc.insights(D.match_insights(sd, m), f"Match · {m}", f"{team} vs {m} — insights")

    # ---- detailed, per type
    if o.detailed:
        for ph, row, g in D.groups(sd):
            if ph not in o.phases or (o.types and row not in o.types):
                continue
            att = ph == "Attacking"
            lab = D.type_label(row)
            s_ = f"{ph} · {lab}"
            title = f"{team} · {lab}"
            direct = g["category"].iloc[0] == "Penalty" or g["variant"].iloc[0] == "Direct"
            if o.include_insights:
                doc.insights(D.type_insights(sd, row, g, section=lab), s_, f"{title} — insights")
            if not direct:
                if "maps" in sec and g["x"].notna().any():
                    doc.safe(s_, lambda g=g, title=title: X.delivery_map(sd, g, f"{title} — deliveries", theme=th), half=True)
                if "zones" in sec and g["x"].notna().any():
                    doc.safe(s_, lambda g=g, title=title: X.zone_map(sd, g, f"{title} — delivery zones", theme=th), half=True)
            if "details" in sec:
                doc.safe(s_, lambda row=row: S.qualifiers(sd, row, theme=th))
            if "funnel" in sec:
                doc.safe(s_, lambda g=g, title=title, att=att: X.funnel_chart(sd, g, f"{title} — from delivery to goal", att, theme=th))
            if "shots" in sec and g["is_shot"].any():
                doc.safe(s_, lambda g=g, title=title: X.shots_chart(sd, g, f"{title} — shots", theme=th))
            if "players" in sec and ((g["fc_won_by"] != "").any() or (g["fc_lost_by"] != "").any()):
                doc.safe(s_, lambda g=g, title=title: X.players_chart(sd, g, f"{title} — 1st-contact players", theme=th))
            if att and "takers" in sec and (g["taker"] != "").any():
                doc.safe(s_, lambda g=g, title=title: S.takers_chart(sd, g, f"{title} — takers", theme=th))
            if "sides" in sec and not direct and (g["side"] == "Left").sum() >= 3 and (g["side"] == "Right").sum() >= 3:
                groups = [(s, g[g["side"] == s]) for s in ("Left", "Right")]
                doc.safe(s_, lambda groups=groups, title=title, row=row: S.compare_groups(
                    sd, groups, f"{title} — left vs right", lines=A.side_insights(sd, row), theme=th))
                if "maps" in sec and g["x"].notna().any():
                    for s in ("Left", "Right"):
                        gs = g[g["side"] == s]
                        if gs["x"].notna().any():
                            doc.safe(s_, lambda gs=gs, s=s, title=title: X.delivery_map(sd, gs, f"{title} — from the {s.lower()}",
                                                                                     theme=th), half=True)
            if "second balls" in sec and g["sb_x"].notna().any():
                doc.safe(s_, lambda row=row: S.contact_map(sd, row, 2, theme=th), half=True)
                doc.safe(s_, lambda row=row: S.contact_map(sd, row, 1, theme=th), half=True)
            if "goal" in sec and g["goal_x_m"].notna().any():
                doc.safe(s_, lambda row=row: S.goal_placement(sd, row, theme=th))

    buf = io.BytesIO()
    header = (o.title or f"{team} — set pieces").upper()
    with PdfPages(buf) as pdf:
        _cover(pdf, sd, o, th)
        _write(pdf, doc, header)
        info = pdf.infodict()
        info["Title"] = header
    return buf.getvalue()
