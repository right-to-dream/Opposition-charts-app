"""Downloads: the code-window build sheet, the analysed data workbook and a sample Sportscode XML."""
from __future__ import annotations

import io
import random
from xml.sax.saxutils import escape

import pandas as pd

from fap.playertag import stats as S
from fap.playertag import taxonomy as T
from fap.playertag.window import build

HOW_TO = [
    ("1", "Sportscode → File → New → Code Window. Turn Edit mode on (pencil)."),
    ("2", "Codes sheet: add one CODE button per row (name EXACTLY as written), set its colour, hotkey and "
          "lead / lag (seconds before / after the click)."),
    ("3", "Select all code buttons → Links → make them EXCLUSIVE to each other, so one action is open at a time."),
    ("4", "Labels sheet: add one LABEL button per row and type its Group name (Outcome, Zone, Pass Value ...). "
          "The group is what the export uses — keep the names identical."),
    ("5", "Groups marked exclusive (Outcome, Zone, Direction ...): link their labels as exclusive to each other."),
    ("6", "Use the x / y / width / height columns to reproduce the layout (pixels, top-left origin); the PNG shows it."),
    ("7", "Tagging order per action: CODE (hotkey) → ZONE (click the pitch) → OUTCOME (1 / 2) → any extra labels."),
    ("8", "One timeline per player per match. After the game: Timeline → Export → XML (or the instances CSV) and "
          "upload the file on the Player Tag Panel page."),
]


def build_sheet(positions: list[str] | None = None) -> bytes:
    positions = positions or list(T.POSITIONS) + ["ALL"]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(HOW_TO, columns=["Step", "How to build the code window"]).to_excel(xw, sheet_name="How to build",
                                                                                       index=False)
        groups = [{"Label group": g.name, "Labels": ", ".join(g.labels),
                   "Exclusive": "Yes" if g.exclusive else "No (several allowed)",
                   "Used by": ", ".join(a.name for a in T.ACTIONS if g.name in a.groups)} for g in T.LABEL_GROUPS]
        pd.DataFrame(groups).to_excel(xw, sheet_name="Label groups", index=False)
        acts = [{"Code": a.name, "Hotkey": a.hotkey, "Category": a.category, "Lead (s)": a.lead, "Lag (s)": a.lag,
                 "Positions": ", ".join(a.positions), "Label groups": ", ".join(a.groups), "Definition": a.help}
                for a in T.ACTIONS]
        pd.DataFrame(acts).to_excel(xw, sheet_name="Codes (all)", index=False)
        for p in positions:
            win = build(p)
            rows = [{"Button": b.name, "Type": "Code" if b.kind == "code" else "Label",
                     "Group / category": b.group, "Hotkey": b.hotkey, "Colour": b.colour, "Lead (s)": b.lead or "",
                     "Lag (s)": b.lag or "", "x": b.x, "y": b.y, "width": b.w, "height": b.h, "Definition": b.help}
                    for b in win.buttons]
            pd.DataFrame(rows).to_excel(xw, sheet_name=f"Window {p}", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(80, max(8, width + 2))
    return buf.getvalue()


def data_workbook(df: pd.DataFrame, position: str, minutes: dict[str, float] | None = None,
                  player: str = "", team: str = "") -> bytes:
    minutes = minutes or {}
    total_min = sum(minutes.get(m, 0) for m in dict.fromkeys(df["match"])) or None
    events = df.drop(columns=[c for c in df.columns if c.startswith("is_")], errors="ignore")
    canonical = pd.DataFrame({
        "match": df["match"], "minute": df["minute"], "second": (df["t"] % 60).round(1),
        "player": df["player"].replace("", player), "team": team, "position": df["position"].replace("", position),
        "event_type": df["action"].str.lower().str.replace(" ", "_"), "outcome": df["Outcome"].str.lower(),
        "shot_result": df["Shot Outcome"].str.lower(), "x": df["x"], "y": df["y"], "zone": df["Zone"],
        "under_pressure": df["under_pressure"].astype(int), "progressive": df["is_progressive"].astype(int),
        "key_pass": df["is_key_pass"].astype(int), "assist": df["is_assist"].astype(int),
        "set_piece": df["set_piece"].astype(int)})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        S.stats_table(df, position, total_min).to_excel(xw, sheet_name="Summary", index=False)
        if df["match"].nunique() > 1:
            S.per_match(df, position, minutes).to_excel(xw, sheet_name="Per match", index=False)
        S.action_summary(df).to_excel(xw, sheet_name="Actions", index=False)
        S.zone_table(df).to_excel(xw, sheet_name="Zones", index=False)
        events.to_excel(xw, sheet_name="Events", index=False)
        canonical.to_excel(xw, sheet_name="Events (app format)", index=False)
    return buf.getvalue()


def canonical_csv(df: pd.DataFrame, player: str = "", team: str = "") -> bytes:
    out = pd.DataFrame({"match": df["match"], "minute": df["minute"], "player": df["player"].replace("", player),
                        "team": team, "event_type": df["action"].str.lower().str.replace(" ", "_"),
                        "outcome": df["Outcome"].str.lower(), "shot_result": df["Shot Outcome"].str.lower(),
                        "x": df["x"], "y": df["y"], "zone": df["Zone"]})
    return out.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ sample export (for testing the page)
_WEIGHTS = {"Pass": 40, "Receive": 22, "Carry": 6, "Dribble": 4, "Cross": 4, "Shot": 3, "Ball Lost": 4,
            "Ground Duel": 7, "Aerial Duel": 5, "Tackle": 4, "Interception": 4, "Recovery": 6, "Clearance": 4,
            "Block": 2, "Press": 8, "Dribbled Past": 2, "Foul": 1, "Foul Won": 1, "Offside": 1, "Save": 4,
            "Goal Conceded": 1, "High Claim": 3, "Sweeper Action": 1}


def sample_xml(position: str = "CM", n: int = 120, seed: int = 7) -> bytes:
    """A realistic-looking Sportscode timeline XML for one player (to try the page without a real file)."""
    rnd = random.Random(seed)
    acts = [a for a in T.actions_for(position)]
    w = [_WEIGHTS.get(a.name, 2) for a in acts]
    t, parts = 30.0, ['<?xml version="1.0" encoding="utf-8"?>', "<file>", "<ALL_INSTANCES>"]
    for i in range(1, n + 1):
        a = rnd.choices(acts, w)[0]
        t += rnd.uniform(10, 50)
        labels: list[tuple[str, str]] = []
        col = rnd.choices(range(1, 7), weights={"GK": (8, 2, 0, 0, 0, 0), "CB": (4, 5, 3, 1, .5, .2),
                                                "ST": (.2, .5, 1, 3, 5, 5), "W": (.5, 1, 2, 4, 5, 3)}
                          .get(position, (1, 2, 4, 4, 3, 1)))[0]
        lane = rnd.choice((1, 2, 3))
        labels.append(("Zone", T.zone_name(col, lane)))
        if "Outcome" in a.groups and a.name not in ("Save",):
            labels.append(("Outcome", "Successful" if rnd.random() < .72 else "Unsuccessful"))
        if a.name == "Shot":
            labels.append(("Shot Outcome", rnd.choice(("Goal", "On Target", "On Target", "Off Target", "Blocked"))))
            labels.append(("Body Part", rnd.choice(("Right Foot", "Left Foot", "Head"))))
        if a.name == "Pass":
            labels.append(("Direction", rnd.choice(("Forward", "Forward", "Sideways", "Backward"))))
            labels.append(("Pass Length", rnd.choice(("Short", "Short", "Short", "Long"))))
            if rnd.random() < .2:
                labels.append(("Pass Value", "Progressive"))
            if rnd.random() < .06:
                labels.append(("Pass Value", "Key Pass"))
            if position == "GK":
                labels.append(("Distribution", rnd.choice(("Hand", "Goal Kick", "Open Play Kick"))))
        if a.name in ("Pass", "Receive", "Dribble") and rnd.random() < .3:
            labels.append(("Pressure", "Under Pressure"))
        if a.name == "Press":
            labels.append(("Press Outcome", rnd.choice(("Regain", "No Regain", "No Regain", "Forced Error"))))
        if a.name == "Receive" and rnd.random() < .25:
            labels.append(("Reception", rnd.choice(("Between Lines", "In Behind"))))
        lab_xml = "".join(f"<label><group>{escape(g)}</group><text>{escape(x)}</text></label>" for g, x in labels)
        parts.append(f"<instance><ID>{i}</ID><start>{t - a.lead:.2f}</start><end>{t + a.lag:.2f}</end>"
                     f"<code>{escape(a.name)}</code>{lab_xml}</instance>")
    parts += ["</ALL_INSTANCES>", "</file>"]
    return "\n".join(parts).encode("utf-8")
