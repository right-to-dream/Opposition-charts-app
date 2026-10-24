"""Read a Sportscode export of the player code window into one tidy event table.

Accepted exports (one file = one match of one player):

* **Timeline / edit-list XML** — ``<instance><code>Pass</code><start>..</start>
  <label><group>Outcome</group><text>Successful</text></label>...</instance>``.
* **Instances CSV / XLSX** (the matrix "Row" export) — a ``Row`` (or ``Code``) column with the action and
  one column per label group (comma-separated when several labels were tagged).

Names are matched case-insensitively against :mod:`fap.playertag.taxonomy` (names + synonyms), so small
spelling differences and ungrouped labels still resolve. A code that is not an action of the window
(half start, notes, ...) is skipped and reported in ``TagData.ignored``; an unknown label is kept in
``TagData.unknown_labels`` so nothing disappears silently.
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fap.playertag import taxonomy as T

ALWAYS_SUCCESS = {"Interception", "Recovery", "Clearance", "Block"}
NEGATIVE = {"Ball Lost", "Dribbled Past", "Foul", "Offside", "Goal Conceded", "Foul Won"}
_OUTCOME_WORDS = {"successful": "Successful", "success": "Successful", "succ": "Successful",
                  "complete": "Successful", "completed": "Successful", "won": "Successful",
                  "unsuccessful": "Unsuccessful", "unsuccess": "Unsuccessful", "fail": "Unsuccessful",
                  "failed": "Unsuccessful", "incomplete": "Unsuccessful", "lost": "Unsuccessful"}
_CODE_COLS = ("row", "code", "action", "event", "category")
_TIME_COLS = ("start time", "start", "clip start", "time", "timestamp")
_SKIP_COLS = ("instance number", "original instance", "duration", "end time", "end", "flags", "id", "notes",
              "timeline", "player", "position", "match", "minute")


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z✓✗ ]+", " ", str(s).lower())).strip()


_ACTION_LOOKUP: dict[str, str] = {}
for _a in T.ACTIONS:
    for _s in (_a.name, *_a.synonyms):
        _ACTION_LOOKUP.setdefault(_norm(_s), _a.name)

_LABEL_LOOKUP: dict[str, tuple[str, str]] = {}          # text -> (group, label), first group wins
_GROUP_LABELS: dict[str, dict[str, str]] = {}           # group -> text -> label
for _g in T.LABEL_GROUPS:
    m = _GROUP_LABELS.setdefault(_norm(_g.name), {})
    for _lab in _g.labels:
        for _s in (_lab, *_g.synonyms.get(_lab, ())):
            m.setdefault(_norm(_s), _lab)
            _LABEL_LOOKUP.setdefault(_norm(_s), (_g.name, _lab))
_GROUP_NAME = {_norm(g.name): g.name for g in T.LABEL_GROUPS}
_GROUP_NAME.update({"result": "Outcome", "outcome": "Outcome", "zones": "Zone", "pitch zone": "Zone",
                    "location": "Zone", "area": "Zone"})


def resolve_action(code: Any) -> tuple[str | None, str]:
    """Code text → (action, outcome-from-code). "Pass Successful" / "Succ Pass" → ("Pass", "Successful")."""
    key = _norm(code)
    if not key:
        return None, ""
    if key in _ACTION_LOOKUP:
        return _ACTION_LOOKUP[key], ""
    words = key.split()
    outcome = next((_OUTCOME_WORDS[w] for w in words if w in _OUTCOME_WORDS), "")
    rest = " ".join(w for w in words if w not in _OUTCOME_WORDS)
    if outcome and rest in _ACTION_LOOKUP:
        return _ACTION_LOOKUP[rest], outcome
    if rest.endswith("s") and rest[:-1] in _ACTION_LOOKUP:
        return _ACTION_LOOKUP[rest[:-1]], outcome
    return None, ""


def resolve_label(text: Any, group: Any = "") -> tuple[str, str] | None:
    """Label text (+ optional group name) → (group, label) of the taxonomy, or None."""
    key = _norm(text)
    if not key:
        return None
    gname = _GROUP_NAME.get(_norm(group)) if group else None
    if gname == "Zone" or re.fullmatch(r"(z|zone )?\d{1,2}", key):
        num = re.sub(r"\D", "", key)
        if num and 1 <= int(num) <= T.N_COLS * T.N_LANES and (gname == "Zone" or key[0] == "z"):
            return "Zone", f"Z{int(num)}"
    if gname and gname in T.GROUPS:
        lab = _GROUP_LABELS[_norm(gname)].get(key)
        if lab:
            return gname, lab
    return _LABEL_LOOKUP.get(key)


def parse_time(v: Any) -> float:
    """Seconds from 75.2 / "75.2" / "01:15.2" / "00:01:15.2"."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip()
    if not s:
        return np.nan
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        return np.nan
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


@dataclass
class TagData:
    df: pd.DataFrame
    ignored: Counter = field(default_factory=Counter)
    unknown_labels: Counter = field(default_factory=Counter)
    sources: list[str] = field(default_factory=list)

    @property
    def matches(self) -> list[str]:
        return list(dict.fromkeys(self.df["match"])) if len(self.df) else []

    def subset(self, match: str | None = None) -> "TagData":
        d = self.df if match in (None, "", "All matches") else self.df[self.df["match"] == match]
        return TagData(d.copy(), self.ignored, self.unknown_labels, self.sources)


# ------------------------------------------------------------------ raw readers → [(code, start, [(group, text)])]
def _read_bytes(src: Any, name: str) -> tuple[bytes, str]:
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes(), name or str(src)
    if isinstance(src, (bytes, bytearray)):
        return bytes(src), name
    return src.read(), name or getattr(src, "name", "")


def _xml_instances(data: bytes) -> list[tuple[str, float, list[tuple[str, str]], str]]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Not a valid Sportscode XML export: {exc}") from exc
    out = []
    for inst in root.iter("instance"):
        labels = [((lab.findtext("group") or "").strip(), (lab.findtext("text") or "").strip())
                  for lab in inst.findall("label")]
        out.append(((inst.findtext("code") or "").strip(), parse_time(inst.findtext("start")), labels, ""))
    if not out:
        raise ValueError("No <instance> elements found — export the timeline as XML (File → Export → XML).")
    return out


def _table_instances(df: pd.DataFrame) -> tuple[list, dict]:
    cols = {_norm(c): c for c in df.columns}
    code_col = next((cols[c] for c in _CODE_COLS if c in cols), None)
    if code_col is None:
        raise ValueError("Expected a 'Row' (or 'Code') column with the action names.")
    time_col = next((cols[c] for c in _TIME_COLS if c in cols), None)
    extra = {k: cols[k] for k in ("player", "position", "match", "notes") if k in cols}
    label_cols = [c for c in df.columns if c not in (code_col, time_col) and _norm(c) not in _SKIP_COLS]
    out, meta = [], {k: [] for k in extra}
    for _, r in df.iterrows():
        labels = []
        for c in label_cols:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            for tok in re.split(r"[,;|]", str(v)):
                if tok.strip():
                    labels.append((str(c), tok.strip()))
        note = r[extra["notes"]] if "notes" in extra else ""
        out.append((str(r[code_col]), parse_time(r[time_col]) if time_col else np.nan, labels,
                    "" if pd.isna(note) else str(note)))
        for k in ("player", "position", "match"):
            if k in extra:
                meta[k].append(r[extra[k]])
    return out, meta


def _read_table(data: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith((".csv", ".txt")):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except UnicodeDecodeError:
                continue
    return pd.read_excel(io.BytesIO(data))


# ------------------------------------------------------------------ public
def load_tags(src: Any, name: str = "", *, player: str = "", position: str = "", match: str = "") -> TagData:
    data, name = _read_bytes(src, name)
    meta: dict = {}
    if name.lower().endswith(".xml") or data.lstrip()[:1] == b"<":
        inst = _xml_instances(data)
    else:
        inst, meta = _table_instances(_read_table(data, name))
    match = match or Path(name).stem or "Match"
    ignored, unknown = Counter(), Counter()
    rows = []
    for i, (code, start, labels, note) in enumerate(inst):
        action, code_outcome = resolve_action(code)
        if action is None:
            if code:
                ignored[code] += 1
            continue
        a = T.ACTION_BY_NAME[action]
        rec: dict[str, Any] = {"match": match, "t": start, "action": action, "category": a.category,
                               "raw_code": code, "notes": note}
        tagged: dict[str, list[str]] = {}
        for group, text in labels:
            if _norm(group) in ("player", "position") and text:
                rec[_norm(group)] = text
                continue
            hit = resolve_label(text, group)
            if hit is None:
                if group and _GROUP_NAME.get(_norm(group)) is None and _norm(group) not in ("", "notes"):
                    unknown[f"{group}: {text}"] += 1
                elif text:
                    unknown[text] += 1
                continue
            g, lab = hit
            if lab not in tagged.setdefault(g, []):
                tagged[g].append(lab)
        for g in T.LABEL_GROUPS:
            vals = tagged.get(g.name, [])
            rec[g.name] = ", ".join(vals[:1] if g.exclusive else vals)
        if code_outcome and not rec["Outcome"]:
            rec["Outcome"] = code_outcome
        for k in ("player", "position", "match"):
            vals = meta.get(k)
            if vals and i < len(vals) and not pd.isna(vals[i]) and str(vals[i]).strip() and (k == "match" or k not in rec):
                rec[k] = str(vals[i]).strip()
        rows.append(rec)
    df = pd.DataFrame(rows, columns=["match", "t", "action", "category", "raw_code", "notes",
                                     *[g.name for g in T.LABEL_GROUPS], "player", "position"])
    if player:
        df["player"] = player
    if position:
        df["position"] = position
    df["player"] = df["player"].fillna("").astype(str)
    df["position"] = df["position"].fillna("").astype(str)
    return TagData(enrich(df), ignored, unknown, [name])


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns: minute, success flag, zone geometry, value flags."""
    df = df.copy().reset_index(drop=True)
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    for g in T.LABEL_GROUPS:
        df[g.name] = df[g.name].fillna("").astype(str)
    df["minute"] = (df["t"] // 60).astype("Int64")
    df["success"] = [_success(a, o, s, p) for a, o, s, p in
                     zip(df["action"], df["Outcome"], df["Shot Outcome"], df["Press Outcome"])]
    cen = [T.zone_centre(z) or (np.nan, np.nan) for z in df["Zone"]]
    df["x"] = [c[0] for c in cen]
    df["y"] = [c[1] for c in cen]
    df["third"] = [T.zone_third(z) for z in df["Zone"]]
    df["lane"] = [T.zone_lane(z) for z in df["Zone"]]
    for lab in ("Progressive", "Key Pass", "Assist", "Through Ball", "Switch", "Line Breaking", "Final Third Entry",
                "Box Entry"):
        df[f"is_{lab.lower().replace(' ', '_')}"] = df["Pass Value"].str.contains(lab, regex=False)
    df["under_pressure"] = df["Pressure"].eq("Under Pressure")
    df["set_piece"] = df["Set Piece"].eq("Set Piece")
    df["is_goal"] = (df["action"] == "Shot") & df["Shot Outcome"].eq("Goal")
    return df.sort_values(["match", "t"], kind="stable").reset_index(drop=True)


def _success(action: str, outcome: str, shot: str, press: str):
    if action in ALWAYS_SUCCESS:
        return True
    if action in NEGATIVE:
        return np.nan
    if action == "Shot":
        return np.nan if not shot else shot in ("Goal", "On Target")
    if action == "Press":
        if press:
            return press in ("Regain", "Forced Error")
        return np.nan if not outcome else outcome == "Successful"
    if action == "Save" and not outcome:
        return True
    return np.nan if not outcome else outcome == "Successful"


def combine(parts: list[TagData]) -> TagData:
    parts = [p for p in parts if p is not None]
    if not parts:
        return TagData(enrich(pd.DataFrame(columns=["match", "t", "action", "category", "raw_code", "notes",
                                                    *[g.name for g in T.LABEL_GROUPS], "player", "position"])))
    ign, unk, src = Counter(), Counter(), []
    for p in parts:
        ign.update(p.ignored)
        unk.update(p.unknown_labels)
        src += p.sources
    df = pd.concat([p.df for p in parts], ignore_index=True)
    return TagData(df, ign, unk, src)
