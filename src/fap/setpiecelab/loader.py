"""Read a set-piece coding matrix (Sportscode / Hudl "matrix" export) into one tidy table.

Expected shape (column names matched case-insensitively):
  * ``Row``                — the set-piece type as coded ("Offensive Corner", "Long GK", "Def Throw in" ...)
  * ``Original Instance``  — instance id ("Offensive Corner: 3"), optional
  * ``Start time`` / ``Duration`` / ``Timeline`` / ``Notes`` / ``Flags`` — optional
  * ``x_<type>`` / ``y_<type>`` — landing point of THAT type (metres, 105 × 68). A cell may hold several
    comma-separated values ("101, 80, 80"): the first is the main landing point, the others are extra
    points of the same set piece (kept and drawn, flagged as extra).
  * every other non-empty column is a qualifier ("1st contact", "2nd ball", "Area", "Execution",
    "Location", "Outcome", "Efficiency" ...) — detected automatically, nothing hard-coded.

Perspective: the file belongs to ONE team (``team``); "Offensive" = that team taking the set piece,
"Defensive" = that team defending it. Set pieces without the word are classified by type
("Long GK" / "Long Throw In" = taking it, "Penalty Against" / "Def ..." = defending).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PITCH_L, PITCH_W = 105.0, 68.0

_META = {"timeline", "start time", "duration", "row", "original instance", "notes", "flags", "ungrouped",
         "end time", "instance", "code", "clip"}

YES = {"yes", "y", "true", "1", "won", "successful", "successfull"}
NO = {"no", "n", "false", "0", "lost", "unsuccessful", "unsuccessfull"}

CATEGORIES = (("corner", "Corner"), ("free kick", "Free kick"), ("freekick", "Free kick"), ("throw", "Throw-in"),
              ("gk", "Goal kick"), ("goal kick", "Goal kick"), ("penalty", "Penalty"), ("kick off", "Kick-off"),
              ("kickoff", "Kick-off"))


def _key(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def category_of(row: str) -> str:
    r = _key(row)
    for k, v in CATEGORIES:
        if k in r:
            return v
    return "Other"


def phase_of(row: str) -> str:
    """'Attacking' = the file's team takes the set piece, 'Defending' = it defends one."""
    r = _key(row)
    if r.startswith(("def", "defensive")) or "against" in r or "defending" in r:
        return "Defending"
    return "Attacking"


def parse_points(xv: Any, yv: Any) -> list[tuple[float, float]]:
    def nums(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        out = []
        for part in str(v).replace(";", ",").split(","):
            try:
                out.append(float(part.strip()))
            except ValueError:
                pass
        return out
    xs, ys = nums(xv), nums(yv)
    n = min(len(xs), len(ys))
    return [(xs[i], ys[i]) for i in range(n)]


def _clean_qual(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v).strip()
    k = s.lower()
    if k in ("successfull", "successful"):
        return "Successful"
    if k in ("unsuccessfull", "unsuccessful"):
        return "Unsuccessful"
    if k == "cleareance":
        return "Clearance"
    return s


@dataclass
class SetPieceData:
    df: pd.DataFrame                       # one row per set piece
    team: str
    qualifiers: list[str]                  # qualifier column names (display names)
    binary: list[str]                      # qualifiers that are Yes / No
    source_name: str = ""
    notes: list[str] = field(default_factory=list)

    def of(self, row: str | None = None, category: str | None = None, phase: str | None = None) -> pd.DataFrame:
        d = self.df
        if row:
            d = d[d["row"] == row]
        if category:
            d = d[d["category"] == category]
        if phase:
            d = d[d["phase"] == phase]
        return d

    @property
    def rows(self) -> list[str]:
        return list(self.df["row"].value_counts().index)


def read_table(src: Any, name: str = "") -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()
    if isinstance(src, (str, Path)):
        name = name or str(src)
        data = Path(src).read_bytes()
    elif isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        name = name or getattr(src, "name", "")
        data = src.read()
    if name.lower().endswith((".csv", ".txt")):
        try:
            return pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
    return pd.read_excel(io.BytesIO(data))


def load_setpieces(src: Any, name: str = "", team: str = "") -> SetPieceData:
    raw = read_table(src, name)
    cols = {_key(c): c for c in raw.columns}
    if "row" not in cols:
        raise ValueError("Expected a coding-matrix export with a 'Row' column (the set-piece type).")
    notes: list[str] = []
    row = raw[cols["row"]].astype(str).str.strip()
    xcols = {_key(c)[2:]: c for c in raw.columns if _key(c).startswith("x_")}
    ycols = {_key(c)[2:]: c for c in raw.columns if _key(c).startswith("y_")}
    qual_cols = [c for c in raw.columns if _key(c) not in _META and not _key(c).startswith(("x_", "y_"))
                 and raw[c].notna().any()]
    df = pd.DataFrame({
        "row": row,
        "instance": raw[cols["original instance"]].astype(str) if "original instance" in cols else row + ": " + (raw.index + 1).astype(str),
        "category": row.map(category_of),
        "phase": row.map(phase_of),
        "start": pd.to_numeric(raw[cols["start time"]], errors="coerce") if "start time" in cols else np.nan,
        "duration": pd.to_numeric(raw[cols["duration"]], errors="coerce") if "duration" in cols else np.nan,
        "notes": raw[cols["notes"]].fillna("").astype(str) if "notes" in cols else "",
        "flags": raw[cols["flags"]].fillna("").astype(str) if "flags" in cols else "",
    })
    df["n"] = df.groupby("row").cumcount() + 1
    for c in qual_cols:
        df[str(c).strip()] = raw[c].map(_clean_qual)
    quals = [str(c).strip() for c in qual_cols]
    binary = [q for q in quals if set(df[q].str.lower().unique()) - {""} <= (YES | NO) and (df[q] != "").any()]
    # landing points: x_<row> / y_<row> of the row's own type
    def coord_key(k: str) -> str | None:
        """x_/y_ suffix for a row type: exact name, else a suffix contained in the type ('gk' in 'long gk')
        or the type contained in the suffix; the longest match wins."""
        if k in xcols and k in ycols:
            return k
        cands = [s for s in xcols if s in ycols and (re.search(rf"\b{re.escape(s)}\b", k) or k in s)]
        return max(cands, key=len) if cands else None

    pts = []
    for i, r in enumerate(row):
        ck = coord_key(_key(r))
        xc, yc = (xcols.get(ck), ycols.get(ck)) if ck else (None, None)
        pts.append(parse_points(raw.at[raw.index[i], xc], raw.at[raw.index[i], yc]) if xc is not None and yc is not None else [])
    df["points"] = pts
    df["x"] = [p[0][0] if p else np.nan for p in pts]
    df["y"] = [p[0][1] if p else np.nan for p in pts]
    df["extra_points"] = [len(p) - 1 if p else 0 for p in pts]
    missing_xy = sorted({r for r in df["row"].unique() if coord_key(_key(r)) is None})
    if missing_xy:
        notes.append("No coordinate columns for: " + ", ".join(missing_xy))
    multi = int((df["extra_points"] > 0).sum())
    if multi:
        notes.append(f"{multi} set piece(s) have more than one point in the cell — first = landing point, "
                     "others drawn as extra points.")
    if not team:
        tl = raw[cols["timeline"]].dropna().astype(str) if "timeline" in cols else pd.Series(dtype=str)
        team = re.sub(r"(?i)\s*set\s*pieces?.*$", "", tl.iloc[0]).strip() if len(tl) else "Team"
    return SetPieceData(df=df, team=team or "Team", qualifiers=quals, binary=binary, source_name=name, notes=notes)


def is_yes(v: Any) -> bool | None:
    k = str(v).strip().lower()
    if k in YES:
        return True
    if k in NO:
        return False
    return None
