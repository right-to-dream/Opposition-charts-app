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
    if k in ("cleareance", "clearence"):
        return "Clearance"
    if k in YES or k in NO:
        return s[:1].upper() + s[1:].lower()      # yes / YES → Yes
    # one spelling per value: "Penalty area" / "penalty Area" → "Penalty Area" (GK stays GK)
    return " ".join(w if w.isupper() and len(w) > 1 else w[:1].upper() + w[1:].lower() for w in s.split())


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

    def subset(self, **filters) -> "SetPieceData":
        """Same data restricted to matching rows, e.g. ``subset(side="Left")``, ``subset(phase="Attacking")``,
        ``subset(taker="24")`` — every analytics / chart function then works on the slice unchanged."""
        d = self.df
        for k, v in filters.items():
            if v not in (None, "", "All"):
                d = d[d[k].astype(str) == str(v)]
        return SetPieceData(df=d.copy(), team=self.team, qualifiers=self.qualifiers, binary=self.binary,
                            source_name=self.source_name, notes=self.notes)


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


_COORD = re.compile(r"^([xy])\s*[_\-\s]\s*(.+)$")
_SECOND = ("second ball", "2nd ball", "second", "2nd")

# outcome words, most dangerous first — the set piece's overall Result is the most dangerous one coded
RESULT_RANK = ("goal", "shot", "threat", "cross", "recycle", "foul won", "corner won", "gk punch", "gk claim",
               "clearance", "foul", "out of bounds", "out")
_GOAL_X = ("goal across", "goal x", "goal_x", "across")
_GOAL_Y = ("goal height", "goal y", "goal_y", "height")
_TAKER = ("taker", "kicker", "thrower", "player")


def variant_of(row: str) -> str:
    """Direct / Indirect free kick, Long / Short throw-in (empty for other types)."""
    r = _key(row)
    if "free kick" in r or "freekick" in r:
        return "Direct" if "direct" in r and "indirect" not in r else "Indirect"
    if "throw" in r:
        return "Long" if "long" in r else "Short"
    return ""
RESULT_COL = "Result"


def _coord_cols(columns) -> tuple[dict, dict]:
    """``x_<kind>`` / ``x-<kind>`` / ``X <kind>`` columns → {kind: column} for x and y. A y column whose kind
    is misspelt ('Y_Long Ttrow In' next to 'X_Long Throw IN') is paired with the closest x kind."""
    import difflib
    xs, ys = {}, {}
    for c in columns:
        m = _COORD.match(_key(c))
        if m:
            (xs if m.group(1) == "x" else ys)[m.group(2).strip()] = c
    for yk in [k for k in ys if k not in xs]:
        free = [k for k in xs if k not in ys]
        best = difflib.get_close_matches(yk, free, n=1, cutoff=0.8)
        if best:
            ys[best[0]] = ys.pop(yk)
    return xs, ys


def _is_second(kind: str) -> bool:
    return any(w in kind for w in _SECOND)


def _side_of(kind: str) -> str:
    return "Right" if re.search(r"\bright\b", kind) else "Left" if re.search(r"\bleft\b", kind) else ""


def _split(v: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;/]", str(v)) if p.strip()]


def _result_of(values: list[str]) -> str:
    """Most dangerous outcome among the coded ones ('Threat, Shot' → Shot)."""
    best, best_rank = "", len(RESULT_RANK) + 1
    for v in values:
        for part in _split(v):
            k = _clean_qual(part).lower()
            rank = RESULT_RANK.index(k) if k in RESULT_RANK else len(RESULT_RANK)
            if rank < best_rank:
                best, best_rank = _clean_qual(part), rank
    return best.title() if best and best.islower() else best


def load_setpieces(src: Any, name: str = "", team: str = "") -> SetPieceData:
    raw = read_table(src, name)
    cols = {_key(c): c for c in raw.columns}
    if "row" not in cols:
        raise ValueError("Expected a coding-matrix export with a 'Row' column (the set-piece type).")
    notes: list[str] = []
    row = raw[cols["row"]].astype(str).str.strip()
    xcols, ycols = _coord_cols(raw.columns)
    gx_col = next((c for c in raw.columns if _key(c) in _GOAL_X), None)
    gy_col = next((c for c in raw.columns if _key(c) in _GOAL_Y), None)
    taker_col = next((c for c in raw.columns if _key(c) in _TAKER), None)
    coord_set = set(xcols.values()) | set(ycols.values()) | {gx_col, gy_col, taker_col}
    qual_cols = [c for c in raw.columns if _key(c) not in _META and c not in coord_set and raw[c].notna().any()]
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
    df["variant"] = row.map(variant_of)
    df["taker"] = (raw[taker_col].map(lambda v: "" if pd.isna(v) else str(int(v)) if isinstance(v, float) and v.is_integer()
                                      else str(v).strip()) if taker_col is not None else "")
    df["goal_x_m"] = pd.to_numeric(raw[gx_col], errors="coerce") if gx_col is not None else np.nan
    df["goal_y_m"] = pd.to_numeric(raw[gy_col], errors="coerce") if gy_col is not None else np.nan
    for c in qual_cols:
        df[str(c).strip()] = raw[c].map(_clean_qual)
    quals = [str(c).strip() for c in qual_cols]

    # overall result: from every "... Outcome" column (1st contact Outcome, 2nd contact Outcome, Outcome)
    out_cols = [q for q in quals if "outcome" in q.lower() or q.lower() == "result"]
    if out_cols:
        df[RESULT_COL] = [_result_of([r[q] for q in out_cols]) for _, r in df[out_cols].iterrows()]
        if out_cols == ["Outcome"]:                    # a single Outcome column → it IS the result
            df["Outcome"] = df[RESULT_COL]             # kept as an alias (older reports / designs use it)
            quals = [q for q in quals if q != "Outcome"]
        quals = [RESULT_COL] + quals
        notes.append(f"Result = the most dangerous outcome coded in: {', '.join(out_cols)}.")
    eff = next((q for q in quals if q.lower() == "efficiency"), None)
    if eff:                                            # "Goal" coded in Efficiency (direct free kicks)
        if RESULT_COL not in df:
            df[RESULT_COL] = ""
            quals = [RESULT_COL] + quals
        df.loc[df[eff].str.lower().eq("goal"), RESULT_COL] = "Goal"
    if eff:                                            # penalties: Efficiency Successful = scored
        pen = df["category"].eq("Penalty") & df[eff].str.lower().str.contains("success") \
            & ~df[eff].str.lower().str.contains("unsuccess")
        if RESULT_COL not in df:
            df[RESULT_COL] = ""
            quals = [RESULT_COL] + quals
        df.loc[pen & df[RESULT_COL].eq(""), RESULT_COL] = "Goal"
    binary = [q for q in quals if set(df[q].str.lower().unique()) - {""} <= (YES | NO) and (df[q] != "").any()]

    # coordinates: landing point of the row's own type (+ side columns), second-ball point
    land_kinds = [k for k in xcols if k in ycols and not _is_second(k)]
    sb_kind = next((k for k in xcols if k in ycols and _is_second(k)), None)

    def kinds_for(rk: str) -> list[str]:
        """Coordinate kinds of a row type: exact, a kind contained in the type ('gk' in 'long gk'), or the type
        contained in the kind ('offensive corner' in 'offensive corner right'); longest first."""
        out = []
        for k in land_kinds:
            base = re.sub(r"\b(right|left)\b", "", k).strip()
            if k == rk or base == rk or re.search(rf"\b{re.escape(base)}\b", rk) or rk in k:
                out.append(k)
        return sorted(out, key=len, reverse=True)

    pts, sides, sbx, sby = [], [], [], []
    for i, r in enumerate(row):
        ri = raw.index[i]
        p, side = [], ""
        for k in kinds_for(_key(r)):
            got = parse_points(raw.at[ri, xcols[k]], raw.at[ri, ycols[k]])
            if got:
                p, side = got, _side_of(k)
                break
        pts.append(p)
        sides.append(side)
        sb = parse_points(raw.at[ri, xcols[sb_kind]], raw.at[ri, ycols[sb_kind]]) if sb_kind else []
        sbx.append(sb[0][0] if sb else np.nan)
        sby.append(sb[0][1] if sb else np.nan)
    df["points"] = pts
    df["x"] = [p[0][0] if p else np.nan for p in pts]
    df["y"] = [p[0][1] if p else np.nan for p in pts]
    df["extra_points"] = [len(p) - 1 if p else 0 for p in pts]
    df["landing_side"] = sides
    df["sb_x"], df["sb_y"] = sbx, sby
    loc = next((q for q in quals if q.lower() in ("location", "side")), None)
    side = df[loc].str.title() if loc else pd.Series("", index=df.index)
    df["side"] = side.where(side.isin(["Left", "Right"]), df["landing_side"])
    missing_xy = sorted({r for r in df["row"].unique() if not kinds_for(_key(r))})
    if missing_xy:
        notes.append("No landing-point columns for: " + ", ".join(missing_xy))
    if sb_kind:
        notes.append(f"Second-ball location from '{xcols[sb_kind]}' / '{ycols[sb_kind]}' "
                     f"({int(df['sb_x'].notna().sum())} coded).")
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
