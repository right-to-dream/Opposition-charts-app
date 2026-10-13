"""Read + normalise a video-tagging event export into an analysis-ready :class:`MatchData`.

Coordinates
  The tagging tool writes metres on a 105 x 68 pitch in IMAGE orientation (``y`` grows downward)
  and keeps each team's raw direction of play. Every event is mapped into ONE analysis frame per
  acting team:

      x ∈ [0, 105]  the team attacks left → right (attacked goal at x = 105)
      y ∈ [0, 68]   y = 0 is the team's RIGHT touchline (so on a y-up plot the right flank is
                    at the bottom — the broadcast convention used across FAP)

  The attack direction of each (team, half) is detected from the data (shot locations, then the
  mean forward displacement of completed passes, then "opposite of the other team"), never assumed.

Pass end points
  The export carries no target coordinates. A pass ends where the next ball event happens: the
  team-mate's ``Receive`` (completed) or the opponent's interception / recovery / tackle / block /
  save (lost), or where the ball went out. The inferred end is stored with ``end_source`` so every
  chart can say what it is drawing.

Implicit carries
  ``Receive`` at A followed by the SAME player's next on-ball action at B means the player carried
  the ball A → B. These derived carries (``derived=True``) are what make progressive-carry metrics
  possible on data that tags almost no ``Carry`` events.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fap.pipeline.tagged_events import event_type_for

PITCH_L = 105.0
PITCH_W = 68.0

# ------------------------------------------------------------------ column resolution
_ALIASES: dict[str, tuple[str, ...]] = {
    "tag": ("tag_name", "tag", "tagname", "action", "event", "event_name", "sub_event", "type"),
    "labels": ("labels", "label", "tags", "descriptors", "qualifiers"),
    "team": ("team", "team_name", "squad"),
    "player": ("players", "player", "player_number", "jersey", "shirt", "number", "player_name"),
    "x": ("pos_x_meters", "x", "pos_x", "start_x", "location_x"),
    "y": ("pos_y_meters", "y", "pos_y", "start_y", "location_y"),
    "x2": ("pos_target_x_meters", "end_x", "x2", "target_x", "pass_end_x"),
    "y2": ("pos_target_y_meters", "end_y", "y2", "target_y", "pass_end_y"),
    "clock": ("begin2", "match_clock", "game_time", "clock"),
    "video": ("begin", "start", "video_time", "timestamp"),
    "half": ("half", "period", "half_number"),
}


def _key(c: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def _resolve(df: pd.DataFrame) -> dict[str, str]:
    lookup = {_key(c): c for c in df.columns}
    out: dict[str, str] = {}
    for canon, cands in _ALIASES.items():
        for cand in cands:
            if cand in lookup:
                out[canon] = lookup[cand]
                break
    return out


# ------------------------------------------------------------------ readers
def read_table(src: Any, name: str = "") -> pd.DataFrame:
    """Read an xlsx / xls / csv from a path, bytes, a file-like object or pass a DataFrame through."""
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
    lower = name.lower()
    bio = io.BytesIO(data)
    if lower.endswith((".csv", ".txt")):
        try:
            return pd.read_csv(bio)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
    return pd.read_excel(bio)


# ------------------------------------------------------------------ parsing helpers
def parse_clock(value: Any) -> float:
    """'mm:ss.ff' / 'h:mm:ss.ff' / '90+05:22.94' (stoppage) → seconds. NaN when unparseable."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if hasattr(value, "hour") and hasattr(value, "minute"):          # datetime.time
        return value.hour * 3600 + value.minute * 60 + value.second + getattr(value, "microsecond", 0) / 1e6
    s = str(value).strip()
    if not s:
        return np.nan
    base = 0.0
    if "+" in s:
        head, s = s.split("+", 1)
        try:
            base = float(head) * 60.0
        except ValueError:
            return np.nan
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return np.nan
    secs = 0.0
    for n in nums:
        secs = secs * 60.0 + n
    return base + secs


def split_labels(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return tuple(t.strip().lower() for t in str(value).split(",") if t.strip())


# Event-type normalisation for the tagging vocabulary (specific words first, then the shared
# keyword classifier from the import pipeline so other tagging dialects still work).
_SPECIFIC = {
    "dribble past": "dribbled_past",      # the defender WAS dribbled past (defensive failure)
    "dribble tackle": "tackle",
    "ball out": "ball_out",
    "saved": "save",
    "errors": "error", "error": "error",
    "duels": "duel", "duel": "duel",
    "fouls": "foul", "foul": "foul",
    "cleareance": "clearance", "clearance": "clearance",
    "half start": "half_start", "half end": "half_end",
    "sub in": "sub_in", "sub out": "sub_out",
    "press": "press", "pressure": "press",
}


def normalise_etype(tag: Any) -> str:
    t = str(tag).strip().lower() if tag is not None and tag == tag else ""
    if not t:
        return ""
    if t in _SPECIFIC:
        return _SPECIFIC[t]
    et = event_type_for(t)
    return {"pressure": "press", "cross": "pass", "dispossessed": "ball_lost"}.get(et, et)


_SUCC = ("successfull", "successful", "success", "succ", "won", "complete", "completed", "goal")
_FAIL = ("unsuccessfull", "unsuccessful", "unsucc", "fail", "failed", "lost", "miss")


def outcome_from(labels: tuple[str, ...], tag: str) -> float:
    """1.0 success / 0.0 failure / NaN unknown — from labels first, then tag wording."""
    words = set(labels)
    if words & set(_FAIL):
        return 0.0
    if words & set(_SUCC):
        return 1.0
    t = tag.lower()
    if any(w in t for w in ("unsucc", "fail", "lost")):
        return 0.0
    if any(w in t for w in ("succ", "won", "complete")):
        return 1.0
    return np.nan


# ------------------------------------------------------------------ data container
@dataclass
class MatchData:
    events: pd.DataFrame                    # every event, analysis-frame coordinates
    teams: list[str]
    source_name: str = ""
    orientation: dict = field(default_factory=dict)   # (team, half) -> "→" / "←" (raw)
    notes: list[str] = field(default_factory=list)
    possessions: pd.DataFrame = field(default_factory=pd.DataFrame)
    thirds: tuple = (35.0, 70.0)

    # convenience views -------------------------------------------------------
    def team_events(self, team: str) -> pd.DataFrame:
        return self.events[self.events["team"] == team]

    def opponent(self, team: str) -> str:
        others = [t for t in self.teams if t != team]
        return others[0] if others else ""

    def passes(self, team: str | None = None) -> pd.DataFrame:
        p = self.events[self.events["etype"] == "pass"]
        return p if team is None else p[p["team"] == team]

    def carries(self, team: str | None = None) -> pd.DataFrame:
        c = self.events[self.events["etype"] == "carry"]
        return c if team is None else c[c["team"] == team]

    @property
    def title(self) -> str:
        return " vs ".join(self.teams) if self.teams else (self.source_name or "Match")


# ------------------------------------------------------------------ main entry
_DEF_EVENTS = {"interception", "recovery", "tackle", "block", "save", "clearance", "duel"}
_ON_BALL = {"pass", "shot", "dribble", "carry"}


def load_match(src: Any, name: str = "", *, mirror_y: bool = False,
               thirds: tuple[float, float] = (35.0, 70.0)) -> MatchData:
    """Read + normalise a tagged export. ``mirror_y`` flips left/right flanks for tagging tools
    that write ``y`` upward instead of image-downward; ``thirds`` are the phase lines (metres)."""
    raw = read_table(src, name)
    cols = _resolve(raw)
    for need in ("tag", "x", "y"):
        if need not in cols:
            raise ValueError(f"Missing a '{need}' column — expected a tagged event export "
                             "(tag_name / pos_x_meters / pos_y_meters).")
    notes: list[str] = []
    ev = pd.DataFrame(index=raw.index)
    ev["tag"] = raw[cols["tag"]].astype(str).str.strip()
    ev["etype"] = ev["tag"].map(normalise_etype)
    ev["labels"] = raw[cols["labels"]].map(split_labels) if "labels" in cols else [()] * len(raw)
    ev["team"] = raw[cols["team"]].fillna("").astype(str).str.strip() if "team" in cols else ""
    if "player" in cols:
        pl = raw[cols["player"]]
        num = pd.to_numeric(pl, errors="coerce")
        ev["player"] = np.where(num.notna(), num.fillna(-1).astype(int).astype(str), pl.fillna("").astype(str))
        ev.loc[ev["player"].isin(["-1", "nan"]), "player"] = ""
    else:
        ev["player"] = ""
    ev["raw_x"] = pd.to_numeric(raw[cols["x"]], errors="coerce")
    ev["raw_y"] = pd.to_numeric(raw[cols["y"]], errors="coerce")
    ev["raw_x2"] = pd.to_numeric(raw[cols["x2"]], errors="coerce") if "x2" in cols else np.nan
    ev["raw_y2"] = pd.to_numeric(raw[cols["y2"]], errors="coerce") if "y2" in cols else np.nan
    if mirror_y:
        ev["raw_y"] = PITCH_W - ev["raw_y"]
        ev["raw_y2"] = PITCH_W - ev["raw_y2"]

    # time ----------------------------------------------------------------
    clock = raw[cols["clock"]].map(parse_clock) if "clock" in cols else pd.Series(np.nan, index=raw.index)
    video = raw[cols["video"]].map(parse_clock) if "video" in cols else pd.Series(np.nan, index=raw.index)
    ev["t"] = video if video.notna().any() else clock               # monotonic, for gaps
    if ev["t"].isna().all():
        ev["t"] = np.arange(len(ev), dtype=float) * 3.0
        notes.append("No time columns — event order used as time.")
    ev["t"] = ev["t"].ffill().fillna(0.0)

    # halves ----------------------------------------------------------------
    if "half" in cols:
        ev["half"] = pd.to_numeric(raw[cols["half"]], errors="coerce").ffill().fillna(1).astype(int)
    else:
        starts = (ev["etype"] == "half_start").cumsum()
        ev["half"] = starts.clip(lower=1).astype(int) if starts.max() > 0 else 1
        if starts.max() == 0 and clock.notna().any():
            ev["half"] = np.where(clock >= 45 * 60, 2, 1)
    if clock.notna().any():
        ev["clock"] = clock.ffill().fillna(0.0)
    else:
        ev["clock"] = ev.groupby("half")["t"].transform(lambda s: s - s.min()) + (ev["half"] - 1) * 2700
    ev["minute"] = ev["clock"] / 60.0

    ev = ev.sort_values(["half", "t"], kind="stable").reset_index(drop=True)
    ev["outcome"] = [outcome_from(l, t) for l, t in zip(ev["labels"], ev["tag"])]
    for lab in ("set piece", "transition", "build up", "regular play", "under pressure", "counter press"):
        ev["lab_" + lab.replace(" ", "_")] = ev["labels"].map(lambda ls, w=lab: w in ls)

    teams = [t for t in ev["team"].value_counts().index.tolist() if t]

    # pass end points (raw frame, before orientation) -------------------------
    ev = _fix_late_receives(ev)
    _infer_pass_ends(ev)

    # orientation per (team, half) ---------------------------------------------
    orient: dict = {}
    for half in sorted(ev["half"].unique()):
        dirs = {team: _attack_sign(ev, team, half) for team in teams}
        known = {t: d for t, d in dirs.items() if d != 0}
        for t in teams:
            if dirs[t] == 0:
                other = [d for tt, d in known.items() if tt != t]
                dirs[t] = -other[0] if other else 1
                notes.append(f"Attack direction of {t} (half {half}) inferred from the opponent.")
        for t, d in dirs.items():
            orient[(t, int(half))] = "→" if d > 0 else "←"
    ev["attack_sign"] = [1 if orient.get((tm, int(h)), "→") == "→" else -1
                         for tm, h in zip(ev["team"], ev["half"])]
    sign = ev["attack_sign"].to_numpy()
    for src_x, src_y, dx, dy in (("raw_x", "raw_y", "x", "y"), ("end_raw_x", "end_raw_y", "end_x", "end_y")):
        rx, ry = ev[src_x].to_numpy(float), ev[src_y].to_numpy(float)
        ev[dx] = np.where(sign > 0, rx, PITCH_L - rx).clip(0, PITCH_L)
        ev[dy] = np.where(sign > 0, PITCH_W - ry, ry).clip(0, PITCH_W)

    ev = _add_carries(ev)
    md = MatchData(events=ev, teams=teams, source_name=name, orientation=orient, notes=notes,
                   thirds=(float(thirds[0]), float(thirds[1])))
    from fap.matchlab import metrics   # local import: metrics depends on loader
    metrics.enrich(md)
    return md


def _attack_sign(ev: pd.DataFrame, team: str, half: int) -> int:
    """+1 when the team attacks toward raw x = 105 in this half, -1 toward x = 0, 0 unknown."""
    g = ev[(ev["team"] == team) & (ev["half"] == half)]
    shots = g[g["etype"] == "shot"]["raw_x"].dropna()
    if len(shots) >= 2:
        return 1 if shots.mean() > PITCH_L / 2 else -1
    p = g[(g["etype"] == "pass") & (g["end_source"] == "receive")]
    dx = (p["end_raw_x"] - p["raw_x"]).dropna()
    if len(dx) >= 15 and abs(dx.mean()) > 0.5:
        return 1 if dx.mean() > 0 else -1
    if len(shots) == 1:
        return 1 if shots.iloc[0] > PITCH_L / 2 else -1
    return 0


def _known(p: Any) -> bool:
    """A real player id ('' and '0' are the tagger's 'unknown player')."""
    return bool(p) and p not in ("0", "nan")


def _fix_late_receives(ev: pd.DataFrame, max_lag: float = 2.5) -> pd.DataFrame:
    """Tagging clips have different pre-rolls, so a player's ``Receive`` is often stamped just AFTER
    his own next pass / shot / dribble. Move such a Receive in front of that action so the event
    chain reads receive → action (needed for pass end points and carries)."""
    et, tm, pl, t = (ev[c].tolist() for c in ("etype", "team", "player", "t"))
    order = list(range(len(ev)))
    i = 0
    while i < len(order) - 1:
        a, b = order[i], order[i + 1]
        if (et[b] == "receive" and et[a] in ("pass", "shot", "dribble", "carry") and _known(pl[a]) and pl[a] == pl[b]
                and tm[a] == tm[b] and 0 <= t[b] - t[a] <= max_lag):
            order[i], order[i + 1] = b, a
            i = max(i - 1, 0)
            continue
        i += 1
    return ev.iloc[order].reset_index(drop=True)


def _infer_pass_ends(ev: pd.DataFrame, horizon_rows: int = 7, horizon_s: float = 12.0) -> None:
    n = len(ev)
    ex = np.full(n, np.nan)
    ey = np.full(n, np.nan)
    src = np.array([""] * n, dtype=object)
    receiver = np.array([""] * n, dtype=object)
    et = ev["etype"].to_numpy()
    tm = ev["team"].to_numpy()
    pl = ev["player"].to_numpy()
    lab = ev["labels"].to_numpy()
    oc = ev["outcome"].to_numpy(float)
    x, y, t = ev["raw_x"].to_numpy(float), ev["raw_y"].to_numpy(float), ev["t"].to_numpy(float)
    x2, y2 = ev["raw_x2"].to_numpy(float), ev["raw_y2"].to_numpy(float)
    half = ev["half"].to_numpy()
    restarts = ("throw in", "corner", "goal kick", "free kick", "kick off")
    for i in np.where(et == "pass")[0]:
        failed = oc[i] == 0.0
        if not np.isnan(x2[i]) and not np.isnan(y2[i]):
            ex[i], ey[i], src[i] = x2[i], y2[i], "tagged"
        for j in range(i + 1, min(n, i + 1 + horizon_rows)):
            if half[j] != half[i] or t[j] - t[i] > horizon_s:
                break
            if np.isnan(x[j]) or et[j] in ("press", "sub_in", "sub_out"):
                continue
            same = tm[j] == tm[i]
            if et[j] in ("ball_out", "offside") or (et[j] == "pass" and any(r in lab[j] for r in restarts)):
                if not src[i]:
                    ex[i], ey[i], src[i] = x[j], y[j], "ball_out" if et[j] != "offside" else "offside"
                break
            if same and _known(pl[i]) and pl[j] == pl[i]:
                continue                                   # the passer's own (late-tagged) events
            if same and et[j] == "receive":
                if failed and oc[j] != 0.0:
                    continue                               # contradiction: keep looking for the opponent
                if not src[i]:
                    ex[i], ey[i], src[i] = x[j], y[j], "receive"
                receiver[i] = pl[j]
                break
            if not same and et[j] in _DEF_EVENTS | {"pass", "receive", "foul"}:
                if not src[i]:
                    ex[i], ey[i], src[i] = x[j], y[j], "opponent"
                break
            if same and et[j] in _ON_BALL and (pl[j] != pl[i] or not _known(pl[i])):
                if failed:
                    continue
                if not src[i]:                             # receive not tagged — next team-mate action
                    ex[i], ey[i], src[i] = x[j], y[j], "next_action"
                receiver[i] = pl[j]
                break
    ev["end_raw_x"] = np.where(et == "pass", ex, np.nan)
    ev["end_raw_y"] = np.where(et == "pass", ey, np.nan)
    ev["end_source"] = src
    ev["receiver"] = receiver
    # outcome fallback for passes without a success label
    po = ev["outcome"].to_numpy(float).copy()
    for i in np.where((et == "pass") & np.isnan(po))[0]:
        if src[i] in ("receive", "next_action"):
            po[i] = 1.0
        elif src[i] in ("opponent", "ball_out", "offside"):
            po[i] = 0.0
    ev["outcome"] = po


def _add_carries(ev: pd.DataFrame, min_dist: float = 5.0, max_gap: float = 15.0) -> pd.DataFrame:
    """Derive carries: Receive at A → the same player's next on-ball action at B (same team, same
    half, within ``max_gap`` s, no opponent event in between). Tagged ``Carry`` rows without an end
    point get their end from the same rule."""
    rows = []
    n = len(ev)
    et, tm, pl = ev["etype"].to_numpy(), ev["team"].to_numpy(), ev["player"].to_numpy()
    x, y, t, half = ev["x"].to_numpy(float), ev["y"].to_numpy(float), ev["t"].to_numpy(float), ev["half"].to_numpy()
    for i in range(n):
        if et[i] not in ("receive", "recovery", "interception") or not _known(pl[i]) or np.isnan(x[i]):
            continue
        if et[i] != "receive" and ev.at[i, "outcome"] == 0.0:
            continue
        for j in range(i + 1, min(n, i + 4)):
            if tm[j] != tm[i] or half[j] != half[i] or t[j] - t[i] > max_gap:
                break
            if et[j] in ("press", "duel", "foul", "error"):
                continue
            if et[j] in ("pass", "shot", "dribble", "carry") and pl[j] == pl[i] and not np.isnan(x[j]):
                d = float(np.hypot(x[j] - x[i], y[j] - y[i]))
                if d >= min_dist:
                    r = ev.iloc[i].to_dict()
                    r.update(etype="carry", tag="Carry (derived)", x=x[i], y=y[i], end_x=x[j], end_y=y[j],
                             end_source="derived", outcome=1.0, t=t[i] + 0.01, derived=True,
                             labels=(), receiver="")
                    rows.append(r)
            break
    ev = ev.copy()
    ev["derived"] = False
    if rows:
        ev = pd.concat([ev, pd.DataFrame(rows)], ignore_index=True)
        ev = ev.sort_values(["half", "t"], kind="stable").reset_index(drop=True)
    return ev
