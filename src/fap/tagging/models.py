"""Tagging data model + session with snapshot undo/redo.

``TagEvent`` is one tagged event in canonical coordinates. ``TaggingSession`` owns
the ordered events plus optional match context and an attack-direction flag, and
provides create/edit/delete/move as history-recorded operations so undo/redo works
uniformly for every kind of change (JSON-serialisable snapshots, exactly like the
Tactical Board's ``History``). The session is fully round-trippable to a dict, so
autosave and project save can persist and restore it (events, metadata AND history).
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

_COORD_FIELDS = ("x", "y", "x2", "y2", "goal_x", "goal_y")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class TagEvent:
    id: str = ""
    team: str = ""
    player: str = ""
    period: str = "1H"
    minute: int | None = None
    second: int | None = None
    event_type: str = ""
    outcome: str = ""
    coordinate_space: str = "pitch"           # pitch | goal
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    goal_x: float | None = None
    goal_y: float | None = None
    notes: str = ""
    video_timestamp: float | None = None      # seconds into the source video (nullable)
    frame: int | None = None
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "e_" + uuid.uuid4().hex[:12]
        self.minute, self.second = _int(self.minute), _int(self.second)
        self.frame = _int(self.frame)
        self.video_timestamp = _num(self.video_timestamp)
        for f in _COORD_FIELDS:
            setattr(self, f, _num(getattr(self, f)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TagEvent":
        known = {f for f in cls.__slots__}                # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class History:
    """Undo/redo over JSON-serialisable session snapshots (deterministic)."""

    def __init__(self, limit: int = 100) -> None:
        self._undo: list[dict[str, Any]] = []
        self._redo: list[dict[str, Any]] = []
        self._limit = limit

    def record(self, snapshot: dict[str, Any]) -> None:
        self._undo.append(snapshot)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self, current: dict[str, Any]) -> dict[str, Any] | None:
        if not self._undo:
            return None
        self._redo.append(current)
        return self._undo.pop()

    def redo(self, current: dict[str, Any]) -> dict[str, Any] | None:
        if not self._redo:
            return None
        self._undo.append(current)
        return self._redo.pop()

    def to_dict(self) -> dict[str, Any]:
        return {"undo": self._undo, "redo": self._redo, "limit": self._limit}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "History":
        h = cls(limit=int((d or {}).get("limit", 100)))
        h._undo = list((d or {}).get("undo") or [])
        h._redo = list((d or {}).get("redo") or [])
        return h


_META_FIELDS = ("match_id", "competition", "match_date", "opponent", "analyst",
                "home_team", "away_team", "attack_direction", "preset", "pitch_length", "pitch_width")


@dataclass(slots=True)
class TaggingSession:
    events: list[TagEvent] = field(default_factory=list)
    match_id: str = ""
    competition: str = ""
    match_date: str = ""
    opponent: str = ""
    analyst: str = ""
    home_team: str = "Team A"
    away_team: str = "Team B"
    attack_direction: str = "lr"              # lr = left->right, rl = right->left
    preset: str = "All"
    pitch_length: float = 105.0               # metres — canonical 0..100 maps onto these
    pitch_width: float = 68.0
    _history: History = field(default_factory=History, repr=False)

    # -- snapshot state (events + metadata; history is separate) ---------------
    def _state(self) -> dict[str, Any]:
        return {"events": [e.to_dict() for e in self.events],
                "meta": {k: getattr(self, k) for k in _META_FIELDS}}

    def _restore(self, state: dict[str, Any]) -> None:
        self.events = [TagEvent.from_dict(e) for e in state.get("events", [])]
        for k, v in (state.get("meta") or {}).items():
            if k in _META_FIELDS:
                setattr(self, k, v)

    def _checkpoint(self) -> None:
        self._history.record(self._state())

    # -- queries ---------------------------------------------------------------
    def get(self, event_id: str) -> TagEvent | None:
        return next((e for e in self.events if e.id == event_id), None)

    def index_of(self, event_id: str) -> int:
        return next((i for i, e in enumerate(self.events) if e.id == event_id), -1)

    def __len__(self) -> int:
        return len(self.events)

    # -- history-recorded operations ------------------------------------------
    def add_event(self, event: TagEvent) -> TagEvent:
        self._checkpoint()
        self.events.append(event)
        return event

    def delete_event(self, event_id: str) -> bool:
        idx = self.index_of(event_id)
        if idx < 0:
            return False
        self._checkpoint()
        self.events.pop(idx)
        return True

    def edit_event(self, event_id: str, **fields: Any) -> TagEvent | None:
        event = self.get(event_id)
        if event is None:
            return None
        self._checkpoint()
        coords = {"x", "y", "x2", "y2", "goal_x", "goal_y", "video_timestamp"}
        ints = {"minute", "second", "frame"}
        for k, v in fields.items():
            if not hasattr(event, k) or k == "id":
                continue
            setattr(event, k, _num(v) if k in coords else _int(v) if k in ints else v)
        return event

    def move_event(self, event_id: str, **coords: Any) -> TagEvent | None:
        allowed = {k: v for k, v in coords.items()
                   if k in ("x", "y", "x2", "y2", "goal_x", "goal_y")}
        return self.edit_event(event_id, **allowed) if allowed else self.get(event_id)

    def set_meta(self, **fields: Any) -> None:
        changed = {k: v for k, v in fields.items() if k in _META_FIELDS
                   and getattr(self, k) != v}
        if not changed:
            return
        self._checkpoint()
        for k, v in changed.items():
            setattr(self, k, v)

    def clear(self) -> None:
        if self.events:
            self._checkpoint()
        self.events = []

    # -- undo / redo -----------------------------------------------------------
    def can_undo(self) -> bool:
        return self._history.can_undo()

    def can_redo(self) -> bool:
        return self._history.can_redo()

    def undo(self) -> bool:
        state = self._history.undo(self._state())
        if state is None:
            return False
        self._restore(state)
        return True

    def redo(self) -> bool:
        state = self._history.redo(self._state())
        if state is None:
            return False
        self._restore(state)
        return True

    # -- serialisation (autosave / project save) ------------------------------
    def to_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"events": [e.to_dict() for e in self.events]}
        out.update({k: getattr(self, k) for k in _META_FIELDS})
        if include_history:
            out["history"] = self._history.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaggingSession":
        d = d or {}
        s = cls(events=[TagEvent.from_dict(e) for e in d.get("events", [])])
        for k in _META_FIELDS:
            if k in d and d[k] is not None:
                setattr(s, k, d[k])
        if "history" in d:
            s._history = History.from_dict(d["history"])
        return s
