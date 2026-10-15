"""Per-match memoisation for the engine's heavier analyses.

A loaded :class:`~fap.matchlab.loader.MatchData` never changes after ``load_match`` (settings such as
the phase lines or the counter-press window produce a NEW MatchData), so results can be cached on the
object itself. Callers get a copy of DataFrame results (and a shallow copy of dicts) so nothing they do
can alter the cached value.
"""
from __future__ import annotations

import functools

import pandas as pd


def _copy(v):
    if isinstance(v, pd.DataFrame | pd.Series):
        return v.copy()
    if isinstance(v, dict):
        return dict(v)
    return v


def per_match(fn):
    @functools.wraps(fn)
    def wrapper(md, *args, **kwargs):
        store = md.__dict__.setdefault("_memo", {})
        try:
            key = (fn.__module__, fn.__qualname__, args, tuple(sorted(kwargs.items())))
            hash(key)
        except TypeError:                       # unhashable argument → no caching
            return fn(md, *args, **kwargs)
        if key not in store:
            store[key] = fn(md, *args, **kwargs)
        return _copy(store[key])
    return wrapper
