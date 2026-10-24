"""Player Tag Panel — a position-aware Hudl Sportscode code window for individual player analysis.

* :mod:`taxonomy` — the code / label vocabulary (one schema for every position) and the 18-zone grid.
* :mod:`window`   — the button layout per position (mock-up PNG + build sheet share it).
* :mod:`loader`   — reads the Sportscode export (timeline XML or instances CSV/XLSX) into a tidy table.
* :mod:`stats`    — per-position KPIs, per-match table, zone and action summaries.
* :mod:`charts`   — static PNG charts; :mod:`export` — build sheet, data workbook, sample XML.
"""
from fap.playertag.loader import TagData, combine, load_tags

__all__ = ["TagData", "combine", "load_tags"]
