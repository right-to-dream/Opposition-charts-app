"""Set Piece Lab — set-piece analysis from a coding-matrix export (Sportscode / Hudl style) and a
routine & drill designer.

* :mod:`.loader`    — one row per set piece (``Row`` = type), every other column is a qualifier,
                      ``x_<type>`` / ``y_<type>`` = landing point(s) of that type on a 105 × 68 pitch.
* :mod:`.analytics` — counts, qualifier breakdowns, contest rates, cross-tabs, deterministic insights.
* :mod:`.charts`    — interactive (hover) charts reusing the Match Lab Plotly pitch.
* :mod:`.designer`  — set-piece templates (multi-step boards for animation) on the Tactical Board model,
                      a data-driven "their pattern" board, and the routine / drill library.
"""
from fap.setpiecelab.loader import SetPieceData, load_setpieces  # noqa: F401
