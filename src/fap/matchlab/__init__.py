"""Match Lab — professional match-analysis engine for video-tagging event exports.

Built for the free-text tagging format (``tag_name`` / ``labels`` / ``team`` / ``players`` /
``pos_x_meters`` / ``pos_y_meters`` on a 105 x 68 m pitch) — the shape produced by the club's
video-tagging tool. Pure (no Streamlit): the page ``fap.ui.builtin.match_lab`` is one consumer.

Layers
  * :mod:`.loader`  — read + normalise: labels → flags, clock → minutes, halves, per-team
    attack-direction normalisation (both teams attack left → right, ``y = 0`` = right touchline),
    pass end-point inference (the export has no target coordinates) and implicit carries.
  * :mod:`.metrics` — pass taxonomy (progressive [Wyscout], direction, length, height, type,
    final-third / box entries, channels), possessions, team + player tables, PPDA, xG.
  * :mod:`.phases`  — phases of play from x / y: build-up, progression, final third in possession;
    high press / mid block / low block + counter-press out of possession; possession funnel.
  * :mod:`.charts`  — publication-grade matplotlib/mplsoccer charts with a Dark / Light theme.
"""
from fap.matchlab.loader import MatchData, load_match, read_table  # noqa: F401

# Bump on any change to loading / metrics / export so cached results from an older engine are
# never served after an update (it is part of every Match Lab cache key).
ENGINE_VERSION = "2026.10.16-3"
