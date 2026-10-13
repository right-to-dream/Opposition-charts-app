"""Match Lab engine: orientation, pass end inference, taxonomy, phases and chart smoke tests."""
from __future__ import annotations

import pandas as pd
import pytest

from fap.matchlab import load_match
from fap.matchlab import metrics as M
from fap.matchlab import phases as P


def _row(tag, team, player, x, y, t, labels=""):
    return {"tag_name": tag, "begin": f"00:{t:05.2f}" if t < 60 else f"{int(t // 60):02d}:{t % 60:05.2f}",
            "players": player, "labels": labels, "team": team, "pos_x_meters": x, "pos_y_meters": y}


def _match() -> pd.DataFrame:
    """'A' attacks toward raw x = 0 (like the real export), 'B' toward x = 105. Image-y (down)."""
    rows = [_row("Half Start", None, None, None, None, 0.0)]
    t = 1.0
    # A: GK build-up → CB → progressive pass → receive in final third → shot
    rows += [
        _row("Pass", "A", 1, 100, 34, t, "Ground, Successfull"),
        _row("Receive", "A", 4, 90, 50, t + 1, "Successfull"),
        _row("Pass", "A", 4, 88, 50, t + 3, "Ground, Successfull, Line break, Mid"),
        _row("Receive", "A", 10, 40, 30, t + 5, "Successfull"),
        _row("Shot", "A", 10, 12, 34, t + 7, "Right Foot, Off target"),
        _row("Shot", "A", 10, 10, 30, t + 9, "Right Foot, Goal"),
    ]
    t = 30.0
    # A: cross whose passer's own Receive is tagged late, then an opponent clearance
    rows += [
        _row("Pass", "A", 7, 20, 60, t, "Cross, Low, Unsuccessfull"),
        _row("Receive", "A", 7, 25, 62, t + 0.5, "Successfull"),
        _row("Interception", "B", 0, 8, 36, t + 1.5, "Clear"),
        # B: two passes, B attacks toward x = 105, players unknown ("0")
        _row("Recovery", "B", 0, 20, 30, t + 4),
        _row("Pass", "B", 0, 22, 30, t + 5, "Ground, Successfull"),
        _row("Receive", "B", 0, 50, 20, t + 7, "Successfull"),
        _row("Pass", "B", 0, 52, 20, t + 8, "High, Unsuccessfull"),
        _row("Interception", "A", 3, 70, 10, t + 10, "Won"),
        _row("Shot", "B", 0, 95, 34, t + 20, "Head, Off target"),
        _row("Shot", "B", 0, 92, 30, t + 25, "Right Foot, Off target"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def md():
    return load_match(_match(), "synthetic.xlsx")


def test_orientation_detected_and_normalised(md):
    assert md.orientation[("A", 1)] == "←"
    assert md.orientation[("B", 1)] == "→"
    shots = md.events[md.events["etype"] == "shot"]
    assert (shots["x"] > 80).all()                    # every team now attacks toward x = 105


def test_flank_convention_y0_is_right(md):
    # A attacks toward raw x=0 in image coords: raw y small = top = A's right → analysis y small.
    ev = md.events
    a_int = ev[(ev["team"] == "A") & (ev["etype"] == "interception")].iloc[0]
    assert a_int["y"] == pytest.approx(10)
    assert M.channel_of([a_int["y"]])[0] == "Right flank"


def test_pass_end_from_receive_and_receiver(md):
    p = md.events[(md.events["etype"] == "pass") & (md.events["player"] == "4")].iloc[0]
    assert p["end_source"] == "receive"
    assert p["receiver"] == "10"
    assert (p["x"], p["end_x"]) == pytest.approx((105 - 88, 105 - 40))
    assert p["progressive"] and p["direction"] == "Forward" and p["line_break"]


def test_late_receive_skipped_failed_cross_ends_at_opponent(md):
    c = md.events[md.events["cross"]].iloc[0]
    assert c["end_source"] == "opponent"
    assert c["outcome"] == 0
    assert c["end_x"] == pytest.approx(105 - 8)       # clearance location in the passer's frame


def test_unknown_player_zero_not_treated_as_same_player(md):
    b = md.events[(md.events["team"] == "B") & (md.events["etype"] == "pass")]
    assert (b["end_source"] != "").all()
    assert b.iloc[0]["end_source"] == "receive"
    assert not md.events[(md.events["team"] == "B") & md.events["derived"]].shape[0]


def test_progressive_thresholds():
    # own half → own half needs 30 m gain; 20 m gain is not progressive
    assert not M.progressive_mask([10], [34], [30], [34])[0]
    assert M.progressive_mask([10], [34], [45], [34])[0]
    # opponent half: 10 m enough
    assert M.progressive_mask([60], [34], [71], [34])[0]


def test_phases_and_possessions(md):
    ev = md.events
    first = ev[(ev["team"] == "A") & (ev["etype"] == "pass")].iloc[0]
    assert first["phase"] == "Build-up"
    tab = P.in_possession_table(md, "A")
    assert list(tab["Phase"]) == ["Build-up", "Progression", "Final third"]
    assert not md.possessions.empty
    fn = P.funnel(md, "A")
    assert fn[0][1] >= fn[-1][1]
    dt = P.out_of_possession_table(md, "A")
    assert dt["Actions"].sum() >= 1


def test_shots_and_summary(md):
    s = M.team_summary(md, "A")
    assert s["Goals"] == 1 and s["Shots"] == 2
    assert s["xG"] > 0


def test_all_charts_render(md):
    from fap.matchlab import charts as CH
    for spec in CH.CHARTS:
        for theme in (CH.DARK, CH.LIGHT):
            fig = CH.render(spec.key, md, "A", theme=theme, player="10" if spec.needs_player else None)
            assert CH.to_png(fig, dpi=40)[:4] == b"\x89PNG"


def test_real_file_if_present():
    import pathlib
    p = pathlib.Path(__file__).parent.parent / "sample_data" / "Masar Vs Mansoura Final.xlsx"
    if not p.exists():
        pytest.skip("real match file not in sample_data")
    md = load_match(p)
    passes = md.passes()
    assert (passes["end_source"] != "").mean() > 0.95
