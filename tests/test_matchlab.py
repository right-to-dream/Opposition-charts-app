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


# ---------------------------------------------------------------- phase 1: dialects, audit, IDs, export
def _proxy_style() -> pd.DataFrame:
    """Outcome-in-tag dialect: 'Succ Pass' / 'Fail Cross' / 'Shot On Target' / 'Goal' / '1 Half Start'."""
    r = lambda tag, team, pl, x, y, t, lab="": {"tag_name": tag, "begin": f"00:{t:05.2f}", "end": "",
                                                "begin_match": f"00:{t:05.2f}", "end_match": "", "players": pl,
                                                "labels": lab, "team": team, "pos_x_meters": x, "pos_y_meters": y}
    return pd.DataFrame([
        r("1 Half Start", None, None, None, None, 0.0),
        r("Succ Pass", "Masar", 33, 90, 34, 1.0, "Line Break, Under Press"),
        r("Receive", "Masar", 3, 60, 30, 2.0, "Successfull"),
        r("Fail Cross", "Masar", 3, 20, 60, 4.0),
        r("Interception", "Proxy", 0, 10, 36, 5.0, "Clear"),
        r("Shot On Target", "Masar", 10, 12, 34, 9.0, "Head"),
        r("Goal", "Masar", "0, 10", 10, 32, 12.0, "Right Foot, Penalty"),
        r("Shot Off Target", "Proxy", 0, 92, 30, 20.0),
        r("Shot Blocked", "Proxy", 0, 95, 36, 22.0),
        r("2 Half Start", None, None, None, None, 30.0),
        r("Pass", "Masar", 33, 90, 34, 31.0, "Successfull, Unsuccessfull"),
        r("Receive", "Masar", 3, 70, 30, 32.0, "Successful"),
    ])


@pytest.fixture(scope="module")
def proxy():
    return load_match(_proxy_style(), "proxy.xlsx")


def test_outcome_in_tag_dialect(proxy):
    ev = proxy.events
    assert ev["half"].max() == 2
    p = ev[ev["etype"] == "pass"]
    assert list(p["outcome"][:2]) == [1.0, 0.0]
    assert bool(p.iloc[1]["cross"]) and bool(p.iloc[0]["line_break"]) and bool(p.iloc[0]["under_pressure"])
    s = ev[ev["etype"] == "shot"]
    assert list(s["shot_result"]) == ["Saved", "Goal", "Off target", "Blocked"]
    assert ev[ev["tag"] == "Goal"]["player"].iloc[0] == "10"           # '0, 10' -> 10
    assert proxy.our_team == "Masar"


def test_contradictory_labels_resolved_from_next_event(proxy):
    p = proxy.events[(proxy.events["etype"] == "pass") & (proxy.events["half"] == 2)].iloc[0]
    assert p["outcome"] == 1.0 and p["end_source"] == "receive"


def test_audit_has_no_failures(md, proxy):
    from fap.matchlab import audit
    for m in (md, proxy):
        a = audit.run(m)
        assert not a.empty and (a["Status"] != "FAIL").all(), a[a["Status"] == "FAIL"].to_string()


def test_team_matching_case_insensitive_and_normalised():
    from fap.matchlab.reference import find_matches, match_team
    teams = pd.DataFrame({"team_id": ["a", "b", "c", "d"],
                          "team": ["FC Masar Men", "Proxy SC", "El Dakhleya", "Baladiyet El Mahalla"]})
    assert match_team("MASAR", teams).team_id == "a"
    assert match_team("proxy", teams).team_id == "b"
    assert match_team("Dakhlya", teams).team_id == "c"
    assert match_team("mahalla", teams).team_id == "d"
    assert match_team("Zamalek", teams).team_id == ""
    matches = pd.DataFrame({"match_id": ["m1", "m2"], "match": ["x", "y"],
                            "home_team": ["FC Masar Men", "El Dakhleya"], "away_team": ["Proxy SC", "FC Masar Men"]})
    assert list(find_matches("Masar", "Proxy", matches, teams)["match_id"]) == ["m1"]


def test_squad_file_and_ambiguous_shirts():
    from fap.matchlab.reference import squad_from_table
    sq = squad_from_table(pd.DataFrame({"player_id": ["p1", "p2", "p3"], "player_name": ["A", "B", "C"],
                                        "shirt": [10, 15, 15]}))
    assert sq.lookup("10") == ("p1", "A")
    assert sq.lookup("15") == ("", "") and sq.ambiguous == ("15",)


def test_export_schema_ids_and_label_columns(proxy):
    from fap.matchlab import export as X
    from fap.matchlab.reference import Reference, Squad
    ref = Reference(pd.DataFrame({"team_id": ["T1", "T2"], "team": ["FC Masar Men", "Proxy SC"]}), None,
                    Squad({"33": ("P33", "Kamel"), "3": ("P3", "Abdelrahman")}))
    e = X.build_events(proxy, ref, "M1")
    assert list(e.columns[:45]) == X.SCHEMA_44
    assert list(e.columns[7:9]) == ["event_type", "outcome"]
    assert e.loc[e["event_id"] == 3, "outcome"].iloc[0] == "successful"      # Receive labelled Successfull
    assert "label_successful" not in e.columns and "outcome_source" not in e.columns
    assert len(e) == len(proxy.raw)
    assert (e["match_id"] == "M1").all()
    first = e[e["event_id"] == 2].iloc[0]
    assert (first["team_id"], first["player_id"], first["player_name"]) == ("T1", "P33", "Kamel")
    assert first["pass_line_break"] is True or first["pass_line_break"] == True  # noqa: E712
    assert first["pass_under_pressure"] == True  # noqa: E712
    assert (first["start_x"], first["end_x"]) == (90, 60)            # file orientation kept
    assert first["receiver_name"] == "Abdelrahman"
    assert e[e["team"] == "Proxy SC"]["player_id"].eq("").all()        # opponent: no player ids
    assert {"label_line_break", "label_under_pressure", "label_clear"} <= set(e.columns)
    assert e.loc[e["event_id"] == 2, "labels"].iloc[0] == "Line Break, Under Press"   # raw text kept
    assert e["pass_cross"].dtype == bool
    wb = X.build_workbook(proxy, ref, "M1")
    assert set(pd.ExcelFile(__import__("io").BytesIO(wb)).sheet_names) >= {"Events", "Data audit", "Info"}


# ---------------------------------------------------------------- phase 2: counter-press, sequences, halves, panels
def test_squad_details_file_shape():
    from fap.matchlab.reference import squad_from_table
    sq = squad_from_table(pd.DataFrame({"shirt_number": [24, 33], "player_app_id": ["a", "b"], "first_name": ["Hamdy", "Mohamed"],
                                        "middle_name": ["Yasser", None], "last_name": ["Moussa", "Kamel"],
                                        "short_name": ["H.Moussa", "M.Kalousha"], "role_code": ["DEF", "MID"],
                                        "status": ["active", "active"]}))
    assert sq.lookup("24") == ("a", "Hamdy Yasser Moussa")
    assert sq.label("33") == "#33 M.Kalousha" and sq.role["24"] == "DEF"


def _cp_match() -> pd.DataFrame:
    """Masar loses the ball twice: once presses in 2 s and regains in 4 s, once nothing happens."""
    r = lambda tag, team, pl, x, y, t, lab="": _row(tag, team, pl, x, y, t, lab)
    return pd.DataFrame([
        r("Half Start", None, None, None, None, 0.0),
        r("Shot", "Masar", 9, 12, 34, 1.0, "Off target"), r("Shot", "Masar", 9, 10, 30, 2.0, "Off target"),
        r("Shot", "Opp", 0, 95, 34, 3.0, "Off target"), r("Shot", "Opp", 0, 92, 30, 4.0, "Off target"),
        r("Pass", "Masar", 6, 60, 30, 10.0, "Unsuccessful"),
        r("Interception", "Opp", 0, 40, 35, 11.0, "Won"),
        r("Pass", "Opp", 0, 42, 35, 11.5, "Successful"),
        r("Press", "Masar", 8, 55, 30, 13.0),
        r("Recovery", "Masar", 8, 58, 31, 15.0),
        r("Pass", "Masar", 8, 58, 31, 16.0, "Successful"),
        r("Receive", "Masar", 9, 40, 30, 17.0, "Successful"),
        r("Pass", "Masar", 9, 40, 30, 20.0, "Unsuccessful"),
        r("Interception", "Opp", 0, 70, 30, 21.0, "Won"),
        r("Pass", "Opp", 0, 72, 30, 22.0, "Successful"),
        r("Receive", "Opp", 0, 85, 30, 24.0, "Successful"),
    ])


def test_counterpress_window():
    from fap.matchlab import counterpress as CP
    m = load_match(_cp_match(), "cp.xlsx", cp_window=8)
    d = CP.analyse(m, "Masar")
    assert len(d) == 2
    a, b = d.iloc[0], d.iloc[1]
    assert a["pressed"] and a["seconds_to_press"] == pytest.approx(2.0) and a["regained"]
    assert a["seconds_to_regain"] == pytest.approx(4.0) and a["category"] == "Pressed & regained"
    assert not b["pressed"] and not b["regained"] and b["category"] == "Not pressed, not regained"
    # a 3-second window drops the regain (it came 4 s after the loss)
    assert not CP.analyse(m, "Masar", 3)["regained"].iloc[0]
    s = CP.summary(d)
    assert s["losses"] == 2 and s["pressed"] == 1 and s["regained"] == 1


def test_sequence_table_lists_actions(md):
    from fap.matchlab.charts import sequence_table
    pid = int(md.possessions[md.possessions["team"] == "A"].sort_values("n_passes").iloc[-1]["poss_id"])
    t = sequence_table(md, pid)
    assert not t.empty and t["Action"].str.contains("Pass").any()


def test_panels_and_halves(proxy):
    from fap.matchlab import charts as CH
    fig = CH.render("pass_map_halves", proxy, "Masar", theme=CH.DARK)
    assert CH.panels_of(fig) == ["1st half", "2nd half"]
    png = CH.to_png(fig, dpi=40, panel="2nd half")
    assert png[:4] == b"\x89PNG"
    fig = CH.render("pass_profile", proxy, "Masar", theme=CH.LIGHT)
    assert "Direction rose" in CH.panels_of(fig)
    assert CH.to_png(fig, dpi=40, panel="Direction rose")[:4] == b"\x89PNG"


def test_bundled_reference_fills_ids_without_uploads(proxy):
    from fap.matchlab import export as X
    from fap.matchlab.reference import Reference
    ref = Reference.bundled()
    assert len(ref.squad.by_shirt) >= 31 and ref.squad.lookup("24")[0]
    assert ref.teams is not None and ref.matches is not None
    e = X.build_events(proxy)                       # no reference passed -> built-in tables
    m = e[e["team"] == "FC Masar Men"]
    assert m["team_id"].eq("25bd1277-b0e7-5e6c-b692-1716a60099f5").all()
    known = m["player_number"].isin(list(ref.squad.by_shirt))
    assert m.loc[known, "player_id"].ne("").all()
