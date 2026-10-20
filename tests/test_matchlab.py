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
    assert first["pass_line_break"] == 1 and first["pass_under_pressure"] == 1
    assert (first["start_x"], first["end_x"]) == (90, 60)            # file orientation kept
    assert first["receiver_name"] == "Abdelrahman"
    assert e[e["team"] == "Proxy SC"]["player_id"].eq("").all()        # opponent: no player ids
    assert {"label_line_break", "label_under_pressure", "label_clear"} <= set(e.columns)
    assert e.loc[e["event_id"] == 2, "labels"].iloc[0] == "Line Break, Under Press"   # raw text kept
    assert e["pass_cross"].dtype.kind == "i" and set(e["pass_cross"]) <= {0, 1}
    assert not [c for c in e.columns if e[c].dtype == bool]
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
    # a 1-second window: the press (2 s) comes after the opponent completed a pass -> not even late
    tiny = CP.analyse(m, "Masar", 1).iloc[0]
    assert not tiny["pressed"] and not tiny["regained"]
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


def _late_match(extra_rows) -> pd.DataFrame:
    r = lambda tag, team, pl, x, y, t, lab="": _row(tag, team, pl, x, y, t, lab)
    base = [r("Half Start", None, None, None, None, 0.0),
            r("Shot", "Masar", 9, 12, 34, 1.0, "Off target"), r("Shot", "Masar", 9, 10, 30, 2.0, "Off target"),
            r("Shot", "Opp", 0, 95, 34, 3.0, "Off target"), r("Shot", "Opp", 0, 92, 30, 4.0, "Off target"),
            r("Pass", "Masar", 6, 60, 30, 10.0, "Unsuccessful"),
            r("Interception", "Opp", 0, 40, 35, 11.0, "Won")]
    return pd.DataFrame(base + [r(*x) for x in extra_rows])


def test_late_counterpress_counts_before_opponent_settles():
    from fap.matchlab import counterpress as CP
    # press 11.3 s after the loss, nothing in between -> late counter-press, failed
    m = load_match(_late_match([("Press", "Masar", 7, 50, 30, 22.3), ("Pass", "Opp", 0, 45, 30, 30.0, "Successful")]), "l.xlsx")
    d = CP.analyse(m, "Masar", 8).iloc[0]
    assert d["pressed"] and d["press_timing"] == "late" and d["category"] == "Pressed, not regained"
    assert d["seconds_to_press"] == pytest.approx(11.3)
    # same press, but the opponent completed a pass first -> settled possession, not a counter-press
    m2 = load_match(_late_match([("Pass", "Opp", 0, 42, 35, 15.0, "Successful"), ("Receive", "Opp", 0, 50, 35, 16.0, "Successful"),
                                 ("Press", "Masar", 7, 50, 30, 22.3)]), "l2.xlsx")
    assert not CP.analyse(m2, "Masar", 8).iloc[0]["pressed"]


def test_under_pressure_label_and_duel_count_as_pressure():
    from fap.matchlab import counterpress as CP
    m = load_match(_late_match([("Pass", "Opp", 0, 42, 35, 14.0, "Unsuccessful, Under Press"),
                                ("Recovery", "Masar", 8, 58, 31, 16.0), ("Pass", "Masar", 8, 58, 31, 17.0, "Successful")]), "u.xlsx")
    d = CP.analyse(m, "Masar", 8).iloc[0]
    assert d["pressed"] and d["press_evidence"] == "Opponent under pressure" and d["regained"]


def test_dead_ball_after_loss_is_not_an_open_play_loss():
    from fap.matchlab import counterpress as CP
    r = lambda tag, team, pl, x, y, t, lab="": _row(tag, team, pl, x, y, t, lab)
    df = pd.DataFrame([r("Half Start", None, None, None, None, 0.0),
                       r("Shot", "Masar", 9, 12, 34, 1.0, "Off target"), r("Shot", "Masar", 9, 10, 30, 2.0, "Off target"),
                       r("Shot", "Opp", 0, 95, 34, 3.0, "Off target"), r("Shot", "Opp", 0, 92, 30, 4.0, "Off target"),
                       r("Pass", "Masar", 6, 60, 30, 10.0, "Unsuccessful"), r("Ball Out", "Masar", 6, 60, -1, 11.0),
                       r("Press", "Masar", 7, 50, 30, 13.0), r("Pass", "Opp", 0, 45, 0, 30.0, "Throw In, Successful")])
    assert CP.analyse(load_match(df, "d.xlsx"), "Masar", 8).empty


def test_interactive_hover_figures(md, proxy):
    from fap.matchlab import interactive as IX
    from fap.matchlab.charts import DARK
    f = IX.build("pass_map", md, "A", theme=DARK, names={"4": "Kamel"}, kind="all")
    hovers = [t for tr in f.data if tr.text is not None for t in tr.text]
    assert any("#4 Kamel" in h and "→ #10" in h and "file pos" in h for h in hovers)
    g = IX.build("counterpress", proxy, "Masar", theme=DARK)
    assert g is not None and len(g.data) >= 0
    assert IX.build("comparison", md, "A") is None          # static-only charts are untouched


def test_all_interactive_charts_build(md, proxy):
    from fap.matchlab import interactive as IX
    from fap.matchlab.charts import DARK, LIGHT
    for key in IX.INTERACTIVE:
        for m, team in ((md, "A"), (proxy, "Masar"), (proxy, "Proxy")):
            out = IX.build(key, m, team, theme=LIGHT if team == "Proxy" else DARK, names={})
            figs = out if isinstance(out, list) else [out]
            assert figs and all(f is not None for f in figs), key
    assert IX.build("pass_map_halves", proxy, "Masar", theme=DARK) is not None


def test_fallback_xg_penalty(proxy):
    pen = proxy.events[(proxy.events["etype"] == "shot") & proxy.events["set_piece_kind"].eq("penalty")]
    assert len(pen) == 1 and pen["xg"].iloc[0] > 0.5


def test_player_profile_and_dashboard(proxy):
    from fap.matchlab import interactive as IX
    from fap.matchlab import player as PP
    from fap.matchlab.charts import DARK
    pr = PP.profile(proxy, "Masar", "33")
    assert pr["passing"]["Passes"] == 2 and pr["passing"]["Completed"] == 2   # 2nd = conflicting labels, resolved
    assert pr["by_direction"]["attempted"].sum() == 2
    assert pr["shooting"]["Shots"] == 0
    c = pr["comparison"]
    assert not c.empty and {"Rank", "Of", "Team average"} <= set(c.columns)
    for fn in (IX.player_rose, IX.player_breakdown, IX.player_compare):
        assert fn(proxy, "Masar", "33", theme=DARK) is not None
    assert IX.player_receptions(proxy, "Masar", "3", theme=DARK) is not None
    assert IX.player_defence(proxy, "Masar", "33", theme=DARK) is not None


def test_player_stats_leaders_and_compare(proxy):
    from fap.matchlab import interactive as IX
    from fap.matchlab import player as PP
    from fap.matchlab.charts import DARK
    st_ = PP.player_stats(proxy, "Masar")
    ev = proxy.events[(proxy.events["team"] == "Masar") & proxy.events["player"].ne("0") & proxy.events["player"].ne("")]
    assert int(st_["Passes"].sum()) == int((ev["etype"] == "pass").sum())
    for grp, specs in PP.LEADERBOARDS.items():
        for title, col, num, den in specs:
            assert col in st_.columns
            assert IX.leaderboard(st_, title, col, num, den, theme=DARK) is not None
    c = PP.compare_players(proxy, "Masar", ["33", "3"])
    assert {"33", "3", "Team best"} <= set(c.columns)
    assert IX.compare_radar(proxy, "Masar", ["33", "3"], theme=DARK).data


def test_team_deep_dive(md, proxy):
    from fap.matchlab import interactive_team as XT
    from fap.matchlab import team as T
    from fap.matchlab.charts import DARK
    for m in (md, proxy):
        side = T.stats_side_by_side(m)
        assert {"Category", "Metric"} <= set(side.columns) and len(side) > 40
        for cat in side["Category"].unique():
            assert XT.stats_category(side, cat, m.teams, theme=DARK) is not None
        team = m.teams[0]
        for fn in (XT.buildup_exits, XT.buildup_flow, XT.gk_distribution, XT.f3_entry_routes, XT.entry_breakdown,
                   XT.progressive_origins, XT.time_to_final_third, XT.lines_broken, XT.f3_channels, XT.chance_creation,
                   XT.shot_sources, XT.crosses_by_flank):
            assert fn(m, team, theme=DARK) is not None
    # fouls: 'Foul Won' is not a foul committed
    ev = proxy.events
    st_ = T.team_stats(proxy, "Masar").set_index("Metric")
    assert st_.at["Fouls committed", "Value"] == int((ev[(ev["team"] == "Masar")]["etype"] == "foul").sum())
    side = T.team_stats(proxy, "Masar").set_index("Metric")
    assert side.at["Passes", "Attempts"] == int(((ev["team"] == "Masar") & (ev["etype"] == "pass")).sum())


def test_switch_definition_side_to_side():
    ev = pd.DataFrame({"x": [50.0] * 4, "y": [10.0, 30.0, 10.0, 60.0], "end_x": [60.0] * 4,
                       "end_y": [58.0, 66.0, 38.0, 20.0]})
    from fap.matchlab.metrics import CHANNEL_EDGES, SWITCH_MIN_M
    lo, hi = CHANNEL_EDGES[2], CHANNEL_EDGES[3]
    side = lambda v: -1 if v < lo else 1 if v > hi else 0  # noqa: E731
    want = [abs(b - a) >= SWITCH_MIN_M and side(a) * side(b) == -1 for a, b in zip(ev["y"], ev["end_y"])]
    assert want == [True, False, False, True]      # from the centre / ending central are not switches


def test_heat_kinds_and_new_charts(md):
    from fap.matchlab import charts as CH
    from fap.matchlab import heat as HT
    from fap.matchlab import interactive as IX
    team = md.teams[0]
    for k in HT.KINDS:
        pts = HT.points(md, team, k)
        assert pts["x"].notna().all()
    assert set(HT.points(md, team, "losses").index) <= set(md.events.index)
    for key, kw in (("switches", {}), ("heat_sheet", {}), ("heat_compare", {"kind": "regains"}),
                    ("heatmap", {"kind": "press", "style": "zones"})):
        CH.to_png(CH.render(key, md, team, **kw), dpi=30)
    assert len(IX.build("heat_compare", md, team, kind="passes")) == 2
    IX.build("switches", md, team)
    sw = md.events[md.events["switch"]]
    assert (sw["etype"] == "pass").all() and ((sw["end_y"] - sw["y"]).abs() >= 30).all()


def test_loss_drawn_where_the_ball_was_lost(proxy):
    from fap.matchlab.loader import at_loss, loss_mask
    ev = proxy.events
    lo = at_loss(ev[loss_mask(ev)])
    assert lo["loss_at"].ne("").all()
    lp = lo[(lo["etype"] == "pass") & lo["end_x"].notna()]
    assert (lp["x"] == lp["end_x"]).all() and (lp["from_x"] == ev.loc[lp.index, "x"]).all()


def test_second_balls(proxy):
    from fap.matchlab import interactive_team as XT
    from fap.matchlab import secondball as SB
    c = SB.contests(proxy)
    assert set(c["kind"]) <= set(SB.KINDS)
    won = c[c["result"] == "Won"]
    assert (won["seconds"] <= SB.WINDOW).all() and won["winner"].ne("").all()
    a, b = proxy.teams[:2]
    sa, sb = SB.summary(proxy, a), SB.summary(proxy, b)
    assert sa["decided"] == sb["decided"] and sa["won"] + sb["won"] == sa["decided"]   # one winner per contest
    for f in (XT.second_balls_map, XT.second_balls_breakdown, XT.second_balls_players):
        f(proxy, a)


def test_multimatch_combine_table_and_outputs(md, proxy):
    from fap.matchlab import charts as CH
    from fap.matchlab import multimatch as MM
    items = [MM.MatchItem("1. A", md, "a"), MM.MatchItem("2. B", proxy, "b")]
    cm = MM.combine(items)
    assert cm.teams[1] == MM.OPP and len(cm.events) == len(md.events) + len(proxy.events)
    assert cm.events.groupby("match")["t"].min().is_monotonic_increasing          # matches never overlap in time
    pid = cm.possessions["poss_id"]
    assert pid.is_unique
    tb = MM.comparison_table(items, cm)
    assert list(tb.columns) == ["1. A", "2. B", "Match average", "All matches"]
    assert tb.loc["Passes", "All matches"] == tb.loc["Passes", "1. A"] + tb.loc["Passes", "2. B"]
    png = CH.to_png(CH.render("pass_map", cm, cm.teams[0]), dpi=30)
    grid = MM.grid_png([("1. A", png), ("2. B", png)], "t", "s")
    pdf = MM.pdf_report("t", "s", ["1. A", "2. B"], [("grid", grid),
                        ("trend", MM.fig_png(MM.trend_figure(tb, ["Passes"], "t"), 40))])
    assert grid[1:4] == b"PNG" and pdf[:4] == b"%PDF"


def test_player_comparison_minutes_per90_percentiles(md, proxy):
    from fap.matchlab import multimatch as MM
    from fap.matchlab import playercharts as PCH
    from fap.matchlab import playercompare as PC
    mins = PC.minutes_played(proxy, proxy.teams[0])
    assert (mins["minutes"] >= 0).all() and mins["started"].dtype == bool
    items = [MM.MatchItem("1. A", md, "a"), MM.MatchItem("2. B", proxy, "b")]
    pool = PC.player_pool(items)
    assert len(pool) and (pool["Matches"] >= 1).all()
    r = pool[pool["Minutes"] > 0].iloc[0]
    assert abs(r["Passes p90"] - r["Passes"] / r["Minutes"] * 90) < 1e-6          # per 90 is exact
    pp = PC.add_percentiles(pool, by="team", min_minutes=0)
    p = pp["pct:Passes"].dropna()
    assert p.between(0, 100).all() and p.max() == 100
    worst = pp.loc[pp["Ball losses p90"].idxmax(), "pct:Ball losses"]
    assert worst == pp["pct:Ball losses"].min()                                    # lower-is-better inverted
    keys = list(pp["player"][:2])
    for fig in (PCH.percentile_profile(pp, keys[0], pool_label="team"),
                PCH.compare_radar(pp, keys, ["Passes", "Pass accuracy %", "Ball losses"], pool_label="team"),
                PCH.compare_bars(pp, keys, ["Passes", "Ball losses"], pool_label="team"),
                PCH.scatter(pp, "Passes", "Defensive actions", min_minutes=0),
                PCH.ranking(pp, "Passes", min_minutes=0),
                PCH.vs_position(pp, keys[0], PC.position_averages(pp, 0))):
        assert MM.fig_png(fig, 30)[1:4] == b"PNG"


def test_match_scorecard_and_radar(md, proxy):
    from fap.matchlab import multimatch as MM
    items = [MM.MatchItem("1. A", md, "a"), MM.MatchItem("2. B", proxy, "b")]
    tb = MM.comparison_table(items, MM.combine(items))
    assert "Match average" in tb.columns and "xG against" in tb.index
    assert MM.metric_group("PPDA") == "Pressing" and MM.metric_group("xG against") == "Defending"
    assert MM.fig_png(MM.scorecard_figure(tb, "t"), 30)[1:4] == b"PNG"
    assert MM.fig_png(MM.match_radar(tb, ["Passes", "xG", "Shots", "PPDA"], "t"), 30)[1:4] == b"PNG"


def test_ppda_falls_back_to_possessions_and_regains_are_defensive(md):
    from fap.matchlab import metrics as M
    team = md.teams[0]
    pr = M.pressing(md, team)
    assert pr["PPDA method"] in ("passes", "possessions", "")          # "" = no defensive action in the zone
    if pr["PPDA method"] == "passes":
        assert pr["PPDA"] == pr["PPDA (passes)"]
    elif pr["PPDA method"] == "possessions":
        assert pr["PPDA"] == pr["PPDA (possessions)"]
    reg = M.defensive_regains(md, team)
    assert reg["etype"].isin(["recovery", "interception", "tackle"]).all()
    assert pr["High regains"] == int((reg["x"] >= 52.5).sum())
    # remove the opponent's passes → PPDA is computed from their possessions instead
    import copy
    md2 = copy.copy(md)
    md2.__dict__ = dict(md.__dict__)
    md2.__dict__.pop("_memo", None)
    opp = md.opponent(team)
    md2.events = md.events[~((md.events["team"] == opp) & (md.events["etype"] == "pass"))]
    pr2 = M.pressing(md2, team)
    assert pr2["PPDA method"] in ("possessions", "") and pr2["PPDA (passes)"] != pr2["PPDA (passes)"]


def test_error_ball_lost_with_failed_dribble_counts_once():
    from fap.matchlab.loader import loss_mask
    ev = pd.DataFrame({"etype": ["dribble", "error", "error", "pass"], "outcome": [0, None, None, 0],
                       "labels": [[], ["ball lost"], ["ball lost"], []], "team": ["A"] * 4,
                       "player": ["7", "7", "9", "9"], "half": [1] * 4, "t": [10.0, 11.0, 40.0, 60.0]})
    assert loss_mask(ev).tolist() == [True, False, True, True]        # #7's error = the same loss as his dribble


def test_player_defensive_actions_exclude_presses_and_fouls(proxy):
    from fap.matchlab import player as PL
    from fap.matchlab.metrics import PLAYER_DEF_ACTIONS
    team = proxy.our_team
    ps = PL.player_stats(proxy, team).set_index("player")
    ev = proxy.events[(proxy.events["team"] == team)]
    for pl in ps.index[:5]:
        g = ev[ev["player"] == pl]
        assert ps.at[pl, "Defensive actions"] == int(g["etype"].isin(PLAYER_DEF_ACTIONS).sum())
    assert "press" not in PLAYER_DEF_ACTIONS and "foul" not in PLAYER_DEF_ACTIONS
    assert (ps["Tackles successful"] >= ps["Tackles won"]).all() and "Dribbled past" in ps


def test_team_comparison_buildup_progression_box_and_insights(md, proxy):
    from fap.matchlab import insights as INS
    from fap.matchlab import multimatch as MM
    from fap.matchlab import playercompare as PC
    items = [MM.MatchItem("1. A", md, "a"), MM.MatchItem("2. B", proxy, "b"), MM.MatchItem("3. B", proxy, "c")]
    tb = MM.comparison_table(items, MM.combine(items))
    for m in ("Build-up exit success %", "Final-third entries (pass + carry)", "Box entry success %",
              "Box entries by carry", "Touches in box"):
        assert m in tb.index
    assert MM.metric_group("Box entry success %") == "Box entries"
    assert all("PPDA" not in m.label for m in PC.METRICS) and all(m.help for m in PC.METRICS)
    one = INS.match_insights(proxy, proxy.our_team)
    assert one and all(i.tone in ("strength", "concern", "note") and i.detail for i in one)
    many = INS.multi_insights(tb, pool=PC.player_pool(items), min_minutes=0)
    assert any(i.section == "Summary" for i in many)
