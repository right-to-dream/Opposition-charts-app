"""Set Piece Lab: matrix loader, analytics, charts and the routine/drill designer."""
from __future__ import annotations

import pandas as pd
import pytest

from fap.setpiecelab import load_setpieces
from fap.setpiecelab import analytics as A
from fap.setpiecelab import charts as CH
from fap.setpiecelab import designer as D


def _matrix() -> pd.DataFrame:
    return pd.DataFrame({
        "Timeline": ["Baladya Set Pieces"] * 6,
        "Start time": [10, 50, 90, 130, 170, 210],
        "Duration": [10] * 6,
        "Row": ["Offensive Corner", "Offensive Corner", "Offensive Corner", "Long GK", "Defensive Corner", "Def Throw in"],
        "Original Instance": ["Offensive Corner: 1", "Offensive Corner: 2", "Offensive Corner: 3", "Long GK: 1",
                              "Defensive Corner: 1", "Def Throw in: 1"],
        "1st contact": ["Yes", "No", "No", "Yes", "No", None],
        "2nd ball": ["Yes", None, "Yes", "No", None, None],
        "Outcome": ["Goal", "Cleareance", "Shot", None, "Threat", None],
        "Execution": ["Inswing", "Inswing", "Outswing", None, "Inswing", None],
        "x_Offensive Corner": ["100", "99", "101, 80, 80", None, None, None],
        "y_Offensive Corner": ["30", "36", "34, 40, 42", None, None, None],
        "x_GK": [None, None, None, "60", None, None],
        "y_GK": [None, None, None, "20", None, None],
        "x_Defensive Corner": [None, None, None, None, "8", None],
        "y_Defensive Corner": [None, None, None, None, "33", None],
    })


@pytest.fixture()
def sd():
    return load_setpieces(_matrix(), "m.xlsx")


def test_loader_shape(sd):
    assert sd.team == "Baladya"
    assert len(sd.df) == 6
    assert set(sd.df["phase"]) == {"Attacking", "Defending"}
    assert sd.df.loc[sd.df["row"] == "Def Throw in", "phase"].iloc[0] == "Defending"
    assert "1st contact" in sd.binary and "Outcome" not in sd.binary
    # coordinate matching: own type, suffix (gk ⊂ long gk), multi-value cell
    gk = sd.of(row="Long GK").iloc[0]
    assert (gk["x"], gk["y"]) == (60.0, 20.0)
    third = sd.of(row="Offensive Corner").iloc[2]
    assert (third["x"], third["y"], third["extra_points"]) == (101.0, 34.0, 2)
    assert pd.isna(sd.of(row="Def Throw in").iloc[0]["x"])
    assert sd.of(row="Offensive Corner").iloc[1]["Outcome"] == "Clearance"
    assert any("Def Throw in" in n for n in sd.notes)


def test_analytics_counts(sd):
    ov = A.overview(sd).set_index("Set piece")
    oc = ov.loc["Offensive Corner"]
    assert oc["Count"] == 3 and oc["Goals"] == 1 and oc["Shots"] == 1
    assert oc["Dangerous (goal/shot/threat)"] == 2
    assert (oc["1st contact won"], oc["1st contact decided"]) == (1, 3)
    assert (oc["2nd ball won"], oc["2nd ball decided"]) == (2, 2)
    heads = [t for r, t in A.insights(sd) if r == "Headline"]
    assert any("Offensive Corner" in t for t in heads)
    assert not A.crosstab(sd, "Offensive Corner", "Execution", "Outcome").empty


def test_charts_build(sd):
    for fig in (CH.overview_bars(sd), CH.contest_rates(sd), CH.landing_map(sd, "Offensive Corner", color_by="Outcome"),
                CH.landing_map(sd, "Def Throw in"), CH.qualifier_grid(sd, "Offensive Corner"),
                CH.cross_heat(sd, "Offensive Corner", "Execution", "Outcome"),
                CH.outcome_by(sd, "Offensive Corner", "Execution")):
        assert fig.to_dict()["layout"]


def test_templates_and_library(sd):
    assert len(D.TEMPLATES) >= 12
    for name in D.TEMPLATES:
        d = D.template(name)
        assert D.clean_doc(d) is not None and len(d["steps"]) >= 2, name
        moved = any(d["steps"][0]["pos"].get(k) != d["steps"][-1]["pos"].get(k) for k in d["steps"][0]["pos"])
        assert moved, name                                   # something actually moves = animation
    pd_ = D.pattern_doc(sd, "Offensive Corner")
    marks = [o for o in pd_["objects"] if o["kind"] == "mark"]
    assert len(marks) == 3 and pd_["steps"][0]["zones"]
    lib = D.routine_doc(pd_, {"title": "Near post", "category": "Corner", "phase": "Attacking"})
    back = D.board_of(lib)
    assert back["name"] == pd_["name"] and len(back["steps"]) == len(pd_["steps"])
    assert "Corner" in D.summary_text(lib)
    assert D.drill_doc(pd_, {"title": "2nd balls", "focus": "Second balls"})["kind"] == "drill"


def test_advance_and_animation_follow_the_line():
    d = D.new_doc()
    d["objects"].append({"id": "p", "kind": "player", "team": "home", "gk": False, "num": "9", "label": "", "color": ""})
    s0 = d["steps"][0]
    s0["pos"]["p"] = [90.0, 30.0]
    s0["lines"].append({"id": "l", "kind": "run", "a": [90.0, 30.0], "b": [100.0, 30.0], "c": [95.0, 40.0], "obj": "p"})
    d["steps"].append(D.advance(s0, "Run"))
    assert d["steps"][1]["pos"]["p"] == [100.0, 30.0] and d["steps"][1]["lines"] == []
    x, y, a = D.frame_positions(d, 0, 0.5)["p"]
    assert a == 1.0 and y > 34                               # follows the curve, not the straight line


def test_clean_doc_rejects_garbage():
    assert D.clean_doc(None) is None and D.clean_doc({"steps": []}) is None
    doc = D.clean_doc({"steps": [{"pos": {"ghost": [1, 2]}, "lines": [{"kind": "rm -rf", "a": [0, 0], "b": [1, 1]}]}],
                       "objects": [{"id": "x", "kind": "<script>"}], "view": "evil", "colors": {"home": "red;"}})
    assert doc["objects"] == [] and doc["steps"][0]["pos"] == {} and doc["steps"][0]["lines"] == []
    assert doc["view"] == "half" and doc["colors"]["home"].startswith("#")


def test_exports_render(sd):
    from fap.setpiecelab import render as R
    from fap.setpiecelab import static as S
    d = D.template("Penalty · taker and rebounds")
    for style in ("grass", "dark", "light"):
        d["style"] = style
        assert R.step_png(d, 0, dpi=40)[1:4] == b"PNG"
    assert R.sheet_png(d, {"coaching_points": "Strike low."}, dpi=40)[1:4] == b"PNG"
    assert R.gif_bytes(d, fps=4, width_px=300)[:3] == b"GIF"
    for fig in (S.overview(sd), S.contests(sd), S.landing(sd, "Offensive Corner", color_by="Outcome"),
                S.landing(sd, "Defensive Corner"), S.qualifiers(sd, "Offensive Corner"),
                S.cross(sd, "Offensive Corner", "Execution", "Outcome"), S.outcome_by(sd, "Offensive Corner", "Execution"),
                S.report_card(sd, "Offensive Corner")):
        assert S.png(fig, 30)[1:4] == b"PNG"


def test_bad_file():
    with pytest.raises(ValueError):
        load_setpieces(pd.DataFrame({"a": [1]}), "x.csv")


def test_split_outcome_columns_side_coords_and_second_ball():
    """Pharco-style export: '1st contact Outcome' / '2nd contact Outcome', corner coords split by side,
    a second-ball location and 'X-GK' (hyphen) coordinates."""
    from fap.setpiecelab.loader import RESULT_COL
    raw = pd.DataFrame({
        "Timeline": ["Pharco Set pieces"] * 5,
        "Row": ["Offensive Corner", "Offensive Corner", "Defensive Corner", "Long GK", "Penalty"],
        "1st contact": ["No", "No", "Yes", "No", None],
        "1st contact Outcome": ["GK Punch", "Cleareance", "Cleareance", None, None],
        "2nd ball": [None, "yes", "Yes", "Yes", None],
        "2nd contact Outcome": [None, "Threat, Shot", "Cleareance", None, None],
        "Efficiency": [None, None, None, None, "Successful"],
        "X-GK": [None, None, None, 60, None], "Y-GK": [None, None, None, 46, None],
        "X_Second ball": [96, None, None, None, None], "Y_Second ball": [30, None, None, None, None],
        "X_offensive corner Right": [103, None, None, None, None], "Y_offensive corner Right": [40, None, None, None, None],
        "X_offensive corner left": [None, 99, None, None, None], "Y_offensive corner left": [None, 38, None, None, None],
    })
    sd = load_setpieces(raw, "p.csv")
    df = sd.df
    assert sd.team == "Pharco" and sd.qualifiers[0] == RESULT_COL and "X-GK" not in sd.qualifiers
    assert list(df[RESULT_COL]) == ["GK Punch", "Shot", "Clearance", "", "Goal"]     # most dangerous wins
    assert (df.at[0, "x"], df.at[0, "landing_side"], df.at[0, "sb_x"]) == (103.0, "Right", 96.0)
    assert (df.at[1, "x"], df.at[1, "landing_side"]) == (99.0, "Left")
    assert (df.at[3, "x"], df.at[3, "y"]) == (60.0, 46.0)
    assert df.at[1, "2nd ball"] == "Yes"
    assert A.categories(sd) == ["Corner", "Goal kick", "Penalty"]
    t = A.category_table(sd, "Corner").set_index("Phase")
    assert t.loc["Attacking", "Dangerous (goal/shot/threat)"] == 1 and t.loc["Defending", "1st contact won"] == 1
    assert any("Second balls" in s for s in A.category_insights(sd, "Corner"))
    from fap.setpiecelab import static as S
    assert S.png(S.category_compare(sd, "Corner"), 30)[1:4] == b"PNG"
    assert S.png(S.landing(sd, "Offensive Corner", color_by=RESULT_COL), 30)[1:4] == b"PNG"
    CH.category_compare(sd, "Corner")


def test_contact_maps():
    from fap.setpiecelab import static as S
    raw = pd.DataFrame({"Timeline": ["P set pieces"] * 2, "Row": ["Offensive Corner"] * 2,
                        "1st contact": ["No", "Yes"], "1st contact Outcome": ["Clearance", "Threat"],
                        "2nd ball": ["Yes", None], "2nd contact Outcome": ["Shot", None],
                        "X_offensive corner Right": [100, 99], "Y_offensive corner Right": [30, 40],
                        "X_Second ball": [90, None], "Y_Second ball": [34, None]})
    sd = load_setpieces(raw, "p.csv")
    assert A.contact_columns(sd, 1) == ("1st contact Outcome", "1st contact")
    assert len(A.contact_points(sd, "Offensive Corner", 1)) == 2 and len(A.contact_points(sd, "Offensive Corner", 2)) == 1
    for w in (1, 2):
        CH.contact_map(sd, "Offensive Corner", w)
        assert S.png(S.contact_map(sd, "Offensive Corner", w), 30)[1:4] == b"PNG"


def test_v2_variants_sides_takers_goal_and_typo_coords():
    from fap.setpiecelab import static as S
    raw = pd.DataFrame({
        "Timeline": ["Pharco Set pieces"] * 5,
        "Row": ["Long Throw In", "Throw In", "Direct Offensive Free kick", "Offensive Free kick", "Penalty"],
        "1st contact": ["Yes", None, None, "No", None], "1st contact Outcome": ["Threat", None, None, "Out of bounds", None],
        "Efficiency": [None, None, "Goal", None, "Successful"],
        "Location": ["Left", "Right", None, "Right", None],
        "X_Long Throw IN": [98.4, None, None, None, None], "Y_Long Ttrow In": [37.3, None, None, None, None],
        "Goal across ": [None, None, None, None, 6.57], "Goal Height": [None, None, None, None, 2.26],
        "Taker": [24, 24, 7, 22, 24],
    })
    sd = load_setpieces(raw, "p.csv")
    df = sd.df
    assert "Taker" not in sd.qualifiers and "Goal across" not in sd.qualifiers
    assert list(df["variant"]) == ["Long", "Short", "Direct", "Indirect", ""]
    assert list(df["side"][:2]) == ["Left", "Right"] and list(df["taker"]) == ["24", "24", "7", "22", "24"]
    assert (df.at[0, "x"], df.at[0, "y"]) == (98.4, 37.3)                       # misspelt Y column paired
    assert df.at[2, "Result"] == "Goal" and df.at[4, "Result"] == "Goal" and (df.at[4, "goal_x_m"], df.at[4, "goal_y_m"]) == (6.57, 2.26)
    left = sd.subset(side="Left")
    assert list(left.df["row"]) == ["Long Throw In"]
    t = A.takers(sd).set_index("Taker")
    assert t.loc["#24", "Taken"] == 3 and t.loc["#7", "Goals"] == 1
    assert any("Left" in s for s in A.side_insights(sd, "Long Throw In"))
    CH.takers_chart(sd, sd.df, "t"); CH.goal_placement(sd, "Penalty")
    CH.compare_groups(sd, [("A", df[:2]), ("B", df[2:])], "t")
    for fig in (S.takers_chart(sd, sd.df, "t"), S.goal_placement(sd, "Penalty"),
                S.compare_groups(sd, [("A", df[:2]), ("B", df[2:])], "t", lines=["**x** y"])):
        assert S.png(fig, 30)[1:4] == b"PNG"
