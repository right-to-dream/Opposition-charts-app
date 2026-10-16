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
