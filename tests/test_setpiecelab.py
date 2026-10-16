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
    assert len(D.TEMPLATES) >= 10
    for name in D.TEMPLATES:
        b = D.template(name)
        assert len(b.frames) >= 2
        ids = {o.id for o in b.frames[0].objects}
        assert ids & {o.id for o in b.frames[-1].objects}       # same objects across steps = animation
    pb = D.pattern_board(sd, "Offensive Corner")
    assert pb.frames and pb.frames[0].objects
    doc = D.routine_doc(pb, {"title": "Near post", "category": "Corner", "phase": "Attacking"})
    back = D.board_of(doc)
    assert back.name == pb.name and len(back.frames) == len(pb.frames)
    assert "Corner" in D.summary_text(doc)
    ddoc = D.drill_doc(pb, {"title": "2nd balls", "focus": "Second balls"})
    assert ddoc["kind"] == "drill"


def test_bad_file():
    with pytest.raises(ValueError):
        load_setpieces(pd.DataFrame({"a": [1]}), "x.csv")
