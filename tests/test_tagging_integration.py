"""Tagging Studio integration: exported tags feed the FAP visualization engine,
the canvas trust boundary is safe, and the page is a first-class Analysis tab.
"""
import os
os.environ["FAP_TEST"] = "1"
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from fap.core.types import RenderContext
from fap.tagging import coordinates as TC
from fap.tagging.export import session_to_canonical_frame
from fap.tagging.models import TagEvent, TaggingSession
from fap.themes import ThemeManager
from fap.visuals import Renderer, visual_registry
from fap.visuals.base import load_builtin_visuals
import fap.visuals.setpieces.library  # noqa: F401 - registers penalty viz

load_builtin_visuals()
THEME = ThemeManager("assets/themes").get("opta_light")


def _mixed_session() -> TaggingSession:
    s = TaggingSession(match_id="M1")
    s.add_event(TagEvent(event_type="pass", coordinate_space="pitch",
                         x=30, y=45, x2=62, y2=41, team="Team A", player="P8",
                         outcome="Successful", minute=12, second=34))
    s.add_event(TagEvent(event_type="shot", coordinate_space="pitch",
                         x=88, y=52, team="Team A", player="P9", outcome="Saved"))
    s.add_event(TagEvent(event_type="shot_on_target", coordinate_space="goal",
                         goal_x=62, goal_y=48, player="P9", outcome="Goal"))
    s.add_event(TagEvent(event_type="save", coordinate_space="goal",
                         goal_x=20, goal_y=30, player="GK", outcome="Saved"))
    return s


def _render(viz_id, df):
    viz = visual_registry.create(viz_id)
    fig = Renderer().render(viz, RenderContext(df=df, theme=THEME,
                                               controls={"title": viz_id}, meta={}))
    plt.close(fig)
    return fig


def test_canonical_frame_has_stable_viz_columns():
    df = session_to_canonical_frame(_mixed_session())
    for col in ("event_type", "x", "y", "end_x", "end_y", "goal_x", "goal_y",
                "gx", "gy", "shot_result"):
        assert col in df.columns
    # goal events are emitted as shots so the shot/goal-mouth maps consume them
    goal_rows = df[df["coordinate_space"] == "goal"]
    assert (goal_rows["event_type"] == "shot").all()
    assert goal_rows["end_y"].between(44, 56).all()          # mapped across the goal
    assert goal_rows["gx"].between(0, 2).all() and goal_rows["gy"].between(0, 2).all()


@pytest.mark.parametrize("viz_id", ["goal_mouth_map", "save_zones", "pass_map",
                                    "sp_pen_placement", "sp_gk_reach"])
def test_tagged_data_renders_in_maps(viz_id):
    df = session_to_canonical_frame(_mixed_session())
    fig = _render(viz_id, df)
    assert fig.axes and len(fig.axes[0].get_children()) > 0


def test_exported_csv_imports_through_pipeline_and_goalmouth_is_not_empty():
    """The reported bug: tag goal-mouth -> export CSV -> load in Open Play -> the
    Goal Mouth Map must render the shots (not empty). Replays the whole path:
    CSV text -> pandas -> DataPipeline (Data Hub ingestion) -> GoalMouthMap."""
    import io
    import pandas as pd
    from fap.pipeline.pipeline import DataPipeline
    from fap.providers.base import RawDataset
    from fap.tagging.export import session_to_csv
    from fap.visuals import analysis as A

    s = TaggingSession(match_id="M1")
    for gx, gy, out in [(62, 48, "Goal"), (20, 30, "Saved"), (80, 60, "Saved"),
                        (48, 20, "Missed")]:
        s.add_event(TagEvent(event_type="shot_on_target", coordinate_space="goal",
                             goal_x=gx, goal_y=gy, player="P9", outcome=out))
    csv_text = session_to_csv(s)                             # what the analyst downloads

    raw = pd.read_csv(io.StringIO(csv_text))                 # what the Data Hub reads
    frame = DataPipeline().run(RawDataset(frame=raw))        # canonical event frame
    # the pipeline sees them as shots with an across-goal end_y in the goal-mouth band
    shots = A.shots(frame)
    assert len(shots) == 4
    assert shots["end_y"].dropna().between(38, 62).all()

    fig = _render("goal_mouth_map", frame)
    ax = fig.axes[0]
    from matplotlib.patches import Circle
    markers = [p for p in ax.patches if isinstance(p, Circle)]
    assert len(markers) >= 4, "Goal Mouth Map rendered empty for tagged goal data"
    plt.close(fig)


def test_pitch_and_goal_tags_coexist_when_switching_layers():
    s = _mixed_session()
    pitch = [e for e in s.events if e.coordinate_space == "pitch"]
    goal = [e for e in s.events if e.coordinate_space == "goal"]
    assert len(pitch) == 2 and len(goal) == 2       # both retained regardless of active layer
    # undo/redo does not mix or lose either space
    s.undo(); s.redo()
    assert len(s) == 4


# ------------------------------------------------------------------ trust boundary
def test_parse_result_validates_and_clamps():
    from fap.ui.builtin.tagging_canvas import parse_result
    assert parse_result(None) is None
    assert parse_result({"ts": 1, "action": "bogus"}) is None
    assert parse_result({"action": "point", "ifx": 0.5, "ify": 0.5}) is None   # no ts
    ok = parse_result({"ts": 3, "action": "point", "ifx": 1.4, "ify": -0.2})
    assert ok == {"ts": 3.0, "action": "point", "ifx": 1.0, "ify": 0.0}        # clamped
    line = parse_result({"ts": 4, "action": "line", "ifx": 0.1, "ify": 0.2,
                         "ifx2": 0.3, "ify2": 0.4})
    assert line["action"] == "line" and line["ifx2"] == 0.3
    assert parse_result({"ts": 5, "action": "select", "select": "e_1"})["select"] == "e_1"
    assert parse_result({"ts": 6, "action": "delete"})["action"] == "delete"


def test_canvas_click_round_trips_to_the_same_event_marker():
    """A click fraction -> canonical -> marker fraction returns to the click."""
    fx, fy = 0.62, 0.30
    x, y = TC.canonical_from_pitch_fraction(fx, fy)
    ifx, ify = TC.pitch_fraction_from_canonical(x, y)
    assert ifx == pytest.approx(fx, abs=1e-3) and ify == pytest.approx(fy, abs=1e-3)
    gx, gy = TC.canonical_from_goal_fraction(fx, fy)
    gifx, gify = TC.goal_fraction_from_canonical(gx, gy)
    assert gifx == pytest.approx(fx, abs=1e-3) and gify == pytest.approx(fy, abs=1e-3)


# ------------------------------------------------------------------ one-click Data Hub bridge
def _settings(tmp_path):
    from dataclasses import replace
    from fap.config.settings import (AppSettings, CacheSettings, DatabaseSettings,
                                     StorageSettings)
    return replace(AppSettings(environment="development"),
                   user_data_dir=str(tmp_path / "ud"),
                   database=DatabaseSettings(path=str(tmp_path / "ud" / "fap.sqlite3")),
                   cache=CacheSettings(backend="memory"), storage=StorageSettings(backend="local"))


def _user():
    from fap.identity.models import User
    from fap.identity.roles import Role
    return User(email="analyst@club.com", name="A", role=Role.SUPER_ADMIN, provider_id="dev")


def test_send_to_datahub_activates_a_renderable_open_play_dataset(tmp_path):
    """The one-click bridge: tag goal-mouth -> send_to_datahub -> it becomes the ACTIVE
    dataset and the active frame renders on the Goal Mouth Map (no manual CSV step)."""
    from fap.bootstrap import init_platform
    platform = init_platform(settings=_settings(tmp_path))
    user = _user()
    ws = platform.workspace_manager.ensure_workspace(user)
    s = TaggingSession(match_id="Tagged vs Rivals")
    for gx, gy, out in [(62, 48, "Goal"), (20, 30, "Saved"), (80, 60, "Saved")]:
        s.add_event(TagEvent(event_type="shot_on_target", coordinate_space="goal",
                             goal_x=gx, goal_y=gy, player="P9", outcome=out))
    try:
        ds = platform.tagging.send_to_datahub(user, s, name="Tagged vs Rivals",
                                              workspace_id=ws.id)
        assert ds is not None
        # it is now the active dataset every module reads
        assert platform.workspace_manager.active_dataset_id(user) == ds.id
        frame = platform.workspace_manager.active_frame(user)
        assert frame is not None and (frame["event_type"] == "shot").sum() == 3
        fig = _render("goal_mouth_map", frame)
        from matplotlib.patches import Circle
        assert len([p for p in fig.axes[0].patches if isinstance(p, Circle)]) >= 3
        plt.close(fig)
    finally:
        try:
            platform.db.close()
        except Exception:
            pass


# ------------------------------------------------------------------ page registration
def test_tagging_page_is_a_first_class_analysis_tab():
    from fap.ui.page import load_builtin_pages, page_registry
    load_builtin_pages()
    page = next((cls() for cls in page_registry if cls().info.id == "tagging"), None)
    assert page is not None
    assert page.section == "Analysis"
    assert page.info.name == "Tagging"


def test_interior_bbox_maps_corners_exactly():
    """A click on a pitch corner must resolve to the exact canonical corner. The pitch
    renderer pins aspect='equal', which letterboxes the data inside the figure; the
    bbox is derived from the real axes transform so the mapping stays exact through
    that margin (a naive xlim/ylim ratio put a corner click at x≈95.7 instead of 100)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fap.tagging.coordinates import canonical_from_pitch_fraction
    from fap.themes import ThemeManager
    from fap.ui.builtin.tagging import TaggingStudioPage
    from fap.visuals.pitch import DISPLAY_WIDTH, PitchFactory, get_spec

    theme = ThemeManager("assets/themes").get("opta_light")
    fig = plt.figure(figsize=(11.5, 7.4))
    ax = fig.add_axes([0, 0, 1, 1])
    PitchFactory().draw_pitch(ax, theme, get_spec("uefa"), vertical=False)
    ax.set_xlim(-4, 104); ax.set_ylim(-4, DISPLAY_WIDTH + 4); ax.axis("off")
    fig.canvas.draw()
    bbox = TaggingStudioPage._interior_bbox(fig, ax, 0.0, 100.0, 0.0, DISPLAY_WIDTH)
    assert 0.0 <= bbox["left"] < bbox["right"] <= 1.0
    assert 0.0 <= bbox["top"] < bbox["bottom"] <= 1.0

    tf = fig.transFigure.inverted()

    def img_frac(x, y):                                  # data -> saved-image fraction
        fx, fy = tf.transform(ax.transData.transform((x, y)))
        return fx, 1.0 - fy

    # each visual pitch corner/goal-line point must round-trip to its canonical value
    for dx, dy, cx, cy in [(100.0, 0.0, 100.0, 0.0), (0.0, 0.0, 0.0, 0.0),
                           (100.0, DISPLAY_WIDTH, 100.0, 100.0)]:
        ifx_img, ify_img = img_frac(dx, dy)
        ifx = (ifx_img - bbox["left"]) / (bbox["right"] - bbox["left"])
        ify = (ify_img - bbox["top"]) / (bbox["bottom"] - bbox["top"])
        gx, gy = canonical_from_pitch_fraction(ifx, ify)
        assert gx == pytest.approx(cx, abs=0.4)
        assert gy == pytest.approx(cy, abs=0.4)
    plt.close(fig)


def test_board_batch_is_idempotent_and_clamped():
    from fap.ui.builtin.tagging_canvas import parse_result
    b = parse_result({"kind": "batch", "token": "t", "seq": 3,
                      "events": [{"cid": "t-1", "action": "point", "ifx": 0.5, "ify": 1.7},
                                 {"cid": "t-2", "action": "line", "ifx": 0, "ify": 0, "ifx2": 1, "ify2": 1},
                                 {"cid": "t-3", "action": "point", "ifx": "x"},          # malformed → dropped
                                 {"action": "point", "ifx": 0.1, "ify": 0.1}],           # no id → dropped
                      "ops": [{"aid": "t-op4", "action": "undo"}, {"aid": "t-op5", "action": "rm -rf"}]})
    assert b["kind"] == "batch" and [e["cid"] for e in b["events"]] == ["t-1", "t-2"]
    assert b["events"][0]["ify"] == 1.0 and [o["action"] for o in b["ops"]] == ["undo"]
    assert parse_result({"kind": "batch", "events": []}) is None                        # no seq


def test_metres_follow_the_session_pitch_size():
    from fap.tagging import coordinates as TC
    from fap.tagging.export import CSV_COLUMNS, METRE_COLUMNS, session_to_csv, session_to_rows
    s = TaggingSession(pitch_length=100.0, pitch_width=64.0)
    s.add_event(TagEvent(event_type="pass", coordinate_space="pitch", x=50, y=25, x2=100, y2=50))
    s.add_event(TagEvent(event_type="shot_on_target", coordinate_space="goal", goal_x=50, goal_y=50))
    assert TC.to_metres(50, 105) == 52.5 and TC.from_metres(34, 68) == 50.0
    rows = session_to_rows(s, metres=True)
    assert (rows[0]["x_m"], rows[0]["y_m"], rows[0]["end_x_m"], rows[0]["end_y_m"]) == (50.0, 16.0, 100.0, 32.0)
    assert (rows[1]["goal_x_m"], rows[1]["goal_y_m"]) == (3.66, 1.22)
    assert session_to_csv(s).splitlines()[0] == ",".join(CSV_COLUMNS)                  # Open Play schema unchanged
    assert session_to_csv(s, metres=True).splitlines()[0] == ",".join(CSV_COLUMNS + METRE_COLUMNS)
    assert TaggingSession.from_dict(s.to_dict()).pitch_length == 100.0
