"""Set Piece Lab page — opponent set-piece analysis from a coding-matrix file + a routine & drill designer.

Its own import (xlsx / csv from Sportscode / Hudl matrix exports), separate from Match Lab and from the
database-backed Set Piece Analysis page. Analysis = :mod:`fap.setpiecelab` (Plotly, hover, PNG camera +
HTML download). Designer = the Tactical Board editor (same canvas, frames = animation steps, GIF) loaded
with set-piece templates or a board built from the opponent's data; routines and drills are saved with
their coaching details as WorkspaceManager presets (kind ``setpiece_lab_library``).
"""
from __future__ import annotations

import base64
import hashlib
import io
import re

import pandas as pd
import streamlit as st

from fap.core.plugin import PluginInfo
from fap.identity.roles import Role
from fap.theme import components as C
from fap.ui.page import Page, page_registry

SPL_FILE = "_spl_file"              # {"name", "team", "data"}
SPL_FILE_KIND = "setpiece_lab_file"
SPL_LIB_KIND = "setpiece_lab_library"
SPL_LIB_SESSION = "_spl_library"    # session fallback when presets are unavailable


@st.cache_resource(show_spinner=False, max_entries=8)
def _load(sig: str, _data: bytes, name: str, team: str):
    from fap.setpiecelab import load_setpieces
    return load_setpieces(_data, name, team=team)


def _wm(shell):
    return getattr(shell, "wm", None)


def _show(fig, key: str, fname: str) -> None:
    cfg = {"displaylogo": False, "scrollZoom": True, "displayModeBar": True,
           "modeBarButtonsToRemove": ["lasso2d", "select2d"],
           "toImageButtonOptions": {"format": "png", "scale": 3, "filename": re.sub(r"[^A-Za-z0-9]+", "_", fname)}}
    st.plotly_chart(fig, use_container_width=True, key=f"spl_{key}", theme=None, config=cfg)
    st.download_button("Download chart (interactive HTML)", fig.to_html(include_plotlyjs="cdn", config=cfg).encode("utf-8"),
                       file_name=re.sub(r"[^A-Za-z0-9]+", "_", fname) + ".html", mime="text/html", key=f"spl_dl_{key}")


# ---------------------------------------------------------------- library persistence
def _lib_list(shell) -> list[dict]:
    wm = _wm(shell)
    if wm is not None:
        try:
            return [{"id": p.id, "name": getattr(p, "name", ""), "doc": getattr(p, "document", None) or {}}
                    for p in wm.list_presets(shell.user, kind=SPL_LIB_KIND)]
        except Exception:
            pass
    return list(st.session_state.get(SPL_LIB_SESSION, []))


def _lib_save(shell, name: str, doc: dict) -> str:
    wm = _wm(shell)
    if wm is not None:
        try:
            wm.save_preset(shell.user, kind=SPL_LIB_KIND, name=name, document=doc)
            return f"Saved '{name}' to the library."
        except Exception as exc:  # pragma: no cover
            return f"Save failed: {exc}"
    lib = st.session_state.setdefault(SPL_LIB_SESSION, [])
    lib.append({"id": hashlib.sha1(f"{name}{len(lib)}".encode()).hexdigest()[:10], "name": name, "doc": doc})
    return f"Saved '{name}' (this session only — saving is unavailable here)."


def _lib_delete(shell, item: dict) -> None:
    wm = _wm(shell)
    if wm is not None:
        try:
            wm.delete_preset(shell.user, item["id"])
            return
        except Exception:
            pass
    st.session_state[SPL_LIB_SESSION] = [x for x in st.session_state.get(SPL_LIB_SESSION, []) if x["id"] != item["id"]]


@page_registry.register
class SetPieceLabPage(Page):
    info = PluginInfo(id="set_piece_lab", name="Set Piece Lab", category="page")
    section = "Analysis"
    icon = "setpiece"
    order = 6

    def render(self, shell) -> None:
        can_edit = shell.user.role >= Role.PERFORMANCE_ANALYST
        st.markdown("### Set Piece Lab")
        st.caption("Opponent set-piece analysis from a coding-matrix file (Sportscode / Hudl), plus a designer "
                   "for your own routines and drills (steps + animation).")
        self._source(shell, can_edit)
        f = st.session_state.get(SPL_FILE)
        theme_name = st.radio("Chart theme", ["Dark", "Light"], horizontal=True, key="spl_theme")
        from fap.matchlab.charts import THEMES
        th = THEMES[theme_name]
        sd = None
        if f:
            sig = hashlib.sha1(f["data"]).hexdigest()[:16] + "|" + f.get("team", "")
            try:
                sd = _load(sig, f["data"], f["name"], f.get("team", ""))
            except Exception as exc:
                C.render_alert(f"Could not read this file: {exc}", "danger")
        tabs = st.tabs(["Overview", "Explorer", "Designer", "Library", "Data"])
        with tabs[0]:
            self._overview(sd, th) if sd else self._empty()
        with tabs[1]:
            self._explorer(sd, th) if sd else self._empty()
        with tabs[2]:
            self._designer(shell, sd, can_edit)
        with tabs[3]:
            self._library(shell, can_edit)
        with tabs[4]:
            self._data(sd) if sd else self._empty()

    @staticmethod
    def _empty() -> None:
        C.render_empty_state("Upload a set-piece file to start",
                             "Coding-matrix export (.xlsx / .csv) with a 'Row' column (set-piece type), qualifier "
                             "columns and x_/y_ landing coordinates.", icon_name="upload")

    # ---- import -----------------------------------------------------------------
    def _source(self, shell, can_edit: bool) -> None:
        c = st.columns([3, 1.5, 2])
        up = c[0].file_uploader("Set-piece file", type=["xlsx", "xls", "csv"], key="spl_upload")
        cur = st.session_state.get(SPL_FILE)
        if up is not None:
            data = up.getvalue()
            if not cur or cur.get("data") != data:
                st.session_state[SPL_FILE] = cur = {"name": up.name, "data": data, "team": ""}
        if cur:
            team = c[1].text_input("Team in the file", value=cur.get("team", ""), key="spl_team",
                                   help="Whose set pieces these are (blank = read from the file's Timeline)")
            if team != cur.get("team", ""):
                cur["team"] = team
        wm = _wm(shell)
        if wm is not None:
            try:
                saved = wm.list_presets(shell.user, kind=SPL_FILE_KIND)
            except Exception:
                saved = []
            if saved:
                names = {getattr(p, "name", "file"): p for p in saved}
                pick = c[2].selectbox("Saved files", ["—"] + list(names), key="spl_saved")
                if pick != "—" and c[2].button("Open", key="spl_open"):
                    doc = getattr(names[pick], "document", None) or {}
                    st.session_state[SPL_FILE] = {"name": doc.get("name") or pick, "team": doc.get("team", ""),
                                                  "data": base64.b64decode(doc.get("data_b64", ""))}
                    st.rerun()
            if cur and can_edit and c[2].button("Save file", key="spl_savefile"):
                try:
                    wm.save_preset(shell.user, kind=SPL_FILE_KIND, name=cur["name"].rsplit(".", 1)[0],
                                   document={"name": cur["name"], "team": cur.get("team", ""),
                                             "data_b64": base64.b64encode(cur["data"]).decode("ascii")})
                    st.toast("File saved.")
                except Exception as exc:
                    st.toast(f"Save failed: {exc}")

    # ---- overview -----------------------------------------------------------------
    def _overview(self, sd, th) -> None:
        from fap.setpiecelab import analytics as A
        from fap.setpiecelab import charts as CH
        ov = A.overview(sd)
        att, de = ov[ov["Phase"] == "Attacking"], ov[ov["Phase"] == "Defending"]
        g = lambda d, c: int(d[c].sum()) if c in d else 0  # noqa: E731
        C.render_metric_row([
            C.metric_card_html("Set pieces coded", str(int(ov["Count"].sum())), hint=f"{len(ov)} types"),
            C.metric_card_html(f"{sd.team} taking", str(g(att, "Count")),
                               hint=f"{g(att, 'Goals')} goals · {g(att, 'Dangerous (goal/shot/threat)')} dangerous"),
            C.metric_card_html(f"{sd.team} defending", str(g(de, "Count")),
                               hint=f"{g(de, 'Goals')} conceded · {g(de, 'Dangerous (goal/shot/threat)')} dangerous"),
            C.metric_card_html("With a landing point", str(g(ov, "With landing point")), hint="x / y coded"),
        ])
        if sd.notes:
            st.caption(" · ".join(sd.notes))
        st.markdown("#### Key findings")
        for row, text in A.insights(sd):
            st.markdown(f"- **{row}** — {text}" if row != "Headline" else f"- **{text}**")
        c1, c2 = st.columns(2)
        with c1:
            _show(CH.overview_bars(sd, theme=th), "ov_bars", f"{sd.team}_set_pieces")
        with c2:
            _show(CH.contest_rates(sd, theme=th), "ov_contest", f"{sd.team}_contests")
        st.dataframe(ov, use_container_width=True, hide_index=True)

    # ---- explorer -----------------------------------------------------------------
    def _explorer(self, sd, th) -> None:
        from fap.setpiecelab import analytics as A
        from fap.setpiecelab import charts as CH
        c = st.columns([2, 1.5])
        row = c[0].selectbox("Set piece", sd.rows, key="spl_row",
                             format_func=lambda r: f"{r}  ({len(sd.of(row=r))})")
        quals = [q for q in sd.qualifiers if (sd.of(row=row)[q] != "").any()]
        color_by = c[1].selectbox("Colour landing points by", ["—"] + quals, key="spl_colorby",
                                  index=(1 + quals.index("Outcome")) if "Outcome" in quals else 0)
        for r_, text in A.insights(sd):
            if r_ == row:
                st.info(text)
        _show(CH.landing_map(sd, row, color_by=None if color_by == "—" else color_by, theme=th), "map", f"{sd.team}_{row}_landing")
        _show(CH.qualifier_grid(sd, row, theme=th), "grid", f"{sd.team}_{row}_details")
        if len(quals) >= 2:
            cc = st.columns(2)
            a = cc[0].selectbox("Cross-tab: rows", quals, key="spl_ct_a")
            b = cc[1].selectbox("Cross-tab: columns", [q for q in quals if q != a], key="spl_ct_b")
            c1, c2 = st.columns(2)
            with c1:
                _show(CH.cross_heat(sd, row, a, b, theme=th), "ct", f"{sd.team}_{row}_{a}_x_{b}")
            with c2:
                if A._col(sd, "Outcome") and a != A._col(sd, "Outcome"):
                    _show(CH.outcome_by(sd, row, a, theme=th), "ob", f"{sd.team}_{row}_outcome_by_{a}")
        st.markdown("**Every instance**")
        cols = ["instance"] + quals + ["x", "y", "notes"]
        st.dataframe(sd.of(row=row)[[c_ for c_ in cols if c_ in sd.df.columns]], use_container_width=True, hide_index=True)
        if st.button("Open their pattern in the Designer", key="spl_to_designer"):
            from fap.setpiecelab import designer as D
            self._load_board(D.pattern_board(sd, row))
            st.success("Loaded — open the Designer tab.")

    # ---- designer -----------------------------------------------------------------
    @staticmethod
    def _load_board(board) -> None:
        from fap.ui.builtin import tactical_board as TBM
        TBM._load(board)

    def _designer(self, shell, sd, can_edit: bool) -> None:
        from fap.setpiecelab import designer as D
        from fap.ui.builtin import tactical_board as TBM
        st.caption("Pick a starting point, then edit on the board: drag players, draw runs and ball paths, add frames "
                   "(each frame = one step of the routine). Play the frames in the board's timeline; export a GIF below.")
        c = st.columns([1.4, 3, 1])
        src = c[0].radio("Start from", ["Template", "Opponent data", "Blank"], key="spl_src")
        if src == "Template":
            name = c[1].selectbox("Template", list(D.TEMPLATES), key="spl_tpl")
            if c[2].button("Load", key="spl_load_tpl", use_container_width=True, disabled=not can_edit):
                self._load_board(D.template(name))
                st.rerun()
        elif src == "Opponent data":
            if sd is None:
                c[1].caption("Upload a set-piece file first.")
            else:
                row = c[1].selectbox("Their set piece", [r for r in sd.rows if sd.of(row=r)["x"].notna().any()] or sd.rows,
                                     key="spl_pat_row")
                if c[2].button("Load", key="spl_load_pat", use_container_width=True, disabled=not can_edit):
                    self._load_board(D.pattern_board(sd, row))
                    st.rerun()
        else:
            if c[2].button("New", key="spl_new", use_container_width=True, disabled=not can_edit):
                from fap.tactical.models import new_board
                b = new_board("New set-piece routine")
                b.pitch.area = "att_half"
                self._load_board(b)
                st.rerun()

        # the Tactical Board editor itself (same canvas, frames, tools)
        page = TBM.TacticalBoardPage()
        page._inject_css()
        board, _hist = TBM._state()
        TBM._consume_canvas_intent(can_edit)
        page._board_view(shell, board, can_edit)
        board = st.session_state[TBM.TB_BOARD]
        st.caption(f"Board: **{board.name}** · {len(board.frames)} step(s): "
                   + " → ".join(f.name or f"Frame {i + 1}" for i, f in enumerate(board.frames)))

        # exports
        from fap.tactical.service import TacticalService
        svc = TacticalService(_wm(shell))
        colors = TBM._resolve_board_colors(shell, board)
        sig = hashlib.sha1(repr(board.to_dict()).encode("utf-8")).hexdigest()[:16]
        ec = st.columns(3)
        if ec[0].button("Prepare animation (GIF)", key="spl_gif", use_container_width=True):
            data, _, _ = svc.export(board, 0, fmt="gif", colors=colors)
            st.session_state["_spl_gif"] = (sig, data)
        if ec[1].button("Prepare every step (PNG, ZIP)", key="spl_pngs", use_container_width=True):
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for i, fr in enumerate(board.frames):
                    data, _, _ = svc.export(board, i, fmt="png", colors=colors)
                    z.writestr(f"{i + 1:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', fr.name or 'step')}.png", data)
            st.session_state["_spl_pngs"] = (sig, buf.getvalue())
        base = re.sub(r"[^A-Za-z0-9]+", "_", board.name) or "routine"
        got = st.session_state.get("_spl_gif")
        if got and got[0] == sig:
            ec[0].download_button("Download GIF", got[1], file_name=f"{base}.gif", mime="image/gif", key="spl_gif_dl",
                                  use_container_width=True)
        got = st.session_state.get("_spl_pngs")
        if got and got[0] == sig:
            ec[1].download_button("Download steps (ZIP)", got[1], file_name=f"{base}_steps.zip", mime="application/zip",
                                  key="spl_pngs_dl", use_container_width=True)

        # save to library with coaching details
        bk = board.id  # form fields reset when another board is loaded
        with st.expander("Save to the library (routine or drill, with coaching details)", expanded=False):
            kind = st.radio("Save as", ["Routine", "Drill"], horizontal=True, key=f"spl_kind_{bk}")
            meta = board.meta or {}
            if kind == "Routine":
                cc = st.columns(3)
                d = {"title": cc[0].text_input("Title", value=board.name, key=f"spl_r_title_{bk}"),
                     "category": cc[1].selectbox("Set piece", ["Corner", "Free kick", "Throw-in", "Goal kick", "Penalty", "Kick-off", "Other"],
                                                 index=["Corner", "Free kick", "Throw-in", "Goal kick", "Penalty", "Kick-off", "Other"].index(meta.get("category"))
                                                 if meta.get("category") in ["Corner", "Free kick", "Throw-in", "Goal kick", "Penalty", "Kick-off"] else 6,
                                                 key=f"spl_r_cat_{bk}"),
                     "phase": cc[2].selectbox("Phase", ["Attacking", "Defending"],
                                              index=1 if meta.get("phase") == "Defending" else 0, key=f"spl_r_phase_{bk}")}
                cc = st.columns(3)
                d["side"] = cc[0].selectbox("Side", ["Any", "Left", "Right", "Central"], key=f"spl_r_side_{bk}")
                d["target"] = cc[1].text_input("Target zone", value=str(meta.get("target", "")), key=f"spl_r_target_{bk}")
                d["code_word"] = cc[2].text_input("Code word / signal", key=f"spl_r_code_{bk}")
                cc = st.columns(2)
                d["taker"] = cc[0].text_input("Taker(s)", key=f"spl_r_taker_{bk}")
                d["key_roles"] = cc[1].text_input("Key roles (who does what)", key=f"spl_r_roles_{bk}")
                d["coaching_points"] = st.text_area("Coaching points", key=f"spl_r_cp_{bk}")
                d["trigger"] = st.text_input("When to use it (trigger / vs which defence)", key=f"spl_r_trig_{bk}")
                d["video_link"] = st.text_input("Video link", key=f"spl_r_video_{bk}")
                d["notes"] = st.text_area("Notes", key=f"spl_r_notes_{bk}")
                doc = D.routine_doc(board, d)
            else:
                cc = st.columns(3)
                d = {"title": cc[0].text_input("Title", value=board.name, key=f"spl_d_title_{bk}"),
                     "focus": cc[1].selectbox("Focus", ["Attacking corners", "Defending corners", "Free kicks", "Throw-ins",
                                                        "Second balls", "Goal kicks", "Penalties", "Other"], key=f"spl_d_focus_{bk}"),
                     "intensity": cc[2].selectbox("Intensity", ["Low", "Medium", "High"], index=1, key=f"spl_d_int_{bk}")}
                cc = st.columns(3)
                d["players"] = cc[0].text_input("Players", key=f"spl_d_players_{bk}")
                d["area"] = cc[1].text_input("Area size", key=f"spl_d_area_{bk}")
                d["duration_min"] = cc[2].text_input("Duration (min)", key=f"spl_d_dur_{bk}")
                d["objective"] = st.text_input("Objective", key=f"spl_d_obj_{bk}")
                d["setup"] = st.text_area("Setup", key=f"spl_d_setup_{bk}")
                d["instructions"] = st.text_area("Instructions / rules", key=f"spl_d_instr_{bk}")
                d["coaching_points"] = st.text_area("Coaching points", key=f"spl_d_cp_{bk}")
                d["progressions"] = st.text_area("Progressions / variations", key=f"spl_d_prog_{bk}")
                d["video_link"] = st.text_input("Video link", key=f"spl_d_video_{bk}")
                doc = D.drill_doc(board, d)
            if st.button(f"Save {kind.lower()}", key=f"spl_save_lib_{bk}", type="primary", disabled=not can_edit):
                st.toast(_lib_save(shell, d.get("title") or board.name, doc))

    # ---- library -----------------------------------------------------------------
    def _library(self, shell, can_edit: bool) -> None:
        from fap.setpiecelab import designer as D
        items = _lib_list(shell)
        if not items:
            st.info("No routines or drills saved yet — design one in the Designer tab and save it.")
            return
        c = st.columns([2, 2])
        kind = c[0].radio("Show", ["All", "Routines", "Drills"], horizontal=True, key="spl_lib_kind")
        q = c[1].text_input("Search", key="spl_lib_q")
        for it in items:
            doc = it["doc"]
            k = doc.get("kind", "routine")
            if kind == "Routines" and k != "routine" or kind == "Drills" and k != "drill":
                continue
            det = doc.get("details", {})
            text = " ".join(str(v) for v in det.values()).lower()
            if q and q.lower() not in text + it["name"].lower():
                continue
            with st.container(border=True):
                cc = st.columns([4, 1, 1])
                cc[0].markdown(f"**{it['name']}** · {'Drill' if k == 'drill' else 'Routine'}  \n"
                               f"<span style='opacity:.7'>{D.summary_text(doc)}</span>", unsafe_allow_html=True)
                if cc[1].button("Open", key=f"spl_lib_open_{it['id']}", use_container_width=True):
                    self._load_board(D.board_of(doc))
                    st.success("Loaded into the Designer tab.")
                if can_edit and cc[2].button("Delete", key=f"spl_lib_del_{it['id']}", use_container_width=True):
                    _lib_delete(shell, it)
                    st.rerun()
                with st.expander("Details", expanded=False):
                    for key, val in det.items():
                        if val:
                            st.markdown(f"**{key.replace('_', ' ').capitalize()}:** {val}")

    # ---- data -----------------------------------------------------------------
    def _data(self, sd) -> None:
        df = sd.df.drop(columns=["points"]).copy()
        df["points"] = sd.df["points"].map(lambda p: "; ".join(f"{x:g},{y:g}" for x, y in p))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download tidy table (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{sd.team}_set_pieces_tidy.csv", mime="text/csv", key="spl_csv")
