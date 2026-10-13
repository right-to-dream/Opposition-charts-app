"""Match Lab page — professional match analysis from a video-tagging event export.

Upload the tagging export (xlsx / csv: tag_name, labels, team, players, pos_x_meters, pos_y_meters on
a 105 x 68 m pitch) and every metric is derived from it by :mod:`fap.matchlab`: pass taxonomy
(progressive, direction, length, height, type, lines broken), entries, crosses, shots + xG,
possessions, phases of play (build-up / progression / final third; high press / mid block / low
block / counter-press), defending, momentum and player tables. Charts switch between a Dark and a
Light theme and download as 300 dpi PNGs. Matches can be saved as WorkspaceManager presets
(kind ``matchlab_match``) so they reopen without re-uploading.
"""
from __future__ import annotations

import base64
import hashlib
import io

import pandas as pd
import streamlit as st

from fap.core.plugin import PluginInfo
from fap.identity.roles import Role
from fap.theme import components as C
from fap.ui.page import Page, page_registry

ML_FILE = "_ml_file"            # {"name", "data" (bytes)}
ML_KIND = "matchlab_match"
ML_REF = "_ml_ref"              # {"teams": df|None, "matches": df|None, "squad": df|None}
ML_REF_KIND = "matchlab_reference"
HALVES = {"Full match": None, "1st half": 1, "2nd half": 2}


# ---------------------------------------------------------------- cached engine calls
@st.cache_resource(show_spinner=False, max_entries=8)
def _load(sig: str, _data: bytes, name: str, mirror_y: bool, t1: float, t2: float, cp: float = 8.0):
    from fap.matchlab import load_match
    return load_match(_data, name, mirror_y=mirror_y, thirds=(t1, t2), cp_window=cp)


@st.cache_data(show_spinner=False, max_entries=300)
def _png(sig: str, key: str, team: str, half, player, theme: str, opts: tuple, dpi: int, _md,
         panel: str | None = None) -> tuple[bytes, list]:
    """(PNG bytes, panel names). ``panel`` renders only that part of the chart."""
    from fap.matchlab import charts as CH
    fig = CH.render(key, _md, team, half=half, player=player, theme=CH.THEMES[theme], **dict(opts))
    names = CH.panels_of(fig)
    return CH.to_png(fig, dpi=dpi, panel=panel if panel in names else None), names


@st.cache_data(show_spinner=False, max_entries=16)
def _audit(sig: str, _md) -> pd.DataFrame:
    from fap.matchlab import audit
    return audit.run(_md)


# ---------------------------------------------------------------- reference data (IDs)
def _ref_state(shell) -> dict:
    """Reference tables live in the session; on first use they are restored from the last saved
    preset so the team / match / squad files are uploaded once."""
    if ML_REF not in st.session_state:
        st.session_state[ML_REF] = {"teams": None, "matches": None, "squad": None}
        wm = _wm(shell)
        if wm is not None:
            try:
                saved = wm.list_presets(shell.user, kind=ML_REF_KIND)
                if saved:
                    doc = getattr(saved[-1], "document", None) or {}
                    for k in ("teams", "matches", "squad"):
                        if doc.get(k):
                            st.session_state[ML_REF][k] = pd.read_json(io.StringIO(doc[k]), orient="split", dtype=False)
            except Exception:
                pass
    return st.session_state[ML_REF]


def _save_ref(shell, state: dict) -> str:
    wm = _wm(shell)
    if wm is None:
        return "Kept for this session (saving is unavailable here)."
    try:
        doc = {k: (v.to_json(orient="split") if isinstance(v, pd.DataFrame) else "") for k, v in state.items()}
        wm.save_preset(shell.user, kind=ML_REF_KIND, name="Match Lab reference data", document=doc)
        return "Reference data saved."
    except Exception as exc:  # pragma: no cover
        return f"Save failed: {exc}"


def _reference(state: dict):
    """Uploaded tables win; anything not uploaded falls back to the built-in club tables."""
    from fap.matchlab.reference import Reference, squad_from_table
    base = Reference.bundled()
    squad = base.squad
    if isinstance(state.get("squad"), pd.DataFrame):
        try:
            squad = squad_from_table(state["squad"], "uploaded squad file")
        except ValueError:
            pass
    teams = state["teams"] if isinstance(state.get("teams"), pd.DataFrame) else base.teams
    matches = state["matches"] if isinstance(state.get("matches"), pd.DataFrame) else base.matches
    return Reference(teams, matches, squad)


@st.cache_data(show_spinner=False, max_entries=8)
def _workbook(sig: str, ref_sig: str, match_id: str, _md, _ref) -> bytes:
    from fap.matchlab import export as X
    return X.build_workbook(_md, _ref, match_id)


def _ref_sig(state: dict) -> str:
    parts = []
    for k in ("teams", "matches", "squad"):
        v = state.get(k)
        parts.append(hashlib.sha1(pd.util.hash_pandas_object(v.astype(str), index=False).values.tobytes()).hexdigest()[:10]
                     if isinstance(v, pd.DataFrame) else "-")
    return "|".join(parts)


# ---------------------------------------------------------------- persistence
def _wm(shell):
    return getattr(shell, "wm", None)


def _save(shell, name: str, data: bytes) -> str:
    wm = _wm(shell)
    if wm is None:
        return "Saving is unavailable here."
    try:
        wm.save_preset(shell.user, kind=ML_KIND, name=name,
                       document={"name": name, "data_b64": base64.b64encode(data).decode("ascii")})
        return f"Saved '{name}'."
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        return f"Save failed: {exc}"


def _list(shell) -> list:
    wm = _wm(shell)
    if wm is None:
        return []
    try:
        return wm.list_presets(shell.user, kind=ML_KIND)
    except Exception:
        return []


def _fmt(v, d=0, pct=False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, str):
        return v
    return f"{v:.{d}f}{'%' if pct else ''}"


# ---------------------------------------------------------------- page
@page_registry.register
class MatchLabPage(Page):
    info = PluginInfo(id="match_lab", name="Match Lab", category="page")
    section = "Analysis"
    icon = "analysis"
    order = 4

    def render(self, shell) -> None:
        can_edit = shell.user.role >= Role.PERFORMANCE_ANALYST
        st.markdown("### Match Lab")
        st.caption("Professional match analysis from a tagged event file (105 × 68 m). Every metric — "
                   "progressive passes, pass types & directions, entries, phases of play, defending — is "
                   "derived from the file's x / y and labels.")
        self._source(shell, can_edit)
        f = st.session_state.get(ML_FILE)
        if not f:
            C.render_empty_state("Upload a match file to start",
                                 "Tagging export (.xlsx / .csv) with tag_name, labels, team, players, "
                                 "pos_x_meters, pos_y_meters.", icon_name="upload")
            return

        with st.expander("Settings", expanded=False):
            c = st.columns(5)
            t1 = c[0].slider("Build-up line (m)", 20.0, 45.0, 35.0, 1.0, key="ml_t1",
                             help="Phase boundary: build-up ends here")
            t2 = c[1].slider("Final-third line (m)", 55.0, 85.0, 70.0, 1.0, key="ml_t2")
            mirror = c[2].checkbox("Mirror flanks (left ↔ right)", key="ml_mirror",
                                   help="Use if the tagging tool writes y upward (flanks come out swapped)")
            dpi = c[3].select_slider("Download quality (dpi)", [150, 200, 300, 400], value=300, key="ml_dpi")
            cpw = c[4].slider("Counter-press window (s)", 6, 8, 8, 1, key="ml_cpw",
                              help="Seconds after a ball loss in which a press / a regain counts as counter-press")
        data = f["data"]
        from fap.matchlab import ENGINE_VERSION
        sig = f"{ENGINE_VERSION}|" + hashlib.sha1(data).hexdigest()[:16] + f"|{mirror}|{t1}|{t2}|{cpw}"
        try:
            with st.spinner("Analysing match…"):
                md = _load(sig, data, f["name"], mirror, t1, t2, float(cpw))
        except Exception as exc:
            C.render_alert(f"Could not read this file: {exc}", "danger")
            return

        c = st.columns([2, 2, 2, 1.4])
        if st.session_state.get("ml_team") not in md.teams:
            st.session_state["ml_team"] = md.our_team
        team = c[0].selectbox("Team", md.teams, key="ml_team")
        half_label = c[1].selectbox("Period", list(HALVES), key="ml_half")
        half = HALVES[half_label]
        players = sorted({p for p in md.team_events(team)["player"] if p and p != "0"}, key=lambda s: (len(s), s))
        ref = _reference(_ref_state(shell))
        names = ref.squad.short if team == md.our_team else {}
        pl_label = c[2].selectbox("Player", ["All players"] + [f"#{p} {names.get(p, '')}".strip() for p in players],
                                  key="ml_player")
        player = None if pl_label == "All players" else pl_label[1:].split(" ", 1)[0]
        theme = c[3].radio("Theme", ["Dark", "Light"], horizontal=True, key="ml_theme")
        if not players:
            st.caption(f"{team} has no player numbers in this file — team-level analysis only.")

        self._kpis(md, team, half)
        aud = _audit(sig, md)
        n = aud["Status"].value_counts().to_dict() if not aud.empty else {}
        st.session_state["_ml_sig"] = sig
        cc = st.columns([3, 1.2])
        cc[0].caption(f"Data check against the raw file: {n.get('PASS', 0)} passed · {n.get('WARN', 0)} warnings · "
                      f"{n.get('FAIL', 0)} failed — details in the 'Data & export' tab.")
        state = _ref_state(shell)
        ref_top = _reference(state)
        mid = st.session_state.get("_ml_match_id") or ref_top.auto_match_id(md)
        base_name = f["name"].rsplit(".", 1)[0]
        try:
            wb_top = _workbook(sig, _ref_sig(state), mid, md, ref_top)
            cc[1].download_button("Download enriched Excel", wb_top, file_name=f"{base_name}_enriched.xlsx",
                                  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  key="ml_dl_xlsx_top", type="primary", use_container_width=True)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            cc[1].caption(f"Excel export unavailable: {exc}")
        ctx = dict(sig=sig, md=md, team=team, half=half, player=player, theme=theme, dpi=int(dpi), names=names)

        tabs = st.tabs(["Overview", "Passing", "Progression & attack", "Phases of play", "Defending",
                        "Territory", "Players", "Data & export"])
        with tabs[0]:
            self._chart(ctx, "comparison")
            self._chart(ctx, "momentum")
        with tabs[1]:
            from fap.matchlab.charts import PASS_FILTERS
            cc = st.columns([2, 1, 1])
            kind = cc[0].selectbox("Pass filter", list(PASS_FILTERS), format_func=lambda k: PASS_FILTERS[k][0],
                                   key="ml_pass_kind")
            failed = cc[1].checkbox("Show unsuccessful", value=True, key="ml_pass_failed")
            op = cc[2].checkbox("Open play only", value=False, key="ml_pass_op")
            self._chart(ctx, "pass_map", kind=kind, show_failed=failed, open_play_only=op)
            self._chart(ctx, "pass_map_halves", kind=kind, show_failed=failed, open_play_only=op)
            self._chart(ctx, "pass_profile")
            self._chart(ctx, "pass_sonars")
            self._chart(ctx, "pass_flow")
            cc = st.columns([1, 1, 2])
            mn = cc[0].number_input("Min. passes per link", 1, 10, 3, key="ml_net_min")
            subs = cc[1].checkbox("Include substitutes", value=False, key="ml_net_subs")
            self._chart(ctx, "pass_network", team_level=True, min_passes=int(mn), include_subs=subs)
            self._chart(ctx, "passing_matrix", team_level=True)
            self._breakdown_tables(md, team, half, player)
        with tabs[2]:
            self._chart(ctx, "progression")
            op = st.checkbox("Entries: open play only", value=True, key="ml_entries_op")
            self._chart(ctx, "entries", open_play_only=op)
            self._chart(ctx, "crosses")
            self._chart(ctx, "shots")
        with tabs[3]:
            self._phase_tables(md, team, half)
            self._chart(ctx, "phases_in", team_level=True)
            self._chart(ctx, "phases_out", team_level=True)
            self._chart(ctx, "phase_flow", team_level=True)
            mp = st.slider("Build-up sequences: minimum passes", 1, 10, 3, key="ml_bu_min")
            self._chart(ctx, "buildup", team_level=True, min_passes=int(mp))
            self._sequence_explorer(ctx)
            self._chart(ctx, "sequences", team_level=True)
        with tabs[4]:
            self._counterpress(ctx)
            pr = st.checkbox("Include presses", value=False, key="ml_def_press")
            self._chart(ctx, "defensive", include_press=pr)
            self._chart(ctx, "losses")
        with tabs[5]:
            style = st.radio("Heat map style", ["smooth", "zones"], horizontal=True, key="ml_heat_style",
                             format_func=lambda s: {"smooth": "Smooth density", "zones": "Positional zones"}[s])
            self._chart(ctx, "heatmap", style=style)
        with tabs[6]:
            self._players(ctx)
        with tabs[7]:
            self._data_export(shell, can_edit, md, aud)
            self._method(md, team)

    # ---- source ------------------------------------------------------------
    def _source(self, shell, can_edit: bool) -> None:
        c = st.columns([3, 2])
        up = c[0].file_uploader("Match file", type=["xlsx", "xls", "csv"], key="ml_upload")
        if up is not None:
            data = up.getvalue()
            cur = st.session_state.get(ML_FILE)
            if not cur or cur.get("data") != data:
                st.session_state[ML_FILE] = {"name": up.name, "data": data}
        saved = _list(shell)
        if saved:
            names = {getattr(p, "name", "match"): p for p in saved}
            pick = c[1].selectbox("Saved matches", ["—"] + list(names), key="ml_saved")
            if pick != "—" and c[1].button("Open", key="ml_open"):
                doc = getattr(names[pick], "document", None) or {}
                try:
                    st.session_state[ML_FILE] = {"name": doc.get("name") or pick,
                                                 "data": base64.b64decode(doc.get("data_b64", ""))}
                    st.rerun()
                except Exception:
                    C.render_alert("This saved match could not be opened.", "danger")
        f = st.session_state.get(ML_FILE)
        if f and can_edit and _wm(shell) is not None:
            cc = st.columns([3, 1])
            nm = cc[0].text_input("Save as", value=f["name"].rsplit(".", 1)[0], key="ml_save_name",
                                  label_visibility="collapsed")
            if cc[1].button("Save match", key="ml_save", use_container_width=True):
                st.toast(_save(shell, nm, f["data"]))

    # ---- KPI strip ---------------------------------------------------------
    def _kpis(self, md, team, half) -> None:
        from fap.matchlab import metrics as M
        a = M.team_summary(md, team, half)
        cards = [
            C.metric_card_html("Goals · xG", f"{a['Goals']} · {a['xG']:.2f}", hint=f"{a['Shots']} shots, {a['On target']} on target"),
            C.metric_card_html("Possession", _fmt(a["Possession % (passes)"], 0, True), hint=f"Field tilt {_fmt(a['Field tilt %'], 0, True)}"),
            C.metric_card_html("Passing", _fmt(a["Pass accuracy %"], 0, True), hint=f"{a['Passes']} passes"),
            C.metric_card_html("Progressive", f"{a['Progressive passes']} + {a['Progressive carries']}", hint="passes + carries"),
            C.metric_card_html("Entries", f"{a['Final-third entries']} · {a['Box entries']}", hint="final third · box"),
            C.metric_card_html("PPDA", _fmt(a["PPDA"], 1), hint=f"Def. height {_fmt(a['Def. action height (m)'], 1)} m"),
        ]
        C.render_metric_row(cards)

    # ---- one chart -----------------------------------------------------------
    def _chart(self, ctx: dict, key: str, team_level: bool = False, **opts) -> None:
        from fap.matchlab.charts import CHART_BY_KEY
        spec = CHART_BY_KEY[key]
        player = None if team_level else ctx["player"]
        args = (ctx["sig"], key, ctx["team"], ctx["half"], player, ctx["theme"], tuple(sorted(opts.items())))
        try:
            png, panels = _png(*args, 110, ctx["md"])
        except Exception as exc:
            C.render_alert(f"{spec.label}: could not render ({exc})", "warning")
            return
        st.image(png, use_container_width=True)
        cc = st.columns([2, 1.3, 1.3])
        part = cc[0].selectbox("Download", ["Whole chart"] + panels, key=f"ml_part_{key}",
                               label_visibility="collapsed",
                               help="Download the whole chart, or only one part of it (pitch, table, bars ...)")
        panel = None if part == "Whole chart" else part
        base = f"{ctx['team']}_{key}{'_' + str(ctx['half']) + 'H' if ctx['half'] else ''}{'_' + player if player else ''}"
        fname = f"{base}{'_' + panel if panel else ''}_{ctx['theme'].lower()}.png".replace(" ", "_").replace("#", "")
        if cc[1].button(f"Prepare PNG ({ctx['dpi']} dpi)", key=f"ml_prep_{key}", use_container_width=True):
            st.session_state[f"_ml_hi_{key}"] = (args, panel, _png(*args, ctx["dpi"], ctx["md"], panel)[0])
        hi = st.session_state.get(f"_ml_hi_{key}")
        if hi and hi[0] == args and hi[1] == panel:
            cc[2].download_button("Download PNG", hi[2], file_name=fname, mime="image/png",
                                  key=f"ml_dl_{key}", use_container_width=True)
        st.divider()

    # ---- counter-press ------------------------------------------------------
    def _counterpress(self, ctx: dict) -> None:
        from fap.matchlab import counterpress as CP
        md, team, half, names = ctx["md"], ctx["team"], ctx["half"], ctx.get("names") or {}
        st.markdown(f"#### Counter-press — first {md.cp_window:.0f} s after losing the ball")
        st.caption("Change the window (6–8 s) in Settings. Loss = where the opponent won the ball; pressed = a Press "
                   "tag inside the window; regained = an open-play regain inside the window.")
        self._chart(ctx, "counterpress", team_level=True)
        d = CP.analyse(md, team, md.cp_window, half)
        if d.empty:
            return
        with st.expander(f"Every ball loss ({len(d)})", expanded=False):
            show = d.copy()
            show["player"] = show["player"].map(lambda p: f"#{p} {names.get(p, '')}".strip() if p and p != "0" else "")
            show["first_presser"] = show["first_presser"].map(lambda p: f"#{p} {names.get(p, '')}".strip() if p and p != "0" else "")
            show["minute"] = show["minute"].round(1)
            cols = ["loss_event_id", "half", "minute", "player", "loss_action", "third", "won_by_opponent_with",
                    "pressed", "seconds_to_press", "presses", "first_presser", "regained", "seconds_to_regain",
                    "regain_to_shot", "opp_reached_final_third", "opp_shot", "category"]
            st.dataframe(show[cols].round(2), use_container_width=True, hide_index=True)
            pr = d[d["pressed"] & d["first_presser"].ne("") & d["first_presser"].ne("0")]
            if len(pr):
                top = pr.groupby("first_presser").agg(presses=("pressed", "size"), regained=("regained", "sum"),
                                                     avg_seconds=("seconds_to_press", "mean"))
                top.index = [f"#{p} {names.get(p, '')}".strip() for p in top.index]
                st.markdown("**First presser after a loss**")
                st.dataframe(top.sort_values("presses", ascending=False).round(2), use_container_width=True)

    # ---- sequence explorer -------------------------------------------------
    def _sequence_explorer(self, ctx: dict) -> None:
        from fap.matchlab.charts import sequence_table
        md, team, half = ctx["md"], ctx["team"], ctx["half"]
        poss = md.possessions
        if poss.empty:
            return
        st.markdown("#### Sequence explorer — every action of one possession")
        c = st.columns([1.2, 1, 1, 3])
        start = c[0].selectbox("Started in", ["Anywhere", "Build-up", "Progression", "Final third"], key="ml_seq_start")
        mn = c[1].number_input("Min. passes", 0, 30, 3, key="ml_seq_min")
        only = c[2].selectbox("Ended", ["Any", "Shot", "Reached final third", "Lost"], key="ml_seq_end")
        p = poss[(poss["team"] == team) & (poss["n_passes"] >= mn)]
        if half is not None:
            p = p[p["half"] == half]
        t1, t2 = md.thirds
        if start == "Build-up":
            p = p[p["start_x"] < t1]
        elif start == "Progression":
            p = p[(p["start_x"] >= t1) & (p["start_x"] < t2)]
        elif start == "Final third":
            p = p[p["start_x"] >= t2]
        if only == "Shot":
            p = p[p["shot"]]
        elif only == "Reached final third":
            p = p[p["reached_f3"]]
        elif only == "Lost":
            p = p[p["end_reason"] == "Lost"]
        if p.empty:
            st.info("No possession matches these filters.")
            return
        opts = {f"#{r.poss_id} · {r.start_min:.0f}' · {r.n_passes} passes · {r.duration:.0f}s · {r.start_type} → {r.end_reason}":
                int(r.poss_id) for r in p.sort_values("start_min").itertuples()}
        pick = c[3].selectbox(f"Sequence ({len(opts)})", list(opts), key="ml_seq_pick")
        pid = opts[pick]
        names = tuple(sorted((ctx.get("names") or {}).items()))
        self._chart(ctx, "sequence", team_level=True, poss_id=pid, names=names)
        st.dataframe(sequence_table(md, pid, dict(names)), use_container_width=True, hide_index=True)

    # ---- tables --------------------------------------------------------------
    def _breakdown_tables(self, md, team, half, player) -> None:
        from fap.matchlab import metrics as M
        with st.expander("Pass breakdown tables", expanded=False):
            bd = M.pass_breakdown(md, team, half, player)
            cols = st.columns(3)
            names = {"pass_type": "Type", "direction": "Direction", "length_class": "Length", "height": "Height",
                     "line": "Line broken", "channel": "Channel (origin)", "third": "Third (origin)"}
            for i, (k, g) in enumerate(bd.items()):
                with cols[i % 3]:
                    st.markdown(f"**{names[k]}**")
                    st.dataframe(g.round(1), use_container_width=True)

    def _phase_tables(self, md, team, half) -> None:
        from fap.matchlab import phases as P
        st.markdown("**In possession** — phase = where the ball is")
        st.dataframe(P.in_possession_table(md, team, half).round(1), use_container_width=True, hide_index=True)
        st.markdown("**Out of possession** — phase = where the defensive action happens")
        st.dataframe(P.out_of_possession_table(md, team, half).round(1), use_container_width=True, hide_index=True)
        c = st.columns(2)
        with c[0]:
            st.markdown("**Build-up possessions**")
            bo = P.buildup_outcomes(md, team, half)
            if bo.get("count"):
                st.write({"Possessions": bo["count"], "Avg passes": round(bo["avg_passes"], 1),
                          "Avg duration (s)": round(bo["avg_duration"], 1), "Furthest third": bo["furthest"],
                          "How they ended": bo["ended"], "Lost in own third": bo["lost_in_own_third"],
                          "Shots": bo["shots"]})
        with c[1]:
            st.markdown("**Attacking transitions (after a regain)**")
            st.write(P.transitions(md, team, half))

    def _players(self, ctx: dict) -> None:
        from fap.matchlab import metrics as M
        md, team, half = ctx["md"], ctx["team"], ctx["half"]
        tab = M.player_table(md, team, half)
        if tab.empty:
            st.info("No player numbers for this team in the file.")
            return
        names = ctx.get("names") or {}
        if names:
            tab.insert(1, "Name", tab["Player"].str.lstrip("#").map(lambda p: names.get(p, "")))
        st.dataframe(tab, use_container_width=True, hide_index=True)
        st.download_button("Player table (CSV)", tab.to_csv(index=False).encode("utf-8"),
                           file_name=f"{team}_players.csv", mime="text/csv", key="ml_pl_csv")
        pick = st.selectbox("Player dashboard", list(tab["Player"]), key="ml_pl_dash")
        self._chart(dict(ctx, player=pick.lstrip("#")), "player")

    def _data_export(self, shell, can_edit: bool, md, aud: pd.DataFrame) -> None:
        from fap.matchlab import export as X
        st.markdown("#### Data check")
        st.caption("Every number below is re-counted straight from the raw file and compared with the engine.")
        if not aud.empty:
            st.dataframe(aud, use_container_width=True, hide_index=True,
                         column_config={"Detail": st.column_config.TextColumn(width="large")})

        st.markdown("#### Team, match & player IDs")
        state = _ref_state(shell)
        c = st.columns(3)
        uploads = (("teams", "Teams file (team_id, team)"), ("matches", "Matches file (match_id, home_team, away_team)"),
                   ("squad", "Squad file (player_id, player_name, shirt number)"))
        changed = False
        for col, (k, label) in zip(c, uploads):
            up = col.file_uploader(label, type=["xlsx", "xls", "csv"], key=f"ml_ref_{k}")
            if up is not None:
                sigk = f"{up.name}:{up.size}"
                if st.session_state.get(f"_ml_ref_sig_{k}") != sigk:
                    from fap.matchlab.loader import read_table
                    try:
                        state[k] = read_table(up.getvalue(), up.name)
                        st.session_state[f"_ml_ref_sig_{k}"] = sigk
                        changed = True
                    except Exception as exc:
                        col.error(f"Could not read: {exc}")
            have = state.get(k)
            col.caption(f"Uploaded: {len(have)} rows" if isinstance(have, pd.DataFrame)
                        else "Using the built-in club file")
        if changed and can_edit:
            st.toast(_save_ref(shell, state))
        ref = _reference(state)
        res = ref.resolve(md)
        rows = [{"Team in file": t, "Official name": r.official or "—", "team_id": r.team_id or "—", "Matched by": r.method}
                for t, r in res["teams"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        cand = res["matches"]
        match_id = ""
        if isinstance(cand, pd.DataFrame) and not cand.empty:
            opts = {f"{r['match']}  ·  GW {r.get('gameweek', '')}  ·  {r['match_id']}": r["match_id"] for _, r in cand.iterrows()}
            match_id = opts[st.selectbox("Match", list(opts), key="ml_match_pick")]
        elif isinstance(state.get("matches"), pd.DataFrame):
            st.caption("No row in the matches file has these two teams — match_id will be empty.")
        missing = X.unmatched_shirts(md, ref)
        if missing:
            st.warning(f"No player_id for {md.our_team} shirt(s): " + ", ".join(f"#{m}" for m in missing)
                       + f" — {ref.squad.source}. Upload the squad file to fill them."
                       + (f" Ambiguous shirts (shared by 2 players): {', '.join(ref.squad.ambiguous)}." if ref.squad.ambiguous else ""))

        st.markdown("#### Download the enriched file")
        st.caption("The club's 44 columns + all derived columns (end points, progressive, direction, phases, "
                   "possession, xG ...) + one column per label, with team / match / player IDs. "
                   "Extra sheets: carries, possessions, players, team summary, data check.")
        base = (st.session_state.get(ML_FILE) or {}).get("name", "match").rsplit(".", 1)[0]
        st.session_state["_ml_match_id"] = match_id
        with st.spinner("Preparing the Excel file..."):
            wb = _workbook(st.session_state.get("_ml_sig", ""), _ref_sig(state), match_id, md, ref)
        st.download_button("Download Excel (.xlsx)", wb, file_name=f"{base}_enriched.xlsx", type="primary",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="ml_dl_xlsx")
        ev_csv = X.build_events(md, ref, match_id)
        st.download_button("Download events only (.csv)", ev_csv.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{base}_events.csv", mime="text/csv", key="ml_dl_evcsv")
        with st.expander("Preview (first 200 events)", expanded=False):
            st.dataframe(ev_csv.head(200), use_container_width=True, hide_index=True)

    def _method(self, md, team) -> None:
        st.markdown("#### Method")
        st.markdown(
            """
**Coordinates** — every team is normalised to attack left → right on a 105 × 68 m pitch
(`y = 0` = the team's right touchline, drawn at the bottom). Attack direction is detected per team
and half from the data (shots → pass direction → opposite of the opponent).

**Pass end points** — the file has no target coordinates, so a pass ends at the next event: the
team-mate's *Receive* (completed), the opponent's interception / recovery / tackle / clearance /
save (lost), or the restart / ball-out location. A player's *Receive* that is time-stamped just
after his own pass is re-ordered first (tagging pre-roll).

**Carries** — derived: a player's *Receive* followed by his own next action ≥ 5 m away.

**Progressive (Wyscout)** — ≥ 30 m closer to the goal centre (both points in own half), ≥ 15 m
(crossing halfway), ≥ 10 m (both in opponent half).

**Direction** — forward |θ| ≤ 45°, backward |θ| ≥ 135°, else left / right. **Length** — short
< 15 m, medium 15–30 m, long ≥ 30 m. **Entries** — final third from < 70 m to ≥ 70 m; box =
ends inside the penalty area. **Channels** — split at the box / six-yard-box edges.

**Phases** — in possession by ball x: build-up / progression / final third (lines adjustable in
*Settings*); transition = within 10 s of a regain. Out of possession by defensive-action x: low
block / mid block / high press; counter-press = within 5 s of losing the ball or tagged *Counter
Press*. **PPDA** = opponent passes in their own 60 % ÷ our tackles, interceptions (not *Clear*),
duels and fouls in our attacking 60 %.

**Possession** = a sequence of one team's control ending on an opponent's control, a dead ball or a
shot. **xG** — FAP internal model when available, else a transparent distance/angle model.
"""
        )
        if md.notes:
            st.caption(" · ".join(dict.fromkeys(md.notes)))
        st.markdown("**Detected attack direction (raw file)**")
        st.write({f"{t} – half {h}": d for (t, h), d in md.orientation.items()})
