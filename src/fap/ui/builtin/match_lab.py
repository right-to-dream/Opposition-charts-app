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
import re
import uuid

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

        tabs = st.tabs(["Overview", "Team analysis", "Passing", "Progression & attack", "Phases of play", "Defending",
                        "Heat maps", "Players", "Data & export"])
        with tabs[0]:
            self._chart(ctx, "comparison")
            self._chart(ctx, "momentum")
        with tabs[1]:
            self._team_analysis(ctx)
        with tabs[2]:
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
        with tabs[3]:
            self._chart(ctx, "progression")
            self._chart(ctx, "switches")
            op = st.checkbox("Entries: open play only", value=True, key="ml_entries_op")
            self._chart(ctx, "entries", open_play_only=op)
            self._chart(ctx, "crosses")
            self._chart(ctx, "shots")
        with tabs[4]:
            self._phase_tables(md, team, half)
            self._chart(ctx, "phases_in", team_level=True)
            self._chart(ctx, "phases_out", team_level=True)
            self._chart(ctx, "phase_flow", team_level=True)
            mp = st.slider("Build-up sequences: minimum passes", 1, 10, 3, key="ml_bu_min")
            self._chart(ctx, "buildup", team_level=True, min_passes=int(mp))
            self._sequence_explorer(ctx)
            self._chart(ctx, "sequences", team_level=True)
        with tabs[5]:
            self._counterpress(ctx)
            pr = st.checkbox("Include presses", value=False, key="ml_def_press")
            self._chart(ctx, "defensive", include_press=pr)
            self._chart(ctx, "losses")
        with tabs[6]:
            self._heat_maps(ctx)
        with tabs[7]:
            self._players(ctx)
        with tabs[8]:
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
        uid = opts.pop("_uid", key)               # widget-key namespace when one chart appears several times
        player = None if team_level else ctx["player"]
        args = (ctx["sig"], key, ctx["team"], ctx["half"], player, ctx["theme"], tuple(sorted(opts.items())))
        try:
            png, panels = _png(*args, 110, ctx["md"])
        except Exception as exc:
            C.render_alert(f"{spec.label}: could not render ({exc})", "warning")
            return
        shown = False
        from fap.matchlab import interactive as IX
        if key in IX.INTERACTIVE:
            try:
                from fap.matchlab.charts import THEMES
                ix_kw = {"names": ctx.get("names") or {}, **opts}        # a chart's own `names` wins
                fig = IX.build(key, ctx["md"], ctx["team"], half=ctx["half"], player=player,
                               theme=THEMES[ctx["theme"]], **ix_kw)
                cfg = {"displaylogo": False, "scrollZoom": True,
                       "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"]}
                if isinstance(fig, list):
                    for i, (col, f) in enumerate(zip(st.columns(len(fig)), fig)):
                        f.update_layout(height=470)
                        col.plotly_chart(f, use_container_width=True, key=f"ml_ix_{uid}_{i}", theme=None, config=cfg)
                else:
                    st.plotly_chart(fig, use_container_width=True, key=f"ml_ix_{uid}", theme=None, config=cfg)
                st.caption("Interactive: hover any point for details · drag to zoom, double-click to reset · "
                           "click a legend item to hide / show it. The PNG download below is the report version.")
                shown = True
            except Exception as exc:  # fall back to the static chart
                st.caption(f"Interactive view unavailable ({exc}); showing the static chart.")
        if not shown:
            st.image(png, use_container_width=True)
        cc = st.columns([2, 1.3, 1.3])
        part = cc[0].selectbox("Download", ["Whole chart"] + panels, key=f"ml_part_{uid}",
                               label_visibility="collapsed",
                               help="Download the whole chart, or only one part of it (pitch, table, bars ...)")
        panel = None if part == "Whole chart" else part
        base = f"{ctx['team']}_{uid}{'_' + str(ctx['half']) + 'H' if ctx['half'] else ''}{'_' + player if player else ''}"
        fname = f"{base}{'_' + panel if panel else ''}_{ctx['theme'].lower()}.png".replace(" ", "_").replace("#", "")
        if cc[1].button(f"Prepare PNG ({ctx['dpi']} dpi)", key=f"ml_prep_{uid}", use_container_width=True):
            st.session_state[f"_ml_hi_{uid}"] = (args, panel, _png(*args, ctx["dpi"], ctx["md"], panel)[0])
        hi = st.session_state.get(f"_ml_hi_{uid}")
        if hi and hi[0] == args and hi[1] == panel:
            cc[2].download_button("Download PNG", hi[2], file_name=fname, mime="image/png",
                                  key=f"ml_dl_{uid}", use_container_width=True)
        st.divider()

    # ---- heat maps ------------------------------------------------------------
    def _heat_maps(self, ctx: dict) -> None:
        from fap.matchlab import heat as HT
        md, team = ctx["md"], ctx["team"]
        opp = md.opponent(team)
        c = st.columns([1.6, 1.7, 1.4])
        kind = c[0].selectbox("Heat map", list(HT.KINDS), format_func=lambda k: HT.KINDS[k][0], key="ml_heat_kind")
        view = c[1].radio("Team", [team, opp, "Both teams"], horizontal=True, key="ml_heat_view")
        style = c[2].radio("Style", ["smooth", "zones"], horizontal=True, key="ml_heat_style",
                           format_func=lambda s: {"smooth": "Smooth density", "zones": "Positional zones"}[s])
        st.caption(f"{HT.KINDS[kind][0]}: {HT.KINDS[kind][1]}. Every map reads left → right in that team's attacking "
                   "direction. Zones = 3 thirds × 5 lanes (hover a zone for the players).")
        if view == "Both teams":
            self._chart(ctx, "heat_compare", team_level=True, style=style, kind=kind, _uid=f"heatcmp_{kind}")
        else:
            c2 = ctx if view == team else dict(ctx, team=opp, player=None)
            self._chart(c2, "heatmap", style=style, kind=kind, _uid=f"heat_{kind}_{'us' if view == team else 'opp'}")
        st.markdown("#### All heat maps on one page")
        who = st.radio("Sheet for", [team, opp], horizontal=True, key="ml_heat_sheet_team")
        c3 = ctx if who == team else dict(ctx, team=opp, player=None)
        self._chart(c3, "heat_sheet", team_level=who != team, style=style,
                    _uid=f"heatsheet_{'us' if who == team else 'opp'}")

    # ---- team deep-dive ------------------------------------------------------
    def _team_analysis(self, ctx: dict) -> None:
        from fap.matchlab import interactive_team as XT
        from fap.matchlab import team as T
        from fap.matchlab.charts import THEMES
        md, team, half = ctx["md"], ctx["team"], ctx["half"]
        th = THEMES[ctx["theme"]]
        names = ctx.get("names") or {}
        half_tag = {None: "", 1: "_1H", 2: "_2H"}.get(half, "")
        bucket: list = []                     # (filename, html) of the current sub-tab, for "download all"

        def show(fig, key, w: int = 1300):
            """Interactive chart + "Prepare PNG" → "Download PNG" (rendered in the browser at 3× = report quality)."""
            from fap.ui.builtin import plotly_png as PP
            title = re.sub(r"<[^>]+>", "", (fig.layout.title.text or key).split("<br>")[0]).strip() or key
            fname = re.sub(r"[^A-Za-z0-9]+", "_", f"{team}_{title}{half_tag}").strip("_")
            cfg = {"displaylogo": False, "scrollZoom": True, "displayModeBar": True,
                   "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"]}
            st.plotly_chart(fig, use_container_width=True, key=f"ml_ta_{key}", theme=None, config=cfg)
            size = (w, int(fig.layout.height or 520))
            stamp = hashlib.md5(fig.to_json().encode("utf-8")).hexdigest()
            bucket.append((fname, fig, size))
            req, got = st.session_state.get(f"_ml_pngreq_{key}"), st.session_state.get(f"_ml_png_{key}")
            c1, c2, _ = st.columns([1, 1, 2.2])
            if c1.button("Prepare PNG", key=f"ml_ta_prep_{key}", use_container_width=True):
                req = (stamp, f"{key}-{stamp[:8]}-{uuid.uuid4().hex[:8]}")
                st.session_state[f"_ml_pngreq_{key}"] = req
            if req and req[0] == stamp and not (got and got[0] == req[1]):
                with c2:
                    res = PP.render([(fname, fig, *size)], req[1], key=f"ml_ta_pp_{key}")
                if res:
                    got = (req[1], res[0][1])
                    st.session_state[f"_ml_png_{key}"] = got
                else:
                    c2.caption("Rendering…")
            if req and req[0] == stamp and got and got[0] == req[1]:
                c2.download_button("Download PNG", got[1], file_name=f"{fname}.png", mime="image/png", type="primary",
                                   key=f"ml_ta_dl_{key}", use_container_width=True)

        def download_all(section: str):
            """ZIP of the section's charts as PNGs, rendered only on request (keeps page refreshes fast)."""
            import zipfile
            from fap.ui.builtin import plotly_png as PP
            if not bucket:
                return
            st.divider()
            stamp = hashlib.md5("".join(f.to_json() for _, f, _ in bucket).encode("utf-8")).hexdigest()
            rkey, zkey = f"_ml_zipreq_{section}", f"_ml_zip_{section}"
            req, got = st.session_state.get(rkey), st.session_state.get(zkey)
            c1, c2 = st.columns([1, 1])
            if c1.button(f"Prepare all {section} charts (PNG, ZIP)", key=f"ml_ta_zipprep_{section}", use_container_width=True):
                req = (stamp, f"zip-{stamp[:8]}-{uuid.uuid4().hex[:8]}")
                st.session_state[rkey] = req
            if req and req[0] == stamp and not (got and got[0] == req[1]):
                with c2:
                    res = PP.render([(n, f, *sz) for n, f, sz in bucket], req[1], key=f"ml_ta_ppz_{section}")
                if res:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                        for i, (n, data) in enumerate(res, start=1):
                            z.writestr(f"{i:02d}_{n}.png", data)
                    got = (req[1], buf.getvalue())
                    st.session_state[zkey] = got
                else:
                    c2.caption(f"Rendering {len(bucket)} charts…")
            if req and req[0] == stamp and got and got[0] == req[1]:
                c2.download_button(f"Download {section} (ZIP of PNGs)", got[1],
                                   file_name=re.sub(r"[^A-Za-z0-9]+", "_", f"{team}_{section}{half_tag}") + ".zip",
                                   mime="application/zip", key=f"ml_ta_zip_{section}", type="primary",
                                   use_container_width=True)
            bucket.clear()

        def cards(items):
            C.render_metric_row([C.metric_card_html(a, b, hint=c) for a, b, c in items])

        st.caption("Download: press “Prepare PNG” under a chart, then “Download PNG” (high resolution, for reports). "
                   "Each section ends with a ZIP of all its charts as PNGs.")
        sub = st.tabs(["Team stats", "Build-up", "Progression", "Final third", "Second balls"])
        with sub[0]:
            side = T.stats_side_by_side(md, half)
            order = ["Possession", "Passing", "Attacking", "Duels", "Defending", "Discipline", "Set pieces", "Ball losses"]
            cats = [c for c in order if c in set(side["Category"])]
            st.caption("Both teams side by side. Every row shows the count and, where it has attempts, the success "
                       "rate (e.g. duels won / duels decided). Hover a bar for the exact numbers.")
            for i in range(0, len(cats), 2):
                cc = st.columns(2)
                for col, cat in zip(cc, cats[i:i + 2]):
                    with col:
                        show(XT.stats_category(side, cat, md.teams, theme=th), f"stats_{cat}", w=900)
            with st.expander("Full statistics table", expanded=False):
                st.dataframe(side.round(1), use_container_width=True, hide_index=True)
                st.download_button("Team statistics (CSV)", side.round(2).to_csv(index=False).encode("utf-8-sig"),
                                   file_name="team_statistics.csv", mime="text/csv", key="ml_ta_csv")
            download_all("Team stats")
        with sub[1]:
            b = T.buildup(md, team, half)
            bp = b["possessions"]
            if bp.empty:
                st.info("No possessions started in the own third.")
            else:
                ex = b.get("exits", pd.DataFrame())
                gk = b["gk"]
                cards([
                    ("Build-up possessions", str(len(bp)), f"avg {bp['n_passes'].mean():.1f} passes · {bp['duration'].mean():.0f} s"),
                    ("Left our third", f"{int((bp['max_x'] >= md.thirds[0]).sum())}/{len(bp)}",
                     f"{(bp['max_x'] >= md.thirds[0]).mean() * 100:.0f}% of build-ups"),
                    ("Reached final third", f"{int(bp['reached_f3'].sum())}/{len(bp)}", f"{bp['reached_f3'].mean() * 100:.0f}% · {int(bp['shot'].sum())} ended in a shot"),
                    ("Lost in own third", str(len(b["losses"])), "ball losses before the halfway third"),
                    ("Goalkeeper passes", f"{int((gk['outcome'] == 1).sum())}/{len(gk)}" if len(gk) else "–",
                     f"{int((gk['length_class'] == 'Long').sum())} long" if len(gk) else ""),
                ])
                show(XT.buildup_exits(md, team, half=half, theme=th, names=names), "exits")
                show(XT.buildup_flow(md, team, half=half, theme=th), "flow")
                c1, c2 = st.columns(2)
                with c1:
                    show(XT.pass_pairs_network(md, team, b["passes"], "Passing in our own third", theme=th, names=names), "net", w=900)
                with c2:
                    show(XT.gk_distribution(md, team, half=half, theme=th, names=names), "gk", w=900)
                c1, c2 = st.columns(2)
                with c1:
                    show(XT.zone_losses(md, team, b["losses"], "Ball losses in our own third", theme=th, names=names), "loss", w=900)
                with c2:
                    st.markdown("**Who starts the build-up**")
                    stt = b["starters"].head(8)
                    st.dataframe(pd.DataFrame({"Player": [f"#{k} {names.get(k, '')}".strip() for k in stt.index],
                                               "Build-ups started": stt.values}), use_container_width=True, hide_index=True)
                    st.markdown("**Top passing pairs in our third**")
                    pr_ = b["pairs"].head(8)
                    st.dataframe(pd.DataFrame({"From": [f"#{a} {names.get(a, '')}".strip() for a, _ in pr_.index],
                                               "To": [f"#{c} {names.get(c, '')}".strip() for _, c in pr_.index],
                                               "Passes": pr_.values}), use_container_width=True, hide_index=True)
                    if not ex.empty:
                        st.markdown("**Exits by method**")
                        st.dataframe(ex.groupby("method")["success"].agg(exits="size", kept="sum").sort_values("exits", ascending=False),
                                     use_container_width=True)
            download_all("Build-up")
        with sub[2]:
            pr = T.progression(md, team, half)
            mv, tt = pr["entries"], pr["time_to_f3"]
            cards([
                ("Final-third entries", f"{int(mv['success'].sum())}/{len(mv)}", f"{mv['success'].mean() * 100:.0f}% successful" if len(mv) else ""),
                ("By carry", str(int((mv["method"] == "Carry").sum())), "entries carried in"),
                ("Progressive actions", str(len(pr["prog_origins"])), "completed passes + carries"),
                ("Switches of play", f"{int((pr['switches']['outcome'] == 1).sum())}/{len(pr['switches'])}", "completed / attempted"),
                ("Time to final third", f"{tt['seconds'].median():.1f} s" if len(tt) else "–",
                 f"median · {tt['passes'].median():.0f} passes" if len(tt) else ""),
            ])
            show(XT.f3_entry_routes(md, team, half=half, theme=th, names=names), "routes")
            show(XT.entry_breakdown(md, team, half=half, theme=th, names=names), "entrybars")
            c1, c2 = st.columns([1.4, 1])
            with c1:
                show(XT.progressive_origins(md, team, half=half, theme=th, names=names), "origins", w=900)
            with c2:
                show(XT.lines_broken(md, team, half=half, theme=th), "lines", w=900)
            from fap.matchlab import interactive as IX
            show(IX.switches(md, team, half=half, theme=th, names=names), "switches")
            show(XT.time_to_final_third(md, team, half=half, theme=th), "speed")
            show(XT.pass_pairs_network(md, team, pr["middle_passes"], "Passing in the middle third", theme=th, names=names), "midnet")
            download_all("Progression")
        with sub[3]:
            f3 = T.final_third(md, team, half)
            sm = f3["summary"]
            cards([(k, str(v), "") for k, v in list(sm.items())[:6]])
            show(XT.f3_channels(md, team, half=half, theme=th), "channels")
            c1, c2 = st.columns([1.4, 1])
            with c1:
                show(XT.chance_creation(md, team, half=half, theme=th, names=names), "chances", w=900)
            with c2:
                show(XT.shot_sources(md, team, half=half, theme=th), "sources", w=900)
                show(XT.crosses_by_flank(md, team, half=half, theme=th), "crossflank", w=900)
            c1, c2 = st.columns(2)
            with c1:
                show(XT.zone_losses(md, team, f3["losses"], "Ball losses in the final third", theme=th, names=names), "f3loss", w=900)
            with c2:
                f3p = f3["events"][f3["events"]["etype"] == "pass"]
                show(XT.pass_pairs_network(md, team, f3p, "Passing in the final third", theme=th, names=names), "f3net", w=900)
            download_all("Final third")
        with sub[4]:
            from fap.matchlab import secondball as SB
            sb = SB.summary(md, team, half)
            so = SB.summary(md, md.opponent(team), half)
            st.caption(f"Second ball = the first team to control the loose ball within {SB.WINDOW:.0f} s of a contested "
                       "first contact (clearance, aerial duel, or a long / high ball nobody controlled cleanly). "
                       "Out of play / fouls first = no winner.")
            pct = lambda v: "–" if v != v else f"{v:.0f}%"  # noqa: E731
            cards([
                ("Second balls won", f"{sb['won']}/{sb['decided']}", f"{pct(sb['won %'])} · {md.opponent(team)} {so['won']}"),
                ("Won → our final third", str(sb.get("won → final third", 0)), f"{sb.get('won → shot', 0)} led to a shot"),
                ("Lost → their final third", str(sb.get("lost → opp. final third", 0)),
                 f"{sb.get('lost → opp. shot', 0)} led to their shot"),
                ("Aerial duels", f"{sb.get('Aerial duel: won', 0)}/{sb.get('Aerial duel: decided', 0)}", "second ball won"),
                ("After clearances", f"{sb.get('Clearance: won', 0)}/{sb.get('Clearance: decided', 0)}", "second ball won"),
                ("Situations", str(sb["contests"]), f"{sb['contests'] - sb['decided']} without a winner"),
            ])
            show(XT.second_balls_map(md, team, half=half, theme=th, names=names), "sb_map")
            c1, c2 = st.columns([1.5, 1])
            with c1:
                show(XT.second_balls_breakdown(md, team, half=half, theme=th), "sb_break", w=900)
            with c2:
                show(XT.second_balls_players(md, team, half=half, theme=th, names=names), "sb_players", w=700)
            with st.expander("Every second-ball situation", expanded=False):
                tb = SB.for_team(md, team, half)
                if not tb.empty:
                    st.dataframe(tb[["minute", "kind", "contact_team", "contact_player", "contact_etype", "outcome",
                                     "winner", "winner_player", "win_etype", "seconds", "third", "then_final_third",
                                     "then_shot"]].round(2), use_container_width=True, hide_index=True)
            download_all("Second balls")

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
        labels = {f"{r['Player']} {names.get(r['Player'].lstrip('#'), '')}".strip(): r["Player"].lstrip("#")
                  for _, r in tab.iterrows()}
        st.divider()
        self._leaderboards(ctx)
        st.divider()
        self._compare_players(ctx, labels)
        st.divider()
        st.markdown("### Player dashboard")
        pick = st.selectbox("Player", list(labels), key="ml_pl_dash")
        self._player_dashboard(ctx, labels[pick])

    def _leaderboards(self, ctx: dict) -> None:
        from fap.matchlab import interactive as IX
        from fap.matchlab import player as PP
        from fap.matchlab.charts import THEMES
        md, team, half = ctx["md"], ctx["team"], ctx["half"]
        names = ctx.get("names") or {}
        th = THEMES[ctx["theme"]]
        stats = PP.player_stats(md, team, half)
        if stats.empty:
            return
        st.markdown("### Leaders")
        st.caption("Top players per action. Bar colour = success rate where the action has a tagged outcome "
                   "(red → amber → green); hover a bar for the exact numbers.")
        grp = st.radio("Category", list(PP.LEADERBOARDS), horizontal=True, key="ml_lead_grp",
                       label_visibility="collapsed")
        specs = PP.LEADERBOARDS[grp]
        cfg = {"displaylogo": False, "displayModeBar": False}
        for i in range(0, len(specs), 3):
            cols = st.columns(3)
            for col, (title, c, num, den) in zip(cols, specs[i:i + 3]):
                with col:
                    st.plotly_chart(IX.leaderboard(stats, title, c, num, den, theme=th, names=names),
                                    use_container_width=True, key=f"ml_lead_{grp}_{c}", theme=None, config=cfg)
        with st.expander("Full table (all players, all actions)", expanded=False):
            show = stats.copy()
            show.insert(1, "Name", show["player"].map(lambda p: names.get(p, "")))
            show["player"] = "#" + show["player"]
            st.dataframe(show.round(1), use_container_width=True, hide_index=True)
            st.download_button("Leaders table (CSV)", show.round(2).to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{team}_player_actions.csv", mime="text/csv", key="ml_lead_csv")

    def _compare_players(self, ctx: dict, labels: dict) -> None:
        from fap.matchlab import interactive as IX
        from fap.matchlab import player as PP
        from fap.matchlab.charts import THEMES
        md, team, half = ctx["md"], ctx["team"], ctx["half"]
        names = ctx.get("names") or {}
        st.markdown("### Compare players")
        opts = list(labels)
        picked = st.multiselect("Players (2–4)", opts, default=opts[:2], max_selections=4, key="ml_cmp_pick")
        if len(picked) < 2:
            st.caption("Pick at least two players.")
            return
        ids = [labels[p] for p in picked]
        st.plotly_chart(IX.compare_radar(md, team, ids, half=half, theme=THEMES[ctx["theme"]], names=names),
                        use_container_width=True, key="ml_cmp_radar", theme=None,
                        config={"displaylogo": False, "displayModeBar": False})
        c = PP.compare_players(md, team, ids, half)
        if c.empty:
            return
        table = c.set_index("Metric")[[str(i) for i in ids] + ["Team best"]]
        table.columns = [f"#{i} {names.get(str(i), '')}".strip() for i in ids] + ["Team best"]
        hib = c.set_index("Metric")["Higher is better"]

        def best(row):
            vals = row.iloc[:-1]
            target = vals.max() if hib[row.name] else vals.min()
            return ["font-weight:700; color:#2ed8a3" if (v == target and not pd.isna(v)) else "" for v in vals] + [""]

        def num(v):
            if pd.isna(v):
                return "–"
            return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"

        st.dataframe(table.style.apply(best, axis=1).format(num), use_container_width=True)
        st.caption("Green = best of the selected players for that metric (for ball losses: fewest). "
                   "Team best is among team-mates with ≥ 15 actions.")

    def _player_dashboard(self, ctx: dict, pl: str) -> None:
        """Interactive player dashboard: KPI cards, pass map, directions, receptions, defending + losses,
        pass details, shots (when he has any) and comparison with the team. Every chart has hover."""
        from fap.matchlab import interactive as IX
        from fap.matchlab import player as PP
        from fap.matchlab.charts import THEMES
        md, team, half = ctx["md"], ctx["team"], ctx["half"]
        th = THEMES[ctx["theme"]]
        names = ctx.get("names") or {}
        pr = PP.profile(md, team, pl, half)
        cfg = {"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"]}
        st.markdown(f"#### #{pl} {names.get(pl, '')}".rstrip())
        pa, rc, ca, sh, de, lo = (pr["passing"], pr["receiving"], pr["carrying"], pr["shooting"], pr["defending"],
                                  pr["losses"])
        cards = [
            C.metric_card_html("Passing", f"{pa['Completed']}/{pa['Passes']}", hint=f"{_fmt(pa['Accuracy %'], 0, True)} accurate · {_fmt(pa['Forward %'], 0, True)} forward"),
            C.metric_card_html("Progression", f"{pa['Progressive (completed)']} + {ca['Progressive carries']}", hint="prog. passes + carries"),
            C.metric_card_html("Received", str(rc["Receptions"]), hint=f"{rc['Progressive receptions']} progressive · {rc['In final third']} in final third"),
            C.metric_card_html("Chance creation", f"{pa['Key passes']} · {pa['Final-third entries']}", hint="key passes · final-third entries"),
            C.metric_card_html("Defending", str(de["Defensive actions"]), hint=f"{de['Recoveries']} recoveries · duels {de['Duels won / lost']}"),
            C.metric_card_html("Ball losses", str(lo["Ball losses"]), hint=f"{lo['in defensive third']} in own third"),
        ]
        if sh["Shots"]:
            cards.insert(4, C.metric_card_html("Shooting", f"{sh['Shots']} · {sh['Goals']}G", hint=f"xG {sh['xG']:.2f} · {sh['On target']} on target"))
        C.render_metric_row(cards)

        def show(fig, key):
            st.plotly_chart(fig, use_container_width=True, key=f"ml_pd_{key}", theme=None, config=cfg)

        c1, c2 = st.columns([1.55, 1])
        with c1:
            f = IX.pass_map(md, team, half=half, player=pl, theme=th, names=names)
            f.update_layout(height=470)
            show(f, "passes")
        with c2:
            show(IX.player_rose(md, team, pl, half=half, theme=th), "rose")
        show(IX.player_breakdown(md, team, pl, half=half, theme=th), "breakdown")
        c1, c2 = st.columns(2)
        with c1:
            show(IX.player_receptions(md, team, pl, half=half, theme=th, names=names), "receptions")
        with c2:
            show(IX.player_defence(md, team, pl, half=half, theme=th, names=names), "defence")
        if sh["Shots"]:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                f = IX.shots(md, team, half=half, player=pl, theme=th, names=names)
                f.update_layout(height=470)
                show(f, "shots")
            with c2:
                st.markdown("**Shooting**")
                st.dataframe(pd.DataFrame({"value": [(_fmt(v, 2) if isinstance(v, float) else str(v)) for v in sh.values()]},
                                          index=list(sh.keys())), use_container_width=True)
        show(IX.player_compare(md, team, pl, half=half, theme=th), "compare")

        with st.expander("All numbers", expanded=False):
            cc = st.columns(3)
            blocks = [("Passing", pa), ("Receiving", rc), ("Carrying", ca), ("Defending", de), ("Ball losses", lo),
                      ("Counter-press", pr["counterpress"])]
            if sh["Shots"]:
                blocks.insert(3, ("Shooting", sh))
            for i, (title, d) in enumerate(blocks):
                with cc[i % 3]:
                    st.markdown(f"**{title}**")
                    if d:
                        st.dataframe(pd.DataFrame({"value": [(_fmt(v, 1) if isinstance(v, float) else str(v)) for v in d.values()]},
                                                  index=list(d.keys())), use_container_width=True)
            cc = st.columns(3)
            with cc[0]:
                st.markdown("**Passes to**")
                st.dataframe(pr["to_whom"].rename(lambda k: f"#{k} {names.get(k, '')}".strip()).to_frame("passes"),
                             use_container_width=True)
            with cc[1]:
                st.markdown("**Received from**")
                st.dataframe(pr["from_whom"].rename(lambda k: f"#{k} {names.get(k, '')}".strip()).to_frame("passes"),
                             use_container_width=True)
            with cc[2]:
                st.markdown("**On-ball actions by phase (%)**")
                st.dataframe(pr["phases"].round(1).rename(columns={"him": "player", "team": "team"}), use_container_width=True)
            st.markdown("**Compared with the team**")
            st.dataframe(pr["comparison"].round(2), use_container_width=True, hide_index=True)
        st.markdown("##### Report version (PNG)")
        self._chart(dict(ctx, player=pl), "player")

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
