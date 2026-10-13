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

import pandas as pd
import streamlit as st

from fap.core.plugin import PluginInfo
from fap.identity.roles import Role
from fap.theme import components as C
from fap.ui.page import Page, page_registry

ML_FILE = "_ml_file"            # {"name", "data" (bytes)}
ML_KIND = "matchlab_match"
HALVES = {"Full match": None, "1st half": 1, "2nd half": 2}


# ---------------------------------------------------------------- cached engine calls
@st.cache_resource(show_spinner=False, max_entries=8)
def _load(sig: str, _data: bytes, name: str, mirror_y: bool, t1: float, t2: float):
    from fap.matchlab import load_match
    return load_match(_data, name, mirror_y=mirror_y, thirds=(t1, t2))


@st.cache_data(show_spinner=False, max_entries=300)
def _png(sig: str, key: str, team: str, half, player, theme: str, opts: tuple, dpi: int, _md) -> bytes:
    from fap.matchlab import charts as CH
    fig = CH.render(key, _md, team, half=half, player=player, theme=CH.THEMES[theme], **dict(opts))
    return CH.to_png(fig, dpi=dpi)


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
            c = st.columns(4)
            t1 = c[0].slider("Build-up line (m)", 20.0, 45.0, 35.0, 1.0, key="ml_t1",
                             help="Phase boundary: build-up ends here")
            t2 = c[1].slider("Final-third line (m)", 55.0, 85.0, 70.0, 1.0, key="ml_t2")
            mirror = c[2].checkbox("Mirror flanks (left ↔ right)", key="ml_mirror",
                                   help="Use if the tagging tool writes y upward (flanks come out swapped)")
            dpi = c[3].select_slider("Download quality (dpi)", [150, 200, 300, 400], value=300, key="ml_dpi")
        data = f["data"]
        sig = hashlib.sha1(data).hexdigest()[:16] + f"|{mirror}|{t1}|{t2}"
        try:
            with st.spinner("Analysing match…"):
                md = _load(sig, data, f["name"], mirror, t1, t2)
        except Exception as exc:
            C.render_alert(f"Could not read this file: {exc}", "danger")
            return

        c = st.columns([2, 2, 2, 1.4])
        team = c[0].selectbox("Team", md.teams, key="ml_team")
        half_label = c[1].selectbox("Period", list(HALVES), key="ml_half")
        half = HALVES[half_label]
        players = sorted({p for p in md.team_events(team)["player"] if p and p != "0"}, key=lambda s: (len(s), s))
        pl_label = c[2].selectbox("Player", ["All players"] + [f"#{p}" for p in players], key="ml_player")
        player = None if pl_label == "All players" else pl_label[1:]
        theme = c[3].radio("Theme", ["Dark", "Light"], horizontal=True, key="ml_theme")
        if not players:
            st.caption(f"{team} has no player numbers in this file — team-level analysis only.")

        self._kpis(md, team, half)
        ctx = dict(sig=sig, md=md, team=team, half=half, player=player, theme=theme, dpi=int(dpi))

        tabs = st.tabs(["Overview", "Passing", "Progression & attack", "Phases of play", "Defending",
                        "Territory", "Players", "Data & method"])
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
            self._chart(ctx, "sequences", team_level=True)
        with tabs[4]:
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
            png = _png(*args, 110, ctx["md"])
        except Exception as exc:
            C.render_alert(f"{spec.label}: could not render ({exc})", "warning")
            return
        st.image(png, use_container_width=True)
        fname = f"{ctx['team']}_{key}{'_' + str(ctx['half']) + 'H' if ctx['half'] else ''}{'_' + player if player else ''}_{ctx['theme'].lower()}.png"
        if st.button(f"Prepare {spec.label} PNG ({ctx['dpi']} dpi)", key=f"ml_prep_{key}"):
            st.session_state[f"_ml_hi_{key}"] = (args, _png(*args, ctx["dpi"], ctx["md"]))
        hi = st.session_state.get(f"_ml_hi_{key}")
        if hi and hi[0] == args:
            st.download_button(f"Download {spec.label}", hi[1], file_name=fname.replace(" ", "_"),
                               mime="image/png", key=f"ml_dl_{key}")
        st.divider()

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
        st.dataframe(tab, use_container_width=True, hide_index=True)
        st.download_button("Player table (CSV)", tab.to_csv(index=False).encode("utf-8"),
                           file_name=f"{team}_players.csv", mime="text/csv", key="ml_pl_csv")
        pick = st.selectbox("Player dashboard", list(tab["Player"]), key="ml_pl_dash")
        self._chart(dict(ctx, player=pick.lstrip("#")), "player")

    def _method(self, md, team) -> None:
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
        ev = md.events
        cols = ["half", "minute", "team", "player", "etype", "tag", "x", "y", "end_x", "end_y", "end_source",
                "receiver", "outcome", "direction", "length", "length_class", "pass_type", "progressive",
                "f3_entry", "box_entry", "line", "height", "phase", "def_phase", "transition", "counter_press",
                "poss_id", "xg", "derived"]
        out = ev[[c for c in cols if c in ev.columns]].copy()
        out["labels"] = ev["labels"].map(lambda l: ", ".join(l))
        st.dataframe(out.head(300), use_container_width=True, hide_index=True)
        st.download_button("Enriched events (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="matchlab_events.csv", mime="text/csv", key="ml_ev_csv")
        if not md.possessions.empty:
            st.download_button("Possessions (CSV)", md.possessions.to_csv(index=False).encode("utf-8"),
                               file_name="matchlab_possessions.csv", mime="text/csv", key="ml_poss_csv")
