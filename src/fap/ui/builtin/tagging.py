"""Tagging Studio — a first-class Analysis workspace for manual event tagging.

A thin Streamlit view over the pure ``fap.tagging`` core (schema, coordinate
engine, session + undo/redo, validation, export). The pitch and the goal are drawn
by the canonical renderers (``PitchFactory`` and ``fap.visuals.goal``) — never
duplicated — and the interactive canvas (``tagging_canvas``) only reports clicks as
interior fractions, which the coordinate engine converts to canonical football
coordinates. A native coordinate-input fallback keeps tagging fully usable if the
JS canvas cannot mount. The live session is autosaved through the WorkspaceManager
so reruns/navigation never lose work.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import streamlit as st

from fap.core.plugin import PluginInfo
from fap.identity.roles import Role
from fap.tagging import coordinates as TC
from fap.tagging.export import session_to_canonical_frame, session_to_csv, to_project_dict
from fap.tagging.models import TagEvent, TaggingSession
from fap.tagging.schema import (PERIODS, PRESETS, TEAMS, preset_tags, shortcut_map,
                                tag_by_key, tags_for_space)
from fap.tagging.validation import validate_session
from fap.theme import components as C
from fap.ui.page import Page, page_registry

_SESSION = "_tag_session"          # TaggingSession in session_state
_SEL = "_tag_selected"             # selected event id
_TS = "_tag_last_ts"               # last processed canvas action ts
_LOADED = "_tag_loaded"
_SEEN = "_tag_seen"                # ids of canvas events / ops already applied (idempotent sync)
_ACKED = "_tag_acked"              # ids to confirm back to the browser on the next render
TEAM_COLORS = ("#1f6feb", "#e5484d", "#8b96a8")


@page_registry.register
class TaggingStudioPage(Page):
    info = PluginInfo(id="tagging", name="Tagging", category="page")
    section = "Analysis"
    icon = "target"
    order = 45                      # after Open Play / Set Piece / Tactical Board
    min_role = Role.PERFORMANCE_ANALYST

    # ------------------------------------------------------------------ render
    def render(self, shell) -> None:
        svc = getattr(shell.platform, "tagging", None) if shell.platform else None
        session = self._session(shell, svc)

        C.render_section_title(
            "Tagging Studio", eyebrow="Analysis", icon_name="target",
            subtitle="Tag events on the pitch and in the goal (metres, default 105 × 68), then export analysis-ready CSV.")

        # ---- toolbar ----
        bar = st.columns([1, 4, 1.6, 1, 1])
        layer = bar[0].radio("Layer", ["Pitch", "Goal"], key="tag_layer",
                             horizontal=True, label_visibility="collapsed")
        space = "pitch" if layer == "Pitch" else "goal"
        if bar[2].button("→ Open Play", use_container_width=True, key="tag_send_op",
                         type="primary", help="Import these tags into the Data Hub and "
                         "activate them so they render on the Open Play maps."):
            self._send_open_play(shell, svc, session)
        if bar[3].button("Save", use_container_width=True, key="tag_save_btn"):
            self._save_project(shell, svc, session)
        self._export_button(bar[4], session)
        self._sent_banner(shell)

        controls = self._controls(session, space)
        self._canvas(shell, svc, session, space, controls)
        left, right = st.columns([1, 1])
        with left:
            self._selected_panel(shell, svc, session)
        with right:
            self._native_add(shell, svc, session, space, controls)

        self._event_table(session)
        self._status_bar(shell, svc, session)
        self._autosave(shell, svc, session)

    # ------------------------------------------------------------------ session
    def _session(self, shell, svc) -> TaggingSession:
        if _SESSION not in st.session_state:
            loaded = None
            if svc is not None:
                try:
                    loaded, _ = svc.load_autosave(shell.user)
                except Exception:
                    loaded = None
            st.session_state[_SESSION] = loaded or TaggingSession(
                analyst=getattr(shell.user, "email", "") or "")
        return st.session_state[_SESSION]

    def _autosave(self, shell, svc, session) -> None:
        if svc is not None:
            svc.autosave(shell.user, session, ui_state={"layer": st.session_state.get("tag_layer")})

    # ------------------------------------------------------------------ controls
    def _controls(self, session, space) -> dict[str, Any]:
        st.caption("TAGGING CONTROLS")
        c = st.columns([2, 2, 2, 1, 2])
        preset = c[0].selectbox("Preset", list(PRESETS), key="tag_preset")
        # event types = preset ∩ current layer space (always non-empty via fallback)
        options = [t for t in preset_tags(preset) if t.coordinate_space == space]
        if not options:
            options = tags_for_space(space)
        labels = {t.key: t.label for t in options}
        etype = c[1].selectbox("Event type", [t.key for t in options],
                               format_func=lambda k: labels.get(k, k), key=f"tag_etype_{space}")
        tag = tag_by_key(etype)
        teams = [session.home_team, session.away_team, "Neutral"]
        team = c[2].selectbox("Team", teams, key="tag_team")
        player = c[3].text_input("Player", key="tag_player", placeholder="e.g. P9")
        period = c[4].selectbox("Period", list(PERIODS), key="tag_period")

        c2 = st.columns([2, 2, 1, 1, 2])
        outcomes = ["—", *(tag.outcomes if tag else ())]
        outcome = c2[0].selectbox("Outcome", outcomes, key=f"tag_outcome_{etype}",
                                  help="Only outcomes relevant to this event are shown.")
        minute = c2[1].number_input("Minute", 0, 130, 0, key="tag_minute")
        second = c2[2].number_input("Second", 0, 59, 0, key="tag_second")
        direction = c2[3].selectbox("Attack", ["L→R", "R→L"], key="tag_direction",
                                    help="Recorded per session; coordinates are never silently flipped.")
        video_ts = c2[4].number_input("Video ts (s)", 0.0, step=1.0, key="tag_video",
                                      help="Optional — leave 0 if not syncing video.")
        session.set_meta(attack_direction=("lr" if direction == "L→R" else "rl"), preset=preset)

        with st.expander("Match context & precision toggles", expanded=False):
            m = st.columns(4)
            session.match_id = m[0].text_input("Match id", value=session.match_id, key="tag_match")
            session.competition = m[1].text_input("Competition", value=session.competition, key="tag_comp")
            session.match_date = m[2].text_input("Date", value=session.match_date, key="tag_date")
            session.opponent = m[3].text_input("Opponent", value=session.opponent, key="tag_opp")
            d = st.columns([1, 1, 2])
            pl = d[0].number_input("Pitch length (m)", 90.0, 120.0, float(session.pitch_length or 105.0), 0.5,
                                   key="tag_pitch_len", help="Default 105 m")
            pw = d[1].number_input("Pitch width (m)", 45.0, 90.0, float(session.pitch_width or 68.0), 0.5,
                                   key="tag_pitch_wid", help="Default 68 m")
            session.set_meta(pitch_length=float(pl), pitch_width=float(pw))
            d[2].caption("Coordinates are shown and exported in metres on this pitch (x along the length from the "
                         "left goal line, y across from the bottom touchline). The canonical 0–100 columns stay "
                         "the same for Open Play.")
            t = st.columns(7)
            toggles = {
                "grid": t[0].checkbox("Grid", key="tag_tg_grid"),
                "zones": t[1].checkbox("Zones", key="tag_tg_zones"),
                "penalty": t[2].checkbox("Penalty area", value=True, key="tag_tg_pen"),
                "halfway": t[3].checkbox("Halfway", value=True, key="tag_tg_half"),
                "thirds": t[4].checkbox("Thirds", key="tag_tg_thirds"),
                "channels": t[5].checkbox("Channels", key="tag_tg_chan"),
                "coords": t[6].checkbox("Coordinates", value=True, key="tag_tg_coords"),
            }
        notes = st.text_input("Notes (optional)", key="tag_notes")
        return {"event_type": etype, "tag": tag, "team": team, "player": player,
                "period": period, "outcome": ("" if outcome == "—" else outcome),
                "minute": int(minute), "second": int(second), "notes": notes,
                "video_timestamp": (float(video_ts) or None), "toggles": toggles}

    # ------------------------------------------------------------------ canvas
    def _team_color(self, session, team: str) -> str:
        return {session.home_team: TEAM_COLORS[0], session.away_team: TEAM_COLORS[1]}.get(team, TEAM_COLORS[2])

    def _overlay(self, session, space) -> list[dict]:
        out = []
        for i, e in enumerate(session.events, 1):
            if e.coordinate_space != space:
                continue
            if space == "goal":
                if e.goal_x is None or e.goal_y is None:
                    continue
                fx, fy = TC.goal_fraction_from_canonical(e.goal_x, e.goal_y)
            else:
                if e.x is None or e.y is None:
                    continue
                fx, fy = TC.pitch_fraction_from_canonical(e.x, e.y)
            tag = tag_by_key(e.event_type)
            m = {"id": e.id, "ifx": fx, "ify": fy, "color": self._team_color(session, e.team),
                 "label": f"#{i} {tag.label if tag else e.event_type}" + (f" · {e.player}" if e.player else "")
                          + (f" · {e.outcome}" if e.outcome else "") + (f" · {self._clock(e)}" if e.minute else "")}
            if space == "pitch" and e.x2 is not None and e.y2 is not None:
                m["ifx2"], m["ify2"] = TC.pitch_fraction_from_canonical(e.x2, e.y2)
            out.append(m)
        return out

    def _canvas(self, shell, svc, session, space, controls) -> None:
        tag = controls["tag"]
        mode = "line" if (tag and tag.geometry == "line" and space == "pitch") else "point"
        ctx = {"label": " · ".join(x for x in ((tag.label if tag else controls["event_type"]), controls["team"],
                                                controls["player"], controls["outcome"]) if x),
               "color": self._team_color(session, controls["team"]), "attack": session.attack_direction}
        dims = {"length": float(session.pitch_length or 105.0), "width": float(session.pitch_width or 68.0)}
        rendered, intent = False, None
        try:
            from fap.ui.builtin.tagging_canvas import tagging_board
            rendered, intent = tagging_board(space=space, dims=dims, overlay=self._overlay(session, space), mode=mode,
                                             toggles=controls["toggles"], ctx=ctx,
                                             acked=st.session_state.get(_ACKED, []),
                                             selected=st.session_state.get(_SEL), key=f"tagboard_{space}")
        except Exception:
            rendered, intent = False, None
        if not rendered:
            st.caption("Interactive board unavailable — use the coordinate inputs on the right.")
        elif intent is not None and intent.get("kind") == "batch":
            self._apply_batch(shell, svc, session, space, controls, intent)

    def _apply_batch(self, shell, svc, session, space, controls, batch) -> None:
        """Apply every event / op the browser sent exactly once (ids remembered), then confirm them."""
        seen = st.session_state.setdefault(_SEEN, [])
        seen_set = set(seen)
        changed = False
        for e in batch["events"]:
            if e["cid"] in seen_set:
                continue
            if space == "goal":
                gx, gy = TC.canonical_from_goal_fraction(e["ifx"], e["ify"])
                self._create(session, controls, space, goal_x=gx, goal_y=gy)
            elif e["action"] == "line":
                x, y = TC.canonical_from_pitch_fraction(e["ifx"], e["ify"])
                x2, y2 = TC.canonical_from_pitch_fraction(e["ifx2"], e["ify2"])
                self._create(session, controls, space, x=x, y=y, x2=x2, y2=y2)
            else:
                x, y = TC.canonical_from_pitch_fraction(e["ifx"], e["ify"])
                self._create(session, controls, space, x=x, y=y)
            seen.append(e["cid"]); seen_set.add(e["cid"]); changed = True
        for o in batch["ops"]:
            if o["aid"] in seen_set:
                continue
            act = o["action"]
            if act == "select":
                st.session_state[_SEL] = o.get("select")
            elif act == "delete":
                if o.get("select"):
                    st.session_state[_SEL] = o["select"]
                self._delete_selected(session)
            elif act == "undo":
                session.undo()
            elif act == "redo":
                session.redo()
            elif act == "cancel":
                st.session_state.pop(_SEL, None)
            seen.append(o["aid"]); seen_set.add(o["aid"]); changed = True
        del seen[:-5000]
        ids = [e["cid"] for e in batch["events"]] + [o["aid"] for o in batch["ops"]]
        if ids != st.session_state.get(_ACKED):
            st.session_state[_ACKED] = ids
            changed = True
        if changed:
            self._autosave(shell, svc, session)
            st.rerun()

    def _create(self, session, controls, space, **coords) -> None:
        e = TagEvent(event_type=controls["event_type"], coordinate_space=space,
                     team=controls["team"], player=controls["player"],
                     period=controls["period"], outcome=controls["outcome"],
                     minute=controls["minute"], second=controls["second"],
                     notes=controls["notes"], video_timestamp=controls["video_timestamp"],
                     **coords)
        session.add_event(e)
        st.session_state[_SEL] = e.id

    # ------------------------------------------------------------------ native add (fallback + precision)
    def _native_add(self, shell, svc, session, space, controls) -> None:
        tag = controls["tag"]
        is_line = bool(tag and tag.geometry == "line" and space == "pitch")
        L, W = float(session.pitch_length or 105.0), float(session.pitch_width or 68.0)
        sx, sy = (TC.GOAL_WIDTH_M, TC.GOAL_HEIGHT_M) if space == "goal" else (L, W)
        with st.expander("Add by coordinates in metres (precise / no-mouse)", expanded=False):
            fx = "across goal (m)" if space == "goal" else f"x (m, 0–{L:g})"
            fy = "height (m)" if space == "goal" else f"y (m, 0–{W:g})"
            a = st.columns(2)
            xm = a[0].number_input(fx, 0.0, sx, sx / 2, 0.1, key=f"tag_nx_{space}")
            ym = a[1].number_input(fy, 0.0, sy, sy / 2, 0.1, key=f"tag_ny_{space}")
            xv, yv = TC.from_metres(xm, sx), TC.from_metres(ym, sy)
            x2v = y2v = None
            if is_line:
                b = st.columns(2)
                x2m = b[0].number_input(f"end x (m)", 0.0, L, min(L, L / 2 + 10), 0.1, key="tag_nx2")
                y2m = b[1].number_input(f"end y (m)", 0.0, W, W / 2, 0.1, key="tag_ny2")
                x2v, y2v = TC.from_metres(x2m, L), TC.from_metres(y2m, W)
            if st.button("Add event", type="primary", key="tag_native_add", use_container_width=True):
                if space == "goal":
                    self._create(session, controls, space, goal_x=xv, goal_y=yv)
                elif is_line:
                    self._create(session, controls, space, x=xv, y=yv, x2=x2v, y2=y2v)
                else:
                    self._create(session, controls, space, x=xv, y=yv)
                self._autosave(shell, svc, session)
                st.rerun()

    # ------------------------------------------------------------------ selected panel (edit/delete)
    def _selected_panel(self, shell, svc, session) -> None:
        sel = st.session_state.get(_SEL)
        e = session.get(sel) if sel else None
        if e is None:
            st.caption("Select an event (canvas marker or table row) to edit it.")
            return
        tag = tag_by_key(e.event_type)
        with st.container(border=True):
            st.markdown(f"**Editing** `{e.id[:8]}` · {tag.label if tag else e.event_type}")
            c = st.columns(2)
            player = c[0].text_input("Player", value=e.player, key=f"ed_player_{e.id}")
            team = c[1].text_input("Team", value=e.team, key=f"ed_team_{e.id}")
            outs = ["—", *(tag.outcomes if tag else ())]
            idx = outs.index(e.outcome) if e.outcome in outs else 0
            outcome = st.selectbox("Outcome", outs, index=idx, key=f"ed_out_{e.id}")
            coords = st.columns(4)
            vals: dict[str, Any] = {}
            L, W = float(session.pitch_length or 105.0), float(session.pitch_width or 68.0)
            size = {"x": L, "x2": L, "y": W, "y2": W, "goal_x": TC.GOAL_WIDTH_M, "goal_y": TC.GOAL_HEIGHT_M}
            for i, f in enumerate(tag.required_fields if tag else ()):
                sz = size.get(f, 100.0)
                mv = coords[i].number_input(f"{f} (m)", 0.0, float(sz), float(TC.to_metres(getattr(e, f) or 0.0, sz)),
                                            0.1, key=f"ed_{f}_{e.id}")
                vals[f] = TC.from_metres(mv, sz)
            notes = st.text_input("Notes", value=e.notes, key=f"ed_notes_{e.id}")
            b = st.columns(2)
            if b[0].button("Update", type="primary", key=f"ed_upd_{e.id}", use_container_width=True):
                session.edit_event(e.id, player=player, team=team, notes=notes,
                                   outcome=("" if outcome == "—" else outcome), **vals)
                self._autosave(shell, svc, session)
                st.rerun()
            if b[1].button("Delete", key=f"ed_del_{e.id}", use_container_width=True):
                self._delete_selected(session)
                self._autosave(shell, svc, session)
                st.rerun()

    def _delete_selected(self, session) -> None:
        sel = st.session_state.get(_SEL)
        if sel and session.delete_event(sel):
            st.session_state.pop(_SEL, None)

    # ------------------------------------------------------------------ event table
    def _event_table(self, session) -> None:
        st.markdown("#### Tagged events")
        if not session.events:
            st.caption("No events yet — tag on the canvas or add by coordinates.")
            return
        rows = []
        L, W = float(session.pitch_length or 105.0), float(session.pitch_width or 68.0)
        for i, e in enumerate(session.events, 1):
            rows.append({"#": i, "id": e.id, "Time": self._clock(e), "Team": e.team,
                         "Player": e.player, "Event": e.event_type, "Out": e.outcome,
                         "Space": e.coordinate_space,
                         "X (m)": TC.to_metres(e.x, L), "Y (m)": TC.to_metres(e.y, W),
                         "X2 (m)": TC.to_metres(e.x2, L), "Y2 (m)": TC.to_metres(e.y2, W),
                         "Goal across (m)": TC.to_metres(e.goal_x, TC.GOAL_WIDTH_M),
                         "Goal height (m)": TC.to_metres(e.goal_y, TC.GOAL_HEIGHT_M)})
        ids = [r["id"] for r in rows]
        display = [{k: v for k, v in r.items() if k != "id"} for r in rows]
        try:
            ev = st.dataframe(display, use_container_width=True, hide_index=True,
                              on_select="rerun", selection_mode="single-row",
                              key="tag_table")
            picked = ev.selection.rows if ev and hasattr(ev, "selection") else []
            if picked:
                st.session_state[_SEL] = ids[picked[0]]
        except Exception:
            st.dataframe(display, use_container_width=True, hide_index=True)
            choice = st.selectbox("Select event", ["—", *ids],
                                  format_func=lambda x: "—" if x == "—" else x[:8], key="tag_pick")
            if choice != "—":
                st.session_state[_SEL] = choice

    @staticmethod
    def _clock(e: TagEvent) -> str:
        if e.minute is None:
            return ""
        return f"{e.minute:02d}:{(e.second or 0):02d}"

    # ------------------------------------------------------------------ status bar
    def _status_bar(self, shell, svc, session) -> None:
        st.divider()
        cols = st.columns([3, 1, 1, 1, 1])
        problems = validate_session(session)
        state = f"{len(session)} events"
        state += f" · {len(problems)} issue(s)" if problems else " · valid"
        cols[0].caption(f"Status: {state}")
        if cols[1].button("Undo", disabled=not session.can_undo(), key="tag_undo",
                          use_container_width=True):
            session.undo(); self._autosave(shell, svc, session); st.rerun()
        if cols[2].button("Redo", disabled=not session.can_redo(), key="tag_redo",
                          use_container_width=True):
            session.redo(); self._autosave(shell, svc, session); st.rerun()
        if cols[3].button("Delete", disabled=not st.session_state.get(_SEL),
                          key="tag_del", use_container_width=True):
            self._delete_selected(session); self._autosave(shell, svc, session); st.rerun()
        if cols[4].button("Clear all", key="tag_clear", use_container_width=True):
            session.clear(); st.session_state.pop(_SEL, None)
            self._autosave(shell, svc, session); st.rerun()

    # ------------------------------------------------------------------ export
    def _export_button(self, col, session) -> None:
        problems = validate_session(session)
        if problems:
            col.button("Export", disabled=True, use_container_width=True, key="tag_export_disabled",
                       help=f"{len(problems)} validation issue(s) — fix before export.")
            return
        col.download_button("Export", data=session_to_csv(session, metres=True),
                            file_name="tagging.csv", mime="text/csv",
                            use_container_width=True, key="tag_export_csv",
                            disabled=not session.events)

    def _send_open_play(self, shell, svc, session) -> None:
        if svc is None:
            st.warning("Tagging service unavailable.")
            return
        name = (session.match_id or "").strip() or "Tagging session"
        try:
            ds = svc.send_to_datahub(shell.user, session, name=name,
                                     workspace_id=shell.workspace_id)
            st.session_state["_tag_sent"] = ds.name
            st.toast(f"Imported & activated '{ds.name}'.")
        except ValueError as exc:
            st.warning(str(exc))

    def _sent_banner(self, shell) -> None:
        name = st.session_state.get("_tag_sent")
        if not name:
            return
        C.render_alert(f"Imported **{name}** into the Data Hub and set it active. "
                       "Open the Open Play Studio to render the maps.", "success")
        cols = st.columns([1, 4])
        if cols[0].button("Open Play Studio", key="tag_goto_op", type="primary"):
            st.session_state.pop("_tag_sent", None)
            shell.goto("openplay_studio")
            st.rerun()

    def _save_project(self, shell, svc, session) -> None:
        import json
        data = json.dumps(to_project_dict(session, name="Tagging session"), indent=2)
        st.download_button("Download project (.json)", data=data,
                           file_name="tagging_project.json", mime="application/json",
                           key="tag_project_dl")
        st.toast("Project ready to download below the toolbar.")

    # ------------------------------------------------------------------ rendering (reuses canonical renderers)
    @staticmethod
    def _theme():
        from fap.themes import ThemeManager
        return ThemeManager("assets/themes").get("opta_light")

    @staticmethod
    def _nonce(session) -> str:
        return f"{len(session)}_{st.session_state.get(_SEL, '')}"

    def _render(self, session, space, toggles) -> tuple[str, dict[str, float], list[dict]]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        theme = self._theme()
        sel = st.session_state.get(_SEL)
        fig = plt.figure(figsize=(11.5, 7.4))
        ax = fig.add_axes([0, 0, 1, 1])
        if space == "goal":
            interior, overlay = self._draw_goal(ax, theme, session, sel)
        else:
            interior, overlay = self._draw_pitch(ax, theme, session, sel, toggles)
        fig.patch.set_facecolor(theme.colors.get("bg", "#ECECEC"))
        # Resolve the interior→image mapping from the REAL axes transform, after a
        # draw applies the pitch's aspect='equal'. A raw xlim/ylim ratio would ignore
        # the equal-aspect letterbox and shift every tag off the pitch lines.
        fig.canvas.draw()
        bbox = self._interior_bbox(fig, ax, *interior)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return uri, bbox, overlay

    @staticmethod
    def _interior_bbox(fig, ax, ix0, ix1, iy0, iy1) -> dict[str, float]:
        """Playing-area rectangle as a fraction of the SAVED IMAGE (top-left origin).

        Computed from the live data→figure transform, so it stays exact under the
        equal-aspect letterboxing the pitch/goal renderers apply — a naive xlim/ylim
        ratio assumes the data fills the figure edge-to-edge and is off by the margin,
        which is what shifted a tagged corner off the corner. ``fig.canvas.draw()``
        must have run first so the aspect is applied to the transform. The image has
        no ``bbox_inches='tight'`` crop, so figure fraction == image fraction (y up →
        y down flip only)."""
        tf = fig.transFigure.inverted()
        fx0, fy0 = tf.transform(ax.transData.transform((ix0, iy0)))
        fx1, fy1 = tf.transform(ax.transData.transform((ix1, iy1)))
        lo_y, hi_y = min(fy0, fy1), max(fy0, fy1)
        return {"left": min(fx0, fx1), "right": max(fx0, fx1),
                "top": 1.0 - hi_y, "bottom": 1.0 - lo_y}

    def _draw_pitch(self, ax, theme, session, sel, toggles):
        from fap.visuals.pitch import DISPLAY_WIDTH, PitchFactory, get_spec
        pf = PitchFactory()
        pf.draw_pitch(ax, theme, get_spec("uefa"), vertical=False)
        pf.draw_overlays(ax, theme, vertical=False,
                         show_thirds=toggles.get("thirds"), show_lanes=toggles.get("channels"))
        c = theme.colors
        if toggles.get("halfway"):
            ax.plot([50, 50], [0, DISPLAY_WIDTH], color=c["lines"], lw=1.4, alpha=0.8)
        if toggles.get("grid"):
            import numpy as np
            for gx in np.linspace(0, 100, 11):
                ax.plot([gx, gx], [0, DISPLAY_WIDTH], color=c["grid"], lw=0.5, ls=":", alpha=0.5)
            for gy in np.linspace(0, DISPLAY_WIDTH, 7):
                ax.plot([0, 100], [gy, gy], color=c["grid"], lw=0.5, ls=":", alpha=0.5)
        xlim, ylim = (-4.0, 104.0), (-4.0, DISPLAY_WIDTH + 4.0)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
        team_color = {session.home_team: c["accent"], session.away_team: c["accent_2"],
                      "Neutral": c["grey"]}
        # batched drawing: ONE scatter for all points + ONE LineCollection for all
        # lines, so the canvas stays responsive with thousands of events.
        from matplotlib.collections import LineCollection
        segs, seg_cols, px, py, pc, endx, endy = [], [], [], [], [], [], []
        sel_pt = None
        overlay = []
        for e in session.events:
            if e.coordinate_space != "pitch" or e.x is None or e.y is None:
                continue
            col = team_color.get(e.team, c["accent"])
            selected = (e.id == sel)
            dx, dy = e.x, e.y / 100.0 * DISPLAY_WIDTH
            tag = tag_by_key(e.event_type)
            ifx, ify = TC.pitch_fraction_from_canonical(e.x, e.y)
            m = {"id": e.id, "ifx": ifx, "ify": ify, "kind": "pitch", "selected": selected}
            if tag and tag.geometry == "line" and e.x2 is not None and e.y2 is not None:
                d2x, d2y = e.x2, e.y2 / 100.0 * DISPLAY_WIDTH
                segs.append([(dx, dy), (d2x, d2y)]); seg_cols.append(col)
                endx.append(d2x); endy.append(d2y)
                m["ifx2"], m["ify2"] = TC.pitch_fraction_from_canonical(e.x2, e.y2)
            px.append(dx); py.append(dy); pc.append(col)
            if selected:
                sel_pt = (dx, dy)
            overlay.append(m)
        if segs:
            ax.add_collection(LineCollection(segs, colors=seg_cols, linewidths=2.0,
                                             alpha=0.9, zorder=5))
            ax.scatter(endx, endy, s=42, c=seg_cols, edgecolors="none", zorder=6)
        if px:
            ax.scatter(px, py, s=90, c=pc, edgecolors=c["bg"], linewidths=1.0, zorder=6)
        if sel_pt is not None:
            ax.scatter([sel_pt[0]], [sel_pt[1]], s=210, facecolors="none",
                       edgecolors=c["text"], linewidths=2.0, zorder=9)
        return (0.0, 100.0, 0.0, DISPLAY_WIDTH), overlay

    def _draw_goal(self, ax, theme, session, sel):
        from fap.visuals import goal as G
        G.draw_goal(ax, theme)
        xs, ys, is_goal, ov = [], [], [], []
        for e in session.events:
            if e.coordinate_space != "goal" or e.goal_x is None or e.goal_y is None:
                continue
            gx = e.goal_x / 100.0 * G.GOAL_WIDTH
            gy = e.goal_y / 100.0 * G.GOAL_HEIGHT
            goalish = e.event_type == "goal" or (e.outcome or "").lower() == "goal"
            xs.append(gx); ys.append(gy); is_goal.append(goalish)
            ifx, ify = TC.goal_fraction_from_canonical(e.goal_x, e.goal_y)
            ov.append({"id": e.id, "ifx": ifx, "ify": ify, "kind": "goal",
                       "selected": (e.id == sel)})
        if xs:
            G.draw_shots(ax, theme, xs=xs, ys=ys, is_goal=is_goal)
            for e, marker in zip([m for m in session.events
                                  if m.coordinate_space == "goal" and m.goal_x is not None], ov):
                if marker["selected"]:
                    gx = e.goal_x / 100.0 * G.GOAL_WIDTH
                    gy = e.goal_y / 100.0 * G.GOAL_HEIGHT
                    ax.scatter([gx], [gy], s=260, facecolors="none",
                               edgecolors=theme.colors["text"], linewidths=2.0, zorder=9)
        return (0.0, G.GOAL_WIDTH, 0.0, G.GOAL_HEIGHT), ov
