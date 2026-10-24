"""Set Piece Lab page — opponent set-piece analysis from a coding-matrix file + a routine & drill designer.

Its own import (xlsx / csv from Sportscode / Hudl matrix exports), separate from Match Lab and from the
database-backed Set Piece Analysis page. Analysis = :mod:`fap.setpiecelab` (interactive Plotly on screen,
every download a static 300-dpi PNG from :mod:`fap.setpiecelab.static`). Designer = a dedicated set-piece
canvas (``setpiece_designer``): players / ball / equipment, runs and deliveries that animate along their
curves, zones, text, steps with coaching notes; exports are rendered server-side by
:mod:`fap.setpiecelab.render` (step PNG, one-page coaching sheet PNG, GIF). Routines and drills are saved
with their coaching details as WorkspaceManager presets (kind ``setpiece_lab_library``).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import zipfile

import streamlit as st

from fap.core.plugin import PluginInfo
from fap.identity.roles import Role
from fap.theme import components as C
from fap.ui.page import Page, page_registry

SPL_FILE = "_spl_file"              # {"name", "team", "data"}
SPL_FILE_KIND = "setpiece_lab_file"
SPL_LIB_KIND = "setpiece_lab_library"
SPL_LIB_SESSION = "_spl_library"    # session fallback when presets are unavailable
SPL_DOC, SPL_LOAD, SPL_NONCE, SPL_CUR = "_spl_doc", "_spl_doc_load", "_spl_nonce", "_spl_cur"
CATEGORIES = ["Corner", "Free kick", "Throw-in", "Goal kick", "Penalty", "Kick-off", "Other"]


def _fname(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "chart"


@st.cache_resource(show_spinner=False, max_entries=8)
def _load(sig: str, _data: bytes, name: str, team: str):
    from fap.setpiecelab import load_setpieces
    return load_setpieces(_data, name, team=team)


@st.cache_data(show_spinner=False, max_entries=96)
def _static_png(sig: str, chart: str, args: tuple, theme_name: str, _sd) -> bytes:
    from fap.matchlab.charts import THEMES
    from fap.setpiecelab import static as S
    fn = {"overview": S.overview, "contests": S.contests, "landing": S.landing, "qualifiers": S.qualifiers,
          "cross": S.cross, "outcome_by": S.outcome_by, "card": S.report_card,
          "category": S.category_compare, "contact": S.contact_map}[chart]
    kw = dict(args)
    pos = kw.pop("_pos", ())
    return S.png(fn(_sd, *pos, theme=THEMES[theme_name], **kw), 300)


@st.cache_data(show_spinner=False, max_entries=300)
def _cached_png(key: str, _make) -> bytes:
    from fap.setpiecelab import static as S
    return S.png(_make(), 300)


TYPE_LABELS = {"Offensive Free kick": "Indirect free kicks", "Direct Offensive Free kick": "Direct free kicks",
               "Long Throw In": "Long throw-ins", "Throw In": "Throw-ins (short)", "Def Throw in": "Throw-ins against",
               "Defensive Free kick": "Free kicks against", "Offensive Corner": "Corners", "Defensive Corner": "Corners against",
               "Long GK": "Long goal kicks", "Penalty": "Penalties", "Penalty Against": "Penalties against"}


def _type_label(row: str) -> str:
    return TYPE_LABELS.get(row, row)


def _wm(shell):
    return getattr(shell, "wm", None)


def _show(fig, key: str) -> None:
    cfg = {"displaylogo": False, "scrollZoom": False, "displayModeBar": True,
           "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage"]}
    st.plotly_chart(fig, use_container_width=True, key=f"spl_{key}", theme=None, config=cfg)


def _png_button(label: str, data_fn, fname: str, key: str) -> None:
    try:
        data = data_fn()
    except Exception as exc:  # pragma: no cover
        st.caption(f"PNG unavailable: {exc}")
        return
    st.download_button(label, data, file_name=f"{_fname(fname)}.png", mime="image/png", key=f"spl_png_{key}",
                       icon=":material/download:")


def _static_fig(key: str, make, fname: str, bkey: str) -> None:
    """A static chart (cached PNG) on screen + its PNG download."""
    try:
        data = _cached_png(key, make)
    except Exception as exc:  # pragma: no cover
        st.caption(f"Chart unavailable: {exc}")
        return
    st.image(data, use_container_width=True)
    st.download_button("Download PNG", data, file_name=f"{_fname(fname)}.png", mime="image/png", key=f"spl_png_{bkey}",
                       icon=":material/download:")


_TONE = {"strength": ("#16a34a", "Strength"), "concern": ("#dc2626", "Watch"), "note": ("#3b82f6", "Note")}


def _render_insights(items, group: bool = False) -> None:
    import html as _h
    last = None
    for it in items:
        if group and it.section != last:
            st.markdown(f"##### {it.section}")
            last = it.section
        col, lab = _TONE.get(it.tone, _TONE["note"])
        st.markdown(f"<div style='border-left:4px solid {col};padding:6px 12px;margin:0 0 6px 0;border-radius:6px;"
                    f"background:rgba(127,127,127,0.07)'><span style='font-size:0.7rem;font-weight:700;color:{col};"
                    f"text-transform:uppercase;letter-spacing:.04em'>{lab}</span><br>{_h.escape(it.text)}</div>",
                    unsafe_allow_html=True)


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


# ---------------------------------------------------------------- designer state
def _load_doc(doc: dict, details: dict | None = None) -> None:
    st.session_state[SPL_DOC] = doc
    st.session_state[SPL_LOAD] = doc
    st.session_state[SPL_CUR] = 0
    st.session_state[SPL_NONCE] = st.session_state.get(SPL_NONCE, 0) + 1
    st.session_state["_spl_details"] = dict(details or {})


def _doc_sig(doc: dict) -> str:
    return hashlib.sha1(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@st.cache_data(show_spinner=False, max_entries=48)
def _render_step(sig: str, i: int, _doc: dict) -> bytes:
    from fap.setpiecelab import render as R
    return R.step_png(_doc, i, dpi=300)


@st.cache_data(show_spinner=False, max_entries=24)
def _render_sheet(sig: str, details_json: str, kind: str, _doc: dict) -> bytes:
    from fap.setpiecelab import render as R
    return R.sheet_png(_doc, json.loads(details_json), kind, dpi=220)


@st.cache_data(show_spinner=False, max_entries=60)
def _thumb(sig: str, _doc: dict) -> bytes:
    import matplotlib.pyplot as plt
    from fap.setpiecelab import render as R
    fig = R.step_figure(_doc, len(_doc["steps"]) - 1, width_in=4.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=70, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


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
                   "for your own routines and drills — steps, animation and coaching sheets.")
        self._source(shell, can_edit)
        f = st.session_state.get(SPL_FILE)
        sd = None
        if f:
            sig = hashlib.sha1(f["data"]).hexdigest()[:16] + "|" + f.get("team", "")
            try:
                sd = _load(sig, f["data"], f["name"], f.get("team", ""))
                sd._spl_sig = sig
            except Exception as exc:
                C.render_alert(f"Could not read this file: {exc}", "danger")
        if sd is not None:
            from fap.setpiecelab import deep as D
            if D.has_matches(sd):
                ms = D.matches(sd)
                pick = st.selectbox("Match", ["All matches"] + ms, key="spl_match",
                                    format_func=lambda m: m if m == "All matches" else
                                    f"{m}  ({int((sd.df['match'] == m).sum())} set pieces)",
                                    help="Every tab (and the report) shows the chosen match — or all matches together")
                if pick != "All matches":
                    sig0 = sd._spl_sig
                    sd = sd.subset(match=pick)
                    sd._spl_sig = f"{sig0}|m={pick}"
                sd._spl_match = pick
            else:
                sd._spl_match = ""
            C.render_alert(f"Analysed team: {sd.team} — every chart and map is from {sd.team}'s point of view. "
                           f"Attacking = {sd.team} taking the set piece · Defending = the opponent's set pieces "
                           f"against {sd.team} · “won” = {sd.team} won it, “Opponent won” = whichever team {sd.team} "
                           "played. Change the name in “Team in the file” if it is read wrongly.", "info")
        tabs = st.tabs(["Overview", "By type", "Insights", "Explorer", "Report", "Designer", "Library", "Data"])
        with tabs[0]:
            self._overview(sd) if sd else self._empty()
        with tabs[1]:
            self._by_type(sd) if sd else self._empty()
        with tabs[2]:
            self._insights(sd) if sd else self._empty()
        with tabs[3]:
            self._explorer(sd) if sd else self._empty()
        with tabs[4]:
            self._report(sd) if sd else self._empty()
        with tabs[5]:
            self._designer(shell, sd, can_edit)
        with tabs[6]:
            self._library(shell, can_edit)
        with tabs[7]:
            self._data(sd) if sd else self._empty()

    @staticmethod
    def _empty() -> None:
        C.render_empty_state("Upload a set-piece file to start",
                             "Coding-matrix export (.xlsx / .csv) with a 'Row' column (set-piece type), qualifier "
                             "columns and x_/y_ landing coordinates.", icon_name="upload")

    @staticmethod
    def _theme():
        from fap.matchlab.charts import THEMES
        name = st.session_state.get("spl_theme", "Dark")
        return name, THEMES[name]

    # ---- import -----------------------------------------------------------------
    def _source(self, shell, can_edit: bool) -> None:
        c = st.columns([3, 1.5, 1.3, 2])
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
        c[2].radio("Chart theme", ["Dark", "Light"], key="spl_theme")
        wm = _wm(shell)
        if wm is not None:
            try:
                saved = wm.list_presets(shell.user, kind=SPL_FILE_KIND)
            except Exception:
                saved = []
            if saved:
                names = {getattr(p, "name", "file"): p for p in saved}
                pick = c[3].selectbox("Saved files", ["—"] + list(names), key="spl_saved")
                if pick != "—" and c[3].button("Open", key="spl_open"):
                    doc = getattr(names[pick], "document", None) or {}
                    st.session_state[SPL_FILE] = {"name": doc.get("name") or pick, "team": doc.get("team", ""),
                                                  "data": base64.b64decode(doc.get("data_b64", ""))}
                    st.rerun()
            if cur and can_edit and c[3].button("Save file", key="spl_savefile"):
                try:
                    wm.save_preset(shell.user, kind=SPL_FILE_KIND, name=cur["name"].rsplit(".", 1)[0],
                                   document={"name": cur["name"], "team": cur.get("team", ""),
                                             "data_b64": base64.b64encode(cur["data"]).decode("ascii")})
                    st.toast("File saved.")
                except Exception as exc:
                    st.toast(f"Save failed: {exc}")

    # ---- overview -----------------------------------------------------------------
    def _overview(self, sd) -> None:
        from fap.setpiecelab import analytics as A
        from fap.setpiecelab import charts as CH
        from fap.setpiecelab import static as S
        tn, th = self._theme()
        ov = A.overview(sd)
        att, de = ov[ov["Phase"] == "Attacking"], ov[ov["Phase"] == "Defending"]
        g = lambda d, c: int(d[c].sum()) if c in d else 0  # noqa: E731
        C.render_metric_row([
            C.metric_card_html("Set pieces coded", str(int(ov["Count"].sum())), hint=f"{len(ov)} types"),
            C.metric_card_html(f"{sd.team} taking", str(g(att, "Count")),
                               hint=f"{g(att, 'Shots')} shots · {g(att, 'Goals')} goals"),
            C.metric_card_html(f"{sd.team} defending", str(g(de, "Count")),
                               hint=f"{g(de, 'Shots')} shots against · {g(de, 'Goals')} conceded"),
            C.metric_card_html("With a landing point", str(g(ov, "With landing point")), hint="x / y coded"),
        ])
        if sd.notes:
            st.caption(" · ".join(sd.notes))
        from fap.setpiecelab import deep as D
        from fap.setpiecelab import deep_static as X
        if D.has_matches(sd) and getattr(sd, "_spl_match", "") == "All matches":
            st.markdown("#### Match by match")
            _static_fig(f"{sd._spl_sig}|{tn}|matches", lambda: X.match_chart(sd, f"{sd.team} — set pieces match by match", theme=th),
                        f"{sd.team}_match_by_match", "ov_matches")
            bm = D.by_match(sd)
            st.dataframe(bm.round(1), use_container_width=True, hide_index=True)
        tabs = st.tabs([f"Attacking ({g(att, 'Count')})", f"Defending ({g(de, 'Count')})"])
        for phase, tab in zip(("Attacking", "Defending"), tabs):
            with tab:
                sub = sd.subset(phase=phase)
                if sub.df.empty:
                    st.info(f"No {phase.lower()} set pieces coded.")
                    continue
                psig = f"{sd._spl_sig}|{phase}"
                st.markdown("#### Key findings")
                for row, text in A.insights(sub):
                    st.markdown(f"- **{_type_label(row)}** — {text}" if row != "Headline" else f"- **{text}**")
                c1, c2 = st.columns(2)
                with c1:
                    _show(CH.overview_bars(sub, theme=th), f"ov_bars_{phase}")
                    _png_button("Download PNG", lambda sub=sub, psig=psig: _static_png(psig, "overview", (), tn, sub),
                                f"{sd.team}_{phase}_set_pieces", f"ov_{phase}")
                with c2:
                    _show(CH.contest_rates(sub, theme=th), f"ov_contest_{phase}")
                    _png_button("Download PNG", lambda sub=sub, psig=psig: _static_png(psig, "contests", (), tn, sub),
                                f"{sd.team}_{phase}_first_contact_second_ball", f"ct_{phase}")
                st.dataframe(A.overview(sub), use_container_width=True, hide_index=True)
        st.markdown("#### Opposition report cards")
        st.caption("One PNG page per set piece — landing map, key numbers and every qualifier.")
        if st.button("Prepare all report cards (PNG, ZIP)", key="spl_cards"):
            buf = io.BytesIO()
            with st.spinner("Rendering report cards…"), zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for i, r in enumerate(sd.rows, start=1):
                    z.writestr(f"{i:02d}_{_fname(sd.team)}_{_fname(r)}.png",
                               _static_png(sd._spl_sig, "card", (("_pos", (r,)),), tn, sd))
            st.session_state["_spl_cards"] = (sd._spl_sig + tn, buf.getvalue())
        got = st.session_state.get("_spl_cards")
        if got and got[0] == sd._spl_sig + tn:
            st.download_button("Download report cards (ZIP of PNGs)", got[1], file_name=f"{_fname(sd.team)}_set_piece_cards.zip",
                               mime="application/zip", key="spl_cards_dl", icon=":material/download:")

    # ---- explorer -----------------------------------------------------------------
    def _by_type(self, sd) -> None:
        """Attacking | Defending → Corners / Free kicks / Throw-ins … → each coded type (direct / indirect,
        long / short …) with a left / right filter, a left-vs-right comparison and the takers."""
        from fap.setpiecelab import analytics as A
        from fap.setpiecelab import charts as CH
        from fap.setpiecelab import static as S
        tn, th = self._theme()
        sig = sd._spl_sig
        phases = [p for p in ("Attacking", "Defending") if (sd.df["phase"] == p).any()]
        if not phases:
            st.info("No set-piece types recognised.")
            return
        top = st.tabs([f"{p} ({int((sd.df['phase'] == p).sum())})" for p in phases])
        for phase, ptab in zip(phases, top):
            with ptab:
                sub = sd.subset(phase=phase)
                cats = A.categories(sub)
                ctabs = st.tabs([f"{A.CATEGORY_PLURAL.get(c, c)} ({len(sub.of(category=c))})" for c in cats])
                for cat, ctab in zip(cats, ctabs):
                    with ctab:
                        rows = [r for r in sub.rows if sub.of(row=r)["category"].iloc[0] == cat]
                        rows.sort(key=lambda r: (_type_label(r), r))
                        labels = [f"{_type_label(r)} ({len(sub.of(row=r))})" for r in rows]
                        if len(rows) > 1:
                            labels.append(" vs ".join(_type_label(r) for r in rows))
                        inner = st.tabs(labels) if len(labels) > 1 else [st.container()]
                        for r, t in zip(rows, inner):
                            with t:
                                self._type_section(sub, r, tn, th, sig)
                        if len(rows) > 1:
                            with inner[-1]:
                                groups = [(_type_label(r), sub.of(row=r)) for r in rows]
                                title = f"{sd.team} — " + " vs ".join(lab for lab, _ in groups)
                                lines = [f"**{_type_label(r)}** — " + text for r_, text in A.insights(sub) for r in rows if r_ == r]
                                for line in lines:
                                    st.markdown(f"- {line}")
                                _show(CH.compare_groups(sub, groups, title, theme=th), f"cmp_{phase}_{cat}")
                                _png_button("Download PNG", lambda groups=groups, title=title, lines=lines: _cached_png(
                                    f"{sig}|{tn}|cmp|{phase}|{cat}",
                                    lambda: S.compare_groups(sub, groups, title, lines=lines, theme=th)),
                                    _fname(title), f"cmp_{phase}_{cat}")
                                if phase == "Attacking" and (sub.of(category=cat)["taker"] != "").any():
                                    tt = f"{sd.team} — {A.CATEGORY_PLURAL.get(cat, cat).lower()}: takers"
                                    _show(CH.takers_chart(sub, sub.of(category=cat), tt, theme=th), f"cmp_tk_{cat}")
                                    _png_button("Download PNG", lambda cat=cat, tt=tt: _cached_png(
                                        f"{sig}|{tn}|tk|{cat}", lambda: S.takers_chart(sub, sub.of(category=cat), tt, theme=th)),
                                        _fname(tt), f"cmp_tk_{cat}")

    def _type_section(self, sd_all, row: str, tn: str, th, sig: str) -> None:
        """Everything for ONE coded set-piece type, with a left / right filter."""
        from fap.setpiecelab import analytics as A
        from fap.setpiecelab import charts as CH
        from fap.setpiecelab import static as S
        g_all = sd_all.of(row=row)
        att = g_all["phase"].iloc[0] == "Attacking"
        k = f"{row}".replace(" ", "_")
        sides = [s_ for s_ in ("Left", "Right") if (g_all["side"] == s_).any()]
        side = "All"
        if sides:
            side = st.radio("Side", ["All"] + sides, horizontal=True, key=f"bt_side_{k}",
                            format_func=lambda v: v if v == "All" else f"{v} ({int((g_all['side'] == v).sum())})")
        sd = sd_all.subset(side=side) if side != "All" else sd_all
        ssig = f"{sig}|{side}"
        g = sd.of(row=row)
        label = _type_label(row) + ("" if side == "All" else f" · {side.lower()} side")
        sm = A.summarise(sd, g)
        c1 = f"{sm['1st contact won']}/{sm['1st contact decided']}" if sm.get("1st contact decided") else "–"
        c2 = f"{sm['2nd ball won']}/{sm['2nd ball decided']}" if sm.get("2nd ball decided") else "–"
        C.render_metric_row([
            C.metric_card_html(label, str(sm["Count"]), hint=f"{sd.team} {'taking' if att else 'defending'} · coded as {row}"),
            C.metric_card_html("1st contact won", c1, hint=f"{sd.team} won it"),
            C.metric_card_html("2nd ball won", c2, hint=f"{sd.team} won it"),
            C.metric_card_html("Danger " + ("created" if att else "conceded"), str(sm.get("Dangerous (goal/shot/threat)", 0)),
                               hint=f"{sm.get('Shots', 0)} shot(s) · {sm.get('Goals', 0)} goal(s)"),
        ])
        from fap.setpiecelab import deep as D
        from fap.setpiecelab import deep_static as X
        st.markdown("##### Key findings")
        _render_insights(D.type_insights(sd, row, g, section=label))
        direct = g["category"].iloc[0] == "Penalty" or g["variant"].iloc[0] == "Direct"
        if not direct and g["x"].notna().any():
            c1, c2 = st.columns(2)
            with c1:
                _static_fig(f"{ssig}|{tn}|deliv|{row}", lambda: X.delivery_map(sd, g, f"{sd.team} · {label} — deliveries", theme=th),
                            f"{sd.team}_{label}_deliveries", f"bt_deliv_{k}")
            with c2:
                _static_fig(f"{ssig}|{tn}|zones|{row}", lambda: X.zone_map(sd, g, f"{sd.team} · {label} — delivery zones", theme=th),
                            f"{sd.team}_{label}_zones", f"bt_zones_{k}")
            zt = D.zone_table(sd, g)
            if len(zt):
                st.dataframe(zt.round(0), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            if not direct:                              # direct free kicks / penalties have no contacts
                _static_fig(f"{ssig}|{tn}|funnel|{row}", lambda: X.funnel_chart(sd, g, f"{sd.team} · {label} — from delivery to goal",
                                                                             att, theme=th), f"{sd.team}_{label}_funnel", f"bt_funnel_{k}")
        with c2:
            if g["is_shot"].any():
                _static_fig(f"{ssig}|{tn}|shots|{row}", lambda: X.shots_chart(sd, g, f"{sd.team} · {label} — shots", theme=th),
                            f"{sd.team}_{label}_shots", f"bt_shots_{k}")
        if (g["fc_won_by"] != "").any() or (g["fc_lost_by"] != "").any():
            _static_fig(f"{ssig}|{tn}|players|{row}", lambda: X.players_chart(sd, g, f"{sd.team} · {label} — 1st-contact players",
                                                                           theme=th), f"{sd.team}_{label}_players", f"bt_pl_{k}")
        ea = D.exec_area(sd, g)
        if len(ea):
            with st.expander("Execution × target — counts, 1st contact, shots, danger", expanded=False):
                st.dataframe(ea.round(0), use_container_width=True, hide_index=True)
        if side == "All" and len(sides) == 2:
            st.markdown(f"##### {_type_label(row)} — left vs right")
            for line in A.side_insights(sd_all, row):
                st.markdown(f"- {line}")
            groups = [(s_, g_all[g_all["side"] == s_]) for s_ in sides]
            title = f"{sd.team} — {_type_label(row)}: left vs right"
            _show(CH.compare_groups(sd_all, groups, title, theme=th), f"bt_lr_{k}")
            _png_button("Download PNG", lambda: _cached_png(
                f"{sig}|{tn}|lr|{row}", lambda: S.compare_groups(sd_all, groups, title, lines=A.side_insights(sd_all, row),
                                                                theme=th)), _fname(title), f"bt_lr_{k}")
        if att and (g["taker"] != "").any():
            st.markdown(f"##### {label} — takers")
            title = f"{sd.team} — {label}: takers"
            _show(CH.takers_chart(sd, g, title, theme=th), f"bt_tk_{k}")
            _png_button("Download PNG", lambda: _cached_png(f"{ssig}|{tn}|tk|{row}", lambda: S.takers_chart(sd, g, title, theme=th)),
                        _fname(title), f"bt_tk_{k}")
        st.markdown(f"##### {label} — how it is {'taken' if att else 'defended'}")
        _show(CH.qualifier_grid(sd, row, theme=th), f"bt_grid_{k}")
        _png_button("Download PNG", lambda: _static_png(ssig, "qualifiers", (("_pos", (row,)),), tn, sd),
                    f"{sd.team}_{label}_details", f"bt_grid_{k}")
        area = A._col(sd, "Area")
        if area and A.result_col(sd) and (g[area] != "").any():
            st.markdown(f"##### {label} — result by target area")
            _show(CH.outcome_by(sd, row, area, theme=th), f"bt_ob_{k}")
            _png_button("Download PNG", lambda: _static_png(ssig, "outcome_by", (("_pos", (row, area)),), tn, sd),
                        f"{sd.team}_{label}_result_by_area", f"bt_ob_{k}")
        if "goal_x_m" in g and g["goal_x_m"].notna().any():
            st.markdown(f"##### {label} — placement in the goal")
            _show(CH.goal_placement(sd, row, theme=th), f"bt_goal_{k}")
            _png_button("Download PNG", lambda: _cached_png(f"{ssig}|{tn}|goal|{row}", lambda: S.goal_placement(sd, row, theme=th)),
                        f"{sd.team}_{label}_goal_placement", f"bt_goal_{k}")
        maps = False
        for which, title in ((1, "1st contact outcome — where the ball lands"),
                             (2, "2nd contact outcome — where the second ball is played")):
            if len(A.contact_points(sd, row, which)):
                maps = True
                st.markdown(f"##### {label} — {title}")
                _show(CH.contact_map(sd, row, which, theme=th), f"bt_c{which}_{k}")
                _png_button("Download PNG", lambda w=which: _static_png(ssig, "contact", (("_pos", (row, w)),), tn, sd),
                            f"{sd.team}_{label}_contact{which}_outcome", f"bt_c{which}_{k}")
        if g["x"].notna().any():
            st.markdown(f"##### {label} — landing points & second balls (all results)")
            rc = A.result_col(sd)
            _show(CH.landing_map(sd, row, color_by=rc, theme=th), f"bt_map_{k}")
            _png_button("Download PNG", lambda: _static_png(ssig, "landing", (("_pos", (row,)), ("color_by", rc)), tn, sd),
                        f"{sd.team}_{label}_landing", f"bt_map_{k}")
        elif not maps:
            st.caption("No pitch coordinates coded for this type — no pitch maps.")
        with st.expander(f"Every {label.lower()}", expanded=False):
            cols = ["instance", "side", "taker"] + [q for q in sd.qualifiers if (g[q] != "").any()] + ["x", "y", "sb_x", "sb_y"]
            st.dataframe(g[[c for c in cols if c in g.columns]], use_container_width=True, hide_index=True)

    def _insights(self, sd) -> None:
        from fap.setpiecelab import deep as D
        st.caption("Every finding is counted from the file — rates are judged (strength / point to watch) only from "
                   f"{D.MIN_N}+ set pieces; smaller groups are marked “small sample”.")
        items = D.headline_insights(sd)
        _render_insights(items, group=True)
        phases = [p for p in ("Attacking", "Defending") if (sd.df["phase"] == p).any()]
        for ph, t in zip(phases, st.tabs(phases)):
            with t:
                for ph_, row, g in D.groups(sd):
                    if ph_ == ph:
                        st.markdown(f"##### {D.type_label(row)} ({len(g)})")
                        _render_insights(D.type_insights(sd, row, g, section=D.type_label(row)))
        if D.has_matches(sd) and getattr(sd, "_spl_match", "") == "All matches":
            st.markdown("#### Match by match")
            for m in D.matches(sd):
                with st.expander(m, expanded=False):
                    _render_insights(D.match_insights(sd, m))

    def _report(self, sd) -> None:
        from fap.setpiecelab import deep as D
        from fap.setpiecelab.report import DEFAULT_SECTIONS, SECTIONS, ReportOptions, build_report
        scope = getattr(sd, "_spl_match", "") or ""
        st.caption("A PDF (A4 landscape): the general numbers first, then every set-piece type on its own pages — "
                   + ("for " + (f"the match vs {scope}" if scope and scope != "All matches" else "all matches") + ". "
                      if scope else "") + "Choose what goes in.")
        c = st.columns([2, 2, 1])
        title = c[0].text_input("Title", value=f"{sd.team} — set pieces", key="spl_rep_title")
        sub_default = (f"vs {scope}" if scope and scope != "All matches" else
                       f"{len(D.matches(sd))} matches" if D.has_matches(sd) else f"{len(sd.df)} set pieces")
        subtitle = c[1].text_input("Subtitle", value=sub_default, key="spl_rep_sub")
        theme = c[2].radio("Theme", ["Dark", "Light"], key="spl_rep_theme")
        footer = st.text_input("Footer (bottom of every page — leave empty for none)", value=f"{sd.team} · Performance Analysis",
                               key="spl_rep_footer")
        c = st.columns(2)
        inc_ins = c[0].checkbox("Include insights (blue = attacking · red = defending)", value=True, key="spl_rep_ins")
        phases = c[1].multiselect("Phases", ["Attacking", "Defending"], default=["Attacking", "Defending"], key="spl_rep_ph")
        types = st.multiselect("Set-piece types (empty = all)", sd.rows, key="spl_rep_types", format_func=D.type_label)
        secs = st.multiselect("Charts per set-piece type", list(SECTIONS), default=list(DEFAULT_SECTIONS), key="spl_rep_secs",
                              help="'sides' adds a left and a right delivery map per type (longer report)")
        per_match = match_ins = False
        if D.has_matches(sd) and scope == "All matches":
            c = st.columns(2)
            per_match = c[0].checkbox("Match-by-match page", value=True, key="spl_rep_pm")
            match_ins = c[1].checkbox("Insights per match", value=False, key="spl_rep_mi", disabled=not inc_ins)
        opts = ReportOptions(title=title, subtitle=subtitle, footer=footer, theme=theme, include_insights=inc_ins,
                             phases=tuple(phases), per_match=per_match, match_insights=match_ins and inc_ins,
                             sections=tuple(secs), types=tuple(types))
        stamp = json.dumps([sd._spl_sig, title, subtitle, footer, theme, inc_ins, phases, types, secs, per_match, match_ins])
        if st.button("Build PDF report", type="primary", key="spl_rep_build", icon=":material/picture_as_pdf:"):
            with st.spinner("Building the report — every chart is rendered, this takes up to a minute…"):
                data = build_report(sd, opts)
            st.session_state["_spl_report"] = (stamp, data, list(opts.log))
        got = st.session_state.get("_spl_report")
        if got and got[0] == stamp:
            st.download_button("Download PDF report", got[1], file_name=f"{_fname(title)}.pdf", mime="application/pdf",
                               key="spl_rep_dl", type="primary", icon=":material/download:")
            st.caption(f"{len(got[1]) / 1e6:.1f} MB")
            if got[2]:
                st.warning("Some charts were skipped: " + "; ".join(got[2]))

    def _explorer(self, sd) -> None:
        from fap.setpiecelab import analytics as A
        from fap.setpiecelab import charts as CH
        from fap.setpiecelab import designer as D
        tn, th = self._theme()
        sig = sd._spl_sig
        c = st.columns([2, 1.5, 1.5])
        row = c[0].selectbox("Set piece", sd.rows, key="spl_row", format_func=lambda r: f"{r}  ({len(sd.of(row=r))})")
        quals = [q for q in sd.qualifiers if (sd.of(row=row)[q] != "").any()]
        color_by = c[1].selectbox("Colour landing points by", ["—"] + quals, key="spl_colorby",
                                  index=(1 + quals.index("Result")) if "Result" in quals else 0)
        cb = None if color_by == "—" else color_by
        with c[2]:
            st.write("")
            _png_button("Report card (PNG)", lambda: _static_png(sig, "card", (("_pos", (row,)),), tn, sd),
                        f"{sd.team}_{row}_report_card", "x_card")
        for r_, text in A.insights(sd):
            if r_ == row:
                st.info(text)
        _show(CH.landing_map(sd, row, color_by=cb, theme=th), "map")
        _png_button("Download PNG", lambda: _static_png(sig, "landing", (("_pos", (row,)), ("color_by", cb)), tn, sd),
                    f"{sd.team}_{row}_landing", "x_map")
        for which, title in ((1, "1st contact outcome"), (2, "2nd contact outcome (second ball)")):
            if len(A.contact_points(sd, row, which)):
                _show(CH.contact_map(sd, row, which, theme=th), f"x_contact{which}")
                _png_button("Download PNG", lambda w=which: _static_png(sig, "contact", (("_pos", (row, w)),), tn, sd),
                            f"{sd.team}_{row}_contact{which}_outcome", f"x_contact{which}")
        _show(CH.qualifier_grid(sd, row, theme=th), "grid")
        _png_button("Download PNG", lambda: _static_png(sig, "qualifiers", (("_pos", (row,)),), tn, sd),
                    f"{sd.team}_{row}_details", "x_grid")
        if len(quals) >= 2:
            cc = st.columns(2)
            a = cc[0].selectbox("Cross-tab: rows", quals, key="spl_ct_a")
            b = cc[1].selectbox("Cross-tab: columns", [q for q in quals if q != a], key="spl_ct_b")
            c1, c2 = st.columns(2)
            with c1:
                _show(CH.cross_heat(sd, row, a, b, theme=th), "ct")
                _png_button("Download PNG", lambda: _static_png(sig, "cross", (("_pos", (row, a, b)),), tn, sd),
                            f"{sd.team}_{row}_{a}_x_{b}", "x_ct")
            out_c = A._col(sd, "Outcome")
            with c2:
                if out_c and a != out_c:
                    _show(CH.outcome_by(sd, row, a, theme=th), "ob")
                    _png_button("Download PNG", lambda: _static_png(sig, "outcome_by", (("_pos", (row, a)),), tn, sd),
                                f"{sd.team}_{row}_outcome_by_{a}", "x_ob")
        st.markdown("**Every instance**")
        cols = ["instance"] + quals + ["x", "y", "notes"]
        st.dataframe(sd.of(row=row)[[c_ for c_ in cols if c_ in sd.df.columns]], use_container_width=True, hide_index=True)
        if sd.of(row=row)["x"].notna().any() and st.button("Open their pattern in the Designer", key="spl_to_designer",
                                                            icon=":material/edit:"):
            _load_doc(D.pattern_doc(sd, row))
            st.success("Loaded — open the Designer tab.")

    # ---- designer -----------------------------------------------------------------
    def _designer(self, shell, sd, can_edit: bool) -> None:
        from fap.setpiecelab import designer as D
        from fap.ui.builtin.setpiece_designer_canvas import setpiece_designer
        if SPL_DOC not in st.session_state:
            _load_doc(D.template(next(iter(D.TEMPLATES))))
        c = st.columns([1.3, 3, 1])
        src = c[0].selectbox("Start from", ["Template", "Opponent data", "Blank board"], key="spl_src")
        if src == "Template":
            name = c[1].selectbox("Template", list(D.TEMPLATES), key="spl_tpl")
            loader = lambda: D.template(name)  # noqa: E731
        elif src == "Opponent data":
            rows = [r for r in sd.rows if sd.of(row=r)["x"].notna().any()] if sd is not None else []
            if not rows:
                c[1].selectbox("Their set piece", ["Upload a set-piece file with landing points first"], disabled=True,
                               key="spl_pat_none")
                loader = None
            else:
                row = c[1].selectbox("Their set piece", rows, key="spl_pat_row")
                loader = lambda: D.pattern_doc(sd, row)  # noqa: E731
        else:
            view = c[1].selectbox("Pitch view", ["third", "half", "full"], key="spl_blank_view",
                                  format_func={"third": "Final third", "half": "Half pitch", "full": "Full pitch"}.get)
            loader = lambda: D.blank(view)  # noqa: E731
        with c[2]:
            st.write("")
            if st.button("Load", key="spl_load", use_container_width=True, type="primary",
                         disabled=loader is None or not can_edit):
                _load_doc(loader())
                st.rerun()

        ok, res = setpiece_designer(doc=st.session_state[SPL_LOAD], load_nonce=st.session_state[SPL_NONCE],
                                    editable=can_edit, key="spl_designer", cur=st.session_state.get(SPL_CUR, 0))
        if not ok:
            C.render_alert("The designer could not load in this browser.", "danger")
            return
        if res and res.get("nonce") == st.session_state[SPL_NONCE]:
            st.session_state[SPL_DOC] = res["doc"]
            st.session_state[SPL_CUR] = res["cur"]
        doc = st.session_state[SPL_DOC]
        sig = _doc_sig(doc)
        n = len(doc["steps"])

        st.markdown("##### Export")
        e = st.columns([1.2, 1.2, 1.2, 1.4])
        with e[0]:
            i = st.selectbox("Step", list(range(n)), index=min(st.session_state.get(SPL_CUR, 0), n - 1),
                             key=f"spl_exp_step_{sig}", label_visibility="collapsed",
                             format_func=lambda k: f"{k + 1} · {doc['steps'][k]['name'] or 'Step'}")
            _png_button("Step PNG", lambda: _render_step(sig, i, doc), f"{doc['name']}_step{i + 1}", f"step_{sig}_{i}")
        with e[1]:
            if st.button("Prepare all steps (ZIP)", key="spl_zip", use_container_width=True):
                buf = io.BytesIO()
                with st.spinner("Rendering steps…"), zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for k in range(n):
                        z.writestr(f"{k + 1:02d}_{_fname(doc['steps'][k]['name'] or 'step')}.png", _render_step(sig, k, doc))
                st.session_state["_spl_zip"] = (sig, buf.getvalue())
            got = st.session_state.get("_spl_zip")
            if got and got[0] == sig:
                st.download_button("Download steps (ZIP of PNGs)", got[1], file_name=f"{_fname(doc['name'])}_steps.zip",
                                   mime="application/zip", key="spl_zip_dl", icon=":material/download:",
                                   use_container_width=True)
        details = st.session_state.get("_spl_details", {})
        kind = "drill" if (doc.get("meta") or {}).get("kind") == "drill" else "routine"
        with e[2]:
            _png_button("Coaching sheet (PNG)", lambda: _render_sheet(sig, json.dumps(details, sort_keys=True), kind, doc),
                        f"{doc['name']}_sheet", f"sheet_{sig}")
        with e[3]:
            if st.button("Prepare animation (GIF)", key="spl_gif", use_container_width=True, disabled=n < 2):
                from fap.setpiecelab import render as R
                with st.spinner("Rendering animation…"):
                    st.session_state["_spl_gif"] = (sig, R.gif_bytes(doc))
            got = st.session_state.get("_spl_gif")
            if got and got[0] == sig:
                st.download_button("Download GIF", got[1], file_name=f"{_fname(doc['name'])}.gif", mime="image/gif",
                                   key="spl_gif_dl", icon=":material/download:", use_container_width=True)

        self._save_form(shell, doc, can_edit)

    def _save_form(self, shell, doc: dict, can_edit: bool) -> None:
        from fap.setpiecelab import designer as D
        meta = doc.get("meta") or {}
        det = st.session_state.get("_spl_details", {})
        bk = str(st.session_state[SPL_NONCE])     # fields reset when another board is loaded
        with st.expander("Save to the library — routine or drill, with coaching details", expanded=False,
                         icon=":material/bookmark_add:"):
            kind = st.radio("Save as", ["Routine", "Drill"], horizontal=True, key=f"spl_kind_{bk}",
                            index=1 if meta.get("kind") == "drill" else 0)
            if kind == "Routine":
                cc = st.columns(3)
                cat0 = det.get("category") or meta.get("category")
                d = {"title": cc[0].text_input("Title", value=det.get("title") or doc["name"], key=f"spl_r_title_{bk}"),
                     "category": cc[1].selectbox("Set piece", CATEGORIES, key=f"spl_r_cat_{bk}",
                                                 index=CATEGORIES.index(cat0) if cat0 in CATEGORIES else 6),
                     "phase": cc[2].selectbox("Phase", ["Attacking", "Defending"], key=f"spl_r_phase_{bk}",
                                              index=1 if (det.get("phase") or meta.get("phase")) == "Defending" else 0)}
                cc = st.columns(3)
                sides = ["Any", "Left", "Right", "Central"]
                d["side"] = cc[0].selectbox("Side", sides, index=sides.index(det["side"]) if det.get("side") in sides else 0,
                                            key=f"spl_r_side_{bk}")
                d["target"] = cc[1].text_input("Target zone", value=det.get("target", ""), key=f"spl_r_target_{bk}")
                d["code_word"] = cc[2].text_input("Code word / signal", value=det.get("code_word", ""), key=f"spl_r_code_{bk}")
                cc = st.columns(2)
                d["taker"] = cc[0].text_input("Taker(s)", value=det.get("taker", ""), key=f"spl_r_taker_{bk}")
                d["key_roles"] = cc[1].text_input("Key roles (who does what)", value=det.get("key_roles", ""),
                                                  key=f"spl_r_roles_{bk}")
                d["coaching_points"] = st.text_area("Coaching points", value=det.get("coaching_points", ""), key=f"spl_r_cp_{bk}")
                d["trigger"] = st.text_input("When to use it (trigger / vs which defence)", value=det.get("trigger", ""),
                                             key=f"spl_r_trig_{bk}")
                d["video_link"] = st.text_input("Video link", value=det.get("video_link", ""), key=f"spl_r_video_{bk}")
                d["notes"] = st.text_area("Notes", value=det.get("notes", ""), key=f"spl_r_notes_{bk}")
                lib = D.routine_doc(doc, d)
            else:
                cc = st.columns(3)
                focus = ["Attacking corners", "Defending corners", "Free kicks", "Throw-ins", "Second balls", "Goal kicks",
                         "Penalties", "Other"]
                ints = ["Low", "Medium", "High"]
                d = {"title": cc[0].text_input("Title", value=det.get("title") or doc["name"], key=f"spl_d_title_{bk}"),
                     "focus": cc[1].selectbox("Focus", focus, key=f"spl_d_focus_{bk}",
                                              index=focus.index(det["focus"]) if det.get("focus") in focus else 0),
                     "intensity": cc[2].selectbox("Intensity", ints, key=f"spl_d_int_{bk}",
                                                  index=ints.index(det["intensity"]) if det.get("intensity") in ints else 1)}
                cc = st.columns(3)
                d["players"] = cc[0].text_input("Players", value=det.get("players", ""), key=f"spl_d_players_{bk}")
                d["area"] = cc[1].text_input("Area size", value=det.get("area", ""), key=f"spl_d_area_{bk}")
                d["duration_min"] = cc[2].text_input("Duration (min)", value=det.get("duration_min", ""), key=f"spl_d_dur_{bk}")
                d["objective"] = st.text_input("Objective", value=det.get("objective", ""), key=f"spl_d_obj_{bk}")
                d["setup"] = st.text_area("Setup", value=det.get("setup", ""), key=f"spl_d_setup_{bk}")
                d["instructions"] = st.text_area("Instructions / rules", value=det.get("instructions", ""), key=f"spl_d_instr_{bk}")
                d["coaching_points"] = st.text_area("Coaching points", value=det.get("coaching_points", ""), key=f"spl_d_cp_{bk}")
                d["progressions"] = st.text_area("Progressions / variations", value=det.get("progressions", ""),
                                                 key=f"spl_d_prog_{bk}")
                d["video_link"] = st.text_input("Video link", value=det.get("video_link", ""), key=f"spl_d_video_{bk}")
                lib = D.drill_doc(doc, d)
            st.session_state["_spl_details"] = d
            if st.button(f"Save {kind.lower()}", key=f"spl_save_lib_{bk}", type="primary", disabled=not can_edit):
                st.toast(_lib_save(shell, d.get("title") or doc["name"], lib))

    # ---- library -----------------------------------------------------------------
    def _library(self, shell, can_edit: bool) -> None:
        from fap.setpiecelab import designer as D
        items = [it for it in _lib_list(shell) if it["doc"].get("v") == 2]
        if not items:
            st.info("No routines or drills saved yet — design one in the Designer tab and save it.")
            return
        c = st.columns([2, 2])
        kind = c[0].radio("Show", ["All", "Routines", "Drills"], horizontal=True, key="spl_lib_kind")
        q = c[1].text_input("Search", key="spl_lib_q")
        shown = []
        for it in items:
            doc = it["doc"]
            k = doc.get("kind", "routine")
            if kind == "Routines" and k != "routine" or kind == "Drills" and k != "drill":
                continue
            text = " ".join(str(v) for v in doc.get("details", {}).values()).lower()
            if q and q.lower() not in text + it["name"].lower():
                continue
            shown.append(it)
        for start in range(0, len(shown), 3):
            cols = st.columns(3)
            for it, col in zip(shown[start:start + 3], cols):
                doc = it["doc"]
                board = D.board_of(doc)
                if board is None:
                    continue
                det = doc.get("details", {})
                sig = _doc_sig(board)
                with col, st.container(border=True):
                    st.image(_thumb(sig, board), use_container_width=True)
                    st.markdown(f"**{it['name']}**  \n<span style='opacity:.7;font-size:.85em'>"
                                f"{'Drill' if doc.get('kind') == 'drill' else 'Routine'} · {D.summary_text(doc)} · "
                                f"{len(board['steps'])} steps</span>", unsafe_allow_html=True)
                    b = st.columns(3)
                    if b[0].button("Open", key=f"spl_lib_open_{it['id']}", use_container_width=True):
                        _load_doc(board, det)
                        st.success("Loaded into the Designer tab.")
                    with b[1]:
                        _png_button("Sheet", lambda: _render_sheet(sig, json.dumps(det, sort_keys=True),
                                                                   doc.get("kind", "routine"), board),
                                    f"{it['name']}_sheet", f"lib_{it['id']}")
                    if can_edit and b[2].button("Delete", key=f"spl_lib_del_{it['id']}", use_container_width=True):
                        _lib_delete(shell, it)
                        st.rerun()

    # ---- data -----------------------------------------------------------------
    def _data(self, sd) -> None:
        df = sd.df.drop(columns=["points"]).copy()
        df["points"] = sd.df["points"].map(lambda p: "; ".join(f"{x:g},{y:g}" for x, y in p))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download tidy table (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{sd.team}_set_pieces_tidy.csv", mime="text/csv", key="spl_csv")
