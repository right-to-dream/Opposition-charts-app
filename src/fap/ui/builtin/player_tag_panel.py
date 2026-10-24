"""Player Tag Panel page — a position-aware Hudl Sportscode code window for individual player analysis.

Two halves:

* **Code window** — the button layout per position (mock-up PNG), the build sheet (xlsx: every button with
  type, label group, hotkey, colour, lead/lag and pixel position), definitions and a sample export.
* **Analyse / Data** — upload the Sportscode export(s) of one player (timeline XML or instances CSV/XLSX,
  one file per match) → position KPIs, zone maps, success by action, match-by-match, workbook download.

Engine = :mod:`fap.playertag` (pure, Streamlit-free). Saved players = WorkspaceManager presets
(kind ``playertag_player``) holding the original files, so a re-open re-reads them with the current engine.
"""
from __future__ import annotations

import base64
import hashlib
import re

import pandas as pd
import streamlit as st

from fap.core.plugin import PluginInfo
from fap.identity.roles import Role
from fap.theme import components as C
from fap.ui.page import Page, page_registry

PT_STATE = "_pt_player"            # {"player", "position", "team", "files": [{"name", "match", "minutes", "data"}]}
PT_KIND = "playertag_player"


def _fname(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "chart"


def _sig(f: dict) -> str:
    return hashlib.sha1(f["data"]).hexdigest()[:16]


@st.cache_data(show_spinner=False, max_entries=64)
def _load_one(sig: str, _data: bytes, name: str, match: str, player: str, position: str):
    from fap.playertag import load_tags
    return load_tags(_data, name, player=player, position=position, match=match)


@st.cache_data(show_spinner=False, max_entries=200)
def _png(key: str, _make) -> bytes:
    from fap.playertag.charts import png
    return png(_make(), 300)


def _theme():
    from fap.matchlab.charts import THEMES
    name = st.session_state.get("pt_theme", "Dark")
    return name, THEMES[name]


def _chart(key: str, make, filename: str, wkey: str) -> None:
    data = _png(key, make)
    st.image(data, use_container_width=True)
    st.download_button("Download PNG", data, f"{_fname(filename)}.png", "image/png", key=f"pt_dl_{wkey}")


@page_registry.register
class PlayerTagPanelPage(Page):
    info = PluginInfo(id="player_tag_panel", name="Player Tag Panel", category="page")
    section = "Analysis"
    icon = "target"
    order = 8

    def render(self, shell) -> None:
        can_edit = shell.user.role >= Role.PERFORMANCE_ANALYST
        st.markdown("### Player Tag Panel")
        st.caption("Individual player tagging in Hudl Sportscode — one code window per position (same names "
                   "everywhere), then upload the export to get the player's numbers, zone maps and data.")
        st.radio("Chart theme", ["Dark", "Light"], key="pt_theme", horizontal=True)
        tabs = st.tabs(["Code window", "Analyse", "Data"])
        with tabs[0]:
            self._window()
        with tabs[1]:
            td = self._source(shell, can_edit)
            if td is not None:
                self._analyse(td)
        with tabs[2]:
            td = st.session_state.get("_pt_td")
            self._data(td) if td is not None else self._empty()

    # ---- code window ----------------------------------------------------------------
    def _window(self) -> None:
        from fap.playertag import charts as CH
        from fap.playertag import export as E
        from fap.playertag import taxonomy as T
        from fap.playertag.window import build
        tn, th = _theme()
        opts = list(T.POSITIONS) + ["ALL"]
        c = st.columns([2, 1.2, 1.2, 1.2])
        pos = c[0].selectbox("Position", opts, key="pt_win_pos",
                             format_func=lambda p: f"{p} · {T.POSITIONS[p]}" if p in T.POSITIONS else "All positions")
        win = build(pos)
        c[1].download_button("Build sheet (xlsx)", _bs(), "player_code_window_build_sheet.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="pt_bs",
                             help="Every button of every position: type, label group, hotkey, colour, lead/lag and "
                                  "x / y / size in pixels", use_container_width=True)
        c[2].download_button("Sample export (XML)", E.sample_xml(pos if pos in T.POSITIONS else "CM"),
                             f"sample_{pos}_timeline.xml", "application/xml", key="pt_sample",
                             help="A made-up Sportscode timeline export to try the Analyse tab",
                             use_container_width=True)
        c[3].metric("Buttons", f"{len(win.codes())} codes · {len(win.labels())} labels")
        _chart(f"win|{pos}|{tn}", lambda: CH.code_window(win, th), f"code_window_{pos}", "win")
        with st.expander("How to build it in Sportscode and how to tag", expanded=False):
            for step, txt in E.HOW_TO:
                st.markdown(f"**{step}.** {txt}")
            st.markdown("**Zones** — 18 zones: 6 columns from your own goal (Z1–Z3) to the opponent goal "
                        "(Z16–Z18) × 3 lanes (Left / Centre / Right, attacking to the right). Tag where the "
                        "action STARTED (a pass: where it was played from).")
            st.markdown("**Outcome** — Successful / Unsuccessful on passes, crosses, dribbles, carries, duels, "
                        "tackles, claims. Duel / tackle Successful = won. Shots use Shot Outcome instead. An "
                        "untagged outcome is left out of the % — never counted as a failure.")
        with st.expander("Code definitions", expanded=False):
            acts = [a for a in T.actions_for(pos)]
            st.dataframe(pd.DataFrame([{"Code": a.name, "Hotkey": a.hotkey, "Type": a.category,
                                        "Definition": a.help, "Labels": ", ".join(a.groups)} for a in acts]),
                         hide_index=True, use_container_width=True)
        with st.expander("Label groups", expanded=False):
            st.dataframe(pd.DataFrame([{"Group": g.name, "Labels": ", ".join(g.labels),
                                        "One per instance": "Yes" if g.exclusive else "No"}
                                       for g in T.LABEL_GROUPS]), hide_index=True, use_container_width=True)

    # ---- source -----------------------------------------------------------------------
    def _source(self, shell, can_edit: bool):
        from fap.playertag import taxonomy as T
        cur = st.session_state.setdefault(PT_STATE, {"player": "", "position": "CM", "team": "", "files": []})
        c = st.columns([2.2, 1.4, 1.4, 1.4])
        ups = c[0].file_uploader("Sportscode export(s) — one file per match", type=["xml", "csv", "xlsx", "xls"],
                                 accept_multiple_files=True, key="pt_upload",
                                 help="Timeline → Export → XML (recommended), or the instances CSV / XLSX")
        known = {(f["name"], _sig(f)) for f in cur["files"]}
        for up in ups or []:
            data = up.getvalue()
            f = {"name": up.name, "match": up.name.rsplit(".", 1)[0], "minutes": 90, "data": data}
            if (up.name, _sig(f)) not in known:
                cur["files"].append(f)
                known.add((up.name, _sig(f)))
        cur["player"] = c[1].text_input("Player", value=cur["player"], key="pt_player")
        pos_opts = list(T.POSITIONS)
        cur["position"] = c[2].selectbox("Position", pos_opts, index=pos_opts.index(cur["position"])
                                         if cur["position"] in pos_opts else 4, key="pt_pos",
                                         format_func=lambda p: f"{p} · {T.POSITIONS[p]}")
        cur["team"] = c[3].text_input("Team", value=cur["team"], key="pt_team")
        self._saved(shell, can_edit, cur)
        if not cur["files"]:
            st.session_state.pop("_pt_td", None)
            self._empty()
            return None
        st.markdown("##### Matches")
        ed = st.data_editor(pd.DataFrame([{"File": f["name"], "Match": f["match"], "Minutes": f["minutes"],
                                           "Remove": False} for f in cur["files"]]),
                            hide_index=True, use_container_width=True, key=f"pt_files_{len(cur['files'])}",
                            disabled=["File"],
                            column_config={"Minutes": st.column_config.NumberColumn(min_value=0, max_value=130,
                                                                                    help="Minutes played (per 90)")})
        keep = []
        for f, (_, r) in zip(cur["files"], ed.iterrows()):
            if not r["Remove"]:
                f["match"] = str(r["Match"] or f["name"]).strip()
                f["minutes"] = float(r["Minutes"] or 0)
                keep.append(f)
        if len(keep) != len(cur["files"]):
            cur["files"] = keep
            st.rerun()
        from fap.playertag import combine
        parts = []
        for f in cur["files"]:
            try:
                parts.append(_load_one(_sig(f), f["data"], f["name"], f["match"], cur["player"], cur["position"]))
            except Exception as exc:
                C.render_alert(f"Could not read {f['name']}: {exc}", "danger")
        td = combine(parts)
        td.minutes = {f["match"]: f["minutes"] for f in cur["files"]}
        st.session_state["_pt_td"] = td
        if not len(td.df):
            C.render_alert("No actions of the player code window were found in the file(s). Check the code "
                           "names against the build sheet.", "warning")
            return None
        return td

    def _saved(self, shell, can_edit: bool, cur: dict) -> None:
        wm = getattr(shell, "wm", None)
        if wm is None:
            return
        try:
            saved = wm.list_presets(shell.user, kind=PT_KIND)
        except Exception:
            saved = []
        c = st.columns([2.2, 1, 1, 2.6])
        names = {getattr(p, "name", "player"): p for p in saved}
        if names:
            pick = c[0].selectbox("Saved players", ["—"] + list(names), key="pt_saved")
            if pick != "—" and c[1].button("Open", key="pt_open", use_container_width=True):
                doc = getattr(names[pick], "document", None) or {}
                st.session_state[PT_STATE] = {
                    "player": doc.get("player", ""), "position": doc.get("position", "CM"),
                    "team": doc.get("team", ""),
                    "files": [{"name": f["name"], "match": f.get("match", f["name"]),
                               "minutes": f.get("minutes", 90), "data": base64.b64decode(f["data_b64"])}
                              for f in doc.get("files", [])]}
                for k in ("pt_player", "pt_pos", "pt_team"):
                    st.session_state.pop(k, None)
                st.rerun()
        if cur["files"] and can_edit and c[2].button("Save player", key="pt_save", use_container_width=True):
            name = cur["player"] or "Player"
            try:
                prev = names.get(name)
                wm.save_preset(shell.user, kind=PT_KIND, name=name, preset_id=getattr(prev, "id", None),
                               document={"player": cur["player"], "position": cur["position"], "team": cur["team"],
                                         "files": [{"name": f["name"], "match": f["match"], "minutes": f["minutes"],
                                                    "data_b64": base64.b64encode(f["data"]).decode("ascii")}
                                                   for f in cur["files"]]})
                st.toast(f"{name} saved with {len(cur['files'])} match file(s).")
            except Exception as exc:
                st.toast(f"Save failed: {exc}")

    @staticmethod
    def _empty() -> None:
        C.render_empty_state("Upload the player's Sportscode export",
                             "Timeline XML or instances CSV / XLSX tagged with the Player Tag Panel code window — "
                             "one file per match. No file yet? Download the sample export in the Code window tab.",
                             icon_name="upload")

    # ---- analyse ------------------------------------------------------------------------
    def _analyse(self, td) -> None:
        from fap.playertag import charts as CH
        from fap.playertag import stats as S
        from fap.playertag import taxonomy as T
        cur = st.session_state[PT_STATE]
        pos, player = cur["position"], cur["player"] or "Player"
        tn, th = _theme()
        matches = td.matches
        pick = "All matches"
        if len(matches) > 1:
            pick = st.selectbox("Match", ["All matches"] + matches, key="pt_match")
        sub = td.subset(pick)
        df = sub.df
        mins = (sum(td.minutes.get(m, 0) for m in matches) if pick == "All matches" else td.minutes.get(pick)) or None
        sig = "|".join(_sig(f) for f in cur["files"]) + f"|{pos}|{pick}|{mins}|{tn}"
        self._quality(td, df)

        st.markdown(f"#### {player} · {T.POSITIONS[pos]}")
        st.caption(f"{pick} · {len(df)} tagged actions" + (f" · {mins:.0f} minutes" if mins else ""))
        cards = [C.metric_card_html(v.metric.label, v.display, hint=v.detail) for v in S.headline(df, pos, mins)]
        for i in range(0, len(cards), 4):
            C.render_metric_row(cards[i:i + 4])

        t = st.tabs(["Player sheet", "All numbers", "Zones", "By action"] +
                    (["Match by match"] if len(matches) > 1 else []))
        subtitle = f"{pick}" + (f" · {cur['team']}" if cur["team"] else "") + (f" · {mins:.0f} min" if mins else "")
        with t[0]:
            _chart(f"sheet|{sig}|{player}", lambda: CH.player_sheet(df, player, pos, th, minutes=mins,
                                                                    subtitle=subtitle),
                   f"{player}_{pos}_{pick}_sheet", "sheet")
        with t[1]:
            for sec, vals in S.sections(df, pos, mins):
                st.markdown(f"##### {sec}")
                cards = [C.metric_card_html(v.metric.label, v.display, hint=v.detail) for v in vals]
                for i in range(0, len(cards), 4):
                    C.render_metric_row(cards[i:i + 4])
        with t[2]:
            groups = {"All actions": None,
                      "In possession": tuple(a.name for a in T.ACTIONS if a.category == T.IN_POSS),
                      "Out of possession": tuple(a.name for a in T.ACTIONS if a.category == T.OUT_POSS),
                      "Goalkeeping": tuple(a.name for a in T.ACTIONS if a.category == T.GOALKEEPING)}
                      | {a: (a,) for a in dict.fromkeys(df["action"])}
            c = st.columns([2, 1.4])
            which = c[0].selectbox("Actions", list(groups), key="pt_zone_act")
            mode = c[1].radio("Colour by", ["Volume", "Success %"], key="pt_zone_mode", horizontal=True)
            m = "count" if mode == "Volume" else "success"
            title = f"{player} · {which} · {'actions' if m == 'count' else 'success %'} by zone"
            _chart(f"zone|{sig}|{which}|{m}", lambda: CH.zone_map(df, title, th, actions=groups[which], mode=m),
                   f"{player}_{which}_zones", "zone")
            st.dataframe(S.zone_table(df, groups[which]), hide_index=True, use_container_width=True)
        with t[3]:
            _chart(f"bars|{sig}", lambda: CH.action_bars(df, f"{player} · actions & success rate", th),
                   f"{player}_actions", "bars")
            st.dataframe(S.action_summary(df), hide_index=True, use_container_width=True)
        if len(matches) > 1:
            with t[4]:
                pm = S.per_match(td.df, pos, td.minutes)
                st.dataframe(pm, hide_index=True, use_container_width=True)
                metric = st.selectbox("Trend", list(dict.fromkeys(pm["Metric"])), key="pt_trend")
                _chart(f"trend|{sig}|{metric}", lambda: CH.per_match_trend(pm, metric, th),
                       f"{player}_{metric}_trend", "trend")

    @staticmethod
    def _quality(td, df) -> None:
        from fap.playertag import stats as S
        msgs = []
        if td.ignored:
            msgs.append("Skipped codes (not in the player code window): " +
                        ", ".join(f"{k} ×{v}" for k, v in td.ignored.most_common(8)))
        if td.unknown_labels:
            msgs.append("Unknown labels (kept out of the stats): " +
                        ", ".join(f"{k} ×{v}" for k, v in td.unknown_labels.most_common(8)))
        cov = S.coverage(df)
        low = [f"{k} {v:.0f}%" for k, v in cov.items() if v == v and v < 90]
        if low:
            msgs.append("Tagging coverage below 90%: " + " · ".join(low) + " — those instances are left out of "
                        "the zone maps / success rates.")
        if msgs:
            C.render_alert("<br>".join(msgs), "warning", title="Tagging check")

    # ---- data -------------------------------------------------------------------------------
    def _data(self, td) -> None:
        from fap.playertag import export as E
        cur = st.session_state.get(PT_STATE, {})
        player, pos = cur.get("player") or "Player", cur.get("position", "CM")
        c = st.columns(2)
        c[0].download_button("Download workbook (xlsx)",
                             E.data_workbook(td.df, pos, td.minutes, player=player, team=cur.get("team", "")),
                             f"{_fname(player)}_{pos}_tag_data.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="pt_xlsx",
                             help="Summary, per match, actions, zones, every event, and the events in the app's "
                                  "canonical format (x / y from the zone centre)", use_container_width=True)
        c[1].download_button("Events CSV (app format)", E.canonical_csv(td.df, player, cur.get("team", "")),
                             f"{_fname(player)}_{pos}_events.csv", "text/csv", key="pt_csv",
                             help="event_type / outcome / x / y — can be imported in Data Hub",
                             use_container_width=True)
        show = td.df.drop(columns=[c for c in td.df.columns if c.startswith("is_")])
        st.dataframe(show, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner=False)
def _bs() -> bytes:
    from fap.playertag import export as E
    return E.build_sheet()
