"""Code-window layout per position — one source for the on-screen mock-up, the PNG and the build sheet.

Coordinates are pixels on a 1600-wide canvas (Sportscode's code-window editor works in pixels too), so the
build sheet tells the analyst exactly where each button goes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from fap.playertag import taxonomy as T

CANVAS_W = 1600
_CODE_W, _CODE_H, _LAB_W, _LAB_H, _GAP, _HDR = 168, 46, 124, 38, 8, 26
OUTCOME_COLOURS = {"Successful": "#0f9f6e", "Unsuccessful": "#d6334f"}
LABEL_COLOUR, ZONE_COLOUR = "#5b6577", "#2f6b3a"
CENTRE_GROUPS = ("Outcome", "Pressure", "Direction")


@dataclass(frozen=True)
class Button:
    name: str
    kind: str          # "code" | "label"
    group: str         # category for codes, label group for labels
    hotkey: str
    colour: str
    x: int
    y: int
    w: int
    h: int
    lead: float = 0.0
    lag: float = 0.0
    help: str = ""


@dataclass(frozen=True)
class Header:
    text: str
    x: int
    y: int
    w: int


@dataclass
class Window:
    position: str
    title: str
    buttons: list[Button]
    headers: list[Header]
    height: int
    zone_box: tuple[int, int, int, int]       # x, y, w, h of the zone grid

    def codes(self) -> list[Button]:
        return [b for b in self.buttons if b.kind == "code"]

    def labels(self) -> list[Button]:
        return [b for b in self.buttons if b.kind == "label"]


def _groups(position: str) -> list[T.LabelGroup]:
    gs = T.groups_for(position)
    if position != "GK":
        gs = [g for g in gs if g.name not in ("Distribution", "Save Type")]
    return gs


def build(position: str) -> Window:
    """position = one of T.POSITIONS, or "ALL" for the full window."""
    acts = T.actions_for(position)
    title = T.POSITIONS.get(position, "All positions")
    buttons: list[Button] = []
    headers: list[Header] = [Header("CODES  ·  one click = one instance", 20, 70, 540)]

    # ---- code buttons, by category (3 per row)
    y = 104
    for cat in (T.IN_POSS, T.OUT_POSS, T.GOALKEEPING):
        cat_acts = [a for a in acts if a.category == cat]
        if not cat_acts:
            continue
        headers.append(Header(cat.upper(), 20, y, 540))
        y += _HDR
        for i, a in enumerate(cat_acts):
            r, c = divmod(i, 3)
            buttons.append(Button(a.name, "code", cat, a.hotkey, T.CATEGORY_COLOURS[cat],
                                  20 + c * (_CODE_W + _GAP), y + r * (_CODE_H + _GAP), _CODE_W, _CODE_H,
                                  a.lead, a.lag, a.help))
        y += math.ceil(len(cat_acts) / 3) * (_CODE_H + _GAP) + 10
    code_bottom = y

    # ---- zone grid (pitch layout, attacking → right, LEFT lane on top)
    zx, zy, cw, rh = 600, 104, 68, 70
    headers.append(Header("ZONE  ·  where the action happened  (attack →)", zx, 70, cw * T.N_COLS))
    for col in range(1, T.N_COLS + 1):
        for lane in range(1, T.N_LANES + 1):
            z = T.zone_name(col, lane)
            buttons.append(Button(z, "label", "Zone", "", ZONE_COLOUR, zx + (col - 1) * cw, zy + (lane - 1) * rh,
                                  cw - 4, rh - 4))
    zone_box = (zx, zy, cw * T.N_COLS, rh * T.N_LANES)

    # ---- centre: outcome (big), pressure, direction
    groups = {g.name: g for g in _groups(position)}
    cy = zy + rh * T.N_LANES + 20
    if "Outcome" in groups:
        headers.append(Header("OUTCOME", zx, cy, 408))
        cy += _HDR
        g = groups["Outcome"]
        for i, lab in enumerate(g.labels):
            buttons.append(Button(lab, "label", g.name, g.hotkeys[i], OUTCOME_COLOURS[lab], zx + i * 206, cy, 200,
                                  56))
        cy += 56 + 14
    for gname in ("Pressure", "Direction"):
        if gname in groups:
            cy = _place_group(groups[gname], zx, cy, 408, buttons, headers)
    centre_bottom = cy

    # ---- right column: every other label group (flow layout), overflow under the centre
    regions = [[1040, 70, 540], [zx, centre_bottom + 6, 408], [20, code_bottom + 6, 540]]
    rest = [g for g in _groups(position) if g.name not in CENTRE_GROUPS and g.name != "Zone"]
    ri, ry = 0, regions[0][1]
    limit = 880
    for g in rest:
        rx, _, rw = regions[ri]
        cols = max(1, rw // (_LAB_W + _GAP))
        need = _HDR + math.ceil(len(g.labels) / cols) * (_LAB_H + _GAP) + 6
        while ry + need > limit and ri < len(regions) - 1:
            ri += 1
            rx, ry, rw = regions[ri]
        ry = _place_group(g, rx, ry, rw, buttons, headers)
    height = max([b.y + b.h for b in buttons] + [limit]) + 24
    return Window(position, title, buttons, headers, height, zone_box)


def _place_group(g: T.LabelGroup, x: int, y: int, width: int, buttons: list, headers: list) -> int:
    headers.append(Header(g.name.upper() + ("" if g.exclusive else "  · several allowed"), x, y, width))
    y += _HDR
    cols = max(1, width // (_LAB_W + _GAP))
    for i, lab in enumerate(g.labels):
        r, c = divmod(i, cols)
        hk = g.hotkeys[i] if i < len(g.hotkeys) else ""
        buttons.append(Button(lab, "label", g.name, hk, LABEL_COLOUR, x + c * (_LAB_W + _GAP),
                              y + r * (_LAB_H + _GAP), _LAB_W, _LAB_H))
    return y + math.ceil(len(g.labels) / cols) * (_LAB_H + _GAP) + 6
