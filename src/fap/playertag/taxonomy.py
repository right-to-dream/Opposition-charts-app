"""The individual-player code window — one vocabulary for every position.

A Sportscode code window has two kinds of buttons:

* **Code buttons** (``kind="code"``) create an instance on the timeline — one per player action
  (Pass, Duel, Save ...). They are mutually exclusive and use a fixed lead/lag around the click.
* **Label buttons** (``kind="label"``) attach a descriptor to the instance that is open/last created
  (Successful, Z7, Under Pressure ...). Each label belongs to a *label group*; the group name is what
  Sportscode writes in the export, so the names here are the contract with :mod:`fap.playertag.loader`.

Every position uses the SAME names and hotkeys; a position window only shows the subset of actions and
label groups that matter for that role, so data from a GK window and a striker window land in one schema.

Zones: 18 zones = 6 thirds-of-thirds along the pitch (own goal → opponent goal) × 3 lanes. With the
team attacking left → right, the LEFT lane is at the top of the pitch. ``Z1..Z3`` = first column
(L, C, R), ..., ``Z16..Z18`` = last column next to the opponent goal. Canonical metres (105 × 68,
y = 0 = RIGHT touchline) as everywhere else in the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PITCH_L, PITCH_W = 105.0, 68.0
N_COLS, N_LANES = 6, 3
LANES = ("Left", "Centre", "Right")

POSITIONS: dict[str, str] = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB": "Full-back / Wing-back",
    "DM": "Defensive midfielder",
    "CM": "Central midfielder",
    "AM": "Attacking midfielder",
    "W": "Winger",
    "ST": "Striker",
}

IN_POSS, OUT_POSS, GOALKEEPING = "In possession", "Out of possession", "Goalkeeping"
CATEGORY_COLOURS = {IN_POSS: "#1d6fe0", OUT_POSS: "#d6334f", GOALKEEPING: "#0f9f6e"}


@dataclass(frozen=True)
class Action:
    name: str                       # code button name = the Sportscode "code" / "Row"
    hotkey: str
    category: str
    positions: tuple[str, ...]      # positions whose window shows it
    groups: tuple[str, ...]         # label groups that describe it
    help: str = ""
    synonyms: tuple[str, ...] = ()
    lead: float = 5.0
    lag: float = 3.0


@dataclass(frozen=True)
class LabelGroup:
    name: str
    labels: tuple[str, ...]
    exclusive: bool = True          # one label of the group per instance
    hotkeys: tuple[str, ...] = ()   # aligned with ``labels`` ("" = click only)
    synonyms: dict = field(default_factory=dict)   # label -> other spellings


ALL = tuple(POSITIONS)
OUTFIELD = tuple(p for p in POSITIONS if p != "GK")
ATTACKERS = ("AM", "W", "ST")

# ------------------------------------------------------------------ label groups
LABEL_GROUPS: tuple[LabelGroup, ...] = (
    LabelGroup("Outcome", ("Successful", "Unsuccessful"), hotkeys=("1", "2"),
               synonyms={"Successful": ("success", "succ", "complete", "completed", "won", "accurate", "yes", "✓"),
                         "Unsuccessful": ("unsuccess", "fail", "failed", "incomplete", "lost", "inaccurate", "no",
                                          "✗")}),
    LabelGroup("Zone", tuple(f"Z{i}" for i in range(1, N_COLS * N_LANES + 1))),
    LabelGroup("Pressure", ("Under Pressure",), hotkeys=("3",), synonyms={"Under Pressure": ("pressure",
                                                                                             "under press",
                                                                                             "pressed")}),
    LabelGroup("Direction", ("Forward", "Sideways", "Backward"), hotkeys=("4", "5", "6"),
               synonyms={"Sideways": ("lateral", "side"), "Backward": ("back", "backwards")}),
    LabelGroup("Pass Length", ("Short", "Long"), hotkeys=("", "0")),
    LabelGroup("Pass Value", ("Progressive", "Key Pass", "Assist", "Through Ball", "Switch", "Line Breaking",
                              "Final Third Entry", "Box Entry"), exclusive=False, hotkeys=("7", "8", "9"),
               synonyms={"Key Pass": ("keypass", "chance created", "shot assist"), "Through Ball": ("throughball",
                                                                                                     "through"),
                         "Switch": ("switch of play",), "Line Breaking": ("line break", "line-breaking"),
                         "Final Third Entry": ("f3 entry", "final 3rd entry"), "Box Entry": ("penalty box entry",)}),
    LabelGroup("Cross Type", ("Low", "High", "Cutback", "Early"), synonyms={"Cutback": ("cut back", "cut-back")}),
    LabelGroup("Shot Outcome", ("Goal", "On Target", "Off Target", "Blocked"),
               synonyms={"On Target": ("saved", "on-target"), "Off Target": ("off-target", "wide", "over", "post"),
                         "Blocked": ("block",)}),
    LabelGroup("Body Part", ("Right Foot", "Left Foot", "Head", "Other"),
               synonyms={"Right Foot": ("right", "rf"), "Left Foot": ("left", "lf"), "Head": ("header",)}),
    LabelGroup("Reception", ("Between Lines", "In Behind", "On The Turn"), exclusive=False,
               synonyms={"Between Lines": ("between the lines",), "In Behind": ("behind", "in-behind")}),
    LabelGroup("Press Outcome", ("Regain", "Forced Error", "No Regain"),
               synonyms={"Regain": ("regained", "won back"), "No Regain": ("bypassed", "no effect")}),
    LabelGroup("Loss Type", ("Bad Pass", "Bad Touch", "Dispossessed"),
               synonyms={"Dispossessed": ("tackled",)}),
    LabelGroup("Save Type", ("Catch", "Parry", "Punch", "Smother"), synonyms={"Smother": ("feet", "1v1")}),
    LabelGroup("Distribution", ("Hand", "Goal Kick", "Open Play Kick"),
               synonyms={"Hand": ("throw", "hand throw", "roll"), "Goal Kick": ("gk", "goalkick")}),
    LabelGroup("Card", ("Yellow Card", "Red Card"), synonyms={"Yellow Card": ("yellow",), "Red Card": ("red",)}),
    LabelGroup("Set Piece", ("Set Piece",), synonyms={"Set Piece": ("set play", "dead ball")}),
)
GROUPS = {g.name: g for g in LABEL_GROUPS}

_P = ("Outcome", "Zone", "Pressure", "Direction", "Pass Length", "Pass Value", "Set Piece")
_D = ("Outcome", "Zone")

# ------------------------------------------------------------------ code buttons
ACTIONS: tuple[Action, ...] = (
    # in possession
    Action("Pass", "P", IN_POSS, ALL, _P + ("Distribution",), "Every pass attempt (GK: tag the Distribution label)",
           ("passes", "pass attempt")),
    Action("Cross", "C", IN_POSS, ("FB", "CM", "AM", "W", "ST"), ("Outcome", "Zone", "Cross Type", "Pressure",
                                                                   "Pass Value", "Set Piece"),
           "Delivery from the wide areas into the box", ("crosses",)),
    Action("Shot", "S", IN_POSS, OUTFIELD, ("Shot Outcome", "Zone", "Body Part", "Pressure", "Set Piece"),
           "Every attempt on goal — Shot Outcome is required", ("shots", "attempt", "shoot")),
    Action("Dribble", "D", IN_POSS, ("FB", "DM", "CM", "AM", "W", "ST"), ("Outcome", "Zone", "Pressure"),
           "1v1 take-on to beat an opponent", ("take on", "take-on", "dribbles")),
    Action("Carry", "K", IN_POSS, OUTFIELD, ("Outcome", "Zone", "Pass Value"),
           "Ball carry of 5 m+ (label Progressive when it moves play 10 m+ forward)", ("carries", "run with ball")),
    Action("Receive", "V", IN_POSS, OUTFIELD, ("Zone", "Pressure", "Reception"),
           "Ball received under control", ("reception", "receipt", "received")),
    Action("Ball Lost", "L", IN_POSS, ALL, ("Zone", "Loss Type", "Pressure"),
           "Possession lost by a touch / dispossession (not a failed pass — that is Pass + Unsuccessful)",
           ("loss", "possession lost", "turnover", "dispossessed")),
    Action("Foul Won", "F", IN_POSS, OUTFIELD, ("Zone",), "Fouled by an opponent", ("fouled", "foul suffered")),
    Action("Offside", "Q", IN_POSS, ("FB", "AM", "W", "ST"), ("Zone",), "Caught offside", ("offsides",)),
    # out of possession
    Action("Ground Duel", "G", OUT_POSS, OUTFIELD, _D, "1v1 ground contest (Successful = won)", ("duel", "duels",
                                                                                                   "ground duels")),
    Action("Aerial Duel", "A", OUT_POSS, ALL, _D + ("Set Piece",), "Header contest (Successful = won)",
           ("aerial", "header duel", "air duel", "aerials")),
    Action("Tackle", "T", OUT_POSS, OUTFIELD, _D, "Tackle attempt (Successful = ball won)", ("tackles",)),
    Action("Interception", "I", OUT_POSS, OUTFIELD, ("Zone",), "Reads and cuts out a pass", ("interceptions",
                                                                                              "intercept")),
    Action("Recovery", "R", OUT_POSS, OUTFIELD, ("Zone",), "Picks up a loose ball", ("ball recovery", "recoveries",
                                                                                      "regain")),
    Action("Clearance", "E", OUT_POSS, ("GK", "CB", "FB", "DM", "CM"), ("Zone", "Body Part", "Set Piece"),
           "Defensive clearance", ("clearances", "clear")),
    Action("Block", "B", OUT_POSS, ("CB", "FB", "DM", "CM"), ("Zone",), "Blocks a shot or cross", ("blocks",
                                                                                                     "blocked shot")),
    Action("Press", "X", OUT_POSS, OUTFIELD, ("Zone", "Press Outcome"), "Pressure on the ball carrier",
           ("pressure", "pressing", "presses")),
    Action("Dribbled Past", "Y", OUT_POSS, OUTFIELD, ("Zone",), "Beaten by an opponent's take-on",
           ("dribbled", "beaten", "dribble past")),
    Action("Foul", "O", OUT_POSS, ALL, ("Zone", "Card"), "Foul committed", ("foul committed", "fouls")),
    # goalkeeping
    Action("Save", "W", GOALKEEPING, ("GK",), ("Save Type", "Zone", "Outcome"),
           "Shot on target saved (Unsuccessful = rebound conceded to the opponent)", ("saves", "gk save"), 8.0, 3.0),
    Action("Goal Conceded", "N", GOALKEEPING, ("GK",), ("Zone", "Set Piece"), "Goal conceded",
           ("conceded", "goal against"), 10.0, 3.0),
    Action("High Claim", "H", GOALKEEPING, ("GK",), ("Outcome", "Save Type", "Zone", "Set Piece"),
           "Comes for a cross (Successful = claimed / punched clear)", ("claim", "cross claim", "high ball")),
    Action("Sweeper Action", "J", GOALKEEPING, ("GK",), ("Outcome", "Zone"),
           "Comes off the line outside the box to deal with a ball in behind", ("sweeper", "sweeping")),
)
ACTION_BY_NAME = {a.name: a for a in ACTIONS}


def actions_for(position: str) -> list[Action]:
    return [a for a in ACTIONS if position in a.positions] if position in POSITIONS else list(ACTIONS)


def groups_for(position: str) -> list[LabelGroup]:
    used = {g for a in actions_for(position) for g in a.groups}
    return [g for g in LABEL_GROUPS if g.name in used]


# ------------------------------------------------------------------ zones
def zone_name(col: int, lane: int) -> str:
    """col 1..6 (own goal → opponent goal), lane 1..3 (Left, Centre, Right)."""
    return f"Z{(col - 1) * N_LANES + lane}"


def zone_cell(zone: str) -> tuple[int, int] | None:
    try:
        n = int(str(zone).strip().upper().lstrip("Z"))
    except ValueError:
        return None
    if not 1 <= n <= N_COLS * N_LANES:
        return None
    return (n - 1) // N_LANES + 1, (n - 1) % N_LANES + 1


def zone_bounds(zone: str) -> tuple[float, float, float, float] | None:
    """(x0, x1, y0, y1) in metres. Left lane = high y (y = 0 is the RIGHT touchline)."""
    cell = zone_cell(zone)
    if cell is None:
        return None
    col, lane = cell
    w, h = PITCH_L / N_COLS, PITCH_W / N_LANES
    y1 = PITCH_W - (lane - 1) * h
    return (col - 1) * w, col * w, y1 - h, y1


def zone_centre(zone: str) -> tuple[float, float] | None:
    b = zone_bounds(zone)
    return None if b is None else ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2)


def zone_third(zone: str) -> str:
    cell = zone_cell(zone)
    if cell is None:
        return ""
    return ("Defensive third", "Middle third", "Final third")[(cell[0] - 1) // 2]


def zone_lane(zone: str) -> str:
    cell = zone_cell(zone)
    return "" if cell is None else LANES[cell[1] - 1]
