"""Reference data: team IDs, match IDs and the Masar squad (player IDs by shirt number).

Team names in tagging files ("Masar", "Proxy", "mansoura") never match the official names
("FC Masar Men", "Proxy SC", "El Mansoura") exactly, so names are matched case-insensitively on a
normalised key (punctuation, "FC / SC / Club / Men / El / Al ..." dropped), then by token
containment, then by close spelling. Every match result says HOW it matched, so nothing is guessed
silently.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_STOP = {"fc", "sc", "club", "men", "el", "al", "the", "afc", "cf", "team", "sporting"}


def team_key(name: Any) -> str:
    s = str(name or "").lower()
    s = re.sub(r"[^a-z0-9؀-ۿ]+", " ", s)
    toks = [t for t in s.split() if t not in _STOP]
    return " ".join(toks) or s.strip()


def _cols(df: pd.DataFrame, *cands: str) -> str | None:
    look = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c for c in df.columns}
    for c in cands:
        if c in look:
            return look[c]
    return None


# ------------------------------------------------------------------ teams
@dataclass
class TeamMatch:
    team_id: str = ""
    official: str = ""
    method: str = "not found"


def match_team(name: str, teams: pd.DataFrame | None) -> TeamMatch:
    """Find ``name`` in a team table (team_id, team). Case-insensitive + normalised."""
    if teams is None or teams.empty or not str(name or "").strip():
        return TeamMatch()
    idc, nc = _cols(teams, "team_id", "id"), _cols(teams, "team", "team_name", "name")
    if not idc or not nc:
        return TeamMatch(method="team table needs team_id + team columns")
    names = teams[nc].astype(str).tolist()
    ids = teams[idc].astype(str).tolist()
    low = str(name).strip().lower()
    for n, i in zip(names, ids):
        if n.strip().lower() == low:
            return TeamMatch(i, n, "exact (case-insensitive)")
    k = team_key(name)
    keys = [team_key(n) for n in names]
    hits = [j for j, kk in enumerate(keys) if kk == k]
    if len(hits) == 1:
        return TeamMatch(ids[hits[0]], names[hits[0]], "normalised name")
    kt = set(k.split())
    hits = [j for j, kk in enumerate(keys) if kt and (kt <= set(kk.split()) or set(kk.split()) <= kt)]
    if len(hits) == 1:
        return TeamMatch(ids[hits[0]], names[hits[0]], "name contained")
    close = difflib.get_close_matches(k, keys, n=2, cutoff=0.82)
    if len(close) == 1 or (len(close) == 2 and difflib.SequenceMatcher(None, k, close[0]).ratio() > 0.92):
        j = keys.index(close[0])
        return TeamMatch(ids[j], names[j], f"close spelling ({close[0]})")
    if len(hits) > 1:
        return TeamMatch(method="ambiguous: " + ", ".join(names[j] for j in hits))
    return TeamMatch()


# ------------------------------------------------------------------ matches
def find_matches(team_a: str, team_b: str, matches: pd.DataFrame | None,
                 teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of the match table between the two teams (either home/away order)."""
    if matches is None or matches.empty:
        return pd.DataFrame()
    hc, ac = _cols(matches, "home_team", "home"), _cols(matches, "away_team", "away")
    if not hc or not ac:
        return pd.DataFrame()
    ka, kb = team_key(match_team(team_a, teams).official or team_a), team_key(match_team(team_b, teams).official or team_b)

    def same(x, k):
        kx = team_key(x)
        return kx == k or (set(k.split()) and (set(k.split()) <= set(kx.split()) or set(kx.split()) <= set(k.split())))

    m = matches.apply(lambda r: (same(r[hc], ka) and same(r[ac], kb)) or (same(r[hc], kb) and same(r[ac], ka)), axis=1)
    return matches[m]


# ------------------------------------------------------------------ squad
# Masar 2026/27 squad (converter notebook + 'Masar Squad Details 26-27.xlsx'). An uploaded squad file
# replaces it.
DEFAULT_MASAR_SQUAD = {
    "24": ("0350843a-cf0f-5081-95b3-f8ea3dfb4324", "Hamdy Yasser Moussa"),
    "15": ("ee8ee58b-a46e-5d57-96f3-52dc41b821af", "Adam Babajide"),
    "1": ("2a5c7210-e415-53ff-bac6-70db0fb63896", "Ibrahim Zayed"),
    "26": ("4c47e2dd-555c-542e-913f-646f088dca4f", "Ahmed Haggag"),
    "16": ("1a22d871-c996-5399-8f84-08331c0c3057", "Ahmed Mostafa"),
    "2": ("5e37f721-3ee2-5892-a3de-896a9def8dec", "Abdelrahman Youssef"),
    "3": ("be29993c-7ff8-54b0-aeb6-c4a2b44f80c0", "Ahmed Abdelrahman"),
    "28": ("5a3ceb33-7c99-504d-82b4-272ef08b6c68", "Ahmed Sayed Talha"),
    "5": ("787a721d-1a77-5b8f-883c-4af9bef4cb2c", "Mohamed Ataq"),
    "6": ("4b98e625-746b-51ec-b2e4-c8180b3f1c1a", "Abdelrahman Abdelmawgood Fathalla"),
    "20": ("f7e3e6d0-ce99-52dd-9fcf-b971f2178069", "Abdul-Aziz Coulibaly"),
    "13": ("fc46b6a1-b9e1-53b4-94be-a45d2f833436", "Ahmed Shaaban"),
    "33": ("2d61bb9f-aee0-5b87-92c1-7f211dde0354", "Mohamed Kamel"),
    "14": ("c9d4fdb0-afb3-54ce-b9b9-32323c61eb8f", "Ahmed Essam"),
    "8": ("4d7035b8-3922-5fc1-a0e1-7822969d69df", "Ahmed Ibrahim"),
    "17": ("f34e4408-ca8a-5f89-8119-45a97d3dc686", "Mohamed Farhan"),
    "41": ("fb5bda3c-099c-547a-93fa-a1efe684d9ff", "Zeyad Yasser"),
    "74": ("23e72d05-0163-54d3-acd4-7120b8208a3a", "Zeyad Shadad"),
    "12": ("740127cc-3f6c-5e48-b405-3ab656061579", "Mostafa Osman Anany"),
    "18": ("d899c523-efe8-5124-aa9a-ed3e4b97548e", "Islam Marzouk"),
    "22": ("a41431d7-1805-53ad-ae71-ed1829e4408b", "Samir Essam"),
    "7": ("7173c322-3463-56d6-ad76-a0d608f94bea", "Abdelrahman Mahmoud Abouda"),
    "29": ("c9dfc637-2cc0-5816-b79b-d9dd584d7c12", "Ahmed El-Shamy"),
    "11": ("765a1efc-0311-5ef4-a3cb-bb5b1529d6dd", "Ayman Ashraf Nada"),
    "30": ("a5830983-844c-5a63-be19-397c673ebcf6", "Godwin Friday"),
    "21": ("ab898557-36ad-5bf5-b13c-c5d74e7d2057", "Marwan Magdy"),
    "10": ("d5fb0b57-64a7-5d27-9c3c-b0ec9ff485d1", "Mohamed Abdelqawy"),
    "25": ("57f2400b-7ae9-5c10-8e48-860a3274e28e", "Mahmoud Ageeb"),
    "9": ("41054417-5af4-54f0-9b3c-b4fa47bfea87", "Ahmed Raafat"),
    "23": ("700388fe-c690-5dca-b1c8-6e967e7027a8", "Mazen Yasser"),
    "19": ("90df6f03-6fb6-56e6-8634-7dcf5492b480", "Mostafa Sultan"),
}
DEFAULT_AMBIGUOUS: tuple = ()


@dataclass
class Squad:
    by_shirt: dict = field(default_factory=dict)        # "20" -> (player_id, player_name)
    ambiguous: tuple = ()                                # shirt numbers shared by >1 player
    source: str = "notebook default"
    short: dict = field(default_factory=dict)           # "20" -> "A.Coulibaly" (chart labels)
    role: dict = field(default_factory=dict)            # "20" -> "DEF"

    def label(self, shirt: str) -> str:
        """'#20 A.Coulibaly' for charts / pickers ('#20' when unknown)."""
        s = str(shirt or "").strip()
        nm = self.short.get(s) or self.lookup(s)[1]
        return f"#{s} {nm}".strip() if s else ""

    def lookup(self, shirt: str) -> tuple[str, str]:
        s = str(shirt or "").strip()
        if not s or s == "0" or s in self.ambiguous:
            return "", ""
        return self.by_shirt.get(s, ("", ""))


def default_squad() -> Squad:
    return Squad(dict(DEFAULT_MASAR_SQUAD), DEFAULT_AMBIGUOUS, "built-in Masar squad 2026/27")


def squad_from_table(df: pd.DataFrame, source: str = "uploaded squad") -> Squad:
    """Squad file: player_id + player_name/name + shirt/number/player_number. Duplicate shirts are
    reported as ambiguous and never auto-assigned."""
    idc = _cols(df, "player_id", "player_app_id", "app_id", "id", "playerid")
    nc = _cols(df, "player_name", "name", "player", "full_name")
    sc = _cols(df, "shirt_number", "player_number", "shirt", "number", "jersey", "jersey_number", "no")
    if not idc or not sc:
        raise ValueError("Squad file needs a player id column (player_id / player_app_id) and a "
                         "shirt-number column (shirt_number / number / player_number).")
    d = df.dropna(subset=[idc, sc]).copy()
    st = _cols(d, "status")
    if st:                                               # inactive players never take a shirt
        d = d[~d[st].astype(str).str.lower().isin(["inactive", "left", "released"])]
    if nc:
        d["_name"] = d[nc].astype(str)
    else:                                                # first + middle + last (same as the notebook)
        parts = [c for c in (_cols(d, "first_name"), _cols(d, "middle_name"), _cols(d, "last_name")) if c]
        d["_name"] = d[parts].apply(lambda r: " ".join(str(v).strip() for v in r if pd.notna(v) and str(v).strip()), axis=1) if parts else ""
    shc, rc = _cols(d, "short_name"), _cols(d, "role_code", "position", "role")
    d["_shirt"] = pd.to_numeric(d[sc], errors="coerce").astype("Int64").astype(str)
    counts = d["_shirt"].value_counts()
    amb = tuple(sorted(counts[counts > 1].index, key=lambda s: (len(s), s)))
    ok = d[~d["_shirt"].isin(amb) & (d["_shirt"] != "<NA>")]
    by = {r["_shirt"]: (str(r[idc]), r["_name"]) for _, r in ok.iterrows()}
    short = {r["_shirt"]: str(r[shc]) for _, r in ok.iterrows() if shc and pd.notna(r[shc])}
    role = {r["_shirt"]: str(r[rc]) for _, r in ok.iterrows() if rc and pd.notna(r[rc])}
    return Squad(by, amb, source, short, role)


# ------------------------------------------------------------------ bundled defaults
_DATA = Path(__file__).with_name("data")


def bundled_tables() -> dict:
    """The club's reference tables shipped with the app (EPL 2A teams + matches, Masar squad 26/27).
    Used whenever nothing newer has been uploaded."""
    out = {}
    for k, f in (("teams", "teams.csv"), ("matches", "matches.csv"), ("squad", "masar_squad.csv")):
        fp = _DATA / f
        out[k] = pd.read_csv(fp, dtype=str, keep_default_na=False).replace({"": None}) if fp.exists() else None
    return out


# ------------------------------------------------------------------ bundle
@dataclass
class Reference:
    teams: pd.DataFrame | None = None
    matches: pd.DataFrame | None = None
    squad: Squad = field(default_factory=default_squad)

    @classmethod
    def bundled(cls) -> "Reference":
        t = bundled_tables()
        squad = default_squad()
        if isinstance(t.get("squad"), pd.DataFrame):
            try:
                squad = squad_from_table(t["squad"], "Masar Squad Details 26-27 (built-in)")
            except ValueError:
                pass
        return cls(t.get("teams"), t.get("matches"), squad)

    def auto_match_id(self, md) -> str:
        """The match_id when exactly one row of the match table has these two teams."""
        m = self.resolve(md)["matches"]
        return str(m["match_id"].iloc[0]) if isinstance(m, pd.DataFrame) and len(m) == 1 else ""

    def resolve(self, md) -> dict:
        """IDs for one loaded match: per team {id, official, method} + candidate match rows."""
        out = {"teams": {t: match_team(t, self.teams) for t in md.teams}}
        a, b = (md.teams + ["", ""])[:2]
        out["matches"] = find_matches(a, b, self.matches, self.teams)
        return out
