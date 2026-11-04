"""One id per person across competitions.

Each source has its own player ids (EuroLeague and EuroCup share theirs; the ACB, FEB, FIBA and
NBA each have their own) and writes names differently ("LLULL, SERGIO" vs "Sergio Llull").
People are linked in two passes:

1. Same normalised name (accents, case, punctuation and order removed). A name that belongs to
   more than one id inside the same source is ambiguous and is left out of this pass.
2. For players still unlinked: same surname, a compatible first name (same initial, or one name
   contained in the other) and the same club in the same season. Clubs are matched from the
   players linked in pass 1 (a club is the team most of its players come from).

Every link records how it was made, so that it can be audited (links.csv).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

ID_FAMILY = {"euroleague": "euroleague", "eurocup": "euroleague"}   # competitions that share player ids
SUFFIXES = {"JR", "SR", "II", "III", "IV"}


def family(competition: str) -> str:
    return ID_FAMILY.get(competition, competition)


def tokens(name: str) -> list[str]:
    if not isinstance(name, str):
        return []
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper()
    if "," in n:                                    # "LLULL, SERGIO" -> "SERGIO LLULL"
        last, _, first = n.partition(",")
        n = f"{first} {last}"
    return [t for t in re.findall(r"[A-Z]+", n) if t not in SUFFIXES]


def name_key(name: str) -> str:
    return " ".join(sorted(tokens(name)))


def surname(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper()
    if "," in n:
        return " ".join(re.findall(r"[A-Z]+", n.partition(",")[0]))
    t = [x for x in re.findall(r"[A-Z]+", n) if x not in SUFFIXES]
    return t[-1] if t else ""


def link(ps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds `person` to the player-season table; returns (table, links)."""
    ps = ps.copy()
    ps["family"] = ps["competition"].map(family)
    ps["source_id"] = ps["family"] + ":" + ps["player_id"].astype(str)
    ps["key"] = ps["player_name"].map(name_key)
    people = ps.drop_duplicates("source_id")[["source_id", "family", "key", "player_name"]]

    # ids per (family, name): ambiguous names have more than one id in the same family
    per_family = people.groupby(["family", "key"])["source_id"].nunique()
    ambiguous = set(per_family[per_family > 1].index)
    person = {}
    links = []
    by_key = defaultdict(list)
    for r in people.itertuples():
        if (r.family, r.key) not in ambiguous and r.key:
            by_key[r.key].append(r.source_id)
    for key, ids in by_key.items():
        root = sorted(ids)[0]
        for sid in ids:
            person[sid] = root
            if sid != root:
                links.append({"source_id": sid, "person": root, "how": "same name", "name": key})

    # pass 2: same surname + compatible first name + same club in the same season
    ps["person"] = ps["source_id"].map(person).fillna(ps["source_id"])
    club = _club_map(ps)
    ps["club"] = [club.get((f, t), f"{f}:{t}") for f, t in zip(ps["family"], ps["team"])]
    linked_ids = {sid for sid, p in person.items() if sid != p} | set(person.values())
    multi = ps.groupby("person")["family"].nunique()
    singles = ps[ps["person"].map(multi) == 1]
    cand = defaultdict(list)
    for r in singles.drop_duplicates(["source_id", "season", "club"]).itertuples():
        cand[(r.season, r.club, surname(r.player_name))].append(r)
    for (season, clb, sur), rows in cand.items():
        fams = defaultdict(list)
        for r in rows:
            fams[r.family].append(r)
        if len(fams) < 2 or not sur:
            continue
        fam_list = list(fams)
        base = fams[fam_list[0]]
        for other in fam_list[1:]:
            if len(base) == 1 and len(fams[other]) == 1:
                a, b = base[0], fams[other][0]
                if _first_names_agree(a.player_name, b.player_name):
                    root = person.get(a.source_id, a.source_id)
                    for sid in (b.source_id,):
                        if sid in linked_ids:
                            continue
                        person[sid] = root
                        links.append({"source_id": sid, "person": root, "how": "surname + club + season",
                                      "name": f"{a.player_name} = {b.player_name}"})
    ps["person"] = ps["source_id"].map(person).fillna(ps["source_id"])
    # person name: the most readable spelling (one without a comma)
    readable = ps.assign(_c=ps["player_name"].str.contains(",", na=False)).sort_values("_c") \
        .groupby("person")["player_name"].first()
    ps["person_name"] = ps["person"].map(readable)
    return ps.drop(columns=["key"]), pd.DataFrame(links)


def _first_names_agree(a: str, b: str) -> bool:
    ta, tb = tokens(a), tokens(b)
    sa, sb = surname(a), surname(b)
    fa = [t for t in ta if t not in sa.split()]
    fb = [t for t in tb if t not in sb.split()]
    if not fa or not fb:
        return True
    return fa[0][0] == fb[0][0] or any(x in y or y in x for x in fa for y in fb)


def _club_map(ps: pd.DataFrame) -> dict:
    """(family, team code) -> club label, from players already linked across families."""
    linked = ps[ps.groupby("person")["family"].transform("nunique") > 1]
    votes = defaultdict(Counter)
    for (person, season), g in linked.groupby(["person", "season"]):
        teams = list(zip(g["family"], g["team"]))
        for f1, t1 in teams:
            for f2, t2 in teams:
                if f1 < f2:
                    votes[(f1, t1)][(f2, t2)] += 1
    club = {}
    for a, c in votes.items():
        (b, n), = c.most_common(1)
        if n >= 3:                                  # at least three shared players that season
            label = f"{a[0]}:{a[1]}"
            club[a] = label
            club.setdefault(b, label)
    return club
