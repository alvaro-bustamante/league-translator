"""Player-season table from the hoops-warehouse game tables.

One row per player, competition and season, with possessions on court and totals of every
counting statistic. Per-100-possession rates and the league-season reference come from here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

COUNTS = ["pts", "fg2_made", "fg2_missed", "fg3_made", "fg3_missed", "ft_made", "ft_missed", "orb", "drb",
          "ast", "stl", "tov", "blk", "blk_against", "pf", "fouls_drawn", "pir", "plus_minus"]
NAMES = {"euroleague": "EuroLeague", "eurocup": "EuroCup", "acb": "Liga Endesa ACB", "bcl": "Basketball Champions League",
         "fibaeurocup": "FIBA Europe Cup", "primerafeb": "Primera FEB", "segundafeb": "Segunda FEB", "nba": "NBA",
         "lfendesa": "LF Endesa", "elwomen": "EuroLeague Women"}


def warehouse_dir() -> Path:
    for c in (os.environ.get("HOOPS_WAREHOUSE"), Path(__file__).resolve().parents[2] / "hoops-warehouse" / "data" /
              "warehouse", Path("data") / "warehouse"):
        if c and Path(c).is_dir() and any(Path(c).glob("*_player_games.csv")):
            return Path(c)
    raise FileNotFoundError("No hoops-warehouse game tables found: set HOOPS_WAREHOUSE to the folder with "
                            "<comp>_<season>_player_games.csv files.")


def available(base: Path | None = None) -> pd.DataFrame:
    base = base or warehouse_dir()
    rows = []
    for f in base.glob("*_player_games.csv"):
        comp, _, season = f.name[:-len("_player_games.csv")].rpartition("_")
        if season.isdigit():
            rows.append({"competition": comp, "season": int(season), "file": f})
    return pd.DataFrame(rows).sort_values(["competition", "season"]).reset_index(drop=True)


def player_games(competitions: list[str] | None = None, seasons: list[int] | None = None,
                 base: Path | None = None) -> pd.DataFrame:
    av = available(base)
    if competitions:
        av = av[av["competition"].isin(competitions)]
    if seasons:
        av = av[av["season"].isin(seasons)]
    frames = []
    for r in av.itertuples():
        df = pd.read_csv(r.file, dtype={"game_id": str, "player_id": str}, low_memory=False)
        df["competition"], df["season"] = r.competition, r.season
        frames.append(df)
    if not frames:
        raise ValueError("No game tables match that selection.")
    return pd.concat(frames, ignore_index=True)


def player_seasons(pg: pd.DataFrame) -> pd.DataFrame:
    """Totals per player, competition and season (players who played)."""
    pg = pg[pg["seconds"].fillna(0) > 0].copy()
    # possessions on court: counted from the play by play; where lineups were unreliable (some old ACB
    # games) or the source is box-only, estimated from minutes and the team's possessions
    team_poss = pg.groupby(["competition", "season", "game_id", "team"])["seconds"].transform("sum") / 5
    if "possessions" not in pg:
        pg["possessions"] = np.nan
    est = pg["seconds"] / team_poss.replace(0, np.nan)
    tg_poss = pg.groupby(["competition", "season", "game_id", "team"])["possessions"].transform("sum") / 5
    pg["poss"] = pg["possessions"].where(pg["possessions"].notna() & (tg_poss > 0), est * tg_poss.where(tg_poss > 0))
    pg["poss"] = pg["poss"].fillna(pg["seconds"] / 60 * 1.8)          # last resort: ~72 possessions per 40'
    team_net = team_strength(pg)
    keys = ["competition", "season", "player_id"]
    g = pg.groupby(keys)
    out = g[[c for c in COUNTS if c in pg]].sum()
    # spread of the per-game values, for the sampling variance of ratio statistics (PIR, +/- per 100)
    for c in ("pir", "plus_minus"):
        if c in pg:
            rate = g[c].transform("sum") / g["poss"].transform("sum").replace(0, np.nan)
            pg[f"_{c}_res2"] = (pg[c].fillna(0) - rate * pg["poss"]) ** 2
            out[f"{c}_res2"] = pg.groupby(keys)[f"_{c}_res2"].sum()
    out["poss"] = g["poss"].sum()
    out["seconds"] = g["seconds"].sum()
    out["games"] = g["game_id"].nunique()
    out["player_name"] = g["player_name"].last()
    out["team"] = g["team"].agg(lambda s: s.value_counts().index[0])
    out["team_name"] = g["team_name"].agg(lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None)
    out = out.reset_index()
    out = out.merge(team_net, on=["competition", "season", "team"], how="left")
    out["fga"] = out["fg2_made"] + out["fg2_missed"] + out["fg3_made"] + out["fg3_missed"]
    out["fta"] = out["ft_made"] + out["ft_missed"]
    out["fg3a"] = out["fg3_made"] + out["fg3_missed"]
    out["fg2a"] = out["fg2_made"] + out["fg2_missed"]
    return out


def team_strength(pg: pd.DataFrame) -> pd.DataFrame:
    """Net rating of every team-season (points scored minus allowed per 100 possessions) and the same
    value in standard deviations of its league-season (`team_z`). Built from the player rows only, so
    it also works for box-score-only sources: the opponent's points are the opponent's player points
    in the same game. Needs a `poss` column (see player_seasons)."""
    keys = ["competition", "season", "game_id", "team"]
    tg = pg.groupby(keys).agg(pts=("pts", "sum"), poss=("poss", "sum"), opponent=("opponent", "first")).reset_index()
    tg["poss"] = tg["poss"] / 5
    opp = tg[keys + ["pts"]].rename(columns={"team": "opponent", "pts": "pts_against"})
    tg = tg.merge(opp, on=["competition", "season", "game_id", "opponent"], how="left")
    tg = tg[tg["pts_against"].notna() & (tg["poss"] > 0)]
    ts = tg.groupby(["competition", "season", "team"]).agg(margin=("pts", "sum"), against=("pts_against", "sum"),
                                                           poss=("poss", "sum")).reset_index()
    ts["team_net"] = (ts["margin"] - ts["against"]) / ts["poss"] * 100
    g = ts.groupby(["competition", "season"])["team_net"]
    ts["team_z"] = ((ts["team_net"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    return ts[["competition", "season", "team", "team_net", "team_z"]]
