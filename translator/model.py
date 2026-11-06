"""League translation factors with their uncertainty.

Every statistic is first expressed **relative to its league-season** (so that the era and the
style of each league are taken out), then the change in that relative value is measured on the
players who played in both competitions:

* **Same season** (design S): players in A and B in the same season (e.g. a EuroLeague club that
  also plays the ACB). Both values are measured at the same moment, so the difference is the
  league effect plus noise.
* **Consecutive seasons** (design C): players in A in season t and in B in season t+1, and in
  only one of them each season (real moves). The difference also contains a year-to-year change
  (development, ageing, regression to the mean); it is removed with the change observed, in
  the same years, on players who stayed in A.

Each design is pooled with a random-effects model (DerSimonian-Laird): the league effect δ has a
standard error and a between-player spread τ (not every player translates the same way). The
two designs are combined by precision and their agreement is tested (I² and a z test): if
both agree, the factor is solid.

Scales:
  rates per 100 possessions   rel = log(rate / league rate)      factor = exp(δ)  (×)
  percentages                 rel = logit(p) − logit(league p)   applied on the logit scale
  PIR and +/- per 100         rel = value − league value         added (+)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATES = {   # metric: (counts column, label)
    "pts": ("pts", "Points /100"), "fga": ("fga", "FGA /100"), "fg3a": ("fg3a", "3PA /100"),
    "fta": ("fta", "FTA /100"), "ast": ("ast", "Assists /100"), "tov": ("tov", "Turnovers /100"),
    "orb": ("orb", "Off. rebounds /100"), "drb": ("drb", "Def. rebounds /100"), "stl": ("stl", "Steals /100"),
    "blk": ("blk", "Blocks /100"), "pf": ("pf", "Fouls /100"), "fd": ("fouls_drawn", "Fouls drawn /100"),
}
PCTS = {    # metric: (made, attempts, label)
    "fg2_pct": ("fg2_made", "fg2a", "2P%"), "fg3_pct": ("fg3_made", "fg3a", "3P%"), "ft_pct": ("ft_made", "fta", "FT%"),
    "ts_pct": ("pts_half", "tsa", "TS%"),
}
LEVELS = {  # metric: (column, residual column, label)
    "pir": ("pir", "pir_res2", "PIR /100"), "net": ("plus_minus", "plus_minus_res2", "+/- /100"),
}
LABELS = {**{k: v[1] for k, v in RATES.items()}, **{k: v[2] for k, v in PCTS.items()}, **{k: v[2] for k, v in LEVELS.items()}}
KIND = {**{k: "rate" for k in RATES}, **{k: "pct" for k in PCTS}, **{k: "level" for k in LEVELS}}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def relative(ps: pd.DataFrame, min_poss: float = 150) -> pd.DataFrame:
    """Long table: one row per player-season and metric, with rel (vs league-season) and its variance."""
    ps = ps[ps["poss"] >= min_poss].copy()
    ps["tsa"] = ps["fga"] + 0.44 * ps["fta"]
    ps["pts_half"] = ps["pts"] / 2                      # TS% = pts / (2·TSA): "made" = pts/2 over TSA attempts
    league = ps.groupby(["competition", "season"])
    rows = []
    for m, (col, _) in RATES.items():
        L = league[col].transform("sum") / league["poss"].transform("sum")
        x = ps[col].clip(lower=0)
        rel = np.log((x + 0.5) / ps["poss"]) - np.log(L.where(L > 0))   # a stat a league does not record: NaN
        rows.append(_frame(ps, m, rel, 1 / (x + 0.5), x / ps["poss"] * 100, L * 100))
    for m, (made, att, _) in PCTS.items():
        n = ps[att]
        ok = n >= (20 if m != "ft_pct" else 10)
        Lp = league[made].transform("sum") / league[att].transform("sum")
        p = (ps[made] + 0.5) / (n + 1)
        rel = _logit(p) - _logit(Lp)
        var = 1 / (n.clip(lower=1) * p * (1 - p))
        rows.append(_frame(ps[ok], m, rel[ok], var[ok], (ps[made] / n * 100)[ok], (Lp * 100)[ok]))
    for m, (col, res, _) in LEVELS.items():
        if col not in ps:
            continue
        L = league[col].transform("sum") / league["poss"].transform("sum") * 100
        val = ps[col] / ps["poss"] * 100
        var = (ps[res] / ps["poss"] ** 2 * 1e4).clip(lower=1e-3) if res in ps else (val.var() / 30)
        rows.append(_frame(ps, m, val - L, var, val, L))
    return pd.concat(rows, ignore_index=True)


def _frame(ps, m, rel, var, value, league_value):
    return pd.DataFrame({"person": ps["person"].values, "person_name": ps["person_name"].values,
                         "competition": ps["competition"].values, "season": ps["season"].values,
                         "team": ps["team"].values, "poss": ps["poss"].values, "metric": m,
                         "team_z": ps["team_z"].fillna(0.0).values if "team_z" in ps else 0.0,
                         "value": np.asarray(value), "league_value": np.asarray(league_value),
                         "rel": np.asarray(rel), "var": np.asarray(var)})


# ─────────────────────────────────────────────────────────────── pairs
def pairs_same_season(rel: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    A, B = rel[rel["competition"] == a], rel[rel["competition"] == b]
    m = A.merge(B, on=["person", "season", "metric"], suffixes=("_a", "_b"))
    m["d"], m["v"] = m["rel_b"] - m["rel_a"], m["var_a"] + m["var_b"]
    m["design"] = "same season"
    return m


def pairs_consecutive(rel: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Moves a(t) -> b(t+1): the player is not in b at t nor in a at t+1."""
    where = rel.groupby(["person", "season"])["competition"].agg(set)
    A, B = rel[rel["competition"] == a].copy(), rel[rel["competition"] == b].copy()
    A["season_next"] = A["season"] + 1
    m = A.merge(B, left_on=["person", "season_next", "metric"], right_on=["person", "season", "metric"],
                suffixes=("_a", "_b"))
    move = [b not in where.get((p, s), set()) and a not in where.get((p, s + 1), set())
            for p, s in zip(m["person"], m["season_a"])]
    m = m[np.array(move, dtype=bool)].copy()
    m["d"], m["v"] = m["rel_b"] - m["rel_a"], m["var_a"] + m["var_b"]
    m["design"] = "consecutive"
    return m


def year_to_year(rel: pd.DataFrame, comp: str) -> pd.DataFrame:
    """Players who stay in the same competition from t to t+1: the typical change of a returning player."""
    A = rel[rel["competition"] == comp].copy()
    A["season_next"] = A["season"] + 1
    m = A.merge(A, left_on=["person", "season_next", "metric"], right_on=["person", "season", "metric"],
                suffixes=("_a", "_b"))
    m["d"], m["v"] = m["rel_b"] - m["rel_a"], m["var_a"] + m["var_b"]
    return m


def shrink(rel: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes value of every player-season: rel pulled toward his league by its noise.
    Adds rel_s (shrunk value) and var_s (its posterior variance)."""
    rel = rel.copy()
    g = rel.groupby(["competition", "season", "metric"])
    prior = (g["rel"].transform("var") - g["var"].transform("mean")).clip(lower=1e-4)
    k = prior / (prior + rel["var"])
    rel["rel_s"], rel["var_s"] = k * rel["rel"], k * rel["var"]
    return rel


def move_factor(rel: pd.DataFrame, a: str, b: str, min_pairs: int = 8) -> pd.DataFrame:
    """Next-season projection from a (season t) to b (season t+1), for every metric.

    Measured on the real moves a(t) → b(t+1) (or, when a == b, on players who stayed), from the
    player's shrunk value in t to his observed value in t+1. It therefore includes everything that
    comes with a move (league level, new role, development and ageing), which is what a projection
    for a signing needs. With fewer than `min_pairs` moves it falls back to the league effect plus
    the usual year-to-year change, and says so in `move_source`."""
    rs = shrink(rel)
    pairs = year_to_year(rs, a) if a == b else pairs_consecutive(rs, a, b)
    if len(pairs):
        pairs = pairs.assign(d=pairs["rel_b"] - pairs["rel_s_a"], v=pairs["var_b"] + pairs["var_s_a"])
    out = []
    fac = factors(rel, a, b, min_pairs) if a != b else None
    for m in rel["metric"].unique():
        pm = pairs[pairs["metric"] == m] if len(pairs) else pairs
        r = random_effects(pm["d"], pm["v"]) if len(pm) else random_effects([], [])
        src = "moves" if a != b else "stayers"
        if r["n"] < min_pairs and fac is not None:
            f = fac[fac["metric"] == m].iloc[0]
            y = f["year_to_year"] if np.isfinite(f["year_to_year"]) else 0.0
            r = {"n": r["n"], "delta": f["delta"] + y, "se": f["se"], "tau": f["tau"], "i2": np.nan}
            src = "league effect + year-to-year"
        out.append({"from": a, "to": b, "metric": m, "label": LABELS[m], "kind": KIND[m], "n_move": r["n"],
                    "delta": r["delta"], "se": r["se"], "tau": r["tau"], "move_source": src})
    f = pd.DataFrame(out)
    f["factor"] = np.where(f["kind"] == "rate", np.exp(f["delta"]), np.nan)
    return f.sort_values("metric").reset_index(drop=True)


# ─────────────────────────────────────────────────────────── pooling
def random_effects(d: np.ndarray, v: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects mean, its SE, between-player SD (tau) and I²."""
    d, v = np.asarray(d, float), np.asarray(v, float)
    ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
    d, v = d[ok], v[ok]
    k = len(d)
    if k == 0:
        return {"n": 0, "delta": np.nan, "se": np.nan, "tau": np.nan, "i2": np.nan}
    w = 1 / v
    fixed = np.sum(w * d) / np.sum(w)
    q = np.sum(w * (d - fixed) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    w2 = 1 / (v + tau2)
    delta = np.sum(w2 * d) / np.sum(w2)
    se = np.sqrt(1 / np.sum(w2))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"n": int(k), "delta": float(delta), "se": float(se), "tau": float(np.sqrt(tau2)), "i2": float(i2)}


def factors(rel: pd.DataFrame, a: str, b: str, min_pairs: int = 8) -> pd.DataFrame:
    """Translation from a to b for every metric, by design and combined."""
    same = pairs_same_season(rel, a, b)
    cons = pairs_consecutive(rel, a, b)
    stay = year_to_year(rel, a)
    out = []
    for m in rel["metric"].unique():
        rs = random_effects(*same.loc[same["metric"] == m, ["d", "v"]].to_numpy().T) \
            if (same["metric"] == m).any() else random_effects([], [])
        rc = random_effects(*cons.loc[cons["metric"] == m, ["d", "v"]].to_numpy().T) \
            if (cons["metric"] == m).any() else random_effects([], [])
        # year-to-year change of stayers in the seasons where the moves happened
        yrs = set(cons.loc[cons["metric"] == m, "season_a"])
        st = stay[(stay["metric"] == m) & stay["season_a"].isin(yrs)]
        ry = random_effects(st["d"], st["v"]) if len(st) else random_effects([], [])
        if rc["n"] and ry["n"]:
            rc_adj = {**rc, "delta": rc["delta"] - ry["delta"], "se": float(np.hypot(rc["se"], ry["se"]))}
        else:
            rc_adj = rc
        parts = [r for r in (rs, rc_adj) if r["n"] >= min_pairs and np.isfinite(r["se"])]
        if parts:
            w = np.array([1 / r["se"] ** 2 for r in parts])
            comb = float(np.sum(w * np.array([r["delta"] for r in parts])) / w.sum())
            comb_se = float(np.sqrt(1 / w.sum()))
            tau = float(np.sqrt(np.sum(w * np.array([r["tau"] ** 2 for r in parts])) / w.sum()))
        else:
            comb, comb_se, tau = np.nan, np.nan, np.nan
        z = (rs["delta"] - rc_adj["delta"]) / np.hypot(rs["se"], rc_adj["se"]) \
            if rs["n"] >= min_pairs and rc_adj["n"] >= min_pairs else np.nan
        out.append({"from": a, "to": b, "metric": m, "label": LABELS[m], "kind": KIND[m],
                    "n_same": rs["n"], "delta_same": rs["delta"], "se_same": rs["se"],
                    "n_consecutive": rc["n"], "delta_consecutive": rc_adj["delta"], "se_consecutive": rc_adj["se"],
                    "year_to_year": ry["delta"], "delta": comb, "se": comb_se, "tau": tau,
                    "designs_agree_z": z})
    f = pd.DataFrame(out)
    f["factor"] = np.where(f["kind"] == "rate", np.exp(f["delta"]), np.nan)
    f["lo95"], f["hi95"] = f["delta"] - 1.96 * f["se"], f["delta"] + 1.96 * f["se"]
    return f.sort_values("metric").reset_index(drop=True)


# ───────────────────────────────────────────────────────────── translate
def translate(rel: pd.DataFrame, fac: pd.DataFrame, person: str, season: int, a: str, b: str,
              target_season: int | None = None, next_season: bool = False) -> pd.DataFrame:
    """Expected stats of one player in competition b, from his season in a, with 80% intervals.

    The player's relative value is shrunk toward his league (empirical Bayes, how much depends on
    how many possessions he played), then moved by the league effect; the interval adds the
    uncertainty of the factor and the spread between players (tau)."""
    target_season = target_season or (season + 1 if next_season else season)
    me = rel[(rel["person"] == person) & (rel["season"] == season) & (rel["competition"] == a)]
    seasons_b = sorted(rel.loc[rel["competition"] == b, "season"].unique())
    if target_season not in seasons_b and seasons_b:     # nearest season of b (the latest one before, if any)
        before = [x for x in seasons_b if x <= target_season]
        target_season = before[-1] if before else seasons_b[0]
    ref_b = rel[(rel["competition"] == b) & (rel["season"] == target_season)].groupby("metric")["league_value"].first()
    rows = []
    for r in me.itertuples():
        f = fac[fac["metric"] == r.metric]
        if f.empty or not np.isfinite(f["delta"].iloc[0]) or r.metric not in ref_b:
            continue
        f = f.iloc[0]
        pool = rel[(rel["competition"] == a) & (rel["season"] == season) & (rel["metric"] == r.metric)]
        prior = max(pool["rel"].var() - pool["var"].mean(), 1e-4)     # true spread between players
        k = prior / (prior + r.var)
        post, post_var = k * r.rel, k * r.var
        mu = post + f["delta"]
        sd = np.sqrt(post_var + f["se"] ** 2 + f["tau"] ** 2)
        lo, hi = mu - 1.2816 * sd, mu + 1.2816 * sd
        L = ref_b[r.metric]
        conv = _from_rel(r.metric, L)
        rows.append({"metric": r.metric, "label": LABELS[r.metric], "in_a": r.value, "league_a": r.league_value,
                     "expected_b": conv(mu), "lo80": conv(lo), "hi80": conv(hi), "league_b": L,
                     "shrinkage": 1 - k})
    return pd.DataFrame(rows, columns=["metric", "label", "in_a", "league_a", "expected_b", "lo80", "hi80", "league_b",
                                       "shrinkage"])


def _from_rel(metric: str, league_value: float):
    kind = KIND[metric]
    if kind == "rate":
        return lambda x: league_value * np.exp(x)
    if kind == "pct":
        lp = _logit(league_value / 100)
        return lambda x: 100 / (1 + np.exp(-(lp + x)))
    return lambda x: league_value + x


def league_trends(rel: pd.DataFrame) -> pd.DataFrame:
    """League value of each metric by competition and season (the era drift)."""
    return rel.groupby(["competition", "season", "metric"])["league_value"].first().unstack("metric").reset_index()


def common_scale(all_factors: pd.DataFrame, reference: str = "euroleague") -> pd.DataFrame:
    """Every competition on one scale per metric (reference = 0), by weighted least squares over all
    the pairwise effects. The residuals show how far the pairs are from being transitive
    (A→B + B→C = A→C); large residuals point to selection or team-context effects."""
    out = []
    for m, f in all_factors.dropna(subset=["delta", "se"]).groupby("metric"):
        comps = sorted(set(f["from"]) | set(f["to"]))
        if reference not in comps:
            continue
        free = [c for c in comps if c != reference]
        idx = {c: i for i, c in enumerate(free)}
        X = np.zeros((len(f), len(free)))
        for row, r in enumerate(f.itertuples()):
            if r.to in idx:
                X[row, idx[r.to]] += 1
            if r._1 in idx:            # "from" is a keyword: itertuples names it _1
                X[row, idx[r._1]] -= 1
        y, w = f["delta"].to_numpy(), 1 / f["se"].to_numpy() ** 2
        W = np.diag(w)
        cov = np.linalg.pinv(X.T @ W @ X)
        theta = cov @ X.T @ W @ y
        resid = y - X @ theta
        for c in comps:
            i = idx.get(c)
            out.append({"metric": m, "competition": c, "theta": 0.0 if i is None else float(theta[i]),
                        "se": 0.0 if i is None else float(np.sqrt(cov[i, i])),
                        "chi2_inconsistency": float(np.sum(w * resid ** 2)), "df": len(f) - len(free)})
    return pd.DataFrame(out)
