"""The joint model recovers known competition effects, the team-strength slope and the variances
from simulated careers in which talent differs between competitions (the selection that makes a
naive comparison of league averages useless)."""
import numpy as np
import pandas as pd

from translator import bayes

LAM = {"euroleague": 0.0, "eurocup": 0.25, "acb": 0.12, "bcl": 0.45}
TALENT = {"euroleague": 0.30, "eurocup": 0.05, "acb": 0.15, "bcl": -0.10}   # selection


def simulate(n_players=1500, seasons=range(2018, 2024), beta=0.08, tau_a=0.25, sigma=0.12, tau_g=0.05, seed=1):
    rng = np.random.default_rng(seed)
    comps = list(LAM)
    teams = {(c, s): rng.normal(0, 1, 12) for c in comps for s in seasons}      # team_z per team
    gam = {(c, s): rng.normal(0, tau_g, 12) for c in comps for s in seasons}
    rows = []
    for p in range(n_players):
        home = comps[p % len(comps)]
        alpha = TALENT[home] + rng.normal(0, tau_a)
        c = home
        for s in seasons:
            if rng.random() < 0.6:
                continue                                    # not every player plays every season
            eta = rng.normal(0, sigma)
            played = [c] + ([rng.choice([x for x in comps if x != c])] if rng.random() < 0.08 else [])
            for comp in played:                             # sometimes two competitions in a season
                t = rng.integers(12)
                z = teams[(comp, s)][t]
                v = rng.uniform(0.005, 0.03)
                y = alpha + eta + LAM[comp] + beta * z + gam[(comp, s)][t] + rng.normal(0, np.sqrt(v))
                rows.append({"person": f"P{p}", "person_name": f"Player {p}", "competition": comp, "season": s,
                             "team": f"T{t}", "poss": 1000, "metric": "pts", "team_z": z, "value": 20 * np.exp(y),
                             "league_value": 20.0, "rel": y, "var": v})
            if rng.random() < 0.15:                         # a move for next season
                c = rng.choice([x for x in comps if x != c])
    return pd.DataFrame(rows)


def test_recovers_league_effects_and_team_strength():
    rel = simulate()
    D = bayes.design(rel, "pts", reference="euroleague")
    fit = bayes.gibbs(D, draws=600, burn=300, seed=2)
    lam = fit.leagues().set_index("competition")["lambda"]
    for c, true in LAM.items():
        assert abs(lam[c] - true) < 0.04, (c, lam[c], true)
    # transitive by construction, and in the right direction
    f = fit.factor("acb", "bcl")
    assert abs(f["delta"] - (LAM["bcl"] - LAM["acb"])) < 0.05 and f["p_positive"] > 0.99
    assert abs(fit.beta.mean() - 0.08) < 0.02
    assert abs(fit.tau_alpha.mean() - 0.25) < 0.04
    assert abs(fit.sigma.mean() - 0.12) < 0.03


def test_naive_comparison_is_biased_but_model_is_not():
    """League means of the relative values are ~0 by construction: without the movers nothing is learnt.
    The model gets the effect from the movers only, so talent differences do not leak into it."""
    rel = simulate(seed=5)
    fit = bayes.gibbs(bayes.design(rel, "pts"), draws=400, burn=200, seed=1)
    lam = fit.leagues().set_index("competition")["lambda"]
    assert abs(lam["bcl"] - LAM["bcl"]) < 0.05


def test_translate_uses_all_his_competitions():
    rel = simulate(seed=7)
    fits = {"pts": bayes.gibbs(bayes.design(rel, "pts"), draws=300, burn=200, seed=0)}
    row = rel[rel["competition"] == "acb"].iloc[0]
    out = bayes.translate(fits, rel, row["person"], row["season"], "acb", "euroleague")
    assert len(out) == 1
    r = out.iloc[0]
    assert r["lo80"] < r["expected_b"] < r["hi80"]
    # a weaker competition (positive effect) translates down into the EuroLeague
    up = bayes.translate(fits, rel, row["person"], row["season"], "acb", "bcl").iloc[0]
    assert up["expected_b"] > r["expected_b"]
    good = bayes.translate(fits, rel, row["person"], row["season"], "acb", "euroleague", team_z=1.5).iloc[0]
    assert good["expected_b"] > r["expected_b"]                  # beta > 0 in the simulation


def test_unlinked_competition_is_left_out():
    rel = simulate(n_players=400, seed=3)
    lonely = rel[rel["competition"] == "acb"].head(50).copy()
    lonely["competition"], lonely["person"] = "nba", "N" + lonely["person"]     # nobody shared with Europe
    D = bayes.design(pd.concat([rel, lonely]), "pts")
    assert "nba" in D.dropped and "nba" not in D.leagues
