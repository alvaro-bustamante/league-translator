"""Translator v2: one hierarchical Bayesian model for every competition at once.

For each metric, the relative value of a player-season in a competition (see model.relative) is

    rel[p, c, s] = theta[p, s] + lambda[c] + beta * team_z + gamma[team-season] + e

    theta[p, s] = alpha[p] + eta[p, s]           his level that season (shared by every competition
                                                  he played that season)
    alpha[p]  ~ N(m[group(p)], tau_alpha²)        career level; group = the set of competitions he
                                                  played, so a league's typical talent is not forced
                                                  to be the same everywhere
    eta[p, s] ~ N(0, sigma²)                      season-to-season change
    gamma     ~ N(0, tau_gamma²)                  team context not explained by team strength
    e         ~ N(0, var)                         sampling noise, known from his possessions/attempts
    lambda[c]                                     competition effect, reference competition = 0
    beta                                          team-strength effect (team net rating, in SD of
                                                  its league-season)

The difference lambda[b] − lambda[a] is the translation from a to b. Compared with v1 (pairwise
random-effects meta-analysis), every competition is fitted jointly, so the factors are transitive
by construction (A→B + B→C = A→C), a player who moves through three competitions informs all of
them, and the role of team strength is separated from the league effect.

Two backends with the same model:
* `gibbs` (default, numpy only): blocked Gibbs sampler, every step conjugate. The player level is
  sampled with the season terms integrated out, and exact "ridge" moves shift each competition
  effect against the career level of the players who played there (the direction that only movers
  inform) and against its team effects, and the team-strength slope against the team effects.
  It needs no extra package.
* `pymc` (PyMC ≥ 5, optional): the same model with NUTS, for checking the Gibbs results or
  extending the model (other priors, other likelihoods).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model import KIND, LABELS, _from_rel


# ─────────────────────────────────────────────────────────────── data
@dataclass
class Design:
    metric: str
    y: np.ndarray
    v: np.ndarray
    z: np.ndarray
    player: np.ndarray          # obs -> player
    pseason: np.ndarray         # obs -> player-season
    league: np.ndarray          # obs -> competition
    team: np.ndarray            # obs -> team-season
    ps_player: np.ndarray       # player-season -> player
    group: np.ndarray           # player -> group
    leagues: list
    players: np.ndarray
    pseasons: pd.DataFrame      # person, season
    teams: np.ndarray
    groups: list
    reference: str
    dropped: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.y)


def connected(d: pd.DataFrame, reference: str, min_movers: int = 5) -> set:
    """Competitions reachable from the reference through pairs with at least `min_movers` shared players."""
    comps = d.groupby("person")["competition"].agg(lambda s: sorted(set(s)))
    edges = {}
    for cs in comps[comps.map(len) > 1]:
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                edges[(a, b)] = edges.get((a, b), 0) + 1
    seen, todo = {reference}, [reference]
    while todo:
        a = todo.pop()
        for (x, y), n in edges.items():
            if n >= min_movers and a in (x, y):
                other = y if a == x else x
                if other not in seen:
                    seen.add(other)
                    todo.append(other)
    return seen


def design(rel: pd.DataFrame, metric: str, reference: str = "euroleague", min_movers: int = 5) -> Design:
    d = rel[(rel["metric"] == metric) & np.isfinite(rel["rel"]) & np.isfinite(rel["var"]) & (rel["var"] > 0)]
    d = d.reset_index(drop=True)
    leagues = sorted(d["competition"].unique())
    if reference not in leagues:
        reference = leagues[0]
    # only competitions linked to the reference through players who played in both can be placed on
    # its scale (e.g. NBA rows whose names cannot be matched to the European ones are left out)
    linked = connected(d, reference, min_movers)
    dropped = [c for c in leagues if c not in linked]
    d = d[d["competition"].isin(linked)].reset_index(drop=True)
    leagues = sorted(linked)
    leagues = [reference] + [c for c in leagues if c != reference]
    player_codes, players = pd.factorize(d["person"])
    ps_key = d["person"].astype(str) + "|" + d["season"].astype(str)
    ps_codes, ps_uniques = pd.factorize(ps_key)
    team_key = d["competition"] + ":" + d["season"].astype(str) + ":" + d["team"].astype(str)
    team_codes, teams = pd.factorize(team_key)
    league_codes = d["competition"].map({c: i for i, c in enumerate(leagues)}).to_numpy()
    ps_player = np.zeros(len(ps_uniques), int)
    ps_player[ps_codes] = player_codes
    comps = d.groupby(player_codes)["competition"].agg(lambda s: tuple(sorted(set(s))))
    group_codes, groups = pd.factorize(comps.sort_index())
    ps_table = d.groupby(ps_codes)[["person", "season"]].first().sort_index()
    z = d["team_z"].to_numpy(float) if "team_z" in d else np.zeros(len(d))
    return Design(metric, d["rel"].to_numpy(float), d["var"].to_numpy(float), np.nan_to_num(z), player_codes,
                  ps_codes, league_codes, team_codes, ps_player, group_codes, leagues, np.asarray(players),
                  ps_table.reset_index(drop=True), np.asarray(teams), list(groups), reference, dropped)


# ─────────────────────────────────────────────────────────────── gibbs
@dataclass
class Fit:
    design: Design
    lam: np.ndarray             # draws x leagues (reference column = 0)
    beta: np.ndarray
    tau_alpha: np.ndarray
    tau_gamma: np.ndarray
    sigma: np.ndarray
    alpha_mean: np.ndarray
    alpha_var: np.ndarray
    theta_mean: np.ndarray      # per player-season
    theta_var: np.ndarray
    theta_lam_cov: np.ndarray   # player-season x leagues: Cov(theta, lambda_c)
    alpha_lam_cov: np.ndarray
    backend: str = "gibbs"
    extra: dict = field(default_factory=dict)

    def leagues(self) -> pd.DataFrame:
        """Competition effects on the reference scale, with 90% intervals."""
        rows = []
        for j, c in enumerate(self.design.leagues):
            x = self.lam[:, j]
            rows.append({"metric": self.design.metric, "competition": c, "lambda": x.mean(), "sd": x.std(),
                         "q05": np.quantile(x, .05), "q95": np.quantile(x, .95),
                         "ess": ess(x) if j else np.nan,
                         "players": int(np.sum([c in self.design.groups[g] for g in self.design.group])),
                         "movers": int(np.sum([c in self.design.groups[g] and len(self.design.groups[g]) > 1
                                               for g in self.design.group]))})
        return pd.DataFrame(rows)

    def factor(self, a: str, b: str) -> dict:
        L = self.design.leagues
        x = self.lam[:, L.index(b)] - self.lam[:, L.index(a)]
        return {"from": a, "to": b, "metric": self.design.metric, "label": LABELS[self.design.metric],
                "kind": KIND[self.design.metric], "delta": x.mean(), "se": x.std(),
                "lo95": np.quantile(x, .025), "hi95": np.quantile(x, .975),
                "p_positive": (x > 0).mean(),
                "factor": np.exp(x.mean()) if KIND[self.design.metric] == "rate" else np.nan}

    def hyper(self) -> dict:
        def s(x):
            return {"mean": float(np.mean(x)), "q05": float(np.quantile(x, .05)), "q95": float(np.quantile(x, .95))}
        return {"metric": self.design.metric, "n_obs": self.design.n, "n_players": len(self.design.players),
                "n_movers": int(np.sum([len(self.design.groups[g]) > 1 for g in self.design.group])),
                "beta": s(self.beta), "tau_alpha": s(self.tau_alpha), "tau_gamma": s(self.tau_gamma),
                "sigma": s(self.sigma)}


def ess(x: np.ndarray) -> float:
    """Effective sample size of one chain (initial positive sequence of autocorrelations)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or not np.any(x):
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (x @ x)
    s = 0.0
    for k in range(1, n - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        s += pair
    return float(n / (1 + 2 * s))


def _ig(rng, a, b):
    return b / rng.gamma(a)


def gibbs(D: Design, draws: int = 1500, burn: int = 500, seed: int = 0, thin: int = 1) -> Fit:
    rng = np.random.default_rng(seed)
    y, v, z = D.y, D.v, D.z
    w = 1 / v
    L, P, Q, T, G = len(D.leagues), len(D.players), len(D.pseasons), len(D.teams), len(D.groups)
    scale = max(np.var(y), 1e-3)
    s_lam2, s_beta2, big = 4 * scale, 4 * scale, 100 * scale      # weak priors
    a0, b0 = 1.0, 1e-3 * scale                                   # inverse-gamma priors of the variances

    # sufficient structure
    w_q = np.bincount(D.pseason, w, Q)
    n_g = np.bincount(D.group, minlength=G)
    in_group = np.array([[D.leagues[j] in D.groups[g] for j in range(L)] for g in range(G)])  # G x L
    player_has = in_group[D.group]                              # P x L
    obs_has = player_has[D.player]                              # obs x L
    team_league = np.zeros(T, int)
    team_league[D.team] = D.league
    lam = np.zeros(L)
    beta = 0.0
    alpha = np.zeros(P)
    eta = np.zeros(Q)
    gamma = np.zeros(T)
    m = np.zeros(G)
    ta2, tg2, s2 = 0.5 * scale, 0.1 * scale, 0.2 * scale
    step, n_acc, n_try = 0.1, 0, 0

    keep = []
    acc = {"a1": np.zeros(P), "a2": np.zeros(P), "t1": np.zeros(Q), "t2": np.zeros(Q),
           "tl": np.zeros((Q, L)), "al": np.zeros((P, L)), "l1": np.zeros(L)}
    total = burn + draws * thin
    for it in range(total):
        # 1) alpha with the season terms integrated out
        r = y - lam[D.league] - beta * z - gamma[D.team]
        rbar = np.bincount(D.pseason, w * r, Q) / w_q
        vq = 1 / w_q + s2
        prec = 1 / ta2 + np.bincount(D.ps_player, 1 / vq, P)
        mean = (m[D.group] / ta2 + np.bincount(D.ps_player, rbar / vq, P)) / prec
        alpha = mean + rng.standard_normal(P) / np.sqrt(prec)
        # 2) sigma² with the season terms still integrated out (random-walk Metropolis on log sigma²),
        #    then eta given alpha: much better mixing than drawing sigma² from the eta draws alone
        dev2 = (rbar - alpha[D.ps_player]) ** 2

        def logp(ls2):
            vv = 1 / w_q + np.exp(ls2)
            return -0.5 * np.sum(np.log(vv) + dev2 / vv) - a0 * ls2 - b0 / np.exp(ls2)

        cur, lp = np.log(s2), None
        for _ in range(3):
            lp = logp(cur) if lp is None else lp
            prop = cur + step * rng.standard_normal()
            lq = logp(prop)
            if np.log(rng.random()) < lq - lp:
                cur, lp = prop, lq
                n_acc += 1
            n_try += 1
        s2 = np.exp(cur)
        if it < burn and it % 50 == 49:                     # tune the step during burn-in only
            step *= np.exp(n_acc / n_try - 0.35)
            n_acc = n_try = 0
        prec_e = 1 / s2 + w_q
        eta = w_q * (rbar - alpha[D.ps_player]) / prec_e + rng.standard_normal(Q) / np.sqrt(prec_e)
        theta = alpha[D.ps_player] + eta
        # 3) team effects
        r = y - theta[D.pseason] - lam[D.league] - beta * z
        prec_t = 1 / tg2 + np.bincount(D.team, w, T)
        gamma = np.bincount(D.team, w * r, T) / prec_t + rng.standard_normal(T) / np.sqrt(prec_t)
        # 4) competition effects and team-strength slope (weighted least squares, weak normal prior)
        r = y - theta[D.pseason] - gamma[D.team]
        k = L                                   # L − 1 free competition effects + the slope
        XtWX = np.zeros((k, k))
        XtWr = np.zeros(k)
        wl = np.bincount(D.league, w, L)
        XtWX[np.arange(L - 1), np.arange(L - 1)] = wl[1:]
        wz = np.bincount(D.league, w * z, L)
        XtWX[:L - 1, -1] = XtWX[-1, :L - 1] = wz[1:]
        XtWX[-1, -1] = np.sum(w * z * z)
        XtWr[:L - 1] = np.bincount(D.league, w * r, L)[1:]
        XtWr[-1] = np.sum(w * z * r)
        XtWX += np.diag([1 / s_lam2] * (L - 1) + [1 / s_beta2])
        C = np.linalg.cholesky(XtWX)
        mu = np.linalg.solve(XtWX, XtWr)
        draw = mu + np.linalg.solve(C.T, rng.standard_normal(k))
        lam = np.concatenate([[0.0], draw[:L - 1]])
        beta = draw[-1]
        # 5) group means of the career level
        prec_m = n_g / ta2 + 1 / big
        m = np.bincount(D.group, alpha, G) / ta2 / prec_m + rng.standard_normal(G) / np.sqrt(prec_m)
        # 6) ridge moves: lambda[c] + x and the career level of everyone who played in c − x.
        #    Only the other-competition rows of the movers change: that is all the information.
        for j in range(1, L):
            hit = obs_has[:, j] & (D.league != j)
            e = y[hit] - theta[D.pseason[hit]] - lam[D.league[hit]] - beta * z[hit] - gamma[D.team[hit]]
            gs = in_group[:, j]
            prec_x = np.sum(w[hit]) + 1 / s_lam2 + gs.sum() / big
            mean_x = (-np.sum(w[hit] * e) - lam[j] / s_lam2 + np.sum(m[gs]) / big) / prec_x
            x = mean_x + rng.standard_normal() / np.sqrt(prec_x)
            lam[j] += x
            alpha[player_has[:, j]] -= x
            m[gs] -= x
            theta = alpha[D.ps_player] + eta
        #    lambda[c] + x and every team effect of c − x: again only the priors change
        for j in range(1, L):
            tj = team_league == j
            prec_x = tj.sum() / tg2 + 1 / s_lam2
            mean_x = (np.sum(gamma[tj]) / tg2 - lam[j] / s_lam2) / prec_x
            x = mean_x + rng.standard_normal() / np.sqrt(prec_x)
            lam[j] += x
            gamma[tj] -= x
        #    beta + x and every team effect − x · z: the likelihood does not change, only the priors
        zt = np.bincount(D.team, z, T) / np.maximum(np.bincount(D.team, minlength=T), 1)
        prec_x = np.sum(zt ** 2) / tg2 + 1 / s_beta2
        mean_x = (np.sum(gamma * zt) / tg2 - beta / s_beta2) / prec_x
        x = mean_x + rng.standard_normal() / np.sqrt(prec_x)
        beta += x
        gamma -= x * zt
        # 7) variances
        ta2 = _ig(rng, a0 + P / 2, b0 + 0.5 * np.sum((alpha - m[D.group]) ** 2))
        tg2 = _ig(rng, a0 + T / 2, b0 + 0.5 * np.sum(gamma ** 2))

        if it >= burn and (it - burn) % thin == 0:
            keep.append((lam.copy(), beta, np.sqrt(ta2), np.sqrt(tg2), np.sqrt(s2)))
            acc["a1"] += alpha
            acc["a2"] += alpha ** 2
            acc["t1"] += theta
            acc["t2"] += theta ** 2
            acc["tl"] += theta[:, None] * lam[None, :]
            acc["al"] += alpha[:, None] * lam[None, :]
            acc["l1"] += lam
    n = len(keep)
    lam_d = np.array([k[0] for k in keep])
    a_m, t_m, l_m = acc["a1"] / n, acc["t1"] / n, acc["l1"] / n
    return Fit(D, lam_d, np.array([k[1] for k in keep]), np.array([k[2] for k in keep]),
               np.array([k[3] for k in keep]), np.array([k[4] for k in keep]),
               a_m, acc["a2"] / n - a_m ** 2, t_m, acc["t2"] / n - t_m ** 2,
               acc["tl"] / n - t_m[:, None] * l_m[None, :], acc["al"] / n - a_m[:, None] * l_m[None, :])


# ─────────────────────────────────────────────────────────────── pymc
def pymc_fit(D: Design, draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 0, **kw) -> Fit:
    """The same model with PyMC (NUTS). Optional: pip install pymc (and nutpie for speed)."""
    import pymc as pm

    L, P, Q, T, G = len(D.leagues), len(D.players), len(D.pseasons), len(D.teams), len(D.groups)
    sd_y = float(np.std(D.y)) or 1.0
    with pm.Model() as model:
        lam_free = pm.Normal("lam_free", 0, 2 * sd_y, shape=L - 1)
        lam = pm.math.concatenate([np.zeros(1), lam_free])
        beta = pm.Normal("beta", 0, 2 * sd_y)
        m = pm.Normal("m", 0, 10 * sd_y, shape=G)
        tau_a = pm.HalfNormal("tau_alpha", sd_y)
        tau_g = pm.HalfNormal("tau_gamma", sd_y)
        sigma = pm.HalfNormal("sigma", sd_y)
        alpha = pm.Deterministic("alpha", m[D.group] + tau_a * pm.Normal("alpha_z", 0, 1, shape=P))
        eta = sigma * pm.Normal("eta_z", 0, 1, shape=Q)
        theta = pm.Deterministic("theta", alpha[D.ps_player] + eta)
        gamma = tau_g * pm.Normal("gamma_z", 0, 1, shape=T)
        mu = theta[D.pseason] + lam[D.league] + beta * D.z + gamma[D.team]
        pm.Normal("y", mu, np.sqrt(D.v), observed=D.y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, target_accept=0.9,
                          progressbar=True, **kw)
    post = trace.posterior
    lf = post["lam_free"].values.reshape(-1, L - 1)
    lam_d = np.column_stack([np.zeros(len(lf)), lf])
    th = post["theta"].values.reshape(len(lf), Q)
    al = post["alpha"].values.reshape(len(lf), P)

    def cov(x):
        return ((x - x.mean(0)).T @ (lam_d - lam_d.mean(0))) / len(lf)

    flat = lambda k: post[k].values.reshape(-1)          # noqa: E731
    return Fit(D, lam_d, flat("beta"), flat("tau_alpha"), flat("tau_gamma"), flat("sigma"),
               al.mean(0), al.var(0), th.mean(0), th.var(0), cov(th), cov(al), backend="pymc",
               extra={"trace": trace})


# ─────────────────────────────────────────────────────────────── whole fit
def fit_all(rel: pd.DataFrame, metrics=None, reference: str = "euroleague", backend: str = "gibbs",
            draws: int = 3000, burn: int = 1000, seed: int = 0, verbose: bool = True) -> dict:
    metrics = metrics or [m for m in LABELS if m in set(rel["metric"])]
    out = {}
    for m in metrics:
        D = design(rel, m, reference)
        if len(D.leagues) < 2:
            continue
        if verbose and D.dropped and m == metrics[0]:
            print(f"  not linked to {D.reference} by enough shared players, left out: {', '.join(D.dropped)}")
        f = gibbs(D, draws=draws, burn=burn, seed=seed) if backend == "gibbs" else \
            pymc_fit(D, draws=draws, tune=burn, seed=seed)
        out[m] = f
        if verbose:
            h = f.hyper()
            print(f"  {LABELS[m]:20} {D.n:>7} rows  beta {h['beta']['mean']:+.3f}  "
                  f"tau_alpha {h['tau_alpha']['mean']:.3f}  sigma {h['sigma']['mean']:.3f}  "
                  f"tau_gamma {h['tau_gamma']['mean']:.3f}")
    return out


def tables(fits: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(competition effects, pairwise factors, hyperparameters) for every metric."""
    scale = pd.concat([f.leagues() for f in fits.values()], ignore_index=True)
    pairs = []
    for f in fits.values():
        for a in f.design.leagues:
            for b in f.design.leagues:
                if a != b:
                    pairs.append(f.factor(a, b))
    hyp = pd.json_normalize([f.hyper() for f in fits.values()])
    return scale, pd.DataFrame(pairs), hyp


# ─────────────────────────────────────────────────────────────── translate
def translate(fits: dict, rel: pd.DataFrame, person: str, season: int, a: str, b: str,
              team_z: float = 0.0, next_season: bool = False, target_season: int | None = None) -> pd.DataFrame:
    """Expected stats of one player in competition b (80% intervals), from the joint model.

    Same season: his level that season (theta, estimated from every competition he played) plus
    the effect of b and of the strength of his new team (team_z, in SD: 0 = average team, 1 = a
    clearly good one). Next season: his career level plus a new season-to-season change. The
    interval adds the posterior uncertainty and an unknown new team context (tau_gamma)."""
    target_season = target_season or (season + 1 if next_season else season)
    seasons_b = sorted(rel.loc[rel["competition"] == b, "season"].unique())
    if target_season not in seasons_b and seasons_b:
        before = [x for x in seasons_b if x <= target_season]
        target_season = before[-1] if before else seasons_b[0]
    ref_b = rel[(rel["competition"] == b) & (rel["season"] == target_season)].groupby("metric")["league_value"].first()
    me = rel[(rel["person"] == person) & (rel["season"] == season) & (rel["competition"] == a)].set_index("metric")
    rows = []
    for m, f in fits.items():
        D = f.design
        if m not in ref_b or b not in D.leagues or m not in me.index:
            continue
        jb = D.leagues.index(b)
        lam_b = f.lam[:, jb]
        if next_season:
            hit = np.flatnonzero(D.players == person)
            if not len(hit):
                continue
            i = hit[0]
            mean_x, var_x, cov_x = f.alpha_mean[i], f.alpha_var[i], f.alpha_lam_cov[i, jb]
            extra = np.mean(f.sigma ** 2)
        else:
            hit = np.flatnonzero((D.pseasons["person"] == person).to_numpy() & (D.pseasons["season"] == season).to_numpy())
            if not len(hit):
                continue
            i = hit[0]
            mean_x, var_x, cov_x = f.theta_mean[i], f.theta_var[i], f.theta_lam_cov[i, jb]
            extra = 0.0
        mu = mean_x + lam_b.mean() + f.beta.mean() * team_z
        var = var_x + lam_b.var() + 2 * cov_x + f.beta.var() * team_z ** 2 + np.mean(f.tau_gamma ** 2) + extra
        sd = np.sqrt(max(var, 1e-12))
        conv = _from_rel(m, ref_b[m])
        r = me.loc[m]
        rows.append({"metric": m, "label": LABELS[m], "in_a": r["value"], "league_a": r["league_value"],
                     "expected_b": conv(mu), "lo80": conv(mu - 1.2816 * sd), "hi80": conv(mu + 1.2816 * sd),
                     "league_b": ref_b[m]})
    return pd.DataFrame(rows, columns=["metric", "label", "in_a", "league_a", "expected_b", "lo80", "hi80",
                                       "league_b"])
