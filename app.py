"""League Translator: what would this player's numbers look like in another competition?

    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path
import pickle

from translator import bayes, data, identity, model

BAYES = Path("results") / "_bayes.pkl"

st.set_page_config(page_title="League Translator", page_icon="🏀", layout="wide")
T = {"es": {"title": "Traductor de ligas", "from": "Desde", "to": "Hacia", "season": "Temporada", "player": "Jugador",
            "min_poss": "Mínimo de posesiones", "exp": "Esperado", "obs": "Observado", "league": "Media de la liga",
            "factors": "Factores de traducción", "how": "Cómo se calcula",
            "none": "No hay jugadores con esos filtros.",
            "note": "Intervalo del 80%: incluye la incertidumbre del factor y la variación entre jugadores. "
                    "La columna 'encogimiento' indica cuánto se acerca el valor del jugador a la media de su liga "
                    "por tener pocas posesiones.",
            "method": "Cada estadística se expresa respecto a la media de su liga y temporada. El efecto de liga se "
                      "estima con los jugadores que jugaron en ambas competiciones, en la misma temporada o en "
                      "temporadas consecutivas (descontando el cambio típico de un año a otro), con un modelo de "
                      "efectos aleatorios. Datos: ACB, Euroleague Basketball."},
     "en": {"title": "League Translator", "from": "From", "to": "To", "season": "Season", "player": "Player",
            "min_poss": "Minimum possessions", "exp": "Expected", "obs": "Observed", "league": "League average",
            "factors": "Translation factors", "how": "How it works",
            "none": "No players with these filters.",
            "note": "80% interval: includes the uncertainty of the factor and the spread between players. "
                    "'Shrinkage' is how much the player's value is pulled toward his league's average because "
                    "of few possessions.",
            "method": "Every statistic is expressed relative to its league-season. The league effect is estimated "
                      "on the players who played in both competitions, in the same season or in consecutive "
                      "seasons (removing the usual year-to-year change), with a random-effects model. "
                      "Data: ACB, Euroleague Basketball."}}


@st.cache_data(show_spinner="Reading the warehouse…")
def load():
    pg = data.player_games()
    ps = data.player_seasons(pg)
    ps, _ = identity.link(ps)
    return ps, model.relative(ps)


@st.cache_data(show_spinner=False)
def factors(a: str, b: str):
    _, rel = load()
    return model.factors(rel, a, b)


@st.cache_resource(show_spinner=False)
def bayes_fits():
    return pickle.loads(BAYES.read_bytes())["fits"] if BAYES.exists() else None


@st.cache_data(show_spinner=False)
def move(a: str, b: str):
    _, rel = load()
    return model.move_factor(rel, a, b)


lang = st.sidebar.radio("🌐", ["es", "en"], horizontal=True)
t = T[lang]
st.title(f"🏀 {t['title']}")
ps, rel = load()
comps = sorted(rel["competition"].unique())
name = lambda c: data.NAMES.get(c, c)  # noqa: E731
es = lang == "es"
mode = st.sidebar.radio("Proyección" if es else "Projection", ["same", "next"],
                        format_func=lambda m: {"same": "Misma temporada (nivel de liga)" if es else "Same season (league level)",
                                               "next": "Próxima temporada (fichaje)" if es else "Next season (signing)"}[m],
                        help=("Misma temporada: cuánto rendiría en la otra liga ese mismo año. Próxima temporada: qué "
                              "esperar si ficha por la otra liga (incluye el cambio de rol y de un año a otro, medido "
                              "en los traspasos reales). Si eliges la misma liga, proyecta su próxima temporada allí."
                              if es else
                              "Same season: how he would do in the other league that year. Next season: what to expect "
                              "if he signs there (includes the change of role and of one year, measured on real moves). "
                              "Choosing the same league projects his next season there."))
a = st.sidebar.selectbox(t["from"], comps, index=comps.index("acb") if "acb" in comps else 0, format_func=name)
b = st.sidebar.selectbox(t["to"], [c for c in comps if c != a] + ([a] if mode == "next" else []), format_func=name)
seasons = sorted(rel.loc[rel["competition"] == a, "season"].unique(), reverse=True)
season = st.sidebar.selectbox(t["season"], seasons, format_func=lambda y: f"{y}-{str(y + 1)[-2:]}")
here = ps[(ps["competition"] == a) & (ps["season"] == season)]
default_poss = 300 if (here["poss"] >= 300).sum() >= 15 else 150      # early in the season, a lower bar
min_poss = st.sidebar.slider(t["min_poss"], 150, 1500, value=default_poss, step=50)
pool = ps[(ps["competition"] == a) & (ps["season"] == season) & (ps["poss"] >= min_poss)] \
    .sort_values("pir", ascending=False)
if pool.empty:
    st.info(t["none"])
    st.stop()
who = st.sidebar.selectbox(t["player"], pool["person"], format_func=lambda p: pool.set_index("person").loc[p, "person_name"]
                           + f" ({pool.set_index('person').loc[p, 'team']})")

fits = bayes_fits()
engine = st.sidebar.radio("Modelo" if es else "Model", ["v1", "bayes"] if fits else ["v1"],
                          format_func=lambda m: {"v1": "v1 · pares de ligas" if es else "v1 · league pairs",
                                                 "bayes": "v2 · bayesiano conjunto" if es else "v2 · joint Bayesian"}[m],
                          help=("v2: un único modelo jerárquico para todas las ligas, que separa el efecto de la liga "
                                "del de la fuerza del equipo. Se ajusta con `python -m translator.cli fit --model bayes`."
                                if es else "v2: one hierarchical model for every league, separating the league effect "
                                "from team strength. Fitted with `python -m translator.cli fit --model bayes`."))
team_z = 0.0
if engine == "bayes":
    team_z = st.sidebar.slider("Fuerza del equipo de destino (DE)" if es else "Strength of the new team (SD)",
                               -2.0, 2.0, 0.0, 0.5,
                               help="0 = equipo medio de la liga de destino; +1 = claramente bueno; −1 = de la zona baja."
                               if es else "0 = average team of the target league; +1 = clearly good; −1 = bottom half.")
fac = factors(a, b) if mode == "same" else move(a, b)
if engine == "bayes":
    out = bayes.translate(fits, rel, who, season, a, b, team_z=team_z, next_season=mode == "next")
    out["shrinkage"] = np.nan
else:
    out = model.translate(rel, fac, who, season, a, b, next_season=mode == "next")
pname = pool.set_index("person").loc[who, "person_name"]
target = season if mode == "same" else season + 1
st.subheader(f"{pname} · {name(a)} {season}-{str(season + 1)[-2:]} → {name(b)} {target}-{str(target + 1)[-2:]}")
if mode == "next" and engine == "v1":
    nm = int(fac["n_move"].median())
    src = fac["move_source"].iloc[0]
    st.caption((f"Basado en {nm} {'jugadores que se quedaron' if a == b else 'traspasos reales'} "
                f"{name(a)} → {name(b)}." if src != "league effect + year-to-year" else
                "Pocos traspasos directos entre estas ligas: se usa el efecto de liga más el cambio típico de un año "
                "a otro.") if es else
               (f"Based on {nm} {'players who stayed' if a == b else 'real moves'} {name(a)} → {name(b)}."
                if src != "league effect + year-to-year" else
                "Few direct moves between these leagues: league effect plus the usual year-to-year change."))

if out.empty:
    st.info("No hay factores de traducción para este par de competiciones (faltan jugadores que hayan jugado en "
            "ambas). Prueba con otro par o con otra temporada." if lang == "es" else
            "No translation factors for this pair of competitions (not enough players who played in both). "
            "Try another pair or season.")
    st.stop()
fig = go.Figure()
show = out[out["metric"].isin(["pts", "ast", "tov", "orb", "drb", "fta", "fg3a", "stl"])]
y = show["label"]
fig.add_trace(go.Scatter(x=show["league_b"], y=y, mode="markers", name=t["league"],
                         marker=dict(symbol="line-ns", size=16, line=dict(width=2, color="#9ca3af"))))
fig.add_trace(go.Scatter(x=show["in_a"], y=y, mode="markers", name=f"{t['obs']} ({name(a)})",
                         marker=dict(size=9, color="#f59e0b")))
fig.add_trace(go.Scatter(x=show["expected_b"], y=y, mode="markers", name=f"{t['exp']} ({name(b)})",
                         marker=dict(size=11, color="#2563eb", symbol="diamond"),
                         error_x=dict(type="data", symmetric=False, array=show["hi80"] - show["expected_b"],
                                      arrayminus=show["expected_b"] - show["lo80"], color="#2563eb")))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15),
                  xaxis_title="/100 poss.")
st.plotly_chart(fig, use_container_width=True)
tab = out[["label", "in_a", "league_a", "expected_b", "lo80", "hi80", "league_b", "shrinkage"]].copy()
tab.columns = ["Métrica" if lang == "es" else "Metric", f"{t['obs']} {name(a)}", f"{t['league']} {name(a)}",
               f"{t['exp']} {name(b)}", "P10", "P90", f"{t['league']} {name(b)}",
               "encogimiento" if lang == "es" else "shrinkage"]
st.dataframe(tab.round(2), hide_index=True, use_container_width=True)
st.caption(t["note"])
with st.expander(t["factors"]):
    cols = (["label", "n_same", "delta_same", "n_consecutive", "delta_consecutive", "delta", "se", "tau", "factor",
             "designs_agree_z"] if mode == "same" else ["label", "n_move", "delta", "se", "tau", "factor", "move_source"])
    st.dataframe(fac[cols].round(3), hide_index=True, use_container_width=True)
with st.expander(t["how"]):
    st.write(t["method"])
