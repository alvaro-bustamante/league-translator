"""League translator, command line.

    python -m translator.cli fit                                  # every pair of competitions in the warehouse
    python -m translator.cli fit --competitions acb euroleague eurocup --from-season 2016
    python -m translator.cli translate --player "Shermadini" --from acb --season 2025 --to euroleague
    python -m translator.cli translate --player "Shermadini" --from acb --season 2024 --to euroleague --next-season
    python -m translator.cli pairs --from acb --to euroleague      # the players behind a factor

Output in results/: factors.csv, common_scale.csv, links.csv, league_trends.csv and
factors_<from>_<to>.png (forest plots). The warehouse folder is found next to this project
(../hoops-warehouse/data/warehouse) or in HOOPS_WAREHOUSE.
"""
from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, identity, model

RESULTS = Path("results")
CACHE = RESULTS / "_relative.pkl"


def prepare(competitions: list[str] | None, first: int | None, last: int | None, refresh: bool = False):
    key = (tuple(competitions or []), first, last)
    if CACHE.exists() and not refresh:
        k, ps, rel, links = pickle.loads(CACHE.read_bytes())
        if k == key:
            return ps, rel, links
    av = data.available()
    seasons = [s for s in av["season"].unique() if (first is None or s >= first) and (last is None or s <= last)]
    pg = data.player_games(competitions, seasons)
    ps = data.player_seasons(pg)
    ps, links = identity.link(ps)
    rel = model.relative(ps)
    RESULTS.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps((key, ps, rel, links)))
    return ps, rel, links


def fit(args) -> None:
    ps, rel, links = prepare(args.competitions, args.from_season, args.to_season, refresh=True)
    comps = sorted(rel["competition"].unique())
    frames = []
    for a, b in itertools.permutations(comps, 2):
        f = model.factors(rel, a, b)
        if (f["n_same"] + f["n_consecutive"]).max() >= 8:
            frames.append(f)
    F = pd.concat(frames, ignore_index=True)
    F.to_csv(RESULTS / "factors.csv", index=False)
    model.common_scale(F, reference=args.reference).to_csv(RESULTS / "common_scale.csv", index=False)
    links.to_csv(RESULTS / "links.csv", index=False)
    model.league_trends(rel).to_csv(RESULTS / "league_trends.csv", index=False)
    for (a, b), f in F.groupby(["from", "to"]):
        forest(f, RESULTS / f"factors_{a}_{b}.png")
    multi = ps.groupby("person")["family"].nunique()
    print(f"{len(ps)} player-seasons, {ps['person'].nunique()} people ({(multi > 1).sum()} in more than one "
          f"competition family; {len(links)} links, see results/links.csv)")
    show = F[F["metric"].isin(["pts", "ts_pct", "ast", "tov", "pir", "net"])]
    print(show[["from", "to", "label", "n_same", "n_consecutive", "delta", "se", "factor", "designs_agree_z"]]
          .round(3).to_string(index=False))
    print(f"\nSaved to {RESULTS.resolve()}")


def forest(f: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = f.dropna(subset=["delta"]).copy()
    f = f[f["kind"] != "level"].sort_values("delta")
    if f.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(f) + 1.4), dpi=160)
    y = np.arange(len(f))
    for off, col, se, lab, c in ((-0.18, "delta_same", "se_same", "same season", "#2563eb"),
                                (0.18, "delta_consecutive", "se_consecutive", "consecutive seasons", "#f59e0b")):
        ok = f[col].notna()
        ax.errorbar(f.loc[ok, col], y[ok.to_numpy()] + off, xerr=1.96 * f.loc[ok, se], fmt="o", ms=3.5, lw=1,
                    color=c, alpha=0.75, label=lab)
    ax.errorbar(f["delta"], y, xerr=1.96 * f["se"], fmt="D", ms=5, lw=2, color="#111827", label="combined")
    ax.axvline(0, color="#9ca3af", lw=1, ls=":")
    ax.set_yticks(y, f["label"])
    a, b = f["from"].iloc[0], f["to"].iloc[0]
    ax.set_xlabel("change relative to league (log ratio for rates, logit for %)")
    ax.set_title(f"{data.NAMES.get(a, a)} → {data.NAMES.get(b, b)}", loc="left", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def translate(args) -> None:
    ps, rel, _ = prepare(args.competitions, args.from_season, args.to_season)
    me = ps[(ps["competition"] == args.source) & (ps["season"] == args.season)
            & ps["person_name"].str.contains(args.player, case=False, na=False)]
    if me.empty:
        raise SystemExit(f"No player matching '{args.player}' in {args.source} {args.season}.")
    if me["person"].nunique() > 1:
        print("Several players match:", ", ".join(sorted(me["person_name"].unique())))
    person = me.sort_values("poss", ascending=False)["person"].iloc[0]
    if args.next_season:
        f = model.move_factor(rel, args.source, args.target)
    else:
        f = model.factors(rel, args.source, args.target)
    out = model.translate(rel, f, person, args.season, args.source, args.target, args.target_season,
                          next_season=args.next_season)
    name = me.loc[me["person"] == person, "person_name"].iloc[0]
    print(f"\n{name}: {data.NAMES.get(args.source, args.source)} {args.season} → "
          f"{data.NAMES.get(args.target, args.target)} (80% interval)\n")
    print(out[["label", "in_a", "league_a", "expected_b", "lo80", "hi80", "league_b", "shrinkage"]].round(2)
          .to_string(index=False))


def pairs(args) -> None:
    _, rel, _ = prepare(args.competitions, args.from_season, args.to_season)
    s = model.pairs_same_season(rel, args.source, args.target)
    c = model.pairs_consecutive(rel, args.source, args.target)
    both = pd.concat([s, c])
    both = both[both["metric"] == args.metric]
    cols = ["design", "person_name_a", "season_a", "team_a", "value_a", "season_b", "team_b", "value_b", "d"]
    print(both[cols].sort_values(["design", "season_a"]).round(2).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fit", "translate", "pairs"])
    ap.add_argument("--competitions", nargs="+")
    ap.add_argument("--from-season", type=int)
    ap.add_argument("--to-season", type=int)
    ap.add_argument("--reference", default="euroleague")
    ap.add_argument("--player")
    ap.add_argument("--from", dest="source")
    ap.add_argument("--to", dest="target")
    ap.add_argument("--season", type=int)
    ap.add_argument("--target-season", type=int)
    ap.add_argument("--metric", default="pts")
    ap.add_argument("--next-season", action="store_true",
                    help="translate: project his next season in the target league (a signing), from real moves.")
    args = ap.parse_args()
    {"fit": fit, "translate": translate, "pairs": pairs}[args.command](args)


if __name__ == "__main__":
    main()
