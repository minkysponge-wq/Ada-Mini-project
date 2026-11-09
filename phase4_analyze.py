#!/usr/bin/env python3
"""
Phase 4c - Does the measured memorisation drop validate the inflation proxy?

Per model (paired by seed question):
  acc_orig            accuracy on the original GSM8K test questions
  drop_symbolic       acc(original) - acc(GSM-Symbolic re-instantiations of the same question)
  drop_numsub         acc(original) - acc(GSM-Plus numerical substitution)
  drop_paraphrase     acc(original) - acc(GSM-Plus problem understanding)
  drop                primary: mean of drop_symbolic and drop_numsub (same problem, new numbers)
  drop_rel            drop / acc_orig
95% CIs by bootstrap over seed questions.

Across models:
  0. harness check: Spearman(our acc_orig, leaderboard v1 GSM8K) - must be clearly positive
  1. Spearman(drop, proxy) for infl_math, inflation, resid_inflation, pedigree_pred
  2. the same, partialling out acc_orig (a stronger model has more to lose)
  3. permutation p-values (n is small)

Usage:  python phase4_analyze.py --models out_phase1/phase4/models.csv \
            --preds out_phase1/phase4/preds --out out_phase1/phase4
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part2_analysis import style, C

PROXIES = {"infl_math": "GSM8K-vs-MATH gap", "inflation": "overall v1-v2 gap",
           "resid_inflation": "residual gap (Phase 2)", "pedigree_pred": "predicted from relatives only"}


def paired_drop(df, variant, rng, n_boot=1000):
    """acc(original) - acc(variant), paired within seed groups; bootstrap over groups."""
    o = df[df.variant == "original"].groupby("group").correct.mean()
    v = df[df.variant == variant].groupby("group").correct.mean()
    g = o.index.intersection(v.index)
    if len(g) == 0:
        return np.nan, (np.nan, np.nan), 0
    d = (o[g] - v[g]).values
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))), int(len(g))


def per_model(pred_dir, rng):
    rows = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        df = pd.read_json(f, lines=True)
        if df.empty:
            continue
        mid = os.path.basename(f)[:-6].replace("__", "/")
        r = {"model_id": mid, "n_items": len(df),
             "acc_orig": df[df.variant == "original"].correct.mean(),
             "acc_symbolic": df[df.variant == "symbolic"].correct.mean(),
             "acc_numsub": df[df.variant == "numerical substitution"].correct.mean(),
             "acc_paraphrase": df[df.variant == "problem understanding"].correct.mean(),
             "no_answer_rate": df.pred.isna().mean()}
        for key, var, src in [("symbolic", "symbolic", "symbolic"), ("numsub", "numerical substitution", "plus"),
                              ("paraphrase", "problem understanding", "plus")]:
            m, ci, n = paired_drop(df[df.source == src], var, rng)
            r[f"drop_{key}"], r[f"drop_{key}_lo"], r[f"drop_{key}_hi"], r[f"n_{key}"] = m, ci[0], ci[1], n
        r["drop"] = np.nanmean([r["drop_symbolic"], r["drop_numsub"]])
        r["drop_rel"] = r["drop"] / r["acc_orig"] if r["acc_orig"] > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(stats.pearsonr(ex, ey)[0])


def perm_p(x, y, rng, n=5000):
    obs = stats.spearmanr(x, y).correlation
    null = np.array([stats.spearmanr(x, rng.permutation(y)).correlation for _ in range(n)])
    return float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="out_phase1/phase4/models.csv")
    ap.add_argument("--preds", default="out_phase1/phase4/preds")
    ap.add_argument("--out", default="out_phase1/phase4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    figdir = os.path.join(a.out, "figs"); os.makedirs(figdir, exist_ok=True)
    style()

    pm = per_model(a.preds, rng)
    if pm.empty:
        print("[analyze] no predictions found"); return
    meta = pd.read_csv(a.models)
    df = pm.merge(meta, on="model_id", how="left")
    df.to_csv(f"{a.out}/memorisation_by_model.csv", index=False)
    print(df[["model_id", "acc_orig", "acc_symbolic", "acc_numsub", "drop", "infl_math"]].round(3).to_string(index=False))

    summ = {"n_models": int(len(df)), "mean_acc_orig": float(df.acc_orig.mean()),
            "mean_drop": float(df["drop"].mean()), "mean_drop_paraphrase": float(df.drop_paraphrase.mean())}
    ok = df.dropna(subset=["GSM8K", "acc_orig"])
    if len(ok) >= 4:
        summ["harness_check_spearman_vs_leaderboard_gsm8k"] = float(stats.spearmanr(ok.GSM8K, ok.acc_orig).correlation)
    tests = []
    if len(df) >= 5:
        for p, lab in PROXIES.items():
            d = df.dropna(subset=[p, "drop", "acc_orig"])
            if len(d) < 5:
                continue
            for target in ["drop", "drop_rel", "drop_paraphrase"]:
                dd = d.dropna(subset=[target])
                if len(dd) < 5:
                    continue
                rho = stats.spearmanr(dd[p], dd[target]).correlation
                tests.append(dict(proxy=p, target=target, n=len(dd), spearman=float(rho),
                                  p_perm=perm_p(dd[p].values, dd[target].values, rng),
                                  partial_spearman_given_acc=partial_spearman(dd[p], dd[target], dd.acc_orig)))
        tests = pd.DataFrame(tests)
        tests.to_csv(f"{a.out}/proxy_validation.csv", index=False)
        summ["proxy_validation"] = tests.round(4).to_dict("records")
        print(tests.round(3).to_string(index=False))
    else:
        print(f"[analyze] only {len(df)} models - need >= 5 for the proxy correlations")
    json.dump(summ, open(f"{a.out}/summary_phase4.json", "w"), indent=2, default=float)

    # ---------------- figures
    if len(ok) >= 4:
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        ax.scatter(ok.GSM8K, ok.acc_orig * 100, color=C["ink2"], s=28)
        ax.plot([0, 100], [0, 100], color=C["grid"], ls="--")
        ax.set_xlabel("leaderboard v1 GSM8K (%)"); ax.set_ylabel("our accuracy on original questions (%)")
        ax.set_title(f"Harness check: ρ = {summ['harness_check_spearman_vs_leaderboard_gsm8k']:.2f}", fontsize=11)
        fig.savefig(f"{figdir}/P4_harness_check.png"); plt.close(fig)

    d = df.sort_values("drop")
    fig, ax = plt.subplots(figsize=(7.4, max(2.6, .26 * len(d) + 1)))
    y = np.arange(len(d))
    ax.errorbar(d.drop_symbolic * 100, y + .15, xerr=[(d.drop_symbolic - d.drop_symbolic_lo) * 100,
                (d.drop_symbolic_hi - d.drop_symbolic) * 100], fmt="o", ms=4, color=C["blue"],
                capsize=2, label="GSM-Symbolic")
    ax.errorbar(d.drop_numsub * 100, y - .15, xerr=[(d.drop_numsub - d.drop_numsub_lo) * 100,
                (d.drop_numsub_hi - d.drop_numsub) * 100], fmt="s", ms=4, color=C["ink2"],
                capsize=2, label="GSM-Plus numerical substitution")
    ax.axvline(0, color=C["ink"], lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(d.model_id, fontsize=7.5); ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy drop: original - perturbed [pct pts, 95% CI]")
    ax.legend(fontsize=8, loc="lower right"); ax.set_title("Memorisation drop per model")
    fig.savefig(f"{figdir}/P4_drop_by_model.png"); plt.close(fig)

    if len(df) >= 5:
        fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
        for ax, p in zip(axs, ["infl_math", "pedigree_pred"]):
            dd = df.dropna(subset=[p, "drop"])
            ax.scatter(dd[p], dd["drop"] * 100, color=C["blue"] if p == "infl_math" else C["ink2"], s=30)
            row = tests[(tests.proxy == p) & (tests.target == "drop")]
            ttl = f"ρ = {row.spearman.iloc[0]:+.2f} (perm p = {row.p_perm.iloc[0]:.2g})" if len(row) else ""
            ax.set_title(ttl, fontsize=10.5); ax.set_xlabel(PROXIES[p])
            ax.axhline(0, color=C["ink2"], lw=.8)
        axs[0].set_ylabel("memorisation drop [pct pts]")
        fig.suptitle("Do proxy-inflated models lose more on perturbed questions?", x=.02, ha="left",
                     fontweight="bold", fontsize=12)
        fig.tight_layout(); fig.savefig(f"{figdir}/P4_drop_vs_proxy.png"); plt.close(fig)
    print(f"[done] phase 4 analysis in {a.out}/")


if __name__ == "__main__":
    main()
