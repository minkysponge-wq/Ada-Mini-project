#!/usr/bin/env python3
"""
Phase 4a - Pick the models for the GSM-Symbolic / GSM-Plus validation (offline).

A model is eligible if it is in the v1/v2 panel (so it has an inflation proxy), its Hub
page still exists, it fits the GPU tier, and it scored >= 15% on GSM8K in v1 (a model at
0% cannot show a memorisation drop).

For every eligible model it records the four quantities Phase 4c will test against the
measured drop:
  infl_math         GSM8K-vs-MATH rank gap (the benchmark-matched proxy)
  inflation         overall v1-v2 rank gap (Part 2 proxy)
  resid_inflation   capability-adjusted residual gap (Phase 2)
  pedigree_pred     inflation predicted from the model's RELATIVES ONLY (leave-one-out
                    pedigree BLUP; the model's own scores are never used) -> tests the
                    "lineage-only contamination detector"

Tiers:  local  <= 2B  (6 GB laptop GPU, fp16)
        colab  <= 4B  (free Colab T4, 16 GB)
        kaggle <= 8B  (free Kaggle 2x T4)

Usage:  python phase4_select.py --data out_phase1/data --out out_phase1
"""
import argparse, os
import numpy as np
import pandas as pd

from part2_analysis import kinship, AnimalModel, design, gp_predict
from phase2_decouple import residual_gap

TIERS = {"local": 2.0, "colab": 4.0, "kaggle": 8.0}


def loo_pedigree_prediction(panel, edges, targets, kin="clonal"):
    """Leave-one-out pedigree BLUP of inflation for each target model."""
    X = design(panel).values
    y = panel.inflation.values.astype(float)
    A, _ = kinship(panel.model_id.tolist(), edges, kin)
    ys = (y - y.mean()) / y.std()
    h2 = AnimalModel(A).fit(ys, X).h2
    idx = {m: i for i, m in enumerate(panel.model_id)}
    out = {}
    for m in targets:
        te = np.array([idx[m]])
        tr = np.array([i for i in range(len(y)) if i != idx[m]])
        blup, cov = gp_predict(A, y, X, tr, te, h2)
        out[m] = (float(blup[0]), float(cov[0]))
    return out, h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out_phase1/data")
    ap.add_argument("--out", default="out_phase1")
    ap.add_argument("--min-gsm8k", type=float, default=15.0)
    a = ap.parse_args()
    odir = os.path.join(a.out, "phase4"); os.makedirs(odir, exist_ok=True)

    panel = pd.read_csv(f"{a.data}/panel.csv").dropna(subset=["inflation"]) \
              .drop_duplicates("model_id").reset_index(drop=True)
    nodes = pd.read_csv(f"{a.data}/nodes.csv").drop_duplicates("model_id").set_index("model_id")
    edges = pd.read_csv(f"{a.data}/edges.csv")
    for c in ["created_at", "n_params", "founder", "generation"]:
        if c not in panel and c in nodes:
            panel[c] = nodes[c].reindex(panel.model_id).values
    # leaderboard '#Params (B)' is 0 for some models (e.g. Mixtral-8x22B) -> treat 0 as missing
    lb_size = pd.to_numeric(panel.params_b, errors="coerce").where(lambda s: s > 0)
    hub_size = pd.to_numeric(panel.n_params, errors="coerce").where(lambda s: s > 0) / 1e9
    panel["size_b"] = lb_size.fillna(hub_size)          # unknown size -> NaN -> not eligible
    panel["hub_ok"] = nodes.ok.reindex(panel.model_id).values == True
    panel["resid_inflation"] = residual_gap(panel.v1_avg, panel.v2_avg).values

    elig = panel[panel.hub_ok & (panel.size_b <= TIERS["kaggle"]) & (panel.GSM8K >= a.min_gsm8k)].copy()
    pred, h2 = loo_pedigree_prediction(panel, edges, elig.model_id.tolist())
    elig["pedigree_pred"] = elig.model_id.map(lambda m: pred[m][0])
    elig["covariate_pred"] = elig.model_id.map(lambda m: pred[m][1])
    elig["has_relative_in_panel"] = (elig.pedigree_pred - elig.covariate_pred).abs() > 1e-6
    elig["tier"] = np.select([elig.size_b <= TIERS["local"], elig.size_b <= TIERS["colab"]],
                             ["local", "colab"], "kaggle")

    cols = ["model_id", "tier", "size_b", "architecture", "type", "founder", "GSM8K", "MATH Lvl 5",
            "infl_math", "inflation", "resid_inflation", "pedigree_pred", "covariate_pred",
            "has_relative_in_panel"]
    elig = elig[cols].sort_values(["size_b", "model_id"]).reset_index(drop=True)
    elig.to_csv(f"{odir}/models.csv", index=False)

    print(f"[select] h2 used for the pedigree BLUP = {h2:.2f}")
    for t in TIERS:
        k = elig[elig.tier.isin(list(TIERS)[:list(TIERS).index(t) + 1])]
        print(f"[select] up to tier '{t}' ({TIERS[t]}B): {len(k)} models, {k.founder.nunique()} families, "
              f"{int(k.has_relative_in_panel.sum())} with a relative in the panel; "
              f"infl_math {k.infl_math.min():.0f}..{k.infl_math.max():.0f}")
    print(f"[done] {odir}/models.csv")


if __name__ == "__main__":
    main()
