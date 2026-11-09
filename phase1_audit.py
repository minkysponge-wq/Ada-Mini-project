#!/usr/bin/env python3
"""
Phase 1 - Lineage coverage audit + heritability sensitivity.

Answers the TA's question: "does the Hugging Face API have sufficient structured
lineage data to map this pedigree?"  Reads Part 1 outputs only (no internet).

Outputs (in <out>/phase1/):
  coverage.json          every coverage number used in the report
  coverage_funnel.csv    leaderboard models -> Hub page -> lineage metadata -> relatives in panel
  h2_sensitivity.csv     h2 of inflation on progressively better-documented subsets
  figs/P1_coverage_funnel.png, figs/P1_lineage_source.png, figs/P1_h2_sensitivity.png

Usage:  python phase1_audit.py --data out_phase1/data --out out_phase1
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part1_pedigree import parents_of
from part2_analysis import kinship, AnimalModel, design, style, C


def load(ddir):
    v1 = pd.read_csv(f"{ddir}/v1_scores.csv")
    v2 = pd.read_csv(f"{ddir}/v2_scores.csv")
    nodes = pd.read_csv(f"{ddir}/nodes.csv")
    edges = pd.read_csv(f"{ddir}/edges.csv")
    panel = pd.read_csv(f"{ddir}/panel.csv")
    meta = {}
    for line in open(f"{ddir}/hub_meta.jsonl", encoding="utf-8"):
        r = json.loads(line)
        meta[r["model_id"]] = r          # later lines win (resumed runs)
    alias_p = f"{ddir}/id_aliases.csv"
    alias = pd.read_csv(alias_p) if os.path.exists(alias_p) else None
    return v1, v2, nodes, edges, panel, meta, alias


def lineage_source(mid, meta, edges_in):
    """Where a model's parent links come from."""
    r = meta.get(mid)
    if r is None:
        hub = "not fetched"
    elif r.get("ok") is False:
        hub = "hub page missing/gated"
    else:
        hub = "hub page ok"
    has_tags = bool(r and r.get("rel_tags"))
    has_card = bool(r and r.get("base_models"))
    n_edges = edges_in.get(mid, 0)
    if has_tags:
        src = "relation tags"
    elif has_card:
        src = "card base_model only"
    elif n_edges:
        src = "leaderboard column only"
    else:
        src = "no lineage (founder or undocumented)"
    return hub, src


def audit(v1, v2, nodes, edges, panel, meta, alias):
    lb = sorted(set(nodes.model_id[nodes.v1_avg.notna() | nodes.v2_avg.notna()]))
    e_nq = edges[edges.relation != "quantized"]
    edges_in = e_nq.groupby("child").size().to_dict()
    rows = []
    for m in lb:
        hub, src = lineage_source(m, meta, edges_in)
        rows.append({"model_id": m, "hub": hub, "source": src, "in_panel": m in set(panel.model_id)})
    lbdf = pd.DataFrame(rows)

    # parents referenced anywhere: do they resolve to a real Hub page?
    parents = set(e_nq.parent)
    p_ok = sum(1 for p in parents if meta.get(p, {}).get("ok"))
    p_missing = sum(1 for p in parents if meta.get(p, {}).get("ok") is False)
    p_unfetched = len(parents) - p_ok - p_missing

    # panel structure: how many panel models have at least one relative in the panel?
    ids = panel.model_id.tolist()
    A, _ = kinship(ids, edges, "clonal")
    off = A.copy(); np.fill_diagonal(off, 0)
    has_rel = (off > 1e-9).any(axis=1)
    n_pairs = int((np.triu(off, 1) > 1e-9).sum())
    fam = panel.groupby("founder").size()
    in_multi = panel.founder.map(fam).ge(2)

    pan = lbdf[lbdf.in_panel]
    cov = {
        "leaderboard_models": len(lb),
        "v1_models": int(v1.model_id.nunique()), "v2_models": int(v2.model_id.nunique()),
        "panel_models": len(panel),
        "renames_confirmed": int(alias.accepted.sum()) if alias is not None else 0,
        "renames_candidates": int(len(alias)) if alias is not None else 0,
        "hub_status_leaderboard": lbdf.hub.value_counts().to_dict(),
        "lineage_source_leaderboard": lbdf.source.value_counts().to_dict(),
        "lineage_source_panel": pan.source.value_counts().to_dict(),
        "panel_with_any_parent": int((pan.source != "no lineage (founder or undocumented)").sum()),
        "distinct_parents_referenced": len(parents),
        "parents_hub_ok": p_ok, "parents_hub_missing_or_gated": p_missing,
        "parents_not_fetched_depth_limit": p_unfetched,
        "relation_counts": edges.relation.value_counts().to_dict(),
        "panel_families": int(panel.founder.nunique()),
        "panel_families_with_2plus": int((fam >= 2).sum()),
        "panel_models_in_2plus_families": int(in_multi.sum()),
        "panel_models_with_a_relative_in_panel": int(has_rel.sum()),
        "related_pairs_in_panel": n_pairs,
    }

    funnel = pd.DataFrame([
        ("Scored on the leaderboard (v1 or v2)", len(lb)),
        ("Hub page still exists", int((lbdf.hub == "hub page ok").sum())),
        ("Has any parent link", int((lbdf.source != "no lineage (founder or undocumented)").sum())),
        ("Scored on BOTH v1 and v2 (panel)", len(panel)),
        ("Panel model with a parent link", cov["panel_with_any_parent"]),
        ("Panel model with a relative in the panel", cov["panel_models_with_a_relative_in_panel"]),
    ], columns=["stage", "models"])
    return cov, funnel, lbdf, has_rel, in_multi


def h2_sensitivity(panel, edges, has_rel, in_multi):
    """h2 of inflation (and capability) on progressively better-documented subsets."""
    subsets = {
        "All panel models": np.ones(len(panel), bool),
        "Families with 2+ panel members": in_multi.values,
        "Models with a relative in the panel": has_rel,
    }
    out = []
    for name, mask in subsets.items():
        sub = panel[mask].reset_index(drop=True)
        if len(sub) < 30:
            continue
        X = design(sub).values
        for kin in ["clonal", "diploid"]:
            A, _ = kinship(sub.model_id.tolist(), edges, kin)
            base = AnimalModel(A)
            for trait in ["inflation", "v2_pct"]:
                y = sub[trait].fillna(sub[trait].median()).values.astype(float)
                y = (y - y.mean()) / y.std()
                m = AnimalModel.__new__(AnimalModel); m.s, m.U = base.s, base.U
                m.fit(y, X)
                out.append(dict(subset=name, n=len(sub), kinship=kin, trait=trait,
                                h2=round(m.h2, 4), ci_lo=round(m.ci[0], 3),
                                ci_hi=round(m.ci[1], 3), p_lrt=float(m.p_lrt)))
                print(f"  {name:38s} n={len(sub):4d} {kin:7s} {trait:9s} "
                      f"h2={m.h2:.2f} [{m.ci[0]:.2f},{m.ci[1]:.2f}] p={m.p_lrt:.2g}")
    return pd.DataFrame(out)


def plots(funnel, lbdf, sens, figdir):
    style()
    # funnel
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = np.arange(len(funnel))[::-1]
    top = funnel.models.iloc[0]
    cols = [C["ink2"]] * 3 + [C["blue"]] * 3
    ax.barh(y, funnel.models, color=cols, height=.6)
    for yi, (s, n) in zip(y, funnel.values):
        ax.text(n + top * .01, yi, f"{n:,}  ({n / top:.0%})", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(funnel.stage, fontsize=9)
    ax.set_xlim(0, top * 1.25); ax.grid(axis="y", visible=False)
    ax.set_xlabel("models")
    ax.set_title("Lineage coverage: from leaderboard to usable pedigree")
    fig.savefig(f"{figdir}/P1_coverage_funnel.png"); plt.close(fig)

    # lineage source, leaderboard vs panel
    order = ["relation tags", "card base_model only", "leaderboard column only",
             "no lineage (founder or undocumented)"]
    colr = [C["blue"], C["aqua"], C["yellow"], C["grid"]]
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    for row, (lab, d) in enumerate([("All leaderboard models", lbdf),
                                    ("Panel (v1 and v2)", lbdf[lbdf.in_panel])]):
        share = d.source.value_counts(normalize=True).reindex(order).fillna(0)
        left = 0
        for k, c in zip(order, colr):
            ax.barh(row, share[k], left=left, color=c, height=.55, label=k if row == 0 else None)
            if share[k] > .06:
                ax.text(left + share[k] / 2, row, f"{share[k]:.0%}", ha="center", va="center", fontsize=9)
            left += share[k]
    ax.set_yticks([0, 1]); ax.set_yticklabels(["All leaderboard models", "Panel (v1 and v2)"])
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.grid(visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.25))
    ax.set_title("Where each model's parent link comes from")
    fig.savefig(f"{figdir}/P1_lineage_source.png"); plt.close(fig)

    # h2 sensitivity
    s = sens[sens.trait == "inflation"]
    subsets = list(dict.fromkeys(s.subset))
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for j, (kin, col) in enumerate([("clonal", C["blue"]), ("diploid", C["ink2"])]):
        d = s[s.kinship == kin].set_index("subset").reindex(subsets)
        x = np.arange(len(subsets)) + (j - .5) * .22
        ax.errorbar(x, d.h2, yerr=[d.h2 - d.ci_lo, d.ci_hi - d.h2], fmt="o", color=col,
                    capsize=3, lw=1.5, label=f"{kin} kinship")
    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels([f"{t}\n(n={int(s[s.subset == t].n.iloc[0])})" for t in subsets], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("h² of inflation (95% CI)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Is h² driven by the well-documented families?")
    fig.savefig(f"{figdir}/P1_h2_sensitivity.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out_phase1/data")
    ap.add_argument("--out", default="out_phase1")
    a = ap.parse_args()
    odir = os.path.join(a.out, "phase1"); figdir = os.path.join(odir, "figs")
    os.makedirs(figdir, exist_ok=True)

    v1, v2, nodes, edges, panel, meta, alias = load(a.data)
    panel = panel.dropna(subset=["inflation"]).drop_duplicates("model_id").reset_index(drop=True)
    for c in ["created_at", "n_params", "founder", "generation"]:
        if c not in panel and c in nodes:
            panel = panel.merge(nodes[["model_id", c]], on="model_id", how="left")

    cov, funnel, lbdf, has_rel, in_multi = audit(v1, v2, nodes, edges, panel, meta, alias)
    print(json.dumps(cov, indent=1))
    print("[h2 sensitivity]")
    sens = h2_sensitivity(panel, edges, has_rel, in_multi)

    json.dump(cov, open(f"{odir}/coverage.json", "w"), indent=2)
    funnel.to_csv(f"{odir}/coverage_funnel.csv", index=False)
    lbdf.to_csv(f"{odir}/lineage_by_model.csv", index=False)
    sens.to_csv(f"{odir}/h2_sensitivity.csv", index=False)
    plots(funnel, lbdf, sens, figdir)
    print(f"[done] phase 1 audit in {odir}/")


if __name__ == "__main__":
    main()
