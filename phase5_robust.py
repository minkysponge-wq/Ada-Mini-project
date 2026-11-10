#!/usr/bin/env python3
"""
Phase 5 - Robustness of the main result (h2 of inflation).

Re-estimates h2 (clonal kinship, the conservative model) of the raw and residual inflation proxies
under alternative choices, to show the headline result is not an artefact of one decision:

  - kinship model: clonal vs diploid
  - residual proxy: linear / cubic / quintic fit of v1 rank on v2 rank
  - sample: without merges, without MoE models,
    only models above / below the median size
  - leave-one-family-out: drop each of the 5 largest founder families in turn
    (is a single family driving the result?)

Outputs (in <out>/phase5/): robustness.csv, summary_phase5.json, figs/P5_robustness.png

Usage:  python phase5_robust.py --data out_phase1/data --out out_phase1
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part2_analysis import kinship, AnimalModel, design, style, C
from phase2_decouple import residual_gap, zfill


def h2(panel, edges, y, variant="clonal"):
    A, _ = kinship(panel.model_id.tolist(), edges, variant)
    m = AnimalModel(A).fit(zfill(y), design(panel).values)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out_phase1/data")
    ap.add_argument("--out", default="out_phase1")
    a = ap.parse_args()
    odir = os.path.join(a.out, "phase5"); figdir = os.path.join(odir, "figs")
    os.makedirs(figdir, exist_ok=True)
    style()

    panel = pd.read_csv(f"{a.data}/panel.csv")
    edges = pd.read_csv(f"{a.data}/edges.csv")
    panel = panel.dropna(subset=["inflation"]).drop_duplicates("model_id").reset_index(drop=True)
    panel["resid3"] = residual_gap(panel.v1_avg, panel.v2_avg, 3)
    panel["resid1"] = residual_gap(panel.v1_avg, panel.v2_avg, 1)
    panel["resid5"] = residual_gap(panel.v1_avg, panel.v2_avg, 5)
    size = pd.to_numeric(panel.params_b, errors="coerce").replace(0, np.nan)
    size = size.fillna(pd.to_numeric(panel.n_params, errors="coerce") / 1e9)
    truthy = lambda s: s.astype(str).str.lower().isin(["true", "1", "1.0"])

    specs = [("baseline", "all", np.ones(len(panel), bool), "clonal")]
    specs.append(("kinship: diploid", "kinship", np.ones(len(panel), bool), "diploid"))
    specs += [("without merges", "sample", ~truthy(panel.is_merge).values, "clonal"),
              ("without MoE models", "sample", ~truthy(panel.moe).values, "clonal"),
              ("size <= median", "sample", (size <= size.median()).fillna(False).values, "clonal"),
              ("size > median", "sample", (size > size.median()).fillna(False).values, "clonal")]
    fam = panel.founder.fillna(panel.model_id)
    for f in fam.value_counts().index[:5]:
        specs.append((f"drop family {f.split('/')[-1]}", "leave-one-family-out", (fam != f).values, "clonal"))

    rows = []
    for name, group, mask, variant in specs:
        sub = panel[mask].reset_index(drop=True)
        proxies = [("raw", "inflation"), ("residual (cubic)", "resid3")]
        if name == "baseline":
            proxies += [("residual (linear)", "resid1"), ("residual (quintic)", "resid5")]
        for plab, col in proxies:
            m = h2(sub, edges, sub[col], variant)
            rows.append(dict(check=name, group=group, proxy=plab, n=len(sub), h2=round(m.h2, 3),
                             ci_lo=round(m.ci[0], 3), ci_hi=round(m.ci[1], 3), p_lrt=float(m.p_lrt)))
            print(f"[{group:22s}] {name:32s} {plab:18s} n={len(sub):3d} h2={m.h2:.2f} "
                  f"[{m.ci[0]:.2f},{m.ci[1]:.2f}] p={m.p_lrt:.2g}")
    res = pd.DataFrame(rows)
    res.to_csv(f"{odir}/robustness.csv", index=False)

    # figure: forest plot, one row per check, raw and residual side by side
    order = res.drop_duplicates("check").check.tolist()
    base_rows = res[res.check == "baseline"]
    extra = base_rows[~base_rows.proxy.isin(["raw", "residual (cubic)"])]
    labels = order + [f"baseline, {p}" for p in extra.proxy]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(labels) + 1.2))
    for i, lab in enumerate(labels):
        yy = -i
        if lab.startswith("baseline, "):
            r = extra[extra.proxy == lab.split(", ", 1)[1]].iloc[0]
            pts = [(r, C["blue"], 0)]
        else:
            g = res[res.check == lab].set_index("proxy")
            pts = [(g.loc["raw"], C["ink2"], .12), (g.loc["residual (cubic)"], C["blue"], -.12)]
        for r, col, off in pts:
            ax.errorbar(r.h2, yy + off, xerr=[[r.h2 - r.ci_lo], [r.ci_hi - r.h2]], fmt="o", color=col,
                        ms=5, capsize=2, lw=1.2)
        n = res[res.check == lab.split(", ")[0]].n.iloc[0]
        ax.text(1.02, yy, f"n={n}", va="center", fontsize=8, color=C["ink2"],
                transform=ax.get_yaxis_transform())
    b = base_rows.set_index("proxy")
    ax.axvline(b.loc["raw"].h2, color=C["ink2"], ls=":", lw=1)
    ax.axvline(b.loc["residual (cubic)"].h2, color=C["blue"], ls=":", lw=1)
    ax.axvline(0, color=C["ink2"], lw=.8)
    ax.set_yticks([-i for i in range(len(labels))]); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(-.02, 1.02); ax.grid(axis="y", visible=False)
    ax.set_xlabel("h² of inflation (95% CI)")
    ax.errorbar([], [], fmt="o", color=C["ink2"], label="raw gap")
    ax.errorbar([], [], fmt="o", color=C["blue"], label="residual gap")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title("Inflation stays heritable under every alternative choice", loc="left")
    fig.tight_layout(); fig.savefig(f"{figdir}/P5_robustness.png"); plt.close(fig)

    nb = res[res.check != "baseline"]
    summ = {"n_checks": int(res.check.nunique()),
            "baseline": base_rows.to_dict("records"),
            "raw_h2_range": [float(nb[nb.proxy == "raw"].h2.min()), float(nb[nb.proxy == "raw"].h2.max())],
            "residual_h2_range": [float(nb[nb.proxy == "residual (cubic)"].h2.min()),
                                  float(nb[nb.proxy == "residual (cubic)"].h2.max())],
            "n_significant_p05": int((res.p_lrt < .05).sum()), "n_fits": int(len(res))}
    json.dump(summ, open(f"{odir}/summary_phase5.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))
    print(f"[done] phase 5 in {odir}/")


if __name__ == "__main__":
    main()
