#!/usr/bin/env python3
"""
Phase 2 - Decoupling contamination from genuine capability.

TA question: "how will you decouple contamination from genuine capability in the v1-v2 gap?"
Four tests, all on the Part 1 panel (no internet, reuses Part 2's REML code):

  A. Residual proxy: the raw gap v1_pct - v2_pct is mechanically correlated with v2_pct
     (regression to the mean). Residual inflation = v1_pct minus a cubic fit of v1_pct on
     v2_pct, so it is uncorrelated with measured capability by construction.
  B. Capability as a covariate: h2 of inflation with v2_pct added to the fixed effects.
  C. Genetic correlation r_g between inflation and capability: bivariate animal model
     (ML on covariate-adjusted traits, same kinship matrix), profile-likelihood 95% CI.
     r_g near 0 = the inherited part of inflation is not inherited capability.
  D. Placebo control: the same residual gap computed between pairs of CLEAN v2 tasks.
     If contaminated-vs-clean gaps are more heritable than clean-vs-clean gaps, the
     heritability is specific to contamination, not a generic "skill profile" effect.

Outputs (in <out>/phase2/): summary_phase2.json, h2_tests.csv, placebo.csv,
  figs/P2_regression_to_mean.png, P2_placebo.png, P2_rg_profile.png

Usage:  python phase2_decouple.py --data out_phase1/data --out out_phase1
"""
import argparse, itertools, json, os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part2_analysis import kinship, AnimalModel, design, style, C

V1 = ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"]
V2 = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
# contaminated v1 task -> closest clean v2 task (as in Part 1)
CONTAM_PAIRS = {"overall": ("v1_avg", "v2_avg"),
                "math (GSM8K vs MATH)": ("GSM8K", "MATH Lvl 5"),
                "knowledge (MMLU vs MMLU-Pro)": ("MMLU", "MMLU-PRO")}


# --------------------------------------------------------------------------- proxies
def pct(s):
    return s.rank(pct=True) * 100


def residual_gap(a, b, deg=3):
    """Residual of pct(a) on a cubic in pct(b): the part of a's rank not explained by b."""
    x, y = pct(b).values, pct(a).values
    ok = ~(np.isnan(x) | np.isnan(y))
    coef = np.polyfit(x[ok], y[ok], deg)
    r = np.full(len(x), np.nan)
    r[ok] = y[ok] - np.polyval(coef, x[ok])
    return pd.Series(r, index=a.index)


def zfill(v):
    v = pd.Series(v).astype(float)
    v = v.fillna(v.median())
    return ((v - v.mean()) / v.std()).values


def fit_h2(base, y, X):
    m = AnimalModel.__new__(AnimalModel); m.s, m.U = base.s, base.U
    m.fit(y, X)
    return m


# --------------------------------------------------------------------------- bivariate
class Bivariate:
    """Bivariate animal model on covariate-adjusted traits (ML).
    In the eigenbasis of A (A = U diag(s) U'), rotated observations are independent with
    2x2 covariance  s_i * G + E.  G = genetic (co)variance, E = residual (co)variance.
    r_g is fixed on a grid and the other 5 parameters are optimised -> profile likelihood."""

    def __init__(self, base, Y):
        self.s = base.s
        self.Z = base.U.T @ Y            # n x 2

    def nll(self, theta, rg):
        g1, g2 = np.exp(theta[0]), np.exp(theta[1])
        e1, e2 = np.exp(theta[2]), np.exp(theta[3])
        re = np.tanh(theta[4])
        g12 = rg * np.sqrt(g1 * g2)
        e12 = re * np.sqrt(e1 * e2)
        a = self.s * g1 + e1
        d = self.s * g2 + e2
        b = self.s * g12 + e12
        det = a * d - b * b
        if np.any(det <= 0):
            return 1e12
        z1, z2 = self.Z[:, 0], self.Z[:, 1]
        quad = (d * z1 * z1 - 2 * b * z1 * z2 + a * z2 * z2) / det
        return 0.5 * np.sum(np.log(det) + quad)

    def profile(self, rg, start):
        best = None
        for x0 in [start, np.r_[np.log(.3), np.log(.3), np.log(.7), np.log(.7), 0]]:
            r = minimize(self.nll, x0, args=(rg,), method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        return best

    def fit(self):
        grid = np.round(np.r_[np.linspace(-.98, .98, 50), 0.0], 4)
        grid = np.unique(grid)
        start = np.r_[np.log(.4), np.log(.4), np.log(.6), np.log(.6), 0]
        ll = []
        for g in grid:
            r = self.profile(g, start)
            ll.append(-r.fun); start = r.x
        ll = np.array(ll)
        k = int(np.argmax(ll))
        self.grid, self.ll, self.rg = grid, ll, grid[k]
        ok = grid[ll >= ll.max() - 1.92]
        self.ci = (ok.min(), ok.max())
        self.p0 = stats.chi2.sf(max(0, 2 * (ll.max() - ll[grid == 0.0][0])), 1)
        return self


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out_phase1/data")
    ap.add_argument("--out", default="out_phase1")
    ap.add_argument("--kinship", default="clonal", choices=["clonal", "diploid"],
                    help="primary kinship model (clonal = conservative)")
    a = ap.parse_args()
    odir = os.path.join(a.out, "phase2"); figdir = os.path.join(odir, "figs")
    os.makedirs(figdir, exist_ok=True)
    style()

    panel = pd.read_csv(f"{a.data}/panel.csv")
    nodes = pd.read_csv(f"{a.data}/nodes.csv")
    edges = pd.read_csv(f"{a.data}/edges.csv")
    panel = panel.dropna(subset=["inflation"]).drop_duplicates("model_id").reset_index(drop=True)
    for c in ["created_at", "n_params", "founder", "generation"]:
        if c not in panel and c in nodes:
            panel = panel.merge(nodes[["model_id", c]], on="model_id", how="left")
    n = len(panel)
    print(f"[load] panel {n} models")

    X = design(panel)
    Xv = X.values
    kin = {}
    for k in ["clonal", "diploid"]:
        A, _ = kinship(panel.model_id.tolist(), edges, k)
        kin[k] = AnimalModel(A)

    # ---------------- A. proxies
    raw = panel.inflation
    resid = residual_gap(panel.v1_avg, panel.v2_avg)
    cap = panel.v2_pct
    rtm = {"corr_raw_gap_vs_capability": float(stats.spearmanr(raw, cap).correlation),
           "corr_residual_gap_vs_capability": float(stats.spearmanr(resid, cap).correlation)}
    print(f"[A] corr(raw gap, capability)={rtm['corr_raw_gap_vs_capability']:.3f}  "
          f"corr(residual gap, capability)={rtm['corr_residual_gap_vs_capability']:.3f}")

    # ---------------- B. h2 tests
    tests = []
    Xcap = np.c_[Xv, zfill(cap)]
    specs = [("raw gap (Part 2 proxy)", zfill(raw), Xv),
             ("raw gap + capability covariate", zfill(raw), Xcap),
             ("residual gap", zfill(resid), Xv),
             ("capability (v2 rank)", zfill(cap), Xv)]
    for label, (va, vb) in CONTAM_PAIRS.items():
        if label == "overall":
            continue
        specs.append((f"residual gap: {label}", zfill(residual_gap(panel[va], panel[vb])), Xv))
    for k in ["clonal", "diploid"]:
        for label, y, XX in specs:
            m = fit_h2(kin[k], y, XX)
            tests.append(dict(kinship=k, test=label, h2=round(m.h2, 4), ci_lo=round(m.ci[0], 3),
                              ci_hi=round(m.ci[1], 3), p_lrt=float(m.p_lrt), n=n))
            print(f"[B] {k:7s} {label:45s} h2={m.h2:.2f} [{m.ci[0]:.2f},{m.ci[1]:.2f}] p={m.p_lrt:.2g}")
    tests = pd.DataFrame(tests)

    # ---------------- C. genetic correlation inflation x capability
    def adj(y):
        b = np.linalg.lstsq(Xv, y, rcond=None)[0]
        r = y - Xv @ b
        return r / r.std()
    Y = np.c_[adj(zfill(resid)), adj(zfill(cap))]
    bv = Bivariate(kin[a.kinship], Y).fit()
    print(f"[C] r_g(residual inflation, capability) = {bv.rg:+.2f} "
          f"[{bv.ci[0]:+.2f},{bv.ci[1]:+.2f}]  p(r_g=0)={bv.p0:.2g}")

    # ---------------- D. placebo: clean-vs-clean residual gaps
    plac = []
    base = kin[a.kinship]
    for t1, t2 in itertools.permutations(V2, 2):
        if t1 not in panel or t2 not in panel:
            continue
        y = zfill(residual_gap(panel[t1], panel[t2]))
        m = fit_h2(base, y, Xv)
        plac.append(dict(kind="placebo (clean vs clean)", pair=f"{t1} vs {t2}", h2=m.h2,
                         ci_lo=m.ci[0], ci_hi=m.ci[1], p_lrt=float(m.p_lrt)))
    # fair placebo for the OVERALL gap (an average of 6 vs 6 tasks): averaging lowers
    # noise and noise alone lowers h2, so compare against averages of clean tasks too
    for grp in itertools.combinations(V2, 3):
        rest = [t for t in V2 if t not in grp]
        ga, gb = panel[list(grp)].mean(axis=1), panel[rest].mean(axis=1)
        y = zfill(residual_gap(ga, gb))
        m = fit_h2(base, y, Xv)
        plac.append(dict(kind="placebo avg (3 clean vs 3 clean)", pair=f"{'+'.join(grp)} vs rest",
                         h2=m.h2, ci_lo=m.ci[0], ci_hi=m.ci[1], p_lrt=float(m.p_lrt)))
    for label, (va, vb) in CONTAM_PAIRS.items():
        y = zfill(residual_gap(panel[va], panel[vb]))
        m = fit_h2(base, y, Xv)
        plac.append(dict(kind="contaminated vs clean", pair=label, h2=m.h2,
                         ci_lo=m.ci[0], ci_hi=m.ci[1], p_lrt=float(m.p_lrt)))
    plac = pd.DataFrame(plac)
    pl = plac[plac.kind == "placebo (clean vs clean)"].h2.values           # single-task
    pla = plac[plac.kind == "placebo avg (3 clean vs 3 clean)"].h2.values  # averaged
    contam = plac[plac.kind.str.startswith("contaminated")].set_index("pair").h2
    # like-for-like reference set: overall gap vs averaged placebos, task gaps vs single-task
    ref = {k: (pla if k == "overall" else pl) for k in contam.index}
    placebo_rank = {k: float((ref[k] >= v).mean()) for k, v in contam.items()}
    print(f"[D] single-task placebo h2 median={np.median(pl):.2f} (n={len(pl)}), "
          f"averaged placebo median={np.median(pla):.2f} (n={len(pla)})")
    for k, v in contam.items():
        print(f"[D]   {k}: h2={v:.2f}; share of like-for-like placebos >= it = {placebo_rank[k]:.2f}")

    # ---------------- figures
    # regression to the mean
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=False)
    for ax, (v, lab) in zip(axs, [(raw, "raw gap  (v1 pct - v2 pct)"), (resid, "residual gap")]):
        ax.scatter(cap, v, s=9, color=C["ink2"], alpha=.45, lw=0)
        rho = stats.spearmanr(v, cap).correlation
        ax.axhline(0, color=C["ink2"], lw=.8)
        ax.set_xlabel("capability (v2 percentile)"); ax.set_ylabel(lab)
        ax.set_title(f"Spearman ρ with capability = {rho:+.2f}", fontsize=10.5)
    fig.suptitle("The raw gap is tied to capability; the residual gap is not",
                 x=.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{figdir}/P2_regression_to_mean.png"); plt.close(fig)

    # placebo
    # one row per comparison: placebo cloud (grey) + the contaminated gap (blue) on the same row
    rows_ = [("overall", pla, "overall gap\nvs 3-vs-3 clean averages"),
             ("math (GSM8K vs MATH)", pl, "GSM8K vs MATH\nvs single clean-task gaps"),
             ("knowledge (MMLU vs MMLU-Pro)", pl, "MMLU vs MMLU-Pro\nvs single clean-task gaps")]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    rng_j = np.random.default_rng(0)
    for i, (k, ref_h2, lab) in enumerate(rows_):
        yy = -i
        ax.scatter(ref_h2, yy + rng_j.uniform(-.13, .13, len(ref_h2)), color=C["grid"],
                   edgecolor=C["ink2"], s=26, lw=.6, zorder=2,
                   label="placebo: clean vs clean gaps" if i == 0 else None)
        r = plac[plac.pair == k].iloc[0]
        ax.errorbar(r.h2, yy, xerr=[[r.h2 - r.ci_lo], [r.ci_hi - r.h2]], fmt="o", color=C["blue"],
                    capsize=3, ms=7, zorder=3, label="contaminated vs clean gap" if i == 0 else None)
        ax.text(1.03, yy, f"{placebo_rank[k]:.0%} of placebos\nas high", va="center", fontsize=8,
                color=C["ink2"], transform=ax.get_yaxis_transform())
    ax.set_yticks([0, -1, -2]); ax.set_yticklabels([r[2] for r in rows_], fontsize=8.5)
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-2.5, .5); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"h² of residual gap ({a.kinship} kinship, 95% CI)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.45, -.22), ncol=2, frameon=False)
    ax.set_title("Are contamination gaps more heritable than ordinary skill gaps?")
    fig.savefig(f"{figdir}/P2_placebo.png"); plt.close(fig)

    # r_g profile
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(bv.grid, bv.ll - bv.ll.max(), color=C["blue"], lw=2)
    ax.axhline(-1.92, color=C["ink2"], ls="--", lw=1)
    ax.axvline(0, color=C["ink2"], lw=.8)
    ax.text(.98, -1.80, "95% CI cut-off", ha="right", fontsize=8.5, color=C["ink2"])
    ax.set_ylim(max(-8, (bv.ll - bv.ll.max()).min()), .3)
    ax.set_xlabel("genetic correlation r_g (residual inflation, capability)")
    ax.set_ylabel("profile log-likelihood")
    ax.set_title(f"r_g = {bv.rg:+.2f}  [{bv.ci[0]:+.2f}, {bv.ci[1]:+.2f}]")
    fig.savefig(f"{figdir}/P2_rg_profile.png"); plt.close(fig)

    summ = {"n_panel": n, "primary_kinship": a.kinship,
            "A_regression_to_mean": rtm,
            "B_h2_tests": tests.to_dict("records"),
            "C_genetic_correlation": {"rg": float(bv.rg), "ci95": [float(bv.ci[0]), float(bv.ci[1])],
                                      "p_rg_eq_0": float(bv.p0)},
            "D_placebo": {"placebo_h2_median": float(np.median(pl)),
                          "placebo_h2_max": float(pl.max()), "n_placebo_pairs": int(len(pl)),
                          "placebo_avg_h2_median": float(np.median(pla)),
                          "placebo_avg_h2_max": float(pla.max()), "n_placebo_avg": int(len(pla)),
                          "contaminated_h2": {k: float(v) for k, v in contam.items()},
                          "share_of_placebos_at_or_above": placebo_rank}}
    json.dump(summ, open(f"{odir}/summary_phase2.json", "w"), indent=2)
    tests.to_csv(f"{odir}/h2_tests.csv", index=False)
    plac.round(4).to_csv(f"{odir}/placebo.csv", index=False)
    print(f"[done] phase 2 in {odir}/")


if __name__ == "__main__":
    main()
