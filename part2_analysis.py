#!/usr/bin/env python3
"""
ADA Jackfruit — Parts 2-4: heritability, inheritance mode, lineage-only detection.

Reads Part 1 outputs (out/data/{nodes,edges,panel}.csv) — no internet needed.

Part 2  Animal model  y = Xb + u + e,  u ~ N(0, s2a*A),  e ~ N(0, s2e*I)
        REML estimate of h2 = s2a/(s2a+s2e) for inflation vs genuine capability,
        under two kinship (A) definitions, + profile-likelihood CI, LRT, permutation null.
Part 3  Inheritance mode: mid-parent regression by relation type, merge additivity
        (child vs mean of parents), accumulation across generations.
Part 4  Lineage-only prediction: 5-fold CV of pedigree BLUP vs non-lineage baselines,
        contamination detector ROC, proxy validation against leaderboard flags.

Usage:  python part2_analysis.py --data out/data --out out
        python part2_analysis.py --data out_mock/data --out out_mock   (smoke test)
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular
from scipy.optimize import minimize_scalar
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", surface="#fcfcfb")
TRAITS = {"inflation": "Inflation (overall)", "infl_math": "Inflation: GSM8K vs MATH",
          "infl_knowledge": "Inflation: MMLU vs MMLU-Pro", "v2_pct": "Capability (v2 rank)"}


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "axes.edgecolor": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False})


# =========================================================================== kinship
def kinship(panel_ids, edges, variant):
    """Relationship matrix A among panel models via u = T m, A = T D T'.
    P[child, parent] = weight of parent's genetic value in child's.
      diploid : textbook animal model — fine-tune = 0.5*parent + 0.5*unknown,
                merge of k parents = mean of parents.  D = 1 - sum(w^2).
      clonal  : fine-tune ≈ copy of parent + drift (w=1, D=0.25);
                merge = mean of parents, D = max(0.25, 1 - sum(w^2)).
    Founders: D = 1. Quantized edges dropped (same weights, not a new model)."""
    e = edges[edges.relation != "quantized"]
    par = e.groupby("child").parent.apply(list).to_dict()
    # prune pedigree to panel models + all their ancestors
    keep, stack = set(), list(panel_ids)
    while stack:
        n = stack.pop()
        if n in keep: continue
        keep.add(n); stack.extend(par.get(n, []))
    # topological order (parents first)
    order, seen = [], set()
    def visit(n):
        st = [(n, False)]
        while st:
            x, done = st.pop()
            if done: order.append(x); continue
            if x in seen: continue
            seen.add(x); st.append((x, True))
            st.extend((p, False) for p in par.get(x, []) if p in keep and p not in seen)
    for n in sorted(keep): visit(n)
    idx = {n: i for i, n in enumerate(order)}
    N = len(order); rows, cols, vals = [], [], []
    D = np.ones(N)
    for n in order:
        ps = [p for p in par.get(n, []) if p in idx]
        if not ps: continue
        k = len(ps)
        if variant == "diploid":
            w = [0.5] if k == 1 else [1.0 / k] * k
            D[idx[n]] = 1 - sum(x * x for x in w)
        else:
            w = [1.0] if k == 1 else [1.0 / k] * k
            D[idx[n]] = 0.25 if k == 1 else max(0.25, 1 - sum(x * x for x in w))
        for p, wi in zip(ps, w):
            rows.append(idx[n]); cols.append(idx[p]); vals.append(wi)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    # T = (I-P)^-1 ; need rows of T for panel models: solve (I-P)^T X = E
    IPt = (sp.identity(N, format="csr") - P).T.tocsr()   # upper-triangular in topo order
    pi = np.array([idx[m] for m in panel_ids])
    E = np.zeros((N, len(pi))); E[pi, np.arange(len(pi))] = 1
    X = spsolve_triangular(IPt, E, lower=False)          # X = T^T[:, panel]
    A = (X * D[:, None]).T @ X
    return A, N


# =========================================================================== REML
class AnimalModel:
    """Exact REML via eigendecomposition of A (FaST-LMM style): one eigh, then each
    h2 evaluation is O(n p^2). V = s2 * (h2*A + (1-h2)*I)."""
    def __init__(self, A):
        s, U = np.linalg.eigh(A)
        self.s, self.U = np.clip(s, 0, None), U

    def fit(self, y, X):
        self.ys, self.Xs = self.U.T @ y, self.U.T @ X
        n, p = X.shape
        self.n, self.p = n, p
        grid = np.r_[np.linspace(0, 0.99, 100), 0.995, 0.999]
        ll = np.array([self.loglik(h) for h in grid])
        g0 = grid[np.argmax(ll)]
        r = minimize_scalar(lambda h: -self.loglik(h), bounds=(max(0, g0 - .02), min(.999, g0 + .02)),
                            method="bounded")
        self.h2, self.ll = (r.x, -r.fun) if -r.fun > ll.max() else (g0, ll.max())
        self.grid, self.prof = grid, ll
        self.ll0 = self.loglik(0.0)
        lr = max(0, 2 * (self.ll - self.ll0))
        self.p_lrt = 0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0   # boundary mixture
        ok = np.r_[grid[ll >= self.ll - 1.92], self.h2]               # 95% profile CI
        self.ci = (ok.min(), ok.max())
        return self

    def loglik(self, h):
        d = h * self.s + (1 - h)
        w = 1 / d
        XtW = self.Xs.T * w
        XWX = XtW @ self.Xs
        b = np.linalg.solve(XWX, XtW @ self.ys)
        r = self.ys - self.Xs @ b
        s2 = (r * w) @ r / (self.n - self.p)
        sign, ld = np.linalg.slogdet(XWX)
        return -0.5 * ((self.n - self.p) * np.log(s2) + np.log(d).sum() + ld)

    def perm_h2(self, X, n_perm, rng):
        out = []
        for _ in range(n_perm):
            yp = rng.permutation(self._y)
            out.append(AnimalModel.__new__(AnimalModel)._quick(self, yp, X))
        return np.array(out)

    def _quick(self, base, y, X):
        self.s, self.U = base.s, base.U
        self.ys, self.Xs, self.n, self.p = base.U.T @ y, base.U.T @ X, *X.shape
        grid = np.linspace(0, 0.99, 50)
        return grid[np.argmax([self.loglik(h) for h in grid])]


def design(df):
    """Non-lineage covariates: size, date, architecture, model type, MoE.
    Missing values → median + indicator, so no rows are dropped."""
    X = pd.DataFrame(index=df.index)
    X["intercept"] = 1.0
    size = pd.to_numeric(df.get("params_b"), errors="coerce")
    if "n_params" in df:
        size = size.fillna(pd.to_numeric(df["n_params"], errors="coerce") / 1e9)
    date = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
    num = {"log_params": np.log1p(size),
           "year": date.dt.year + date.dt.dayofyear / 366 if date.notna().any() else pd.Series(np.nan, index=df.index)}
    for k, v in num.items():
        v = pd.to_numeric(v, errors="coerce")
        if v.notna().sum() > 10 and v.std() > 0:
            X[k + "_miss"] = v.isna().astype(float)
            X[k] = (v.fillna(v.median()) - v.mean()) / v.std()
            if X[k + "_miss"].sum() == 0: X.drop(columns=k + "_miss", inplace=True)
    for c in ["architecture", "type"]:
        if c in df and df[c].notna().sum() > 10:
            top = df[c].value_counts().index[:8]
            for t in top[1:]:
                X[f"{c}={t}"] = (df[c] == t).astype(float)
    if "moe" in df and df["moe"].notna().any():
        X["moe"] = df["moe"].astype(str).str.lower().eq("true").astype(float)
    X = X.loc[:, X.std(ddof=0).gt(0) | (X.columns == "intercept")]
    return X


# =========================================================================== part 2
def part2(panel, edges, figdir, n_perm, rng):
    ids = panel.model_id.tolist()
    X = design(panel)
    res, models = [], {}
    for variant in ["diploid", "clonal"]:
        A, N = kinship(ids, edges, variant)
        am = AnimalModel(A)
        print(f"[part2] A ({variant}): {len(ids)} panel models, {N} pedigree nodes, "
              f"mean off-diag {A[np.triu_indices_from(A, 1)].mean():.4f}")
        for t in TRAITS:
            if t not in panel or panel[t].isna().all(): continue
            y = panel[t].fillna(panel[t].median()).values.astype(float)
            y = (y - y.mean()) / y.std()
            m = AnimalModel.__new__(AnimalModel); m.s, m.U = am.s, am.U
            m.fit(y, X.values); m._y = y
            res.append(dict(trait=t, kinship=variant, h2=m.h2, ci_lo=m.ci[0], ci_hi=m.ci[1],
                            p_lrt=m.p_lrt, n=len(y)))
            models[(variant, t)] = (m, A)
            print(f"   {t:15s} h2={m.h2:.3f} [{m.ci[0]:.2f},{m.ci[1]:.2f}] p={m.p_lrt:.2g}")
    res = pd.DataFrame(res)

    # permutation null for the headline trait (diploid)
    m, _ = models[("diploid", "inflation")]
    null = m.perm_h2(X.values, n_perm, rng)
    p_perm = (1 + (null >= m.h2).sum()) / (1 + n_perm)

    # --- fig 08: h2 bars
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    tr = [t for t in TRAITS if t in res.trait.values]
    xs = np.arange(len(tr)); wdt = 0.36
    for j, (v, col) in enumerate([("diploid", C["blue"]), ("clonal", C["orange"])]):
        r = res[res.kinship == v].set_index("trait").loc[tr]
        ax.bar(xs + (j - .5) * wdt, r.h2, wdt * .92, color=col, label=f"kinship: {v}")
        ax.errorbar(xs + (j - .5) * wdt, r.h2, yerr=[r.h2 - r.ci_lo, r.ci_hi - r.h2],
                    fmt="none", ecolor=C["ink"], elinewidth=1, capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels([TRAITS[t].replace(": ", ":\n") for t in tr], fontsize=9)
    ax.set_ylabel("heritability h²"); ax.set_ylim(0, 1.05)
    ax.set_title("Heritability of inflation vs capability (REML, 95% CI)", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.savefig(f"{figdir}/08_h2_bars.png"); plt.close(fig)

    # --- fig 09: profile likelihood
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for t, col in [("inflation", C["orange"]), ("v2_pct", C["blue"])]:
        if ("diploid", t) not in models: continue
        mm, _ = models[("diploid", t)]
        ax.plot(mm.grid, mm.prof - mm.prof.max(), color=col, lw=2,
                label=f"{TRAITS[t]} (ĥ²={mm.h2:.2f})")
    ax.axhline(-1.92, color=C["ink2"], ls="--", lw=1)
    ax.text(0.01, -1.75, "95% CI cut-off", color=C["ink2"], fontsize=8)
    ax.set_ylim(-12, 0.5); ax.set_xlabel("h²"); ax.set_ylabel("Δ REML log-likelihood")
    ax.set_title("Profile likelihood of h²"); ax.legend(loc="lower center")
    fig.savefig(f"{figdir}/09_profile_likelihood.png"); plt.close(fig)

    # --- fig 10: permutation null
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(null, bins=np.linspace(0, max(1e-3, null.max(), m.h2) * 1.1, 30), color=C["ink2"],
            alpha=.6, label=f"shuffled labels (n={n_perm})")
    ax.axvline(m.h2, color=C["orange"], lw=2.5, label=f"observed ĥ² = {m.h2:.2f}")
    ax.set_xlabel("h² of inflation"); ax.set_ylabel("count")
    ax.set_title(f"Permutation test (p = {p_perm:.3f})"); ax.legend()
    fig.savefig(f"{figdir}/10_permutation_null.png"); plt.close(fig)
    return res, p_perm, X, models


# =========================================================================== part 3
def part3(panel, edges, nodes, figdir):
    out = {}
    pi = panel.set_index("model_id").inflation
    e = edges[(edges.relation != "quantized") & edges.child.isin(pi.index)]
    # children whose *all* non-quantized parents are scored
    g = e.groupby("child").agg(parents=("parent", list), rel=("relation", lambda r: "merge"
                               if (r == "merge").any() or len(r) > 1 else "finetune"))
    g = g[g.parents.apply(lambda ps: all(p in pi.index for p in ps))]
    g["mid"] = g.parents.apply(lambda ps: pi.loc[ps].mean())
    g["pmin"] = g.parents.apply(lambda ps: pi.loc[ps].min())
    g["pmax"] = g.parents.apply(lambda ps: pi.loc[ps].max())
    g["child"] = pi.loc[g.index].values
    g["delta"] = g.child - g.mid

    # fig 11: mid-parent regression by relation
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for rel, col in [("finetune", C["blue"]), ("merge", C["orange"])]:
        d = g[g.rel == rel]
        if len(d) < 5: continue
        sl = stats.linregress(d.mid, d.child)
        out[f"midparent_{rel}"] = dict(n=len(d), slope=sl.slope, slope_se=sl.stderr,
                                        intercept=sl.intercept, r=sl.rvalue, p=sl.pvalue)
        ax.scatter(d.mid, d.child, s=10, alpha=.45, color=col, linewidths=0)
        xs = np.linspace(d.mid.min(), d.mid.max(), 20)
        ax.plot(xs, sl.intercept + sl.slope * xs, color=col, lw=2.2,
                label=f"{rel}: β={sl.slope:.2f}±{1.96*sl.stderr:.2f} (n={len(d)})")
    lim = np.nanpercentile(np.abs(np.r_[g.mid, g.child]), 99) if len(g) else 50
    ax.plot([-lim, lim], [-lim, lim], color=C["ink2"], ls="--", lw=1, label="child = mid-parent")
    ax.set_xlabel("mid-parent inflation"); ax.set_ylabel("child inflation")
    ax.set_title("Inheritance by relation type"); ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(f"{figdir}/11_midparent_by_relation.png"); plt.close(fig)

    # fig 12: merge additivity — child minus mean(parents)
    mg = g[g.rel == "merge"]
    if len(mg) >= 5:
        t = stats.ttest_1samp(mg.delta, 0)
        within = ((mg.child >= mg.pmin) & (mg.child <= mg.pmax)).mean()
        above = (mg.child > mg.pmax).mean()
        out["merge_additivity"] = dict(n=len(mg), mean_delta=mg.delta.mean(),
                                       ci95=1.96 * mg.delta.std() / np.sqrt(len(mg)),
                                       t=t.statistic, p=t.pvalue, frac_within_parents=within,
                                       frac_above_both=above)
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.hist(mg.delta, bins=30, color=C["orange"], alpha=.8)
        ax.axvline(0, color=C["ink2"], lw=1)
        ax.axvline(mg.delta.mean(), color=C["ink"], lw=2,
                   label=f"mean = {mg.delta.mean():+.1f} pts (p = {t.pvalue:.2g})")
        ax.set_xlabel("child inflation − mean(parents) [percentile pts]"); ax.set_ylabel("merges")
        ax.set_title(f"Merge additivity — {within:.0%} of merges fall within parents' range")
        ax.legend(loc="upper left")
        fig.savefig(f"{figdir}/12_merge_additivity.png"); plt.close(fig)

    # fig 13: accumulation across generations
    gen = panel.dropna(subset=["generation"])
    gg = gen.groupby(gen.generation.clip(upper=6)).inflation.agg(["mean", "sem", "count"])
    gg = gg[gg["count"] >= 5]
    sl = stats.linregress(gen.generation, gen.inflation) if len(gen) > 10 else None
    step = g.delta  # per-generation change along edges
    out["generation"] = dict(slope=sl.slope if sl else None, p=sl.pvalue if sl else None,
                             mean_step_change=step.mean() if len(step) else None,
                             step_p=stats.ttest_1samp(step, 0).pvalue if len(step) > 5 else None)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.errorbar(gg.index, gg["mean"], yerr=1.96 * gg["sem"], fmt="o-", color=C["blue"],
                lw=2, ms=7, capsize=3)
    for x, (mu, n) in zip(gg.index, gg[["mean", "count"]].values):
        ax.annotate(f"n={int(n)}", (x, mu), textcoords="offset points", xytext=(9, -3),
                    ha="left", fontsize=8, color=C["ink2"])
    ax.axhline(0, color=C["ink2"], lw=1)
    ax.set_xticks(gg.index); ax.set_xticklabels([f"{int(i)}" + ("+" if i == 6 else "") for i in gg.index])
    ax.set_xlabel("generation (0 = base model)"); ax.set_ylabel("mean inflation (±95% CI)")
    ttl = f"Accumulation across generations (slope {sl.slope:+.2f}/gen, p={sl.pvalue:.2g})" if sl else "Inflation by generation"
    ax.set_title(ttl, fontsize=11)
    fig.savefig(f"{figdir}/13_generation_trend.png"); plt.close(fig)
    return out


# =========================================================================== part 4
def gp_predict(A, y, X, tr, te, h2):
    """BLUP of u for test models from training models only, plus fixed effects
    fitted on train. lam = (1-h2)/h2."""
    lam = (1 - h2) / max(h2, 1e-3)
    Xt = X[tr]
    b = np.linalg.lstsq(Xt, y[tr], rcond=None)[0]
    r = y[tr] - Xt @ b
    K = A[np.ix_(tr, tr)] + lam * np.eye(len(tr))
    alpha = np.linalg.solve(K, r)
    return X[te] @ b + A[np.ix_(te, tr)] @ alpha, X[te] @ b


def part4(panel, models, X, figdir, rng, folds=5):
    out = {}
    y = panel.inflation.values.astype(float)
    m, A = models[("diploid", "inflation")]
    Xv = X.values
    fam = panel.founder.values
    k = rng.permutation(len(y)) % folds
    pred = {"Covariates only\n(size, arch, date, type)": np.zeros_like(y),
            "Family mean\n(founder)": np.zeros_like(y),
            "Pedigree BLUP\n+ covariates": np.zeros_like(y)}
    keys = list(pred)
    for f in range(folds):
        tr, te = np.where(k != f)[0], np.where(k == f)[0]
        # refit h2 on training fold only (no leakage)
        sub = AnimalModel(A[np.ix_(tr, tr)])
        yt = (y[tr] - y[tr].mean()) / y[tr].std()
        h2 = sub.fit(yt, Xv[tr]).h2
        blup, cov = gp_predict(A, y, Xv, tr, te, h2)
        pred[keys[2]][te], pred[keys[0]][te] = blup, cov
        fm = pd.Series(y[tr]).groupby(fam[tr]).mean()
        pred[keys[1]][te] = pd.Series(fam[te]).map(fm).fillna(y[tr].mean()).values
    r2 = {kk: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum() for kk, p in pred.items()}
    rho = {kk: stats.spearmanr(y, p).correlation for kk, p in pred.items()}
    out["cv_r2"] = {kk.replace("\n", " "): v for kk, v in r2.items()}
    out["cv_spearman"] = {kk.replace("\n", " "): v for kk, v in rho.items()}

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cols = [C["ink2"], C["aqua"], C["orange"]]
    ax.bar(range(3), [r2[kk] for kk in keys], color=cols, width=.55)
    for i, kk in enumerate(keys):
        ax.text(i, r2[kk], f"R²={r2[kk]:.2f}\nρ={rho[kk]:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel(f"{folds}-fold CV R²"); ax.axhline(0, color=C["ink2"], lw=1)
    ax.set_ylim(min(0, min(r2.values())) - .05, max(r2.values()) * 1.35 + .05)
    ax.set_title("Predicting a model's inflation without its scores"); ax.grid(axis="x", visible=False)
    fig.savefig(f"{figdir}/14_cv_prediction.png"); plt.close(fig)

    # detector: flag top-20% inflation using held-out predictions
    lab = (y >= np.quantile(y, .8)).astype(int)
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    out["detector_auc"] = {}
    for kk, col in zip(keys, cols):
        fpr, tpr, _ = roc_curve(lab, pred[kk]); auc = roc_auc_score(lab, pred[kk])
        out["detector_auc"][kk.replace("\n", " ")] = auc
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{kk.splitlines()[0]} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=C["grid"], ls="--")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("Lineage-only contamination detector\n(top-20% inflation)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(f"{figdir}/15_detector_roc.png"); plt.close(fig)

    # proxy validation: leaderboard 'Flagged' models + agreement between sub-proxies
    if "flagged" in panel and panel.flagged.astype(str).str.lower().eq("true").sum() >= 5:
        fl = panel.flagged.astype(str).str.lower().eq("true")
        u = stats.mannwhitneyu(panel.inflation[fl], panel.inflation[~fl], alternative="greater")
        out["flagged_validation"] = dict(n_flagged=int(fl.sum()),
                                         median_flagged=panel.inflation[fl].median(),
                                         median_other=panel.inflation[~fl].median(), p=u.pvalue)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        bp = ax.boxplot([panel.inflation[~fl], panel.inflation[fl]], widths=.5, patch_artist=True,
                        labels=[f"not flagged\n(n={(~fl).sum()})", f"flagged\n(n={fl.sum()})"])
        for b, col in zip(bp["boxes"], [C["blue"], C["orange"]]):
            b.set_facecolor(col); b.set_alpha(.7)
        ax.set_ylabel("inflation"); ax.set_title(f"Proxy validation: leaderboard flags (p={u.pvalue:.2g})", fontsize=11)
        fig.savefig(f"{figdir}/16_flagged_validation.png"); plt.close(fig)
    if {"infl_math", "infl_knowledge"} <= set(panel):
        d = panel[["infl_math", "infl_knowledge"]].dropna()
        out["subproxy_spearman"] = stats.spearmanr(d.infl_math, d.infl_knowledge).correlation
    return out


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--max-panel", type=int, default=6000,
                    help="subsample panel if larger (eigh is O(n^3))")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    figdir = os.path.join(a.out, "figs"); os.makedirs(figdir, exist_ok=True)
    style()
    panel = pd.read_csv(f"{a.data}/panel.csv")
    edges = pd.read_csv(f"{a.data}/edges.csv")
    nodes = pd.read_csv(f"{a.data}/nodes.csv")
    panel = panel.dropna(subset=["inflation"]).drop_duplicates("model_id").reset_index(drop=True)
    if len(panel) > a.max_panel:
        panel = panel.sample(a.max_panel, random_state=a.seed).reset_index(drop=True)
    for c in ["created_at", "n_params", "founder", "generation"]:
        if c not in panel and c in nodes:
            panel = panel.merge(nodes[["model_id", c]], on="model_id", how="left")
    print(f"[load] panel {len(panel)} models, {len(edges)} edges")

    res, p_perm, X, models = part2(panel, edges, figdir, a.perm, rng)
    p3 = part3(panel, edges, nodes, figdir)
    p4 = part4(panel, models, X, figdir, rng)

    res.to_csv(f"{a.out}/heritability.csv", index=False)
    summ = {"n_panel": len(panel), "covariates": list(X.columns),
            "heritability": res.round(4).to_dict("records"),
            "permutation_p_inflation": p_perm, "inheritance": p3, "prediction": p4}
    json.dump(summ, open(f"{a.out}/summary_part2.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "covariates"}, indent=1, default=float)[:3000])
    print(f"[done] figures 08-16 in {figdir}/  summary in {a.out}/summary_part2.json")


if __name__ == "__main__":
    main()
