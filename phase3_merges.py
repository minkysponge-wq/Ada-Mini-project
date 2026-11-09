#!/usr/bin/env python3
"""
Phase 3 - Inheritance mode under merging (one-pager point 3).

Is a merge's score additive (child ~ mean of its parents), and is any "merge bonus"
concentrated on the contaminated v1 benchmarks rather than the clean v2 ones?

Why per leaderboard: only a handful of merges have the child AND every parent scored on
both v1 and v2, so additivity of the inflation proxy itself cannot be tested. Each
leaderboard on its own has hundreds of fully scored merges. Scores are converted to
percentile ranks within each leaderboard so v1 and v2 effects share one scale.

Tests, separately for v1 (contaminated) and v2 (clean):
  1. delta = child pct - mean(parents pct): mean, 95% CI, t-test and Wilcoxon
  2. share of merges inside the parents' range / above both parents
  3. regression child ~ mid-parent (additive inheritance -> slope 1, intercept 0)
  4. per-task merge bonus: is it largest on the most-suspected tasks (GSM8K, TruthfulQA)?
  5. fine-tunes as a reference: child - parent for single-parent fine-tunes
  6. v1 vs v2 merge bonus (Welch t-test; caveat: the two leaderboards cover different eras)

Outputs (in <out>/phase3/): summary_phase3.json, merge_pairs_v1.csv, merge_pairs_v2.csv,
  per_task_bonus.csv, figs/12_merge_additivity.png, P3_task_bonus.png, P3_midparent.png

Usage:  python phase3_merges.py --data out_phase1/data --out out_phase1
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part2_analysis import style, C

LB = {"v1": {"avg": "v1_avg", "tasks": ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"]},
      "v2": {"avg": "v2_avg", "tasks": ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]}}
SUSPECT = {"GSM8K", "TruthfulQA"}          # named as contaminated in the one-pager


def pct(s):
    return s.rank(pct=True) * 100


def mean_ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3:
        return dict(n=len(x), mean=float(np.mean(x)) if len(x) else None)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.ttest_1samp(x, 0)
    w = stats.wilcoxon(x) if np.any(x != 0) else None
    return dict(n=int(len(x)), mean=float(x.mean()), ci95=[float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)],
                median=float(np.median(x)), t=float(t.statistic), p_t=float(t.pvalue),
                p_wilcoxon=float(w.pvalue) if w else None)


def merge_table(nodes, par, version, strict=True):
    """One row per merge child with the child and (all / >=2) parents scored on `version`."""
    cfg = LB[version]
    scored = nodes.dropna(subset=[cfg["avg"]]).set_index("model_id")
    cols = [cfg["avg"]] + [t for t in cfg["tasks"] if t in scored]
    P = pd.DataFrame({c: pct(scored[c]) for c in cols})        # percentiles within leaderboard
    rows = []
    for child, ps in par.items():
        if len(ps) < 2 or child not in P.index:
            continue
        have = [p for p in ps if p in P.index]
        if (strict and len(have) < len(ps)) or len(have) < 2:
            continue
        pp = P.loc[have]
        r = {"child": child, "n_parents": len(ps), "n_scored_parents": len(have)}
        for c in cols:
            ch, mid = P.at[child, c], pp[c].mean()
            r[f"{c}|child"], r[f"{c}|mid"] = ch, mid
            r[f"{c}|delta"] = ch - mid
            r[f"{c}|pmin"], r[f"{c}|pmax"] = pp[c].min(), pp[c].max()
        rows.append(r)
    return pd.DataFrame(rows), P


def finetune_table(par, rel, P, avg):
    rows = []
    for child, ps in par.items():
        if len(ps) != 1 or rel.get(child) != "finetune":
            continue
        p = ps[0]
        if child in P.index and p in P.index:
            rows.append({"child": child, "parent": p, "delta": P.at[child, avg] - P.at[p, avg],
                         "child_pct": P.at[child, avg], "parent_pct": P.at[p, avg]})
    return pd.DataFrame(rows)


def summarise(mt, avg):
    d = mt[f"{avg}|delta"]
    within = ((mt[f"{avg}|child"] >= mt[f"{avg}|pmin"]) & (mt[f"{avg}|child"] <= mt[f"{avg}|pmax"])).mean()
    above = (mt[f"{avg}|child"] > mt[f"{avg}|pmax"]).mean()
    below = (mt[f"{avg}|child"] < mt[f"{avg}|pmin"]).mean()
    sl = stats.linregress(mt[f"{avg}|mid"], mt[f"{avg}|child"])
    t1 = (sl.slope - 1) / sl.stderr
    return {"delta": mean_ci(d), "share_within_parents": float(within),
            "share_above_both": float(above), "share_below_both": float(below),
            "midparent_slope": float(sl.slope), "slope_se": float(sl.stderr),
            "p_slope_eq_1": float(2 * stats.t.sf(abs(t1), len(mt) - 2)),
            "intercept": float(sl.intercept), "r": float(sl.rvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out_phase1/data")
    ap.add_argument("--out", default="out_phase1")
    a = ap.parse_args()
    odir = os.path.join(a.out, "phase3"); figdir = os.path.join(odir, "figs")
    os.makedirs(figdir, exist_ok=True)
    style()

    nodes = pd.read_csv(f"{a.data}/nodes.csv").drop_duplicates("model_id")
    edges = pd.read_csv(f"{a.data}/edges.csv")
    e = edges[edges.relation != "quantized"]
    par = e.groupby("child").parent.apply(list).to_dict()
    rel = e.groupby("child").relation.first().to_dict()

    out, tables, ft, task_rows = {}, {}, {}, []
    for v, cfg in LB.items():
        mt, P = merge_table(nodes, par, v, strict=True)
        mt_l, _ = merge_table(nodes, par, v, strict=False)
        tables[v] = mt
        mt.to_csv(f"{odir}/merge_pairs_{v}.csv", index=False)
        avg = cfg["avg"]
        out[v] = {"strict_all_parents_scored": summarise(mt, avg),
                  "lenient_2plus_parents_scored": summarise(mt_l, avg)}
        f = finetune_table(par, rel, P, avg); ft[v] = f
        out[v]["finetune_reference"] = mean_ci(f.delta)
        for t in cfg["tasks"]:
            if f"{t}|delta" in mt:
                s = mean_ci(mt[f"{t}|delta"])
                task_rows.append(dict(leaderboard=v, task=t, suspect=t in SUSPECT, n=s["n"],
                                      mean_delta=s["mean"], ci_lo=s["ci95"][0], ci_hi=s["ci95"][1],
                                      p=s["p_t"]))
        s = out[v]["strict_all_parents_scored"]
        print(f"[{v}] merges n={s['delta']['n']}: child - mean(parents) = {s['delta']['mean']:+.1f} pct pts "
              f"[{s['delta']['ci95'][0]:+.1f},{s['delta']['ci95'][1]:+.1f}] p={s['delta']['p_t']:.2g}; "
              f"within parents {s['share_within_parents']:.0%}, above both {s['share_above_both']:.0%}; "
              f"slope {s['midparent_slope']:.2f} (p slope=1: {s['p_slope_eq_1']:.2g})")
        fr = out[v]["finetune_reference"]
        print(f"[{v}] fine-tunes n={fr['n']}: child - parent = {fr['mean']:+.1f} [{fr['ci95'][0]:+.1f},{fr['ci95'][1]:+.1f}]")

    # v1 vs v2 merge bonus
    d1, d2 = tables["v1"]["v1_avg|delta"], tables["v2"]["v2_avg|delta"]
    w = stats.ttest_ind(d1, d2, equal_var=False)
    out["v1_vs_v2_merge_bonus"] = {"v1_mean": float(d1.mean()), "v2_mean": float(d2.mean()),
                                   "difference": float(d1.mean() - d2.mean()), "welch_t": float(w.statistic),
                                   "p": float(w.pvalue),
                                   "caveat": "v1 and v2 merges are mostly different models from different eras"}
    # adjusted for parent rank (regression to the mean): delta ~ is_v1 + midparent.
    # Robustness: "broken" merges (child below the 5th pct while parents average above the
    # 30th) are failed weight merges; they collapse mostly on v2's generative tasks, so the
    # comparison is repeated without them.
    def adjusted(drop_broken):
        parts = []
        for v, flag in [("v1", 1.0), ("v2", 0.0)]:
            avg = LB[v]["avg"]; t = tables[v]
            d = pd.DataFrame({"mid": t[f"{avg}|mid"], "delta": t[f"{avg}|delta"],
                              "child": t[f"{avg}|child"], "is_v1": flag})
            if drop_broken:
                d = d[~((d.child < 5) & (d.mid > 30))]
            parts.append(d)
        mm = pd.concat(parts, ignore_index=True)
        Xa = np.c_[np.ones(len(mm)), mm.is_v1, mm.mid - 50]
        b, *_ = np.linalg.lstsq(Xa, mm.delta.values, rcond=None)
        res = mm.delta.values - Xa @ b
        s2 = res @ res / (len(mm) - Xa.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Xa.T @ Xa)))
        p_v = 2 * stats.t.sf(abs(b[1] / se[1]), len(mm) - Xa.shape[1])
        return {"v1_minus_v2_bonus": float(b[1]),
                "ci95": [float(b[1] - 1.96 * se[1]), float(b[1] + 1.96 * se[1])], "p": float(p_v),
                "n_v1": int(len(parts[0])), "n_v2": int(len(parts[1])),
                "mean_parent_pct_v1": float(parts[0].mid.mean()), "mean_parent_pct_v2": float(parts[1].mid.mean())}

    for key, drop in [("all_merges", False), ("without_broken_merges", True)]:
        r = adjusted(drop)
        out.setdefault("v1_vs_v2_adjusted_for_parent_rank", {})[key] = r
        print(f"[v1 vs v2 | parent rank, {key}] v1 bonus minus v2 bonus = {r['v1_minus_v2_bonus']:+.1f} "
              f"[{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}] p={r['p']:.2g}  (n v1={r['n_v1']}, v2={r['n_v2']})")
    for v in LB:
        avg = LB[v]["avg"]; t = tables[v]
        broken = (t[f"{avg}|child"] < 5) & (t[f"{avg}|mid"] > 30)
        out[v]["broken_merges"] = int(broken.sum())
        out[v]["delta_without_broken"] = mean_ci(t.loc[~broken, f"{avg}|delta"])

    # merges scored on both leaderboards (direct, within-model contrast; usually tiny)
    both = tables["v1"].merge(tables["v2"], on="child", suffixes=("", "_"))
    out["merges_scored_on_both"] = int(len(both))
    if len(both) >= 5:
        dd = both["v1_avg|delta"] - both["v2_avg|delta"]
        out["within_model_v1_minus_v2_bonus"] = mean_ci(dd)
    print(f"[v1 vs v2] merge bonus v1 {d1.mean():+.1f} vs v2 {d2.mean():+.1f}  (Welch p={w.pvalue:.2g}); "
          f"merges scored on both: {len(both)}")

    tb = pd.DataFrame(task_rows)
    tb.to_csv(f"{odir}/per_task_bonus.csv", index=False)
    v1t = tb[tb.leaderboard == "v1"]
    out["per_task"] = tb.round(3).to_dict("records")
    out["v1_suspect_minus_other_bonus"] = float(v1t[v1t.suspect].mean_delta.mean() - v1t[~v1t.suspect].mean_delta.mean())
    print(tb.round(2).to_string(index=False))

    # ---------------- figures
    # 12: additivity histogram, v1 vs v2 (same bins)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bins = np.linspace(-60, 60, 41)
    ax.hist(d2, bins=bins, color=C["ink2"], alpha=.45, label=f"v2 clean (n={len(d2)}, mean {d2.mean():+.1f})")
    ax.hist(d1, bins=bins, color=C["blue"], alpha=.55, label=f"v1 contaminated (n={len(d1)}, mean {d1.mean():+.1f})")
    ax.axvline(0, color=C["ink"], lw=1)
    ax.axvline(d1.mean(), color=C["blue"], lw=2); ax.axvline(d2.mean(), color=C["ink2"], lw=2, ls="--")
    ax.set_xlabel("merge minus mean of its parents  [percentile points within leaderboard]")
    ax.set_ylabel("merges"); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Merge additivity: is a merge the average of its parents?")
    fig.savefig(f"{figdir}/12_merge_additivity.png"); plt.close(fig)

    # per-task bonus
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    tb_ = tb.reset_index(drop=True)
    y = np.arange(len(tb_))[::-1]
    for yi, r in zip(y, tb_.itertuples()):
        col = C["orange"] if r.suspect else (C["blue"] if r.leaderboard == "v1" else C["ink2"])
        ax.errorbar(r.mean_delta, yi, xerr=[[r.mean_delta - r.ci_lo], [r.ci_hi - r.mean_delta]],
                    fmt="o", color=col, capsize=3)
    ax.axvline(0, color=C["ink"], lw=.8)
    ax.axhline(len(tb_[tb_.leaderboard == "v2"]) - .5, color=C["grid"], lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.leaderboard}  {r.task}" for r in tb_.itertuples()], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("merge minus mean of parents, per task [pct pts, 95% CI]")
    ax.set_title("Where do merges gain? (orange = tasks named as contaminated)")
    fig.savefig(f"{figdir}/P3_task_bonus.png"); plt.close(fig)

    # mid-parent regression, both leaderboards
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
    for ax, v in zip(axs, ["v1", "v2"]):
        mt, avg = tables[v], LB[v]["avg"]
        s = out[v]["strict_all_parents_scored"]
        ax.scatter(mt[f"{avg}|mid"], mt[f"{avg}|child"], s=8, alpha=.4, lw=0,
                   color=C["blue"] if v == "v1" else C["ink2"])
        xs = np.linspace(0, 100, 10)
        ax.plot(xs, s["intercept"] + s["midparent_slope"] * xs, color=C["ink"], lw=2,
                label=f"fit: slope {s['midparent_slope']:.2f}")
        ax.plot(xs, xs, color=C["ink2"], ls="--", lw=1, label="child = mid-parent")
        ax.set_title(f"{v}: n={len(mt)} merges", fontsize=10.5)
        ax.set_xlabel("mean of parents (pct)"); ax.legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("merge (pct)")
    fig.suptitle("Merges track the average of their parents", x=.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{figdir}/P3_midparent.png"); plt.close(fig)

    json.dump(out, open(f"{odir}/summary_phase3.json", "w"), indent=2, default=float)
    print(f"[done] phase 3 in {odir}/")


if __name__ == "__main__":
    main()
