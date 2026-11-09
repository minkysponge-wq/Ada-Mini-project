#!/usr/bin/env python3
"""
ADA Jackfruit — Part 1: Pedigree reconstruction + v1/v2 linkage.

Is Benchmark Contamination Heritable? (Ullas T, Sampreeth Jaladi, Deepthi A E)

Outputs (in --out, default ./out):
  data/v1_scores.csv       per-model Open LLM Leaderboard v1 scores
  data/v2_scores.csv       per-model Open LLM Leaderboard v2 scores
  data/hub_meta.jsonl      cached Hub metadata (resumable)
  data/nodes.csv           pedigree nodes (+ generation, founder, scores)
  data/edges.csv           parent -> child edges with relation type
  data/panel.csv           models scored on BOTH v1 and v2 (+ inflation proxy)
  figs/*.png               result plots for slides

Usage:
  pip install -r requirements.txt
  python part1_pedigree.py                  # real run (needs internet + HF)
  python part1_pedigree.py --mock           # synthetic smoke test, no network
  HF_TOKEN=hf_xxx python part1_pedigree.py  # token raises API rate limits
"""
import argparse, json, os, re, sys, time, glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_REPO = "open-llm-leaderboard/contents"
V1_RESULTS_REPO = "open-llm-leaderboard-old/results"

V1_TASKS = ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"]
V2_TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]

# palette (validated categorical slots 1-4 + ink) — light surface
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", surface="#fcfcfb")
REL_COLOR = {"finetune": C["blue"], "merge": C["orange"],
             "adapter": C["aqua"], "quantized": C["yellow"]}


def norm_id(s):
    """Canonical repo id. The v2 leaderboard 'Base Model' column appends display tags
    such as '(Merge)'; Hub repo ids cannot contain parentheses, so a trailing '(...)'
    is always a label, never part of the id."""
    if not isinstance(s, str):
        return None
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s.strip())
    return s.strip().strip("/").replace(" ", "")


# --------------------------------------------------------------------------- v2
def fetch_v2():
    from datasets import load_dataset
    df = load_dataset(V2_REPO, split="train").to_pandas()

    def col(*cands):
        for c in cands:
            for k in df.columns:
                if k.lower().replace("⬆️", "").strip() == c.lower():
                    return k
        return None

    out = pd.DataFrame({"model_id": df[col("fullname", "eval_name")].map(norm_id)})
    for t in V2_TASKS:
        k = col(t)
        out[t] = pd.to_numeric(df[k], errors="coerce") if k else np.nan
    out["v2_avg"] = pd.to_numeric(df[col("Average")], errors="coerce")
    extra = {"precision": ("Precision",), "type": ("Type",),
             "architecture": ("Architecture",), "params_b": ("#Params (B)",),
             "lb_base_model": ("Base Model",), "lb_merged": ("Merged",),
             "lb_generation": ("Generation",), "flagged": ("Flagged",),
             "moe": ("MoE",), "hub_license": ("Hub License",),
             "upload_date": ("Upload To Hub Date",), "chat_template": ("Chat Template",),
             "official": ("Official Providers",)}
    for name, cands in extra.items():
        k = col(*cands)
        out[name] = df[k].values if k else np.nan
    return dedupe(out, "v2_avg")


# --------------------------------------------------------------------------- v1
def _v1_parse_results_json(d):
    name = (d.get("config_general") or {}).get("model_name") or \
           (d.get("config") or {}).get("model_name")
    prec = (d.get("config_general") or {}).get("model_dtype") or ""
    res = d.get("results", {})
    row, mmlu = {}, []
    for k, v in res.items():
        if not isinstance(v, dict):
            continue
        if "arc:challenge" in k:
            row["ARC"] = v.get("acc_norm", v.get("acc"))
        elif "hellaswag" in k:
            row["HellaSwag"] = v.get("acc_norm", v.get("acc"))
        elif "hendrycksTest" in k or "|mmlu:" in k:
            if v.get("acc") is not None:
                mmlu.append(v["acc"])
        elif "truthfulqa" in k:
            row["TruthfulQA"] = v.get("mc2")
        elif "winogrande" in k:
            row["Winogrande"] = v.get("acc")
        elif "gsm8k" in k:
            row["GSM8K"] = v.get("acc")
    if mmlu:
        row["MMLU"] = float(np.mean(mmlu))
    return name, prec, row


def fetch_v1(cache_dir, workers=16):
    """Read every results_*.json in the archived v1 repo straight into memory
    (filenames contain ':' which Windows can't write to disk), parse, and cache the
    parsed rows in v1_parsed.jsonl so an interrupted run resumes.
    A model may have several runs (GSM8K was added later) → latest non-null per task."""
    # Direct /resolve/ GETs = "resolver" quota (5000 / 5 min, free user).
    # HfFileSystem would also hit the tree API per file (1000 / 5 min) → 429 storm.
    import requests
    from huggingface_hub import HfApi, hf_hub_url
    tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    sess = requests.Session()
    if tok:
        sess.headers["Authorization"] = f"Bearer {tok}"
    files = [f for f in api.list_repo_files(V1_RESULTS_REPO, repo_type="dataset")
             if re.search(r"(^|/)results_[^/]*\.json$", f)]
    cache = os.path.join(cache_dir, "v1_parsed.jsonl")
    done = {}
    if os.path.exists(cache):
        for line in open(cache, encoding="utf-8"):
            r = json.loads(line); done[r["file"]] = r
    todo = [f for f in files if f not in done]
    print(f"[v1] {len(files)} result files, {len(todo)} left to fetch")

    def fetch(f):
        url = hf_hub_url(V1_RESULTS_REPO, f, repo_type="dataset")
        for attempt in range(8):
            try:
                r = sess.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(5); continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(min(300, 15 * 2 ** attempt)); continue
            r.raise_for_status()
        raise RuntimeError("retries exhausted")

    def get(f):
        try:
            d = fetch(f)
            name, prec, row = _v1_parse_results_json(d)
            return {"file": f, "name": name or "/".join(f.split("/")[:2]),
                    "prec": prec, "row": row}
        except Exception as e:
            return {"file": f, "err": type(e).__name__}

    with open(cache, "a", encoding="utf-8") as fout, ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(get, todo)):
            if "err" not in r:
                fout.write(json.dumps(r) + "\n"); done[r["file"]] = r
            if i % 500 == 0:
                fout.flush(); print(f"[v1] {i}/{len(todo)}")

    agg = defaultdict(dict)
    for f in sorted(done, key=lambda f: os.path.basename(f)):  # timestamp order
        r = done[f]
        # merge runs per model (precision label is missing on some later runs)
        key = norm_id(r["name"])
        if r.get("prec"):
            agg[key]["_prec"] = r["prec"]
        for t, v in r["row"].items():
            if v is not None:
                agg[key][t] = v
    rows = [{"model_id": k, "precision": v.get("_prec", ""), **{t: v.get(t) for t in V1_TASKS}}
            for k, v in agg.items()]
    df = pd.DataFrame(rows)
    for t in V1_TASKS:
        df[t] = pd.to_numeric(df[t], errors="coerce") * 100
    df = df.dropna(subset=V1_TASKS)          # require the full 6-task v1 average
    df["v1_avg"] = df[V1_TASKS].mean(axis=1)
    return dedupe(df, "v1_avg")


def dedupe(df, score):
    """One row per model: prefer bf16/fp16 runs, then the highest score."""
    pr = df.get("precision", pd.Series("", index=df.index)).astype(str).str.lower()
    df = df.assign(_p=(~pr.str.contains("bfloat16|float16|bf16|fp16")).astype(int))
    df = df.sort_values(["model_id", "_p", score], ascending=[True, True, False])
    return df.drop_duplicates("model_id").drop(columns="_p").reset_index(drop=True)


# --------------------------------------------------------------------------- Hub
REL_TAG = re.compile(r"^base_model:(finetune|merge|adapter|quantized):(.+)$")


def _hub_one(api, mid):
    """429 (rate limit) / network errors → back off and retry; never cached as 'missing'.
    Only a real 404/401/gated answer is recorded as ok=False."""
    for attempt in range(8):
        try:
            info = api.model_info(mid, expand=["cardData", "tags", "createdAt",
                                               "safetensors", "downloads", "likes"])
            break
        except Exception as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403, 404) or type(e).__name__ in (
                    "RepositoryNotFoundError", "GatedRepoError"):
                return {"model_id": mid, "ok": False, "err": type(e).__name__}
            time.sleep(min(300, 20 * 2 ** attempt) if code == 429 else 5)
    else:
        return {"model_id": mid, "retry": True}
    card = info.card_data.to_dict() if info.card_data else {}
    bm = card.get("base_model")
    bm = [bm] if isinstance(bm, str) else (bm or [])
    rels = defaultdict(list)
    for t in info.tags or []:
        m = REL_TAG.match(t)
        if m:
            rels[m.group(1)].append(m.group(2))
    n_params = None
    if info.safetensors is not None:
        n_params = getattr(info.safetensors, "total", None)
    return {"model_id": mid, "ok": True,
            "base_models": [norm_id(b) for b in bm if isinstance(b, str)],
            "rel_tags": dict(rels),
            "created_at": str(info.created_at) if info.created_at else None,
            "n_params": n_params, "downloads": info.downloads, "likes": info.likes,
            "tags": [t for t in (info.tags or []) if ":" not in t][:30]}


def fetch_hub_meta(seed_ids, cache_path, max_depth=8, workers=16):
    """BFS up the lineage: fetch metadata for seed models, then for any ancestor
    not yet seen (ancestors off the leaderboard are still needed for kinship)."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    meta = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line); meta[r["model_id"]] = r
    # resume fix: parents of models already in the cache must be fetched too,
    # otherwise an interrupted run leaves those branches permanently cut off
    pending = {p for r in meta.values() for p in parents_of(r) if p not in meta}
    frontier = [m for m in set(seed_ids) | pending if m and m not in meta]
    depth = 0
    if pending:
        print(f"[hub] resume: {len(pending)} parents of cached models not yet fetched")
    with open(cache_path, "a", encoding="utf-8") as fout:
        while frontier and depth <= max_depth:
            print(f"[hub] depth {depth}: fetching {len(frontier)} models")
            with ThreadPoolExecutor(workers) as ex:
                futs = {ex.submit(_hub_one, api, m): m for m in frontier}
                for i, f in enumerate(as_completed(futs)):
                    r = f.result()
                    if r.get("retry"):
                        continue  # not cached → refetched on the next run
                    meta[r["model_id"]] = r
                    fout.write(json.dumps(r) + "\n")
                    if i % 250 == 0:
                        fout.flush(); print(f"[hub]   {i}/{len(frontier)}", flush=True)
            nxt = set()
            for m in frontier:
                for p in parents_of(meta.get(m, {})):
                    if p not in meta:
                        nxt.add(p)
            frontier, depth = list(nxt), depth + 1
    return meta


def parents_of(r):
    """Returns {parent_id: relation}. Prefers explicit relation tags; falls back to card."""
    out = {}
    for rel, ps in (r.get("rel_tags") or {}).items():
        for p in ps:
            out[norm_id(p)] = rel
    for p in r.get("base_models") or []:
        p = norm_id(p)
        if not p:
            continue
        out.setdefault(p,"merge" if len(r.get("base_models")) > 1 else "finetune")
    out.pop(r.get("model_id"), None)
    return out


def resolve_renames(v1, v2, cache_path):
    """Repos renamed between v1 and v2 (org moves such as psmathur/* -> pankajmathur/*,
    legacy ids such as gpt2-xl -> openai-community/gpt2-xl) appear as two different
    models. Candidates = v1-only and v2-only ids sharing a repo name; a pair is accepted
    ONLY if the Hub redirects the v1 id to the v2 id (name matches alone are unsafe:
    MaziyarPanahi/Mistral-7B-Instruct-v0.3 is not mistralai/Mistral-7B-Instruct-v0.3).
    Returns {old_v1_id: canonical_v2_id}; cached in id_aliases.csv."""
    if os.path.exists(cache_path):
        c = pd.read_csv(cache_path)
        return dict(zip(c.v1_id[c.accepted], c.v2_id[c.accepted]))
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    a, b = set(v1.model_id), set(v2.model_id)
    by_name = defaultdict(list)
    for x in a - b:
        by_name[x.split("/")[-1].lower()].append(x)
    pairs = [(old, new) for new in sorted(b - a) for old in by_name.get(new.split("/")[-1].lower(), [])]
    rows = []
    for old, new in pairs:
        try:
            canon = api.model_info(old).id
        except Exception as e:
            canon = f"<{type(e).__name__}>"
        rows.append({"v1_id": old, "v2_id": new, "hub_canonical": canon, "accepted": canon == new})
        time.sleep(0.3)
    c = pd.DataFrame(rows, columns=["v1_id", "v2_id", "hub_canonical", "accepted"])
    c.to_csv(cache_path, index=False)
    print(f"[alias] {len(c)} rename candidates, {int(c.accepted.sum())} confirmed by Hub redirect")
    return dict(zip(c.v1_id[c.accepted], c.v2_id[c.accepted]))


# --------------------------------------------------------------------------- pedigree
def build_pedigree(meta, v1, v2):
    G = nx.DiGraph()
    for mid, r in meta.items():
        G.add_node(mid)
        for p, rel in parents_of(r).items():
            G.add_edge(p, mid, relation=rel)
    # leaderboard fallback for models whose Hub page is gone / has no card
    lb = v2.set_index("model_id")
    for mid in v2.model_id:
        if G.in_degree(mid) if mid in G else 0:
            continue
        bm = norm_id(lb.at[mid, "lb_base_model"]) if "lb_base_model" in lb else None
        if bm and bm != mid and bm.lower() not in ("", "nan", "removed"):
            rel = "merge" if str(lb.at[mid, "lb_merged"]).lower() == "true" else "finetune"
            G.add_edge(bm, mid, relation=rel)
    for mid in set(v1.model_id) | set(v2.model_id):
        G.add_node(mid)
    # break cycles (mislabelled cards) by dropping the edge closing each cycle
    n_cut = 0
    while True:
        try:
            cyc = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break
        G.remove_edge(*cyc[-1][:2]); n_cut += 1
    print(f"[pedigree] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{n_cut} cycle edges removed")

    # quantized copies inherit scores trivially → exclude them as edges for kinship,
    # but keep the info. Generation = longest path from any founder.
    gen, founder = {}, {}
    for n in nx.topological_sort(G):
        ps = [p for p in G.predecessors(n) if G.edges[p, n]["relation"] != "quantized"]
        if not ps:
            gen[n], founder[n] = 0, n
        else:
            gen[n] = 1 + max(gen[p] for p in ps)
            # primary founder = founder of the deepest parent (ties: alphabetical)
            founder[n] = founder[sorted(ps, key=lambda p: (-gen[p], p))[0]]

    edges = pd.DataFrame([(u, v, d["relation"]) for u, v, d in G.edges(data=True)],
                         columns=["parent", "child", "relation"])
    nodes = pd.DataFrame({"model_id": list(G.nodes)})
    nodes["generation"] = nodes.model_id.map(gen)
    nodes["founder"] = nodes.model_id.map(founder)
    nodes["n_parents"] = nodes.model_id.map(lambda n: sum(
        1 for p in G.predecessors(n) if G.edges[p, n]["relation"] != "quantized"))
    nodes["is_merge"] = nodes.model_id.map(lambda n: any(
        G.edges[p, n]["relation"] == "merge" for p in G.predecessors(n))) | (nodes.n_parents > 1)
    m = pd.DataFrame(meta.values()).set_index("model_id") if meta else pd.DataFrame()
    for c in ["created_at", "n_params", "downloads", "likes", "ok"]:
        nodes[c] = nodes.model_id.map(m[c]) if c in m else np.nan
    nodes = nodes.merge(v1[["model_id", "v1_avg"] + V1_TASKS], on="model_id", how="left") \
                 .merge(v2[["model_id", "v2_avg"] + V2_TASKS], on="model_id", how="left",
                        suffixes=("", "_v2"))
    return G, nodes, edges


def make_panel(nodes, v2):
    """Inflation proxy: percentile rank on v1 minus percentile rank on v2, computed
    within the overlap set so both ranks refer to the same population."""
    p = nodes.dropna(subset=["v1_avg", "v2_avg"]).copy()
    p["v1_pct"] = p.v1_avg.rank(pct=True) * 100
    p["v2_pct"] = p.v2_avg.rank(pct=True) * 100
    p["inflation"] = p.v1_pct - p.v2_pct
    # benchmark-specific contrasts: contaminated v1 task vs closest fresh v2 task
    pairs = {"infl_math": ("GSM8K", "MATH Lvl 5"), "infl_knowledge": ("MMLU", "MMLU-PRO")}
    for name, (a, b) in pairs.items():
        if a in p and b in p:
            p[name] = p[a].rank(pct=True) * 100 - p[b].rank(pct=True) * 100
    keep = ["model_id", "type", "architecture", "params_b", "flagged", "moe", "official"]
    p = p.merge(v2[[c for c in keep if c in v2]], on="model_id", how="left")
    return p


# --------------------------------------------------------------------------- plots
def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False})


def plots(nodes, edges, panel, figdir):
    style(); os.makedirs(figdir, exist_ok=True)
    on_lb = nodes[nodes.v1_avg.notna() | nodes.v2_avg.notna()]

    # 1. coverage funnel
    n = {"v1 scored": nodes.v1_avg.notna().sum(), "v2 scored": nodes.v2_avg.notna().sum(),
         "Both (panel)": len(panel),
         "Panel with ≥1 parent": (panel.model_id.isin(edges.child)).sum()}
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.barh(list(n)[::-1], list(n.values())[::-1], color=C["blue"], height=0.55)
    for i, v in enumerate(list(n.values())[::-1]):
        ax.text(v, i, f"  {v:,}", va="center", color=C["ink"])
    ax.set_title("Dataset coverage"); ax.set_xlabel("models"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(n.values()) * 1.15)
    fig.savefig(f"{figdir}/01_coverage.png"); plt.close(fig)

    # 2. edge relation mix
    rc = edges.relation.value_counts()
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.barh(rc.index[::-1], rc.values[::-1], height=0.55,
            color=[REL_COLOR.get(r, C["ink2"]) for r in rc.index[::-1]])
    for i, v in enumerate(rc.values[::-1]):
        ax.text(v, i, f"  {v:,}", va="center")
    ax.set_title("Lineage edges by relation type"); ax.set_xlabel("edges")
    ax.grid(axis="y", visible=False); ax.set_xlim(0, rc.max() * 1.15)
    fig.savefig(f"{figdir}/02_relations.png"); plt.close(fig)

    # 3. generation depth of leaderboard models
    g = on_lb.generation.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.bar(g.index, g.values, color=C["blue"], width=0.7)
    ax.set_title("Generation depth of leaderboard models")
    ax.set_xlabel("generation (0 = founder / base model)"); ax.set_ylabel("models")
    ax.set_xticks(g.index); ax.grid(axis="x", visible=False)
    fig.savefig(f"{figdir}/03_generations.png"); plt.close(fig)

    # 4. largest families
    fam = on_lb.founder.value_counts().head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.barh([f.split("/")[-1][:34] for f in fam.index], fam.values, color=C["blue"], height=0.6)
    ax.set_title("Largest model families (by founder)"); ax.set_xlabel("leaderboard descendants")
    ax.grid(axis="y", visible=False)
    fig.savefig(f"{figdir}/04_families.png"); plt.close(fig)

    # 5. v1 vs v2 percentile rank — the inflation proxy
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    mg = panel.is_merge.fillna(False).astype(bool)
    for mask, lab, col in [(~mg, "fine-tune / base", C["blue"]), (mg, "merge", C["orange"])]:
        ax.scatter(panel.v2_pct[mask], panel.v1_pct[mask], s=10, alpha=0.55,
                   color=col, label=f"{lab} (n={mask.sum():,})", linewidths=0)
    ax.plot([0, 100], [0, 100], color=C["ink2"], lw=1, ls="--")
    ax.text(4, 94, "above line ⇒ ranks better on\ncontaminated v1 (inflated)",
            fontsize=8.5, color=C["ink2"], va="top")
    r = panel[["v1_pct", "v2_pct"]].corr(method="spearman").iloc[0, 1]
    ax.set_title(f"v1 vs v2 rank (Spearman ρ = {r:.2f})")
    ax.set_xlabel("v2 percentile (fresh benchmarks)"); ax.set_ylabel("v1 percentile (contaminated)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.legend(loc="lower right", markerscale=2.5)
    fig.savefig(f"{figdir}/05_rank_scatter.png"); plt.close(fig)

    # 6. inflation distribution: merge vs non-merge
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bins = np.linspace(-80, 80, 41)
    for mask, lab, col in [(~mg, "fine-tune / base", C["blue"]), (mg, "merge", C["orange"])]:
        x = panel.inflation[mask]
        ax.hist(x, bins=bins, density=True, alpha=0.55, color=col,
                label=f"{lab}: median {x.median():+.1f}")
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.set_title("Inflation proxy (v1 pct − v2 pct)"); ax.set_xlabel("percentile points")
    ax.set_ylabel("density"); ax.legend()
    fig.savefig(f"{figdir}/06_inflation_hist.png"); plt.close(fig)

    # 7. parent → child inflation (first look at heritability)
    pi = panel.set_index("model_id").inflation
    e = edges[edges.relation != "quantized"]
    e = e[e.parent.isin(pi.index) & e.child.isin(pi.index)].copy()
    if len(e) > 5:
        # mid-parent value for merges
        mp = e.groupby("child").parent.apply(lambda ps: pi.loc[list(ps)].mean())
        d = pd.DataFrame({"mid": mp, "child": pi.loc[mp.index]})
        slope, icpt = np.polyfit(d.mid, d.child, 1)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.scatter(d.mid, d.child, s=12, alpha=0.6, color=C["blue"], linewidths=0)
        xs = np.linspace(d.mid.min(), d.mid.max(), 50)
        ax.plot(xs, slope * xs + icpt, color=C["orange"], lw=2,
                label=f"OLS slope = {slope:.2f}  (n={len(d):,} pairs)")
        ax.axhline(0, color=C["grid"]); ax.axvline(0, color=C["grid"])
        ax.set_title("Parent → child inflation")
        ax.set_xlabel("(mid-)parent inflation"); ax.set_ylabel("child inflation")
        ax.legend(loc="upper left")
        fig.savefig(f"{figdir}/07_parent_child.png"); plt.close(fig)
        print(f"[plots] parent-offspring regression slope = {slope:.3f} on {len(d)} pairs "
              f"(naive h² proxy; Part 2 does this properly with the animal model)")


# --------------------------------------------------------------------------- mock
def mock_data(n_founders=40, n_models=4000, seed=0):
    """Synthetic world with a known heritable inflation component — tests the pipeline."""
    rng = np.random.default_rng(seed)
    ids, par, rel, cap, infl = [], [], [], [], []
    for i in range(n_founders):
        ids.append(f"base-org/founder-{i}"); par.append([]); rel.append(None)
        cap.append(rng.normal(0, 1)); infl.append(rng.normal(0, 0.3))
    for i in range(n_models - n_founders):
        k = len(ids)
        if rng.random() < 0.22 and k > 2:
            ps = list(rng.choice(k, 2, replace=False)); r = "merge"
        else:
            ps = [int(rng.integers(0, k))]; r = "finetune" if rng.random() > 0.05 else "adapter"
        ids.append(f"user{rng.integers(0, 600)}/m{i}")
        par.append([ids[p] for p in ps]); rel.append(r)
        cap.append(np.mean([cap[p] for p in ps]) + rng.normal(0, 0.3))
        # inflation: inherited mean of parents + own contamination; merges add a bit
        infl.append(np.mean([infl[p] for p in ps]) + rng.normal(0.05, 0.35)
                    + (0.15 if r == "merge" else 0))
    cap, infl = np.array(cap), np.array(infl)
    meta = {}
    for m, ps, r in zip(ids, par, rel):
        meta[m] = {"model_id": m, "ok": True, "base_models": ps,
                   "rel_tags": {r: ps} if ps else {}, "created_at": None,
                   "n_params": None, "downloads": 0, "likes": 0}
    scored = rng.random(len(ids)) < 0.7
    v1s = 50 + 8 * cap + 6 * infl + rng.normal(0, 1.5, len(ids))
    v2s = 20 + 7 * cap + rng.normal(0, 1.5, len(ids))
    in_v1 = scored & (rng.random(len(ids)) < 0.75)
    in_v2 = scored & (rng.random(len(ids)) < 0.75)
    v1 = pd.DataFrame({"model_id": np.array(ids)[in_v1], "v1_avg": v1s[in_v1]})
    for t in V1_TASKS: v1[t] = v1.v1_avg + rng.normal(0, 3, len(v1))
    v2 = pd.DataFrame({"model_id": np.array(ids)[in_v2], "v2_avg": v2s[in_v2]})
    for t in V2_TASKS: v2[t] = v2.v2_avg + rng.normal(0, 3, len(v2))
    for c in ["type", "architecture", "params_b", "lb_base_model", "lb_merged",
              "flagged", "moe", "official"]:
        v2[c] = np.nan
    return meta, v1, v2


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-alias", action="store_true",
                    help="skip matching repos renamed between v1 and v2")
    a = ap.parse_args()
    ddir, fdir = os.path.join(a.out, "data"), os.path.join(a.out, "figs")
    os.makedirs(ddir, exist_ok=True)

    if a.mock:
        meta, v1, v2 = mock_data()
    else:
        p1, p2 = f"{ddir}/v1_scores.csv", f"{ddir}/v2_scores.csv"
        v2 = pd.read_csv(p2) if os.path.exists(p2) else fetch_v2()
        v2.to_csv(p2, index=False); print(f"[v2] {len(v2):,} models")
        v1 = pd.read_csv(p1) if os.path.exists(p1) else fetch_v1(ddir, a.workers)
        v1.to_csv(p1, index=False); print(f"[v1] {len(v1):,} models")
        if not a.no_alias:
            alias = resolve_renames(v1, v2, f"{ddir}/id_aliases.csv")
            if alias:
                v1 = dedupe(v1.assign(model_id=v1.model_id.replace(alias)), "v1_avg")
                print(f"[alias] {len(alias)} v1 ids relabelled to their current Hub id")
        lb_parents = set(v2.lb_base_model.map(norm_id).dropna()) if "lb_base_model" in v2 else set()
        lb_parents -= {"", "nan", "removed"}
        meta = fetch_hub_meta(set(v1.model_id) | set(v2.model_id) | lb_parents,
                              f"{ddir}/hub_meta.jsonl", a.max_depth, a.workers)

    G, nodes, edges = build_pedigree(meta, v1, v2)
    panel = make_panel(nodes, v2)
    panel = panel.merge(nodes[["model_id", "is_merge"]], on="model_id", how="left") \
        if "is_merge" not in panel else panel
    nodes.to_csv(f"{ddir}/nodes.csv", index=False)
    edges.to_csv(f"{ddir}/edges.csv", index=False)
    panel.to_csv(f"{ddir}/panel.csv", index=False)

    s = {"nodes": len(nodes), "edges": len(edges), "panel": len(panel),
         "panel_merges": int(panel.is_merge.sum()),
         "max_generation": int(nodes.generation.max()),
         "families_in_panel": int(panel.founder.nunique()),
         "spearman_v1_v2": float(panel[["v1_pct", "v2_pct"]].corr("spearman").iloc[0, 1])}
    json.dump(s, open(f"{a.out}/summary_part1.json", "w"), indent=2)
    print(json.dumps(s, indent=2))
    plots(nodes, edges, panel, fdir)
    print(f"[done] figures in {fdir}/")


if __name__ == "__main__":
    main()
