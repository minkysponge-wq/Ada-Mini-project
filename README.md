# Is Benchmark Contamination Heritable?

A quantitative-genetics analysis of the Hugging Face LLM family tree (ADA Jackfruit project).

**Team:** Ullas T (PES1UG24AM309) · Sampreeth Jaladi (PES1UG24AM246) · Deepthi A E (PES1UG24AM358)

## Idea

Open-weight LLMs are built from each other through fine-tunes and merges, forming a family tree on the
Hugging Face Hub. We ask whether benchmark contamination is *inherited* along that tree.

- **Inflation proxy:** a model's percentile rank on Open LLM Leaderboard **v1** (older, likely
  contaminated benchmarks: ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K) minus its rank on
  **v2** (newer, harder benchmarks: IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-Pro).
- **Pedigree:** parent links from Hub `base_model` metadata and relation tags.
- **Model:** a pedigree linear mixed model (the "animal model") estimates the heritability h² of
  inflation versus genuine capability.

## Pipeline

| Script | What it does |
|---|---|
| `part1_pedigree.py` | Downloads v1/v2 scores and Hub lineage, builds the pedigree and the v1∩v2 panel, figures 01–07 |
| `part2_analysis.py` | Heritability (REML, two kinship models, permutation test), inheritance mode, lineage-only prediction, figures 08–16 |
| `phase1_audit.py` | Lineage coverage audit and h² sensitivity on well-documented subsets |
| `phase2_decouple.py` | Separates contamination from capability: residual proxy, capability covariate, genetic correlation, placebo control |
| `phase3_merges.py` | Merge additivity on each leaderboard, per-task merge bonus, fine-tune reference, v1-vs-v2 comparison adjusted for parent rank |
| `phase4_select.py` | Picks models for the memorisation test and records each one's proxies, including inflation predicted from its relatives only |
| `phase4_eval.py` | Runs models on original GSM8K questions vs GSM-Symbolic / GSM-Plus versions of the same questions (GPU) |
| `phase4_colab.ipynb` | Colab / Kaggle notebook that runs `phase4_eval.py` with vLLM and saves to Google Drive |
| `phase4_analyze.py` | Memorisation drop per model and its correlation with each inflation proxy |

## Run (Windows)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
set HF_TOKEN=hf_...            (a Hugging Face read token)
.venv\Scripts\python part1_pedigree.py --out out_phase1
.venv\Scripts\python part2_analysis.py --data out_phase1/data --out out_phase1
.venv\Scripts\python phase1_audit.py --data out_phase1/data --out out_phase1
.venv\Scripts\python phase2_decouple.py --data out_phase1/data --out out_phase1
.venv\Scripts\python phase3_merges.py --data out_phase1/data --out out_phase1
```

### Phase 4 (needs a GPU)

The memorisation test runs models, so it needs a GPU. Model selection is already in
`out_phase1/phase4/models.csv` (10 models up to 2B, 26 up to 4B, 106 up to 8B).

- **Colab (recommended):** open `phase4_colab.ipynb` in Colab, set *Runtime → T4 GPU*, add
  `HF_TOKEN` under Secrets, then *Run all*. Results go to Google Drive and the run resumes after a
  disconnect.
- **Local GPU (Windows, slower):**
  ```
  .venv\Scripts\python -m pip install -r requirements-eval.txt --extra-index-url https://download.pytorch.org/whl/cu121
  .venv\Scripts\python phase4_eval.py --tiers local --backend hf --batch 8
  .venv\Scripts\python phase4_analyze.py
  ```

Part 1 resumes from its cache in `<out>/data` and can take up to an hour on a fresh run because of
Hub rate limits. Parts 2 and the audit run offline in a few minutes.

## Outputs

- `out/` — first full run (baseline).
- `out_phase1/` — current results: renamed repos matched between v1 and v2 (panel 394 → 405) and
  lineage fixes. `phase1/` holds the coverage audit, `phase2/` the decoupling tests, `phase3/`
  the merge-additivity tests (including the figure `12_merge_additivity.png`).

## Key results so far (clonal kinship, the conservative model)

| Question | Result |
|---|---|
| Is inflation heritable? | h² = 0.52 [0.28, 0.72], permutation p = 0.015 |
| Is the lineage data good enough? | 99 of 405 panel models (24%) have documented parents; 127 have a relative in the panel (569 related pairs). Restricted to those, h² of inflation stays 0.52 while capability drops to 0.20. |
| Is it just capability? | Residual inflation is uncorrelated with capability (ρ = 0.02) and still heritable (h² = 0.62); genetic correlation with capability r_g = −0.14 [−0.38, 0.34] |
| Placebo: are ordinary skill gaps as heritable? | No. 0 of 20 clean-vs-clean averaged gaps reach the contamination gap's h² (placebo median 0.22) |
| Does family predict inflation? | CV R² 0.23 with the pedigree vs 0.17 covariates only; detector AUC 0.71 |
| Are merges additive? | Yes, roughly: a typical merge sits at its parents' average on both leaderboards (median difference +0.5 on v1, +0.2 on v2; slope 0.91 / 0.88) |
| Does merging compound contamination? | Weakly. At matched parent rank and without broken merges, merges keep +2.3 [0.6, 4.0] percentile points more of their parents' standing on contaminated v1 than on clean v2 (p = 0.008). The larger raw gap (+5.6) is mostly 23 broken merges that collapse on v2's generative tasks. |
| Is the merge gain concentrated on GSM8K / TruthfulQA? | No clear evidence: TruthfulQA +0.9 [−0.0, 1.7], GSM8K −0.3 [−1.7, 1.0] |

### Phase 4: memorisation test (pilot, 5 small models on a local RTX 4050)

Each model answered 1,400 paired items: original GSM8K questions and the same questions with new
numbers (GSM-Symbolic, GSM-Plus numerical substitution) or reworded (GSM-Plus paraphrase).
Models: Qwen2-0.5B, Qwen1.5-0.5B, TinyLlama-Cinder-Agent, Qwen2-1.5B, StableLM-2-1.6B (856 of 1,400 items).

| Question | Result |
|---|---|
| Does the harness measure what the leaderboard measures? | Yes: our original-question accuracy ranks the models like the leaderboard's GSM8K (Spearman 0.90) |
| Do models do worse when only the numbers change? | Yes, every model: 8–15 points lower on re-numbered questions (mean 7.8 points over symbolic and number-substitution); bootstrap CI excludes 0 for 4 of 5 |
| Is that just harder wording? | No: reworded questions with the original numbers cost only 1 point on average, so the drop is tied to the specific numbers, which is the memorisation pattern |
| Does the inflation proxy predict which models memorise more? | Not testable yet: with 5 models no correlation is significant (all p > 0.13). About 20+ models with relatives in the panel are needed; `phase4_select.py` lists 47 such models in the 7–8B tier for a Kaggle run |

Lineage note: on the v2 leaderboard, the "Base Model" of most merges is the model itself tagged
"(Merge)". Those are self-references, not parents, and are excluded.
