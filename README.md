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

## Run (Windows)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
set HF_TOKEN=hf_...            (a Hugging Face read token)
.venv\Scripts\python part1_pedigree.py --out out_phase1
.venv\Scripts\python part2_analysis.py --data out_phase1/data --out out_phase1
.venv\Scripts\python phase1_audit.py --data out_phase1/data --out out_phase1
```

Part 1 resumes from its cache in `<out>/data` and can take up to an hour on a fresh run because of
Hub rate limits. Parts 2 and the audit run offline in a few minutes.

## Outputs

- `out/` — first full run (baseline).
- `out_phase1/` — Phase 1 run: renamed repos matched between v1 and v2, and lineage fixes
  (`(Merge)` suffixes stripped from leaderboard parent ids, missing parents of cached models fetched).
