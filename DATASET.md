# Dataset

Built by `part1_pedigree.py` on 2026-11 from public sources; frozen in `out_phase1/data`
(SHA-256 hashes in `out_phase1/MANIFEST.json`).

## Sources

| Source | What we took |
|---|---|
| Open LLM Leaderboard v1 (`open-llm-leaderboard-old/results`, Hugging Face) | Scores on ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K |
| Open LLM Leaderboard v2 (`open-llm-leaderboard/contents`, Hugging Face) | Scores on IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-Pro, plus type, architecture, size, MoE flag |
| Hugging Face Hub API (`HfApi.model_info`) | Model card `base_model` field and relation tags (`base_model:finetune:`, `base_model:merge:`, …), creation date, downloads, likes, parameter count; also used to resolve renamed repositories |
| GSM-Symbolic (`apple/GSM-Symbolic`, main/test.jsonl) | 100 GSM8K templates with 50 re-numbered instances each (Phase 4) |
| GSM-Plus (`qintongli/GSM-Plus`, testmini) | 300 GSM8K questions with perturbed versions (numerical substitution, paraphrase, …) (Phase 4) |

## Files (`out_phase1/data`)

| File | Rows | Contents |
|---|---|---|
| `v1_scores.csv` | one per v1 model | Leaderboard v1 task scores |
| `v1_parsed.jsonl` | one per v1 result file | Parsed raw v1 result files (cache for `v1_scores.csv`) |
| `v2_scores.csv` | one per v2 model | Leaderboard v2 task scores and metadata |
| `hub_meta.jsonl` | one per model looked up | Raw Hub metadata (base models, tags, dates, size) |
| `id_aliases.csv` | 1 per rename | Old repo id → new repo id, verified through the Hub API |
| `nodes.csv` | 13,148 | Every model in the family tree: generation, founder, number of parents, merge flag, scores |
| `edges.csv` | 11,541 | Parent → child links with relation type (merge 8,535, fine-tune 2,679, adapter 192, quantized 135) |
| `panel.csv` | 405 | Analysis panel: models scored on both leaderboards and on the Hub, with v1/v2 percentile ranks and the inflation proxy (`inflation = v1_pct − v2_pct`, `infl_math`, `infl_knowledge`) |

## Phase 4 files (`out_phase1/phase4`)

| File | Contents |
|---|---|
| `models.csv` | 142 candidate models by GPU tier, with their inflation proxies and pedigree-predicted inflation |
| `preds/<model>.jsonl` | One line per question: item id, dataset, variant (original / symbolic / numerical substitution / problem understanding), gold answer, model answer, correct, raw output |
| `memorisation_by_model.csv` | Accuracy on originals vs variants and the paired drop with bootstrap 95% CIs |
