# Reproducible Tests

Standalone, de-duplicated scripts ported from `Colab_Testing.ipynb` (my testing notebook) — the
same GPT-2-medium hallucination / activation-patching experiments, without
the Colab-only setup (Drive mounting, pip-installing into a persisted Drive
folder) and without the notebook's redefined-three-times helper functions.
Each script is independently runnable and writes its outputs to
`results/<script_name>/`.

## Setup

1. Install PyTorch yourself, matching your CUDA version (or CPU-only) —
   see https://pytorch.org/get-started/locally/. The scripts auto-detect
   CUDA and fall back to CPU if it's not available (CPU will be slow for
   the full head sweeps).
2. `pip install -r requirements.txt`
3. (Optional) No Hugging Face token is needed for the default run — GPT-2
   and the TruthfulQA dataset are public, and the misconception pairs are
   loaded from the local `misconception_pairs.json` in the project root.
   A token is only needed for `07_misconception_dataset_sweep.py --from-hf`,
   which pulls the private `Chukkk/TruthfulTransformer` HF dataset repo
   instead of the local file. If you need it, put `HF_TOKEN=...` in
   `pass.env` at the project root (already gitignored).

> **Note:** `Colab_Testing.ipynb` has a Hugging Face token hardcoded in one
> cell, already committed to git history. Treat that token as compromised
> and rotate it at huggingface.co/settings/tokens — it isn't reused by any
> script here.

## Datasets

Three prompt-pair files live in the project root. See
[`DATASETS.md`](https://github.com/G-cae78/Lancelot/blob/main/Reproducable%20Tests/DATASET_DESCRIPTION.md) for the full breakdown (sources, counts,
answer splits, gap stats and known caveats).

| File | Pairs | What's in it | Used by scripts? |
|---|---|---|---|
| `misconception_pairs.json` | 22,648 | Original unfiltered pool: TruthfulQA + CREAK Yes/No pairs, plus ~21k CounterFact entity pairs | Yes, 07–09 by default |
| `yesno_pairs.json` | 10,000 | Gap-filtered Yes/No pairs (FEVER, CREAK, TruthfulQA) with 10 rotating leading templates | Not yet |
| `entity_completion_pairs.json` | 845 | Gap-filtered open-entity completions, a non-Yes/No control | Not yet |

> **Heads-up:** `misconception_pairs.json` is ordered TruthfulQA → CREAK →
> CounterFact, so with `--limit` above ~1,233 scripts 07–09 start pulling in
> CounterFact pairs, which were rejected for this analysis. Keep the limit
> below that or filter on `source` first.

Dataset helper scripts:

| Script | What it does |
|---|---|
| `build_yesno_pairs.py` | Pulls the Yes/No pairs out of `misconception_pairs.json` and rewraps each corrupt prompt in one of 10 rotating leading templates → `yesno_pairs.json`. No model needed. |
| `expand_yesno_pairs_from_creak.py` | Adds more de-duplicated CREAK claims to `yesno_pairs.json` (`--target-count`, default 5000), with `stable: null` until gap-filtered. |

## Running

Run from inside this folder, in order (each step's output feeds context
for later ones, but every script also works standalone):

| Script | What it does |
|---|---|
| `01_model_sanity_check.py` | Loads GPT-2-medium, confirms hooks work. Run this first. |
| `02_truthfulqa_mc_baseline.py` | Baseline hallucination rate on TruthfulQA multiple-choice, scored by the model's own log-probs. |
| `03_single_pair_layer_patching.py` | Residual-stream patching across layers/positions on one clean/corrupt pair, then narrows to per-head patching at the strongest layer. |
| `04_single_pair_full_head_sweep.py` | Full layer x head patching sweep on the same pair; ranks heads, checks layer clustering. |
| `05_multi_pair_head_sweep.py` | Repeats the full sweep across 8 hand-written sycophancy pairs; checks whether the same heads matter across prompts (overlap, Spearman correlation). |
| `06_combo_head_patching.py` | Patches *sets* of heads jointly to test whether top heads act as a circuit (super-additive), independently (additive), or redundantly (sub-additive). |
| `07_misconception_dataset_sweep.py` | Scales the sweep up to the `misconception_pairs.json` set; gap-filters first, then aggregates which heads recur most often. |
| `08_group_vs_triad_test.py` | Tests a specific 3-head triad vs. whole-layer groupings for consistency across all pairs. |
| `09_ablation_necessity.py` | Ablates (zero- and mean-) the same head / layer to test necessity, not just sufficiency. |

```bash
cd "Reproducable Tests"
python 01_model_sanity_check.py
python 02_truthfulqa_mc_baseline.py --n-examples 50
python 03_single_pair_layer_patching.py
python 04_single_pair_full_head_sweep.py
python 05_multi_pair_head_sweep.py
python 06_combo_head_patching.py
python 07_misconception_dataset_sweep.py --limit 100
python 08_group_vs_triad_test.py --limit 100
python 09_ablation_necessity.py --limit 100
```

`--limit` (default 100) keeps the expensive per-head sweep (384 forward
passes per pair, since GPT-2-medium has 24 layers x 16 heads) from taking
hours on first run. Raise it once you know your hardware's budget, keeping
the CounterFact heads-up above in mind.
