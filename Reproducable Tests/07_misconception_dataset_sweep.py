"""
Scale the per-head patching sweep from a handful of hand-written pairs up to
the full misconception_pairs.json set (clean/corrupt pairs generated from
TruthfulQA's boolean questions).

Steps:
1. Load pairs (local misconception_pairs.json by default; --from-hf pulls
   the private Chukkk/TruthfulTransformer HF dataset instead -- requires
   HF_TOKEN in pass.env).
2. Cheap gap pre-filter (2 forward passes/pair) -- only pairs with a
   stable, correctly-directed gap go into the expensive 144-forward-pass
   sweep.
3. Run the full sweep on the surviving (optionally --limit'd) pairs.
4. Aggregate: which heads show up in the local top-5 most often across
   pairs, and how much that overlaps with the idx5 pair (script 04) and
   the hand-written pairs (script 05).

Usage:
    python "07_misconception_dataset_sweep.py" [--limit 100] [--min-gap 0.3] [--from-hf] [--write-stable]
"""

import argparse
import contextlib
import io
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    HAND_WRITTEN_PAIRS,
    PROJECT_ROOT,
    ensure_results_dir,
    filter_stable_pairs,
    get_device,
    load_local_misconception_pairs,
    load_model,
    maybe_hf_login,
    run_head_patch_sweep,
    to_pair_records,
)


def load_pairs(from_hf: bool) -> list[dict]:
    if from_hf:
        maybe_hf_login()
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id="Chukkk/TruthfulTransformer", filename="misconception_pairs.json", repo_type="dataset")
        with open(path) as f:
            return json.load(f)

    return load_local_misconception_pairs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Max number of pairs to consider (the local dataset has thousands)")
    parser.add_argument("--min-gap", type=float, default=0.3, help="Minimum |clean-corrupt| logit-diff gap to keep a pair")
    parser.add_argument("--from-hf", action="store_true", help="Pull misconception_pairs.json from the private HF dataset instead of the local file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K heads per pair used for the frequency aggregation")
    parser.add_argument("--write-stable", action="store_true", help="Write the stable subset to results/.../stable_pairs.json")
    args = parser.parse_args()

    model = load_model(device=get_device())
    out_dir = ensure_results_dir("07_misconception_dataset_sweep")

    raw_pairs = load_pairs(args.from_hf)
    if args.limit is not None:
        raw_pairs = raw_pairs[: args.limit]
    print(f"Loaded {len(raw_pairs)} pairs (limit={args.limit})")

    records = to_pair_records(raw_pairs)
    stable_pairs, unstable_pairs, skipped_pairs = filter_stable_pairs(model, records, min_gap=args.min_gap)
    print(f"{len(skipped_pairs)}/{len(records)} pairs skipped (e.g. a multi-token correct/predicted word)")
    print(f"{len(stable_pairs)}/{len(records)} pairs clear gap >= {args.min_gap} in the correct direction -- sweeping only these")

    unstable_pairs_sorted = sorted(unstable_pairs, key=lambda p: p["gap"])
    print(f"\n{len(unstable_pairs_sorted)} pairs did NOT clear the gap filter:")
    for p in unstable_pairs_sorted[:20]:
        direction = "backfired" if p["gap"] < 0 else "too small"
        print(f"  gap={p['gap']:+.3f} ({direction}) [{p.get('source', 'n/a')}] {p['clean_prompt'].splitlines()[0]!r}")
    if len(unstable_pairs_sorted) > 20:
        print(f"  ... and {len(unstable_pairs_sorted) - 20} more")

    if args.write_stable:
        stable_out = out_dir / "stable_pairs.json"
        with open(stable_out, "w") as f:
            json.dump(stable_pairs, f, indent=2)
        print(f"Wrote {len(stable_pairs)} stable pairs to {stable_out}")

    # --- Full sweep on stable pairs ---
    misconception_results = {}
    misconception_gaps = {}
    sweep_skipped = []

    start = time.time()
    for i, p in enumerate(stable_pairs):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                recovery, raw_effect, clean_diff, corrupt_diff = run_head_patch_sweep(
                    model, p["clean_prompt"], p["corrupt_prompt"], p["correct_word"], p["predicted_word"]
                )
        except Exception as e:
            sweep_skipped.append((p, str(e)))
            continue
        misconception_results[p["name"]] = recovery
        misconception_gaps[p["name"]] = clean_diff - corrupt_diff
        if (i + 1) % 10 == 0 or i == len(stable_pairs) - 1:
            elapsed = time.time() - start
            print(f"{i + 1}/{len(stable_pairs)} pairs done ({elapsed:.0f}s elapsed, {len(sweep_skipped)} skipped)")

    if not misconception_results:
        print("No pairs survived the sweep -- nothing to aggregate. Try a higher --limit or lower --min-gap.")
        return

    np.savez(out_dir / "misconception_results.npz", **misconception_results)

    # --- Aggregate: top-K head frequency across pairs ---
    stable_names = list(misconception_results.keys())
    stable_recoveries = np.stack([misconception_results[n] for n in stable_names])
    mean_recovery = stable_recoveries.mean(axis=0)
    n_layers, n_heads = mean_recovery.shape

    top_k_counts = np.zeros((n_layers, n_heads), dtype=int)
    for n in stable_names:
        rec = misconception_results[n]
        flat_idx = np.argsort(-np.abs(rec).ravel())[: args.top_k]
        for idx in flat_idx:
            l, h = divmod(idx, n_heads)
            top_k_counts[l, h] += 1

    freq_df = pd.DataFrame(
        [
            {"layer": l, "head": h, "mean_recovery": mean_recovery[l, h], f"top{args.top_k}_frequency": top_k_counts[l, h] / len(stable_names)}
            for l in range(n_layers)
            for h in range(n_heads)
        ]
    ).sort_values(f"top{args.top_k}_frequency", ascending=False).reset_index(drop=True)

    print(f"\nTop heads by top-{args.top_k} frequency across {len(stable_names)} misconception pairs:")
    print(freq_df.head(args.top_k).to_string(index=False))
    freq_df.to_csv(out_dir / "head_frequency.csv", index=False)

    misconception_top_heads = set(zip(freq_df.head(args.top_k)["layer"], freq_df.head(args.top_k)["head"]))

    # Overlap vs idx5 (script 04's output, if present)
    idx5_ranking_path = PROJECT_ROOT / "Reproducable Tests" / "results" / "04_single_pair_full_head_sweep" / "head_ranking_idx5.csv"
    if idx5_ranking_path.exists():
        idx5_ranking = pd.read_csv(idx5_ranking_path)
        idx5_top_heads = set(zip(idx5_ranking.head(args.top_k)["layer"], idx5_ranking.head(args.top_k)["head"]))
        overlap = idx5_top_heads & misconception_top_heads
        print(f"\nOverlap between idx5's top-{args.top_k} heads (script 04) and misconception-set top-{args.top_k}: {overlap}")
    else:
        print("\n(Run script 04 first to compare against idx5's top heads.)")

    # Overlap vs every hand-written pair's own top heads
    handwritten_top_heads_by_pair = {}
    for name, (clean_p, corrupt_p, correct_w, predicted_w) in HAND_WRITTEN_PAIRS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            rec, _, _, _ = run_head_patch_sweep(model, clean_p, corrupt_p, correct_w, predicted_w)
        flat_idx = np.argsort(-np.abs(rec).ravel())[: args.top_k]
        heads = {divmod(i, rec.shape[1]) for i in flat_idx}
        handwritten_top_heads_by_pair[name] = heads

    print(f"\nOverlap between each hand-written pair's own top-{args.top_k} heads and misconception-set top-{args.top_k}:")
    for name, heads in handwritten_top_heads_by_pair.items():
        print(f"  {name}: {heads & misconception_top_heads}")

    union_top_heads = set.union(*handwritten_top_heads_by_pair.values())
    intersection_top_heads = set.intersection(*handwritten_top_heads_by_pair.values())
    print(f"Union of ALL hand-written pairs' top-{args.top_k} heads ∩ misconception-set top-{args.top_k}: {union_top_heads & misconception_top_heads}")
    print(f"Heads common to EVERY hand-written pair's own top-{args.top_k} ∩ misconception-set top-{args.top_k}: {intersection_top_heads & misconception_top_heads}")

    # --- Plots ---
    plt.figure(figsize=(10, 8))
    import seaborn as sns

    sns.heatmap(mean_recovery.T, cmap="RdBu", center=0, vmin=-1, vmax=1, cbar_kws={"label": "Mean recovery"})
    plt.xlabel("Layer")
    plt.ylabel("Head")
    plt.title(f"Mean single-head recovery across {len(stable_names)} misconception-adjacent pairs")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_recovery_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(freq_df)), freq_df[f"top{args.top_k}_frequency"])
    plt.xlabel("Head rank (sorted by frequency)")
    plt.ylabel(f"Fraction of {len(stable_names)} pairs where head is in the local top-{args.top_k}")
    plt.title("How concentrated is the 'small set of heads' hypothesis?")
    plt.tight_layout()
    plt.savefig(out_dir / "head_frequency_bar.png", dpi=150)
    plt.close()

    print(f"\nSaved all results and plots to {out_dir}")


if __name__ == "__main__":
    main()
