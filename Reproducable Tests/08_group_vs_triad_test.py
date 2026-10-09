"""
Individual top heads bounce around per prompt (script 05/07), but a few
specific heads recur constantly: (12,7), (13,2), (13,8). Tests three
groupings -- jointly patched in a single forward pass -- across every pair
in both the hand-written set and the misconception set:

  triad     -- just (12,7), (13,2), (13,8)
  layer_12  -- every head in layer 12
  layer_13  -- every head in layer 13

A grouping with consistently high recovery and low variance across pairs
(even where one of its members individually looks weak) is the one
actually carrying the effect.

Usage:
    python "08_group_vs_triad_test.py" [--limit 100] [--min-gap 0.3]
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch

from common import (
    HAND_WRITTEN_PAIRS,
    combo_recovery,
    ensure_results_dir,
    get_device,
    load_local_misconception_pairs,
    load_model,
    make_logit_diff_fn,
)


def group_recovery_for_pair(model, clean_prompt, corrupt_prompt, correct_word, predicted_word, groups):
    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    logit_diff, _, _ = make_logit_diff_fn(model, correct_word, predicted_word)

    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, _ = model.run_with_cache(corrupt_tokens)
    clean_diff = float(logit_diff(clean_logits))
    corrupt_diff = float(logit_diff(corrupt_logits))

    result = {"gap": clean_diff - corrupt_diff}
    for name, heads in groups.items():
        result[name] = combo_recovery(model, heads, clean_cache, corrupt_tokens, logit_diff, clean_diff, corrupt_diff)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Max misconception pairs to include (on top of all hand-written pairs)")
    parser.add_argument("--min-gap", type=float, default=0.3, help="Minimum gap to keep a pair")
    args = parser.parse_args()

    model = load_model(device=get_device())
    out_dir = ensure_results_dir("08_group_vs_triad_test")

    n_heads = model.cfg.n_heads
    GROUPS = {
        "triad": [(12, 7), (13, 2), (13, 8)],
        "layer_12": [(12, h) for h in range(n_heads)],
        "layer_13": [(13, h) for h in range(n_heads)],
    }

    group_test_rows = []
    for name, (clean_p, corrupt_p, correct_w, predicted_w) in HAND_WRITTEN_PAIRS.items():
        r = group_recovery_for_pair(model, clean_p, corrupt_p, correct_w, predicted_w, GROUPS)
        r.update(name=name, pair_id=name, source="hand_written")
        group_test_rows.append(r)

    misconception_pairs = load_local_misconception_pairs(limit=args.limit)
    for p in misconception_pairs:
        r = group_recovery_for_pair(model, p["clean_prompt"], p["corrupt_prompt"], p["correct_word"], p["predicted_word"], GROUPS)
        r.update(name=p["name"], pair_id=p["name"], source="misconception_set")
        group_test_rows.append(r)

    group_df = pd.DataFrame(group_test_rows)
    stable_group_df = group_df[group_df["gap"] >= args.min_gap].reset_index(drop=True)
    print(f"{len(stable_group_df)}/{len(group_df)} pairs (combined) have a stable gap")
    print(stable_group_df[["name", "source", "gap"] + list(GROUPS)].to_string(index=False))

    print("\nMean +/- std recovery per group, across stable pairs:")
    for g in GROUPS:
        vals = stable_group_df[g]
        print(f"{g}: mean={vals.mean():.3f}  std={vals.std():.3f}")

    group_df.to_csv(out_dir / "group_vs_triad_all_pairs.csv", index=False)
    stable_group_df.to_csv(out_dir / "group_vs_triad_stable_pairs.csv", index=False)

    plt.figure(figsize=(8, 6))
    stable_group_df[list(GROUPS)].plot(kind="box", ax=plt.gca())
    plt.ylabel("Recovery")
    plt.title("Recovery consistency: triad vs. whole layers, across all stable pairs")
    plt.tight_layout()
    plt.savefig(out_dir / "group_vs_triad_boxplot.png", dpi=150)
    plt.close()

    print(f"\nSaved CSVs and plot to {out_dir}")


if __name__ == "__main__":
    main()
