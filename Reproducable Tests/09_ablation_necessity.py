"""
Scripts 05-08 establish that (12,7) is one of the most consistent heads by
*sufficiency* (patching it back in recovers logit-diff). This script checks
*necessity* instead: ablate it (and, separately, all of layer 12) on CLEAN
prompts and see how much of the clean logit-diff advantage is lost, under
both zero-ablation and mean-ablation (mean over the same stable prompts).

Usage:
    python "09_ablation_necessity.py" [--limit 100] [--min-gap 0.3] [--layer 12] [--head 7]
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch

from common import (
    HAND_WRITTEN_PAIRS,
    ensure_results_dir,
    filter_stable_pairs,
    get_device,
    load_local_misconception_pairs,
    load_model,
    mean_clean_activation,
    necessity_drop,
    to_pair_records,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Max misconception pairs to include (on top of all hand-written pairs)")
    parser.add_argument("--min-gap", type=float, default=0.3, help="Minimum gap to keep a pair")
    parser.add_argument("--layer", type=int, default=12, help="Layer to test necessity for")
    parser.add_argument("--head", type=int, default=7, help="Head (within --layer) to test necessity for")
    args = parser.parse_args()

    model = load_model(device=get_device())
    out_dir = ensure_results_dir("09_ablation_necessity")

    records = to_pair_records(HAND_WRITTEN_PAIRS) + load_local_misconception_pairs(limit=args.limit)
    stable_pairs, _, _ = filter_stable_pairs(model, records, min_gap=args.min_gap)
    print(f"{len(stable_pairs)}/{len(records)} pairs have a stable gap -- using these for ablation")

    hook_name = f"blocks.{args.layer}.attn.hook_z"
    clean_prompts = [p["clean_prompt"] for p in stable_pairs]
    mean_z = mean_clean_activation(model, clean_prompts, hook_name)
    zero_z = torch.zeros_like(mean_z)
    print(f"Mean activation computed over {len(clean_prompts)} clean prompts, shape {tuple(mean_z.shape)}")

    all_heads = list(range(model.cfg.n_heads))
    head_label = f"head_{args.layer}_{args.head}"
    layer_label = f"layer{args.layer}"

    necessity_rows = []
    for p in stable_pairs:
        clean_p, correct_w, predicted_w = p["clean_prompt"], p["correct_word"], p["predicted_word"]
        necessity_rows.append(
            {
                "name": p["name"],
                f"{head_label}_zero": necessity_drop(model, clean_p, correct_w, predicted_w, [args.head], zero_z, hook_name),
                f"{layer_label}_zero": necessity_drop(model, clean_p, correct_w, predicted_w, all_heads, zero_z, hook_name),
                f"{head_label}_mean": necessity_drop(model, clean_p, correct_w, predicted_w, [args.head], mean_z, hook_name),
                f"{layer_label}_mean": necessity_drop(model, clean_p, correct_w, predicted_w, all_heads, mean_z, hook_name),
            }
        )

    necessity_df = pd.DataFrame(necessity_rows)
    print(necessity_df.to_string(index=False))

    cols = [f"{head_label}_zero", f"{layer_label}_zero", f"{head_label}_mean", f"{layer_label}_mean"]
    print("\nMean +/- std necessity drop, across the same stable pairs:")
    for col in cols:
        vals = necessity_df[col]
        print(f"  {col}: mean={vals.mean():.3f}  median={vals.median():.3f}  std={vals.std():.3f}")

    ratio = necessity_df[f"{head_label}_mean"].mean() / necessity_df[f"{layer_label}_mean"].mean()
    print(f"\n{head_label}'s share of {layer_label}'s necessity under mean-ablation (ratio of means): {ratio:.2f}")

    necessity_df.to_csv(out_dir / "necessity_drop.csv", index=False)

    plt.figure(figsize=(9, 6))
    necessity_df[cols].plot(kind="box", ax=plt.gca())
    plt.ylabel("Fraction of clean advantage lost")
    plt.title(f"Necessity: mean- vs. zero-ablation, L{args.layer}H{args.head} alone vs. whole layer {args.layer}")
    plt.tight_layout()
    plt.savefig(out_dir / "necessity_boxplot.png", dpi=150)
    plt.close()

    print(f"\nSaved CSV and plot to {out_dir}")


if __name__ == "__main__":
    main()
