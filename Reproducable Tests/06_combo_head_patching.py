"""
Does a prompt's top-ranked heads act as a genuine small circuit, or as
several independent, redundant paths to the same effect? Patches SETS of
heads jointly (all in the same forward pass) and compares the actual joint
recovery against the additive prediction (sum of each head's individual
recovery):

  joint ≈ sum of singles   -> additive: heads act independently
  joint < sum of singles   -> sub-additive / redundant
  joint > sum of singles   -> super-additive: a genuine small circuit

Two experiments:
1. Pairwise interaction among one prompt's top-N heads (--pair, default idx30).
2. Cumulative joint-patching (1, 2, 3, ... top heads) vs. the additive
   prediction, for every hand-written pair.

Usage:
    python "06_combo_head_patching.py" [--pair idx30] [--top-n 6]
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from common import HAND_WRITTEN_PAIRS, combo_recovery, ensure_results_dir, get_device, load_model, make_logit_diff_fn


def run_single_head_sweep_cache(model, clean_tok, corrupt_tok, logit_diff, clean_cache, corrupt_d, clean_d):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    single = np.zeros((n_layers, n_heads))
    gap = clean_d - corrupt_d
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        clean_z_final = clean_cache[hook_name][0, -1, :, :]
        for head in range(n_heads):

            def hook_fn(z, hook, head_idx=head, clean_z_final=clean_z_final):
                z[:, -1, head_idx, :] = clean_z_final[head_idx, :]
                return z

            with torch.inference_mode():
                patched_logits = model.run_with_hooks(corrupt_tok, fwd_hooks=[(hook_name, hook_fn)])
            single[layer, head] = (float(logit_diff(patched_logits)) - corrupt_d) / gap
    return single


def run_combo_experiment(model, clean_prompt, corrupt_prompt, correct_word, predicted_word, top_k=6, tol=0.02, label=""):
    clean_tok = model.to_tokens(clean_prompt)
    corrupt_tok = model.to_tokens(corrupt_prompt)
    logit_diff, _, _ = make_logit_diff_fn(model, correct_word, predicted_word)

    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tok)
        corrupt_logits, _ = model.run_with_cache(corrupt_tok)
    clean_d = float(logit_diff(clean_logits))
    corrupt_d = float(logit_diff(corrupt_logits))
    gap = clean_d - corrupt_d

    single = run_single_head_sweep_cache(model, clean_tok, corrupt_tok, logit_diff, clean_cache, corrupt_d, clean_d)

    def combo_recovery_local(heads):
        return combo_recovery(model, heads, clean_cache, corrupt_tok, logit_diff, clean_d, corrupt_d)

    n_layers, n_heads = single.shape
    ranked = sorted(
        [(l, h, single[l, h]) for l in range(n_layers) for h in range(n_heads)],
        key=lambda r: abs(r[2]),
        reverse=True,
    )[:top_k]
    top_heads = [(l, h) for l, h, _ in ranked]
    single_map = {(l, h): s for l, h, s in ranked}

    cumulative_rows = []
    running = []
    for head in top_heads:
        running.append(head)
        joint = combo_recovery_local(list(running))
        additive_pred = sum(single_map[x] for x in running)
        cumulative_rows.append(
            {
                "prompt": label,
                "n_heads": len(running),
                "last_head_added": f"L{head[0]}H{head[1]}",
                "joint_actual": joint,
                "additive_pred": additive_pred,
                "interaction": joint - additive_pred,
            }
        )

    cumulative_df = pd.DataFrame(cumulative_rows)
    cumulative_df["relationship"] = cumulative_df["interaction"].apply(
        lambda x: "super-additive" if x > tol else ("sub-additive (redundant)" if x < -tol else "additive")
    )
    return {"gap": gap, "top_heads": top_heads, "single": single, "single_map": single_map, "cumulative_df": cumulative_df}


def run_pairwise_interaction(model, clean_prompt, corrupt_prompt, correct_word, predicted_word, pair_name, top_n):
    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    logit_diff, _, _ = make_logit_diff_fn(model, correct_word, predicted_word)

    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, _ = model.run_with_cache(corrupt_tokens)
    clean_diff = float(logit_diff(clean_logits))
    corrupt_diff = float(logit_diff(corrupt_logits))

    single = run_single_head_sweep_cache(model, clean_tokens, corrupt_tokens, logit_diff, clean_cache, corrupt_diff, clean_diff)
    n_heads = single.shape[1]
    ranked = np.argsort(-np.abs(single).ravel())[:top_n]
    top_heads_selected = [divmod(idx, n_heads) for idx in ranked]
    single_recovery = {h: single[h[0], h[1]] for h in top_heads_selected}
    print(f"Top heads for {pair_name} (single-head recovery):")
    for h in top_heads_selected:
        print(f"  L{h[0]}H{h[1]}: {single_recovery[h]:.4f}")

    pair_rows = []
    for i in range(len(top_heads_selected)):
        for j in range(i + 1, len(top_heads_selected)):
            h1, h2 = top_heads_selected[i], top_heads_selected[j]
            joint = combo_recovery(model, [h1, h2], clean_cache, corrupt_tokens, logit_diff, clean_diff, corrupt_diff)
            additive_pred = single_recovery[h1] + single_recovery[h2]
            pair_rows.append(
                {
                    "head_1": f"L{h1[0]}H{h1[1]}",
                    "head_2": f"L{h2[0]}H{h2[1]}",
                    "single_1": single_recovery[h1],
                    "single_2": single_recovery[h2],
                    "additive_pred": additive_pred,
                    "joint_actual": joint,
                    "interaction": joint - additive_pred,
                }
            )

    pair_df = pd.DataFrame(pair_rows)

    def classify(interaction, tol=0.02):
        if interaction > tol:
            return "super-additive"
        if interaction < -tol:
            return "sub-additive (redundant)"
        return "additive"

    pair_df["relationship"] = pair_df["interaction"].apply(classify)
    pair_df = pair_df.sort_values("interaction", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return pair_df, top_heads_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="idx30", choices=list(HAND_WRITTEN_PAIRS.keys()), help="Pair for the pairwise-interaction experiment")
    parser.add_argument("--top-n", type=int, default=6, help="Top-N heads for the pairwise interaction matrix")
    parser.add_argument("--top-k", type=int, default=6, help="Top-K heads for the cumulative combo experiment")
    args = parser.parse_args()

    model = load_model(device=get_device())
    out_dir = ensure_results_dir("06_combo_head_patching")

    # --- Experiment 1: pairwise interaction on one prompt ---
    clean_p, corrupt_p, correct_w, predicted_w = HAND_WRITTEN_PAIRS[args.pair]
    pair_df, top_heads_selected = run_pairwise_interaction(model, clean_p, corrupt_p, correct_w, predicted_w, args.pair, args.top_n)
    print(f"\nPairwise combo-patching results, {args.pair}:")
    print(pair_df.to_string(index=False))
    pair_df.to_csv(out_dir / f"pairwise_interaction_{args.pair}.csv", index=False)

    labels = [f"L{l}H{h}" for l, h in top_heads_selected]
    interaction_matrix = np.zeros((args.top_n, args.top_n))
    for _, row in pair_df.iterrows():
        i = labels.index(row["head_1"])
        j = labels.index(row["head_2"])
        interaction_matrix[i, j] = row["interaction"]
        interaction_matrix[j, i] = row["interaction"]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        interaction_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Interaction (joint_actual - additive_pred)"},
    )
    plt.title(f"Pairwise head interaction, {args.pair} (top {args.top_n} heads)")
    plt.tight_layout()
    plt.savefig(out_dir / f"pairwise_interaction_{args.pair}.png", dpi=150)
    plt.close()

    # --- Experiment 2: cumulative combo test across all hand-written pairs ---
    combo_results = {}
    for name, (clean_p, corrupt_p, correct_w, predicted_w) in HAND_WRITTEN_PAIRS.items():
        combo_results[name] = run_combo_experiment(model, clean_p, corrupt_p, correct_w, predicted_w, top_k=args.top_k, label=name)
        print(f"\n=== {name} (gap={combo_results[name]['gap']:.3f}) ===")
        print(combo_results[name]["cumulative_df"].to_string(index=False))
        combo_results[name]["cumulative_df"].to_csv(out_dir / f"cumulative_combo_{name}.csv", index=False)

    n = len(combo_results)
    ncols = math.ceil(n / 2)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10), sharey=True)
    axes = axes.flatten()

    for ax, (name, res) in zip(axes, combo_results.items()):
        df = res["cumulative_df"]
        ax.plot(df["n_heads"], df["joint_actual"], marker="o", label="Actual joint recovery")
        ax.plot(df["n_heads"], df["additive_pred"], marker="o", linestyle="--", label="Additive prediction")
        ax.axhline(1.0, color="grey", linestyle=":", label="Full recovery")
        ax.set_xlabel("Number of top-ranked heads patched jointly")
        ax.set_title(f"{name} (gap={res['gap']:.3f})")
        ax.legend()

    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].set_ylabel("Recovery")
    if ncols < len(axes):
        axes[ncols].set_ylabel("Recovery")
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_combo_all_pairs.png", dpi=150)
    plt.close()

    print(f"\nSaved all CSVs and plots to {out_dir}")


if __name__ == "__main__":
    main()
