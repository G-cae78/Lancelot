"""
Full per-head patching sweep (all layers x all heads, hook_z, final token
position) on the idx5 clean/corrupt pair. Ranks heads by |recovery| and
checks whether the effect clusters in a few layers or spreads across the
network.

Usage:
    python "04_single_pair_full_head_sweep.py" [--top-k 10]
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import IDX5_PAIR, ensure_results_dir, get_device, load_model, run_head_patch_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10, help="How many top heads to inspect for clustering")
    args = parser.parse_args()

    model = load_model(device=get_device())

    recovery, raw_effect, clean_diff, corrupt_diff = run_head_patch_sweep(
        model,
        IDX5_PAIR["clean_prompt"],
        IDX5_PAIR["corrupt_prompt"],
        IDX5_PAIR["correct_word"],
        IDX5_PAIR["predicted_word"],
    )
    print(f"Clean diff: {clean_diff:.3f}  Corrupt diff: {corrupt_diff:.3f}")

    n_layers, n_heads = recovery.shape
    rows = [{"layer": layer, "head": head, "recovery": recovery[layer, head]} for layer in range(n_layers) for head in range(n_heads)]
    head_ranking = pd.DataFrame(rows)
    head_ranking["abs_recovery"] = head_ranking["recovery"].abs()
    head_ranking = head_ranking.sort_values("abs_recovery", ascending=False).reset_index(drop=True)

    print("\nTop heads by |recovery|:")
    print(head_ranking.to_string(index=False))

    out_dir = ensure_results_dir("04_single_pair_full_head_sweep")
    head_ranking.to_csv(out_dir / "head_ranking_idx5.csv", index=False)

    top_heads = head_ranking.head(args.top_k)
    layer_counts = top_heads["layer"].value_counts().sort_index()
    print(f"\nLayer distribution of top {args.top_k} heads:")
    print(layer_counts.to_string())

    per_layer_max_abs = head_ranking.groupby("layer")["abs_recovery"].max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(
        recovery,
        cmap="RdBu",
        center=0,
        vmin=-1,
        vmax=1,
        ax=axes[0],
        cbar_kws={"label": "Normalized logit diff recovery"},
    )
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Per-head patching recovery (final token position)")

    axes[1].bar(per_layer_max_abs.index, per_layer_max_abs.values)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Max |recovery| across heads in layer")
    axes[1].set_title(f"Strongest head effect per layer (top {args.top_k} heads highlighted)")
    for layer in top_heads["layer"].unique():
        axes[1].axvline(layer, color="red", alpha=0.15)

    plt.tight_layout()
    plt.savefig(out_dir / "head_patching_heatmap.png", dpi=150)
    plt.close()

    print(f"\nSaved ranking CSV and heatmap to {out_dir}")


if __name__ == "__main__":
    main()
