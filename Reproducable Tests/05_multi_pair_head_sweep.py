"""
Run the same per-head patching sweep as script 04 across the full set of
hand-written sycophancy pairs (common.HAND_WRITTEN_PAIRS), then check
whether the same heads matter across prompts:

- top-15 heads per prompt
- pairwise overlap between prompts' top-15 sets
- heads common to every prompt's top-15
- Spearman rank correlation between full recovery matrices
- side-by-side heatmaps + per-layer bar charts

Usage:
    python "05_multi_pair_head_sweep.py"
"""

import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from common import HAND_WRITTEN_PAIRS, ensure_results_dir, get_device, load_model, run_head_patch_sweep


def top_heads_table(recovery, name, k=15):
    n_layers, n_heads = recovery.shape
    rows = [{"prompt": name, "layer": l, "head": h, "recovery": recovery[l, h]} for l in range(n_layers) for h in range(n_heads)]
    df = pd.DataFrame(rows)
    df["abs_recovery"] = df["recovery"].abs()
    return df.sort_values("abs_recovery", ascending=False).head(k).reset_index(drop=True)


def main():
    model = load_model(device=get_device())

    all_results = {}
    for name, (clean_p, corrupt_p, correct_w, predicted_w) in HAND_WRITTEN_PAIRS.items():
        recovery, _, clean_diff, corrupt_diff = run_head_patch_sweep(model, clean_p, corrupt_p, correct_w, predicted_w)
        all_results[name] = recovery
        print(f"{name}  clean_diff={clean_diff:.3f}  corrupt_diff={corrupt_diff:.3f}")

    out_dir = ensure_results_dir("05_multi_pair_head_sweep")
    np.savez(out_dir / "all_results.npz", **all_results)

    names = list(all_results.keys())
    tables = {name: top_heads_table(rec, name) for name, rec in all_results.items()}
    for name, t in tables.items():
        print(f"\n=== {name} - top 15 heads ===")
        print(t[["layer", "head", "recovery", "abs_recovery"]].to_string(index=False))
        t.to_csv(out_dir / f"top15_{name}.csv", index=False)

    top_sets = {name: set(zip(t["layer"], t["head"])) for name, t in tables.items()}
    common = set.intersection(*top_sets.values())
    print(f"\nHeads in top-15 across ALL prompts: {common}")

    overlap_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = top_sets[names[i]] & top_sets[names[j]]
            print(f"{names[i]} ∩ {names[j]}: {len(overlap)} shared heads -> {overlap}")
            overlap_rows.append({"prompt_a": names[i], "prompt_b": names[j], "n_shared": len(overlap), "shared_heads": str(overlap)})

    corr_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr, p = spearmanr(all_results[names[i]].flatten(), all_results[names[j]].flatten())
            print(f"Spearman({names[i]}, {names[j]}) = {corr:.3f} (p={p:.3g})")
            corr_rows.append({"prompt_a": names[i], "prompt_b": names[j], "spearman": corr, "p_value": p})

    pd.DataFrame(overlap_rows).to_csv(out_dir / "top15_overlap.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out_dir / "spearman_correlation.csv", index=False)

    # Side-by-side heatmap + per-layer bar chart for every prompt
    TOP_K = 15
    NCOLS = 4
    n = len(names)
    nrows = math.ceil(n / NCOLS)

    fig = plt.figure(figsize=(5.5 * NCOLS, 6.2 * nrows))
    outer = gridspec.GridSpec(nrows, NCOLS, figure=fig, wspace=0.35, hspace=0.55)

    cmap = "RdBu"
    norm = plt.Normalize(vmin=-1, vmax=1)

    for idx, name in enumerate(names):
        row, col = divmod(idx, NCOLS)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.12)
        ax_heat = fig.add_subplot(inner[0])
        ax_bar = fig.add_subplot(inner[1])

        recovery = all_results[name]
        sns.heatmap(recovery.T, cmap=cmap, center=0, vmin=-1, vmax=1, ax=ax_heat, cbar=False)
        ax_heat.set_title(name, fontsize=12, fontweight="bold", pad=6)
        ax_heat.set_xlabel("")
        ax_heat.set_xticks([])
        ax_heat.set_ylabel("Head" if col == 0 else "")
        ax_heat.tick_params(axis="y", labelsize=7)

        per_layer_max_abs = np.abs(recovery).max(axis=1)
        top_layers = tables[name].head(TOP_K)["layer"].unique()

        ax_bar.bar(range(len(per_layer_max_abs)), per_layer_max_abs, color="steelblue", width=0.85)
        ax_bar.set_xlabel("Layer", fontsize=9)
        ax_bar.set_ylabel("Max |recovery|" if col == 0 else "", fontsize=8)
        ax_bar.tick_params(axis="both", labelsize=7)
        for layer in top_layers:
            ax_bar.axvline(layer, color="red", alpha=0.12, linewidth=1)

    for idx in range(n, nrows * NCOLS):
        row, col = divmod(idx, NCOLS)
        fig.add_subplot(outer[row, col]).axis("off")

    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, label="Recovery")

    fig.suptitle("Per-head patching recovery + strongest layer effect, all prompts", fontsize=14, y=1.01)
    plt.savefig(out_dir / "multi_pair_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved all results and plots to {out_dir}")


if __name__ == "__main__":
    main()
