"""
First activation-patching pass on the idx5 clean/corrupt pair:

1. Residual-stream (hook_resid_pre) patching, swept across every layer and
   the last 5 shared token positions, to see where in the network the
   clean->corrupt logit-diff gap gets recovered.
2. Per-head (hook_z) patching at the single layer that scored highest in
   step 1, at the final token position, to narrow from "this layer" to
   "these heads".

Usage:
    python "03_single_pair_layer_patching.py"
"""

import matplotlib.pyplot as plt
import torch

from common import IDX5_PAIR, ensure_results_dir, get_device, load_model, make_logit_diff_fn


def main():
    model = load_model(device=get_device())

    clean_prompt = IDX5_PAIR["clean_prompt"]
    corrupt_prompt = IDX5_PAIR["corrupt_prompt"]
    logit_diff, _, _ = make_logit_diff_fn(model, IDX5_PAIR["correct_word"], IDX5_PAIR["predicted_word"])

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    print(f"clean/corrupt token shapes: {clean_tokens.shape}, {corrupt_tokens.shape}")

    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, _ = model.run_with_cache(corrupt_tokens)
    clean_diff = float(logit_diff(clean_logits))
    corrupt_diff = float(logit_diff(corrupt_logits))
    print(f"Clean logit diff: {clean_diff:.3f}")
    print(f"Corrupt logit diff: {corrupt_diff:.3f}")

    # --- Step 1: residual-stream patching, all layers x last 5 positions ---
    clean_len = clean_tokens.shape[1]
    corrupt_len = corrupt_tokens.shape[1]
    max_shared_suffix = 5

    n_layers = model.cfg.n_layers
    patch_results = torch.zeros(n_layers, max_shared_suffix)

    for layer in range(n_layers):
        for offset in range(1, max_shared_suffix + 1):
            clean_pos = clean_len - offset
            corrupt_pos = corrupt_len - offset

            def hook_fn(activation, hook, clean_pos=clean_pos, corrupt_pos=corrupt_pos):
                activation[:, corrupt_pos, :] = clean_cache[hook.name][:, clean_pos, :]
                return activation

            with torch.inference_mode():
                patched_logits = model.run_with_hooks(
                    corrupt_tokens, fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook_fn)]
                )
            patched_diff = float(logit_diff(patched_logits))
            gap = clean_diff - corrupt_diff
            patch_results[layer, max_shared_suffix - offset] = (patched_diff - corrupt_diff) / gap

    out_dir = ensure_results_dir("03_single_pair_layer_patching")

    plt.figure(figsize=(10, 6))
    plt.imshow(patch_results.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(label="Normalized logit diff recovery")
    plt.xlabel("Token position (0 = earliest of last 5 shared positions)")
    plt.ylabel("Layer")
    plt.title("Residual stream patching: clean -> corrupt")
    plt.savefig(out_dir / "resid_patching_heatmap.png", dpi=150)
    plt.close()

    best_layer = int(patch_results.abs().amax(dim=1).argmax().item())
    print(f"Layer with the strongest residual-stream effect: {best_layer}")

    # --- Step 2: per-head patching at the best layer, final token position ---
    n_heads = model.cfg.n_heads
    head_results = torch.zeros(n_heads)
    last_pos = clean_tokens.shape[1] - 1

    for head in range(n_heads):

        def hook_fn(activation, hook, head_idx=head, position=last_pos):
            activation[:, position, head_idx, :] = clean_cache[hook.name][:, position, head_idx, :]
            return activation

        with torch.inference_mode():
            patched_logits = model.run_with_hooks(
                corrupt_tokens, fwd_hooks=[(f"blocks.{best_layer}.attn.hook_z", hook_fn)]
            )
        patched_diff = float(logit_diff(patched_logits))
        gap = clean_diff - corrupt_diff
        head_results[head] = (patched_diff - corrupt_diff) / gap

    print(f"\nPer-head recovery at layer {best_layer}:")
    for head in range(n_heads):
        print(f"  H{head}: {head_results[head].item():.4f}")

    import pandas as pd

    df = pd.DataFrame({"head": range(n_heads), "recovery": head_results.numpy()})
    df.to_csv(out_dir / f"layer_{best_layer}_head_recovery.csv", index=False)
    print(f"\nSaved plot and CSV to {out_dir}")


if __name__ == "__main__":
    main()
