"""
Sanity check: load GPT-2-medium via TransformerLens and confirm its hooks
work. Run this first after installing requirements -- if it fails, nothing
downstream will work either.

Usage:
    python "01_model_sanity_check.py"
"""

import torch

from common import get_device, load_model


def main():
    device = get_device()
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"device: {device}")

    model = load_model(device=device)
    print("Model loaded!")

    tokens = model.to_tokens("The Eiffel Tower is located in")
    logits, cache = model.run_with_cache(tokens)
    hook_points = list(cache.keys())
    print(f"Ran with cache: {len(hook_points)} hook points, e.g. {hook_points[:5]}")
    assert "blocks.0.attn.hook_z" in cache, "expected attention hook_z not found in cache"
    print("Hooks confirmed working.")


if __name__ == "__main__":
    main()
