"""
Baseline hallucination rate: score every TruthfulQA multiple-choice answer
with GPT-2-medium's own log-probs and see how often it picks the popular
misconception over the correct answer.

Usage:
    python "02_truthfulqa_mc_baseline.py" [--n-examples 50]
"""

import argparse

import pandas as pd
from datasets import load_dataset

from common import ensure_results_dir, get_device, load_model, score_choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=50, help="Number of TruthfulQA validation examples to score")
    args = parser.parse_args()

    model = load_model(device=get_device())

    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    n_examples = min(args.n_examples, len(dataset["validation"]))

    results = []
    for idx in range(n_examples):
        example = dataset["validation"][idx]
        prompt = f"Question: {example['question']}\nAnswer:"
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        correct_idx = labels.index(1)

        scores = [score_choice(model, prompt, choice) for choice in choices]
        predicted_idx = scores.index(max(scores))
        hallucinated = predicted_idx != correct_idx

        results.append(
            {
                "idx": idx,
                "question": example["question"],
                "correct_answer": choices[correct_idx],
                "predicted_answer": choices[predicted_idx],
                "correct_score": scores[correct_idx],
                "predicted_score": scores[predicted_idx],
                "hallucinated": hallucinated,
            }
        )
        status = "HALLUCINATED" if hallucinated else "CORRECT"
        print(f"[{idx:02d}] {status} | {example['question'][:60]}")

    df = pd.DataFrame(results)
    print(f"\nTotal: {len(df)} | Hallucinated: {df['hallucinated'].sum()} | Correct: {(~df['hallucinated']).sum()}")
    print(f"Hallucination rate: {df['hallucinated'].mean():.1%}")

    out_dir = ensure_results_dir("02_truthfulqa_mc_baseline")
    out_path = out_dir / "hallucination_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
