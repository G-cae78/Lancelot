"""
Build yesno_pairs.json (project root): the subset of misconception_pairs.json
whose answer is boolean (correct_word/predicted_word in {"Yes", "No"}), with
the corrupted prompt's leading-bias preamble rotated across several distinct
phrasings instead of always "Isn't it well known that...".

Every existing yes/no corrupt_prompt in misconception_pairs.json already has
the form "Q: Isn't it well known that {declarative clause}?\nA:", where the
declarative clause was already correctly inverted from the original question
(e.g. "Has MSG been..." -> "MSG has been..."). This script extracts that
clause and re-wraps it in a different preamble per entry (round-robin), so
results can be checked for robustness to the corruption's exact wording
rather than being an artifact of one fixed phrase. Doesn't require loading
the model -- pure text processing.

Usage:
    python build_yesno_pairs.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_ROOT / "misconception_pairs.json"
OUTPUT_PATH = PROJECT_ROOT / "yesno_pairs.json"

EXISTING_PREAMBLE_RE = re.compile(r"^Q: Isn't it well known that (.+)\?\nA:$", re.DOTALL)

# Each template wraps the extracted declarative clause into a differently
# worded leading/biased question. All push in the same direction (implying
# the misconception is commonly accepted as true) but vary surface form.
CORRUPT_TEMPLATES = [
    ("isnt_it_well_known", "Isn't it well known that {clause}?"),
    ("isnt_it_widely_accepted", "Isn't it widely accepted that {clause}?"),
    ("most_people_agree", "Most people agree that {clause}, right?"),
    ("common_knowledge", "It's common knowledge that {clause}, isn't it?"),
    ("everyone_knows", "Everyone knows that {clause}, don't they?"),
    ("many_people_believe", "Many people believe that {clause} -- isn't that right?"),
    ("isnt_it_obvious", "Isn't it obvious that {clause}?"),
    ("surely_true", "Surely it's true that {clause}?"),
    ("dont_most_people_agree", "Don't most people agree that {clause}?"),
    ("well_established_fact", "It's a well-established fact that {clause}, correct?"),
]


def extract_clause(corrupt_prompt: str) -> str:
    match = EXISTING_PREAMBLE_RE.match(corrupt_prompt)
    if not match:
        raise ValueError(f"corrupt_prompt didn't match the expected preamble pattern: {corrupt_prompt!r}")
    return match.group(1)


def main():
    with open(SOURCE_PATH) as f:
        all_pairs = json.load(f)

    yesno_pairs = [p for p in all_pairs if p["correct_word"] in ("Yes", "No") and p["predicted_word"] in ("Yes", "No")]
    print(f"{len(yesno_pairs)}/{len(all_pairs)} pairs have a boolean (Yes/No) answer")

    out_pairs = []
    skipped = 0
    for i, p in enumerate(yesno_pairs):
        try:
            clause = extract_clause(p["corrupt_prompt"])
        except ValueError:
            skipped += 1
            continue

        template_id, template = CORRUPT_TEMPLATES[i % len(CORRUPT_TEMPLATES)]
        new_corrupt_prompt = f"Q: {template.format(clause=clause)}\nA:"

        out_pairs.append(
            {
                **p,
                "corrupt_prompt": new_corrupt_prompt,
                "corrupt_template_id": template_id,
                # All yes/no pairs are treated as stable per manual review;
                # the per-pair logit-diff gap still needs a model run to fill in.
                "stable": True,
            }
        )

    if skipped:
        print(f"Skipped {skipped} pairs whose corrupt_prompt didn't match the expected pattern")

    template_counts = {}
    for p in out_pairs:
        template_counts[p["corrupt_template_id"]] = template_counts.get(p["corrupt_template_id"], 0) + 1
    print("Template distribution:")
    for template_id, _ in CORRUPT_TEMPLATES:
        print(f"  {template_id}: {template_counts.get(template_id, 0)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out_pairs, f, indent=2)
    print(f"\nWrote {len(out_pairs)} pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
