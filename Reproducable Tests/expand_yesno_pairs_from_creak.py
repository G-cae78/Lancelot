"""
Scale yesno_pairs.json up by pulling more claims from the full CREAK dataset
(amydeng2000/CREAK on the HF hub, 11,547 claims total) beyond the 1,203
already converted into misconception_pairs.json. No grammar transformation
needed: every existing CREAK-derived entry just wraps the claim verbatim in
"Is it true that {claim}?" for clean_prompt, so new claims convert the same
safe way.

label 'true'  -> correct_word "Yes", predicted_word "No"
label 'false' -> correct_word "No",  predicted_word "Yes"
(predicted_word is always just the flip of correct_word -- the corrupted
framing is hypothesized to pull toward whichever answer is wrong.)

corrupt_prompt rotates through the same CORRUPT_TEMPLATES as
build_yesno_pairs.py, continuing the round-robin so the overall file stays
balanced across phrasings.

New pairs are written with "stable": null, not true. An earlier version of
this script blanket-asserted "stable": true on every CREAK pair without
checking it against the model -- a real gap-filter run later showed only
~22% of those actually clear MIN_GAP, with most of the rest "backfiring"
(the corrupted prompt scoring *better* than the clean one), not just falling
short. Asserting stability we haven't measured was the bug; null means
"not yet validated" and leaves the real quick_gap filter as the only source
of truth for which pairs are actually usable.

Usage:
    python expand_yesno_pairs_from_creak.py [--target-count 5000]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YESNO_PATH = PROJECT_ROOT / "yesno_pairs.json"

# Must match build_yesno_pairs.py's CORRUPT_TEMPLATES exactly so the
# round-robin continues the same rotation across both scripts' output.
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


def normalize(sentence: str) -> str:
    return sentence.strip().rstrip(".").strip()


def extract_used_claims(existing_pairs: list[dict]) -> set[str]:
    """Reverse-engineer the raw claim out of every existing creak-sourced
    clean_prompt ("Q: Is it true that {claim}?\\nA:") so we don't duplicate it."""
    used = set()
    prefix = "Q: Is it true that "
    suffix = "?\nA:"
    for p in existing_pairs:
        if p.get("source") != "creak":
            continue
        cp = p["clean_prompt"]
        if cp.startswith(prefix) and cp.endswith(suffix):
            used.add(normalize(cp[len(prefix) : -len(suffix)]))
    return used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-count", type=int, default=5000, help="Stop once yesno_pairs.json reaches this many total entries")
    args = parser.parse_args()

    with open(YESNO_PATH) as f:
        existing_pairs = json.load(f)
    print(f"Starting from {len(existing_pairs)} existing pairs")

    if len(existing_pairs) >= args.target_count:
        print(f"Already at or above --target-count {args.target_count}. Nothing to do.")
        return

    used_claims = extract_used_claims(existing_pairs)
    print(f"{len(used_claims)} claims already used (won't be re-added)")

    # Continue the round-robin template rotation from where build_yesno_pairs.py left off.
    template_cursor = len(existing_pairs) % len(CORRUPT_TEMPLATES)

    creak = load_dataset("amydeng2000/CREAK")
    candidates = list(creak["train"]) + list(creak["validation"])
    print(f"{len(candidates)} total CREAK claims available")

    new_pairs = []
    needed = args.target_count - len(existing_pairs)
    skipped_dupe = 0
    skipped_empty = 0

    for row in candidates:
        if len(new_pairs) >= needed:
            break

        claim = normalize(row["sentence"])
        if not claim:
            skipped_empty += 1
            continue
        if claim in used_claims:
            skipped_dupe += 1
            continue
        used_claims.add(claim)

        is_true = row["label"] == "true"
        correct_word = "Yes" if is_true else "No"
        predicted_word = "No" if is_true else "Yes"

        template_id, template = CORRUPT_TEMPLATES[template_cursor % len(CORRUPT_TEMPLATES)]
        template_cursor += 1

        new_pairs.append(
            {
                "name": row["sentence"][:40],
                "category": f"CREAK: {row['entity']}",
                "clean_prompt": f"Q: Is it true that {claim}?\nA:",
                "corrupt_prompt": f"Q: {template.format(clause=claim)}\nA:",
                "correct_word": correct_word,
                "predicted_word": predicted_word,
                "source": "creak",
                "stable": None,
                "gap": None,
                "corrupt_template_id": template_id,
            }
        )

    print(f"Skipped {skipped_dupe} already-used claims, {skipped_empty} empty claims")
    print(f"Adding {len(new_pairs)} new pairs")

    all_pairs = existing_pairs + new_pairs
    with open(YESNO_PATH, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"Wrote {len(all_pairs)} total pairs to {YESNO_PATH}")
    if len(all_pairs) < args.target_count:
        print(f"Note: ran out of unused CREAK claims before reaching --target-count {args.target_count}.")


if __name__ == "__main__":
    main()
