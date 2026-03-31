# classify_gold_standard.py - Run LLM classification on gold standard entries
#
# Reads the 153 hand-labelled entries from data/gold_standard.json, classifies
# each via the LLM pipeline (gpt-4o-mini with gpt-4o escalation), and saves
# results to data/llm_results.json for evaluation by src/evaluate.py.
#
# Usage:
#   python classify_gold_standard.py              # classify all 153 entries
#   python classify_gold_standard.py --sample 10  # classify first 10 only

import os
import sys
import json
import argparse

from dotenv import load_dotenv
load_dotenv()

from src.taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES
from src.llm_classifier import classify_batch

GOLD_STANDARD_PATH = "data/gold_standard.json"
LLM_RESULTS_PATH = "data/llm_results.json"
BATCH_SIZE = 8


def load_gold_standard():
    """Load gold standard entries and normalise field names for the classifier."""
    with open(GOLD_STANDARD_PATH) as f:
        data = json.load(f)

    listings = []
    for g in data:
        listings.append({
            "company_name": g.get("company_name", ""),
            "programme_name": g.get("programme_name") or g.get("programme", ""),
            "opening_date": g.get("opening_date", ""),
            "closing_date": g.get("closing_date") or g.get("closing", ""),
            "latest_stage": g.get("latest_stage") or g.get("stage", ""),
        })

    return listings


def classify_all(listings):
    """Classify all listings in sequential batches."""
    all_results = []
    total_batches = (len(listings) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(listings), BATCH_SIZE):
        batch = listings[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} listings)...")

        results = classify_batch(batch)
        all_results.extend(results)

    return all_results


def validate_results(results):
    """Check every result has valid taxonomy values."""
    errors = []
    for i, r in enumerate(results):
        company = r.get("company_name", f"entry_{i}")
        if r.get("firm_type") not in FIRM_TYPES:
            errors.append(f"{company}: invalid firm_type '{r.get('firm_type')}'")
        if r.get("role_function") not in ROLE_FUNCTIONS:
            errors.append(f"{company}: invalid role_function '{r.get('role_function')}'")
        if r.get("programme_status") not in PROGRAMME_STATUSES:
            errors.append(f"{company}: invalid programme_status '{r.get('programme_status')}'")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Classify gold standard entries via LLM for evaluation"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Process only the first N entries",
    )
    args = parser.parse_args()

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "API_key_here":
        print("ERROR: No valid OPENAI_API_KEY found in .env")
        print("Set your key: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Load gold standard
    print(f"\n[1/4] Loading gold standard from {GOLD_STANDARD_PATH}...")
    listings = load_gold_standard()
    print(f"  Loaded {len(listings)} entries")

    if args.sample and args.sample < len(listings):
        listings = listings[:args.sample]
        print(f"  Limited to {args.sample} entries")

    # Classify
    print(f"\n[2/4] Running LLM classification ({len(listings)} entries, batch size {BATCH_SIZE})...")
    llm_results = classify_all(listings)

    # Validate
    print(f"\n[3/4] Validating results...")
    errors = validate_results(llm_results)
    if errors:
        print(f"  {len(errors)} validation errors:")
        for e in errors[:20]:
            print(f"    {e}")
    else:
        print("  All results valid.")

    # Save
    print(f"\n[4/4] Saving results...")
    with open(LLM_RESULTS_PATH, "w") as f:
        json.dump(llm_results, f, indent=2)
    print(f"  Saved {len(llm_results)} entries to {LLM_RESULTS_PATH}")

    # Summary
    escalated = sum(1 for r in llm_results if r.get("escalated"))
    print(f"\n{'=' * 60}")
    print(f"GOLD STANDARD CLASSIFICATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total classified:     {len(llm_results)}")
    print(f"  Escalated to gpt-4o:  {escalated}")
    print(f"  Validation errors:    {len(errors)}")
    print(f"  Output:               {LLM_RESULTS_PATH}")
    print(f"{'=' * 60}")
    print(f"\nNext step: python -m src.evaluate")


if __name__ == "__main__":
    main()
