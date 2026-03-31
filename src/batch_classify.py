# batch_classify.py - Classify full TRACKR dataset via sequential LLM calls
#
# Reads ~501 listings from FULL TRACKR LISTINGS.csv, classifies each batch
# using the LLM pipeline (gpt-4o-mini with gpt-4o escalation), saves results
# to data/llm_results.json and data/classified_listings.xlsx.

import os
import sys
import json
import argparse

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from src.taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES
from src.export import export_to_excel
import src.llm_classifier as llm_mod
from src.llm_classifier import classify_batch
from src.baseline import rule_based_baseline

TRACKR_CSV = "data/FULL TRACKR LISTINGS.csv"
LLM_RESULTS_PATH = "data/llm_results.json"
BATCH_SIZE = 8

# Import the canonical escalation threshold from llm_classifier
from src.llm_classifier import ESCALATION_THRESHOLD


# ---------------------------------------------------------------------------
# 1. CSV loader
# ---------------------------------------------------------------------------

def load_trackr_csv(path=TRACKR_CSV):
    """Load the FULL TRACKR LISTINGS.csv with its non-standard format.

    The CSV has 5 junk rows at the top, a header row (line 6), and a
    sub-header row (line 7). Data starts at line 8. Semicolon-delimited,
    latin-1 encoding.
    """
    # Read with skiprows to skip junk lines 0-4 (5 rows), keeping line 5 as header
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        skiprows=[0, 1, 2, 3, 4],
        on_bad_lines="skip",
    )

    # Drop the sub-header row (first data row has empty Company Name)
    # and any other rows where Company Name is blank
    company_col = "Company Name"
    if company_col not in df.columns:
        # Try to find the column
        for col in df.columns:
            if "company" in col.lower():
                company_col = col
                break

    df = df[df[company_col].notna() & (df[company_col].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)

    # Rename columns to snake_case
    rename_map = {
        "My Status": "my_status",
        "Company Name": "company_name",
        "Programme Name": "programme_name",
        "Opening Date": "opening_date",
        "Closing Date": "closing_date",
        "Latest Stage": "latest_stage",
        "Last Year Opening": "last_year_opening",
        "Process": "process",
        "Info & Test Prep": "info_test_prep",
        "Rolling": "rolling",
        "CV": "cv",
        "Cover Letter": "cover_letter",
        "Written Answers": "written_answers",
        "Notes": "notes",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.fillna("")

    print(f"[load] Loaded {len(df)} listings from {path}")
    return df


# ---------------------------------------------------------------------------
# 2. Classification function
# ---------------------------------------------------------------------------

def classify_all(listings):
    """Classify all listings in sequential batches."""
    all_results = []
    total_batches = (len(listings) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(listings), BATCH_SIZE):
        batch = listings[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"Classifying batch {batch_num}/{total_batches} ({len(batch)} listings)...")

        results = classify_batch(batch)
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# 3. Validation
# ---------------------------------------------------------------------------

def validate_results(results):
    """Check every result has valid taxonomy values. Returns list of errors."""
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


# ---------------------------------------------------------------------------
# 4. Escalate-only mode
# ---------------------------------------------------------------------------

def escalate_only():
    """Re-run escalation on existing llm_results.json for low-confidence entries."""
    if not os.path.exists(LLM_RESULTS_PATH):
        print(f"ERROR: {LLM_RESULTS_PATH} not found. Run a full classification first.")
        sys.exit(1)

    with open(LLM_RESULTS_PATH) as f:
        results = json.load(f)

    # Find entries with any confidence < threshold
    to_escalate = []
    escalate_indices = []
    for i, r in enumerate(results):
        confs = [
            r.get("firm_type_confidence", 1.0),
            r.get("role_function_confidence", 1.0),
            r.get("programme_status_confidence", 1.0),
        ]
        if any(c < ESCALATION_THRESHOLD for c in confs):
            to_escalate.append({
                "company_name": r.get("company_name", ""),
                "programme_name": (r.get("programme_name")
                                   or r.get("programme", "")),
                "opening_date": r.get("opening_date", ""),
                "closing_date": r.get("closing_date", ""),
                "latest_stage": r.get("latest_stage", ""),
            })
            escalate_indices.append(i)

    if not to_escalate:
        print("No entries below escalation threshold. Nothing to do.")
        return

    print(f"Found {len(to_escalate)} entries below {ESCALATION_THRESHOLD} confidence threshold")
    print("Re-classifying with gpt-4o...")

    # Force gpt-4o for these by temporarily setting threshold very high
    original_threshold = llm_mod.ESCALATION_THRESHOLD
    llm_mod.ESCALATION_THRESHOLD = 0.0  # escalate everything

    total_batches = (len(to_escalate) + BATCH_SIZE - 1) // BATCH_SIZE
    new_results = []
    for i in range(0, len(to_escalate), BATCH_SIZE):
        batch = to_escalate[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Escalation batch {batch_num}/{total_batches}...")
        new_results.extend(classify_batch(batch))

    llm_mod.ESCALATION_THRESHOLD = original_threshold

    # Merge back
    for idx, new_r in zip(escalate_indices, new_results):
        results[idx] = new_r

    with open(LLM_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated {len(new_results)} entries in {LLM_RESULTS_PATH}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify full TRACKR dataset via LLM"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Process only the first N listings",
    )
    parser.add_argument(
        "--sequential", action="store_true", default=True,
        help="Sequential classification (default)",
    )
    parser.add_argument(
        "--escalate-only", action="store_true",
        help="Re-run escalation on existing llm_results.json",
    )
    args = parser.parse_args()

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "API_key_here":
        print("ERROR: No valid OPENAI_API_KEY found in .env")
        print("Set your key: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if args.escalate_only:
        escalate_only()
        return

    # Load data
    print(f"\n[1/5] Loading TRACKR data...")
    df = load_trackr_csv()

    if args.sample and args.sample < len(df):
        df = df.head(args.sample).reset_index(drop=True)
        print(f"  [sample] Limited to {args.sample} listings")

    listings = df.to_dict(orient="records")

    # Classify
    print(f"\n[2/5] Running LLM classification ({len(listings)} listings, batch size {BATCH_SIZE})...")
    llm_results = classify_all(listings)

    # Validate
    print(f"\n[3/5] Validating results...")
    errors = validate_results(llm_results)
    if errors:
        print(f"  {len(errors)} validation errors:")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
    else:
        print("  All results valid.")

    # Save JSON
    print(f"\n[4/5] Saving results...")
    with open(LLM_RESULTS_PATH, "w") as f:
        json.dump(llm_results, f, indent=2)
    print(f"  Saved {len(llm_results)} entries to {LLM_RESULTS_PATH}")

    # Build merged DataFrame for Excel export
    print(f"\n[5/5] Exporting to Excel...")
    # Add baseline results for comparison
    baseline_results = rule_based_baseline(listings)

    for key in ["firm_type", "firm_type_confidence", "role_function",
                "role_function_confidence", "programme_status",
                "programme_status_confidence"]:
        df[f"baseline_{key}"] = [r.get(key, "") for r in baseline_results]

    for key in ["firm_type", "firm_type_confidence", "firm_type_rationale",
                "role_function", "role_function_confidence", "role_function_rationale",
                "programme_status", "programme_status_confidence",
                "programme_status_rationale", "model_used", "escalated"]:
        df[f"llm_{key}"] = [r.get(key, "") for r in llm_results]

    filepath = export_to_excel(df, output_dir="data", filename="classified_listings.xlsx")

    # Summary
    escalated_count = sum(1 for r in llm_results if r.get("escalated"))
    print(f"\n{'=' * 60}")
    print(f"BATCH CLASSIFICATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total classified:     {len(llm_results)}")
    print(f"  Escalated to gpt-4o:  {escalated_count}")
    print(f"  Validation errors:    {len(errors)}")
    print(f"  JSON output:          {LLM_RESULTS_PATH}")
    print(f"  Excel output:         {filepath}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
