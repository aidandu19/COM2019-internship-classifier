# main.py - end-to-end pipeline for COM2019 internship classifier
#
# Orchestrates: CSV ingestion -> baseline classification -> LLM classification
# -> merge results -> export to Excel. Supports --dry-run (baseline only,
# no API calls) and --sample N (process only N listings).

import os
import sys
import argparse
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # must run before OpenAI client initialises

import pandas as pd

from src.taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES
from src.baseline import rule_based_baseline, random_baseline
from src.llm_classifier import classify_batch
from src.export import export_to_excel, export_to_csv


# Batch size for LLM API calls: 8 listings per call balances token usage
# against API overhead. Larger batches risk hitting context limits; smaller
# batches waste API calls on per-request overhead.
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# 1. Load and clean CSV data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Read sample_listings.csv into a cleaned DataFrame.

    The CSV is pre-cleaned during Phase 1 (standardised column names,
    ISO dates, no special delimiters). Works with bare pd.read_csv().
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("")
    df = df[df["company_name"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    print(f"[load] Loaded {len(df)} rows from {path}")
    return df


# ---------------------------------------------------------------------------
# 2. Build programme text for LLM input
# ---------------------------------------------------------------------------

def build_programme_text(row: pd.Series) -> str:
    """Concatenate row fields into a text block for classification.

    Reuses the pattern from src/test_one_row.py but adapted for the
    standardised column names in sample_listings.csv.
    """
    return (
        f"Company: {row.get('company_name', '')}\n"
        f"Programme: {row.get('programme_name', '')}\n"
        f"Opening date: {row.get('opening_date', '')}\n"
        f"Closing date: {row.get('closing_date', '')}\n"
        f"Latest stage: {row.get('latest_stage', '')}"
    )


# ---------------------------------------------------------------------------
# 3. Convert DataFrame rows to list of dicts for classifiers
# ---------------------------------------------------------------------------

def df_to_listings(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of dicts for the classifiers."""
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# 4. Run LLM classification in batches
# ---------------------------------------------------------------------------

def run_llm_classification(listings: list[dict]) -> list[dict]:
    """Classify all listings via the LLM in batches of BATCH_SIZE.

    Returns a flat list of classification dicts in the same order as input.
    """
    all_results = []
    total = len(listings)

    for i in range(0, total, BATCH_SIZE):
        batch = listings[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  [llm] Batch {batch_num}/{total_batches} ({len(batch)} listings)")
        results = classify_batch(batch)
        all_results.extend(results)

    print(f"  [llm] Done. {len(all_results)} listings classified.")
    return all_results


# ---------------------------------------------------------------------------
# 5. Merge classification results into the DataFrame
# ---------------------------------------------------------------------------

def merge_results(df: pd.DataFrame, baseline_results: list[dict],
                  llm_results: list[dict] = None) -> pd.DataFrame:
    """Add baseline and LLM classification columns to the DataFrame."""
    # Baseline columns
    for key in ["firm_type", "firm_type_confidence", "role_function",
                "role_function_confidence", "programme_status",
                "programme_status_confidence"]:
        df[f"baseline_{key}"] = [r.get(key, "") for r in baseline_results]

    # LLM columns (only if we have results)
    if llm_results:
        for key in ["firm_type", "firm_type_confidence", "firm_type_rationale",
                     "role_function", "role_function_confidence", "role_function_rationale",
                     "programme_status", "programme_status_confidence",
                     "programme_status_rationale",
                     "model_used", "escalated"]:
            df[f"llm_{key}"] = [r.get(key, "") for r in llm_results]

    return df


# ---------------------------------------------------------------------------
# 6. Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """Main pipeline: load -> baseline -> (optional) LLM -> merge -> export."""

    api_key = os.getenv("OPENAI_API_KEY", "")
    has_api_key = bool(api_key) and api_key != "API_key_here"

    # --- Step 1: Load data ---
    print(f"\n[1/5] Loading data from {args.data_path}")
    df = load_data(args.data_path)

    # Apply --sample limit if specified
    if args.sample and args.sample < len(df):
        df = df.head(args.sample).reset_index(drop=True)
        print(f"  [sample] Limited to {args.sample} rows")

    listings = df_to_listings(df)

    # --- Step 2: Rule-based baseline ---
    print(f"\n[2/5] Running rule-based baseline")
    baseline_results = rule_based_baseline(listings)

    # --- Step 3: LLM classification ---
    llm_results = None
    if args.dry_run:
        print(f"\n[3/5] --dry-run: skipping LLM classification")
    elif not has_api_key:
        print(f"\n[3/5] No valid OPENAI_API_KEY in .env - skipping LLM classification")
        print(f"  Set your key in .env: OPENAI_API_KEY=sk-...")
    else:
        print(f"\n[3/5] Running LLM classification (gpt-4o-mini -> gpt-4o escalation)")
        try:
            llm_results = run_llm_classification(listings)
        except Exception as e:
            print(f"  [ERROR] LLM classification failed: {e}")
            print(f"  Falling back to baseline-only results.")
            llm_results = None

    # --- Step 4: Merge results ---
    print(f"\n[4/5] Merging results")
    df = merge_results(df, baseline_results, llm_results)

    # --- Step 5: Export ---
    print(f"\n[5/5] Exporting results")
    try:
        filepath = export_to_excel(df)
    except Exception as e:
        print(f"  [WARNING] Excel export failed ({e}), falling back to CSV")
        filepath = export_to_csv(df)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    escalated_count = sum(1 for r in (llm_results or []) if r.get("escalated"))
    print(f"  Rows processed:       {len(df)}")
    print(f"  Baseline classified:  {len(baseline_results)}")
    print(f"  LLM classified:       {len(llm_results) if llm_results else 0}")
    print(f"  Escalated to gpt-4o:  {escalated_count}")
    print(f"  Dry run:              {args.dry_run}")
    print(f"  Output:               {filepath}")
    print(f"{'=' * 60}")

    return filepath


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COM2019 Finance Internship Classifier Pipeline"
    )
    parser.add_argument(
        "--data-path",
        default="data/sample_listings.csv",
        help="Path to the input CSV file (default: data/sample_listings.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run baseline only, skip LLM classification (no API calls)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only the first N listings (useful for testing)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after classification (requires gold standard data)",
    )
    args = parser.parse_args()

    filepath = run_pipeline(args)

    # Optional evaluation (Phase 4)
    if args.evaluate:
        try:
            from src.evaluate import run_full_evaluation
            run_full_evaluation()
        except ImportError:
            print("[evaluate] evaluate.py not yet available, skipping.")
        except Exception as e:
            print(f"[evaluate] Evaluation failed: {e}")
