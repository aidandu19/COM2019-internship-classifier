# classify_schema_b.py - Schema B classification (rationale-first ordering)
#
# Tests whether reversing the field ordering in the JSON schema (forcing
# the model to generate rationale BEFORE the classification label) affects
# confidence calibration. Uses V1 system prompt to isolate the schema variable.
#
# This is for COM2019 Task 6d: schema ordering experiment.

import os
import sys
import json

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from src.taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES
from src.schema import BATCH_VALIDATOR

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOLD_STANDARD_PATH = "data/gold_standard.json"
SCHEMA_B_RESULTS_PATH = "data/llm_results_schema_b.json"
BATCH_SIZE = 8
MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# V1 System Prompt (same as original - isolating schema variable only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a UK finance internship classification engine built for university students navigating summer internship applications. You have deep knowledge of the UK and global financial services landscape: bulge brackets, elite boutiques, buy-side firms, asset managers, hedge funds, quant/prop shops, and consulting firms.

For each listing provided, classify it along three dimensions:

1. **Firm Type** - assign exactly ONE from: {firm_types}
2. **Role Function** - assign exactly ONE from: {role_functions}
3. **Programme Status** - assign exactly ONE from: {programme_statuses}

Rules:
- Use UNKNOWN only when the information is genuinely insufficient.
- Confidence scores must be between 0.0 and 1.0. Use 0.9+ only when classification is unambiguous.
- Rationale strings must be grounded in the input text, not general knowledge.
- Return one classification object per listing in the batch, in the same order as the input.
- company_name in each classification must match the company from the input listing.

IMPORTANT: For each dimension, you MUST provide your reasoning/rationale FIRST, then the classification label, then your confidence score. Think through the rationale before committing to a label.""".format(
    firm_types=", ".join(FIRM_TYPES),
    role_functions=", ".join(ROLE_FUNCTIONS),
    programme_statuses=", ".join(PROGRAMME_STATUSES),
)

FEW_SHOT_EXAMPLES = """
Here are three worked examples showing the expected output format and reasoning:

Example input 1:
Company: Barclays
Programme: Summer Internship Programme 2026
Opening date: 2025-08-25
Closing date:
Latest stage: Offers Out

Example output 1:
{"classifications": [{"company_name": "Barclays", "firm_type_rationale": "Barclays is a globally recognised bulge bracket investment bank.", "firm_type": "BULGE_BRACKET", "firm_type_confidence": 0.97, "role_function_rationale": "Generic 'Summer Internship Programme' title does not specify division, but Barclays internships are most commonly IB-focused. Moderate confidence.", "role_function": "INVESTMENT_BANKING", "role_function_confidence": 0.6, "programme_status_rationale": "Latest stage is 'Offers Out', indicating the application cycle has concluded.", "programme_status": "CLOSED", "programme_status_confidence": 0.95}]}

Example input 2:
Company: Lazard
Programme: 2026 Summer Internship
Opening date: 2025-09-19
Closing date: 2025-10-14
Latest stage: Offers Out

Example output 2:
{"classifications": [{"company_name": "Lazard", "firm_type_rationale": "Lazard is an elite boutique advisory firm. While it also has an asset management arm, the core classification is elite boutique.", "firm_type": "ELITE_BOUTIQUE", "firm_type_confidence": 0.95, "role_function_rationale": "Lazard's summer internships are typically advisory/IB-focused, though the generic title leaves some ambiguity with asset management roles.", "role_function": "INVESTMENT_BANKING", "role_function_confidence": 0.65, "programme_status_rationale": "Offers Out indicates recruitment cycle is complete and closing date has passed.", "programme_status": "CLOSED", "programme_status_confidence": 0.95}]}

Example input 3:
Company: Private Equity Insights
Programme: Global Internship Program
Opening date: 2025-10-27
Closing date:
Latest stage:

Example output 3:
{"classifications": [{"company_name": "Private Equity Insights", "firm_type_rationale": "Private Equity Insights is a media/events company covering the PE industry, not a financial services firm itself.", "firm_type": "OTHER", "firm_type_confidence": 0.85, "role_function_rationale": "This is likely a media, research, or events role rather than a front-office finance function.", "role_function": "OTHER", "role_function_confidence": 0.8, "programme_status_rationale": "No closing date or latest stage information provided. Cannot determine status.", "programme_status": "UNKNOWN", "programme_status_confidence": 0.5}]}
"""


# ---------------------------------------------------------------------------
# Schema B: rationale-first field ordering
# ---------------------------------------------------------------------------

SCHEMA_B_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "internship_classifications",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["classifications"],
            "additionalProperties": False,
            "properties": {
                "classifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "company_name",
                            "firm_type_rationale",
                            "firm_type",
                            "firm_type_confidence",
                            "role_function_rationale",
                            "role_function",
                            "role_function_confidence",
                            "programme_status_rationale",
                            "programme_status",
                            "programme_status_confidence",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "company_name": {"type": "string"},
                            # Firm type: rationale -> label -> confidence
                            "firm_type_rationale": {"type": "string"},
                            "firm_type": {
                                "type": "string",
                                "enum": FIRM_TYPES,
                            },
                            "firm_type_confidence": {"type": "number"},
                            # Role function: rationale -> label -> confidence
                            "role_function_rationale": {"type": "string"},
                            "role_function": {
                                "type": "string",
                                "enum": ROLE_FUNCTIONS,
                            },
                            "role_function_confidence": {"type": "number"},
                            # Programme status: rationale -> label -> confidence
                            "programme_status_rationale": {"type": "string"},
                            "programme_status": {
                                "type": "string",
                                "enum": PROGRAMME_STATUSES,
                            },
                            "programme_status_confidence": {"type": "number"},
                        },
                    },
                }
            },
        },
    },
}


def load_gold_standard():
    with open(GOLD_STANDARD_PATH) as f:
        data = json.load(f)
    listings = []
    for g in data:
        listings.append({
            "company_name": g.get("company_name", ""),
            "programme_name": g.get("programme") or g.get("programme_name", ""),
            "opening_date": g.get("opening_date", ""),
            "closing_date": g.get("closing_date") or g.get("closing", ""),
            "latest_stage": g.get("latest_stage") or g.get("stage", ""),
        })
    return listings


def build_batch_message(listings):
    parts = []
    for i, listing in enumerate(listings, 1):
        text = (
            f"Listing {i}:\n"
            f"Company: {listing.get('company_name', '')}\n"
            f"Programme: {listing.get('programme_name', '')}\n"
            f"Opening date: {listing.get('opening_date', '')}\n"
            f"Closing date: {listing.get('closing_date', '')}\n"
            f"Latest stage: {listing.get('latest_stage', '')}"
        )
        parts.append(text)
    header = f"Classify the following {len(listings)} internship listing(s):\n\n"
    return header + "\n\n---\n\n".join(parts)


def call_api(system_msg, user_msg):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        response_format=SCHEMA_B_RESPONSE_FORMAT,
    )
    raw = response.choices[0].message.content
    return raw


def classify_batch_schema_b(listings):
    """Classify with Schema B (rationale-first). No caching."""
    user_msg = build_batch_message(listings)
    system_msg = SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES

    raw_text = call_api(system_msg, user_msg)
    parsed = json.loads(raw_text)

    classifications = parsed["classifications"]
    for clf in classifications:
        clf["model_used"] = MODEL
        clf["escalated"] = False
    return classifications, raw_text


def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "API_key_here":
        print("ERROR: No valid OPENAI_API_KEY found in .env")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("SCHEMA B CLASSIFICATION (rationale-first ordering)")
    print(f"{'=' * 60}")

    # Load gold standard
    print(f"\n[1/4] Loading gold standard from {GOLD_STANDARD_PATH}...")
    listings = load_gold_standard()
    print(f"  Loaded {len(listings)} entries")

    # Verify field ordering in first API response
    print(f"\n[2/4] Running Schema B classification ({len(listings)} entries, batch size {BATCH_SIZE})...")
    all_results = []
    raw_responses = []
    total_batches = (len(listings) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(listings), BATCH_SIZE):
        batch = listings[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} listings)...")
        try:
            results, raw = classify_batch_schema_b(batch)
            all_results.extend(results)
            raw_responses.append(raw)
        except Exception as e:
            print(f"  ERROR in batch {batch_num}: {e}")
            for listing in batch:
                all_results.append({
                    "company_name": listing.get("company_name", ""),
                    "firm_type": "UNKNOWN", "firm_type_confidence": 0.0,
                    "firm_type_rationale": "API error - fallback",
                    "role_function": "UNKNOWN", "role_function_confidence": 0.0,
                    "role_function_rationale": "API error - fallback",
                    "programme_status": "UNKNOWN", "programme_status_confidence": 0.0,
                    "programme_status_rationale": "API error - fallback",
                    "model_used": "none", "escalated": False,
                })

    # Verify field ordering from raw response
    print(f"\n[3/4] Verifying field ordering in raw API response...")
    if raw_responses:
        first_raw = raw_responses[0]
        # Check if rationale appears before label in raw JSON string
        parsed_first = json.loads(first_raw)
        first_item = parsed_first["classifications"][0]
        keys = list(first_item.keys())
        print(f"  Field order in response: {keys}")

        # Check ordering
        ft_rat_idx = keys.index("firm_type_rationale") if "firm_type_rationale" in keys else -1
        ft_idx = keys.index("firm_type") if "firm_type" in keys else -1
        if ft_rat_idx < ft_idx:
            print("  CONFIRMED: rationale appears BEFORE label (Schema B ordering respected)")
        else:
            print("  WARNING: rationale appears AFTER label (API may have reordered fields)")

    # Save
    print(f"\n[4/4] Saving Schema B results...")
    with open(SCHEMA_B_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {len(all_results)} entries to {SCHEMA_B_RESULTS_PATH}")

    # Save first raw response for verification
    raw_sample_path = "data/schema_b_raw_sample.json"
    if raw_responses:
        with open(raw_sample_path, "w") as f:
            f.write(raw_responses[0])
        print(f"  Saved raw response sample to {raw_sample_path}")

    print(f"\n{'=' * 60}")
    print("SCHEMA B CLASSIFICATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total classified: {len(all_results)}")
    print(f"  Output: {SCHEMA_B_RESULTS_PATH}")


if __name__ == "__main__":
    main()
