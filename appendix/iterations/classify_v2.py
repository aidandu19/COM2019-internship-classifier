# classify_v2.py - Prompt V2 classification for programme_status iteration
#
# Runs the LLM classification with a revised system prompt that explicitly
# instructs the model to prioritise Trackr's "Latest Stage" text over
# calendar dates for programme_status. Saves results separately as V2.
#
# This is for COM2019 Task 6c: demonstrating iterative prompt engineering.

import os
import sys
import json

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from src.taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES
from src.schema import OPENAI_RESPONSE_FORMAT, BATCH_VALIDATOR

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOLD_STANDARD_PATH = "data/gold_standard.json"
V2_RESULTS_PATH = "data/llm_results_v2.json"
BATCH_SIZE = 8
MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# V2 SYSTEM PROMPT - adds explicit stage-text priority for programme_status
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V2 = """You are a UK finance internship classification engine built for university students navigating summer internship applications. You have deep knowledge of the UK and global financial services landscape: bulge brackets, elite boutiques, buy-side firms, asset managers, hedge funds, quant/prop shops, and consulting firms.

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

### IMPORTANT: Programme Status Classification Rules

For programme status, you MUST prioritise the "Latest Stage" text over calendar dates. The stage text is the most reliable indicator of a programme's current status because it reflects real-time recruiter updates on the Trackr platform, whereas dates may be stale or aspirational.

Stage text mappings (highest priority):
- "Offers Out" = CLOSED (the firm has already made offers, regardless of what the dates say)
- "Closed" = CLOSED
- "Closed - Offers Out" = CLOSED
- "Open" = OPEN
- "Not Yet Open" = NOT_YET_OPEN
- Any text mentioning "offer", "rejected", "withdrawn" = CLOSED
- Any text mentioning "accepting", "apply now" = OPEN

If no stage text is provided (empty or missing "Latest stage" field), THEN and only then should you infer status from opening/closing dates.

If neither stage text nor dates provide clear evidence, return UNKNOWN.

DO NOT predict NOT_YET_OPEN based solely on a future opening date if no stage text confirms this. When in doubt between NOT_YET_OPEN and UNKNOWN, prefer UNKNOWN.""".format(
    firm_types=", ".join(FIRM_TYPES),
    role_functions=", ".join(ROLE_FUNCTIONS),
    programme_statuses=", ".join(PROGRAMME_STATUSES),
)

# Same few-shot examples as V1 (status outputs match gold standard conventions)
FEW_SHOT_EXAMPLES = """
Here are three worked examples showing the expected output format and reasoning:

Example input 1:
Company: Barclays
Programme: Summer Internship Programme 2026
Opening date: 2025-08-25
Closing date:
Latest stage: Offers Out

Example output 1:
{"classifications": [{"company_name": "Barclays", "firm_type": "BULGE_BRACKET", "firm_type_confidence": 0.97, "firm_type_rationale": "Barclays is a globally recognised bulge bracket investment bank.", "role_function": "INVESTMENT_BANKING", "role_function_confidence": 0.6, "role_function_rationale": "Generic 'Summer Internship Programme' title does not specify division, but Barclays internships are most commonly IB-focused. Moderate confidence.", "programme_status": "CLOSED", "programme_status_confidence": 0.95, "programme_status_rationale": "Latest stage is 'Offers Out', indicating the application cycle has concluded."}]}

Example input 2:
Company: Lazard
Programme: 2026 Summer Internship
Opening date: 2025-09-19
Closing date: 2025-10-14
Latest stage: Offers Out

Example output 2:
{"classifications": [{"company_name": "Lazard", "firm_type": "ELITE_BOUTIQUE", "firm_type_confidence": 0.95, "firm_type_rationale": "Lazard is an elite boutique advisory firm. While it also has an asset management arm, the core classification is elite boutique.", "role_function": "INVESTMENT_BANKING", "role_function_confidence": 0.65, "role_function_rationale": "Lazard's summer internships are typically advisory/IB-focused, though the generic title leaves some ambiguity with asset management roles.", "programme_status": "CLOSED", "programme_status_confidence": 0.95, "programme_status_rationale": "Offers Out indicates recruitment cycle is complete and closing date has passed."}]}

Example input 3:
Company: Private Equity Insights
Programme: Global Internship Program
Opening date: 2025-10-27
Closing date:
Latest stage:

Example output 3:
{"classifications": [{"company_name": "Private Equity Insights", "firm_type": "OTHER", "firm_type_confidence": 0.85, "firm_type_rationale": "Private Equity Insights is a media/events company covering the PE industry, not a financial services firm itself.", "role_function": "OTHER", "role_function_confidence": 0.8, "role_function_rationale": "This is likely a media, research, or events role rather than a front-office finance function.", "programme_status": "UNKNOWN", "programme_status_confidence": 0.5, "programme_status_rationale": "No closing date or latest stage information provided. Cannot determine status."}]}
"""


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
        response_format=OPENAI_RESPONSE_FORMAT,
    )
    return response.choices[0].message.content


def classify_batch_v2(listings):
    """Classify a batch with V2 prompt. No caching - always calls API."""
    user_msg = build_batch_message(listings)
    system_msg = SYSTEM_PROMPT_V2 + "\n\n" + FEW_SHOT_EXAMPLES

    raw_text = call_api(system_msg, user_msg)
    parsed = json.loads(raw_text)
    errors = list(BATCH_VALIDATOR.iter_errors(parsed))
    if errors:
        error_msgs = [f"{e.path}: {e.message}" for e in errors[:5]]
        raise ValueError(f"Schema validation failed: {error_msgs}")

    classifications = parsed["classifications"]
    for clf in classifications:
        clf["model_used"] = MODEL
        clf["escalated"] = False
    return classifications


def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "API_key_here":
        print("ERROR: No valid OPENAI_API_KEY found in .env")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("PROMPT V2 CLASSIFICATION (stage-text priority)")
    print(f"{'=' * 60}")

    # Load gold standard
    print(f"\n[1/3] Loading gold standard from {GOLD_STANDARD_PATH}...")
    listings = load_gold_standard()
    print(f"  Loaded {len(listings)} entries")

    # Classify with V2
    print(f"\n[2/3] Running V2 classification ({len(listings)} entries, batch size {BATCH_SIZE})...")
    all_results = []
    total_batches = (len(listings) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(listings), BATCH_SIZE):
        batch = listings[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} listings)...")
        try:
            results = classify_batch_v2(batch)
            all_results.extend(results)
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

    # Save
    print(f"\n[3/3] Saving V2 results...")
    with open(V2_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {len(all_results)} entries to {V2_RESULTS_PATH}")

    # Save the V2 prompt for report appendix
    v2_prompt_path = "data/v2_system_prompt.txt"
    with open(v2_prompt_path, "w") as f:
        f.write(SYSTEM_PROMPT_V2)
    print(f"  Saved V2 system prompt to {v2_prompt_path}")

    print(f"\n{'=' * 60}")
    print("V2 CLASSIFICATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total classified: {len(all_results)}")
    print(f"  Output: {V2_RESULTS_PATH}")
    print(f"  Prompt: {v2_prompt_path}")


if __name__ == "__main__":
    main()
