"""
Standalone gold standard classifier for COM2019.

Reads data/gold_standard.json, classifies each entry via OpenAI API,
writes data/llm_results.json in the exact format src/evaluate.py expects.

Usage:
    python classify_gold_standard_standalone.py
    python classify_gold_standard_standalone.py --model gpt-4o --temp 0.0
    python classify_gold_standard_standalone.py --sample 10
"""

import os
import json
import argparse
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────
# Change these as needed before running.

DEFAULT_MODEL = "gpt-4o"           # primary model for all classifications
ESCALATION_MODEL = "gpt-4o"       # model for re-classifying low-confidence items
ESCALATION_THRESHOLD = 0.7         # any confidence below this triggers escalation
TEMPERATURE = 0.0                  # 0.0 = deterministic, raise for more variation
BATCH_SIZE = 8                     # listings per API call
MAX_RETRIES = 3                    # retry attempts on API failure
RETRY_DELAY = 5                    # seconds between retries

GOLD_STANDARD_PATH = "data/gold_standard.json"
OUTPUT_PATH = "data/llm_results.json"

# ── TAXONOMY (must match src/taxonomy.py exactly) ───────────────────

FIRM_TYPES = [
    "BULGE_BRACKET", "ELITE_BOUTIQUE", "MIDDLE_MARKET_IB",
    "BUY_SIDE_PE", "ASSET_MANAGEMENT", "HEDGE_FUND",
    "QUANT_PROP", "ACCOUNTING", "CONSULTING", "OTHER", "UNKNOWN"
]

ROLE_FUNCTIONS = [
    "INVESTMENT_BANKING", "MARKETS_TRADING", "RESEARCH",
    "QUANT", "CONSULTING", "OPERATIONS", "TECH_DATA",
    "RISK_COMPLIANCE", "OTHER", "UNKNOWN"
]

PROGRAMME_STATUSES = ["OPEN", "CLOSED", "NOT_YET_OPEN", "UNKNOWN"]

# ── SYSTEM PROMPT ───────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a UK finance internship classification engine built for university students navigating summer internship applications. You have deep knowledge of the UK and global financial services landscape: bulge brackets, elite boutiques, buy-side firms, asset managers, hedge funds, quant/prop shops, and consulting firms.

For each listing provided, classify it along three dimensions:

1. **Firm Type** - assign exactly ONE from: {", ".join(FIRM_TYPES)}
2. **Role Function** - assign exactly ONE from: {", ".join(ROLE_FUNCTIONS)}
3. **Programme Status** - assign exactly ONE from: {", ".join(PROGRAMME_STATUSES)}

Rules:
- Use UNKNOWN only when the information is genuinely insufficient.
- Confidence scores must be between 0.0 and 1.0. Use 0.9+ only when classification is unambiguous.
- Rationale strings must be grounded in the input text, not general knowledge.
- Return one classification object per listing in the batch, in the same order as the input.
- company_name in each classification must match the company from the input listing.

Here are three worked examples showing the expected output format and reasoning:

Example input 1:
Company: Barclays
Programme: Summer Internship Programme 2026
Opening date: 2025-08-25
Closing date:
Latest stage: Offers Out

Example output 1:
{{"classifications": [{{"company_name": "Barclays", "firm_type": "BULGE_BRACKET", "firm_type_confidence": 0.97, "firm_type_rationale": "Barclays is a globally recognised bulge bracket investment bank.", "role_function": "INVESTMENT_BANKING", "role_function_confidence": 0.6, "role_function_rationale": "Generic 'Summer Internship Programme' title does not specify division, but Barclays internships are most commonly IB-focused. Moderate confidence.", "programme_status": "CLOSED", "programme_status_confidence": 0.95, "programme_status_rationale": "Latest stage is 'Offers Out', indicating the application cycle has concluded."}}]}}

Example input 2:
Company: Lazard
Programme: 2026 Summer Internship
Opening date: 2025-09-19
Closing date: 2025-10-14
Latest stage: Offers Out

Example output 2:
{{"classifications": [{{"company_name": "Lazard", "firm_type": "ELITE_BOUTIQUE", "firm_type_confidence": 0.95, "firm_type_rationale": "Lazard is an elite boutique advisory firm. While it also has an asset management arm, the core classification is elite boutique.", "role_function": "INVESTMENT_BANKING", "role_function_confidence": 0.65, "role_function_rationale": "Lazard's summer internships are typically advisory/IB-focused, though the generic title leaves some ambiguity with asset management roles.", "programme_status": "CLOSED", "programme_status_confidence": 0.95, "programme_status_rationale": "Offers Out indicates recruitment cycle is complete and closing date has passed."}}]}}

Example input 3:
Company: Private Equity Insights
Programme: Global Internship Program
Opening date: 2025-10-27
Closing date:
Latest stage:

Example output 3:
{{"classifications": [{{"company_name": "Private Equity Insights", "firm_type": "OTHER", "firm_type_confidence": 0.85, "firm_type_rationale": "Private Equity Insights is a media/events company covering the PE industry, not a financial services firm itself.", "role_function": "OTHER", "role_function_confidence": 0.8, "role_function_rationale": "This is likely a media, research, or events role rather than a front-office finance function.", "programme_status": "UNKNOWN", "programme_status_confidence": 0.5, "programme_status_rationale": "No closing date or latest stage information provided. Cannot determine status."}}]}}"""

# ── OPENAI RESPONSE FORMAT (structured outputs) ─────────────────────

RESPONSE_FORMAT = {
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
                            "firm_type", "firm_type_confidence", "firm_type_rationale",
                            "role_function", "role_function_confidence", "role_function_rationale",
                            "programme_status", "programme_status_confidence", "programme_status_rationale",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "company_name": {"type": "string"},
                            "firm_type": {"type": "string", "enum": FIRM_TYPES},
                            "firm_type_confidence": {"type": "number"},
                            "firm_type_rationale": {"type": "string"},
                            "role_function": {"type": "string", "enum": ROLE_FUNCTIONS},
                            "role_function_confidence": {"type": "number"},
                            "role_function_rationale": {"type": "string"},
                            "programme_status": {"type": "string", "enum": PROGRAMME_STATUSES},
                            "programme_status_confidence": {"type": "number"},
                            "programme_status_rationale": {"type": "string"},
                        },
                    },
                }
            },
        },
    },
}


# ── HELPERS ──────────────────────────────────────────────────────────

def build_batch_message(listings):
    """Format a batch of listings into a user message string."""
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
    return f"Classify the following {len(listings)} internship listing(s):\n\n" + "\n\n---\n\n".join(parts)


def call_api(client, user_msg, model, temperature):
    """Call OpenAI Chat Completions with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                response_format=RESPONSE_FORMAT,
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            return parsed["classifications"]
        except Exception as e:
            print(f"    Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def needs_escalation(clf):
    """True if any confidence score is below the escalation threshold."""
    for key in ("firm_type_confidence", "role_function_confidence", "programme_status_confidence"):
        try:
            if float(clf.get(key, 0.0)) < ESCALATION_THRESHOLD:
                return True
        except (ValueError, TypeError):
            return True
    return False


def fallback_result(listing):
    """Return UNKNOWN for everything when API fails completely."""
    return {
        "company_name": listing.get("company_name", ""),
        "firm_type": "UNKNOWN",
        "firm_type_confidence": 0.0,
        "firm_type_rationale": "API unavailable - fallback result",
        "role_function": "UNKNOWN",
        "role_function_confidence": 0.0,
        "role_function_rationale": "API unavailable - fallback result",
        "programme_status": "UNKNOWN",
        "programme_status_confidence": 0.0,
        "programme_status_rationale": "API unavailable - fallback result",
        "model_used": "none",
        "escalated": False,
    }


# ── MAIN ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classify gold standard via OpenAI API")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--escalation-model", default=ESCALATION_MODEL, help=f"Escalation model (default: {ESCALATION_MODEL})")
    parser.add_argument("--temp", type=float, default=TEMPERATURE, help=f"Temperature (default: {TEMPERATURE})")
    parser.add_argument("--sample", type=int, default=None, help="Only classify first N entries")
    parser.add_argument("--no-escalation", action="store_true", help="Skip escalation step")
    parser.add_argument("--threshold", type=float, default=ESCALATION_THRESHOLD, help=f"Escalation threshold (default: {ESCALATION_THRESHOLD})")
    parser.add_argument("--output", default=OUTPUT_PATH, help=f"Output path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    # Init client
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "API_key_here":
        print("ERROR: Set OPENAI_API_KEY in .env file")
        return
    client = OpenAI(api_key=api_key)

    # Load gold standard
    print(f"\n[1/4] Loading {GOLD_STANDARD_PATH}...")
    with open(GOLD_STANDARD_PATH) as f:
        gold_data = json.load(f)

    listings = []
    for g in gold_data:
        listings.append({
            "company_name": g.get("company_name", ""),
            "programme_name": g.get("programme_name") or g.get("programme", ""),
            "opening_date": g.get("opening_date", ""),
            "closing_date": g.get("closing_date") or g.get("closing", ""),
            "latest_stage": g.get("latest_stage") or g.get("stage", ""),
        })

    if args.sample:
        listings = listings[:args.sample]

    print(f"  {len(listings)} entries loaded")
    print(f"  Model: {args.model} | Temperature: {args.temp} | Escalation: {'off' if args.no_escalation else args.escalation_model}")

    # Classify in batches
    print(f"\n[2/4] Classifying ({BATCH_SIZE} per batch)...")
    all_results = []
    total_batches = (len(listings) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(listings), BATCH_SIZE):
        batch = listings[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} listings)...")

        user_msg = build_batch_message(batch)
        results = call_api(client, user_msg, args.model, args.temp)

        if results is None:
            print(f"    FAILED - using fallback for {len(batch)} listings")
            results = [fallback_result(l) for l in batch]

        # Tag with model info
        for clf in results:
            clf["model_used"] = args.model
            clf["escalated"] = False

        all_results.extend(results)

    # Escalation pass
    if not args.no_escalation:
        esc_indices = [i for i, clf in enumerate(all_results) if needs_escalation(clf)]
        if esc_indices:
            print(f"\n[3/4] Escalating {len(esc_indices)}/{len(all_results)} low-confidence results to {args.escalation_model}...")

            # Process escalation in batches too
            for batch_start in range(0, len(esc_indices), BATCH_SIZE):
                batch_indices = esc_indices[batch_start:batch_start + BATCH_SIZE]
                batch_listings = [listings[idx] for idx in batch_indices]
                batch_num = (batch_start // BATCH_SIZE) + 1
                total_esc_batches = (len(esc_indices) + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"  Escalation batch {batch_num}/{total_esc_batches}...")

                user_msg = build_batch_message(batch_listings)
                esc_results = call_api(client, user_msg, args.escalation_model, args.temp)

                if esc_results:
                    for idx, esc_clf in zip(batch_indices, esc_results):
                        esc_clf["model_used"] = args.escalation_model
                        esc_clf["escalated"] = True
                        all_results[idx] = esc_clf
                else:
                    print(f"    Escalation failed, keeping original results")
        else:
            print(f"\n[3/4] No escalation needed (all above {args.threshold})")
    else:
        print(f"\n[3/4] Escalation skipped (--no-escalation)")

    # Validate
    errors = 0
    for r in all_results:
        if r.get("firm_type") not in FIRM_TYPES:
            errors += 1
        if r.get("role_function") not in ROLE_FUNCTIONS:
            errors += 1
        if r.get("programme_status") not in PROGRAMME_STATUSES:
            errors += 1

    # Save
    print(f"\n[4/4] Saving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    escalated = sum(1 for r in all_results if r.get("escalated"))
    models_used = {}
    for r in all_results:
        m = r.get("model_used", "unknown")
        models_used[m] = models_used.get(m, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"  Total classified:    {len(all_results)}")
    print(f"  Escalated:           {escalated}")
    print(f"  Validation errors:   {errors}")
    print(f"  Models used:         {models_used}")
    print(f"  Output:              {args.output}")
    print(f"{'=' * 60}")
    print(f"\nNext: python -m src.evaluate")


if __name__ == "__main__":
    main()
