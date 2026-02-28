# llm_classifier.py - LLM-based internship classifier using OpenAI Chat Completions
#
# Uses gpt-4o-mini with structured outputs (response_format) to classify
# finance internship listings by firm type, role function, and programme status.
# Supports batching (8-10 listings per API call), local JSON caching,
# retry logic via tenacity, and a repair loop for schema validation failures.

import os
import json
import hashlib

from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES
from .schema import OPENAI_RESPONSE_FORMAT, BATCH_VALIDATOR

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CACHE_DIR = os.path.join("data", "cache")

# ---------------------------------------------------------------------------
# System prompt: defines the classification engine persona
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
- company_name in each classification must match the company from the input listing.""".format(
    firm_types=", ".join(FIRM_TYPES),
    role_functions=", ".join(ROLE_FUNCTIONS),
    programme_statuses=", ".join(PROGRAMME_STATUSES),
)

# ---------------------------------------------------------------------------
# Few-shot examples embedded in the system prompt
# ---------------------------------------------------------------------------
# Three examples covering: clear-cut (Barclays), ambiguous/dual (Lazard),
# and edge case (Private Equity Insights).

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


# ---------------------------------------------------------------------------
# Caching: hash input text, store API responses locally
# ---------------------------------------------------------------------------

def _cache_key(text: str) -> str:
    """Generate a SHA-256 hash of the input text for cache lookup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_from_cache(key: str):
    """Load a cached API response if it exists. Returns None on miss."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_to_cache(key: str, data: dict):
    """Save an API response to the local cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# API call with tenacity retry
# ---------------------------------------------------------------------------

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def _call_chat_completions(system_msg: str, user_msg: str) -> str:
    """Call OpenAI Chat Completions API with structured output format.

    Uses response_format with json_schema to enforce the output structure
    directly at the API level (ILO 6 requirement). The model is constrained
    to return valid JSON matching our schema.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        response_format=OPENAI_RESPONSE_FORMAT,
    )
    # TODO: model escalation - for low-confidence results, reclassify with
    # gpt-4o here. Would check each classification's confidence scores and
    # re-submit only the low-confidence listings to the stronger model.
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Parse and validate response
# ---------------------------------------------------------------------------

def _parse_and_validate(raw_text: str) -> dict:
    """Parse JSON response and validate against the batch schema.

    If validation fails, raises ValueError with details for the repair loop.
    """
    parsed = json.loads(raw_text)
    errors = list(BATCH_VALIDATOR.iter_errors(parsed))
    if errors:
        error_msgs = [f"{e.path}: {e.message}" for e in errors[:5]]
        raise ValueError(f"Schema validation failed: {error_msgs}")
    return parsed


# ---------------------------------------------------------------------------
# Repair loop: if structured output still fails validation, ask model to fix
# ---------------------------------------------------------------------------

REPAIR_SYSTEM = """You are a JSON repair assistant. The previous API response failed schema validation. Fix the JSON so it matches the required format exactly. Return ONLY the corrected JSON, no explanation."""

def _repair_response(bad_output: str, error_msg: str) -> dict:
    """Send the malformed output back to the model for repair."""
    repair_user = f"""The following JSON output failed validation:

{bad_output}

Error: {error_msg}

Fix it to match the required schema. Each classification must have:
- company_name (string)
- firm_type (one of: {', '.join(FIRM_TYPES)})
- firm_type_confidence (number 0-1)
- firm_type_rationale (string)
- role_function (one of: {', '.join(ROLE_FUNCTIONS)})
- role_function_confidence (number 0-1)
- role_function_rationale (string)
- programme_status (one of: {', '.join(PROGRAMME_STATUSES)})
- programme_status_confidence (number 0-1)
- programme_status_rationale (string)

Wrapped in: {{"classifications": [...]}}"""

    repaired_text = _call_chat_completions(REPAIR_SYSTEM, repair_user)
    return _parse_and_validate(repaired_text)


# ---------------------------------------------------------------------------
# Build user message for a batch of listings
# ---------------------------------------------------------------------------

def _build_batch_message(listings: list[dict]) -> str:
    """Format a batch of listing dicts into a user message string.

    Each listing dict should have keys: company_name, programme_name,
    opening_date, closing_date, latest_stage (and optionally process, notes).
    """
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
        # Include process and notes if available (extra signal for classification)
        if listing.get("process"):
            text += f"\nProcess: {listing['process']}"
        if listing.get("notes"):
            text += f"\nNotes: {listing['notes']}"
        parts.append(text)

    header = f"Classify the following {len(listings)} internship listing(s):\n\n"
    return header + "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Fallback result for when the API fails after all retries
# ---------------------------------------------------------------------------

def _fallback_result(listing: dict) -> dict:
    """Return a safe fallback classification when the API is unreachable.

    Marked with source='fallback' so downstream code can identify these.
    """
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
        "source": "fallback",
    }


# ---------------------------------------------------------------------------
# Main entry point: classify a batch of listings
# ---------------------------------------------------------------------------

def classify_batch(listings: list[dict]) -> list[dict]:
    """Classify a batch of internship listings via the LLM.

    Args:
        listings: list of dicts, each with company_name, programme_name,
                  opening_date, closing_date, latest_stage, etc.

    Returns:
        List of classification dicts in the same order as input.
        Each dict contains firm_type, role_function, programme_status,
        per-field confidence scores, and per-field rationale strings.
    """
    # Build the full user message for this batch
    user_msg = _build_batch_message(listings)

    # Check cache (hash the full user message)
    cache_key = _cache_key(user_msg)
    cached = _load_from_cache(cache_key)
    if cached is not None:
        print(f"  [llm] Cache hit for batch of {len(listings)}")
        return cached.get("classifications", cached)

    # Build system message with few-shot examples
    system_msg = SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES

    # Call API with retry and repair loop
    try:
        raw_text = _call_chat_completions(system_msg, user_msg)
        try:
            result = _parse_and_validate(raw_text)
        except (json.JSONDecodeError, ValueError) as e:
            # One repair attempt if initial parse/validation fails
            print(f"  [llm] Validation failed, attempting repair: {e}")
            result = _repair_response(raw_text, str(e))

        # Cache the successful result
        _save_to_cache(cache_key, result)
        return result["classifications"]

    except Exception as e:
        # Graceful fallback: return UNKNOWN for all listings rather than crash
        print(f"  [llm] API failed after retries: {e}")
        print(f"  [llm] Returning fallback results for {len(listings)} listings")
        return [_fallback_result(listing) for listing in listings]


# ---------------------------------------------------------------------------
# Convenience: classify a single listing (wraps classify_batch)
# ---------------------------------------------------------------------------

def classify_single(listing: dict) -> dict:
    """Classify a single listing. Convenience wrapper around classify_batch."""
    results = classify_batch([listing])
    return results[0] if results else _fallback_result(listing)
