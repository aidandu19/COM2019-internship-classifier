# baseline.py - rule-based and random baseline classifiers for COM2019
#
# Two baselines for comparison against the LLM classifier:
# 1. Random baseline: assigns labels uniformly at random (lower bound)
# 2. Rule-based baseline: firm name lookup + fuzzy matching + keyword inference
#    + date-based status inference (upper bound for non-LLM approaches)

import random
from datetime import datetime

import pandas as pd
from thefuzz import fuzz

from .taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES


# ---------------------------------------------------------------------------
# Firm name lookup table (~50 known firms)
# ---------------------------------------------------------------------------

FIRM_LOOKUP = {
    # Bulge Bracket
    "goldman sachs": "BULGE_BRACKET",
    "jp morgan": "BULGE_BRACKET",
    "jpmorgan": "BULGE_BRACKET",
    "j.p. morgan": "BULGE_BRACKET",
    "morgan stanley": "BULGE_BRACKET",
    "bank of america": "BULGE_BRACKET",
    "citi": "BULGE_BRACKET",
    "citigroup": "BULGE_BRACKET",
    "barclays": "BULGE_BRACKET",
    "deutsche bank": "BULGE_BRACKET",
    "ubs": "BULGE_BRACKET",
    "credit suisse": "BULGE_BRACKET",
    "hsbc": "BULGE_BRACKET",
    # Elite Boutique
    "lazard": "ELITE_BOUTIQUE",
    "evercore": "ELITE_BOUTIQUE",
    "centerview": "ELITE_BOUTIQUE",
    "moelis": "ELITE_BOUTIQUE",
    "pjt partners": "ELITE_BOUTIQUE",
    "perella weinberg": "ELITE_BOUTIQUE",
    "guggenheim": "ELITE_BOUTIQUE",
    "rothschild": "ELITE_BOUTIQUE",
    "qatalyst": "ELITE_BOUTIQUE",
    # Middle Market IB
    "houlihan lokey": "MIDDLE_MARKET_IB",
    "piper sandler": "MIDDLE_MARKET_IB",
    "william blair": "MIDDLE_MARKET_IB",
    "raymond james": "MIDDLE_MARKET_IB",
    "stifel": "MIDDLE_MARKET_IB",
    "rbc capital": "MIDDLE_MARKET_IB",
    "lincoln international": "MIDDLE_MARKET_IB",
    "baird": "MIDDLE_MARKET_IB",
    "greenhill": "MIDDLE_MARKET_IB",
    "harris williams": "MIDDLE_MARKET_IB",
    "dc advisory": "MIDDLE_MARKET_IB",
    # Buy Side / PE
    "blackstone": "BUY_SIDE_PE",
    "kkr": "BUY_SIDE_PE",
    "carlyle": "BUY_SIDE_PE",
    "apollo": "BUY_SIDE_PE",
    "warburg pincus": "BUY_SIDE_PE",
    "advent international": "BUY_SIDE_PE",
    "bain capital": "BUY_SIDE_PE",
    "tpg": "BUY_SIDE_PE",
    # Asset Management
    "blackrock": "ASSET_MANAGEMENT",
    "fidelity": "ASSET_MANAGEMENT",
    "vanguard": "ASSET_MANAGEMENT",
    "schroders": "ASSET_MANAGEMENT",
    "abrdn": "ASSET_MANAGEMENT",
    "invesco": "ASSET_MANAGEMENT",
    "pimco": "ASSET_MANAGEMENT",
    "man group": "ASSET_MANAGEMENT",
    "newton": "ASSET_MANAGEMENT",
    # Hedge Fund / Trading
    "citadel": "HEDGE_FUND",
    "millennium": "HEDGE_FUND",
    "point72": "HEDGE_FUND",
    "two sigma": "HEDGE_FUND",
    "de shaw": "HEDGE_FUND",
    "d.e. shaw": "HEDGE_FUND",
    "bridgewater": "HEDGE_FUND",
    "aqr": "HEDGE_FUND",
    # Quant / Prop Trading
    "jane street": "QUANT_PROP",
    "optiver": "QUANT_PROP",
    "imc": "QUANT_PROP",
    "flow traders": "QUANT_PROP",
    "susquehanna": "QUANT_PROP",
    "sig": "QUANT_PROP",
    # Consulting
    "mckinsey": "CONSULTING",
    "bain": "CONSULTING",
    "bcg": "CONSULTING",
    "deloitte": "ACCOUNTING",
    "pwc": "ACCOUNTING",
    "ey": "ACCOUNTING",
    "kpmg": "ACCOUNTING",
    "bdo": "ACCOUNTING",
    "grant thornton": "ACCOUNTING",
    "rsm": "ACCOUNTING",
    "mazars": "ACCOUNTING",
    "oliver wyman": "CONSULTING",
    "roland berger": "CONSULTING",
}

# Keywords for inferring role function from programme name
ROLE_KEYWORDS = {
    "INVESTMENT_BANKING": ["investment banking", "ibd", "m&a", "mergers", "advisory", "corporate finance"],
    "MARKETS_TRADING": ["trading", "markets", "sales & trading", "s&t", "fixed income", "equities", "global markets"],
    "RESEARCH": ["research", "equity research"],
    "QUANT": ["quant", "quantitative", "strats", "systematic"],
    "CONSULTING": ["consulting", "consultant", "strategy"],
    "OPERATIONS": ["operations", "ops", "middle office", "back office"],
    "TECH_DATA": ["technology", "tech", "engineering", "data", "software", "developer"],
    "RISK_COMPLIANCE": ["risk", "compliance", "regulatory", "audit"],
}

# Fuzzy match threshold: 80 balances recall vs false positives.
# Lower values catch more typos but risk misclassifying similar-sounding firms.
FUZZY_THRESHOLD = 80


# ---------------------------------------------------------------------------
# Firm type classification: exact substring then fuzzy matching
# ---------------------------------------------------------------------------

def classify_firm_type(company_name: str) -> tuple[str, float]:
    """Match a company name to a firm type using exact substring then fuzzy match.

    Returns (firm_type, confidence) tuple.
    """
    name_lower = company_name.lower().strip()

    # Pass 1: exact substring match (fast, high confidence)
    for firm, ftype in FIRM_LOOKUP.items():
        if firm in name_lower:
            return ftype, 0.95

    # Pass 2: fuzzy match for misspellings and partial names
    best_score = 0
    best_type = "UNKNOWN"
    for firm, ftype in FIRM_LOOKUP.items():
        score = fuzz.ratio(name_lower, firm)
        if score > best_score:
            best_score = score
            best_type = ftype

    if best_score >= FUZZY_THRESHOLD:
        return best_type, round(best_score / 100, 2)

    return "UNKNOWN", 0.0


# ---------------------------------------------------------------------------
# Role function classification: keyword matching on programme name
# ---------------------------------------------------------------------------

def classify_role_function(programme_name: str) -> tuple[str, float]:
    """Infer role function from programme name keywords.

    Returns (role_function, confidence) tuple.
    """
    prog_lower = programme_name.lower().strip()

    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in prog_lower for kw in keywords):
            return role, 0.7

    return "UNKNOWN", 0.0


# ---------------------------------------------------------------------------
# Programme status classification: stage text + date-based inference
# ---------------------------------------------------------------------------

def classify_status(row: dict) -> tuple[str, float]:
    """Infer programme status from latest_stage text and closing date.

    Uses stage text first (highest signal), then falls back to date comparison.
    Returns (status, confidence) tuple.
    """
    stage = str(row.get("latest_stage", "")).lower().strip()
    closing = str(row.get("closing_date", "")).strip()

    # Stage text is the strongest signal
    if any(w in stage for w in ["offers out", "closed", "offer", "rejected", "withdrawn"]):
        return "CLOSED", 0.85
    if any(w in stage for w in ["open", "apply", "accepting"]):
        return "OPEN", 0.85
    if any(w in stage for w in ["not yet", "coming soon", "tbc", "tbd"]):
        return "NOT_YET_OPEN", 0.80

    # Fall back to closing date comparison
    if closing and closing != "nan":
        try:
            close_date = pd.to_datetime(closing, format="mixed", dayfirst=True, errors="coerce")
            if close_date is not None and close_date is not pd.NaT:
                if close_date < datetime.now():
                    return "CLOSED", 0.65
                else:
                    return "OPEN", 0.65
        except Exception:
            pass

    return "UNKNOWN", 0.0


# ---------------------------------------------------------------------------
# Random baseline: trivial lower-bound comparator
# ---------------------------------------------------------------------------

def random_baseline(listings: list[dict], class_weights: dict | None = None) -> list[dict]:
    """Assign random labels proportional to class frequency (stratified baseline).

    When *class_weights* is provided it should map dimension names to
    Counter-like dicts, e.g.::

        {"firm_type": {"BULGE_BRACKET": 17, "OTHER": 22, ...}, ...}

    If omitted, falls back to uniform random (legacy behaviour).

    Returns list of classification dicts matching the LLM output schema.
    """
    def _pick(values, weights_dict):
        if weights_dict:
            population = list(weights_dict.keys())
            weights = list(weights_dict.values())
            return random.choices(population, weights=weights, k=1)[0]
        return random.choice(values)

    ft_w = (class_weights or {}).get("firm_type")
    rf_w = (class_weights or {}).get("role_function")
    ps_w = (class_weights or {}).get("programme_status")

    results = []
    for listing in listings:
        results.append({
            "company_name": listing.get("company_name", ""),
            "firm_type": _pick(FIRM_TYPES, ft_w),
            "firm_type_confidence": 0.0,
            "firm_type_rationale": "Stratified random baseline (proportional to class frequency)",
            "role_function": _pick(ROLE_FUNCTIONS, rf_w),
            "role_function_confidence": 0.0,
            "role_function_rationale": "Stratified random baseline (proportional to class frequency)",
            "programme_status": _pick(PROGRAMME_STATUSES, ps_w),
            "programme_status_confidence": 0.0,
            "programme_status_rationale": "Stratified random baseline (proportional to class frequency)",
            "source": "random_baseline",
        })
    return results


# ---------------------------------------------------------------------------
# Rule-based baseline: combines lookup + fuzzy + keywords + dates
# ---------------------------------------------------------------------------

def rule_based_baseline(listings: list[dict]) -> list[dict]:
    """Classify all listings using deterministic rules.

    Combines exact/fuzzy firm lookup, keyword-based role inference,
    and stage/date-based status inference. Returns list of classification
    dicts matching the LLM output schema for direct comparison.
    """
    results = []
    for listing in listings:
        company = str(listing.get("company_name", ""))
        programme = str(listing.get("programme_name", ""))

        firm_type, firm_conf = classify_firm_type(company)
        role_function, role_conf = classify_role_function(programme)
        status, status_conf = classify_status(listing)

        results.append({
            "company_name": company,
            "firm_type": firm_type,
            "firm_type_confidence": firm_conf,
            "firm_type_rationale": f"Rule-based: {'exact match' if firm_conf >= 0.95 else 'fuzzy match' if firm_conf > 0 else 'no match'} in lookup table",
            "role_function": role_function,
            "role_function_confidence": role_conf,
            "role_function_rationale": f"Rule-based: {'keyword match' if role_conf > 0 else 'no keyword match'} in programme name",
            "programme_status": status,
            "programme_status_confidence": status_conf,
            "programme_status_rationale": f"Rule-based: {'stage text' if status_conf >= 0.8 else 'date comparison' if status_conf > 0 else 'no signal'} inference",
            "source": "rule_based_baseline",
        })

    print(f"  [baseline] Done. {len(results)} rows classified.")
    return results
