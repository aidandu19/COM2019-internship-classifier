# taxonomy.py - classification enums for COM2019 internship classifier

FIRM_TYPES = [
    "BULGE_BRACKET",
    "ELITE_BOUTIQUE",
    "MIDDLE_MARKET_IB",
    "BUY_SIDE_PE",
    "ASSET_MANAGEMENT",
    "HEDGE_FUND",
    "QUANT_PROP",
    "ACCOUNTING",
    "CONSULTING",
    "OTHER",
    "UNKNOWN",
]

ROLE_FUNCTIONS = [
    "INVESTMENT_BANKING",
    "MARKETS_TRADING",
    "RESEARCH",
    "QUANT",
    "CONSULTING",
    "OPERATIONS",
    "TECH_DATA",
    "RISK_COMPLIANCE",
    "OTHER",
    "UNKNOWN",
]

PROGRAMME_STATUSES = [
    "OPEN",
    "CLOSED",
    "NOT_YET_OPEN",
    "UNKNOWN",
]

# ---------------------------------------------------------------------------
# Reverse mapping dicts: short codes -> full enum names
# Used by convert_gold_standard.py to translate hand-labelled Excel codes
# ---------------------------------------------------------------------------

FIRM_TYPE_CODES = {
    "BB": "BULGE_BRACKET",
    "EB": "ELITE_BOUTIQUE",
    "MM": "MIDDLE_MARKET_IB",
    "AM": "ASSET_MANAGEMENT",
    "PE": "BUY_SIDE_PE",
    "HF": "HEDGE_FUND",
    "QP": "QUANT_PROP",
    "ACC": "ACCOUNTING",
    "CON": "CONSULTING",
    "OTH": "OTHER",
    "UNK": "UNKNOWN",
}

ROLE_FUNCTION_CODES = {
    "IB": "INVESTMENT_BANKING",
    "MT": "MARKETS_TRADING",
    "RES": "RESEARCH",
    "QNT": "QUANT",
    "CON": "CONSULTING",
    "OPS": "OPERATIONS",
    "TD": "TECH_DATA",
    "RC": "RISK_COMPLIANCE",
    "OTH": "OTHER",
    "UNK": "UNKNOWN",
}

PROGRAMME_STATUS_CODES = {
    "O": "OPEN",
    "C": "CLOSED",
    "NYO": "NOT_YET_OPEN",
    "U": "UNKNOWN",
}
