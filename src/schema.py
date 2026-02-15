# schema.py - unified JSON schema for COM2019 internship classifier
#
# Design: merges the batch-capable structure (Schema B) with enum validation
# (Schema A). Supports response_format for OpenAI structured outputs.
# Each classification has per-field confidence scores and rationale strings.

from jsonschema import Draft7Validator
from .taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES


# ---------------------------------------------------------------------------
# Single classification item schema (used for local validation)
# ---------------------------------------------------------------------------

CLASSIFICATION_ITEM_SCHEMA = {
    "type": "object",
    "required": [
        "company_name",
        "firm_type",
        "firm_type_confidence",
        "firm_type_rationale",
        "role_function",
        "role_function_confidence",
        "role_function_rationale",
        "programme_status",
        "programme_status_confidence",
        "programme_status_rationale",
    ],
    "properties": {
        "company_name": {"type": "string"},
        "firm_type": {"type": "string", "enum": FIRM_TYPES},
        "firm_type_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "firm_type_rationale": {"type": "string"},
        "role_function": {"type": "string", "enum": ROLE_FUNCTIONS},
        "role_function_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "role_function_rationale": {"type": "string"},
        "programme_status": {"type": "string", "enum": PROGRAMME_STATUSES},
        "programme_status_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "programme_status_rationale": {"type": "string"},
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Batch schema: wraps multiple classifications in a single response
# ---------------------------------------------------------------------------

BATCH_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["classifications"],
    "properties": {
        "classifications": {
            "type": "array",
            "items": CLASSIFICATION_ITEM_SCHEMA,
        }
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Validator instance for local schema checking (jsonschema Draft7)
# ---------------------------------------------------------------------------

BATCH_VALIDATOR = Draft7Validator(BATCH_OUTPUT_SCHEMA)
ITEM_VALIDATOR = Draft7Validator(CLASSIFICATION_ITEM_SCHEMA)


# ---------------------------------------------------------------------------
# OpenAI response_format json_schema definition
# ---------------------------------------------------------------------------
# This is the schema passed to client.chat.completions.create() via
# response_format={"type": "json_schema", "json_schema": {...}}.
# OpenAI structured outputs require a strict subset of JSON Schema:
# no $ref, no additionalProperties on nested objects, enum values inline.

OPENAI_RESPONSE_FORMAT = {
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
                            "firm_type",
                            "firm_type_confidence",
                            "firm_type_rationale",
                            "role_function",
                            "role_function_confidence",
                            "role_function_rationale",
                            "programme_status",
                            "programme_status_confidence",
                            "programme_status_rationale",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "company_name": {"type": "string"},
                            "firm_type": {
                                "type": "string",
                                "enum": FIRM_TYPES,
                            },
                            "firm_type_confidence": {"type": "number"},
                            "firm_type_rationale": {"type": "string"},
                            "role_function": {
                                "type": "string",
                                "enum": ROLE_FUNCTIONS,
                            },
                            "role_function_confidence": {"type": "number"},
                            "role_function_rationale": {"type": "string"},
                            "programme_status": {
                                "type": "string",
                                "enum": PROGRAMME_STATUSES,
                            },
                            "programme_status_confidence": {"type": "number"},
                            "programme_status_rationale": {"type": "string"},
                        },
                    },
                }
            },
        },
    },
}
