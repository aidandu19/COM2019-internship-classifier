# Evaluation Results (2026-03-29, updated)

## Setup
- Gold standard: 153 hand-labelled entries (153 matched to classified results, 0 dropped)
- Classified: 501 listings from full pipeline (gpt-4o-mini + gpt-4o escalation, temperature=0.0)
- Random baseline: stratified (proportional to gold standard class frequency)
- Rule-based baseline: firm lookup + fuzzy matching + keyword + date inference
- Role function: UNKNOWN now included as valid class (23 UNKNOWN entries evaluated)
- Data fixes applied: Macquarie firm_type IB->BB, Credit Agricole encoding normalised

## Comparison Table (with UNKNOWN included in role_function)

| Method     | Firm F1 | Role F1 | Status F1 | Firm ECE | Role ECE | Status ECE |
|------------|---------|---------|-----------|----------|----------|------------|
| Random     | 0.09    | 0.10    | 0.30      | 0.09     | 0.22     | 0.52       |
| Rule-based | 0.29    | 0.37    | **0.99**  | 0.08     | 0.30     | 0.99*      |
| LLM        | **0.58**| **0.46**| 0.28      | 0.22     | 0.14     | 0.54       |

*Rule-based ECE inflated because baseline confidence scores not stored for programme_status.

### Impact of including UNKNOWN in role_function
- LLM Role F1 dropped from 0.48 to 0.46: the LLM hallucinates specific roles for ambiguous entries
- Top new error: UNKNOWN -> INVESTMENT_BANKING (17 times out of 23 UNKNOWN entries)
- Rule-based Role F1 improved from 0.32 to 0.37: rule-based correctly returns UNKNOWN for unknowns
- LLM Role ECE increased from 0.05 to 0.14: calibration worse when UNKNOWN is a target class

## Key Findings

### 1. Firm Type: LLM adds clear value
- LLM macro F1 = 0.58, nearly double the rule-based (0.30)
- Rule-based collapses to UNKNOWN for any firm not in its lookup table (67% of entries)
- LLM handles unknown firms well, which is the entire point of using an LLM

### 2. Role Function: LLM wins
- LLM macro F1 = 0.48 vs rule-based 0.32
- Top LLM error: OTHER -> INVESTMENT_BANKING (11 times) -- the model over-infers IB roles from generic programme titles
- LLM role_function calibration is excellent (ECE = 0.045)

### 3. Programme Status: LLM is catastrophically wrong
- LLM scores 0.28, **worse than random baseline** (0.29)
- Rule-based scores 0.99 (near perfect)
- Root cause: the LLM interprets dates literally. If opening_date is in the future, it predicts NOT_YET_OPEN. But the gold standard has **zero** NOT_YET_OPEN labels -- the human annotator uses Trackr stage data (e.g. "Offers Out" = CLOSED regardless of calendar dates)
- Error breakdown: CLOSED -> NOT_YET_OPEN (39 times), UNKNOWN -> NOT_YET_OPEN (27 times), CLOSED -> OPEN (18 times)
- The LLM is confident but wrong: 0.8-1.0 confidence bucket has only 37% accuracy

### 4. Elite Boutique vs Middle Market IB (hypothesis test)
- MM misclassified as EB: **5/13 (38%)**
- EB misclassified as MM: **0/11 (0%)**
- Asymmetric confusion -- the LLM over-promotes firms to elite boutique status
- Top firm_type confusions overall: CONSULTING->OTHER (9), QUANT_PROP->HEDGE_FUND (7), BUY_SIDE_PE->OTHER (5), MM->EB (5)

### 5. Confidence Calibration
- **Role function is well-calibrated** (ECE = 0.045). 0.8-1.0 bucket: 81% accuracy at 87% avg confidence
- **Firm type is overconfident** (ECE = 0.22). 0.8-1.0 bucket: 69% accuracy at 90% avg confidence
- **Programme status is badly miscalibrated** (ECE = 0.53). Confident predictions are systematically wrong

### 6. Unseen Split (346 entries without ground truth)
- 44% escalation rate (153/346 escalated to gpt-4o)
- LLM vs rule-based agreement: firm_type 11%, role_function 19%, programme_status 54%
- Low agreement on firm_type confirms the LLM is classifying where the baseline returns UNKNOWN
- Confidence distributions: firm_type skews high (84% in 0.8-1.0), role_function more spread (58% in 0.6-0.8)

## Class Distribution (Gold Standard)

### Firm Type (n=152, fairly balanced)
| Category | Count | % |
|----------|-------|---|
| OTHER | 22 | 14.5% |
| BULGE_BRACKET | 16 | 10.5% |
| BUY_SIDE_PE | 16 | 10.5% |
| CONSULTING | 16 | 10.5% |
| QUANT_PROP | 16 | 10.5% |
| ASSET_MANAGEMENT | 15 | 9.9% |
| HEDGE_FUND | 15 | 9.9% |
| MIDDLE_MARKET_IB | 13 | 8.6% |
| ACCOUNTING | 12 | 7.9% |
| ELITE_BOUTIQUE | 11 | 7.2% |

### Role Function (n=130, after UNKNOWN exclusion)
| Category | Count | % |
|----------|-------|---|
| OTHER | 61 | 46.9% |
| INVESTMENT_BANKING | 20 | 15.4% |
| MARKETS_TRADING | 16 | 12.3% |
| CONSULTING | 15 | 11.5% |
| QUANT | 15 | 11.5% |
| RISK_COMPLIANCE | 2 | 1.5% ** sparse |
| RESEARCH | 1 | 0.8% ** sparse |

### Programme Status (n=152, heavily skewed)
| Category | Count | % |
|----------|-------|---|
| CLOSED | 102 | 67.1% |
| UNKNOWN | 48 | 31.6% |
| OPEN | 2 | 1.3% ** sparse |
| NOT_YET_OPEN | 0 | 0.0% |

## Data Issues (FIXED 2026-03-29)
1. ~~Macquarie Group: gold standard row 2 has FT="IB"~~ Fixed to "BB" (Bulge Bracket)
2. ~~Credit Agricole: unmatched due to accent character~~ Fixed encoding in classified_listings.xlsx
3. Rule-based baseline confidence scores missing for role_function and programme_status in classified_listings.xlsx

## Prompt V2: Programme Status Iteration (Task 6c)

V2 adds explicit instructions to prioritise Trackr's "Latest Stage" text over calendar dates.
Full V2 prompt saved to `data/v2_system_prompt.txt`.

| Metric          | Prompt V1 | Prompt V2 | Rule-based |
|-----------------|-----------|-----------|------------|
| Status F1       | 0.28      | 0.26      | 0.99       |
| Status ECE      | 0.54      | **0.10**  | N/A        |
| Firm F1         | 0.58      | 0.58      | 0.29       |
| Role F1         | 0.46      | 0.51      | 0.37       |

### Key findings:
- **Calibration dramatically improved**: ECE 0.54 -> 0.10 (81% reduction). The model's confidence now matches its accuracy.
- **NOT_YET_OPEN hallucination eliminated**: 41+27=68 false NOT_YET_OPEN predictions in V1 reduced to 6 in V2.
- **New error mode**: CLOSED -> UNKNOWN (68 times). Model now over-predicts UNKNOWN when stage text is empty.
- **F1 flat**: The headline metric stayed similar because UNKNOWN over-prediction replaced NOT_YET_OPEN hallucination.
- **No degradation on other dimensions**: Firm F1 held at 0.58, Role F1 actually improved to 0.51.
- **Interpretation**: V2 converts "confidently wrong" (high confidence, wrong label) into "appropriately uncertain" (lower confidence, UNKNOWN). This is a qualitative improvement even if F1 doesn't reflect it.

## Schema B: Rationale-First Ordering (Task 6d)

Schema B reverses field ordering to: rationale -> label -> confidence (forcing chain-of-thought before commitment).
Uses V1 system prompt (isolating schema variable). Same model (gpt-4o-mini), temp=0.0.

**Field ordering verified**: API confirmed rationale generated BEFORE label in raw response.

| Metric          | Schema A (label-first) | Schema B (rationale-first) | Delta  |
|-----------------|------------------------|---------------------------|--------|
| Firm F1         | 0.58                   | 0.59                      | +0.01  |
| Role F1         | 0.46                   | 0.50                      | +0.04  |
| Status F1       | 0.28                   | 0.31                      | +0.03  |
| Firm ECE        | 0.22                   | 0.24                      | +0.02  |
| Role ECE        | 0.14                   | 0.22                      | +0.08  |
| Status ECE      | 0.54                   | 0.17                      | -0.37  |

### Key findings:
- **F1 improved slightly on all dimensions** (1-4% gains). Reasoning first may help the model make better decisions.
- **Status calibration dramatically improved**: ECE 0.54 -> 0.17 (68% reduction). Model is more honest about uncertainty.
- **Role calibration degraded**: ECE 0.14 -> 0.22. Model became more uncertain on dimensions where it was already well-calibrated.
- **Rationale quality**: Schema B produces more honest rationales. E.g. Macquarie confidence dropped from 0.8 to 0.5 with rationale "No latest stage information provided" vs V1's overconfident "closing date is in the future".
- **Interpretation**: forcing rationale-first acts as a form of chain-of-thought that makes the model more self-aware of ambiguity. This helps most on dimensions where the model was overconfident (status) but can over-correct on well-calibrated dimensions (role).

## Output Files
- Confusion matrices: `output/figures/confusion_matrix_*.png`
- Reliability diagrams: `output/figures/reliability_*.png`
- V2 results: `data/llm_results_v2.json`
- Schema B results: `data/llm_results_schema_b.json`
- V2 system prompt: `data/v2_system_prompt.txt`
- Schema B raw response sample: `data/schema_b_raw_sample.json`
