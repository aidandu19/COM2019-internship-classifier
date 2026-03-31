# COM2019 Finance Internship Classifier

An LLM-powered classification pipeline for UK finance internship listings. Classifies listings by firm type, role function, and programme status using a two-tier model strategy (gpt-4o-mini with gpt-4o escalation), evaluated against a 153-entry hand-labelled gold standard.

## Branches

| Branch | Description |
|--------|-------------|
| `FINAL-COM2019-project` | Final submission. Cleaned structure, full evaluation, iterative prompt experiments. |
| `draft-com2019-code` | Early draft code from initial development. |

## Project Structure

```
main.py                  # pipeline entry point
requirements.txt         # dependencies
.env.example             # API key placeholder

src/                     # source code
  taxonomy.py            # classification enums
  schema.py              # JSON schema for structured outputs
  llm_classifier.py      # LLM classifier (few-shot, caching, escalation)
  baseline.py            # rule-based + random baselines
  evaluate.py            # evaluation pipeline (F1, ECE, calibration)
  export.py              # Excel/CSV export with formatting
  batch_classify.py      # utility: classify full dataset
  classify_gold_standard.py  # utility: classify gold standard entries
  convert_gold_standard.py   # utility: convert Excel codes to enums

data/                    # pipeline data
  sample_listings.csv    # default input (small sample)
  FULL TRACKR LISTINGS.csv   # full 501-listing dataset
  gold_standard.json     # 153 hand-labelled entries
  classified_listings.xlsx   # pipeline output used by evaluation

output/                  # generated results
  figures/               # confusion matrices + reliability diagrams

appendix/                # iterative development evidence
  prototypes/            # early experiments and tests
  iterations/            # prompt variant experiments (v2, schema B, zero-shot)
  data/                  # experiment data files
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env

python main.py                    # full pipeline (baseline + LLM)
python main.py --dry-run          # baseline only, no API calls
python main.py --sample 10        # process first 10 listings
python main.py --evaluate         # run evaluation after classification
```
