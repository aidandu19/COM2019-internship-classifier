from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = """
You are a UK finance internship classification engine built for aspiring finance professionals specifically UK university students navigating this summer's internship application cycle. You have deep knowledge of the UK and global financial services landscape and know which firms are bulge brackets, which are elite boutiques, which are buy-side, which span multiple categories, and how their hiring cycle works for each firm.(opening dates, rolling applications, "expressions of interest" windows, etc.).

your task is simple I will provide you with manually extracted internship listing data from Trackr (and possibly other platforms). For each listing, you must classify it along three dimensions:

Dimension 1 — Firm Type. Assign exactly ONE category to the data.

Dimension 2 — Application Status. Assign exactly ONE status whether that be open,closed, not yet open or unknown

Dimension 3 — Role Function. Where the role title provides sufficient information, assign one of: INVESTMENT_BANKING, MARKETS_TRADING, ASSET_MANAGEMENT, RESEARCH, CORPORATE_FINANCE, WEALTH_MANAGEMENT, OPERATIONS, TECHNOLOGY, or OTHER. If unclear, return UNKNOWN.

I expect the outputed data in this specific format
Return a valid JSON object with the following structure for EVERY listing. No free text before or after use this JSON only:

{
  "classifications": [
    {
      "id": 1,
      "firm": "Goldman Sachs",
      "role_title": "Summer Analyst, IBD",
      "firm_type": "BB",
      "firm_type_full": "Bulge Bracket",
      "firm_type_confidence": 0.99,
      "firm_type_rationale": "Goldman Sachs is universally recognised as a bulge bracket investment bank.",
      "application_status": "OPEN",
      "status_confidence": 0.95,
      "status_rationale": "Deadline is 15 March 2026, which is in the future.",
      "role_function": "INVESTMENT_BANKING",
      "deadline": "2026-03-15",
      "programme_type": "Summer Internship",
      "location": "London",
      "notes": ""
    }
  ]
}

If any field is ambiguous or you're unsure lower the confidence and explain in the rationale. I DO NOT WANT YOU TO GUESS OR HALLICULLICINATE.

INTERNSHIP DATA FROM TRACKR:
Not Applied
Citi    Summer Analyst 2026    06 Jan 26        Offers Out    30 Aug 24    OA > VI > AC    Plum    Yes    Yes    No    No
Not Applied
Goldman Sachs    2026 Summer Analyst Programme    15 Aug 25        Offers Out    15 Aug 24    HV > AC    Goldman Sachs Prep    Yes    Yes    Yes    No
Not Applied
Deutsche Bank    Internship Programme 2026    07 Sep 25    31 Oct 25    Offers Out    15 Sep 24    OA > HV > AC    SHL    Yes    Yes    No    No
Not Applied
Private Equity Insights    Global Internship Program    27 Oct 25                        Yes    Yes    Optional    No
VI > AC    Suited    No    Yes    Yes
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt,
    temperature=0.0
)

print(response.output_text)