from openai import OpenAI
import json

import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = """
You are a UK finance internship classification engine built for
aspiring finance professionals specifically UK university students
navigating this summer's internship application cycle. You have deep
knowledge of the UK and global financial services landscape and know
which firms are bulge brackets, which are elite boutiques, which are
buy-side, which span multiple categories, and how their hiring cycle
works for each firm.

Your task is simple. For each listing, classify it along two
dimensions:
Dimension 1 — Firm Type. Assign exactly ONE category.
Dimension 2 — Application Status. Assign exactly ONE status:
OPEN, CLOSED, NOT_YET_OPEN, or UNKNOWN.

Return ONLY valid JSON in this format:
{"classifications": [
  {"id": 1, "firm": "...", "role_title": "...",
   "firm_type": "...", "firm_type_full": "...",
   "firm_type_confidence": 0.0,
   "firm_type_rationale": "...",
   "application_status": "...",
   "status_confidence": 0.0,
   "status_rationale": "...",
   "deadline": "...", "programme_type": "...",
   "location": "", "notes": ""}
]}

If any field is ambiguous or you're unsure, lower the confidence
and explain in the rationale. Do not guess or hallucinate.

INTERNSHIP DATA FROM TRACKR:

Private Equity Insights	Global Internship Program	27 Oct 25
Yes	Yes	Optional	No
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)

result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2))
