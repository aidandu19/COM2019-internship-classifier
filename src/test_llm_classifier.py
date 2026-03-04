from src.llm_classifier import classify_programme

text = """
Blackstone is offering a 2026 Summer Analyst position within its Private Equity
investment team in London. Interns will support deal execution, financial modelling,
and due diligence. Applications open in September.
"""

result = classify_programme(text)
print(result)