# src/test_one_row.py

import pandas as pd
from .llm_classifier import classify_programme

EXCEL_PATH = "Stuctured Mock Data.xlsx"  # make sure filename matches exactly


def load_excel(path: str) -> pd.DataFrame:
    # Row 0 contains the real headers ("Company Name", etc.)
    df = pd.read_excel(path, header=0)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop completely empty rows (your spacer rows)
    df = df.dropna(how="all")

    return df


def build_programme_text(row: pd.Series) -> str:
    company = str(row.get("Company Name", "")).strip()
    programme = str(row.get("Programme Name", "")).strip()
    opening = str(row.get("Opening Date", "")).strip()
    closing = str(row.get("Closing Date", "")).strip()
    latest_stage = str(row.get("Latest Stage", "")).strip()
    process = str(row.get("Process", "")).strip()
    notes = str(row.get("Notes", "")).strip()

    text = f"""
Company: {company}
Programme: {programme}
Opening date: {opening}
Closing date: {closing}
Latest stage: {latest_stage}
Process: {process}
Notes: {notes}
""".strip()

    return text


def main():
    df = load_excel(EXCEL_PATH)

    print("Loaded rows (after dropping blanks):", len(df))
    print("Columns:", list(df.columns))

    # Keep only rows that actually have a company name
    df = df[df["Company Name"].notna() & (df["Company Name"].astype(str).str.strip() != "")]

    # Take first 8 real programmes
    test_rows = df.head(8)

    for idx, row in test_rows.iterrows():
        print("\n" + "=" * 60)
        print(f"ROW INDEX {idx}")

        programme_text = build_programme_text(row)

        print("\n--- Programme text sent to LLM ---\n")
        print(programme_text)

        print("\n--- LLM classification ---\n")
        result = classify_programme(programme_text)
        print(result)


if __name__ == "__main__":
    main()