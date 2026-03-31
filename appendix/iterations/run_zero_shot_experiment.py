#!/usr/bin/env python3
# run_zero_shot_experiment.py - Zero-shot vs few-shot (V1) controlled comparison
#
# Classifies all 153 gold standard entries WITHOUT few-shot examples in the
# system prompt. Everything else identical: same taxonomy definitions, same JSON
# schema, same two-tier escalation (gpt-4o-mini -> gpt-4o at 0.7), temp 0.0.
#
# Outputs:
#   data/zero_shot_results.json
#   output/figures/confusion_matrix_zero_shot_firm_type.png
#   output/figures/reliability_combined_zero_shot_{dim}.png
#
# Usage:
#   python run_zero_shot_experiment.py              # full 153 entries
#   python run_zero_shot_experiment.py --sample 10  # first 10 only (test run)
#   python run_zero_shot_experiment.py --eval-only  # skip classification, just evaluate

import os
import sys
import json
import random
import argparse
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from dotenv import load_dotenv
load_dotenv()

from src.llm_classifier import classify_batch
from src.baseline import rule_based_baseline, random_baseline
from src.taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES
from src.evaluate import (
    load_gold_standard, load_classified_results, match_gold_to_classified,
    evaluate_classifier, confusion_matrix_plot, calibration_analysis,
    calculate_ece, CALIBRATION_BINS, FIGURES_DIR
)

GOLD_JSON_PATH = "data/gold_standard.json"
ZERO_SHOT_RESULTS_PATH = "data/zero_shot_results.json"
CLASSIFIED_XLSX_PATH = "data/classified_listings.xlsx"
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# 1. Load gold standard from JSON (has ground truth + classifier input fields)
# ---------------------------------------------------------------------------

def load_gold_json():
    """Load gold standard entries with ground truth labels and input fields."""
    with open(GOLD_JSON_PATH) as f:
        return json.load(f)


def build_listings(gold_entries):
    """Convert gold standard entries to classifier input format."""
    listings = []
    for g in gold_entries:
        listings.append({
            "company_name": g.get("company_name", ""),
            "programme_name": g.get("programme_name") or g.get("programme", ""),
            "opening_date": g.get("opening_date", ""),
            "closing_date": g.get("closing_date") or g.get("closing", ""),
            "latest_stage": g.get("latest_stage") or g.get("stage", ""),
        })
    return listings


# ---------------------------------------------------------------------------
# 2. Zero-shot classification
# ---------------------------------------------------------------------------

def classify_all_zero_shot(listings):
    """Classify all listings in sequential batches with zero_shot=True."""
    all_results = []
    total_batches = (len(listings) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(listings), BATCH_SIZE):
        batch = listings[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} listings)...")
        results = classify_batch(batch, zero_shot=True)
        all_results.extend(results)
    return all_results


# ---------------------------------------------------------------------------
# 3. Extract V1 predictions via xlsx matching (same source as existing eval)
# ---------------------------------------------------------------------------

def load_v1_predictions():
    """Load V1 (few-shot) predictions by matching gold standard against
    classified_listings.xlsx. Returns (gold_entries, v1_preds, rule_preds)
    aligned by index, or None if matching fails.
    """
    gold_xlsx = load_gold_standard()
    if gold_xlsx is None:
        return None, None, None

    classified = load_classified_results(CLASSIFIED_XLSX_PATH)
    if classified is None:
        return None, None, None

    matched, unmatched, _ = match_gold_to_classified(gold_xlsx, classified)
    if unmatched:
        print(f"  WARNING: {len(unmatched)} unmatched gold entries")

    gold_entries = [m[0] for m in matched]

    # V1 LLM predictions
    v1_preds = []
    for _, c in matched:
        v1_preds.append({
            "firm_type": c.get("firm_type", "UNKNOWN"),
            "firm_type_confidence": c.get("firm_type_confidence", 0.5),
            "role_function": c.get("role_function", "UNKNOWN"),
            "role_function_confidence": c.get("role_function_confidence", 0.5),
            "programme_status": c.get("programme_status", "UNKNOWN"),
            "programme_status_confidence": c.get("programme_status_confidence", 0.5),
            "model_used": c.get("model_used", ""),
            "escalated": c.get("escalated", False),
        })

    # Rule-based baseline predictions (from xlsx, same as existing eval)
    rule_preds = []
    for _, c in matched:
        rule_preds.append({
            "firm_type": c.get("baseline_firm_type", "UNKNOWN"),
            "firm_type_confidence": c.get("baseline_firm_type_confidence", 0.0),
            "role_function": c.get("baseline_role_function", "UNKNOWN"),
            "role_function_confidence": 0.0,
            "programme_status": c.get("baseline_programme_status", "UNKNOWN"),
            "programme_status_confidence": 0.0,
        })

    return gold_entries, v1_preds, rule_preds


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def compute_nyo_false_positives(y_true, y_pred):
    """Count NOT_YET_OPEN false positives: predicted NYO when true != NYO."""
    return sum(1 for t, p in zip(y_true, y_pred)
               if p == "NOT_YET_OPEN" and t != "NOT_YET_OPEN")


def compute_escalation_rate(preds):
    """Percentage of entries escalated to gpt-4o."""
    escalated = sum(1 for p in preds
                    if p.get("escalated") in (True, "True", "TRUE", "true"))
    return round(escalated / max(len(preds), 1) * 100, 1)


# ---------------------------------------------------------------------------
# 5. Combined reliability diagram (multi-series line plot)
# ---------------------------------------------------------------------------

def combined_reliability_diagram(methods_data, dimension, save_path):
    """Plot reliability diagram with multiple methods as line+marker series.

    Args:
        methods_data: list of (method_name, y_pred, y_true, confidences)
        dimension: name for the plot title
        save_path: output file path
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    colours = {
        "Random": "#999999",
        "Rule-based": "#E67E22",
        "LLM (V1)": "#2980B9",
        "Zero-shot": "#E74C3C",
    }
    markers = {
        "Random": "s",
        "Rule-based": "^",
        "LLM (V1)": "o",
        "Zero-shot": "D",
    }

    for method_name, y_pred, y_true, confs in methods_data:
        cal = calibration_analysis(y_pred, y_true, confs)
        if cal is None:
            continue

        bins_with_data = [b for b in cal["bins"] if b["n"] > 0]
        if not bins_with_data:
            continue

        avg_confs = [b["avg_conf"] for b in bins_with_data]
        accuracies = [b["accuracy"] for b in bins_with_data]
        ece = calculate_ece(y_pred, y_true, confs)

        ax.plot(
            avg_confs, accuracies,
            marker=markers.get(method_name, "o"),
            color=colours.get(method_name, "#333333"),
            linewidth=2, markersize=8,
            label=f"{method_name} (ECE={ece:.3f})",
        )

    ax.set_xlabel("Mean predicted confidence", fontsize=12)
    ax.set_ylabel("Fraction correct", fontsize=12)
    ax.set_title(f"Reliability Diagram: {dimension}", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# 6. Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot vs few-shot (V1) controlled experiment"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Classify only the first N entries (for testing)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip classification, evaluate existing zero_shot_results.json",
    )
    args = parser.parse_args()

    # Check API key (unless eval-only)
    if not args.eval_only:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "API_key_here":
            print("ERROR: No valid OPENAI_API_KEY in .env")
            sys.exit(1)

    print("=" * 70)
    print("ZERO-SHOT EXPERIMENT: Few-shot examples removed")
    print("=" * 70)

    # --- Load gold standard (JSON has both input fields and ground truth) ---
    gold_entries = load_gold_json()
    listings = build_listings(gold_entries)
    print(f"Loaded {len(gold_entries)} gold standard entries from {GOLD_JSON_PATH}")

    if args.sample and args.sample < len(gold_entries):
        gold_entries = gold_entries[:args.sample]
        listings = listings[:args.sample]
        print(f"  Limited to {args.sample} entries")

    # --- Step 1: Zero-shot classification ---
    if args.eval_only:
        if not os.path.exists(ZERO_SHOT_RESULTS_PATH):
            print(f"ERROR: {ZERO_SHOT_RESULTS_PATH} not found. Run without --eval-only first.")
            sys.exit(1)
        print(f"\n[1/3] Loading existing zero-shot results from {ZERO_SHOT_RESULTS_PATH}")
        with open(ZERO_SHOT_RESULTS_PATH) as f:
            zs_results = json.load(f)
    else:
        print(f"\n[1/3] Running zero-shot classification ({len(listings)} entries, "
              f"batch size {BATCH_SIZE})...")
        zs_results = classify_all_zero_shot(listings)

        # Validate taxonomy compliance
        errors = []
        for i, r in enumerate(zs_results):
            company = r.get("company_name", f"entry_{i}")
            if r.get("firm_type") not in FIRM_TYPES:
                errors.append(f"  {company}: invalid firm_type '{r.get('firm_type')}'")
            if r.get("role_function") not in ROLE_FUNCTIONS:
                errors.append(f"  {company}: invalid role_function '{r.get('role_function')}'")
            if r.get("programme_status") not in PROGRAMME_STATUSES:
                errors.append(f"  {company}: invalid programme_status '{r.get('programme_status')}'")
        if errors:
            print(f"  {len(errors)} validation errors:")
            for e in errors[:10]:
                print(e)
        else:
            print("  All results valid.")

        # Save
        with open(ZERO_SHOT_RESULTS_PATH, "w") as f:
            json.dump(zs_results, f, indent=2)
        print(f"  Saved to {ZERO_SHOT_RESULTS_PATH}")

    print(f"  Zero-shot results: {len(zs_results)} entries")

    # --- Step 2: Load V1 predictions from classified_listings.xlsx ---
    print(f"\n[2/3] Loading V1 (few-shot) predictions from {CLASSIFIED_XLSX_PATH}...")
    v1_gold, v1_preds, rule_preds_xlsx = load_v1_predictions()

    if v1_gold is None:
        print("  ERROR: Could not load V1 predictions. Aborting.")
        sys.exit(1)

    print(f"  Matched {len(v1_preds)} V1 predictions to gold standard")
    print(f"  V1 escalation rate: {compute_escalation_rate(v1_preds)}%")

    # --- Build ground truth arrays ---
    # Use gold_standard.json labels (aligned with zero-shot results by index)
    n = min(len(gold_entries), len(zs_results))
    dims = ["firm_type", "role_function", "programme_status"]
    y_true = {d: [g[d] for g in gold_entries[:n]] for d in dims}

    # For V1 and rule-based, use the xlsx-matched data (aligned with v1_gold by index)
    # These should have the same gold labels; verify alignment
    v1_true = {d: [g[d] for g in v1_gold] for d in dims}

    # Generate random baseline (stratified, seed=42, same as existing eval)
    random.seed(42)
    class_weights = {d: dict(Counter(y_true[d])) for d in dims}
    random_preds = random_baseline(listings[:n], class_weights=class_weights)

    # Also generate fresh rule-based baseline for zero-shot entries
    # (so rule-based is evaluated on same entries as zero-shot)
    rule_fresh = rule_based_baseline(listings[:n])

    # --- Step 3: Compute metrics ---
    print(f"\n[3/3] Computing metrics...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Methods: (name, predictions, ground_truth_labels, n_entries)
    # For Random and Rule-based: use gold_entries ground truth + fresh baselines
    # For V1: use v1_gold ground truth + v1_preds (from xlsx matching)
    # For Zero-shot: use gold_entries ground truth + zs_results

    table = []

    # --- Random baseline ---
    row = {"method": "Random"}
    for dim in dims:
        y_t = y_true[dim]
        y_p = [p[dim] for p in random_preds]
        confs = [float(p.get(f"{dim}_confidence", 0.0)) for p in random_preds]
        all_labels = sorted(set(y_t + y_p))
        report = classification_report(y_t, y_p, labels=all_labels,
                                       zero_division=0, output_dict=True)
        row[f"{dim}_f1"] = round(report["macro avg"]["f1-score"], 4)
        row[f"{dim}_ece"] = calculate_ece(y_p, y_t, confs) or 0.0
    row["nyo_fp"] = compute_nyo_false_positives(
        y_true["programme_status"], [p["programme_status"] for p in random_preds])
    row["escalation_pct"] = 0.0
    table.append(row)

    # --- Rule-based baseline ---
    row = {"method": "Rule-based"}
    for dim in dims:
        # Use xlsx rule-based for consistency with existing report
        y_t = v1_true[dim]
        y_p = [p[dim] for p in rule_preds_xlsx]
        confs = [float(p.get(f"{dim}_confidence", 0.0)) for p in rule_preds_xlsx]
        all_labels = sorted(set(y_t + y_p))
        report = classification_report(y_t, y_p, labels=all_labels,
                                       zero_division=0, output_dict=True)
        row[f"{dim}_f1"] = round(report["macro avg"]["f1-score"], 4)
        row[f"{dim}_ece"] = calculate_ece(y_p, y_t, confs) or 0.0
    row["nyo_fp"] = compute_nyo_false_positives(
        v1_true["programme_status"], [p["programme_status"] for p in rule_preds_xlsx])
    row["escalation_pct"] = 0.0
    table.append(row)

    # --- LLM V1 (few-shot) ---
    row = {"method": "LLM (V1)"}
    for dim in dims:
        y_t = v1_true[dim]
        y_p = [p[dim] for p in v1_preds]
        confs = [float(p.get(f"{dim}_confidence", 0.5)) for p in v1_preds]
        all_labels = sorted(set(y_t + y_p))
        report = classification_report(y_t, y_p, labels=all_labels,
                                       zero_division=0, output_dict=True)
        row[f"{dim}_f1"] = round(report["macro avg"]["f1-score"], 4)
        row[f"{dim}_ece"] = calculate_ece(y_p, y_t, confs) or 0.0
    row["nyo_fp"] = compute_nyo_false_positives(
        v1_true["programme_status"], [p["programme_status"] for p in v1_preds])
    row["escalation_pct"] = compute_escalation_rate(v1_preds)
    table.append(row)

    # --- Zero-shot ---
    row = {"method": "Zero-shot"}
    for dim in dims:
        y_t = y_true[dim]
        y_p = [p[dim] for p in zs_results[:n]]
        confs = [float(p.get(f"{dim}_confidence", 0.5)) for p in zs_results[:n]]
        all_labels = sorted(set(y_t + y_p))
        report = classification_report(y_t, y_p, labels=all_labels,
                                       zero_division=0, output_dict=True)
        row[f"{dim}_f1"] = round(report["macro avg"]["f1-score"], 4)
        row[f"{dim}_ece"] = calculate_ece(y_p, y_t, confs) or 0.0
    row["nyo_fp"] = compute_nyo_false_positives(
        y_true["programme_status"], [p["programme_status"] for p in zs_results[:n]])
    row["escalation_pct"] = compute_escalation_rate(zs_results[:n])
    table.append(row)

    # --- Print detailed zero-shot classification report ---
    print(f"\n{'=' * 70}")
    print("ZERO-SHOT: DETAILED CLASSIFICATION REPORTS")
    print(f"{'=' * 70}")
    for dim in dims:
        y_t = y_true[dim]
        y_p = [p[dim] for p in zs_results[:n]]
        evaluate_classifier(y_p, y_t, label=f"Zero-shot - {dim}")

    # --- Print comparison table ---
    print(f"\n{'=' * 105}")
    print("COMPARISON TABLE")
    print(f"{'=' * 105}")
    header = (f"{'Method':<15} | {'Firm F1':>8} | {'Role F1':>8} | {'Status F1':>10} | "
              f"{'Firm ECE':>9} | {'Role ECE':>9} | {'Status ECE':>11} | "
              f"{'NYO FP':>6} | {'Esc %':>6}")
    print(header)
    print("-" * len(header))
    for r in table:
        print(
            f"{r['method']:<15} | "
            f"{r['firm_type_f1']:.4f}   | "
            f"{r['role_function_f1']:.4f}   | "
            f"{r['programme_status_f1']:.4f}     | "
            f"{r.get('firm_type_ece', 0):.4f}    | "
            f"{r.get('role_function_ece', 0):.4f}    | "
            f"{r.get('programme_status_ece', 0):.4f}      | "
            f"{r['nyo_fp']:>5d}  | "
            f"{r['escalation_pct']:>5.1f}%"
        )
    print(f"{'=' * 105}")

    # --- Confusion matrix: zero-shot firm_type ---
    print(f"\nGenerating figures...")
    zs_ft_pred = [p["firm_type"] for p in zs_results[:n]]
    confusion_matrix_plot(
        zs_ft_pred, y_true["firm_type"],
        label="Zero_Shot_firm_type",
        save_path=os.path.join(FIGURES_DIR, "confusion_matrix_zero_shot_firm_type.png"),
    )

    # --- Combined reliability diagrams (4 series per dimension) ---
    for dim in dims:
        methods_data = []

        # Random
        methods_data.append((
            "Random",
            [p[dim] for p in random_preds],
            y_true[dim],
            [float(p.get(f"{dim}_confidence", 0.0)) for p in random_preds],
        ))
        # Rule-based (from xlsx)
        methods_data.append((
            "Rule-based",
            [p[dim] for p in rule_preds_xlsx],
            v1_true[dim],
            [float(p.get(f"{dim}_confidence", 0.0)) for p in rule_preds_xlsx],
        ))
        # V1
        methods_data.append((
            "LLM (V1)",
            [p[dim] for p in v1_preds],
            v1_true[dim],
            [float(p.get(f"{dim}_confidence", 0.5)) for p in v1_preds],
        ))
        # Zero-shot
        methods_data.append((
            "Zero-shot",
            [p[dim] for p in zs_results[:n]],
            y_true[dim],
            [float(p.get(f"{dim}_confidence", 0.5)) for p in zs_results[:n]],
        ))

        save_path = os.path.join(FIGURES_DIR,
                                 f"reliability_combined_zero_shot_{dim}.png")
        combined_reliability_diagram(methods_data, dim, save_path)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Zero-shot results:  {ZERO_SHOT_RESULTS_PATH}")
    print(f"  Confusion matrix:   {FIGURES_DIR}/confusion_matrix_zero_shot_firm_type.png")
    print(f"  Reliability plots:  {FIGURES_DIR}/reliability_combined_zero_shot_*.png")
    zs_esc = compute_escalation_rate(zs_results[:n])
    v1_esc = compute_escalation_rate(v1_preds)
    print(f"  Escalation rate:    Zero-shot {zs_esc}% vs V1 {v1_esc}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
