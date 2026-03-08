# evaluate.py - evaluation pipeline for COM2019 internship classifier
#
# Compares random baseline vs rule-based baseline vs LLM classifier using:
# - Per-class precision, recall, F1 (sklearn classification_report)
# - Confusion matrix heatmaps
# - Confidence calibration analysis (binned accuracy vs stated confidence)
# - Expected Calibration Error (ECE)
# - Reliability diagrams
#
# All functions handle the case where gold_standard is None gracefully,
# printing a warning and returning early. This allows the evaluation
# module to be imported and called before manual labelling is complete.

import os
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from .taxonomy import FIRM_TYPES, ROLE_FUNCTIONS, PROGRAMME_STATUSES
from .baseline import rule_based_baseline, random_baseline


FIGURES_DIR = os.path.join("output", "figures")
GOLD_STANDARD_PATH = os.path.join("data", "gold_standard.json")

# Calibration bin edges: four bins covering low/medium/high/very-high confidence
CALIBRATION_BINS = [0.0, 0.3, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------------------
# 1. Load gold standard labels
# ---------------------------------------------------------------------------

def load_gold_standard(path=GOLD_STANDARD_PATH):
    """Load hand-labelled ground truth from JSON.

    Returns a list of label dicts, or None if the file is a placeholder
    or doesn't exist. Downstream functions check for None and skip gracefully.
    """
    if not os.path.exists(path):
        print("[evaluate] Gold standard file not found. Skipping evaluation.")
        return None

    with open(path) as f:
        data = json.load(f)

    # Check if this is still the placeholder file
    labels = data.get("labels", data) if isinstance(data, dict) else data

    if not isinstance(labels, list) or len(labels) < 5:
        print("[evaluate] Gold standard not yet labelled (placeholder detected). Skipping evaluation.")
        return None

    # Check for the placeholder comment marker
    if isinstance(data, dict) and "_comment" in data:
        print("[evaluate] Gold standard is still a placeholder. Skipping evaluation.")
        return None

    return labels


# ---------------------------------------------------------------------------
# 2. Per-class metrics (precision, recall, F1)
# ---------------------------------------------------------------------------

def evaluate_classifier(predictions, gold_standard, label=""):
    """Compute per-class precision, recall, F1 for a classification dimension.

    Args:
        predictions: list of predicted label strings.
        gold_standard: list of true label strings.
        label: descriptive name for printing (e.g. "LLM - firm_type").

    Returns:
        Dict with macro/weighted F1 and the full classification report,
        or None if gold_standard is None.
    """
    if gold_standard is None:
        print(f"[evaluate] Gold standard not yet available, skipping evaluation for {label}")
        return None

    all_labels = sorted(set(predictions + gold_standard))

    report_str = classification_report(
        gold_standard, predictions, labels=all_labels, zero_division=0
    )
    report_dict = classification_report(
        gold_standard, predictions, labels=all_labels, zero_division=0, output_dict=True
    )

    if label:
        print(f"\n--- {label} ---")
        print(report_str)

    return {
        "label": label,
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
        "report": report_str,
    }


# ---------------------------------------------------------------------------
# 3. Confusion matrix heatmap
# ---------------------------------------------------------------------------

def confusion_matrix_plot(predictions, gold_standard, label="firm_type", save_path=None):
    """Generate and save a confusion matrix heatmap.

    Returns the confusion matrix array, or None if gold_standard is None.
    """
    if gold_standard is None:
        print(f"[evaluate] Gold standard not yet available, skipping confusion matrix for {label}")
        return None

    all_labels = sorted(set(predictions + gold_standard))
    cm = confusion_matrix(gold_standard, predictions, labels=all_labels)

    fig, ax = plt.subplots(figsize=(max(8, len(all_labels)), max(6, len(all_labels) * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix: {label}")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Annotate cells with counts
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()

    if save_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{label}.png")

    fig.savefig(save_path, dpi=150)
    print(f"  [evaluate] Saved confusion matrix to {save_path}")
    plt.close(fig)

    return cm


# ---------------------------------------------------------------------------
# 4. Confidence calibration analysis
# ---------------------------------------------------------------------------

def calibration_analysis(predictions, gold_standard, confidences, label=""):
    """Group predictions by confidence band and compute actual accuracy.

    Compares stated confidence against real accuracy to assess whether
    the classifier is well-calibrated (i.e. 80% confidence = 80% correct).

    Returns dict with bin stats, or None if gold_standard is None.
    """
    if gold_standard is None:
        print(f"[evaluate] Gold standard not yet available, skipping calibration for {label}")
        return None

    bins = []
    total = len(gold_standard)

    for i in range(len(CALIBRATION_BINS) - 1):
        lo, hi = CALIBRATION_BINS[i], CALIBRATION_BINS[i + 1]
        # Last bin is inclusive on both ends
        if i < len(CALIBRATION_BINS) - 2:
            mask = [(lo <= c < hi) for c in confidences]
        else:
            mask = [(lo <= c <= hi) for c in confidences]

        bin_true = [t for t, m in zip(gold_standard, mask) if m]
        bin_pred = [p for p, m in zip(predictions, mask) if m]
        bin_conf = [c for c, m in zip(confidences, mask) if m]

        n = len(bin_true)
        if n == 0:
            bins.append({"range": f"{lo:.1f}-{hi:.1f}", "n": 0, "accuracy": 0.0, "avg_conf": 0.0})
            continue

        accuracy = sum(1 for t, p in zip(bin_true, bin_pred) if t == p) / n
        avg_conf = sum(bin_conf) / n

        bins.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": n,
            "accuracy": round(accuracy, 4),
            "avg_conf": round(avg_conf, 4),
        })

    if label:
        print(f"\n--- Calibration: {label} ---")
        for b in bins:
            print(f"  {b['range']}: n={b['n']}, accuracy={b['accuracy']:.2f}, avg_conf={b['avg_conf']:.2f}")

    return {"bins": bins}


# ---------------------------------------------------------------------------
# 5. Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def calculate_ece(predictions, gold_standard, confidences):
    """Compute Expected Calibration Error.

    ECE = sum over bins of (bin_weight * |accuracy - avg_confidence|).
    Lower is better; 0.0 means perfectly calibrated.

    Returns float ECE value, or None if gold_standard is None.
    """
    if gold_standard is None:
        print("[evaluate] Gold standard not yet available, skipping ECE.")
        return None

    cal = calibration_analysis(predictions, gold_standard, confidences)
    if cal is None:
        return None

    total = len(gold_standard)
    ece = 0.0
    for b in cal["bins"]:
        if b["n"] > 0:
            ece += (b["n"] / total) * abs(b["accuracy"] - b["avg_conf"])

    return round(ece, 4)


# ---------------------------------------------------------------------------
# 6. Reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram(predictions, gold_standard, confidences, label="", save_path=None):
    """Plot a reliability diagram showing calibration quality.

    Perfect calibration = diagonal line. Bars above = underconfident,
    bars below = overconfident.

    Returns None if gold_standard is None.
    """
    if gold_standard is None:
        print(f"[evaluate] Gold standard not yet available, skipping reliability diagram for {label}")
        return None

    cal = calibration_analysis(predictions, gold_standard, confidences)
    if cal is None:
        return None

    ece = calculate_ece(predictions, gold_standard, confidences)

    avg_confs = [b["avg_conf"] for b in cal["bins"] if b["n"] > 0]
    accuracies = [b["accuracy"] for b in cal["bins"] if b["n"] > 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(avg_confs, accuracies, width=0.15, alpha=0.6, label="Model", edgecolor="black")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of correct predictions")
    ax.set_title(f"Reliability Diagram: {label}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(FIGURES_DIR, f"reliability_{label}.png")

    fig.savefig(save_path, dpi=150)
    print(f"  [evaluate] Saved reliability diagram to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Full evaluation runner
# ---------------------------------------------------------------------------

def run_full_evaluation(data_path="data/sample_listings.csv",
                        baseline_method="rule",
                        llm_results_available=False):
    """Run complete evaluation: random baseline vs rule-based vs LLM.

    Loads gold standard, generates predictions from each method, computes
    metrics, saves figures to output/figures/. Handles missing gold standard
    gracefully.
    """
    print(f"\n{'=' * 60}")
    print("EVALUATION PIPELINE")
    print(f"{'=' * 60}")

    # Load gold standard
    gold_labels = load_gold_standard()
    if gold_labels is None:
        return

    # Load data to generate baseline predictions
    df = pd.read_csv(data_path)
    df = df.fillna("")
    listings = df.to_dict(orient="records")

    # Match gold labels to listings by company_name
    # (gold standard may only cover a subset of listings)
    gold_companies = {g["company_name"] for g in gold_labels}
    matched_listings = [l for l in listings if l.get("company_name", "") in gold_companies]

    if len(matched_listings) < 5:
        print(f"[evaluate] Only {len(matched_listings)} listings match gold standard. Need at least 5.")
        return

    print(f"[evaluate] Matched {len(matched_listings)} listings to gold standard labels")

    # Generate predictions
    random_preds = random_baseline(matched_listings)
    rule_preds = rule_based_baseline(matched_listings)

    # Build gold label lookup
    gold_lookup = {g["company_name"]: g for g in gold_labels}

    summaries = []

    for method_name, preds in [("Random", random_preds), ("Rule-based", rule_preds)]:
        print(f"\n{'=' * 60}")
        print(f"EVALUATION: {method_name}")
        print(f"{'=' * 60}")

        summary = {"method": method_name}

        for dimension in ["firm_type", "role_function", "programme_status"]:
            y_true = [gold_lookup[l["company_name"]][dimension] for l in matched_listings]
            y_pred = [p[dimension] for p in preds]
            conf_key = f"{dimension}_confidence"
            confs = [p.get(conf_key, 0.5) for p in preds]

            metrics = evaluate_classifier(y_pred, y_true, label=f"{method_name} - {dimension}")
            if metrics:
                summary[f"{dimension}_macro_f1"] = metrics["macro_f1"]
                summary[f"{dimension}_weighted_f1"] = metrics["weighted_f1"]

            confusion_matrix_plot(y_pred, y_true, label=f"{method_name}_{dimension}")

            cal = calibration_analysis(y_pred, y_true, confs, label=f"{method_name} - {dimension}")
            if cal:
                ece = calculate_ece(y_pred, y_true, confs)
                summary[f"{dimension}_ece"] = ece
                reliability_diagram(y_pred, y_true, confs, label=f"{method_name}_{dimension}")

        summaries.append(summary)

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}")
    header = f"{'Method':<15} | {'Firm F1':>8} | {'Role F1':>8} | {'Status F1':>10} | {'Firm ECE':>9} | {'Role ECE':>9} | {'Status ECE':>11}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['method']:<15} | "
            f"{s.get('firm_type_macro_f1', 0):.4f}   | "
            f"{s.get('role_function_macro_f1', 0):.4f}   | "
            f"{s.get('programme_status_macro_f1', 0):.4f}     | "
            f"{s.get('firm_type_ece', 0):.4f}    | "
            f"{s.get('role_function_ece', 0):.4f}    | "
            f"{s.get('programme_status_ece', 0):.4f}"
        )
    print(f"{'=' * 80}")

    return summaries
