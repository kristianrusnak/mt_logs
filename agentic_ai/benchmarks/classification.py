"""
Log Anomaly Detection — Benchmark Calculator
=============================================
Calculates F1, Recall, Precision and Accuracy by comparing
model's predictions against the ground truth labels
from seed_Thunderbird.json (LLM-LADE dataset).
"""

import json
import sys
import os


# ─────────────────────────────────────────────
#  What do TP / TN / FP / FN mean?
# ─────────────────────────────────────────────
#
#  We treat "anomaly" as the POSITIVE class (what we care about detecting).
#
#  For each log sample we compare ground truth vs AI prediction:
#
#   Ground Truth │  AI Prediction  │  Name  │ Meaning
#  ──────────────┼─────────────────┼────────┼──────────────────────────────────
#     anomaly    │    anomaly      │  TP    │ True Positive  — correctly caught
#     normal     │    normal       │  TN    │ True Negative  — correctly ignored
#     normal     │    anomaly      │  FP    │ False Positive — false alarm
#     anomaly    │    normal       │  FN    │ False Negative — missed anomaly
#  ──────────────┴─────────────────┴────────┴──────────────────────────────────


def normalize_label(value: str) -> int:
    v = str(value).strip().lower()
    sep = v.index('-')
    classification = v[:sep]
    if classification == "abnormal":
        return 1
    elif classification == "normal":
        return 0
    else:
        raise ValueError(f"Cannot parse label: '{value}'")


def load_ground_truth(json_path: str) -> list[int]:
    """Extract ground-truth labels from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The dataset may be a plain list or wrapped in a dict key
    if isinstance(data, dict):
        data = data.get("data") or data.get("samples") or list(data.values())[0]

    labels = []
    for i, item in enumerate(data):
        raw = item.get("output")
        if raw is None:
            raise KeyError(f"No label field found in item {i}. Keys: {list(item.keys())}")
        labels.append(normalize_label(raw))

    return labels


def load_predictions_from_file(path: str) -> list[int]:
    """Extract predictions by AI from a JSON file — must be same order as ground truth."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("data") or data.get("samples") or list(data.values())[0]

    labels = []
    for i, item in enumerate(data):
        raw = item.get("reasoning")
        if raw is None:
            raise KeyError(f"No verdict field found in item {i}. Keys: {list(item.keys())}")
        labels.append(normalize_label(raw))

    return labels

# ─────────────────────────────────────────────
#  Count TP, TN, FP, FN
# ─────────────────────────────────────────────

def confusion_matrix(ground_truth: list[int], predictions: list[int]) -> dict:
    """Count the four outcomes across all samples."""
    tp = tn = fp = fn = 0
    for gt, pred in zip(ground_truth, predictions):
        if gt == 1 and pred == 1:
            tp += 1   # anomaly correctly flagged
        elif gt == 0 and pred == 0:
            tn += 1   # normal correctly ignored
        elif gt == 0 and pred == 1:
            fp += 1   # false alarm — predicted anomaly but it was normal
        else:
            fn += 1   # miss — predicted normal but it was actually anomaly
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ─────────────────────────────────────────────
#  Calculate the four metrics
# ─────────────────────────────────────────────
#
#  PRECISION  = TP / (TP + FP)
#    "Of all the times I yelled ANOMALY, how often was I right?"
#    Low precision → too many false alarms.
#
#  RECALL     = TP / (TP + FN)
#    "Of all the real anomalies, how many did I catch?"
#    Low recall → missing too many real anomalies (dangerous!).
#
#  F1 SCORE   = 2 × (Precision × Recall) / (Precision + Recall)
#    Harmonic mean — penalises you if EITHER metric is low.
#    Use this as your primary score when classes are imbalanced.
#
#  ACCURACY   = (TP + TN) / Total
#    "What fraction of all predictions were correct?"
#    Warning: misleading on imbalanced datasets.
#    E.g. if 95% of logs are normal, predicting "always normal"
#    gives 95% accuracy but 0% recall — useless for anomaly detection.

def calculate_metrics(cm: dict) -> dict:
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    return {
        "Precision": precision,
        "Recall":    recall,
        "F1":        f1,
        "Accuracy":  accuracy,
    }


# ─────────────────────────────────────────────
#  Pretty-print everything
# ─────────────────────────────────────────────

def print_results(cm: dict, metrics: dict, ground_truth: list, predictions: list):
    total = sum(cm.values())
    sep   = "─" * 52

    print("\n" + "=" * 52)
    print("  LOG ANOMALY DETECTION — BENCHMARK RESULTS")
    print("=" * 52)

    # ── Confusion matrix ──────────────────────────────
    print("\n  CONFUSION MATRIX")
    print(f"  {sep}")
    print(f"  {'':20s}  {'Pred: Normal':>12}  {'Pred: Anomaly':>13}")
    print(f"  {sep}")
    print(f"  {'Actual: Normal':20s}  {'TN = ' + str(cm['TN']):>12}  {'FP = ' + str(cm['FP']):>13}")
    print(f"  {'Actual: Anomaly':20s}  {'FN = ' + str(cm['FN']):>12}  {'TP = ' + str(cm['TP']):>13}")
    print(f"  {sep}")

    # ── Metric formulas + values ───────────────────────
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    tn         = cm["TN"]

    print("\n  METRIC CALCULATIONS")
    print(f"  {sep}")
    print(f"  Precision = TP / (TP + FP)")
    print(f"            = {tp} / ({tp} + {fp}) = {metrics['Precision']:.4f}")
    print()
    print(f"  Recall    = TP / (TP + FN)")
    print(f"            = {tp} / ({tp} + {fn}) = {metrics['Recall']:.4f}")
    print()
    print(f"  F1 Score  = 2 × Precision × Recall / (Precision + Recall)")
    print(f"            = 2 × {metrics['Precision']:.4f} × {metrics['Recall']:.4f} / "
          f"({metrics['Precision']:.4f} + {metrics['Recall']:.4f})")
    print(f"            = {metrics['F1']:.4f}")
    print()
    print(f"  Accuracy  = (TP + TN) / Total")
    print(f"            = ({tp} + {tn}) / {total} = {metrics['Accuracy']:.4f}")
    print(f"  {sep}")

    # ── Summary ───────────────────────────────────────
    print("\n  SUMMARY")
    print(f"  {sep}")
    print(f"  {'F1 Score':<20} {metrics['F1']*100:>7.2f}%   {bar(metrics['F1'])}")
    print(f"  {'Recall':<20} {metrics['Recall']*100:>7.2f}%   {bar(metrics['Recall'])}")
    print(f"  {'Precision':<20} {metrics['Precision']*100:>7.2f}%   {bar(metrics['Precision'])}")
    print(f"  {'Accuracy':<20} {metrics['Accuracy']*100:>7.2f}%   {bar(metrics['Accuracy'])}")
    print(f"  {sep}")

    # ── Label distribution ────────────────────────────
    gt_a = sum(ground_truth)
    gt_n = len(ground_truth) - gt_a
    pr_a = sum(predictions)
    pr_n = len(predictions) - pr_a

    print("\n  LABEL DISTRIBUTION")
    print(f"  {sep}")
    print(f"  Ground Truth  →  anomaly: {gt_a:5d}  |  normal: {gt_n:5d}  |  total: {total:5d}")
    print(f"  Predictions   →  anomaly: {pr_a:5d}  |  normal: {pr_n:5d}  |  total: {total:5d}")
    print(f"  {sep}")

    # ── Interpretation tip ────────────────────────────
    print("\n  INTERPRETATION")
    if metrics["Recall"] < 0.5:
        print("  ⚠  Low Recall — your model misses many real anomalies (high FN).")
    if metrics["Precision"] < 0.5:
        print("  ⚠  Low Precision — your model raises many false alarms (high FP).")
    if metrics["F1"] >= 0.85:
        print("  ✓  Strong F1 — good balance of precision and recall.")
    elif metrics["F1"] >= 0.6:
        print("  ~  Moderate F1 — room for improvement.")
    else:
        print("  ✗  Low F1 — model needs significant improvement.")
    print()


def bar(value: float, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if len(args) == 2:
        # ── Real dataset + predictions file ──────────
        dataset_path  = args[0]
        prediction_path = args[1]
        ground_truth = load_ground_truth(dataset_path)
        predictions  = load_predictions_from_file(prediction_path)
    else:
        print(__doc__)
        sys.exit(1)

    # Validate length
    if len(ground_truth) != len(predictions):
        print(f"\nError: {len(predictions)} predictions but {len(ground_truth)} ground truth labels.")
        sys.exit(1)

    # Calculate and print
    cm = confusion_matrix(ground_truth, predictions)
    metrics = calculate_metrics(cm)
    print_results(cm, metrics, ground_truth, predictions)


if __name__ == "__main__":
    main()