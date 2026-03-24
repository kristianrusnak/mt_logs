import nltk
from nltk.translate.meteor_score import meteor_score
import json
import sys

# Ensure required resources are downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def extract_ground_true(json_path: str, json_key: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = list(data.values())[0]

    reasoning = []
    for i, item in enumerate(data):
        raw = item.get(json_key)
        reasoning_pos = raw.find("-")
        raw = raw[reasoning_pos + 1:].strip()
        reasoning.append(raw)

    return reasoning

def meteor_compare(ground_truth: str, prediction: str) -> float:
    return round(
        meteor_score([ground_truth.split()], prediction.split()),
        4
    )

def print_score(ground_truths: list[str], predictions: list[str], meteor_scores: list[float]) -> None:
    sep = "─" * 62

    avg   = sum(meteor_scores) / len(meteor_scores)
    best  = max(meteor_scores)
    worst = min(meteor_scores)

    print("\n" + "=" * 62)
    print("  LOG ANOMALY REASONING — METEOR RESULTS")
    print("=" * 62)

    # ── Aggregate stats ───────────────────────────────────────────
    print(f"\n  SUMMARY  ({len(meteor_scores)} samples)")
    print(f"  {sep}")
    print(f"  {'Average METEOR':<22} {avg:.4f}   ({avg*100:.1f}%)")
    print(f"  {'Best sentence METEOR':<22} {best:.4f}   ({best*100:.1f}%)")
    print(f"  {'Worst sentence METEOR':<22} {worst:.4f}   ({worst*100:.1f}%)")
    print(f"  {sep}")

    # ── Per-sample table ────────────────────────────────────────────
    print(f"\n  PER-SAMPLE DETAIL")
    print(f"  {sep}")

    def bar(v: float, w: int = 20) -> str:
        return "[" + "█" * round(v * w) + "░" * (w - round(v * w)) + "]"

    def preview(text: str, max_len: int = 50) -> str:
        return (text[:max_len] + "…") if len(text) > max_len else text

    for i, (gt, pred, score) in enumerate(zip(ground_truths, predictions, meteor_scores)):
        print(f"  #{i+1:>3}  METEOR={score:.4f}  {bar(score)}")
        print(f"       REF : {preview(gt)}")
        print(f"       HYP : {preview(pred)}")
        print()

def main() -> None:
    args = sys.argv[1:]

    if len(args) == 2:
        dataset_path = args[0]
        reasoning_path = args[1]

        ground_truth = extract_ground_true(dataset_path, "output")
        predictions = extract_ground_true(reasoning_path, "reasoning")

    else:
        print(__doc__)
        sys.exit(1)

    if len(ground_truth) != len(predictions):
        print(f"\nError: {len(predictions)} predictions but {len(ground_truth)} ground truth labels.")
        sys.exit(1)

    meteor_scores = []
    for gt, pred in zip(ground_truth, predictions):
        score = meteor_compare(gt, pred)
        meteor_scores.append(score)

    print_score(ground_truth, predictions, meteor_scores)

if __name__ == "__main__":
    main()