from rouge import Rouge
import json
import sys

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

def rouge_compare(ground_truth: str, prediction: str) -> int:
    rouge = Rouge()
    scores = rouge.get_scores(prediction, ground_truth)
    return scores


def print_score(ground_truths: list[str], predictions: list[str], rouge_scores: list[list[dict]]) -> None:
    # rouge_scores: each element is what your ROUGE library returns for one pair:
    #   [ { "rouge-1": {"f": ..., "p": ..., "r": ...},
    #       "rouge-2": {"f": ..., "p": ..., "r": ...},
    #       "rouge-l": {"f": ..., "p": ..., "r": ...} } ]

    sep = "─" * 66
    sep2 = "─" * 66

    variants = [("ROUGE-1", "rouge-1"), ("ROUGE-2", "rouge-2"), ("ROUGE-L", "rouge-l")]

    def get(rscore, variant, field):
        return rscore[0][variant][field]

    def avg(variant, field):
        return sum(get(r, variant, field) for r in rouge_scores) / len(rouge_scores)

    def preview(text: str, max_len: int = 50) -> str:
        return (text[:max_len] + "…") if len(text) > max_len else text

    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  LOG ANOMALY REASONING — ROUGE RESULTS")
    print("=" * 66)

    # ── Overview (averages across all samples) ────────────────────
    print(f"\n  OVERVIEW  ({len(rouge_scores)} samples — averages)")
    print(f"  {sep}")
    print(f"  {'Metric':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {sep}")
    for label, key in variants:
        p = avg(key, "p")
        r = avg(key, "r")
        f1 = avg(key, "f")
        print(f"  {label:<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")
    print(f"  {sep}")

    # ── Per-sample detail ─────────────────────────────────────────
    print(f"\n  PER-SAMPLE DETAIL")

    for i, (gt, pred, rscore) in enumerate(zip(ground_truths, predictions, rouge_scores)):
        print(f"\n  {sep2}")
        print(f"  Sample #{i + 1}")
        print(f"  REF : {preview(gt)}")
        print(f"  HYP : {preview(pred)}")
        print(f"  {sep2}")
        print(f"  {'Metric':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {sep2}")
        for label, key in variants:
            p = get(rscore, key, "p")
            r = get(rscore, key, "r")
            f1 = get(rscore, key, "f")
            print(f"  {label:<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

    print(f"\n  {sep2}")


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

    rouge_scores = []
    for gt, pred in zip(ground_truth, predictions):
        score = rouge_compare(gt, pred)
        rouge_scores.append(score)

    print_score(ground_truth, predictions, rouge_scores)

if __name__ == "__main__":
    main()