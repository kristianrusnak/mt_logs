from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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

def bleu_compare(ground_truth: str, prediction: str) -> int:
    reference_tokens = ground_truth.lower().split()  # tokenise reference
    hypothesis_tokens = prediction.lower().split()  # tokenise prediction

    # sentence_bleu requires references as a list-of-lists
    references = [reference_tokens]

    smoother = SmoothingFunction().method1

    return sentence_bleu(references, hypothesis_tokens, smoothing_function=smoother)

def print_score(ground_truths: list[str], predictions: list[str], bleu_scores: list[float]) -> None:
    sep = "─" * 62

    avg   = sum(bleu_scores) / len(bleu_scores)
    best  = max(bleu_scores)
    worst = min(bleu_scores)

    print("\n" + "=" * 62)
    print("  LOG ANOMALY REASONING — BLEU RESULTS")
    print("=" * 62)

    # ── Aggregate stats ───────────────────────────────────────────
    print(f"\n  SUMMARY  ({len(bleu_scores)} samples)")
    print(f"  {sep}")
    print(f"  {'Average BLEU':<22} {avg:.4f}   ({avg*100:.1f}%)")
    print(f"  {'Best sentence BLEU':<22} {best:.4f}   ({best*100:.1f}%)")
    print(f"  {'Worst sentence BLEU':<22} {worst:.4f}   ({worst*100:.1f}%)")
    print(f"  {sep}")

    # ── Interpretation ─────────────────────────────────────────────
    print(f"\n  INTERPRETATION  (single-reference, smoothed BLEU-4)")
    print(f"  {sep}")
    print(f"  > 0.40     Strong   — output closely mirrors reference phrasing")
    print(f"  0.20–0.40  Moderate — key terms match, wording differs")
    print(f"  0.10–0.20  Low      — some overlap, significant divergence")
    print(f"  < 0.10     Very low — little n-gram overlap")
    print(f"  {sep}")
    print(f"  ⚠  With one reference, paraphrases are penalised even when")
    print(f"     semantically correct. Scores tend to be conservative.")
    print(f"  {sep}")

    # ── Per-sample table ────────────────────────────────────────────
    print(f"\n  PER-SAMPLE DETAIL")
    print(f"  {sep}")

    def bar(v: float, w: int = 20) -> str:
        return "[" + "█" * round(v * w) + "░" * (w - round(v * w)) + "]"

    def preview(text: str, max_len: int = 50) -> str:
        return (text[:max_len] + "…") if len(text) > max_len else text

    for i, (gt, pred, score) in enumerate(zip(ground_truths, predictions, bleu_scores)):
        print(f"  #{i+1:>3}  BLEU={score:.4f}  {bar(score)}")
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

    bleu_scores = []
    for gt, pred in zip(ground_truth, predictions):
        score = bleu_compare(gt, pred)
        bleu_scores.append(score)

    print_score(ground_truth, predictions, bleu_scores)

if __name__ == "__main__":
    main()