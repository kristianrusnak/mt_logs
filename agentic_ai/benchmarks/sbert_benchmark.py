"""
Log Anomaly Reasoning — Sentence-BERT Benchmark
================================================
Measures semantic similarity between your model's reasoning outputs
and the reference reasoning using Sentence-BERT embeddings.

Unlike BLEU / ROUGE / METEOR which count word overlaps, Sentence-BERT
encodes each sentence into a dense vector (embedding) that captures
*meaning*, so paraphrases score high even with no shared words.

Usage:
    python sbert_benchmark.py seed_Thunderbird.json my_outputs.json

Requirements:
    pip install sentence-transformers
"""

from sentence_transformers import SentenceTransformer, util
import json
import sys


# ──────────────────────────────────────────────────────────────────────
#  HOW SENTENCE-BERT WORKS
# ──────────────────────────────────────────────────────────────────────
#
#  1. A pretrained BERT model reads a sentence and produces a fixed-size
#     vector (embedding) — typically 384 or 768 numbers — that encodes
#     its meaning. Similar sentences produce similar vectors.
#
#  2. We compare two embeddings using COSINE SIMILARITY:
#
#       cosine_similarity(A, B) = (A · B) / (|A| × |B|)
#
#     This measures the *angle* between the two vectors:
#       1.0  → identical meaning
#       0.0  → completely unrelated
#      -1.0  → opposite meaning (rare in practice)
#
#  3. No word overlap is needed. This pair scores ~0.91:
#       REF : "The kernel detected a fatal disk I/O error"
#       HYP : "A storage device failure was identified by the OS"
#
#  Model used: "all-MiniLM-L6-v2"
#    - Small (80MB), fast, strong performance on semantic similarity
#    - Downloaded automatically on first run from HuggingFace Hub
#    - Other good options: "all-mpnet-base-v2" (slower, more accurate)


# MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = "all-mpnet-base-v2"

def load_model() -> SentenceTransformer:
    print(f"  Loading model '{MODEL_NAME}' (downloads once on first run)...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model ready.\n")
    return model


def extract_ground_true(json_path: str, json_key: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = list(data.values())[0]

    reasoning = []
    for i, item in enumerate(data):
        raw = item.get(json_key)
        if raw is None:
            raise KeyError(f"Key '{json_key}' not found in item {i}. Available keys: {list(item.keys())}")
        reasoning_pos = raw.find("-")
        raw = raw[reasoning_pos + 1:].strip()
        reasoning.append(raw)

    return reasoning


def sbert_compare(
    ground_truths: list[str],
    predictions: list[str],
    model: SentenceTransformer,
) -> list[float]:
    """
    Encode all references and hypotheses in one batched pass (fast),
    then compute cosine similarity for each pair.

    Returns a list of similarity scores in [0.0, 1.0].
    """
    print("  Encoding reference sentences...")
    ref_embeddings  = model.encode(ground_truths, convert_to_tensor=True, show_progress_bar=True)

    print("  Encoding hypothesis sentences...")
    hyp_embeddings  = model.encode(predictions,   convert_to_tensor=True, show_progress_bar=True)

    scores = []
    for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings):
        # util.cos_sim returns a 1×1 tensor — .item() extracts the float
        score = util.cos_sim(ref_emb, hyp_emb).item()
        scores.append(round(score, 4))

    return scores


def print_score(
    ground_truths: list[str],
    predictions: list[str],
    sbert_scores: list[float],
) -> None:
    sep = "─" * 66

    avg   = sum(sbert_scores) / len(sbert_scores)
    best  = max(sbert_scores)
    worst = min(sbert_scores)

    # indices of best and worst samples
    best_idx  = sbert_scores.index(best)
    worst_idx = sbert_scores.index(worst)

    def bar(v: float, w: int = 20) -> str:
        filled = round(max(v, 0.0) * w)
        return "[" + "█" * filled + "░" * (w - filled) + "]"

    def preview(text: str, max_len: int = 55) -> str:
        return (text[:max_len] + "…") if len(text) > max_len else text

    print("\n" + "=" * 66)
    print("  LOG ANOMALY REASONING — SENTENCE-BERT RESULTS")
    print("=" * 66)

    # ── Overview ──────────────────────────────────────────────────
    print(f"\n  OVERVIEW  ({len(sbert_scores)} samples)")
    print(f"  {sep}")
    print(f"  {'Average similarity':<26} {avg:.4f}   {bar(avg)}")
    print(f"  {'Best similarity':<26} {best:.4f}   {bar(best)}")
    print(f"  {'Worst similarity':<26} {worst:.4f}   {bar(worst)}")
    print(f"  {sep}")

    # ── Interpretation ─────────────────────────────────────────────
    print(f"\n  INTERPRETATION  (cosine similarity, higher = more similar)")
    print(f"  {sep}")
    print(f"  0.90 – 1.00   Near-identical meaning")
    print(f"  0.75 – 0.90   Strong semantic match")
    print(f"  0.50 – 0.75   Related but meaningfully different")
    print(f"  0.00 – 0.50   Weak or no semantic overlap")
    print(f"  {sep}")

    # ── Best / worst highlight ─────────────────────────────────────
    print(f"\n  BEST MATCH  (#{best_idx + 1}, score={best:.4f})")
    print(f"  {sep}")
    print(f"  REF : {preview(ground_truths[best_idx])}")
    print(f"  HYP : {preview(predictions[best_idx])}")

    print(f"\n  WORST MATCH  (#{worst_idx + 1}, score={worst:.4f})")
    print(f"  {sep}")
    print(f"  REF : {preview(ground_truths[worst_idx])}")
    print(f"  HYP : {preview(predictions[worst_idx])}")

    # ── Per-sample detail ──────────────────────────────────────────
    print(f"\n  PER-SAMPLE DETAIL")

    for i, (gt, pred, score) in enumerate(zip(ground_truths, predictions, sbert_scores)):
        print(f"\n  {sep}")
        print(f"  Sample #{i + 1}   similarity={score:.4f}   {bar(score)}")
        print(f"  REF : {preview(gt)}")
        print(f"  HYP : {preview(pred)}")

    print(f"\n  {sep}")


def main() -> None:
    args = sys.argv[1:]

    if len(args) == 2:
        dataset_path   = args[0]
        reasoning_path = args[1]

        ground_truth = extract_ground_true(dataset_path, "output")
        predictions  = extract_ground_true(reasoning_path, "reasoning")

    else:
        print(__doc__)
        sys.exit(1)

    if len(ground_truth) != len(predictions):
        print(f"\nError: {len(predictions)} predictions but {len(ground_truth)} ground truth labels.")
        sys.exit(1)

    model       = load_model()
    sbert_scores = sbert_compare(ground_truth, predictions, model)
    print_score(ground_truth, predictions, sbert_scores)


if __name__ == "__main__":
    main()