"""
Log Anomaly Reasoning — OpenAI Embedding Benchmark
====================================================
Measures semantic similarity between your model's reasoning outputs
and the reference reasoning using OpenAI's text-embedding-3-large.

Works identically to the Sentence-BERT version but uses OpenAI's API
instead of a local model — no local download needed, but requires an
OpenAI API key and incurs a small cost per run.

Usage:
    python openai_benchmark.py seed_Thunderbird.json my_outputs.json

Requirements:
    pip install openai

API key:
    Set the OPENAI_API_KEY environment variable before running:
        export OPENAI_API_KEY="sk-..."
    Or place it in a .env file and load with python-dotenv.

Cost estimate:
    text-embedding-3-large: $0.13 per 1M tokens
    A typical log reasoning sentence is ~30 tokens.
    100 samples (ref + hyp) = ~6000 tokens ≈ $0.001
"""

from openai import OpenAI
import json
import sys
import math
import os


# ──────────────────────────────────────────────────────────────────────
#  HOW THIS DIFFERS FROM SENTENCE-BERT
# ──────────────────────────────────────────────────────────────────────
#
#  Sentence-BERT  — local model, free, ~384-dim embeddings, fast
#  text-embedding-3-large — API call, paid, 3072-dim embeddings,
#                           generally stronger on technical/domain text
#
#  Both produce embeddings and compare them with cosine similarity —
#  the math is identical, only the embedding source changes.
#
#  text-embedding-3-large produces 3072-dimensional vectors by default,
#  which capture finer-grained semantic distinctions than smaller models.
#  You can reduce dimensions via the `dimensions` parameter if needed.

MODEL_NAME = "text-embedding-3-large"


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    return OpenAI(api_key=api_key)


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


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in one API call.

    OpenAI's embeddings endpoint accepts up to 2048 inputs per request.
    For very large datasets this function chunks automatically.
    """
    BATCH_SIZE = 100   # stay well within API limits
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(
            model=MODEL_NAME,
            input=batch,
        )
        # response.data is sorted by index, safe to extend in order
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} texts...")

    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors — identical math to Sentence-BERT version.

        cosine_similarity(A, B) = (A · B) / (|A| × |B|)

    OpenAI embeddings are already L2-normalised, so |A| = |B| = 1
    and this simplifies to just the dot product. We compute it fully
    anyway for correctness in case that changes.
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def openai_compare(
    ground_truths: list[str],
    predictions: list[str],
    client: OpenAI,
) -> list[float]:
    """
    Embed all references and hypotheses, then compute cosine similarity
    for each pair.
    """
    print("  Embedding reference sentences...")
    ref_embeddings = embed_batch(client, ground_truths)

    print("  Embedding hypothesis sentences...")
    hyp_embeddings = embed_batch(client, predictions)

    scores = []
    for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings):
        score = cosine_similarity(ref_emb, hyp_emb)
        scores.append(round(score, 4))

    return scores


def print_score(
    ground_truths: list[str],
    predictions: list[str],
    scores: list[float],
) -> None:
    sep = "─" * 66

    avg   = sum(scores) / len(scores)
    best  = max(scores)
    worst = min(scores)

    best_idx  = scores.index(best)
    worst_idx = scores.index(worst)

    def bar(v: float, w: int = 20) -> str:
        filled = round(max(v, 0.0) * w)
        return "[" + "█" * filled + "░" * (w - filled) + "]"

    def preview(text: str, max_len: int = 55) -> str:
        return (text[:max_len] + "…") if len(text) > max_len else text

    print("\n" + "=" * 66)
    print(f"  LOG ANOMALY REASONING — OPENAI {MODEL_NAME} RESULTS")
    print("=" * 66)

    # ── Overview ──────────────────────────────────────────────────
    print(f"\n  OVERVIEW  ({len(scores)} samples)")
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

    for i, (gt, pred, score) in enumerate(zip(ground_truths, predictions, scores)):
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

    client = get_client()
    scores = openai_compare(ground_truth, predictions, client)
    print_score(ground_truth, predictions, scores)


if __name__ == "__main__":
    main()