"""
LangGraph Log Anomaly Detection Pipeline
=========================================
3-node pipeline:
  Node 1 — log_parser_node:       Parse seed_Thunderbird.json, extract fields,
                                   split labels from content.
  Node 2 — anomaly_reasoner_node: LLM classifies each entry as normal/abnormal
                                   with a chain-of-thought reasoning string.
  Node 3 — comparator_node:       Format output labels, diff against ground truth,
                                   print accuracy report.

Usage:
    pip install langgraph langchain langchain-anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python pipeline.py --input seed_Thunderbird.json [--max-entries 20]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


# ─────────────────────────────────────────────
# 1.  Shared state
# ─────────────────────────────────────────────

class LogState(TypedDict):
    """
    Shared state dict flowing through all nodes.

    raw_entries    : list of log sequence loaded straight from the JSON seed file
    parsed_entries : list of dicts with normalised fields extracted by Node 1
    ground_truth   : list of int  (0 = normal, 1 = abnormal) from seed labels
    predictions    : list of dicts produced by Node 2
                      { "label": "normal"|"abnormal", "reasoning": str }
    comparison     : summary dict produced by Node 3
    """
    raw_entries: list[str]
    parsed_entries: list[dict]
    ground_truth: list[int]
    predictions: list[dict]
    comparison: dict


# ─────────────────────────────────────────────
# 2.  Node 1 — Log parser & context builder
# ─────────────────────────────────────────────

_TIMESTAMP_RE = re.compile(
    r"^(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)"
)
_LEVEL_RE = re.compile(r"\b(ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL)\b", re.I)


def _parse_log_line(raw: str) -> dict:
    """
    Extract timestamp, hostname, level, and message body from a raw log line.
    Falls back gracefully when the line doesn't match the standard syslog format.
    """
    line = raw.strip()
    m = _TIMESTAMP_RE.match(line)
    if m:
        timestamp, hostname, rest = m.group(1), m.group(2), m.group(3)
    else:
        timestamp, hostname, rest = "", "", line

    level_m = _LEVEL_RE.search(rest)
    level = level_m.group(0).upper() if level_m else "UNKNOWN"

    return {
        "raw": line,
        "timestamp": timestamp,
        "hostname": hostname,
        "level": level,
        "message": rest,
    }


def log_parser_node(state: LogState) -> LogState:
    """
    Node 1 — Parse raw seed entries into structured dicts.

    Reads:  state["raw_entries"]
    Writes: state["parsed_entries"], state["ground_truth"]
    """
    parsed = []
    labels = []

    for entry in state["raw_entries"]:
        # Seed JSON schema:  {"label": 0|1, "content": "<log line(s)>"}
        label = int(entry.get("label", 0))
        content = entry.get("content", "")

        parsed.append(_parse_log_line(content))
        labels.append(label)

    print(f"[Node 1] Parsed {len(parsed)} log entries.")
    return {**state, "parsed_entries": parsed, "ground_truth": labels}


# ─────────────────────────────────────────────
# 3.  Node 2 — Anomaly reasoner (LLM)
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a log analysis expert.
    You will be given a single parsed system log entry.
    Your task is to classify it as **normal** or **abnormal**, then briefly explain why.

    Rules:
    - "abnormal" means the log indicates an error, failure, crash, security issue,
      unexpected state, or anything that warrants investigation.
    - "normal" means routine operation with no signs of problems.
    - Respond ONLY in the following JSON format, with no extra text:

    {"label": "normal", "reasoning": "<one or two sentences>"}
    or
    {"label": "abnormal", "reasoning": "<one or two sentences>"}
""").strip()


def _build_user_prompt(entry: dict) -> str:
    return (
        f"Timestamp : {entry['timestamp'] or 'n/a'}\n"
        f"Hostname  : {entry['hostname'] or 'n/a'}\n"
        f"Level     : {entry['level']}\n"
        f"Message   : {entry['message']}\n"
    )


def _call_llm(llm: ChatAnthropic, entry: dict) -> dict:
    """Call the LLM for a single log entry, parse JSON response."""
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(entry)),
    ]
    response = llm.invoke(messages)
    raw_text = response.content.strip()

    # Strip markdown code fences if present
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.S).strip()

    try:
        result = json.loads(clean)
        label = result.get("label", "normal").lower()
        if label not in ("normal", "abnormal"):
            label = "normal"
        return {"label": label, "reasoning": result.get("reasoning", "")}
    except json.JSONDecodeError:
        # Graceful fallback: scan the raw text for the label word
        label = "abnormal" if "abnormal" in raw_text.lower() else "normal"
        return {"label": label, "reasoning": raw_text}


def anomaly_reasoner_node(state: LogState) -> LogState:
    """
    Node 2 — Call an LLM for each parsed entry to get label + reasoning.

    Reads:  state["parsed_entries"]
    Writes: state["predictions"]
    """
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    predictions = []
    total = len(state["parsed_entries"])

    for i, entry in enumerate(state["parsed_entries"], 1):
        print(f"[Node 2] Classifying entry {i}/{total} …", end="\r")
        pred = _call_llm(llm, entry)
        predictions.append(pred)

    print(f"\n[Node 2] Finished classifying {total} entries.")
    return {**state, "predictions": predictions}


# ─────────────────────────────────────────────
# 4.  Node 3 — Label formatter & comparator
# ─────────────────────────────────────────────

def comparator_node(state: LogState) -> LogState:
    """
    Node 3 — Format predictions as <label>-<reasoning>, compare to ground truth.

    Reads:  state["predictions"], state["ground_truth"], state["parsed_entries"]
    Writes: state["comparison"]
    """
    preds = state["predictions"]
    truth = state["ground_truth"]
    entries = state["parsed_entries"]

    tp = tn = fp = fn = 0
    output_lines = []

    for i, (pred, gt) in enumerate(zip(preds, truth)):
        label_str = pred["label"]          # "normal" | "abnormal"
        reasoning = pred["reasoning"]
        formatted = f"{label_str}-{reasoning}"

        pred_bin = 1 if label_str == "abnormal" else 0
        match = pred_bin == gt

        if pred_bin == 1 and gt == 1:
            tp += 1
        elif pred_bin == 0 and gt == 0:
            tn += 1
        elif pred_bin == 1 and gt == 0:
            fp += 1
        else:
            fn += 1

        output_lines.append({
            "index": i,
            "raw": entries[i]["raw"][:80],
            "ground_truth": "abnormal" if gt == 1 else "normal",
            "prediction": label_str,
            "reasoning": reasoning,
            "match": match,
            "formatted": formatted,
        })

    total = len(preds)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    comparison = {
        "total": total,
        "correct": tp + tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "details": output_lines,
    }

    _print_report(comparison)
    return {**state, "comparison": comparison}


def _print_report(c: dict) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print("  ANOMALY DETECTION REPORT")
    print(sep)
    print(f"  Total entries : {c['total']}")
    print(f"  Correct       : {c['correct']}")
    print(f"  Accuracy      : {c['accuracy']:.1%}")
    print(f"  Precision     : {c['precision']:.1%}")
    print(f"  Recall        : {c['recall']:.1%}")
    print(f"  F1 score      : {c['f1']:.1%}")
    print(f"  TP={c['tp']}  TN={c['tn']}  FP={c['fp']}  FN={c['fn']}")
    print(sep)

    mismatches = [d for d in c["details"] if not d["match"]]
    if mismatches:
        print(f"\n  MISMATCHES ({len(mismatches)}):")
        for d in mismatches:
            print(f"\n  [{d['index']}] {d['raw']} …")
            print(f"       GT={d['ground_truth']}  PRED={d['prediction']}")
            print(f"       Reason: {d['reasoning'][:120]}")
    else:
        print("\n  No mismatches — perfect score!")
    print(sep)


# ─────────────────────────────────────────────
# 5.  Build the LangGraph
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(LogState)

    graph.add_node("log_parser", log_parser_node)
    graph.add_node("anomaly_reasoner", anomaly_reasoner_node)
    graph.add_node("comparator", comparator_node)

    graph.add_edge(START, "log_parser")
    graph.add_edge("log_parser", "anomaly_reasoner")
    graph.add_edge("anomaly_reasoner", "comparator")
    graph.add_edge("comparator", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph log anomaly detector")
    parser.add_argument(
        "--input", "-i",
        default="seed_Thunderbird.json",
        help="Path to seed_Thunderbird.json (default: ./seed_Thunderbird.json)",
    )
    parser.add_argument(
        "--max-entries", "-n",
        type=int,
        default=None,
        help="Limit to first N entries (useful for quick testing)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to save full JSON results",
    )
    args = parser.parse_args()

    # ── Load seed data ──
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        raw_json: list[dict] = json.load(f)

    if args.max_entries:
        raw_json = raw_json[: args.max_entries]

    print(f"Loaded {len(raw_json)} entries from {input_path}")

    raw_entries = [log_dict.get("input") for log_dict in raw_json]

    # ── Build and run the graph ──
    app = build_graph()

    initial_state: LogState = {
        "raw_entries": raw_entries,
        "parsed_entries": [],
        "ground_truth": [],
        "predictions": [],
        "comparison": {},
    }

    final_state = app.invoke(initial_state)

    # ── Optionally save results ──
    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            # Exclude raw_entries from output to keep it tidy
            output_data = {k: v for k, v in final_state.items() if k != "raw_entries"}
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
