from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from enum import StrEnum

import json
import sys

class Style(StrEnum):
    TECHNICAL = "A brief, technical 1-2 sentence explanation using system terminology. No hedging, no elaboration."
    STEP_BY_STEP = "A numbered list of 3-5 reasoning steps that walk through the evidence in the log sequence and conclude with the label."
    SYSADMIN = "Written as a sysadmin explaining to a colleague what they see in this log and why it is or is not a concern. Conversational tone, first person."
    ROOT_CAUSE = "Focus specifically on identifying which event or pattern in the sequence is the decisive signal that determines the label. Start with that signal, then explain why it matters."

SYSTEM_PROMPT = """
You are an expert in distributed systems and log analysis. You will be given a log sequence from a real system (HDFS, BGL, or Thunderbird) and its ground-truth classification along with an existing explanation.

Your task is to rewrite the explanation in a different style while:
1. Preserving the exact classification label (normal or abnormal) — never change it
2. Preserving all factual claims — do not invent events that are not in the log sequence
3. Preserving logical correctness — the reasoning must actually justify the label
4. Using a clearly different writing style from the original

Output ONLY a STRING of your rewritten explanation.
Do not output anything else.
"""

def create_user_prompt(log_seq: str, label: str, cause: str, style: str):
    return f"""
    Here is the original entry:

    LOG SEQUENCE:
    {log_seq}
    
    GROUND TRUTH LABEL: {label}
    
    ORIGINAL EXPLANATION:
    {cause}
    
    Rewrite the explanation in the following style: {style}
    
    Remember: do not change the label, do not add events that are not in the log sequence, and make sure your explanation logically supports the label.
    """

def build_prompt_from_record(record: dict, style: Style) -> "HumanMessage":
    log_seq = record.get("input")
    record_output = record.get("output")
    label, cause = extract_output(record_output)

    prompt: str = create_user_prompt(
        log_seq=log_seq,
        label=label,
        cause=cause,
        style=style.value
    )
    return HumanMessage(prompt)

def extract_output(output: str) -> tuple[str, str]:
    sep = output.index("-")
    return output[:sep], output[sep+1:]

def append_to_output_file(record: dict, answer: str, output_file: str) -> None:
    record_input = record.get("input")

    record_output = record.get("output")
    label, _ = extract_output(record_output)

    new_record = {
        "input": record_input,
        "output": f"{label}-{answer}",
        "augmented": True
    }

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.append(new_record)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main() -> None:
    args = sys.argv[1:]

    if len(args) == 2:
        input_file = args[0]
        output_file = args[1]
    else:
        raise IndexError("Wrong Input")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f)

    #TODO
    #model = ChatOpenAI(model="gpt-5.4")
    system_message = SystemMessage(SYSTEM_PROMPT)

    for record in data:
        for style in Style:
            user_prompt = build_prompt_from_record(
                record=record,
                style=style
            )
            #TODO
            #answer = model.invoke([system_message, user_prompt])
            answer = "refactored explanation"
            append_to_output_file(
                record=record,
                answer=answer,
                output_file=output_file
            )

if __name__ == "__main__":
    main()