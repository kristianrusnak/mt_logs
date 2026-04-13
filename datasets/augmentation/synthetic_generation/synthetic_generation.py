import json
import sys

from random import Random
from typing import Literal
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from torch.utils.hipify.hipify_python import InputError

RANDOM_SEED = 8361
LIMIT_FEW_SHOT_PROMPT = 4

class SyntheticLog(BaseModel):
    log_seq: str = Field(description="comma-separated list of log event templates, each ending with a comma")
    label: Literal["normal", "abnormal"] = Field(description="normal or abnormal")
    cause: str = Field(description="detailed explanation of why this sequence is normal or abnormal, referencing specific events in the sequence")

def system_prompt():
    return """
        You are an expert in IBM Blue Gene/L (BGL) supercomputer systems and log analysis. 
        You have deep knowledge of BGL hardware components, failure modes, and normal operating 
        patterns as reflected in system logs.
        
        You will generate synthetic BGL log sequences that are realistic, internally consistent, 
        and correctly classified as either normal or abnormal. Each sequence must be accompanied 
        by a label and a detailed reasoning explanation.
        
        BGL log sequences have the following characteristics:
        - They consist of 15-25 repeated or mixed log event templates
        - Dynamic values are replaced with <*> as placeholders
        - Events come from a fixed vocabulary of BGL-specific message types
        - Normal sequences show consistent, expected hardware monitoring patterns
        - Abnormal sequences show failure signals: panics, persistent errors, unrecoverable 
          interrupts, or combinations of correctable errors that escalate
        
        Output ONLY a JSON object in this exact format, nothing else:
        {
          "log_seq": "<comma-separated list of log event templates, each ending with a comma>",
          "label": "<normal or abnormal>",
          "cause": "<detailed explanation of why this sequence is normal or abnormal, referencing 
                     specific events in the sequence>"
        }
    """

def generate_prompt(data: list[dict], label: Literal["normal", "abnormal"]):
    prompt = f"""
        Generate a NEW synthetic BGL log sequence that is labeled {label.upper()}.
        
        Here are real examples of abnormal BGL sequences to learn the style and vocabulary from:
    """

    for i, record in enumerate(data):
        record_input = record.get("input")
        record_output = record.get("output")
        label, cause = extract_output(record_output)
        prompt += f"""\n\n
            --- EXAMPLE {i} ---
            log_seq: "{record_input}",
            label: "{label}",
            cause: "{cause}"
        """

    prompt += """\n\n
        --- GENERATION INSTRUCTIONS ---
        Generate a sequence that:
        1. Uses BGL event vocabulary (panics, TLB errors, instruction cache parity errors, 
           Lustre mount failures, unrecoverable interrupts, signal floods, power deactivation, 
           kernel terminations, etc.)
        2. Is meaningfully DIFFERENT from the examples above — use a different failure mode 
           or combination as the primary signal
        3. Contains 15-25 events
        4. Is genuinely abnormal — the fault signal must be unambiguous, not just "some errors 
           were corrected" (correctable errors alone are normal in BGL)
        5. The cause explanation must identify the specific event(s) that make it abnormal 
           and explain why those events indicate a real failure
        6. Do NOT generate a sequence where all errors are corrected and the system recovers — 
           that would be normal
    """

def extract_output(output: str) -> tuple[str, str]:
    sep = output.index("-")
    return output[:sep], output[sep + 1:]

def append_to_output_file(
        leg_seq: str,
        label: str,
        cause: str,
        output_file: str
) -> None:
    new_record = {
        "input": leg_seq,
        "output": f"{label}-{cause}",
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
        label = args[0]
        input_file = output_file = args[1]
    elif len(args) == 3:
        label = args[0]
        input_file = args[1]
        output_file = args[2]
    else:
        raise InputError("Wrong Input")

    if label in ("abnormal", "normal"):
        raise InputError("Invalid classification label")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = ChatOpenAI(model="gpt-5.4")
    model_with_structure = model.with_structured_output(SyntheticLog)

    #TODO choose system prompt dynamically
    system_message = SystemMessage(system_prompt())

    rnd = Random(RANDOM_SEED)
    rnd.shuffle(data)

    user_prompt = HumanMessage(generate_prompt(
        data=data[:LIMIT_FEW_SHOT_PROMPT],
        label=label
    ))

    answer = model_with_structure.invoke([system_message, user_prompt])
    append_to_output_file(
        leg_seq=answer.log_seq,
        label=answer.label,
        cause=answer.cause,
        output_file=output_file
    )