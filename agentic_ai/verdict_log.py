from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5-nano")

def _create_prompt_for_verdict(file_output: str, llm_output: str) -> str:
    return (f"""
        You are Node 3 in an agentic AI pipeline. Your role is to act as a strict validator. You must evaluate an AI-generated prediction against a known ground truth. You do not have access to the raw log data; you must base your evaluation entirely on the provided inputs.

        INPUTS:
        1. Node 2 Prediction:
           - Predicted Classification (Normal/Abnormal)
           - Predicted Reasoning
        2. Ground Truth:
           - Correct Classification (Normal/Abnormal)
           - Correct Reasoning

        YOUR OBJECTIVE:
        Evaluate whether Node 2's prediction is correct in both its final classification and its underlying logic.

        EVALUATION RULES:
        1. Classification Check: Does the Predicted Classification match the Correct Classification? 
        2. Reasoning Check: Does the Predicted Reasoning capture the same core logical points as the Correct Reasoning? (It does not need to match word-for-word, but the justification must be fundamentally the same).
        3. If both the classification AND the reasoning match the ground truth, the evaluation is a success.
        4. If the classification is wrong, OR if the classification is correct but the predicted reasoning is flawed, hallucinates, or misses the core point of the ground truth, the evaluation is a failure.

        OUTPUT FORMAT:
        Provide your final evaluation using exactly the following text structure:

        VERDICT: [Output either MATCH or MISMATCH]
        ANALYSIS: [If the verdict is MATCH, simply output: "The classification is correct and the reasoning aligns with the ground truth. OK." If the verdict is MISMATCH, clearly explain exactly what went wrong. Detail where Node 2's prediction diverged from the Ground Truth, whether it failed the classification, or if its reasoning missed the actual root cause.]

        INPUT Node 2 Prediction: {llm_output}
        INPUT Ground Truth: {file_output}
    """).strip()


def verdict_logs(dataset_output: str, agentic_output: str) -> str:
    prompt: str = _create_prompt_for_verdict(dataset_output, agentic_output)
    return model.invoke(prompt).content