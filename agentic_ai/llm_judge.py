from langchain_openai import ChatOpenAI
import json
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)

def _create_prompt_for_verdict(dataset_output: str, llm_reasoning_data: str) -> str:
    return (f"""
        You are an expert log analysis evaluator. Your task is to judge the semantic similarity between two reasoning outputs about log sequence analysis.
        
        ## Input
        
        **Reference Reasoning (Ground Truth):**
        {dataset_output}
        
        **LLM Reasoning (To Evaluate):**
        {llm_reasoning_data}
        
        ## Task
        
        Both reasonings follow the same structure:
        1. A classification: `normal` or `abnormal`
        2. A separator `-`
        3. A detailed reasoning explanation about the log sequence
        
        Evaluate the LLM reasoning against the reference on the following dimensions:
        
        ### 1. Classification Match
        - Do both agree on `normal` vs `abnormal`?
        - If not, is the LLM too strict (flags normal as abnormal) or not strict enough (misses an abnormal case)?
        
        ### 2. Semantic Similarity Score
        Rate the overall semantic similarity on a scale of 0–10, where:
        - 10 = identical reasoning and conclusions
        - 7–9 = minor omissions or phrasing differences, core logic intact
        - 4–6 = partially correct, missing key points or adding irrelevant ones
        - 1–3 = significant divergence, major points missed or hallucinated
        - 0 = completely wrong or unrelated reasoning
        
        ### 3. Detailed Divergence Analysis
        Identify specific differences between the two reasonings:
        - **Missing points**: What does the reference mention that the LLM reasoning omits?
        - **Hallucinations**: What does the LLM reasoning mention that is not in the reference or cannot be inferred from context?
        - **Strictness issues**: Is the LLM too strict or too lenient in its interpretation?
        - **Logical errors**: Are there any incorrect inferences or reasoning gaps?
        
        ### 4. Root Cause Assessment
        Based on the divergences found, what is the likely cause of the LLM's mistakes? Choose one or more:
        - Model is not strict enough (misses anomalies)
        - Model is overly strict (flags benign patterns)
        - Model hallucinated details not present in the log
        - Model missed a key event or pattern
        - Model misunderstood the sequence order
        - Model applied incorrect domain knowledge
        - Other (specify)
        
        ## Output Format
        
        Respond in the following structure:
        
        **Classification Match:** [Yes / No — LLM said X, reference says Y]
        
        **Similarity Score:** [0–10] — [one-line justification]
        
        **Divergence Analysis:**
        - Missing points: ...
        - Hallucinations: ...
        - Strictness issues: ...
        - Logical errors: ...
        
        **Root Cause Assessment:** ...
        
        **Summary:** [2–3 sentence overall verdict on the quality of the LLM reasoning]
    """).strip()


def judge_log(dataset_output: str, agentic_output: str) -> str:
    prompt: str = _create_prompt_for_verdict(dataset_output, agentic_output)
    return model.invoke(prompt).content

def main():
    INPUT_PATH = "agentic_ai_results.json"

    with open(INPUT_PATH, encoding="utf-8") as file:
        data = json.load(file)

    results = []
    for i, item in enumerate(data):
        print(f"judging {i}. dataset")
        judge_result = judge_log(item.get("output"), item.get("reasoning"))
        print(f"judging finished for {i}. dataset")
        results.append({
            "dataset_output": item.get("output"),
            "llm_reasoning": item.get("reasoning"),
            "llm_judge": judge_result
        })

    with open("llm_judge.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()