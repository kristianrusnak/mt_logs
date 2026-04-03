import random
import json
import sys

SEED_NUM = 2003

def extract_classification(classification: str, dataset: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for data in dataset:
        output: str = data.get("output")
        sep = output.index('-')
        output_class = output[:sep]
        if classification == output_class:
            result.append(data)
    return result

def calculate_split(num_of_datasets: int) -> tuple[int, int]:
    two_thirds: float = num_of_datasets * (2 / 3)
    one_third: float = num_of_datasets * (1 / 3)

    if num_of_datasets % 3 == 0:
        return int(two_thirds), int(one_third)

    if two_thirds/int(two_thirds) > one_third/int(one_third):
        one_third += 1
    elif two_thirds/int(two_thirds) < one_third/int(one_third):
        two_thirds += 1

    return int(two_thirds), int(one_third)

def main() -> None:
    args = sys.argv[1:]

    with (open(args[0], "r", encoding="utf-8") as f):
        dataset = json.load(f)

    abnormal = extract_classification("abnormal", dataset)
    normal = extract_classification("normal", dataset)

    random.shuffle(abnormal)
    random.shuffle(normal)

    num_of_abnormal, _ = calculate_split(len(abnormal))
    num_of_normal, _ = calculate_split(len(normal))

    finetune = abnormal[:num_of_abnormal] + normal[:num_of_normal]
    eval_set = abnormal[num_of_abnormal:] + normal[num_of_normal:]

    with open(f"{args[1]}/finetune.json", "w", encoding="utf-8") as f:
        json.dump(finetune, f, indent=2, ensure_ascii=False)

    with open(f"{args[1]}/eval_set.json", "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()