import json
import os
from datasets import load_dataset
from typing import List

def extract_t_rules(solution_text: str) -> List[str]:
    """
    Scans the solution text for keywords and maps them to T-rules.
    """
    rules_map = {
        "T_11": ["equation", "solve", "algebra", "polynomial"],
        "T_12": ["differentiate", "integral", "derivative", "calculus", "limit"],
        "T_13": ["angle", "triangle", "circle", "radius", "area", "geometry", "perimeter", "volume", "length"],
        "T_14": ["probability", "statistics", "chance", "mean", "median", "variance", "distribution"],
        "T_15": ["theorem", "proof", "lemma", "assume", "therefore", "hence", "logical"],
        "T_16": ["matrix", "vector", "determinant", "eigen", "linear algebra"],
        "T_17": ["sequence", "series", "summation", "arithmetic progression", "geometric progression"],
        "T_18": ["graph", "plot", "coordinates", "axis", "axes"],
        "T_19": ["substitute", "replace", "plug in", "let x", "let y"],
        "T_20": ["factor", "expand", "simplify", "distribute", "cancel"],
        "T_21": ["add", "subtract", "multiply", "divide", "sum", "difference", "product", "quotient"],
        "T_22": ["fraction", "ratio", "proportion", "percent", "percentage"]
    }
    
    extracted_rules = set()
    lower_text = solution_text.lower()
    
    for rule, keywords in rules_map.items():
        if any(keyword in lower_text for keyword in keywords):
            extracted_rules.add(rule)
            
    # Default rule if none found
    if not extracted_rules:
        extracted_rules.add("T_00")
        
    return sorted(list(extracted_rules))

def main():
    print("Loading AI-MO/NuminaMath-CoT dataset...")
    # Load only train split and take the first 10,000 samples
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    
    num_samples = min(10000, len(dataset))
    dataset = dataset.select(range(num_samples))
    
    print(f"Processing {num_samples} samples...")
    processed_data = []
    for item in dataset:
        problem = item.get("problem", "")
        solution = item.get("solution", "")
        
        t_rules = extract_t_rules(solution)
        processed_data.append({
            "problem": problem,
            "t_rules": t_rules
        })
        
    output_path = "/Users/uchebnick/projects/secretcase/repo/nellm/data/processed_math.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving processed data to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
