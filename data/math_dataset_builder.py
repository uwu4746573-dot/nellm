"""
Math Dataset Builder

This script defines a pipeline to load and preprocess mathematics datasets
like NuminaMath or Math-Shepherd. It extracts T-rules from textual
Chain-of-Thought solutions.
"""

from typing import List, Dict, Any

class MathDatasetBuilder:
    """
    A builder class for loading, preprocessing, and converting mathematics 
    datasets into formats suitable for training with T-rules.
    """
    def __init__(self, dataset_name: str = "NuminaMath"):
        self.dataset_name = dataset_name
        self.data: List[Dict[str, Any]] = []

    def load_dataset(self) -> None:
        """
        Simulates loading the dataset.
        """
        print(f"Loading dataset: {self.dataset_name}...")
        # Dummy data simulating a loaded dataset
        self.data = [
            {
                "problem": "Solve for x: 2x + 4 = 10",
                "solution": "First, subtract 4 from both sides: 2x = 6. Then, divide by 2: x = 3."
            },
            {
                "problem": "Find the area of a circle with radius 3.",
                "solution": "Use the formula A = pi * r^2. Substitute r = 3: A = pi * 3^2. A = 9pi."
            }
        ]
        print(f"Loaded {len(self.data)} examples.")

    def extract_t_rules(self, solution_text: str) -> List[str]:
        """
        Converts textual Chain-of-Thought solutions into sequences of T-rules.
        
        Args:
            solution_text: The textual solution to parse.
            
        Returns:
            A list of T-rule strings extracted from the text.
        """
        # Dummy implementation for extracting T-rules
        rules = []
        lower_text = solution_text.lower()
        
        if "subtract" in lower_text or "add" in lower_text:
            rules.append("Algebraic Manipulation (Addition/Subtraction)")
        if "divide" in lower_text or "multiply" in lower_text:
            rules.append("Algebraic Manipulation (Multiplication/Division)")
        if "formula" in lower_text:
            rules.append("Retrieve Mathematical Formula")
        if "substitute" in lower_text:
            rules.append("Algebraic Substitution")
        if "area" in lower_text or "radius" in lower_text:
            rules.append("Construct Geometric Representation")
            
        # Default rule if none match
        if not rules:
            rules.append("Logical Deduction")
            
        return rules

    def preprocess(self) -> List[Dict[str, Any]]:
        """
        Preprocesses the loaded data, applying the T-rule extraction.
        
        Returns:
            A list of preprocessed data items containing problems, solutions, 
            and extracted T-rules.
        """
        print("Preprocessing dataset and extracting T-rules...")
        processed_data = []
        for item in self.data:
            t_rules = self.extract_t_rules(item["solution"])
            processed_item = {
                "problem": item["problem"],
                "original_solution": item["solution"],
                "extracted_t_rules": t_rules
            }
            processed_data.append(processed_item)
            
        print("Preprocessing complete.")
        return processed_data

    def build(self) -> List[Dict[str, Any]]:
        """
        Executes the full pipeline: load -> preprocess.
        """
        self.load_dataset()
        return self.preprocess()

if __name__ == "__main__":
    builder = MathDatasetBuilder(dataset_name="Math-Shepherd")
    dataset = builder.build()
    
    for i, example in enumerate(dataset):
        print(f"\nExample {i+1}:")
        print(f"Problem: {example['problem']}")
        print(f"T-Rules: {example['extracted_t_rules']}")
