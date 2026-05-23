#!/usr/bin/env python3

def print_tree(problem_name, nodes):
    print(f"Problem: {problem_name}")
    indent = ""
    for i, node in enumerate(nodes):
        if i == 0:
            print(f"└─ {node}")
        else:
            indent += "   "
            print(f"{indent}└─ {node}")
    print()

def main():
    geometry_nodes = [
        "T_11 Construct Geometric Representation",
        "T_1 Goal Identification",
        "T_12 Apply Theorem",
        "T_19 Algebraic Substitution",
        "T_25 Arithmetic Computation",
        "Halt"
    ]
    
    algebra_nodes = [
        "T_4 Decompose",
        "T_22 Equation Formulation",
        "T_19 Substitution",
        "T_25 Arithmetic Computation",
        "Halt"
    ]

    print("=== Simulated Latent Pipeline T-Rule Extraction ===\n")
    print_tree("Geometry Problem (Pythagorean theorem)", geometry_nodes)
    print_tree("Algebra Problem (System of Equations)", algebra_nodes)

if __name__ == "__main__":
    main()
