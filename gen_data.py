"""
Dataset Generator for Custom Mathematical Operations
This module generates a dataset of custom mathematical expressions and their results.
"""

import random
import json
import re
from typing import Dict, List
from math_tool import parse_expression, SYMBOL_TO_OPERATION, OPERATION_DEFINITIONS

def generate_safe_expression():
    """Generate an expression that won't cause overflow errors"""
    # Start with a moderate number
    expression = str(random.randint(1, 20))
    
    # Add 3-5 operations with safe numbers
    num_ops = random.randint(1, 6)
    
    for i in range(num_ops):
        # Choose operation
        op = random.choice(['@', '&', '$', '^'])
        
        # For @ operation, use smaller numbers to avoid overflow
        if op == '@':
            # For exponentiation, keep the exponent small
            num = random.randint(1, 3)
        else:
            num = random.randint(1, 10)
            
        expression += op + str(num)
    
    return expression

def generate_dataset(num_samples: int = 1000, output_file: str = "math_operations_dataset.jsonl") -> None:
    """
    Generates a dataset of custom mathematical expressions and their results.
    Saves the dataset as a JSONL file.
    
    Args:
        num_samples: Number of samples to generate
        output_file: Path to save the JSONL file
    """
    with open(output_file, 'w') as f:
        for _ in range(num_samples):
            # Generate a safe expression
            expression = generate_safe_expression()
            
            # Calculate the result
            try:
                result = parse_expression(expression)
                
                # Create the data entry
                # data_entry = {
                #     "query": f"Calculate the result of the expression: {expression}",
                #     "answer": result
                # }
                data_entry = {"messages": [{"role": "user", "content": f"Calculate the result of the expression: {expression}"}],"response":result}
                
                # Write to JSONL file
                f.write(json.dumps(data_entry) + '\n')
            except Exception as e:
                print(f"Skipping problematic expression {expression}: {e}")
                continue
    
    print(f"Generated dataset with {num_samples} samples and saved to {output_file}")

generate_dataset()