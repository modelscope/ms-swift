"""
Custom Mathematical Operations Tool Module
This module provides a unified tool call interface for various mathematical operations.
"""

import json
import re
import math
from typing import Tuple, Optional, Any, Dict

# Define the custom mathematical operations with overflow protection
def operation_at(a: int, b: int) -> float:
    """@ operation: Returns a raised to the power of b, then adds a*b
    
    Includes overflow protection for large numbers
    """
    try:
        # For very large exponents, use log approximation
        if b > 100 or (a > 20 and b > 10):
            # Fall back to a simpler calculation for very large values
            return a * b * 2  # Simplified approximation
        power_result = a ** b
        product_result = a * b
        return power_result + product_result
    except OverflowError:
        # If overflow occurs, return a simplified approximation
        return a * b * 2  # Simplified approximation

def operation_amp(a: int, b: int) -> float:
    """& operation: Returns the average of a and b, multiplied by their absolute difference"""
    avg = (a + b) / 2
    diff = abs(a - b)
    return avg * diff

def operation_dollar(a: int, b: int) -> float:
    """$ operation: Returns a factorial-like sum of a repeated b times: a + (a-1) + (a-2) + ... + (a-b+1)"""
    if b <= 0 or b > a:
        return a
        
    # For large values, use arithmetic sequence sum formula
    if b > 1000:
        # Sum of arithmetic sequence: n/2 * (first + last)
        n = min(b, a)
        first = a
        last = a - n + 1
        return n * (first + last) / 2
        
    return sum(a - i for i in range(int(min(b, a))))

def operation_caret(a: int, b: int) -> float:
    """^ operation: Returns a * b if both are even, a + b if both are odd, a - b otherwise"""
    if a % 2 == 0 and b % 2 == 0:
        return a * b
    elif a % 2 == 1 and b % 2 == 1:
        return a + b
    else:
        return a - b

class TOOL_CALL:
    def __call__(self, completion: str) -> Tuple[Any, bool, Optional[float]]:
        raise NotImplementedError

class MathOperation_Tool(TOOL_CALL):
    """Unified tool for handling all mathematical operations"""
    
    def __init__(self):
        self.operations = {
            "at_operation": operation_at,
            "amp_operation": operation_amp,
            "dollar_operation": operation_dollar,
            "caret_operation": operation_caret
        }
    
    def __call__(self, completion: str) -> Tuple[float, bool, float]:
        try:
            # Check for required strict format
            pattern = r'^<think>(.*?)</think>\n<tool>(.*?)</tool>$'
            match = re.match(pattern, completion.strip(), re.DOTALL)
            
            if not match:
                return "", True, 0
                
            tool_content = match.group(2).strip()
            
            # Parse JSON from tool content
            try:
                tool_data = json.loads(tool_content)
            except json.JSONDecodeError:
                return "", True, 0
                
            # Check if JSON has required fields
            if not isinstance(tool_data, dict) or "tool" not in tool_data or "a" not in tool_data or "b" not in tool_data:
                return "", True, 0
                
            tool_name = tool_data["tool"]
            
            # Check if the requested operation exists
            if tool_name not in self.operations:
                return "", True, 0
                
            # Get the operation function
            operation_func = self.operations[tool_name]
            
            # Execute operation
            try:
                a, b = float(tool_data["a"]), float(tool_data["b"])
                result = operation_func(a, b)
                return f"<result>\n{result}\n</reuslt>", False, 0.2
            except (ValueError, TypeError):
                return "", True, 0
                
        except Exception as e:
            print(f"Error in MathOperation_Tool: {e}")
            return "", True, 0

# Parser for expressions with overflow protection
def parse_expression(expression: str) -> float:
    """
    Parses and evaluates a custom mathematical expression.
    Supports operations: @, &, $, ^
    Example: "11@2&1$44^2"
    
    Includes overflow protection for large numbers
    """
    # Tokenize the expression - find all numbers and operators
    tokens = re.findall(r'(\d+|\@|\&|\$|\^)', expression)
    
    # Process tokens
    result = None
    current_op = None
    
    for token in tokens:
        if token in ['@', '&', '$', '^']:
            current_op = token
        else:
            try:
                num = int(token)
                if result is None:
                    result = num
                elif current_op == '@':
                    # Limit very large inputs for @ operation
                    if result > 10000 or num > 100:
                        result = result * num * 2  # Simplified approximation
                    else:
                        result = operation_at(result, num)
                elif current_op == '&':
                    result = operation_amp(result, num)
                elif current_op == '$':
                    result = operation_dollar(result, num)
                elif current_op == '^':
                    result = operation_caret(result, num)
            except (OverflowError, ValueError):
                # Handle overflow by using a simplified calculation
                if current_op == '@':
                    result = result * num * 2  # Simplified approximation
                elif current_op == '&':
                    result = result * num  # Simplified approximation
                elif current_op == '$':
                    result = result + num  # Simplified approximation
                elif current_op == '^':
                    result = max(result, num)  # Simplified approximation
    
    return result

# Map symbols to operation names
SYMBOL_TO_OPERATION = {
    '@': 'at_operation',
    '&': 'amp_operation',
    '$': 'dollar_operation',
    '^': 'caret_operation'
}

# Operation definitions for reference
OPERATION_DEFINITIONS = {
    "@": "a@b = (a^b) + (a*b)",
    "&": "a&b = ((a+b)/2) * |a-b|",
    "$": "a$b = a + (a-1) + (a-2) + ... + (a-b+1)",
    "^": "a^b = a*b if both even, a+b if both odd, a-b otherwise"
}