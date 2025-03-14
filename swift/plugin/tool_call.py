from typing import Union, Tuple, Optional

class TOOL_CALL:

    def __call__(self, completion:str) -> Tuple[str, bool, Optional[int]]:
        raise NotImplementedError


"""
Search module for RL training loop.
This module provides functions to search through vectorized documents and retrieve question-answer pairs.
"""

import json
import re
from typing import Tuple, Optional
import traceback

# Load the vectorstore when module is imported
try:
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("Warning: FAISS vectorstore could not be loaded.")
except Exception as e:
    print(f"Error loading vectorstore: {e}")
    vectorstore = None

def search(query: str, results: int = 5):
    """
    Search for relevant chunks using similarity search.
    
    Args:
        query: The search query
        return_type: Return as string or list (default: str)
        results: Number of results to return (default: 5)
        
    Returns:
        Results as string or list depending on return_type
    """
    if vectorstore is None:
        raise ValueError("Vectorstore not loaded. Please ensure FAISS index exists.")
        
    search_results = vectorstore.similarity_search(query, k=results)
    
    result_dict = {}
    for idx, result in enumerate(search_results, start=1):
        result_dict[idx] = result.page_content
    
    result_json = json.dumps(result_dict,indent=2,ensure_ascii=False)
    return f"<result>\n{result_json}\n</result>"

class TOOL_CALL:
    def __call__(self, completion: str) -> Tuple[str, bool, Optional[float]]:
        raise NotImplementedError

class Search_Tool(TOOL_CALL):
    def __call__(self, completion: str) -> Tuple[str, bool, Optional[float]]:
        """
        Checks if the completion strictly follows the format <think>xxx</think><tool_call>xxx</tool_call>
        and if the tool_call contains valid JSON with "tool" and "arg" fields.
        
        Args:
            completion: The text completion to check
            
        Returns:
            Tuple containing:
            - search result or empty string
            - boolean indicating if there was an error
            - score (0.2 if successful, 0 if error)
        """
        try:
            # Check for required strict format using regex
            pattern = r'^<think>(.*?)</think><tool_call>(.*?)</tool_call>$'
            match = re.match(pattern, completion.strip(), re.DOTALL)
            
            if not match:
                return "", True, 0
                
            tool_content = match.group(2).strip()
            
            # Parse JSON from tool_call content
            try:
                tool_data = json.loads(tool_content)
            except json.JSONDecodeError:
                return "", True, 0
                
            # Check if JSON has required fields
            if not isinstance(tool_data, dict) or "tool" not in tool_data or "arg" not in tool_data:
                return "", True, 0
                
            # Check if the tool is "search"
            if tool_data["tool"] != "search":
                return "", True, 0
                
            # Execute search with the provided argument
            search_result = search(tool_data["arg"])
            return search_result, False, 0.2
            
        except Exception as e:
            print(f"Error in Search_Tool: {e}")
            traceback.print_exc()
            return "", True, 0