def extract_xml_answer(text: str) -> str:
    # Extracts the answer portion from the XML formatted text.
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward function to compare correctness of the generated answer.
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", 
          f"\nExtracted:\n{extracted_responses[0]}")
    # Award a reward if the extracted response matches the expected answer.
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# Reward function for integer responses.
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # Provide a reward if the response is a digit.
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# Reward function checking for strict XML format.
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion strictly adheres to the XML format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    # Reward if the response exactly matches the pattern.
    return [0.5 if match else 0.0 for match in matches]

# Reward function checking for a soft XML format (less strict).
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion follows the expected XML format (soft check)."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Helper function to count XML tags and provide a reward score based on counts.
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

# Reward function using XML tag counts to evaluate formatting.
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
