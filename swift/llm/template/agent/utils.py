from typing import Dict, List, Optional, Tuple


def split_str_parts_by(text: str, delimiters: List[str]):
    """Split the text field into parts.

    Args:
        text: A text to be split.
        delimiters: The delimiters.

    Returns:
        The split text in list of dicts.
    """
    assert isinstance(text, str), f'text: {text}'
    all_start_chars = [d[0] for d in delimiters]
    all_length = [len(d) for d in delimiters]

    text_list = []
    last_words = ''

    while len(text) > 0:
        for char_idx, char in enumerate(text):
            match_index = [idx for idx, start_char in enumerate(all_start_chars) if start_char == char]
            is_delimiter = False
            for index in match_index:
                if text[char_idx:char_idx + all_length[index]] == delimiters[index]:
                    if text_list:
                        text_list[-1]['content'] = last_words
                    elif last_words:
                        text_list.append({'key': '', 'content': last_words})
                    last_words = ''
                    text_list.append({'key': delimiters[index]})
                    text = text[char_idx + all_length[index]:]
                    is_delimiter = True
                    break
            if not is_delimiter:
                last_words += char
            else:
                break
        if last_words == text:
            text = ''

    if len(text_list):
        text_list[-1]['content'] = last_words
    else:
        text_list.append({'key': '', 'content': last_words})
    return text_list


def split_parts_by_regex(text_list: list, regex_delimiters: Dict[str, List[float]]) -> None:
    import re
    compiled_patterns = [(re.compile(pattern), scale) for pattern, scale in regex_delimiters.items()]
    for i in range(len(text_list) - 1, -1, -1):
        item = text_list[i]
        if item.get('key') == '':
            res_text = item['content']
            last_idx = 0
            segments = []

            for pattern, scale in compiled_patterns:
                matches = list(re.finditer(pattern, res_text))
                for match in matches:
                    if match.start() > last_idx:
                        segments.append({'key': '', 'content': res_text[last_idx:match.start()]})
                    segments.append({'key': scale[0], 'content': match.group(0)})
                    last_idx = match.end()

            if last_idx < len(res_text):
                segments.insert(0, {'key': '', 'content': res_text[last_idx:]})

            if segments:
                text_list[i:i + 1] = segments


def calculate_loss_scale(query: str,
                         response: str,
                         response_loss_scale_map: Dict[str, list],
                         query_loss_scale_map: Optional[Dict[str, list]] = None) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.

    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf

    Agent response format:

    ```text
        Thought: you should always think about what to do
        Action: the action to take, should be one of the above tools[fire_recognition,
            fire_alert, call_police, call_fireman]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ```
    Returns:
        A tuple of agent response parts and their weights.
    """
    # query loss scale map
    if query_loss_scale_map is not None:
        for key in query_loss_scale_map.keys():
            if key in query:
                if isinstance(query_loss_scale_map[key], (float, int)):
                    query_loss_scale_map[key] = [query_loss_scale_map[key]]
                loss_scale_value = query_loss_scale_map[key][0]
                return [response], [float(loss_scale_value)]
    delimiters = list(k for k in response_loss_scale_map.keys() if len(response_loss_scale_map[k]) == 2)
    agent_parts = split_str_parts_by(response, delimiters)
    regex_delimiters = {k: v for k, v in response_loss_scale_map.items() if len(v) == 1}
    if len(regex_delimiters):
        split_parts_by_regex(agent_parts, regex_delimiters)
    weights = []
    agent_content = []
    for c in agent_parts:
        if isinstance(c['key'], (float, int)):
            weights += [c['key']]
            agent_content.append(c['content'])
        else:
            if c['key'] in response_loss_scale_map:
                weights += [response_loss_scale_map[c['key']][0]]
                weights += [response_loss_scale_map[c['key']][1]]
                agent_content.append(c['key'])
                agent_content.append(c['content'])
            else:
                weights += [1.0]
                agent_content.append(c['content'])
    return agent_content, weights


def split_action_action_input(response: str) -> Tuple[Optional[str], Optional[str]]:
    agent_keyword = [
        'action:', 'Action:', 'ACTION:', 'action input:', 'Action Input:', 'Action input:', 'ACTION INPUT:', 'Thought:',
        'Final Answer:', 'Observation:'
    ]
    agent_parts = split_str_parts_by(response, agent_keyword)
    action = None
    action_input = None
    for c in agent_parts:
        if c['key'].lower() == 'action:':
            action = c['content']
        elif c['key'].lower() == 'action input:':
            action_input = c['content']
    if action:
        action = action.strip().replace('\n', '')
    if action_input:
        action_input.strip().replace('\n', '')
    return action, action_input
