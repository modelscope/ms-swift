# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Tuple
from types import MethodType

from peft import PeftModel as HFPeftModel

from swift import SwiftModel, PeftModel
from swift.utils import (get_logger)

logger = get_logger()


def split_agent_parts_by(text: str, delimiters: List[str]):
    """

    Args:
        text: A text to be split.
        delimiters: The delimiters.

    Returns:
        The split text.
    """
    all_start_chars = [d[0] for d in delimiters]
    all_length = [len(d) for d in delimiters]

    text_list = []
    last_words = ''

    while len(text) > 0:
        for char_idx, char in enumerate(text):
            match_index = [idx for idx, start_char in enumerate(all_start_chars) if start_char == char]
            is_delimiter = False
            for index in match_index:
                if text[char_idx: char_idx+all_length[index]] == delimiters[index]:
                    if last_words:
                        if text_list:
                            text_list[-1]['content'] = last_words
                        else:
                            text_list.append({'key': '', 'content': last_words})
                    last_words = ''
                    text_list.append({'key': delimiters[index]})
                    text = text[char_idx+all_length[index]:]
                    is_delimiter = True
                    break
            if not is_delimiter:
                last_words += char
            else:
                break
        if last_words == text:
            text = ''

    text_list[-1]['content'] = last_words
    return text_list


def calculate_loss_scale(response) -> Tuple[List[str], List[float]]:
    if 'Action:' in response and 'Thought:' in response:
        agent_keyword = ['Action:', 'Action Input:', 'Thought:', 'Final Answer:', 'Observation:']
        agent_parts = split_agent_parts_by(response, agent_keyword)
        assert all([c['key'] for c in agent_parts])
        weights = []
        agent_content = []
        for c in agent_parts:
            if c['key'] in ('Action:', 'Action Input:'):
                weights += [2.0]
                weights += [2.0]
            elif c['key'] in ('Thought:', 'Final Answer:', ''):
                weights += [1.0]
                weights += [1.0]
            elif c['key'] in ('Observation:',):
                weights += [2.0]
                weights += [0.0]
            agent_content.append(c['key'])
            agent_content.append(c['content'])
        return agent_content, weights
    else:
        return [response], [1.0]


def prepare_loss_scale(model):
    if isinstance(model, (SwiftModel, PeftModel, HFPeftModel)):
        model = model.base_model

    if model.__class__.__name__ == 'ChatGLMForConditionalGeneration':
        from .models import ChatGLM3Forward
        model.forward = MethodType(ChatGLM3Forward, model)
        model.support_loss_scale = True
    elif model.__class__.__name__ == 'InternLM2ForCausalLM':
        from .models import InternLMForward
        model.forward = MethodType(InternLMForward, model)
        model.support_loss_scale = True
    else:
        model.support_loss_scale = False
        logger.warn(f'Model {model.__class__.__name__} not supported for weight scaling')




