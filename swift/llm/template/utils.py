# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria

Prompt = List[Union[str, List[int], List[str]]]
Word = Union[str, List[int]]
Context = Word


class ContextType:
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    OTHER = 'other'


class StopWordsCriteria(StoppingCriteria):
    """Adding extra stop words in template to prevent unstoppable generation
        Like suffixes and chat seps in the template.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_words: List[Word], **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.tokenizer_kwargs = tokenizer_kwargs
        self.start_idx = -1
        self.is_done = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.start_idx == -1:
            self.start_idx = len(input_ids[0]) - 1
            self.is_done = torch.full((input_ids.shape[0], ), False, device=input_ids.device, dtype=torch.bool)
        # [-20:]: Assuming the end tokens do not exceed 20 tokens,
        #   to avoid input_ids being too long and affecting efficiency.
        start_idx = max(self.start_idx, input_ids.shape[1] - 20)
        text_list = self.tokenizer.batch_decode(input_ids[:, start_idx:], **self.tokenizer_kwargs)
        for i, text in enumerate(text_list):
            if self.is_done[i]:
                continue
            is_finished = False
            for stop_word in self.stop_words:
                if isinstance(stop_word, str) and stop_word in text or isinstance(
                        stop_word, list) and input_ids[i][-len(stop_word):].tolist() == stop_word:
                    is_finished = True
                    break
            self.is_done[i] = is_finished
        return self.is_done


def fetch_one(element: Union[Tuple, List, Set, Dict, Any], item_type: Optional[Type] = None) -> Any:
    """
    函数功能：
        从嵌套的数据结构中递归提取第一个非空的元素。该方法支持处理多层嵌套的数据结构（列表、
        元组、集合、字典等），通过深度优先搜索的方式递归遍历，找到第一个满足条件的非空元素并
        返回。如果指定了item_type参数，则只返回符合该类型的元素；否则返回第一个非空的任意类型
        元素。该方法常用于从复杂的嵌套结构中快速提取目标元素，例如判断Context类型（可能是字符串
        或token列表）的实际元素类型。
    
    参数：
        element (Union[Tuple, List, Set, Dict, Any]): 待提取的元素或数据结构
            - 可以是单个元素（如str、int等基本类型）
            - 可以是容器类型（tuple、list、set、dict）
            - 可以是多层嵌套的复杂结构（如[[1, 2], [3, 4]]）
            - 示例：'hello'、[1, 2, 3]、{'key': 'value'}、[['a', 'b'], 'c']
        
        item_type (Optional[Type]): 可选的类型过滤参数，默认为None
            - 若为None，返回第一个非空元素（不限类型）
            - 若指定类型（如str、int），只返回符合该类型的第一个非空元素
            - 用于类型过滤，例如item_type=str表示只提取字符串类型的元素
            - 示例：str、int、float等Python类型
    
    返回值：
        Any: 提取到的第一个满足条件的元素
            - 若element是基本类型（非容器），直接返回element本身
            - 若element是容器类型，返回递归找到的第一个非空且符合类型要求的元素
            - 若容器中所有元素都为空或不符合类型要求，返回None（隐式）
            - 返回类型取决于实际提取到的元素类型
            - 示例：从['', 'hello', 'world']中提取到'hello'（跳过空字符串）
    """
    # ===== 情况1：element是序列类型（tuple、set、list） =====
    if isinstance(element, (tuple, set, list)):
        # 遍历序列中的每个元素
        for ele in element:
            # 递归调用fetch_one，处理嵌套结构
            # 这里递归深度优先搜索，找到第一个非空元素
            out = fetch_one(ele)
            
            # 检查提取结果是否满足条件：
            # 条件1：out为真值（非None、非空字符串、非0等）
            # 条件2：item_type为None（不限类型） 或 out的类型符合item_type
            if out and (item_type is None or isinstance(out, item_type)):
                # 找到满足条件的元素，立即返回
                return out
        # 如果遍历完所有元素都没有找到，函数隐式返回None
    
    # ===== 情况2：element是字典类型 =====
    elif isinstance(element, dict):
        # 将字典的所有值转换为列表，然后递归调用fetch_one
        # 相当于从字典的值中提取第一个非空元素
        # 例如：{'a': '', 'b': 'hello'} -> ['', 'hello'] -> 'hello'
        return fetch_one(list(element.values()))
    
    # ===== 情况3：element是基本类型（非容器） =====
    else:
        # 直接返回element本身
        # 这是递归的终止条件
        return element


def findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """Find the index of a token in the token_list."""
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(sub_token_list[0], idx + 1)
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx:idx + len(sub_token_list)]:
                res.append(idx)
    except ValueError:
        pass
    return res


def align_image_inputs(input_ids: List[int], labels: List[int], new_input_ids,
                       image_token: int) -> Tuple[List[int], List[int]]:
    if isinstance(new_input_ids, torch.Tensor):
        new_input_ids = new_input_ids.tolist()

    # Find the tokens after the image_token in input_ids, and then align them.
    i, j = 0, 0
    while i < len(input_ids):
        x = input_ids[i]
        if x == image_token:
            assert i + 1 < len(input_ids), f'input_ids[-10:]: {input_ids[-10:]}'
            assert i - 1 >= 0, f'input_ids[:10]: {input_ids[:10]}'
            # [1, 2, 3(i-1), image_token(i), 4(i+1) ,5, 6]
            # [1, 2, 3(j_begin), a(j'), a, a, a, 4(j) ,5, 6]
            j_begin = j - 1
            for k in range(5):  # Increase robustness.
                if j_begin + k < len(new_input_ids) and new_input_ids[j_begin + k] == input_ids[i - 1]:
                    j_begin += k
                    break
                if j_begin - k >= 0 and new_input_ids[j_begin - k] == input_ids[i - 1]:
                    j_begin -= k
                    break
            else:
                raise ValueError(f'new_input_ids: {new_input_ids}, input_ids: {input_ids}')
            j_begin += 1
            while j < len(new_input_ids) and new_input_ids[j] != input_ids[i + 1]:
                j += 1
            input_ids = input_ids[:i] + new_input_ids[j_begin:j] + input_ids[i + 1:]
            if labels:
                labels = labels[:i] + [-100] * (j - j_begin) + labels[i + 1:]
            i += j - j_begin
        else:
            j += 1
        i += 1
    return input_ids, labels


def _split_str_by_regex(text: str, regex_delimiters: List[str]) -> List[str]:
    """
    函数功能：
        使用正则表达式分隔符列表分割文本字符串，返回交替包含分隔符和内容的列表。该方法将多个正则
        表达式模式组合成一个联合模式，使用re.split()进行分割，并确保返回列表的格式为：
        [delimiter1, content1, delimiter2, content2, ...]，即奇数位置为分隔符，偶数位置为内容。
        
        该方法会对分割结果进行规范化处理：过滤None值、确保首元素为空字符串（或分隔符）、验证
        列表长度为偶数、验证拼接后能还原原始文本。这种固定格式便于上层调用者进行配对处理，
        无需额外判断元素类型。
    
    参数：
        text (str): 待分割的文本字符串
            - 可以包含零个或多个与正则分隔符匹配的子串
            - 示例：'Hello<image>world<video>end'
        
        regex_delimiters (List[str]): 正则表达式分隔符列表
            - 每个元素是一个正则表达式模式字符串
            - 多个模式会被组合成'(pattern1)|(pattern2)|...'的联合模式
            - 使用捕获组确保分隔符本身也会出现在分割结果中
            - 示例：['\\<image\\>', '\\<video\\>', '\\<audio\\>']（已转义的模式）
    
    返回值：
        List[str]: 分割后的字符串列表，格式为[delimiter1, content1, delimiter2, content2, ...]
            - 列表长度必定为偶数
            - 奇数索引位置（0, 2, 4, ...）：分隔符（或空字符串''表示无分隔符）
            - 偶数索引位置（1, 3, 5, ...）：分隔符后的内容文本
            - 若文本开头没有分隔符，首元素为空字符串''
            - 若文本开头有分隔符，首元素为该分隔符（后面插入空字符串''作为索引0）
            - 所有元素拼接后能完全还原原始文本（验证通过断言）
            - 示例：['', 'Hello', '<image>', '<image>', '', 'world']
                   （第0个元素''表示开头无分隔符，第1个元素'Hello'是内容，
                    第2个元素'<image>'是分隔符，第3个元素'<image>'是内容...）
    """
    # ===== 步骤1：组合多个正则表达式模式为联合模式 =====
    # 将每个模式用括号包裹形成捕获组：(pattern1)|(pattern2)|...
    # 捕获组确保re.split()返回结果中包含匹配到的分隔符本身
    # 使用'|'连接表示"或"关系，匹配任意一个分隔符即可
    # 例如：['\\<image\\>', '\\<video\\>'] -> '(\\<image\\>)|(\\<video\\>)'
    combined_pattern = '|'.join(f'({pattern})' for pattern in regex_delimiters)
    
    # ===== 步骤2：使用正则表达式分割文本 =====
    # re.split(pattern, text, flags=re.DOTALL):
    #   - pattern: 组合后的正则表达式模式
    #   - text: 待分割的文本
    #   - flags=re.DOTALL: 使'.'匹配包括换行符在内的所有字符
    # 由于使用了捕获组，返回结果会包含分隔符本身
    # 例如：'Hello<image>world' 分割后 -> ['Hello', '<image>', None, ..., 'world']
    #       （None来自未匹配的捕获组，因为联合模式中只有一个匹配）
    parts = re.split(combined_pattern, text, flags=re.DOTALL)
    
    # ===== 步骤3：过滤None值 =====
    # re.split()使用多个捕获组时，未匹配的组会产生None
    # 列表推导式过滤掉所有None值，只保留有效的字符串
    parts = [part for part in parts if part is not None]
    
    # ===== 步骤4：规范化列表格式，确保首元素为分隔符（或空字符串） =====
    if parts[0] == '':  # 文本开头没有分隔符（如'Hello<image>...'）
        # 移除首个空字符串，因为后续会以[delimiter, content, ...]格式返回
        # 此时parts变为：['<image>', 'world', ...]
        parts.pop(0)
    else:  # 文本开头就是分隔符（如'<image>Hello...'）
        # 在开头插入空字符串，保持[delimiter, content, ...]格式的一致性
        # 例如：['<image>', 'Hello', ...] -> ['', '<image>', 'Hello', ...]
        parts.insert(0, '')
    
    # ===== 步骤5：验证返回列表长度为偶数 =====
    # 确保列表格式为[delimiter1, content1, delimiter2, content2, ...]
    # 长度为偶数是后续配对处理的前提条件
    assert len(parts) % 2 == 0, f'result: {parts}'
    
    # ===== 步骤6：验证拼接后能还原原始文本 =====
    # 将parts中所有元素拼接，应该完全等于原始text
    # 这是正确性验证，确保分割过程没有丢失或改变任何字符
    assert ''.join(parts) == text, f'split_result: {parts}, text: {text}'
    
    # ===== 步骤7：返回规范化后的分割结果 =====
    return parts


def split_str_parts_by(text: str, delimiters: List[str], regex_mode: bool = False) -> List[Dict[str, str]]:
    """
    函数功能：
        根据分隔符列表将文本字符串分割成多个部分，并将每个部分与其对应的分隔符配对返回。该方法
        支持两种模式：普通字符串匹配模式和正则表达式匹配模式。分割后返回字典列表，每个字典包含
        'key'（匹配到的分隔符）和'content'（分隔符后的文本内容或分隔符本身）两个键。该方法常用于
        解析包含特殊标签的文本，例如将'Hello<image>world'拆分为多个带标签标记的片段，便于后续
        对特殊标签进行识别和替换操作。
    
    参数：
        text (str): 待分割的文本字符串
            - 必须是字符串类型，否则触发断言错误
            - 可以包含零个或多个分隔符
            - 示例：'Hello<image>world<video>end'
        
        delimiters (List[str]): 分隔符列表
            - 用于分割文本的分隔符字符串列表
            - 支持多个分隔符，按列表顺序匹配
            - 在普通模式下，分隔符会被转义以避免正则特殊字符的影响
            - 在正则模式下，分隔符会被当作正则表达式模式处理
            - 示例：['<image>', '<video>', '<audio>']
        
        regex_mode (bool): 正则表达式模式开关，默认为False
            - False: 普通字符串匹配模式，分隔符会被re.escape()转义
            - True: 正则表达式匹配模式，分隔符作为正则模式使用
            - 正则模式适用于需要模糊匹配的场景（如匹配任意数字、字母等）
    
    返回值：
        List[Dict[str, str]]: 分割结果的字典列表
            每个字典包含两个键值对：
            - 'key' (str): 匹配到的分隔符
              * 在普通模式下：匹配到的分隔符字符串（如'<image>'）
              * 在正则模式下：匹配到的原始分隔符字符串
              * 若没有匹配到分隔符，则为空字符串''
            - 'content' (str): 文本内容
              * 在普通模式下：分隔符之后的文本内容
              * 在正则模式下：匹配到的分隔符本身或普通文本
            示例返回值：[{'key': '', 'content': 'Hello'}, 
                        {'key': '<image>', 'content': '<image>'}, 
                        {'key': '', 'content': 'world'}]
    """
    # ===== 步骤1：验证输入参数类型 =====
    assert isinstance(text, str), f'text: {text}'
    
    # ===== 步骤2：保存原始分隔符列表 =====
    # 在普通模式下，分隔符会被转义，需要保存原始版本用于后续返回
    delimiters_origin = delimiters
    
    # ===== 步骤3：处理分隔符（普通模式需要转义） =====
    if not regex_mode:  # 普通字符串匹配模式
        # re.escape(): 转义正则表达式中的特殊字符
        # 例如：'<image>' -> '\\<image\\>'，确保'<'和'>'被当作普通字符而非正则元字符
        # 这样可以避免分隔符中的特殊字符（如'<', '>', '.', '*'等）被误解析为正则语法
        delimiters = [re.escape(delimiter) for delimiter in delimiters]
    
    # ===== 步骤4：执行文本分割 =====
    # 若delimiters非空，调用_split_str_by_regex进行正则分割
    # 若delimiters为空，返回默认格式['', text]（无分隔符的情况）
    # _split_str_by_regex返回格式：[key1, content1, key2, content2, ...]
    # 确保返回列表长度为偶数，交替包含分隔符和内容
    parts = _split_str_by_regex(text, delimiters) if delimiters else ['', text]
    
    # ===== 步骤5：初始化结果列表 =====
    res = []

    # ===== 步骤6：根据模式构建返回结果 =====
    if regex_mode:  # 正则表达式匹配模式
        # ===== 步骤6.1：过滤空字符串 =====
        # 在正则模式下，需要过滤掉空字符串，只保留有内容的部分
        parts = [part for part in parts if part]
        
        # ===== 步骤6.2：遍历每个部分，匹配分隔符 =====
        for part in parts:
            # 遍历所有分隔符，尝试匹配当前part
            for delimiter, delimiter_origin in zip(delimiters, delimiters_origin):
                # re.match(): 从字符串开头匹配正则表达式
                # re.DOTALL: 使'.'匹配包括换行符在内的所有字符
                if re.match(delimiter, part, re.DOTALL):
                    # 匹配成功，找到对应的原始分隔符
                    break
            else:
                # for循环正常结束（没有break），说明没有匹配到任何分隔符
                # 将delimiter_origin设为空字符串，表示这是普通文本内容
                delimiter_origin = ''
            
            # 添加到结果列表：key为匹配到的分隔符（或空字符串），content为part本身
            res.append({'key': delimiter_origin, 'content': part})
    
    else:  # 普通字符串匹配模式
        # ===== 步骤6.3：按照固定格式解析parts =====
        # parts格式：[key1, content1, key2, content2, ...]（长度为偶数）
        # parts[::2]: 提取所有偶数位置的元素（索引0, 2, 4, ...），即所有的key（分隔符）
        # parts[1::2]: 提取所有奇数位置的元素（索引1, 3, 5, ...），即所有的content（内容）
        # zip(): 将key和content一一配对
        for key, content in zip(parts[::2], parts[1::2]):
            # 构建字典并添加到结果列表
            # key: 匹配到的分隔符（已转义后匹配，这里使用parts中已分割出的原始分隔符）
            # content: 分隔符后的文本内容
            res.append({'key': key, 'content': content})
    return res
