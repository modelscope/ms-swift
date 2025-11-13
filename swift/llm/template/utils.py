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
    """
    criteria: n. 标准；准则；评判依据
    类功能：
        自定义停止词判断器，用于在文本生成过程中检测特定停止词并终止生成。
        该类继承自 transformers 的 StoppingCriteria 基类，实现了批量并行检测停止词的功能。
        主要用于防止模型生成不可控的长文本，确保在遇到模板定义的结束标记（如 suffix、chat_sep）
        时能够及时停止生成。支持字符串和 token ID 列表两种停止词格式。

    继承关系：
        - 继承自 `transformers.StoppingCriteria` 基类
        - 必须实现 `__call__` 方法，该方法在每次生成新 token 后被 model.generate 调用
        - 返回布尔张量指示每个样本是否应停止生成

    应用场景：
        1. 多轮对话生成：在生成 assistant 回复时，遇到对话分隔符（如 '<|im_end|>'）自动停止
        2. 模板结束标记：检测模板的 suffix 或 chat_sep，防止生成超出模板格式的内容
        3. 自定义停止条件：支持用户定义的停止词列表，灵活控制生成边界
        4. 批量推理：同时处理多个样本，为每个样本独立判断是否停止

    使用示例：
        >>> # 示例1：基础使用 - 单个停止词
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
        >>> model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
        >>> 
        >>> # 定义停止词
        >>> stop_words = ['<|im_end|>', '<|endoftext|>']
        >>> stopping_criteria = StoppingCriteriaList([
        ...     StopWordsCriteria(tokenizer, stop_words)
        ... ])
        >>> 
        >>> # 生成时使用
        >>> input_ids = tokenizer.encode("你好", return_tensors='pt')
        >>> output = model.generate(input_ids, stopping_criteria=stopping_criteria, max_new_tokens=100)
        >>> 
        >>> # 示例2：支持 token ID 列表作为停止词
        >>> stop_words = [
        ...     '<|im_end|>',  # 字符串形式
        ...     [151645, 151643]  # token ID 列表形式
        ... ]
        >>> criteria = StopWordsCriteria(tokenizer, stop_words)
        >>> 
        >>> # 示例3：批量生成场景
        >>> input_ids = tokenizer([
        ...     "用户问题1",
        ...     "用户问题2",
        ...     "用户问题3"
        ... ], return_tensors='pt', padding=True)['input_ids']
        >>> stopping_criteria = StoppingCriteriaList([
        ...     StopWordsCriteria(tokenizer, ['<|im_end|>'])
        ... ])
        >>> outputs = model.generate(input_ids, stopping_criteria=stopping_criteria)
        >>> # 每个样本独立检测停止词，互不影响
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_words: List[Word], **tokenizer_kwargs) -> None:
        """
        功能：
            初始化停止词判断器，设置 tokenizer、停止词列表和状态变量。

        参数：
            tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer 对象，用于解码 token 序列。
            stop_words (List[Word]): 停止词列表，每个元素可以是：
                - str: 字符串形式的停止词（如 '<|im_end|>'）
                - List[int]: token ID 列表形式（如 [151645, 151643]）
            **tokenizer_kwargs: 传递给 tokenizer.batch_decode 的额外参数。
                - 常用参数：skip_special_tokens=False（保留特殊 token）

        返回：
            None

        示例：
            >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
            >>> stop_words = ['<|im_end|>', '<|endoftext|>']
            >>> criteria = StopWordsCriteria(tokenizer, stop_words, skip_special_tokens=False)
        """
        self.tokenizer = tokenizer  # 保存 tokenizer，用于解码 token 序列
        self.stop_words = stop_words  # 停止词列表，支持字符串和 token ID 列表
        self.tokenizer_kwargs = tokenizer_kwargs  # tokenizer.decode 的额外参数
        self.start_idx = -1  # 生成起始位置索引，初始化为-1表示未初始化
        self.is_done = None  # 批量停止状态张量，shape=(batch_size,)，dtype=bool

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        功能：
            在每次生成新 token 后被调用，检测批量样本中是否有停止词出现。
            该方法会解码最近生成的 token 序列（最多 20 个 token），检查是否包含停止词，
            并更新每个样本的停止状态。一旦某个样本匹配到停止词，其 is_done 标志被设为 True。

        参数：
            input_ids (torch.Tensor): 当前生成的完整 token 序列。
                - shape: (batch_size, seq_len)
                - 包含原始输入和已生成的所有 token
                - 示例：batch_size=2, seq_len=50 时，shape=(2, 50)
            scores (torch.Tensor): 当前生成 token 的 logits 分数（本方法未使用）。
                - shape: (batch_size, vocab_size)
            **kwargs: 其他参数（本方法未使用）。

        返回：
            torch.Tensor: 批量停止状态张量。
                - shape: (batch_size,)
                - dtype: torch.bool
                - True: 该样本已匹配停止词，应停止生成
                - False: 该样本未匹配停止词，继续生成
                - 示例：tensor([False, True, False]) 表示第2个样本已停止

        示例：
            >>> # 假设批量大小为2，当前序列长度为50
            >>> input_ids = torch.tensor([
            ...     [1, 2, 3, ..., 151645],  # 第1个样本，最后是 <|im_end|> 的 token ID
            ...     [1, 2, 3, ..., 100]      # 第2个样本，仍在生成中
            ... ])  # shape=(2, 50)
            >>> scores = torch.randn(2, 50000)  # logits
            >>> 
            >>> criteria = StopWordsCriteria(tokenizer, ['<|im_end|>'])
            >>> is_done = criteria(input_ids, scores)
            >>> print(is_done)  # tensor([True, False])（第1个样本已停止）
        """
        # 1> 首次调用初始化：设置生成起始位置和停止状态张量
        if self.start_idx == -1:
            # 记录生成起始位置（原始输入的最后一个 token 索引）
            # 例如：input_ids[0] = [1,2,3,4,5]，则 start_idx=4（从第5个位置开始是新生成的）
            self.start_idx = len(input_ids[0]) - 1
            
            # 初始化批量停止状态张量，全部设为 False（未停止）
            # shape=(batch_size,), dtype=bool, device与input_ids一致
            # 例如：batch_size=3 → is_done=tensor([False, False, False])
            self.is_done = torch.full((input_ids.shape[0], ), False, device=input_ids.device, dtype=torch.bool)

        # 2> 确定解码起始位置：优化性能，仅解码最近20个token
        # 原因：停止词通常出现在序列末尾，不需要解码整个序列
        # max(self.start_idx, seq_len - 20) 确保：
        #   - 至少从生成起始位置开始（self.start_idx）
        #   - 最多只解码最近20个token（seq_len - 20）
        # 例如：self.start_idx=10, seq_len=50 → start_idx=max(10, 30)=30（解码最后20个token）
        start_idx = max(self.start_idx, input_ids.shape[1] - 20)
        
        # 3> 批量解码：将最近的token序列解码为文本
        # input_ids[:, start_idx:] → shape=(batch_size, decode_len)，decode_len <= 20
        # batch_decode返回字符串列表，长度为batch_size
        # 例如：text_list=['Hello world<|im_end|>', 'Continuing...']
        text_list = self.tokenizer.batch_decode(input_ids[:, start_idx:], **self.tokenizer_kwargs)
        
        # 4> 逐样本检测停止词
        for i, text in enumerate(text_list):
            # 跳过已停止的样本（避免重复检测）
            if self.is_done[i]:
                continue
            
            # 初始化当前样本的停止标志
            is_finished = False
            
            # 遍历所有停止词，检查是否匹配
            for stop_word in self.stop_words:
                # 停止词匹配逻辑（支持两种形式）：
                # 1. 字符串形式：检查stop_word是否在解码后的文本中
                #    例如：stop_word='<|im_end|>', text='Hello<|im_end|>' → 匹配
                # 2. token ID列表形式：检查序列末尾的token是否与stop_word完全匹配
                #    例如：stop_word=[151645, 151643], 
                #          input_ids[i]==[..., 151645, 151643] → 匹配
                #    input_ids[i][-len(stop_word):].tolist() 获取序列末尾相应长度的token
                if isinstance(stop_word, str) and stop_word in text or isinstance(
                        stop_word, list) and input_ids[i][-len(stop_word):].tolist() == stop_word:
                    is_finished = True  # 匹配到停止词
                    break  # 跳出停止词循环，无需继续检测
            
            # 更新当前样本的停止状态
            # 例如：第1个样本匹配到停止词 → is_done[0]=True
            self.is_done[i] = is_finished

        # 5> 返回批量停止状态张量
        # model.generate会根据此张量决定每个样本是否继续生成
        # 例如：is_done=tensor([True, False, True]) → 第1、3个样本停止，第2个继续
        return self.is_done


def fetch_one(element: Union[Tuple, List, Set, Dict, Any], item_type: Optional[Type] = None) -> Any:
    """
    fetch 在计算机术语中表示：获取、请求
    功能：
        从嵌套数据结构中递归提取第一个非空元素。使用深度优先搜索遍历容器类型（list/tuple/set/dict），
        跳过空值（None、空字符串等），返回第一个满足条件的元素。支持可选的类型过滤。

    参数：
        element (Union[Tuple, List, Set, Dict, Any]): 待提取的元素或数据结构。
            - 单个元素：直接返回（如 'hello', 123）
            - 容器类型：递归提取（如 [1, 2], {'key': 'value'}）
            - 嵌套结构：深度优先遍历（如 [['', 'a'], 'b'] → 返回 'a'）
        item_type (Optional[Type]): 类型过滤器，默认 None（不限类型）。
            - None: 返回第一个非空元素
            - str/int/list 等：只返回指定类型的元素

    返回：
        Any: 第一个满足条件的非空元素，未找到返回 None。

    示例：
        >>> # 示例1：从嵌套列表提取非空字符串
        >>> fetch_one([['', 'hello'], 'world'])
        'hello'  # 跳过空字符串 ''，返回 'hello'
        
        >>> # 示例2：类型过滤 - 只提取字符串
        >>> fetch_one([[1, 2], 'text', 3.14], item_type=str)
        'text'  # 跳过整数和浮点数，返回字符串
        
        >>> # 示例3：从字典提取值
        >>> fetch_one({'a': '', 'b': None, 'c': 'valid'})
        'valid'  # 跳过空字符串和 None
        
        >>> # 示例4：基本类型直接返回
        >>> fetch_one('hello')
        'hello'
    """
    # 1> 序列类型处理：递归遍历 tuple/set/list
    if isinstance(element, (tuple, set, list)):
        for ele in element:
            # 递归调用处理嵌套结构（深度优先）
            out = fetch_one(ele)
            # 条件1：out 为真值（跳过 None、''、0 等假值）
            # 条件2：无类型限制 或 类型匹配
            # 例如：out='hello', item_type=str → 匹配返回
            if out and (item_type is None or isinstance(out, item_type)):
                return out  # 找到第一个满足条件的元素
        # 遍历完所有元素未找到，隐式返回 None

    # 2> 字典类型处理：转为值列表后递归
    elif isinstance(element, dict):
        # 将字典值转为列表递归处理
        # 例如：{'a': '', 'b': 'hello'} → ['', 'hello'] → 'hello'
        return fetch_one(list(element.values()))

    # 3> 基本类型处理：直接返回（递归终止条件）
    else:
        return element


def findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """
    功能：
        在token列表中查找所有匹配的子序列位置，返回所有匹配位置的起始索引列表。该方法支持两种查找模式：
        (1)查找单个token的所有出现位置；
        (2)查找token子序列的所有出现位置。
        使用滑动窗口方式遍历，首先定位子序列首个token的位置，然后验证从该位置开始的完整子序列是否匹配。
        主要用于在token序列中定位特殊token（如EOS、分隔符）或特定的token模式。
    
    参数：
        token_list (List[int]): 待搜索的token ID列表（通常是编码后的文本序列）
            - 示例：[1, 2, 100, 3, 100, 4]
        
        sub_token_list (Union[int, List[int]]): 要查找的目标token或token子序列
            - 若为int：表示查找单个token的所有出现位置
            - 若为List[int]：表示查找token子序列的所有出现位置
            - 示例：100（单个token）或[2, 3]（子序列）
    
    返回：
        List[int]: 所有匹配位置的起始索引列表
            - 对于单个token：返回该token在列表中所有出现的索引
            - 对于子序列：返回子序列起始位置的所有索引
            - 若无任何匹配：返回空列表[]
    
    示例：
        >>> # 示例1：查找单个token的所有位置
        >>> token_list = [1, 2, 100, 3, 100, 4]
        >>> result = findall(token_list, 100)
        >>> print(result)  # [2, 4]（token 100出现在索引2和索引4）
        
        >>> # 示例2：查找token子序列的所有位置
        >>> token_list = [1, 2, 3, 4, 2, 3, 5]
        >>> result = findall(token_list, [2, 3])
        >>> print(result)  # [1, 4]（子序列[2,3]在索引1和索引4开始出现）
    """
    # 类型统一化：若输入为单个整数，转换为单元素列表
    # 例如：100 -> [100]，便于后续统一处理单token和子序列两种情况
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]

    # 初始化结果列表，用于存储所有匹配位置的起始索引
    res = []
    
    # 初始化搜索起始位置为-1
    # 首次调用token_list.index时从idx+1=0位置开始搜索
    idx = -1
    
    # 使用try-except结构处理查找结束的情况
    # 当list.index找不到更多匹配时会抛出ValueError，通过捕获异常来优雅退出循环
    try:
        # 无限循环，持续查找直到抛出ValueError异常
        while True:
            # 查找sub_token_list的首个token在token_list中下一次出现的位置
            # 参数idx+1确保从上次找到位置的下一个位置开始搜索，避免重复
            # 例如：首次idx=-1，从0开始；若找到位置2，下次从3开始
            idx = token_list.index(sub_token_list[0], idx + 1)
            
            # 验证是否为完整匹配：
            # 条件1：len(sub_token_list)==1 表示只查找单个token
            #        首token匹配即代表完全匹配，直接添加索引
            # 条件2：sub_token_list == token_list[idx:idx+len(sub_token_list)]
            #        验证从idx位置开始的完整子序列是否与目标子序列完全相同
            #        例如：idx=1, sub_token_list=[2,3], token_list[1:3]=[2,3] -> 匹配
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx:idx + len(sub_token_list)]:
                # 完整匹配成功，将起始索引添加到结果列表
                res.append(idx)
            # 若首token匹配但完整子序列不匹配，继续下一轮循环查找下一个首token位置
            # 例如：目标[2,3]，在位置找到2但后续是[2,4]，不匹配，继续搜索
    
    except ValueError:
        # 当token_list.index找不到更多匹配时抛出ValueError
        # 说明已遍历完整个列表，所有匹配都已找到
        pass  # 捕获异常，优雅退出循环
    
    # 返回所有匹配位置的起始索引列表
    # 完整执行示例：
    # token_list=[1,2,100,3,100,4], sub_token_list=[100]
    # 第1次循环：idx=2, res=[2]
    # 第2次循环：idx=4, res=[2,4]
    # 第3次循环：抛出ValueError，退出
    # 返回：[2, 4]
    return res


def align_image_inputs(input_ids: List[int], labels: List[int], new_input_ids,
                       image_token: int) -> Tuple[List[int], List[int]]:
    """
    align: v. 对齐；校准；使一致
    功能：
        将多模态模型生成的新 token 序列（包含图像视觉 token）对齐到原始 input_ids，
        替换占位符（image_token）为实际的视觉 token，并同步更新 labels（用 -100 填充视觉 token 位置）。
        用于多模态模型训练时，将图像占位符替换为经过视觉编码器处理后的实际 token 序列。

    参数：
        input_ids (List[int]): 原始 token 序列，包含图像占位符（image_token）。
        labels (List[int]): 对应的标签序列（可为空列表）。
        new_input_ids (Union[List[int], torch.Tensor]): 新的 token 序列，图像占位符已被视觉 token 替换。
        image_token (int): 图像占位符的 token ID（如 -200）。

    返回：
        Tuple[List[int], List[int]]: 对齐后的 (input_ids, labels)。

    示例：
        >>> # 示例：图像占位符替换为 3 个视觉 token
        >>> input_ids = [1, 2, 3, -200, 4, 5]  # -200 是图像占位符
        >>> labels = [1, 2, 3, -200, 4, 5]
        >>> new_input_ids = [1, 2, 3, 101, 102, 103, 4, 5]  # 占位符被替换为 [101,102,103]
        >>> align_image_inputs(input_ids, labels, new_input_ids, image_token=-200)
        ([1, 2, 3, 101, 102, 103, 4, 5], [1, 2, 3, -100, -100, -100, 4, 5])
        # 视觉 token 位置的 labels 填充为 -100（不计算 loss）
    """
    # 1> 类型转换：确保 new_input_ids 是列表
    if isinstance(new_input_ids, torch.Tensor):
        new_input_ids = new_input_ids.tolist()

    # 2> 双指针遍历：i 遍历 input_ids，j 遍历 new_input_ids
    i, j = 0, 0
    while i < len(input_ids):
        x = input_ids[i]
        
        # 3> 遇到图像占位符：定位并替换为视觉 token 序列
        if x == image_token:
            # 边界检查：确保占位符前后有 token（用于定位）
            assert i + 1 < len(input_ids), f'input_ids[-10:]: {input_ids[-10:]}'
            assert i - 1 >= 0, f'input_ids[:10]: {input_ids[:10]}'
            
            # 关键思路：
            # input_ids:     [1, 2, 3(i-1), image_token(i), 4(i+1), 5, 6]
            # new_input_ids: [1, 2, 3(j_begin), a, a, a, 4(j), 5, 6]
            # 目标：找到 new_input_ids 中 3 和 4 之间的视觉 token [a,a,a]
            
            # 3.1> 定位 j_begin：在 new_input_ids 中找到 input_ids[i-1] 的位置
            j_begin = j - 1
            for k in range(5):  # 容错搜索：±5 范围内查找
                # 向后搜索
                if j_begin + k < len(new_input_ids) and new_input_ids[j_begin + k] == input_ids[i - 1]:
                    j_begin += k
                    break
                # 向前搜索
                if j_begin - k >= 0 and new_input_ids[j_begin - k] == input_ids[i - 1]:
                    j_begin -= k
                    break
            else:
                # 未找到匹配，抛出异常
                raise ValueError(f'new_input_ids: {new_input_ids}, input_ids: {input_ids}')
            
            j_begin += 1  # 移动到视觉 token 序列的起始位置
            
            # 3.2> 定位 j：在 new_input_ids 中找到 input_ids[i+1] 的位置
            while j < len(new_input_ids) and new_input_ids[j] != input_ids[i + 1]:
                j += 1
            
            # 3.3> 替换：将 image_token 替换为 new_input_ids[j_begin:j]
            # 例如：[1,2,3,-200,4,5] → [1,2,3,101,102,103,4,5]
            input_ids = input_ids[:i] + new_input_ids[j_begin:j] + input_ids[i + 1:]
            
            # 3.4> 同步更新 labels：视觉 token 位置填充 -100（不计算 loss）
            # 例如：[1,2,3,-200,4,5] → [1,2,3,-100,-100,-100,4,5]
            if labels:
                labels = labels[:i] + [-100] * (j - j_begin) + labels[i + 1:]
            
            # 3.5> 更新 i：跳过新插入的视觉 token
            i += j - j_begin
        else:
            # 4> 普通 token：同步移动指针
            j += 1
        i += 1
    
    return input_ids, labels


def _split_str_by_regex(text: str, regex_delimiters: List[str]) -> List[str]:
    """
    功能：
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
    
    返回：
        List[str]: 分割后的字符串列表，格式为[delimiter1, content1, delimiter2, content2, ...]
            - 列表长度必定为偶数
            - 偶数索引位置（0, 2, 4, ...）：分隔符（或空字符串''表示无分隔符）
            - 奇数索引位置（1, 3, 5, ...）：分隔符后的内容文本
            - 若文本开头没有分隔符，首元素为空字符串''
            - 若文本开头有分隔符，首元素为该分隔符（后面插入空字符串''作为索引0）
            - 所有元素拼接后能完全还原原始文本（验证通过断言）
            - 示例：['', 'Hello', '<image>', '<image>', '', 'world']
                   （第0个元素''表示开头无分隔符，第1个元素'Hello'是内容，
                    第2个元素'<image>'是分隔符，第3个元素'<image>'是内容...）
    """
    # 1> 组合多个正则表达式模式为联合模式
    # 将每个模式用括号包裹形成捕获组：(pattern1)|(pattern2)|...
    # 捕获组确保re.split()返回结果中包含匹配到的分隔符本身
    # 使用'|'连接表示"或"关系，匹配任意一个分隔符即可
    # 例如：['\\<image\\>', '\\<video\\>'] -> '(\\<image\\>)|(\\<video\\>)'
    combined_pattern = '|'.join(f'({pattern})' for pattern in regex_delimiters)
    
    # 2> 使用正则表达式分割文本
    # re.split(pattern, text, flags=re.DOTALL):
    #   - pattern: 组合后的正则表达式模式
    #   - text: 待分割的文本
    #   - flags=re.DOTALL: 使'.'匹配包括换行符在内的所有字符
    # 由于使用了捕获组，返回结果会包含分隔符本身
    # 例如：'Hello<image>world' 分割后 -> ['Hello', '<image>', None, ..., 'world']
    #       （None来自未匹配的捕获组，因为联合模式中只有一个匹配）
    parts = re.split(combined_pattern, text, flags=re.DOTALL)
    
    # 3> 过滤None值
    # re.split()使用多个捕获组时，未匹配的组会产生None
    # 列表推导式过滤掉所有None值，只保留有效的字符串
    parts = [part for part in parts if part is not None]

    # 4> 规范化列表格式：统一返回 [delimiter1, content1, delimiter2, content2, ...] 的偶数长度列表
    # 目标格式：偶数索引(0,2,4...)=delimiter，奇数索引(1,3,5...)=content
    #
    # re.split 使用捕获组分割的行为：
    # - 文本开头是分隔符 → parts=['', delimiter1, content1, delimiter2, content2, ...]
    #   例如：'<img>Hello<vid>world' → ['', '<img>', 'Hello', '<vid>', 'world']
    # - 文本开头非分隔符 → parts=[content1, delimiter1, content2, delimiter2, ...]
    #   例如：'Hello<img>world<vid>end' → ['Hello', '<img>', 'world', '<vid>', 'end']
    
    if parts[0] == '':
        # 情况A：文本开头是分隔符
        # 当前：['', '<img>', 'Hello', '<vid>', 'world']
        #       [0]   [1]      [2]     [3]      [4]
        # 索引布局：偶数索引[0,2,4]是content/空串，奇数索引[1,3]是delimiter
        # 问题：索引对应反了！我们需要偶数索引=delimiter，奇数索引=content
        # 操作：pop(0)移除首个''
        # 结果：['<img>', 'Hello', '<vid>', 'world']
        #       [0]      [1]      [2]      [3]
        # 此时：偶数索引[0,2]是delimiter，奇数索引[1,3]是content ✓
        parts.pop(0)
    else:
        # 情况B：文本开头非分隔符
        # 当前：['Hello', '<img>', 'world', '<vid>', 'end']
        #       [0]      [1]      [2]      [3]      [4]
        # 索引布局：偶数索引[0,2,4]是content，奇数索引[1,3]是delimiter
        # 问题：缺少"开头无分隔符"的标记（应该用''占位）
        # 操作：insert(0, '')在开头插入''
        # 结果：['', 'Hello', '<img>', 'world', '<vid>', 'end']
        #       [0]   [1]      [2]      [3]      [4]      [5]
        # 此时：偶数索引[0,2,4]是delimiter（[0]为''表示无前置分隔符），奇数索引[1,3,5]是content ✓
        parts.insert(0, '')
    
    # 5> 验证返回列表长度为偶数
    # 确保列表格式为[delimiter1, content1, delimiter2, content2, ...]
    # 长度为偶数是后续配对处理的前提条件
    assert len(parts) % 2 == 0, f'result: {parts}'
    
    # 6> 验证拼接后能还原原始文本
    # 将parts中所有元素拼接，应该完全等于原始text
    # 这是正确性验证，确保分割过程没有丢失或改变任何字符
    assert ''.join(parts) == text, f'split_result: {parts}, text: {text}'
    
    return parts


def split_str_parts_by(text: str, delimiters: List[str], regex_mode: bool = False) -> List[Dict[str, str]]:
    """
    功能：
        根据分隔符列表将文本字符串分割成多个部分，并将每个部分与其对应的分隔符配对返回。
        该方法支持两种模式：普通字符串匹配模式(regex_mode=False) 和正则表达式匹配模式(regex_mode=True)。
        分割后返回字典列表，每个字典包含'key'（匹配到的分隔符）和'content'（分隔符后的文本内容或分隔符本身）两个键。
        该方法常用于解析包含特殊标签的文本，例如将'Hello<image>world'拆分为多个带标签标记的片段，便于后续对特殊标签进行识别和替换操作。
    
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
    
    返回：
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
    # 1> 验证输入参数类型
    assert isinstance(text, str), f'text: {text}'
    
    # 2> 保存原始分隔符列表
    # 在普通模式下，分隔符会被转义，需要保存原始版本用于后续返回
    delimiters_origin = delimiters
    
    # 3> 处理分隔符（普通模式需要转义）
    if not regex_mode:  # 普通字符串匹配模式
        # re.escape(): 转义正则表达式中的特殊字符
        # 例如：'<image>' -> '\\<image\\>'，确保'<'和'>'被当作普通字符而非正则元字符
        # 这样可以避免分隔符中的特殊字符（如'<', '>', '.', '*'等）被误解析为正则语法
        delimiters = [re.escape(delimiter) for delimiter in delimiters]
    
    # 4> 执行文本分割
    # 若delimiters非空，调用_split_str_by_regex进行正则分割
    # 若delimiters为空，返回默认格式['', text]（无分隔符的情况）
    # _split_str_by_regex返回格式：[key1, content1, key2, content2, ...]
    # 确保返回列表长度为偶数，交替包含分隔符和内容
    parts = _split_str_by_regex(text, delimiters) if delimiters else ['', text]
    
    # 5> 初始化结果列表
    res = []

    # 6> 根据模式构建返回结果
    if regex_mode:  # 正则表达式匹配模式
        # 6.1> 过滤空字符串
        # 在正则模式下，需要过滤掉空字符串，只保留有内容的部分
        parts = [part for part in parts if part]
        
        # 6.2> 遍历每个部分，匹配分隔符
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
        # 6.3> 按照固定格式解析parts
        # parts格式：[delimiter1, content1, delimiter2, content2, ...]（长度为偶数）
        # parts[::2]: 提取所有偶数位置的元素（索引0, 2, 4, ...），即所有的delimiter（分隔符）
        # parts[1::2]: 提取所有奇数位置的元素（索引1, 3, 5, ...），即所有的content（内容）
        # zip(): 将delimiter和content一一配对
        for key, content in zip(parts[::2], parts[1::2]):
            # 构建字典并添加到结果列表
            # key: 匹配到的分隔符（已转义后匹配，这里使用parts中已分割出的原始分隔符）
            # content: 分隔符后的文本内容
            res.append({'key': key, 'content': content})
    return res
