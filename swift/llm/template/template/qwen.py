# Copyright (c) Alibaba, Inc. and its affiliates.
"""
模块功能：
    该模块实现了 Qwen 系列模型（包括 Qwen、Qwen2、Qwen2.5、QwQ、Qwen3 等）的模板类。
    提供了文本模型（LLM）和多模态模型（MLLM）的统一输入预处理、编码、解码等功能。
    支持视觉-语言、音频-语言、视频-语言等多模态任务，以及推理链（Thinking）、重排序（Reranker）等特殊任务。

主要类：
    - QwenTemplateMeta: Qwen 系列模板元数据基类
    - Qwen2_5TemplateMeta: Qwen2.5 模板元数据
    - Qwen2_5MathTemplateMeta: Qwen2.5 数学模型模板元数据
    - QwenVLTemplate: Qwen-VL 视觉语言模型模板
    - QwenAudioTemplate: Qwen-Audio 音频模型模板
    - Qwen2AudioTemplate: Qwen2-Audio 音频模型模板
    - Qwen2VLTemplate: Qwen2-VL 视觉语言模型模板（支持图像和视频）
    - Qwen2_5VLTemplate: Qwen2.5-VL 视觉语言模型模板
    - Qwen2_5OmniTemplate: Qwen2.5-Omni 全模态模型模板（支持图像、视频、音频）
    - Ovis1_6Template: Ovis 1.6 视觉语言模型模板
    - Ovis2Template: Ovis2 视觉语言模型模板（支持图像和视频）
    - Qwen3RerankerTemplate: Qwen3 重排序模型模板
    - QwenPRMTemplate: Qwen PRM（Process Reward Model）模板

应用场景：
    1. 对话生成：使用 Qwen/Qwen2/Qwen2.5/QwQ/Qwen3 等模型进行文本对话
    2. 视觉问答：使用 Qwen-VL/Qwen2-VL/Qwen2.5-VL 处理图像+文本输入
    3. 音频理解：使用 Qwen-Audio/Qwen2-Audio 处理音频+文本输入
    4. 视频理解：使用 Qwen2-VL/Qwen2.5-VL 处理视频+文本输入
    5. 全模态理解：使用 Qwen2.5-Omni 处理图像+视频+音频+文本输入
    6. 数学推理：使用 Qwen2.5-Math 进行数学问题求解
    7. 文档重排序：使用 Qwen3-Reranker 进行文档相关性排序
    8. 过程奖励建模：使用 Qwen-PRM 评估推理步骤质量
"""
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
import transformers
from packaging import version

from swift.llm import get_packed_seq_params, to_device, to_float_dtype
from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context, Word, findall
from ..vision_utils import load_audio, load_batch, load_video_ovis2
from .llama import Llama3TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta, ThinkingTemplate


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    """
    类功能：
        Qwen 系列模型的模板元数据类。定义了 Qwen 模型的默认系统提示词、停止词、BOS 处理等配置。
        作为 Qwen 系列模板的基础配置类，被其他变体（如 Qwen2.5、QwQ 等）继承使用。

    继承关系：
        继承自 ChatmlTemplateMeta，采用 ChatML 格式的对话模板（如 <|im_start|>、<|im_end|>）。

    应用场景：
        用于配置 Qwen/Qwen2/Qwen-VL/Qwen2-VL 等模型的输入格式和行为。

    属性：
        default_system (Optional[str]): 默认系统提示词，定义模型的角色和行为准则。
        auto_add_bos (bool): 是否自动添加 BOS（Beginning of Sequence）token，默认 False。
        stop_words (List[Word]): 停止词列表，当生成这些词时停止生成，默认为 ['<|endoftext|>']。
        agent_template (str): Agent 模式使用的模板类型，默认为 'hermes'（Hermes Agent 格式）。

    示例：
        >>> # 创建 Qwen 模板元数据
        >>> meta = QwenTemplateMeta(LLMTemplateType.qwen)
        >>> meta.default_system
        'You are a helpful assistant.'
        >>> meta.stop_words
        ['<|endoftext|>']
    """
    # 默认系统提示词：定义模型的基本身份和角色
    default_system: Optional[str] = DEFAULT_SYSTEM
    
    # 是否自动添加 BOS token：Qwen 不需要自动添加，由 tokenizer 处理
    auto_add_bos: bool = False
    
    # 停止词列表：当生成 <|endoftext|> 时停止生成
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    
    # Agent 模板类型：使用 Hermes 格式的 Agent 对话模板
    agent_template: str = 'hermes'


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    """
    类功能：
        Qwen2.5 模型的模板元数据类。继承自 QwenTemplateMeta，仅覆盖默认系统提示词。

    继承关系：
        继承自 QwenTemplateMeta，保留其他配置不变。

    应用场景：
        专用于 Qwen2.5 系列模型，提供更明确的模型身份说明。

    示例：
        >>> meta = Qwen2_5TemplateMeta(LLMTemplateType.qwen2_5)
        >>> meta.default_system
        'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
    """
    # 默认系统提示词：明确说明模型是由阿里云创建的 Qwen
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


@dataclass
class Qwen2_5MathTemplateMeta(QwenTemplateMeta):
    """
    类功能：
        Qwen2.5-Math 数学模型的模板元数据类。专为数学推理任务定制系统提示词。

    继承关系：
        继承自 QwenTemplateMeta，仅覆盖默认系统提示词。

    应用场景：
        专用于 Qwen2.5-Math 模型，指导模型进行逐步推理并用 LaTeX 格式输出答案。

    示例：
        >>> meta = Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math)
        >>> meta.default_system
        'Please reason step by step, and put your final answer within \\\\boxed{}.'
    """
    # 默认系统提示词：要求模型逐步推理，并将最终答案放在 \boxed{} 中（LaTeX 格式）
    default_system: Optional[str] = 'Please reason step by step, and put your final answer within \\boxed{}.'


# QwQ Preview 模型的系统提示词：强调逐步思考能力
qwq_preview_system = ('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                      'You should think step-by-step.')

# 注册 Qwen 基础模板：用于 Qwen/Qwen2 等基础对话模型
register_template(QwenTemplateMeta(LLMTemplateType.qwen))

# 注册 Qwen2.5 模板：用于 Qwen2.5 系列模型
register_template(Qwen2_5TemplateMeta(LLMTemplateType.qwen2_5))

# 注册 QwQ Preview 模板：用于 QwQ 预览版模型，强调逐步思考
register_template(QwenTemplateMeta(LLMTemplateType.qwq_preview, default_system=qwq_preview_system))

# 注册 QwQ 模板：使用 ThinkingTemplate，在响应前添加 <think> 标签
# response_prefix='<think>\n' 表示模型输出会以 <think> 开头，用于显式的思考过程
register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwq, default_system=None, response_prefix='<think>\n', template_cls=ThinkingTemplate))

# 注册 Qwen3 模板：使用 ThinkingTemplate，自动识别 '<think>\n\n</think>\n\n' 格式
# 不指定 response_prefix，模型会自动在输出中插入思考标签
register_template(QwenTemplateMeta(LLMTemplateType.qwen3, default_system=None, template_cls=ThinkingTemplate))

# 注册 Qwen3 Thinking 模板：明确指定思考前缀为 '<think>\n'
# 与 qwen3 的区别在于显式指定 response_prefix
register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_thinking, default_system=None, response_prefix='<think>\n',
        template_cls=ThinkingTemplate))


class Qwen3RerankerTemplate(Template):
    """
    类功能：
        Qwen3 重排序（Reranker）模型的模板类。用于文档检索任务中对候选文档进行相关性排序。
        通过特定的输入格式（Instruct、Query、Document）引导模型判断文档是否与查询相关。

    继承关系：
        继承自 Template 基类。

    应用场景：
        信息检索、问答系统中的文档重排序，判断给定文档是否能回答用户查询。

    使用示例：
        >>> # 输入查询和文档，模型判断文档是否相关
        >>> inputs = {
        ...     'messages': [
        ...         {'role': 'user', 'content': 'What is Python?'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... }
        >>> # 处理后的格式类似：
        >>> # <Instruct>: Given a web search query, retrieve relevant passages that answer the query
        >>> # <Query>: What is Python?
        >>> # <Document>: {doc}
    """
    # 指令模板：定义重排序任务的指令
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        """
        功能：
            预处理输入，将用户查询转换为重排序任务的标准格式。
            格式化为：<Instruct>: ... <Query>: ... <Document>: {doc}

        参数：
            inputs (StdTemplateInputs): 标准模板输入对象，包含 messages 列表。
                - messages[-2]: 倒数第二条消息（用户查询）
                - messages[-1]: 最后一条消息（助手响应）

        返回：
            None: 直接修改 inputs.messages

        示例：
            >>> inputs = StdTemplateInputs(messages=[
            ...     {'role': 'user', 'content': 'machine learning'},
            ...     {'role': 'assistant', 'content': ''}
            ... ])
            >>> template._preprocess_inputs(inputs)
            >>> inputs.messages[-2]['content']
            '<Instruct>: Given a web search query, retrieve relevant passages that answer the query\\n<Query>: machine learning\\n<Document>: {doc}'
        """
        # 1> 调用父类预处理方法，进行基础处理
        super()._preprocess_inputs(inputs)
        
        # 2> 提取用户查询：从倒数第二条消息中获取查询内容
        # 例如：'machine learning'
        query = inputs.messages[-2]['content']
        
        # 3> 构建重排序格式的用户消息：
        # 格式：<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}
        # {doc} 是占位符，在实际使用时会被替换为候选文档内容
        user_message = '<Instruct>: ' + self.instruction + '\n' + '<Query>: ' + query + '\n' + '<Document>: {doc}'
        
        # 4> 更新用户消息：将原始查询替换为格式化的消息
        inputs.messages[-2]['content'] = user_message


# Qwen3 Reranker 的系统提示词：指导模型进行二元判断（yes/no）
qwen3_reranker_system = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be \"yes\" or \"no\".')

# 注册 Qwen3 Reranker 模板：用于文档重排序任务
# response_prefix='<think>\n\n</think>\n\n' 表示模型会先思考再给出判断结果
register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_reranker,
        default_system=qwen3_reranker_system,
        response_prefix='<think>\n\n</think>\n\n',
        template_cls=Qwen3RerankerTemplate))

# 注册 Qwen2.5-Math 模板：用于数学推理任务
register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math))


class QwenPRMTemplate(Template):
    """
    类功能：
        Qwen PRM（Process Reward Model，过程奖励模型）的模板类。
        用于评估推理链（Chain-of-Thought）中每个推理步骤的质量，为强化学习提供细粒度的奖励信号。
        通过在推理步骤之间插入特殊标记（<extra_0>），模型可以为每个步骤打分。

    继承关系：
        继承自 Template 基类。

    应用场景：
        强化学习训练中的奖励建模、推理过程质量评估、步骤级别的反馈生成。

    使用示例：
        >>> # 输入带有推理步骤的文本
        >>> inputs = StdTemplateInputs(messages=[
        ...     {'role': 'user', 'content': 'Solve: 2+3*4'},
        ...     {'role': 'assistant', 'content': 'Step1: 3*4=12<extra_0>Step2: 2+12=14<extra_0>'}
        ... ])
        >>> # 模型会在每个 <extra_0> 位置输出该步骤的质量分数
    """
    # CoT 过程分隔符：用于标记推理步骤的边界
    cot_process_placeholder = '<extra_0>'

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        """
        功能：
            预处理输入，确保最后一条消息以 CoT 过程分隔符结尾。
            如果输入中没有分隔符，会自动在最后添加一个。

        参数：
            inputs (StdTemplateInputs): 标准模板输入对象。
                - messages: 消息列表，每条消息包含 'content' 字段

        返回：
            None: 直接修改 inputs.messages

        示例：
            >>> inputs = StdTemplateInputs(messages=[
            ...     {'role': 'user', 'content': 'Calculate 5+3'},
            ...     {'role': 'assistant', 'content': 'The answer is 8'}
            ... ])
            >>> template._preprocess_inputs(inputs)
            >>> inputs.messages[-1]['content']
            'The answer is 8<extra_0>'
        """
        # 1> 调用父类预处理方法
        super()._preprocess_inputs(inputs)
        
        # 2> 拼接所有消息内容：用 '\n' 连接所有消息的内容
        # 例如：'User: Calculate 5+3\nAssistant: The answer is 8'
        total_content = '\n'.join([message['content'] or '' for message in inputs.messages])
        
        # 3> 检查是否包含分隔符：如果没有 <extra_0>，则在最后添加
        # 这确保至少有一个步骤边界用于评分
        if self.cot_process_placeholder not in total_content:
            inputs.messages[-1]['content'] = inputs.messages[-1]['content'] + self.cot_process_placeholder

    @staticmethod
    def make_step_rewards(logits, token_masks):
        """
        功能：
            从模型 logits 中提取每个推理步骤的奖励分数。
            通过识别步骤分隔符位置，计算对应位置的"正向"概率作为奖励。

        参数：
            logits (torch.Tensor): 模型输出的 logits。
                - shape: (batch_size, seq_len, num_labels)
                - 通常 num_labels=2（负向、正向）
            token_masks (torch.Tensor): 步骤分隔符的掩码。
                - shape: (batch_size, seq_len)
                - 值为 1 的位置表示是步骤分隔符

        返回：
            List[List[float]]: 每个样本的步骤奖励列表。
                - 外层 list 长度 = batch_size
                - 内层 list 长度 = 该样本的步骤数

        示例：
            >>> # 假设有 2 个样本，第一个有 3 步，第二个有 2 步
            >>> logits = torch.randn(2, 20, 2)  # batch=2, seq=20, labels=2
            >>> token_masks = torch.zeros(2, 20)
            >>> token_masks[0, [5, 10, 15]] = 1  # 第 1 个样本在 5,10,15 位置有分隔符
            >>> token_masks[1, [8, 16]] = 1      # 第 2 个样本在 8,16 位置有分隔符
            >>> rewards = QwenPRMTemplate.make_step_rewards(logits, token_masks)
            >>> len(rewards)
            2
            >>> len(rewards[0]), len(rewards[1])
            (3, 2)  # 第 1 个样本 3 步，第 2 个样本 2 步
        """
        # 1> 计算概率分布：对 logits 进行 softmax，得到每个 token 的类别概率
        # shape: (batch_size, seq_len, num_labels)
        # 例如：probabilities[i, j, 0] 表示第 i 个样本第 j 个 token 的负向概率
        #      probabilities[i, j, 1] 表示第 i 个样本第 j 个 token 的正向概率
        probabilities = F.softmax(logits, dim=-1)
        
        # 2> 应用掩码：只保留步骤分隔符位置的概率
        # token_masks.unsqueeze(-1): (batch_size, seq_len, 1)
        # 广播相乘后，非分隔符位置的概率被置为 0
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        # 3> 遍历批次中的每个样本，提取步骤奖励
        all_scores_res = []
        for i in range(probabilities.size(0)):
            # 4> 获取当前样本的概率：shape (seq_len, num_labels)
            sample = probabilities[i]  # seq_len, num_labels
            
            # 5> 提取有效位置的正向概率：
            # sample[sample != 0]: 过滤掉所有为 0 的元素（非分隔符位置）
            # view(-1, 2): 重塑为 (num_steps, 2) 形状
            # [:, 1]: 提取正向概率（第 2 列，索引为 1）
            # 例如：如果有 3 个步骤，positive_probs shape=(3,)，值为 [0.8, 0.6, 0.9]
            # 技巧：PyTorch 张量布尔索引机制
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            
            # 6> 转换为 Python 列表：将 tensor 转换为 CPU 上的列表
            non_zero_elements_list = positive_probs.cpu().tolist()
            
            # 7> 添加到结果列表
            all_scores_res.append(non_zero_elements_list)
        
        return all_scores_res

    def decode_prm(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Any:
        """
        功能：
            解码 PRM 输出，从 input_ids 中定位步骤分隔符，并计算对应的奖励分数。

        参数：
            input_ids (torch.Tensor): 输入的 token IDs。
                - shape: (batch_size, seq_len)
            logits (torch.Tensor): 模型输出的 logits。
                - shape: (batch_size, seq_len, num_labels)

        返回：
            List[List[float]]: 每个样本的步骤奖励列表。

        示例：
            >>> # 假设 <extra_0> 的 token id 是 151643
            >>> input_ids = torch.tensor([[1, 2, 151643, 4, 151643, 6]])  # 1 个样本，2 个步骤
            >>> logits = torch.randn(1, 6, 2)
            >>> rewards = template.decode_prm(input_ids, logits)
            >>> len(rewards[0])
            2  # 2 个步骤的奖励
        """
        # 1> 编码步骤分隔符：获取 <extra_0> 对应的 token ID
        # 例如：step_sep_id = 151643
        step_sep_id = self.tokenizer.encode(self.cot_process_placeholder)[0]
        
        # 2> 创建掩码：标记出所有步骤分隔符的位置
        # token_masks shape: (batch_size, seq_len)
        # 值为 True 的位置表示是步骤分隔符
        token_masks = (input_ids == step_sep_id)
        
        # 3> 计算步骤奖励：调用 make_step_rewards 提取奖励分数
        return self.make_step_rewards(logits, token_masks)


# 注册 Qwen2.5-Math PRM 模板：用于数学推理过程的奖励建模
register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math_prm, template_cls=QwenPRMTemplate))


class QwenVLTemplate(Template):
    """
    类功能：
        Qwen-VL 视觉语言模型的模板类。处理图像+文本的多模态输入，支持图像标注（grounding）任务。
        使用自定义的图像占位符格式 <img>...</img>，兼容不同的推理后端（vLLM、LMDeploy 等）。

    继承关系：
        继承自 Template 基类。

    应用场景：
        图像问答、图像描述、视觉定位（grounding）、目标检测等视觉-语言任务。

    使用示例：
        >>> # 图像问答示例
        >>> inputs = StdTemplateInputs(
        ...     images=['path/to/image.jpg'],
        ...     messages=[
        ...         {'role': 'user', 'content': '<image>描述这张图片'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... )
        >>> # 处理后格式：Picture 1: <img>path/to/image.jpg</img>\n描述这张图片
    """
    # 不提前加载图像：图像路径以字符串形式传递
    load_images = False

    @staticmethod
    def _load_image(image, load_images: bool):
        """
        功能：
            自定义图像加载逻辑。检测图像是否为 base64 编码或长字符串，决定是否需要加载。

        参数：
            image: 图像对象，可能是路径字符串、URL 或 PIL.Image。
            load_images (bool): 是否加载图像。

        返回：
            加载后的图像对象。

        示例：
            >>> # 普通路径：不加载
            >>> QwenVLTemplate._load_image('/path/to/img.jpg', False)
            '/path/to/img.jpg'
            
            >>> # Base64 编码：强制加载
            >>> QwenVLTemplate._load_image('data:image/png;base64,iVBORw...', False)
            <PIL.Image.Image object>
        """
        # 检测是否为 base64 编码或长字符串（可能是嵌入的图像数据）
        # 1> 如果是字符串且以 'data:' 开头，说明是 base64 编码的图像
        # 2> 如果字符串长度超过 200，可能也是嵌入的图像数据
        # 这两种情况下需要强制加载图像
        if not load_images and isinstance(image, str) and (image.startswith('data:') or len(image) > 200):
            load_images = True
        
        # 调用父类的图像加载方法
        return Template._load_image(image, load_images)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换媒体标签为 Qwen-VL 特定的格式。根据不同的推理模式生成对应的图像占位符。

        参数：
            media_type (Literal['image', 'video', 'audio']): 媒体类型，仅支持 'image'。
            index (int): 图像在列表中的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 上下文列表，包含图像占位符和换行符。

        示例：
            >>> # LMDeploy 模式
            >>> template.mode = 'lmdeploy'
            >>> template.replace_tag('image', 0, inputs)
            ['Picture 1: ', [-100], '\n']
            
            >>> # vLLM 模式
            >>> template.mode = 'vllm'
            >>> template.replace_tag('image', 0, inputs)
            ['Picture 1: <img></img>\n']
            
            >>> # 普通模式
            >>> template.mode = 'pt'
            >>> inputs.images = ['/path/to/img.jpg']
            >>> template.replace_tag('image', 0, inputs)
            ['Picture 1: <img>/path/to/img.jpg</img>\n']
        """
        # 确保只处理图像类型
        assert media_type == 'image'
        
        # 根据推理模式生成不同的占位符
        if self.mode == 'lmdeploy':
            # LMDeploy 模式：使用特殊 token [-100] 作为图像占位符
            return [f'Picture {index + 1}: ', [-100], '\n']
        else:
            # 获取当前图像
            image = inputs.images[index]
            if self.mode == 'vllm':
                # vLLM 模式：使用空标签 <img></img>
                return [f'Picture {index + 1}: <img></img>\n']
            else:
                # 普通模式（PyTorch）：在标签中嵌入图像路径
                assert isinstance(image, str)
                return [f'Picture {index + 1}: <img>{image}</img>\n']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换引用标签（grounding 任务中的目标引用）。

        参数：
            ref (str): 引用文本（如 "person"、"car"）。
            index (int): 引用的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 格式化的引用标签。

        示例：
            >>> template.replace_ref('person', 0, inputs)
            ['<ref>person</ref>']
        """
        # 使用 <ref>...</ref> 标签包裹引用文本
        return [f'<ref>{ref}</ref>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换边界框标签（grounding 任务中的目标位置）。

        参数：
            bbox (List[int]): 边界框坐标 [x1, y1, x2, y2]。
            index (int): 边界框的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 格式化的边界框标签。

        示例：
            >>> template.replace_bbox([100, 200, 300, 400], 0, inputs)
            ['<box>(100,200),(300,400)</box>']
        """
        # 使用 <box>...</box> 标签包裹边界框坐标字符串
        return [f'<box>{self._get_bbox_str(bbox)}</box>']


# 注册 Qwen-VL 模板：用于视觉-语言任务
register_template(QwenTemplateMeta(MLLMTemplateType.qwen_vl, template_cls=QwenVLTemplate))


class QwenAudioTemplate(Template):
    """
    类功能：
        Qwen-Audio 音频语言模型的模板类。处理音频+文本的多模态输入，支持语音识别、音频问答等任务。
        使用自定义的音频占位符格式 <audio>...</audio>，通过 processor 处理音频信息。

    继承关系：
        继承自 Template 基类。

    应用场景：
        语音识别、音频描述、音乐分析、声音事件检测等音频-语言任务。

    使用示例：
        >>> inputs = StdTemplateInputs(
        ...     audios=['path/to/audio.wav'],
        ...     messages=[
        ...         {'role': 'user', 'content': '<audio>这段音频的内容是什么？'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... )
        >>> # 处理后格式：Audio 1:<audio>path/to/audio.wav</audio>\n这段音频的内容是什么？
    """

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换音频标签为 Qwen-Audio 特定的格式。

        参数：
            media_type (Literal['image', 'video', 'audio']): 媒体类型，仅支持 'audio'。
            index (int): 音频在列表中的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 包含音频占位符的上下文列表。

        示例：
            >>> inputs.audios = ['audio.wav']
            >>> template.replace_tag('audio', 0, inputs)
            ['Audio 1:<audio>audio.wav</audio>\\n']
        """
        # 确保只处理音频类型
        assert media_type == 'audio'
        # 获取音频列表和当前音频
        audios = inputs.audios
        audio = audios[index]
        # 确保音频是路径字符串
        assert isinstance(audio, str)
        # 返回格式化的音频标签
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _tokenize(self, context, **tokenizer_kwargs):
        """
        功能：
            tokenize 上下文，附加音频信息。通过 processor 处理音频标签，提取音频元数据。

        参数：
            context (str): 待 tokenize 的文本上下文（包含 <audio> 标签）。
            **tokenizer_kwargs: 传递给 tokenizer 的额外参数。

        返回：
            tokenize 后的结果。

        示例：
            >>> context = 'Audio 1:<audio>/path/audio.wav</audio>\\n描述这段音频'
            >>> tokens = template._tokenize(context)
        """
        # 1> 处理音频信息：从上下文中提取音频路径并处理
        audio_info = self.processor.process_audio(context)
        # 2> 调用父类 tokenize 方法，附加音频信息
        return super()._tokenize(context, audio_info=audio_info)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        功能：
            编码输入，生成模型可用的张量。处理音频信息并添加到编码结果中。

        参数：
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            Dict[str, Any]: 编码结果字典，包含 input_ids、audio_info 等。

        示例：
            >>> inputs = StdTemplateInputs(audios=['audio.wav'], messages=[...])
            >>> encoded = template._encode(inputs)
            >>> 'audio_info' in encoded
            True
        """
        # 1> 调用父类编码方法，获取基础编码结果（input_ids、labels 等）
        encoded = super()._encode(inputs)
        
        # 2> 构建音频文本：将所有音频路径包裹在 <audio> 标签中
        # 例如：'<audio>audio1.wav</audio><audio>audio2.wav</audio>'
        text = ''.join([f'<audio>{audio}</audio>' for audio in inputs.audios])
        
        # 3> 处理音频信息：从构建的文本中提取音频元数据
        audio_info = self.processor.process_audio(text)
        
        # 4> 添加音频信息到编码结果：如果有音频信息，添加到字典中
        if audio_info:
            tokenizer_kwargs = {'audio_info': audio_info}
            encoded.update(tokenizer_kwargs)
            encoded['tokenizer_kwargs'] = tokenizer_kwargs
        
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        功能：
            批处理数据整理器，将批次中的样本合并为模型输入。收集音频信息到批次级别。

        参数：
            batch (List[Dict[str, Any]]): 批次样本列表。
            padding_to (Optional[int]): 填充到的目标长度。

        返回：
            Dict[str, Any]: 批次级别的输入字典。

        示例：
            >>> batch = [{'input_ids': [...], 'audio_info': {...}}, ...]
            >>> collated = template._data_collator(batch)
            >>> 'audio_info' in collated
            True
        """
        # 1> 调用父类数据整理器，处理文本部分（input_ids、attention_mask 等）
        res = super()._data_collator(batch, padding_to=padding_to)
        
        # 2> 收集音频信息：如果批次中有音频信息，收集到列表中
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        
        return res


# 注册 Qwen-Audio 模板：用于音频-语言任务
register_template(QwenTemplateMeta(MLLMTemplateType.qwen_audio, template_cls=QwenAudioTemplate))


class Qwen2AudioTemplate(Template):
    """
    类功能：
        Qwen2-Audio 音频语言模型的模板类（第二代音频模型）。
        使用标准的音频标记格式 <|audio_bos|><|AUDIO|><|audio_eos|>，通过 feature extractor 提取音频特征。
        相比 Qwen-Audio，使用更标准化的音频处理流程。

    继承关系：
        继承自 Template 基类。

    应用场景：
        语音识别、音频理解、音频描述等音频-语言任务，支持更长的音频输入。

    使用示例：
        >>> inputs = StdTemplateInputs(
        ...     audios=['speech.wav'],
        ...     messages=[
        ...         {'role': 'user', 'content': '<audio>识别这段语音'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... )
        >>> # 处理后格式：<|audio_bos|><|AUDIO|><|audio_eos|>\n识别这段语音
    """

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换音频标签为 Qwen2-Audio 的标准格式。根据是否使用 chat 模板选择不同的格式。

        参数：
            media_type (Literal['image', 'video', 'audio']): 媒体类型，仅支持 'audio'。
            index (int): 音频在列表中的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 包含音频占位符的上下文列表。

        示例：
            >>> # 不使用 chat 模板
            >>> template.use_chat_template = False
            >>> template.replace_tag('audio', 0, inputs)
            ['<|audio_bos|><|AUDIO|><|audio_eos|>\\n']
            
            >>> # 使用 chat 模板
            >>> template.use_chat_template = True
            >>> template.replace_tag('audio', 0, inputs)
            ['Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\\n']
        """
        # 确保只处理音频类型
        assert media_type == 'audio'
        
        # 根据是否使用 chat 模板选择格式
        if not self.use_chat_template:
            # 基础格式：只有音频标记
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']
        else:
            # Chat 格式：添加 "Audio N:" 前缀
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        功能：
            编码输入，提取音频特征并生成模型输入。使用 feature extractor 将音频转换为特征向量。

        参数：
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            Dict[str, Any]: 编码结果，包含 input_ids、input_features、feature_attention_mask 等。

        示例：
            >>> inputs = StdTemplateInputs(audios=['audio.wav'], messages=[...])
            >>> encoded = template._encode(inputs)
            >>> 'input_features' in encoded  # 音频特征
            True
            >>> encoded['input_features'].shape  # (batch, time, feature_dim)
            torch.Size([1, 3000, 128])
        """
        # 1> 调用父类编码方法，处理文本部分
        encoded = super()._encode(inputs)
        
        # 2> 处理音频：如果有音频输入，提取特征
        if inputs.audios:
            # 3> 获取采样率：优先使用环境变量，否则使用默认值
            sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)
            
            # 4> 批量加载音频：使用 load_audio 函数加载所有音频文件
            # load_batch 并行加载多个音频，提高效率
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate))
            
            # 5> 提取音频特征：使用 feature extractor 将音频波形转换为特征向量
            # return_attention_mask=True: 返回注意力掩码，处理变长音频
            # return_tensors='pt': 返回 PyTorch 张量
            audio_inputs = self.processor.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt')
            
            # 6> 重命名 attention_mask：避免与文本的 attention_mask 冲突
            # 将 'attention_mask' 改名为 'feature_attention_mask'
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            
            # 7> 更新编码结果：添加音频特征到字典中
            encoded.update(audio_inputs)
        
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        功能：
            批处理数据整理器，合并音频特征和注意力掩码。

        参数：
            batch (List[Dict[str, Any]]): 批次样本列表。
            padding_to (Optional[int]): 填充到的目标长度。

        返回：
            Dict[str, Any]: 批次级别的输入字典。

        示例：
            >>> batch = [
            ...     {'input_ids': [1,2,3], 'input_features': tensor(...), 'feature_attention_mask': tensor(...)},
            ...     {'input_ids': [4,5,6], 'input_features': tensor(...), 'feature_attention_mask': tensor(...)}
            ... ]
            >>> collated = template._data_collator(batch)
            >>> collated['input_features'].shape  # (2, max_time, feature_dim)
            torch.Size([2, 3000, 128])
        """
        # 1> 调用父类数据整理器，处理文本部分
        res = super()._data_collator(batch, padding_to=padding_to)
        
        # 2> 收集音频特征：从批次中提取所有非空的 input_features
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        
        # 3> 收集特征注意力掩码：标记音频中的有效部分
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        
        # 4> 合并音频数据：如果有音频特征，拼接成批次张量
        if input_features:
            # concat 沿批次维度拼接：(batch, time, feature_dim)
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        
        return res


# 注册 Qwen2-Audio 模板：用于音频-语言任务
register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplate))


class Qwen2VLTemplate(Template):
    """
    类功能：
        Qwen2-VL 视觉语言模型的模板类。支持图像和视频输入，使用动态分辨率处理。
        采用 Vision Transformer 架构，将图像/视频切分为多个 patch，每个 patch 用特定 token 表示。
        支持 grounding 任务（目标定位），可以输出边界框坐标。

    继承关系：
        继承自 Template 基类。

    应用场景：
        图像问答、视频理解、视觉定位、OCR、图表分析等视觉-语言任务。

    使用示例：
        >>> # 图像问答
        >>> inputs = StdTemplateInputs(
        ...     images=['cat.jpg'],
        ...     messages=[
        ...         {'role': 'user', 'content': '<image>这是什么动物？'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... )
        >>> 
        >>> # 视频理解
        >>> inputs = StdTemplateInputs(
        ...     videos=['video.mp4'],
        ...     messages=[
        ...         {'role': 'user', 'content': '<video>描述视频内容'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... )
    """
    # 图像 token ID：用于标记图像 patch 的特殊 token
    image_token_id = 151655
    
    # 视频 token ID：用于标记视频帧 patch 的特殊 token
    video_token_id = 151656
    
    # 占位符 tokens：在 tokenize 时临时使用，后续会被替换为实际数量的 token
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    
    # 模板版本：'v2' 表示 Qwen2-VL
    version = 'v2'
    
    # 是否使用模型对象：需要访问模型来获取视觉编码器
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换媒体标签为 Qwen2-VL 的格式。处理图像和视频输入，使用 qwen_vl_utils 加载媒体。

        参数：
            media_type (Literal['image', 'video', 'audio']): 媒体类型，支持 'image' 和 'video'。
            index (int): 媒体在列表中的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 格式化的视觉标签。

        示例：
            >>> # 图像模式
            >>> inputs.images = ['cat.jpg']
            >>> template.replace_tag('image', 0, inputs)
            ['<|vision_start|><|image_pad|><|vision_end|>']
            
            >>> # 视频模式
            >>> inputs.videos = ['video.mp4']
            >>> template.replace_tag('video', 0, inputs)
            ['<|vision_start|><|video_pad|><|vision_end|>']
        """
        # 导入 qwen_vl_utils：用于加载和预处理图像/视频
        from qwen_vl_utils import fetch_image, fetch_video
        
        # 确保媒体类型是图像或视频
        assert media_type in {'image', 'video'}
        
        if media_type == 'image':
            # 1> 加载图像：使用 fetch_image 获取图像数据（PIL.Image 或 numpy array）
            # 支持多种格式：本地路径、URL、base64 等
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            
            # 2> 生成图像占位符：根据推理模式选择格式
            if self.mode == 'lmdeploy':
                # LMDeploy 模式：使用特殊 token [-100]
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                # 普通模式：使用占位符 token，后续会被替换为实际数量的图像 tokens
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            # 3> 处理视频输入
            video = inputs.videos[index]
            
            # 4> 如果视频是目录（包含帧图像），收集所有帧文件
            if os.path.isdir(video):
                video = [os.path.join(video, fname) for fname in os.listdir(video)]
            
            # 5> 加载视频：fetch_video 返回视频张量和元数据（如 fps）
            # return_video_sample_fps=True: 返回视频采样帧率
            video, video_kwargs = fetch_video({'video': video}, return_video_sample_fps=True)
            
            # 6> 转换视频数据类型：确保是 uint8 格式（0-255 范围）
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            
            # 7> 更新视频数据：存储为 (video_tensor, metadata) 元组
            inputs.videos[index] = (video, video_kwargs)
            
            # 8> 生成视频占位符：后续会被替换为实际数量的视频 tokens
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换引用标签（用于 grounding 任务中的目标引用）。

        参数：
            ref (str): 引用文本（如 "cat"、"person"）。
            index (int): 引用的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 格式化的引用标签。

        示例：
            >>> template.replace_ref('cat', 0, inputs)
            ['<|object_ref_start|>cat<|object_ref_end|>']
        """
        # 使用 Qwen2-VL 的引用标签格式
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换边界框标签（用于 grounding 任务中的目标位置）。

        参数：
            bbox (List[int]): 边界框坐标 [x1, y1, x2, y2]。
            index (int): 边界框的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 格式化的边界框标签。

        示例：
            >>> template.replace_bbox([100, 200, 300, 400], 0, inputs)
            ['<|box_start|>(100,200),(300,400)<|box_end|>']
        """
        # 使用 Qwen2-VL 的边界框标签格式
        return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        功能：
            编码输入，处理图像和视频，将占位符 token 替换为实际数量的媒体 tokens。
            核心步骤：
            1. 使用 processor 提取图像/视频特征和网格信息（grid_thw）
            2. 根据网格尺寸计算实际需要的 token 数量
            3. 将单个占位符 token 扩展为多个实际 token
            grid_thw = 特征网格在 时间 × 高 × 宽 三个维度上的结构
            
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            Dict[str, Any]: 编码结果，包含：
                - input_ids: 扩展后的 token IDs
                - labels: 扩展后的标签
                - pixel_values/pixel_values_videos: 图像/视频像素值
                - image_grid_thw/video_grid_thw: 网格尺寸 (temporal, height, width)

        示例：
            >>> # 假设图像被切分为 2×3 的网格，merge_size=2
            >>> # 原始 input_ids: [1, 2, 151655, 3]  (151655 是图像占位符)
            >>> # grid_thw: (1, 2, 3) -> 6 个 grid cells
            >>> # merge_length: 2×2 = 4
            >>> # token_len: 6 // 4 = 1 (但实际会根据 merge 策略计算)
            >>> # 扩展后: [1, 2, 151655, 151655, ..., 3]  (多个图像 tokens)
        """
        # 1> 调用父类编码方法，获取基础编码结果（文本部分）
        encoded = super()._encode(inputs)
        processor = self.processor
        
        # 2> 提取编码后的文本信息
        input_ids = encoded['input_ids']  # token IDs 列表
        labels = encoded['labels']         # 标签列表（训练时使用）
        loss_scale = encoded.get('loss_scale', None)  # 损失缩放因子（可选）
        
        # 3> 准备图像和视频数据
        images = inputs.images
        # 从 (video_tensor, metadata) 元组中提取视频张量
        videos = [video[0] for video in inputs.videos]
        # 提取视频帧率信息
        fps = [video[1] for video in inputs.videos]
        
        # 4> 处理图像和视频（依次处理）
        for media_type in ['images', 'videos']:
            # 检查当前媒体类型是否有数据
            if locals()[media_type]:
                if media_type == 'images':
                    # 5> 处理图像
                    # 图像 token ID: 151655
                    media_token = self.image_token_id
                    
                    # 使用 image_processor 处理图像，提取特征
                    # do_resize=False: 不调整大小（已经在 fetch_image 中处理）
                    media_inputs = processor.image_processor(images=images, return_tensors='pt', do_resize=False)
                    
                    # 提取网格尺寸信息：shape (num_images, 3)，3个维度为 (temporal, height, width)
                    # 对于图像，temporal=1
                    # 例如：tensor([[1, 16, 16]])  表示 1 帧，16×16 个 patches
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    # 6> 处理视频
                    kwargs = {}
                    # 检查是否有专用的 video_processor
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        # 使用 image_processor 处理视频（逐帧处理）
                        processor_func = processor.image_processor
                        kwargs['images'] = None  # 明确指定不处理图像
                    
                    # 处理视频，提取特征
                    media_inputs = processor_func(videos=videos, return_tensors='pt', do_resize=False, **kwargs)
                    
                    # 提取视频网格尺寸：shape (num_videos, 3)，3个维度为 (temporal, height, width)
                    # 例如：tensor([[8, 16, 16]])  表示 8 帧，每帧 16×16 个 patches
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    
                    # 7> Qwen2.5-VL 特殊处理：计算每个 grid 的时间步长
                    if self.version == 'v2_5':
                        # temporal_patch_size: 时间维度的 patch 大小（如 2 表示每 2 帧合并）
                        # fps: 视频采样帧率
                        # second_per_grid_ts: 每个时间 grid 对应的秒数
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / tmp for tmp in fps
                        ]
                
                # 8> 查找占位符 token 的位置
                # findall 返回所有 media_token 在 input_ids 中的索引列表
                # 例如：[2, 15]  表示第 2 和第 15 个位置是媒体占位符
                idx_list = findall(input_ids, media_token)
                
                # 9> 计算合并长度：merge_size×merge_size
                # merge_size: 合并相邻 patches 的窗口大小（如 2 表示 2×2 合并）
                # 例如：merge_size=2 → merge_length=4，即 4 个 patches 合并为 1 个 token
                merge_length = processor.image_processor.merge_size**2

                # 10> 定义 token 扩展函数：计算每个媒体需要的 token 数量
                def _get_new_tokens(i):
                    """
                    计算第 i 个媒体需要的 token 数量。
                    
                    公式：token_len = (T × H × W) // merge_length
                    - T: temporal 维度（帧数）
                    - H: height 维度（patch 行数）
                    - W: width 维度（patch 列数）
                    - merge_length: 合并因子（如 4 表示 2×2 合并）
                    
                    示例：
                        grid_thw[i] = (8, 16, 16)  # 8 帧，16×16 patches
                        merge_length = 4           # 2×2 合并
                        token_len = 8×16×16 // 4 = 512
                        返回：[media_token] × 512
                    """
                    # 计算总 patch 数量并除以合并因子
                    # torch.Tensor.prod() 的功能：计算张量中所有元素的乘积（product）
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    # 返回对应数量的 media_token
                    return [media_token] * token_len

                # 11> 扩展 tokens：将占位符替换为实际数量的 tokens
                # _extend_tokens 会在每个 idx_list 位置替换为 _get_new_tokens 返回的 tokens
                # 同时同步扩展 labels 和 loss_scale
                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                
                # 12> 更新编码结果：添加媒体特征（pixel_values 等）
                encoded.update(media_inputs)

        # 13> 更新编码结果：保存扩展后的 input_ids、labels、loss_scale
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def forward_context(self, model, inputs):
        """
        功能：
            设置模型前向传播的上下文环境。处理位置编码（position_ids），支持 packing 模式和 Flash Attention。
            在 packing 模式下，使用真实的位置 ID（real_position_ids）来正确处理打包序列。

        参数：
            model: 模型对象。
            inputs (dict): 输入字典，包含 input_ids、position_ids 等。

        返回：
            上下文管理器或 None。

        示例：
            >>> # 普通模式
            >>> with template.forward_context(model, inputs):
            ...     output = model(**inputs)
            
            >>> # Packing 模式：使用 real_position_ids
            >>> inputs = {'real_position_ids': tensor(...), ...}
            >>> with template.forward_context(model, inputs):
            ...     output = model(**inputs)
        """
        # 1> 检查是否为 packing 模式：如果没有 real_position_ids，使用默认行为
        if 'real_position_ids' not in inputs:
            return super().forward_context(model, inputs)
        
        # 2> Packing 模式：替换位置 ID
        # position_ids 是原始的（可能被padding的）位置 ID
        position_ids = inputs['position_ids']
        # 使用真实的位置 ID（未padding的）
        inputs['position_ids'] = inputs.pop('real_position_ids')
        
        # 3> 检查 transformers 版本：>=4.53 原生支持 packing
        transformers_ge_453 = version.parse(transformers.__version__) >= version.parse('4.53')
        if transformers_ge_453:
            # 新版本：使用 get_packed_seq_params 获取 packing 参数
            inputs.update(get_packed_seq_params(position_ids))
            return super().forward_context(model, inputs)
        
        # 4> 旧版本：手动patch Flash Attention，根据版本加载对应的模块
        if self.version == 'v2':
            from transformers.models.qwen2_vl import modeling_qwen2_vl as modeling_module
        elif self.version == 'v2_5':
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling_module
        elif self.version == 'omni':
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as modeling_module
        
        # 5> 应用 Flash Attention patch
        return self._patch_flash_attention_forward(modeling_module, position_ids)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        功能：
            后处理编码结果，将视觉 token IDs 替换为视觉 embeddings。
            该方法在训练阶段将离散的视觉 token（如 151655）转换为连续的 embedding 向量，
            使得语言模型能够同时处理文本和视觉信息。主要步骤包括：
            1> 将 input_ids 通过 embed_tokens 转换为文本 embeddings
            2> 使用视觉编码器处理图像/视频像素值，得到视觉 embeddings
            3> 用视觉 embeddings 替换 input_ids 中对应位置的文本 embeddings

        参数：
            model: Qwen2-VL 模型对象，包含语言模型（base_model）和视觉编码器（visual）。
            inputs (Dict[str, Any]): 编码后的输入字典，包含：
                - input_ids: token IDs，shape (batch_size, seq_len)
                - pixel_values: 图像像素值，shape (num_images, C, H, W)（可选）
                - pixel_values_videos: 视频像素值，shape (num_videos, C, H, W)（可选）
                - image_grid_thw: 图像网格尺寸，shape (num_images, 3) 表示 (T, H, W)
                - video_grid_thw: 视频网格尺寸，shape (num_videos, 3) 表示 (T, H, W)

        返回：
            Dict[str, Any]: 包含 inputs_embeds 的字典
                - inputs_embeds: 融合后的 embeddings，shape (batch_size, seq_len, hidden_size)
                  其中视觉 token 位置已被视觉 embeddings 替换

        示例：
            >>> # 示例1：图像输入
            >>> inputs = {
            ...     'input_ids': tensor([[1, 2, 151655, 151655, 3]]),  # 包含 2 个图像 tokens
            ...     'pixel_values': tensor([[[...]]])  # 图像数据
            ... }
            >>> result = template._post_encode(model, inputs)
            >>> result['inputs_embeds'].shape
            torch.Size([1, 5, 4096])  # 位置 2-3 是视觉 embeddings
            
            >>> # 示例2：图像+视频混合输入
            >>> inputs = {
            ...     'input_ids': tensor([[1, 151655, 151656, 2]]),  # 1 个图像 token + 1 个视频 token
            ...     'pixel_values': tensor([[[...]]]),
            ...     'pixel_values_videos': tensor([[[[...]]]])
            ... }
            >>> result = template._post_encode(model, inputs)
        """
        # 1> 推理模式：不需要转换为 embeddings，直接返回 input_ids
        if not self.is_training:
            return inputs
        
        # 2> 提取输入数据
        input_ids = inputs['input_ids']  # shape: (batch_size, seq_len)
        pixel_values = inputs.get('pixel_values')  # 图像像素值（可选）
        pixel_values_videos = inputs.get('pixel_values_videos')  # 视频像素值（可选）
        image_grid_thw = inputs.get('image_grid_thw')  # 图像网格尺寸
        video_grid_thw = inputs.get('video_grid_thw')  # 视频网格尺寸

        # 3> 获取文本 embedding 层，根据模型结构选择正确的路径
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            # 标准结构：base_model.model.embed_tokens
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            # 嵌套结构：base_model.model.language_model.embed_tokens
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        # inputs_embeds shape: (batch_size, seq_len, hidden_size)
        # 例如：(2, 512, 4096) 表示 2 个样本，512 个 tokens，每个 token 4096 维

        # 4> 获取视觉编码器的数据类型，确保类型一致
        dtype = model.visual.get_dtype() if self.version == 'v2' else model.visual.dtype
        
        # 5> 处理纯文本情况（无图像和视频）
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            # 6> DeepSpeed 特殊处理：保持计算图连接
            # 即使没有视觉输入，也创建一个虚拟的视觉 embedding 并乘以 0
            # 这样可以避免 DeepSpeed 训练时出现计算图断裂的问题
            if is_deepspeed_enabled():
                from PIL import Image
                # 创建一个 32×32 的黑色占位图像
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                # 使用 image_processor 处理占位图像
                media_inputs = self.processor.image_processor(images=images, return_tensors='pt')
                device = input_ids.device
                # 将数据移动到与 input_ids 相同的设备
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                # 通过视觉编码器得到占位 embeddings
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                # 将占位 embeddings 的均值乘以 0 加到 inputs_embeds
                # 效果：保持计算图连接但不改变 inputs_embeds 的值
                inputs_embeds += image_embeds.mean() * 0.
        else:
            # 7> 有视觉输入的情况：处理图像和/或视频
            # 7.1> 合并图像和视频数据（如果都存在）
            if pixel_values is None:
                # 情况1：只有视频
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                # 情况2：只有图像
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                # 情况3：既有图像又有视频，沿批次维度拼接
                # pixel_values shape: (num_images, C, H, W)
                # pixel_values_videos shape: (num_videos, C, H, W)
                # 拼接后 shape: (num_images + num_videos, C, H, W)
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                # 拼接网格信息 shape: (num_images + num_videos, 3)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            
            # 7.2> 转换数据类型，确保与模型一致
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            
            # 7.3> 通过视觉编码器提取特征
            # mixed_embeds shape: (total_visual_tokens, hidden_size)
            # total_visual_tokens 是所有图像和视频的 token 总数
            mixed_embeds = model.visual(pixel_values_mixed, grid_thw=grid_thw)

            # 7.4> 分离图像和视频的 embeddings
            if pixel_values is None:
                # 只有视频
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                # 只有图像
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                # 既有图像又有视频：需要切分 mixed_embeds
                # 计算图像占用的 token 数量
                merge_length = self.processor.image_processor.merge_size**2
                # image_grid_thw.prod(dim=-1): 计算每个图像的总 patches 数
                # shape: (num_images,)，例如 [256, 1024] 表示第1张256个patches，第2张1024个
                # // merge_length: 除以合并因子（如 4），得到实际 token 数
                # .sum(): 所有图像的 token 总数
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                # 切分：前 image_tokens 个是图像 embeddings
                image_embeds = mixed_embeds[:image_tokens]
                # 剩余的是视频 embeddings
                video_embeds = mixed_embeds[image_tokens:]

            # 8> 将视觉 embeddings 填充到 input_ids 对应位置
            # 8.1> 处理图像 embeddings
            if image_embeds is not None:
                # ==================== 步骤1：创建图像 token 的布尔掩码 ====================
                # 操作：(input_ids == model.config.image_token_id)
                # 目的：找出所有图像 token 的位置
                # 
                # 实际例子：
                # 假设 input_ids = tensor([[1, 2, 151655, 151655, 3]])  # shape: (1, 5)
                #     model.config.image_token_id = 151655
                # 
                # 比较操作：逐元素判断是否等于 151655
                # 结果：tensor([[False, False, True, True, False]])  # shape: (1, 5), dtype=bool
                # 
                # 解释：位置 2 和 3 是图像 token（值为 151655），标记为 True
                
                # ==================== 步骤2：增加一个维度 (unsqueeze) ====================
                # 操作：.unsqueeze(-1)
                # 目的：为后续的扩展操作准备维度
                # 
                # 输入：tensor([[False, False, True, True, False]])  # shape: (1, 5)
                # 输出：tensor([[[False], [False], [True], [True], [False]]])  # shape: (1, 5, 1)
                # 
                # 解释：在最后添加一个维度，从 2D 变成 3D
                #      原来每个位置是一个标量 bool 值
                #      现在每个位置是一个长度为 1 的向量 [bool]
                
                # ==================== 步骤3：扩展维度 (expand_as) ====================
                # 操作：.expand_as(inputs_embeds)
                # 目的：将掩码扩展到与 embeddings 相同的 shape
                # 
                # 假设 inputs_embeds.shape = (1, 5, 4096)  # 4096 是 hidden_size
                # 输入掩码 shape: (1, 5, 1)
                # 输出掩码 shape: (1, 5, 4096)
                # 
                # 扩展过程：将最后一维从 1 复制 4096 次
                # 例如位置 [0, 2, :] 的变化：
                #   原来：[True]  # shape: (1,)
                #   扩展后：[True, True, True, ..., True]  # shape: (4096,)，所有值都是 True
                # 
                # 完整示例：
                # tensor([
                #   [[False, False, ..., False],  # 位置0：4096个False
                #    [False, False, ..., False],  # 位置1：4096个False
                #    [True,  True,  ..., True ],  # 位置2：4096个True（图像token）
                #    [True,  True,  ..., True ],  # 位置3：4096个True（图像token）
                #    [False, False, ..., False]]  # 位置4：4096个False
                # ])  # shape: (1, 5, 4096)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                
                # ==================== 步骤4：转换设备和数据类型 ====================
                # 操作：.to(inputs_embeds.device, inputs_embeds.dtype)
                # 目的：确保 image_embeds 与 inputs_embeds 在同一设备且类型相同
                # 
                # 实际例子：
                # 假设 image_embeds 在 CPU，dtype=float32
                #     inputs_embeds 在 GPU:0，dtype=float16
                # 
                # 操作后：image_embeds 也会在 GPU:0，dtype=float16
                # 
                # 这一步很重要，因为 PyTorch 不允许不同设备或类型的张量直接运算
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                
                # ==================== 步骤5：填充图像 embeddings (masked_scatter) ====================
                # 操作：inputs_embeds.masked_scatter(image_mask, image_embeds)
                # 目的：将图像 embedding 替换到文本 embedding 的对应位置
                # 
                # 详细的实际例子（简化版，用小数字表示）：
                # 
                # 假设 hidden_size = 3（实际是 4096）
                # 
                # 输入1 - inputs_embeds (文本embeddings):
                # tensor([
                #   [[0.1, 0.2, 0.3],   # 位置0: token_id=1 的 embedding
                #    [0.4, 0.5, 0.6],   # 位置1: token_id=2 的 embedding
                #    [0.7, 0.8, 0.9],   # 位置2: token_id=151655 的 embedding (待替换)
                #    [1.0, 1.1, 1.2],   # 位置3: token_id=151655 的 embedding (待替换)
                #    [1.3, 1.4, 1.5]]   # 位置4: token_id=3 的 embedding
                # ])  # shape: (1, 5, 3)
                # 
                # 输入2 - image_mask (掩码):
                # tensor([
                #   [[False, False, False],  # 位置0: 不替换
                #    [False, False, False],  # 位置1: 不替换
                #    [True,  True,  True ],  # 位置2: 替换（图像token）
                #    [True,  True,  True ],  # 位置3: 替换（图像token）
                #    [False, False, False]]  # 位置4: 不替换
                # ])  # shape: (1, 5, 3)
                # 
                # 输入3 - image_embeds (图像embeddings):
                # tensor([
                #   [9.1, 9.2, 9.3],  # 第1个图像patch的embedding
                #   [9.4, 9.5, 9.6]   # 第2个图像patch的embedding
                # ])  # shape: (2, 3)
                # 
                # masked_scatter 的工作原理：
                # 1. 将 image_embeds 展平：[9.1, 9.2, 9.3, 9.4, 9.5, 9.6]  # 6个元素
                # 2. 将 image_mask 为 True 的位置按顺序填充：
                #    - 位置[0,2,0]: 9.1 (第1个True位置)
                #    - 位置[0,2,1]: 9.2 (第2个True位置)
                #    - 位置[0,2,2]: 9.3 (第3个True位置)
                #    - 位置[0,3,0]: 9.4 (第4个True位置)
                #    - 位置[0,3,1]: 9.5 (第5个True位置)
                #    - 位置[0,3,2]: 9.6 (第6个True位置)
                # 
                # 输出 - inputs_embeds (融合后的embeddings):
                # tensor([
                #   [[0.1, 0.2, 0.3],   # 位置0: 保持不变
                #    [0.4, 0.5, 0.6],   # 位置1: 保持不变
                #    [9.1, 9.2, 9.3],   # 位置2: 已替换为图像embedding
                #    [9.4, 9.5, 9.6],   # 位置3: 已替换为图像embedding
                #    [1.3, 1.4, 1.5]]   # 位置4: 保持不变
                # ])  # shape: (1, 5, 3)
                # 
                # 关键点：
                # - masked_scatter 会按照 mask 中 True 的顺序依次填充
                # - image_embeds 必须有足够的元素（True的数量 × hidden_size）
                # - 本例中：6个True × 1 = 6个元素，正好对应 image_embeds 的 2×3=6 个元素
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 8.2> 处理视频 embeddings（流程与图像相同）
            if video_embeds is not None:
                # 创建视频 token 的掩码（token_id=151656）
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                # 确保设备和数据类型匹配
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                # 填充视频 embeddings
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # 9> 返回包含融合后 embeddings 的字典
        return {'inputs_embeds': inputs_embeds}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        功能：
            多模态数据的批处理整理器，收集和合并批次中的多模态特定数据。
            主要负责整理 Qwen2-VL 特有的网格尺寸信息（grid_thw）和时间信息（second_per_grid_ts）。

        参数：
            batch (List[Dict[str, Any]]): 批次样本列表，每个样本是一个字典，可能包含：
                - image_grid_thw: 图像网格尺寸，shape (3,) 表示 (T, H, W)
                - video_grid_thw: 视频网格尺寸，shape (3,) 表示 (T, H, W)
                - second_per_grid_ts: 每个时间 grid 的秒数（Qwen2.5-VL 专用）

        返回：
            Dict[str, Any]: 批次级别的数据字典，包含：
                - image_grid_thw: 合并后的图像网格尺寸，shape (total_images, 3)
                - video_grid_thw: 合并后的视频网格尺寸，shape (total_videos, 3)
                - second_per_grid_ts: 时间信息列表（如果存在）

        示例：
            >>> # 示例：2 个样本，第 1 个有图像，第 2 个有视频
            >>> batch = [
            ...     {'image_grid_thw': tensor([1, 16, 16])},
            ...     {'video_grid_thw': tensor([8, 16, 16])}
            ... ]
            >>> result = template._data_collator_mm_data(batch)
            >>> result['image_grid_thw'].shape
            torch.Size([1, 3])  # 1 张图像
            >>> result['video_grid_thw'].shape
            torch.Size([1, 3])  # 1 个视频
        """
        # 1> 调用父类方法，处理通用的多模态数据
        res = super()._data_collator_mm_data(batch)
        
        # 2> 收集时间信息（Qwen2.5-VL 专用）
        # gather_list: 从批次中收集所有非空的 second_per_grid_ts
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            # 将时间信息列表添加到结果中
            res['second_per_grid_ts'] = second_per_grid_ts
        
        # 3> 收集图像和视频的网格尺寸信息
        for media_type in ['image', 'video']:
            # concat_tensor: 沿指定维度（dim=0）拼接批次中的网格尺寸
            # 例如：[tensor([1,16,16]), tensor([1,8,8])] → tensor([[1,16,16], [1,8,8]])
            grid_thw = self.concat_tensor(batch, f'{media_type}_grid_thw', 0)
            if grid_thw is not None:
                # 添加到结果字典：'image_grid_thw' 或 'video_grid_thw'
                res[f'{media_type}_grid_thw'] = grid_thw
        
        return res

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        功能：
            打包一行中的多个样本，将它们拼接成一个长序列以提高训练效率。
            Packing 技术可以减少 padding 带来的计算浪费，特别适用于变长序列训练。
            该方法特别处理了 Qwen2-VL 的位置编码，保存真实的位置 ID（real_position_ids）。

        参数：
            row (List[Dict[str, Any]]): 一行中的多个样本，每个样本包含：
                - input_ids: token IDs 列表
                - image_grid_thw/video_grid_thw: 网格尺寸（如果有图像/视频）

        返回：
            Dict[str, Any]: 打包后的数据字典，包含：
                - input_ids: 拼接后的 token IDs
                - real_position_ids: 真实的位置编码（用于 forward_context）
                - 其他字段（由父类处理）

        示例：
            >>> # 示例：打包 3 个样本
            >>> row = [
            ...     {'input_ids': [1, 2, 3]},           # 样本1，长度3
            ...     {'input_ids': [4, 5, 6, 7]},        # 样本2，长度4
            ...     {'input_ids': [8, 9]}               # 样本3，长度2
            ... ]
            >>> packed = template.packing_row(row)
            >>> packed['input_ids']
            [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 总长度9，无padding
            >>> packed['real_position_ids'].shape
            torch.Size([1, 9])  # 每个样本的位置从0开始
        """
        # 1> 为每个样本计算真实的位置 ID
        position_ids = []
        for r in row:
            # 复制样本，避免修改原始数据
            r = r.copy()
            # 将 input_ids 转换为张量并添加批次维度
            # [1, 2, 3] → tensor([[1, 2, 3]]), shape (1, seq_len)
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            # 计算该样本的位置 ID（考虑视觉 tokens 的特殊位置编码）
            position_ids.append(self._get_position_ids(r))
        
        # 2> 调用父类方法，执行实际的打包操作（拼接 input_ids 等）
        packed = super().packing_row(row)
        
        # 3> 拼接所有样本的位置 ID
        # 例如：[tensor([0,1,2]), tensor([0,1,2,3]), tensor([0,1])]
        #      → tensor([0,1,2,0,1,2,3,0,1]), shape (1, 9)
        # 注意：每个样本的位置从 0 开始，这样可以正确处理多个独立序列
        packed['real_position_ids'] = torch.concat(position_ids, dim=-1)
        
        return packed

    def _get_position_ids(self, inputs: Dict[str, Any]):
        """
        功能：
            计算 Qwen2-VL 的位置编码（position IDs），考虑视觉 tokens 的特殊位置信息。
            Qwen2-VL 使用 RoPE（Rotary Position Embedding），需要根据图像/视频的网格结构
            （grid_thw）计算每个 token 在时间、高度、宽度三个维度上的位置。
            
            注：该方法修复了 transformers 库的一个问题（PR #33487）

        参数：
            inputs (Dict[str, Any]): 输入字典，包含：
                - input_ids: token IDs，shape (batch_size, seq_len)
                - image_grid_thw: 图像网格尺寸，shape (num_images, 3)（可选）
                - video_grid_thw: 视频网格尺寸，shape (num_videos, 3)（可选）
                - attention_mask: 注意力掩码，shape (batch_size, seq_len)（可选）
                - second_per_grid_ts: 每个时间 grid 的秒数（Qwen2.5-VL 专用，可选）

        返回：
            torch.Tensor: 位置编码，shape (batch_size, seq_len, 3)
                最后一维表示 (temporal, height, width) 三个维度的位置

        示例：
            >>> # 示例：包含 1 张图像的输入
            >>> inputs = {
            ...     'input_ids': tensor([[1, 2, 151655, 151655, 3]]),  # 2 个图像 tokens
            ...     'image_grid_thw': tensor([[1, 16, 16]])  # 1帧，16×16 patches
            ... }
            >>> position_ids = template._get_position_ids(inputs)
            >>> position_ids.shape
            torch.Size([1, 5, 3])  # 每个 token 有 3 维位置信息
            >>> # position_ids[0, 2:4] 包含图像 tokens 的 2D 位置信息
        """
        # 1> 准备 Qwen2.5-VL 专用参数（时间信息）
        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            # Qwen2.5-VL 需要时间步长信息来正确计算视频的时间位置
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}
        
        # 2> 获取 RoPE 索引计算函数
        base_model = self.get_base_model(self.model)
        if hasattr(base_model, 'get_rope_index'):
            # 直接挂载在 base_model 上
            get_rope_index = base_model.get_rope_index
        else:
            # 嵌套在 base_model.model 中
            get_rope_index = base_model.model.get_rope_index
        
        # 3> 计算位置编码
        # get_rope_index 返回 (position_ids, rope_deltas)，我们只需要 position_ids
        # position_ids shape: (batch_size, seq_len, 3)
        # - 对于文本 tokens：通常是 (t, 0, 0)，t 是文本位置
        # - 对于图像 tokens：是 (0, h, w)，h 和 w 是 patch 在图像中的位置
        # - 对于视频 tokens：是 (t, h, w)，t 是帧索引，h 和 w 是 patch 位置
        position_ids, _ = get_rope_index(
            inputs['input_ids'],
            inputs.get('image_grid_thw'),  # 图像网格信息
            inputs.get('video_grid_thw'),  # 视频网格信息
            attention_mask=inputs.get('attention_mask'),
            **kwargs)  # Qwen2.5-VL 的时间信息
        
        # 4> 返回连续内存的 position_ids
        # contiguous() 确保张量在内存中是连续的，提高后续操作效率
        return position_ids.contiguous()

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        功能：
            批处理数据整理器的主入口，负责整理批次数据并计算位置编码。
            根据是否使用 packing 模式采用不同的位置编码处理策略：
            - Packing 模式：使用预先计算的 real_position_ids
            - 非 Packing 模式：实时计算 position_ids

        参数：
            batch (List[Dict[str, Any]]): 批次样本列表。
            padding_to (Optional[int]): 填充到的目标长度（可选）。

        返回：
            Dict[str, Any]: 批次级别的数据字典，包含：
                - input_ids: 批次 token IDs，shape (batch_size, max_seq_len)
                - attention_mask: 注意力掩码，shape (batch_size, max_seq_len)
                - position_ids: 位置编码（非 packing）或 real_position_ids（packing）
                - pixel_values, image_grid_thw 等多模态数据

        示例：
            >>> # 示例1：非 Packing 模式
            >>> batch = [
            ...     {'input_ids': [1, 2, 3], 'pixel_values': tensor(...)},
            ...     {'input_ids': [4, 5], 'pixel_values': tensor(...)}
            ... ]
            >>> result = template._data_collator(batch)
            >>> result['input_ids'].shape
            torch.Size([2, 3])  # padding 到最长样本的长度
            >>> 'position_ids' in result
            True  # 训练时自动计算 position_ids
            
            >>> # 示例2：Packing 模式
            >>> batch = [
            ...     {'input_ids': [1,2,3,4,5,6], 'real_position_ids': tensor([0,1,2,0,1,2])}
            ... ]
            >>> result = template._data_collator(batch)
            >>> 'real_position_ids' in result
            True  # 使用预先计算的真实位置 ID
        """
        # 1> 调用父类方法，执行基础的批处理操作（padding、拼接等）
        res = super()._data_collator(batch, padding_to=padding_to)
        
        # 2> 根据模式处理位置编码
        if self._packing:
            # Packing 模式：使用预先计算的真实位置 ID
            # 在 packing_row 中已经计算好，这里只需要拼接
            # concat_tensor: 沿序列维度（dim=-1）拼接
            res['real_position_ids'] = self.concat_tensor(batch, 'real_position_ids', -1)
        elif self.is_training:
            # 非 Packing 模式且训练中：实时计算位置编码
            # _get_position_ids 会根据 grid_thw 计算每个 token 的 3D 位置
            res['position_ids'] = self._get_position_ids(res)
        # 推理模式：不需要 position_ids（模型内部会处理）
        
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_vl, template_cls=Qwen2VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qvq,
        default_system=('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                        'Answer in the language of the question. You should think step-by-step.'),
        template_cls=Qwen2VLTemplate,
    ))


class Qwen2_5VLTemplate(Qwen2VLTemplate):
    """
    类功能：
        Qwen2.5-VL 视觉语言模型的模板类。继承自 Qwen2VLTemplate，使用更新的版本配置。

    继承关系：
        继承自 Qwen2VLTemplate。

    应用场景：
        与 Qwen2-VL 相同，支持图像和视频理解任务。

    使用示例：
        >>> # 与 Qwen2VLTemplate 使用方式相同
        >>> inputs = StdTemplateInputs(images=['image.jpg'], messages=[...])
    """
    # 版本标识：v2_5 表示 Qwen2.5-VL
    version = 'v2_5'
    
    # 边界框归一化方式：'none' 表示不进行归一化
    norm_bbox = 'none'


# 注册 Qwen2.5-VL 模板
register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_vl, template_cls=Qwen2_5VLTemplate))

# 注册 MiMo-VL 模板：小米开发的视觉语言模型，基于 Qwen2.5-VL 架构
register_template(
    QwenTemplateMeta(
        MLLMTemplateType.mimo_vl,
        template_cls=Qwen2_5VLTemplate,
        default_system='You are MiMo, an AI assistant developed by Xiaomi.'))


class Qwen2_5OmniTemplate(Qwen2_5VLTemplate):
    """
    类功能：
        Qwen2.5-Omni 全模态模型的模板类。支持图像、视频、音频三种模态的输入。
        相比 Qwen2.5-VL，增加了音频理解能力，并支持视频中的音频提取。
        采用统一的编码器架构处理所有模态。

    继承关系：
        继承自 Qwen2_5VLTemplate。

    应用场景：
        全模态理解任务，包括图像+音频、视频+音频、多模态混合等复杂场景。

    使用示例：
        >>> # 图像+音频混合输入
        >>> inputs = StdTemplateInputs(
        ...     images=['photo.jpg'],
        ...     audios=['speech.wav'],
        ...     messages=[
        ...         {'role': 'user', 'content': '<image><audio>图片和音频的内容是什么？'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... )
    """
    # 版本标识：omni 表示全模态
    version = 'omni'
    
    # 占位符 tokens：分别对应图像、音频、视频
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']

    def init_processor(self, processor) -> None:
        """
        功能：
            初始化 processor，加载全模态处理相关配置。

        参数：
            processor: Qwen2.5-Omni 的 processor 对象。

        返回：
            None

        示例：
            >>> template.init_processor(processor)
            >>> template.seconds_per_chunk  # 视频块的时长（秒）
            2.0
        """
        # 如果 processor 为空，直接返回
        if processor is None:
            return
        
        # 1> 调用父类初始化方法
        super().init_processor(processor)
        
        # 2> 导入 Qwen2.5-Omni 的默认配置
        from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
        default = Qwen2_5OmniProcessorKwargs._defaults
        
        # 3> 设置视频处理参数
        # seconds_per_chunk: 每个视频块的时长（秒），用于分块处理长视频
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        
        # position_id_per_seconds: 每秒对应的位置 ID 数量，用于时间位置编码
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        
        # 4> 设置音频处理参数（可通过环境变量覆盖）
        # use_audio_in_video: 是否从视频中提取音频（默认 False）
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        
        # sampling_rate: 音频采样率（Hz），默认使用 processor 的配置
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换媒体标签为 Qwen2.5-Omni 的格式。支持图像、音频、视频三种模态。
            特别支持从视频中提取音频（use_audio_in_video=True 时）。

        参数：
            media_type (Literal['image', 'video', 'audio']): 媒体类型。
            index (int): 媒体在列表中的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 格式化的媒体标签。

        示例：
            >>> # 图像
            >>> template.replace_tag('image', 0, inputs)
            ['<|vision_bos|><|IMAGE|><|vision_eos|>']
            
            >>> # 音频
            >>> template.replace_tag('audio', 0, inputs)
            ['<|audio_bos|><|AUDIO|><|audio_eos|>']
            
            >>> # 视频（带音频）
            >>> template.use_audio_in_video = True
            >>> template.replace_tag('video', 0, inputs)
            ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
        """
        # 导入 qwen_omni_utils：用于加载图像和视频
        from qwen_omni_utils import fetch_image, fetch_video
        
        if media_type == 'image':
            # 1> 处理图像：使用 fetch_image 加载图像数据
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            # 返回图像标签格式
            return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
        
        elif media_type == 'audio':
            # 2> 处理音频：非 vLLM 模式需要预加载音频
            if self.mode != 'vllm':
                # 使用 load_audio 加载音频文件，指定采样率
                inputs.audios[index] = load_audio(inputs.audios[index], self.sampling_rate)
            # 返回音频标签格式
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        
        elif media_type == 'video':
            # 3> 处理视频
            video = inputs.videos[index]
            
            # 4> 加载视频数据：转换为 uint8 格式
            inputs.videos[index] = fetch_video({'video': video}).to(torch.uint8)
            
            # 5> 如果启用视频音频提取
            if self.use_audio_in_video:
                import librosa
                
                # 6> 处理网络视频：需要使用 audioread 读取远程音频流
                if video.startswith('http://') or video.startswith('https://'):
                    import audioread
                    video = audioread.ffdec.FFmpegAudioFile(video)
                
                # 7> 使用 librosa 加载音频：返回 (audio_data, sampling_rate)
                # [0] 提取音频数据（numpy array）
                video = librosa.load(video, sr=self.sampling_rate)[0]
                
                # 8> 将音频插入到 audios 列表：标记为来自视频的音频
                # (video, 'video') 元组中的 'video' 标记表示音频来源
                inputs.audios.insert(inputs.audio_idx, (video, 'video'))
                inputs.audio_idx += 1  # 更新音频索引
                
                # 9> 返回视频+音频标签格式
                return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
            
            # 10> 仅视频（无音频）：返回纯视频标签格式
            return ['<|vision_bos|><|VIDEO|><|vision_eos|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        video_audios_mask = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
            else:
                video_audios_mask.append(False)
        video_audios_mask = torch.tensor(video_audios_mask)
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=False,
            return_tensors='pt')
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        # audio
        audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = (((audio_feature_lengths - 1) // 2 + 1 - 2) // 2 + 1)
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        if idx_list:
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)

        for media_type in ['image', 'video']:
            token = f'<|{media_type.upper()}|>'
            token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']

                    def _get_new_tokens_use_audio_in_video(i):
                        audio_token_indices = torch.arange(audio_lengths[i])
                        grid_thw = media_grid_thw[i]
                        height = grid_thw[1] // merge_size
                        width = grid_thw[2] // merge_size
                        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = torch.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)).reshape(-1)
                        video_token_indices = (
                            video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
                        tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
                        video_chunk_indexes = processor.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

                        res = []
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                res += token_id * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                res += audio_token_id * audio_seq_length
                        return res

                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded.update(media_inputs)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return Template._post_encode(self, model, inputs)

    def _get_position_ids(self, inputs: Dict[str, Any]):
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self.model.thinker.get_rope_index(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        return position_ids.contiguous()

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res

    def generate(self, model, *args, **kwargs):
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


# 注册 Qwen2.5-Omni 模板：用于全模态理解任务
register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_omni, template_cls=Qwen2_5OmniTemplate))


class Ovis1_6Template(Template):
    """
    类功能：
        Ovis 1.6 视觉语言模型的模板类。使用特殊的视觉 tokenizer 动态生成图像 tokens。
        支持动态分辨率和多分区处理（max_partition）。

    继承关系：
        继承自 Template 基类。

    应用场景：
        高分辨率图像理解、细粒度视觉任务、OCR 等需要保留图像细节的场景。

    使用示例：
        >>> inputs = StdTemplateInputs(
        ...     images=['high_res_image.jpg'],
        ...     messages=[
        ...         {'role': 'user', 'content': '<image>详细描述图片内容'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... )
    """
    # 是否跳过提示词：False 表示保留提示词
    skip_prompt = False
    
    # 是否使用模型对象：需要访问模型的 visual_tokenizer
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换图像标签为 Ovis 特定的占位符。使用 [-200] 作为临时占位符，后续会被替换。

        参数：
            media_type (Literal['image', 'video', 'audio']): 媒体类型，仅支持 'image'。
            index (int): 图像在列表中的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 包含特殊占位符的上下文列表。

        示例：
            >>> template.replace_tag('image', 0, inputs)
            [[-200], '\\n']
        """
        # 确保只处理图像类型
        assert media_type == 'image'
        # 返回特殊占位符：[-200] 会在 _encode 中被替换为实际的图像 tokens
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, [-200])
        added_tokens_len = 0
        pixel_values = []
        for i, idx in enumerate(idx_list):
            max_partition = get_env_args('max_partition', int, 9)
            raw_pixel_values, image_placeholders = self.model.visual_tokenizer.preprocess_image(
                images[i], max_partition=max_partition)
            input_ids = input_ids[:idx] + image_placeholders + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(image_placeholders) + labels[idx + 1:]
            pixel_values.append(raw_pixel_values)
            added_tokens_len += len(image_placeholders) - 1
        dtype = self.model.visual_tokenizer.dtype
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype)
        else:
            pixel_values = torch.zeros((1, 3, 384, 384), dtype=dtype)  # dummpy
        encoded.update({'input_ids': input_ids, 'labels': labels})
        encoded['pixel_values'] = [pixel_values]
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        padding_side = self.padding_side if self.is_training else 'left'
        if self.max_length is not None:
            model.config.multimodal_max_length = self.max_length
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        if labels is None:
            labels = input_ids.new_full(input_ids.shape, -100)
        _, inputs_embeds, labels, attention_mask = model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=torch.ones_like(input_ids),  # not use, only compat
            text_labels=labels,
            pixel_values=inputs['pixel_values'],
            left_padding=padding_side == 'left')
        if inputs.get('labels') is None:
            labels = None
        return {'inputs_embeds': inputs_embeds, 'labels': labels, 'attention_mask': attention_mask}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        res = super()._data_collator(batch, padding_to=padding_to)
        res['pixel_values'] = pixel_values
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.ovis1_6,
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
        template_cls=Ovis1_6Template,
    ))

# 注册 Ovis 1.6 模板：使用 Llama3 格式
register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.ovis1_6_llama3,
        default_system='You are a helpful and honest multimodal assistant.',
        template_cls=Ovis1_6Template,
    ))


class Ovis2Template(Ovis1_6Template):
    """
    类功能：
        Ovis2 视觉语言模型的模板类。继承自 Ovis1_6Template，增加了视频处理能力。
        将视频转换为多帧图像序列进行处理。

    继承关系：
        继承自 Ovis1_6Template。

    应用场景：
        图像理解、视频理解，支持高分辨率输入和多帧视频处理。

    使用示例：
        >>> # 视频理解
        >>> inputs = StdTemplateInputs(
        ...     videos=['video.mp4'],
        ...     messages=[
        ...         {'role': 'user', 'content': '<video>描述视频内容'},
        ...         {'role': 'assistant', 'content': ''}
        ...     ]
        ... )
    """
    # 占位符 tokens：用于图像和视频
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    
    # 默认视频帧数：从视频中提取的帧数
    nframes = 12

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """
        功能：
            替换媒体标签。支持图像和视频，视频会被转换为多帧图像。

        参数：
            media_type (Literal['image', 'video', 'audio']): 媒体类型，支持 'image' 和 'video'。
            index (int): 媒体在列表中的索引。
            inputs (StdTemplateInputs): 标准模板输入对象。

        返回：
            List[Context]: 格式化的媒体标签。

        示例：
            >>> # 图像
            >>> template.replace_tag('image', 0, inputs)
            [[-200], '\\n']
            
            >>> # 视频（12 帧）
            >>> template.replace_tag('video', 0, inputs)
            [[-200, -200, ..., -200], '\\n']  # 12 个 -200
        """
        if media_type == 'image':
            # 1> 处理图像
            if self.mode == 'vllm':
                # vLLM 模式：使用文本标签
                return ['<image>\n']
            # 普通模式：使用特殊占位符
            return [[-200], '\n']
        
        elif media_type == 'video':
            # 2> 处理视频
            # 获取帧数：可通过环境变量覆盖默认值
            nframes = get_env_args('nframes', int, self.nframes)
            
            # 3> 加载视频帧：将视频转换为图像列表
            # load_video_ovis2 返回 nframes 个图像
            inputs.images = load_video_ovis2(inputs.videos[index], nframes)
            
            # 4> 返回多个占位符：每帧一个
            # 例如：nframes=12 → [-200] * 12
            return [[-200] * nframes, '\n']


# 注册 Ovis2 模板：使用 Qwen 格式
register_template(QwenTemplateMeta(
    MLLMTemplateType.ovis2,
    template_cls=Ovis2Template,
))


@dataclass
class MarcoO1TemplateMeta(QwenTemplateMeta):
    """
    类功能：
        Marco-O1 模型的模板元数据类。专为推理任务设计，采用 <Thought>/<Output> 格式分离思考和输出。
        由阿里国际数字商业集团开发，强调逐步推理能力。

    继承关系：
        继承自 QwenTemplateMeta。

    应用场景：
        需要显式推理过程的任务，如数学推理、逻辑推理、复杂问题求解等。

    使用示例：
        >>> meta = MarcoO1TemplateMeta(LLMTemplateType.marco_o1)
        >>> # 模型输出格式：
        >>> # <Thought>
        >>> # Let's think step by step...
        >>> # </Thought>
        >>> # <Output>
        >>> # 最终答案
        >>> # </Output>
    """
    # 默认系统提示词：定义 Marco-O1 的行为准则
    # 关键点：
    # 1. 思考过程在 <Thought> 标签内，尽可能使用英文
    # 2. 最终输出在 <Output> 标签内，使用用户输入的语言
    # 3. 特例：原文引用和数学公式可在 <Thought> 中使用其他语言
    default_system: Optional[str] = """
你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.
        \n## 重要！！！！！
当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。
<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。
        """


# 注册 Marco-O1 模板：用于推理任务
register_template(MarcoO1TemplateMeta(LLMTemplateType.marco_o1))
