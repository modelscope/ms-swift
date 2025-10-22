"""模块功能概述：
该模块定义了采样器（Sampler）的基类，为不同类型的采样器提供统一的接口和基础功能。

核心功能：
1. Sampler 基类：定义采样器的通用接口和初始化流程；
2. 模型和分词器准备：加载和配置处理器（processor/tokenizer）；
3. 奖励模型准备：初始化 PRM（过程奖励模型）和 ORM（结果奖励模型）；
4. 模板准备：加载和配置对话模板（template），设置为训练模式；
5. 输入截断：提供输入数据截断的接口（子类可重写）；
6. 采样接口：定义抽象的 do_sample 方法，由子类实现具体的采样逻辑。

应用场景：
- 作为所有采样器的基类（VanillaSampler、MctsSampler、DistillSampler）；
- 统一管理采样器所需的模型资源（processor、template、PRM、ORM）；
- 为强化学习（RL）训练提供奖励模型支持；
- 为蒸馏训练提供教师模型支持。

继承关系：
    Sampler（基类）
    ├── VanillaSampler（普通采样器）
    ├── MctsSampler（蒙特卡洛树搜索采样器）
    └── DistillSampler（蒸馏采样器）

“vanilla” 在技术语境中指 “原生的、无修饰的、基础版本的”。

典型使用：
    # 不直接使用 Sampler 基类，而是使用其子类
    >>> from swift.llm import SamplingArguments
    >>> from swift.llm.sampling.vanilla_sampler import VanillaSampler
    >>> 
    >>> args = SamplingArguments(
    ...     model_id_or_path='qwen/Qwen-7B',
    ...     prm_model='prm_model_path',  # 过程奖励模型
    ...     orm_model='orm_model_path'   # 结果奖励模型
    ... )
    >>> sampler = VanillaSampler(args)  # VanillaSampler 继承自 Sampler
    >>> # sampler 已自动初始化 processor、template、prm_model、orm_model
"""
from typing import Any, Dict, List  # 引入类型注解，用于参数和返回值的类型提示：Any（任意类型）、Dict（字典）、List（列表）

from swift.llm import SamplingArguments  # 引入采样参数类，包含采样所需的所有配置参数（模型路径、数据集、采样器类型等）
from swift.plugin import orms, prms  # 引入奖励模型插件字典：prms（过程奖励模型映射）、orms（结果奖励模型映射），用于快速加载预定义的奖励模型
from swift.utils import get_logger  # 引入日志工具函数，用于创建模块级日志记录器

logger = get_logger()  # 创建模块级日志记录器，用于输出运行时信息、警告和错误日志


class Sampler:
    """类功能：
    定义采样器基类，为所有采样器提供统一的初始化流程和抽象接口。
    
    核心职责：
        1. 资源初始化：统一管理 processor、template、PRM、ORM 的加载和配置；
        2. 模板配置：加载对话模板并设置为训练模式；
        3. 奖励模型管理：根据参数加载 PRM 和 ORM，支持插件式加载和自定义模型；
        4. 接口定义：定义 truncate_input 和 do_sample 抽象接口，由子类实现；
        5. 资源复用：避免子类重复实现资源加载逻辑。
    
    属性：
        - args (SamplingArguments): 采样参数对象，包含所有配置信息；
        - template: 对话模板实例，用于格式化输入输出；
        - processor: 处理器实例（tokenizer），用于文本编码和解码；
        - prm_model: 过程奖励模型（Process Reward Model），用于评估生成过程中的每一步；
        - orm_model: 结果奖励模型（Outcome Reward Model），用于评估最终生成结果。

    outcome: 结果、结局、成果。
    
    方法：
        - __init__: 初始化采样器，加载所有必需的资源；
        - _prepare_model_tokenizer: 私有方法，准备处理器（tokenizer）；
        - _prepare_rm: 私有方法，准备奖励模型（PRM 和 ORM）；
        - _prepare_template: 私有方法，准备对话模板；
        - truncate_input: 截断输入数据（子类可重写）；
        - do_sample: 抽象方法，执行采样（子类必须实现）。
    
    实际使用示例：
        示例 1：通过子类使用基类功能（VanillaSampler）
        >>> from swift.llm import SamplingArguments
        >>> from swift.llm.sampling.vanilla_sampler import VanillaSampler
        >>> 
        >>> args = SamplingArguments(
        ...     model_id_or_path='qwen/Qwen-7B-Chat',
        ...     dataset=['alpaca-zh']
        ... )
        >>> sampler = VanillaSampler(args)
        >>> # 基类 Sampler.__init__ 已自动调用
        >>> # sampler.processor 已加载
        >>> # sampler.template 已配置
        >>> print(sampler.template.template_backend)  # 输出: 'jinja'
        
        示例 2：使用奖励模型（MctsSampler）
        >>> args = SamplingArguments(
        ...     model_id_or_path='qwen/Qwen-Math-7B',
        ...     dataset=['math'],
        ...     prm_model='Qwen/Qwen2.5-Math-PRM-7B',  # PRM 模型路径
        ...     orm_model='Qwen/Qwen2.5-Math-ORM-7B'   # ORM 模型路径
        ... )
        >>> from swift.llm.sampling.mcts import MctsSampler
        >>> sampler = MctsSampler(args)
        >>> # sampler.prm_model 和 sampler.orm_model 已加载为 PtEngine 实例
        >>> # 可用于评估采样质量
        
        示例 3：使用预定义的奖励模型插件
        >>> args = SamplingArguments(
        ...     model_id_or_path='qwen/Qwen-7B',
        ...     prm_model='builtin_prm_v1',  # 使用内置 PRM 插件
        ...     orm_model='builtin_orm_v1'   # 使用内置 ORM 插件
        ... )
        >>> sampler = VanillaSampler(args)
        >>> # sampler.prm_model 和 sampler.orm_model 从 prms 和 orms 字典中加载
    """

    def __init__(self, input_args: SamplingArguments):  # 定义初始化方法，接受 SamplingArguments 实例作为参数
        """函数功能：
        初始化采样器基类，加载处理器、模板和奖励模型等所有必需的资源。
        
        参数：
            input_args (SamplingArguments): 采样参数对象，包含模型路径、数据集、奖励模型配置等
        
        返回值：
            None
        
        初始化流程：
            1. 保存参数对象到 self.args；
            2. 初始化资源属性为 None（template、processor、prm_model、orm_model）；
            3. 调用 _prepare_model_tokenizer 加载处理器；
            4. 调用 _prepare_template 加载和配置对话模板；
            5. 调用 _prepare_rm 加载奖励模型（PRM 和 ORM）。
        
        实际使用示例：
            示例 1：基本初始化（通过子类）
            >>> from swift.llm import SamplingArguments
            >>> from swift.llm.sampling.vanilla_sampler import VanillaSampler
            >>> args = SamplingArguments(model_id_or_path='qwen/Qwen-7B')
            >>> sampler = VanillaSampler(args)
            >>> # Sampler.__init__ 被自动调用
            >>> print(sampler.args.model_id_or_path)  # 输出: 'qwen/Qwen-7B'
            >>> print(sampler.processor is not None)  # 输出: True
            >>> print(sampler.template is not None)   # 输出: True
            
            示例 2：带奖励模型的初始化
            >>> args = SamplingArguments(
            ...     model_id_or_path='qwen/Qwen-Math-7B',
            ...     prm_model='prm_path',
            ...     orm_model='orm_path'
            ... )
            >>> sampler = VanillaSampler(args)
            >>> # 初始化完成后，所有资源都已加载
            >>> print(sampler.prm_model is not None)  # 输出: True
            >>> print(sampler.orm_model is not None)  # 输出: True
        """
        self.args = input_args  # 保存传入的采样参数对象到实例属性 self.args，供后续方法使用
        self.template = None  # 初始化对话模板属性为 None，稍后在 _prepare_template 中加载
        self.processor = None  # 初始化处理器（tokenizer）属性为 None，稍后在 _prepare_model_tokenizer 中加载
        self.prm_model = None  # 初始化过程奖励模型属性为 None，稍后在 _prepare_rm 中加载（如果配置了）
        self.orm_model = None  # 初始化结果奖励模型属性为 None，稍后在 _prepare_rm 中加载（如果配置了）
        self._prepare_model_tokenizer()  # 调用私有方法加载处理器（tokenizer），为后续的文本编码解码做准备
        self._prepare_template()  # 调用私有方法加载和配置对话模板，设置为训练模式
        self._prepare_rm()  # 调用私有方法加载奖励模型（PRM 和 ORM），如果参数中配置了奖励模型路径

    def _prepare_model_tokenizer(self):
        """函数功能：
        私有方法，加载和准备处理器（tokenizer/processor），用于文本的编码和解码。
        
        参数：
            无（使用 self.args 中的配置）
        
        返回值：
            None（将 processor 保存到 self.processor）
        
        实现逻辑：
            1. 从 self.args 中获取参数对象；
            2. 调用 args.get_model_processor 方法加载处理器（不加载模型本身）；
            3. 将 processor 保存到 self.processor 属性。
        
        注意事项：
            - 设置 load_model=False 是因为采样器可能不需要在当前进程加载完整模型；
            - 模型可能在推理引擎中已加载，这里只需要 processor 用于数据预处理。
        
        实际使用示例：
            示例 1：加载 tokenizer（内部调用）
            >>> # 该方法在 Sampler.__init__ 中自动调用
            >>> sampler = VanillaSampler(args)
            >>> # sampler.processor 已加载
            >>> print(type(sampler.processor).__name__)
            # 输出: 'QWenTokenizer' 或其他 tokenizer 类型
            
            示例 2：使用加载的 processor
            >>> text = "你好，世界！"
            >>> encoded = sampler.processor(text)
            >>> print(encoded['input_ids'])  # 输出: tensor([...])（编码后的 token IDs）
        """
        args = self.args  # 获取采样参数对象（简化代码，提高可读性）
        _, self.processor = args.get_model_processor(load_model=False)  # 调用参数对象的 get_model_processor 方法加载处理器：返回元组 (model, processor)，这里只取 processor（不加载模型，load_model=False），并保存到 self.processor

    def _prepare_rm(self):
        """函数功能：
        私有方法，根据配置加载过程奖励模型（PRM）和结果奖励模型（ORM）。
        
        参数：
            无（使用 self.args 中的配置）
        
        返回值：
            None（将 prm_model 和 orm_model 保存到实例属性）
        
        实现逻辑：
            1. 处理 PRM 模型：
               a. 若 prm_model 为 None，设置为 None 并输出警告；
               b. 若 prm_model 在 prms 插件字典中，从插件加载；
               c. 否则，作为模型路径使用 PtEngine 加载。
            2. 处理 ORM 模型：
               a. 若 orm_model 为 None，设置为 None 并输出警告；
               b. 若 orm_model 在 orms 插件字典中，从插件加载；
               c. 否则，作为模型路径使用 PtEngine 加载。
        
        奖励模型说明：
            - PRM（Process Reward Model）：过程奖励模型，评估生成过程中每一步的质量；
            - ORM（Outcome Reward Model）：结果奖励模型，评估最终生成结果的质量。
        
        实际使用示例：
            示例 1：不使用奖励模型（仅警告）
            >>> args = SamplingArguments(
            ...     model_id_or_path='qwen/Qwen-7B',
            ...     prm_model=None,  # 不使用 PRM
            ...     orm_model=None   # 不使用 ORM
            ... )
            >>> sampler = VanillaSampler(args)
            # 输出警告: 'prm_model is None.'
            # 输出警告: 'orm_model is None.'
            >>> print(sampler.prm_model)  # 输出: None
            >>> print(sampler.orm_model)  # 输出: None
            
            示例 2：使用插件式奖励模型
            >>> # 假设 prms = {'builtin_v1': BuiltinPRM}
            >>> args = SamplingArguments(
            ...     model_id_or_path='qwen/Qwen-7B',
            ...     prm_model='builtin_v1',  # 使用内置 PRM 插件
            ...     orm_model='builtin_v1'   # 使用内置 ORM 插件
            ... )
            >>> sampler = VanillaSampler(args)
            >>> # sampler.prm_model 和 sampler.orm_model 从插件字典实例化
            
            示例 3：使用自定义模型路径
            >>> args = SamplingArguments(
            ...     model_id_or_path='qwen/Qwen-Math-7B',
            ...     prm_model='path/to/prm_model',  # 自定义 PRM 路径
            ...     orm_model='path/to/orm_model'   # 自定义 ORM 路径
            ... )
            >>> sampler = VanillaSampler(args)
            >>> # sampler.prm_model 和 sampler.orm_model 使用 PtEngine 加载
            >>> print(type(sampler.prm_model).__name__)  # 输出: 'PtEngine'
        """
        if self.args.prm_model is None:  # 若参数中的 prm_model 为 None（用户未配置 PRM 模型）
            self.prm_model = None  # 将 self.prm_model 设置为 None（不使用 PRM）
            logger.warning('prm_model is None.')  # 记录警告日志：提示用户未配置 PRM 模型（可能影响采样质量评估）
        elif self.args.prm_model in prms:  # 若 prm_model 存在于 prms 插件字典中（使用预定义的插件式 PRM）
            self.prm_model = prms[self.args.prm_model]()  # 从 prms 字典中获取对应的 PRM 类并实例化（调用无参构造函数）
        else:  # 否则（prm_model 是自定义的模型路径）
            from swift.llm import PtEngine  # 延迟导入 PtEngine 类（PyTorch 推理引擎），避免循环导入
            self.prm_model = PtEngine(self.args.prm_model, max_batch_size=64)  # 使用 PtEngine 加载 PRM 模型：传入模型路径和最大批量大小（64），创建推理引擎实例

        if self.args.orm_model is None:  # 若参数中的 orm_model 为 None（用户未配置 ORM 模型）
            self.orm_model = None  # 将 self.orm_model 设置为 None（不使用 ORM）
            logger.warning('orm_model is None.')  # 记录警告日志：提示用户未配置 ORM 模型（可能影响采样结果质量评估）
        elif self.args.orm_model in orms:  # 若 orm_model 存在于 orms 插件字典中（使用预定义的插件式 ORM）
            self.orm_model = orms[self.args.orm_model]()  # 从 orms 字典中获取对应的 ORM 类并实例化（调用无参构造函数）
        else:  # 否则（orm_model 是自定义的模型路径）
            from swift.llm import PtEngine  # 延迟导入 PtEngine 类（PyTorch 推理引擎），避免循环导入
            self.orm_model = PtEngine(self.args.orm_model, max_batch_size=64)  # 使用 PtEngine 加载 ORM 模型：传入模型路径和最大批量大小（64），创建推理引擎实例

    def _prepare_template(self) -> None:
        """函数功能：
        私有方法，加载对话模板并设置为训练模式。
        
        参数：
            无（使用 self.args 和 self.processor）
        
        返回值：
            None（将 template 保存到 self.template）
        
        实现逻辑：
            1. 调用 args.get_template 方法获取对话模板实例；
            2. 将模板保存到 self.template；
            3. 调用 template.set_mode('train') 设置为训练模式。
        
        模板模式说明：
            - 'train' 模式：用于训练和采样，保留完整的对话格式；
            - 'infer' 模式：用于推理，可能省略某些训练特定的标记。
        
        实际使用示例：
            示例 1：加载默认模板（内部调用）
            >>> # 该方法在 Sampler.__init__ 中自动调用
            >>> sampler = VanillaSampler(args)
            >>> # sampler.template 已加载并设置为训练模式
            >>> print(sampler.template.template_backend)  # 输出: 'jinja'
            
            示例 2：使用模板格式化对话
            >>> messages = [
            ...     {'role': 'system', 'content': '你是一个有帮助的助手'},
            ...     {'role': 'user', 'content': '你好'}
            ... ]
            >>> formatted = sampler.template.encode(messages)
            >>> print(formatted['input_ids'].shape)  # 输出: torch.Size([seq_len])（编码后的 token IDs）
        """
        template = self.args.get_template(self.processor)  # 调用参数对象的 get_template 方法获取对话模板实例：传入 processor 用于文本编码，返回配置好的 template 对象
        self.template = template  # 将获取的模板实例保存到 self.template 属性，供后续的数据格式化使用
        self.template.set_mode('train')  # 调用模板的 set_mode 方法设置为训练模式（'train'），确保生成的数据包含训练所需的所有标记和格式

    def truncate_input(self, slices: List[Dict[str, Any]]):
        """函数功能：
        截断输入数据，接受数据切片列表作为参数，避免超过策略模型的最大长度限制。
        
        参数：
            slices (List[Dict[str, Any]]): 输入数据切片列表，每个元素为字典，包含样本的各个字段
                示例：[
                    {'messages': [{'role': 'user', 'content': '问题1'}], ...},
                    {'messages': [{'role': 'user', 'content': '问题2'}], ...}
                ]
        
        返回值：
            List[Dict[str, Any]]: 截断后的数据切片列表（基类默认不截断，直接返回原数据）
        
        实现说明：
            - 基类提供默认实现：不进行任何截断，直接返回原始数据；
            - 子类可以重写此方法，实现自定义的截断逻辑（如截断过长的对话历史）。
        
        实际使用示例：
            示例 1：基类默认行为（不截断）
            >>> sampler = VanillaSampler(args)
            >>> slices = [
            ...     {'messages': [{'role': 'user', 'content': '非常长的问题...'}]},
            ...     {'messages': [{'role': 'user', 'content': '另一个问题'}]}
            ... ]
            >>> result = sampler.truncate_input(slices)
            >>> print(result == slices)  # 输出: True（未截断，返回原数据）
            
            示例 2：子类重写截断逻辑
            >>> class CustomSampler(Sampler):
            ...     def truncate_input(self, slices):
            ...         # 自定义截断：只保留每条消息的前 100 个字符
            ...         for item in slices:
            ...             for msg in item['messages']:
            ...                 msg['content'] = msg['content'][:100]
            ...         return slices
            >>> sampler = CustomSampler(args)
            >>> result = sampler.truncate_input(slices)
            # result 中的消息内容已被截断到 100 字符
        """
        return slices  # 返回未经修改的输入数据切片列表（基类默认实现，子类可重写以实现自定义截断逻辑）

    def do_sample(self, data):
        """函数功能：
        定义抽象采样方法，接受数据作为参数，执行采样操作，从语言模型中生成输出（抽象方法，子类必须实现）。
        
        参数：
            data: 输入数据，具体格式由子类定义（通常为数据切片列表或批量数据）
        
        返回值：
            具体返回值由子类定义（通常为生成的样本列表或 JSONL 格式字符串）
        
        异常：
            NotImplementedError: 基类未实现此方法，调用时会抛出异常
        
        实现说明：
            - 这是一个抽象方法，基类不提供实现；
            - 子类必须重写此方法，实现具体的采样逻辑；
            - 不同的采样器（VanillaSampler、MctsSampler、DistillSampler）有不同的采样策略。
        
        实际使用示例：
            示例 1：VanillaSampler 的实现（普通采样）
            >>> # VanillaSampler.do_sample 的实现
            >>> class VanillaSampler(Sampler):
            ...     def do_sample(self, data):
            ...         # 1. 格式化输入数据
            ...         # 2. 调用推理引擎生成输出
            ...         # 3. 后处理生成结果
            ...         # 4. 返回 JSONL 格式的字符串列表
            ...         return ['{"prompt": "...", "response": "..."}\n', ...]
            
            示例 2：MctsSampler 的实现（MCTS 采样）
            >>> class MctsSampler(Sampler):
            ...     def do_sample(self, data):
            ...         # 1. 构建搜索树
            ...         # 2. 执行蒙特卡洛树搜索
            ...         # 3. 使用 PRM/ORM 评估节点
            ...         # 4. 选择最优路径
            ...         # 5. 返回采样结果
            ...         return [...]
            
            示例 3：直接调用会抛出异常
            >>> sampler = Sampler(args)  # 不应直接实例化基类
            >>> sampler.do_sample(data)
            # 抛出: NotImplementedError（基类未实现此方法）
        """
        raise NotImplementedError  # 抛出 NotImplementedError 异常，提示子类必须重写此方法实现具体的采样逻辑
