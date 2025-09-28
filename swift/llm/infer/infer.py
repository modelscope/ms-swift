"""
模块功能概述：
本模块实现 LLM 推理主流程：
- SwiftInfer: 推理管道，负责初始化推理引擎、数据集推理、CLI 交互推理、指标计算与日志输出；
- infer_main: 便捷入口函数，构造管道并执行主流程。
"""

# 版权信息：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.
# 导入类型注解：Any/Dict/List/Optional/Union 用于函数签名与可读性
from typing import Any, Dict, List, Optional, Union

# 导入随机数库numpy：用于生成可复现实验的数据随机序列
import numpy as np
# 导入HF数据集类型：用于类型标注与数据接口
from datasets import Dataset as HfDataset
# 导入tqdm进度条：用于显示推理进度
from tqdm import tqdm

# 从swift.llm导入：推理参数与请求类型、基础管道、数据加载与模型模板准备、数据抽样工具
from swift.llm import InferArguments, InferRequest, SwiftPipeline, load_dataset, prepare_model_template, sample_dataset
# 从插件导入：推理统计、均值度量与ROUGE/BLEU计算工具
from swift.plugin import InferStats, MeanMetric, compute_rouge_bleu
# 从工具导入：Jsonl写入、分布式设置/判定、日志器、JSONL读取
from swift.utils import JsonlWriter, get_dist_setting, get_logger, is_dist, is_master, read_from_jsonl
# 导入推理引擎与适配器请求类型（PT推理）
from .infer_engine import AdapterRequest, PtEngine
# 导入请求配置数据类：控制生成参数与流式开关
from .protocol import RequestConfig
# 导入CLI交互状态管理器
from .utils import InferCliState

# 初始化模块级日志器
logger = get_logger()


# 定义推理管道：封装推理引擎初始化、数据集与CLI推理、指标计算
class SwiftInfer(SwiftPipeline):
    """
    类功能：
        构建并管理推理引擎，提供数据集批量推理与交互式推理（CLI），支持多后端（pt/vllm/sglang/lmdeploy），
        并在需要时计算指标与输出结果到JSONL文件。

    关键属性：
        args_class: 推理参数类类型，固定为 InferArguments。
        args: 解析后的推理参数实例。
        infer_engine: 具体推理后端实例（PtEngine/VllmEngine/SglangEngine/LmdeployEngine）。
        template: 模板对象，负责prompt组织与encode/decode。
        infer_kwargs: 传递给infer方法的额外关键字参数（如metrics、adapter_request）。
        random_state: 数据抽样的随机状态，确保可复现实验。
    """

    # 指定该管道的参数类类型
    args_class = InferArguments
    # 类型标注：实例属性 args 为上述类型
    args: args_class

    # 构造函数：初始化推理引擎、模板与辅助参数
    def __init__(self, args: Optional[Union[List[str], InferArguments]] = None) -> None:
        """
        函数功能：
            初始化推理管道：按需合并LoRA、构建推理引擎与模板、设置适配器请求与随机状态。

        入参：
            args (Optional[Union[List[str], InferArguments]]): 参数列表或参数对象。
        """
        # 延迟导入merge_lora：仅在需要时合并LoRA到基座模型
        from swift.llm import merge_lora
        # 调用基类构造（解析参数、基础初始化）
        super().__init__(args)
        # 便捷引用解析后的参数对象
        args = self.args
        # 若要求在推理前合并LoRA，则在CPU上执行合并，避免显存占用
        if args.merge_lora:
            merge_lora(args, device_map='cpu')
        # 初始化传递给infer的关键字参数容器
        self.infer_kwargs = {}
        # 若使用vllm后端，且有adapter目录，则传入adapter_request以支持热加载LoRA
        if args.infer_backend == 'vllm' and args.adapters:
            self.infer_kwargs['adapter_request'] = AdapterRequest('_lora', args.adapters[0])

        # 构建推理引擎：pt后端直接从模型与模板构造PtEngine；否则根据后端类型构造对应引擎
        if args.infer_backend == 'pt':
            # 准备模型与模板（加载权重与tokenizer/processor等）
            model, self.template = prepare_model_template(args)
            # 使用模板构建PtEngine，并设置最大批大小
            self.infer_engine = PtEngine.from_model_template(model, self.template, max_batch_size=args.max_batch_size)
            # 打印底层模型对象，便于确认加载正确
            logger.info(f'model: {self.infer_engine.model}')
        else:
            # 非pt后端：优先从参数获取模板（可能基于别名指定）
            self.template = args.get_template(None)
            # 按后端类型构建推理引擎实例（内部会组合kwargs）
            self.infer_engine = self.get_infer_engine(args, self.template)
        # 初始化随机状态，用于数据抽样与打乱
        self.random_state = np.random.RandomState(args.data_seed)

    # 兜底属性访问：若当前对象无该属性，则从infer_engine中透传
    def __getattr__(self, key: str):
        """
        函数功能：
            当在SwiftInfer实例未找到属性时，尝试从infer_engine对象上获取（实现委托）。
        """
        try:
            # 尝试从父类获取（包括SwiftPipeline）
            return super().__getattr__(key)
        except AttributeError:
            # 若已经初始化了infer_engine，则从其上获取对应属性
            if 'infer_engine' in self.__dict__:
                return getattr(self.infer_engine, key)
            # 否则按原逻辑抛出异常
            raise

    # 静态方法：根据参数构建具体推理引擎实例
    @staticmethod
    def get_infer_engine(args: InferArguments, template=None, **kwargs):
        """
        函数功能：
            依据infer_backend类型返回对应的推理引擎实例，并注入所需kwargs（模型ID、dtype、模板等）。

        入参：
            args (InferArguments): 推理参数对象。
            template: 模板对象，可为None。
            **kwargs: 额外注入参数（可覆盖infer_backend）。

        返回值：
            Any: 具体后端的推理引擎实例。
        """
        # 注入通用构造参数
        kwargs.update({
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
            'template': template,
        })
        # 解析后端类型：kwargs优先，其次args中的infer_backend
        infer_backend = kwargs.pop('infer_backend', None) or args.infer_backend
        # pt后端：使用PtEngine并注入模型构造相关参数
        if infer_backend == 'pt':
            from .infer_engine import PtEngine
            infer_engine_cls = PtEngine
            kwargs.update(args.get_model_kwargs())
            if hasattr(args, 'max_batch_size'):
                kwargs.update({'max_batch_size': args.max_batch_size})
        # vllm后端：注入vllm特有参数与随机种子（分布式场景确保不同dp进程seed不同）
        elif infer_backend == 'vllm':
            from .infer_engine import VllmEngine
            infer_engine_cls = VllmEngine
            kwargs.update(args.get_vllm_engine_kwargs())
            seed = args.seed
            if is_dist():
                # 确保不同数据并行进程拥有不同随机种子（按global_rank与tp size区分）
                seed += get_dist_setting()[0] // args.vllm_tensor_parallel_size
                kwargs['distributed_executor_backend'] = 'external_launcher'
            kwargs['seed'] = seed
        # sglang后端
        elif infer_backend == 'sglang':
            from .infer_engine import SglangEngine
            infer_engine_cls = SglangEngine
            kwargs.update(args.get_sglang_engine_kwargs())
        # 默认：lmdeploy后端
        else:
            from .infer_engine import LmdeployEngine
            infer_engine_cls = LmdeployEngine
            kwargs.update(args.get_lmdeploy_engine_kwargs())
        # 实例化并返回
        return infer_engine_cls(**kwargs)

    # 主流程：选择数据集或CLI推理，并写入结果
    def run(self) -> List[Dict[str, Any]]:
        """
        函数功能：
            执行推理主流程：
            - 若开启eval_human则进入交互式推理；否则运行数据集推理；
            - 若设置了result_path，则初始化JSONL写入器并最终提示保存位置。

        返回值：
            List[Dict[str, Any]]: 推理结果列表（每个元素包含response、messages等信息）。
        """
        # 便捷引用参数
        args = self.args
        # 如指定结果输出路径，初始化JSONL写入器
        self.jsonl_writer = JsonlWriter(args.result_path) if args.result_path else None
        # 选择推理模式：交互或数据集
        if args.eval_human:
            result = self.infer_cli()
        else:
            result = self.infer_dataset()
        # 若指定结果路径，提示保存位置
        if args.result_path:
            logger.info(f'The inference results have been saved to result_path: `{args.result_path}`.')
        # 返回推理结果列表
        return result

    # 解析响应对象，提取可读内容或embedding摘要
    @staticmethod
    def parse_data_from_response(response):
        """
        函数功能：
            将不同类型的响应对象统一解析为可展示的字符串：
            - 对于ChatCompletion样式，返回第一条choice的message.content
            - 对于Embedding样式，返回shape与截断的样本片段
        """
        # 若是聊天补全响应，返回第一条消息内容
        if hasattr(response, 'choices'):
            return response.choices[0].message.content
        # 若是embedding响应，组装摘要字符串
        elif hasattr(response, 'data'):
            emb = response.data[0].embedding
            shape = len(emb)
            sample = str(emb)
            if len(emb) > 6:
                # NOTE: 拼接成压缩预览字符串，显示前 3 个和后 3 个元素，中间以 ..., 代替省略的部分，eg
                # [a, b, c, ..., -x, y, z]
                sample = str(emb[:3])[:-1] + ', ..., ' + str(emb[-3:])[1:]
            return f'Embedding(shape: [1, {shape}]): {sample}'

    # 单条样本推理：支持流式打印与非流式一次性输出
    def infer_single(self, infer_request: Union[InferRequest, Dict[str, Any]], request_config: RequestConfig) -> str:
        """
        函数功能：
            对单条样本进行推理：
            - 流式模式下边接收增量边打印，并拼接为完整字符串返回；
            - 非流式模式直接解析最终响应并打印。

        入参：
            infer_request (Union[InferRequest, Dict[str, Any]]): 单条推理请求或其字典表示。
            request_config (RequestConfig): 生成配置与流式开关。

        返回值：
            str: 模型响应文本或摘要。
        """
        # 调用批量infer但仅传入1条样本，取返回列表的第0项
        res_or_gen = self.infer([infer_request],
                                request_config,
                                template=self.template,
                                use_tqdm=False,
                                **self.infer_kwargs)[0]
        # 若为流式模式：逐增量读取并打印，同时拼接为完整文本
        if request_config and request_config.stream:
            response = ''
            for res in res_or_gen:
                delta = res.choices[0].delta.content
                print(delta, end='', flush=True)
                response += delta
            print()
        else:
            # 非流式模式：直接解析最终响应
            response = self.parse_data_from_response(res_or_gen)
            print(response)
        # 分隔线便于阅读
        print('-' * 50)
        # 返回该样本的响应文本
        return response

    # 交互式推理（命令行）：支持多轮/单轮、输入多模态数据、按需记录JSONL
    def infer_cli(self) -> List[Dict[str, Any]]:
        """
        函数功能：
            提供命令行交互推理：支持多轮对话开关、输入多模态、奖励模型特殊流程，
            将每轮对话结果（含messages）追加保存到JSONL（若设置）。

        返回值：
            List[Dict[str, Any]]: 交互推理的对话记录列表。
        """
        # 便捷引用参数与模板
        args = self.args
        template = self.template
        # 构造请求默认配置并输出
        request_config = args.get_request_config()
        logger.info(f'request_config: {request_config}')

        # 输出使用帮助提示
        logger.info('Input `exit` or `quit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        logger.info('Input `reset-system` to reset the system and clear the history.')
        # 模板是否支持多轮
        support_multi_round = template.template_meta.support_multi_round
        if support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info('The current template only supports single-round dialogues.')

        # 初始化CLI状态与结果列表
        infer_state = InferCliState()
        result_list = []
        # 交互主循环
        while True:
            # 单轮模板下，每轮前清空历史
            if not support_multi_round:
                infer_state.clear()
            # 读取用户输入
            query = infer_state.input_text()
            # 退出指令
            if query.strip().lower() in {'exit', 'quit'}:
                break
            # 文本清洗与校验
            query = infer_state.check_query(query)
            if query is None:
                continue
            # 记录本轮query
            infer_state.add_query(query)
            # 若多模态模型，读取额外输入（如图像）
            if args.model_meta.is_multimodal:
                infer_state.input_mm_data()
            # 奖励模型或PRM任务：读取人工response作为标签，随后推理
            if args.model_meta.is_reward or args.task_type == 'prm':
                # reward model
                response = infer_state.input_text()
                infer_state.add_response(response)
                data = infer_state.to_dict()
                response = self.infer_single(data, request_config)
                data = {'response': response, **data}
            else:
                # 常规对话：推理得到response并添加到messages中
                data = infer_state.to_dict()
                response = self.infer_single(data, request_config)
                infer_state.add_response(response)
                data['messages'].append({'role': 'assistant', 'content': response})
                data = {'response': response, **data}
            # 记录本轮数据
            result_list.append(data)
            if self.jsonl_writer:
                self.jsonl_writer.append(data)

        # 返回整个会话的记录
        return result_list

    # 准备验证集：按照参数选择数据源并做可选抽样
    def _prepare_val_dataset(self) -> HfDataset:
        """
        函数功能：
            准备评测/验证数据集：
            - 优先使用val_dataset参数；否则使用dataset按比例切分
            - 根据val_dataset_sample与随机状态进行抽样

        返回值：
            HfDataset: 准备好的验证集数据。
        """
        # 便捷引用参数
        args = self.args
        # 获取数据集构造所需的kwargs（tokenizer/模板相关）
        dataset_kwargs = args.get_dataset_kwargs()
        # 若指定了独立的验证数据集路径，则仅读取验证集
        if len(args.val_dataset) > 0:
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
        else:
            # 否则从主数据集中切分验证部分
            _, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                **dataset_kwargs)
        # 基本校验：必须成功获得验证集
        assert val_dataset is not None
        # 对验证集进行可选抽样与打乱（随机状态来自构造时设置）
        val_dataset = sample_dataset(val_dataset, args.val_dataset_sample, args.dataset_shuffle, self.random_state)
        # 返回构造好的验证集
        return val_dataset

    # 计算指标：仅主进程执行，读取JSONL并计算acc或ROUGE
    def _calc_metric(self):
        """
        函数功能：
            读取写入的JSONL结果文件，计算指定的指标（acc或rouge），仅在主进程打印。
        """
        # 便捷引用参数
        args = self.args
        # 非主进程直接返回，避免重复计算
        if not is_master():
            return
        # 读取所有样本的记录
        data_list = read_from_jsonl(self.jsonl_writer.fpath)
        preds, labels = [], []
        # 提取预测与标签
        for data in data_list:
            preds.append(data['response'])
            labels.append(data['labels'])
        # 精度指标：逐条比较是否相等，更新均值度量
        if args.metric == 'acc':
            mean_metric = MeanMetric()
            for pred, label in zip(preds, labels):
                mean_metric.update(pred == label)
            res = {'acc': mean_metric.compute()['value']}
        # ROUGE/BLEU：调用工具计算
        elif args.metric == 'rouge':
            res = compute_rouge_bleu(preds, labels)
        # 打印指标结果
        logger.info(res)

    # 数据集推理：支持流式与批量写入模式，按需记录结果与指标
    def infer_dataset(self) -> List[Dict[str, Any]]:
        """
        函数功能：
            对验证集执行推理：
            - 流式模式：逐条推理并打印query/labels/response，实时写JSONL；
            - 非流式模式：按write_batch_size分片批量推理与写入，显示进度；
            - 可选计算指标并输出。

        返回值：
            List[Dict[str, Any]]: 推理结果列表。
        """
        # 便捷引用参数与请求默认配置
        args = self.args
        request_config = args.get_request_config()
        logger.info(f'request_config: {request_config}')

        # 准备验证集并打印信息
        val_dataset = self._prepare_val_dataset()
        logger.info(f'val_dataset: {val_dataset}')

        # 将统计器放入infer_kwargs，供底层记录吞吐/延迟
        self.infer_kwargs['metrics'] = [InferStats()]
        # 流式模式：逐条推理并交互打印
        if request_config and request_config.stream:
            result_list = []
            for data in val_dataset:
                # 从messages中移除最后一条assistant作为labels（因causal LM任务）
                labels = InferRequest.remove_response(data['messages'])
                # 打印查询内容（最后一条user）
                query = data['messages'][-1]['content']
                print(f'[QUERY] {query}')
                if labels:
                    print(f'[LABELS] {labels}')
                print('[RESPONSE] ', end='')
                # 执行单条推理并打印流式结果
                response = self.infer_single(data, request_config)
                # 将response写回messages
                data['messages'].append({'role': 'assistant', 'content': response})
                # 汇总该条样本的输出字典
                data = {'response': response, 'labels': labels, **data}
                result_list.append(data)
                # 如设置了JSONL写入，则记录该条
                if self.jsonl_writer:
                    self.jsonl_writer.append(data)
            # 从infer_kwargs移除metrics并打印统计
            metrics = self.infer_kwargs.pop('metrics')
            print(metrics[0].compute())
        else:
            # 写入批大小<=0：默认一次性写出全部
            if args.write_batch_size <= 0:
                args.write_batch_size = len(val_dataset)
            # 若分批写且指定了result_path，则打印路径提醒
            if args.write_batch_size < len(val_dataset) and args.result_path:
                logger.info(f'args.result_path: {args.result_path}')
            # 初始化进度条：当一次性写出全部时关闭进度条
            prog_bar = tqdm(
                total=len(val_dataset), dynamic_ncols=True, disable=args.write_batch_size >= len(val_dataset))
            result_list = []
            idx = 0
            # 逐批分片推理
            while idx < len(val_dataset):
                shard_size = min(args.write_batch_size, len(val_dataset) - idx)
                shard_dataset = val_dataset.select(range(idx, idx + shard_size))
                result_list += self._batch_infer(shard_dataset, request_config)
                idx += shard_size
                prog_bar.update(shard_size)
            # 关闭进度条
            prog_bar.close()
            # 打印汇总统计
            metrics = self.infer_kwargs.pop('metrics')
            if result_list:
                metric = metrics[0].compute()
                print(f'[rank{args.rank}] {metric}' if args.rank >= 0 else str(metric))
        # 如设置了metric选项，则计算并打印指标
        if args.metric is not None:
            self._calc_metric()
        # 返回结果列表
        return result_list

    # 批量推理：处理分布式切分、标签抽取与结果写入
    def _batch_infer(self, val_dataset, request_config):
        """
        函数功能：
            对一批样本执行推理：
            - 在分布式场景下按rank进行数据分片；
            - 抽取标签或移除assistant响应；
            - 调用底层infer并将响应写回messages，按需写JSONL。

        返回值：
            List[Dict[str, Any]]: 该批次推理结果列表。
        """
        # 便捷引用参数与结果容器
        args = self.args
        result_list = []
        # vllm后端：rank需要除以tp size得到data parallel rank
        if args.infer_backend == 'vllm':
            rank = args.rank // args.vllm_tensor_parallel_size if args.rank >= 0 else -1
            data_parallel_size = args.global_world_size // args.vllm_tensor_parallel_size
        else:
            # 其他后端：直接使用rank与全局并行度
            rank, data_parallel_size = args.rank, args.global_world_size

        # 若开启分布式且有多份数据并行，则按rank做连续切片
        if rank >= 0 and data_parallel_size > 1:
            val_dataset = val_dataset.shard(data_parallel_size, rank, contiguous=True)
        # 将数据集转化为列表以便多次遍历
        val_dataset = list(val_dataset)
        # 抽取每条样本的标签列表
        labels_list = []
        for data in val_dataset:
            if args.task_type == 'causal_lm':
                labels = InferRequest.remove_response(data['messages'])
            else:
                labels = data.pop('label', None)
            labels_list.append(labels)

        # 执行批量推理（底层按infer_backend实现），并显示tqdm
        resp_list = self.infer(val_dataset, request_config, template=self.template, use_tqdm=True, **self.infer_kwargs)
        # 在vllm下，只有tp组内rank为0的dp进程负责回写与收集结果
        if not (args.infer_backend == 'vllm' and rank >= 0 and args.rank % args.vllm_tensor_parallel_size != 0):
            for data, resp, labels in zip(val_dataset, resp_list, labels_list):
                response = resp.choices[0].message.content
                data['messages'].append({'role': 'assistant', 'content': response})
                data = {'response': response, 'labels': labels, 'logprobs': resp.choices[0].logprobs, **data}
                result_list.append(data)
        # 如设置了JSONL写入，则批量写入并在主进程聚合对象
        if self.jsonl_writer:
            self.jsonl_writer.append(result_list, gather_obj=True)
        # 返回该批次的结果
        return result_list


# 推理命令入口：构造并执行主流程
def infer_main(args: Optional[Union[List[str], InferArguments]] = None):
    """
    函数功能：
        便捷入口。构造SwiftInfer并执行其main流程，返回推理结果列表。
    """
    # 创建推理管道并执行
    return SwiftInfer(args).main()
