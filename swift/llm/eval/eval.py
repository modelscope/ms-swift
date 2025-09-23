"""
模块功能概述：
本模块封装了评测流程的管道类与入口函数：
- SwiftEval: 负责根据不同评测后端（Native/OpenCompass/VLM-Eval-Kit）组装任务配置、
  启动/复用部署服务、运行评测并汇总输出报告。
- eval_main: 提供命令行/编程入口，构造管道并执行主流程。
"""

# 版权声明：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入os模块：文件路径与符号链接等操作
import os
# 导入nullcontext：当无需实际上下文管理时提供空上下文
from contextlib import nullcontext
# 导入类型注解：用于标注集合、可选类型与联合类型
from typing import List, Optional, Union

# 从evalscope引入常量：评测后端与评测类型枚举
from evalscope.constants import EvalBackend, EvalType
# 从evalscope引入任务配置与运行入口：构造任务、执行评测
from evalscope.run import TaskConfig, run_task
# 从evalscope引入汇总器：根据任务配置解析评测报告
from evalscope.summarizer import Summarizer

# 从swift工具模块引入：追加写jsonl文件与日志器
from swift.utils import append_to_jsonl, get_logger
# 从上级包引入资源下载工具：用于拉取OpenCompass数据集
from .. import MediaResource
# 引入评测参数定义类：解析与承载评测配置
from ..argument import EvalArguments
# 引入基础管道类：评测管道继承自该基类
from ..base import SwiftPipeline
# 引入部署函数：用于在本地启动服务并返回URL
from ..infer import run_deploy

# 初始化模块级日志器：统一打印评测相关日志
logger = get_logger()


# 定义评测管道类：封装从部署、配置到执行与汇总的完整流程
class SwiftEval(SwiftPipeline):
    """
    类功能：
        评测主流程管道。根据传入的参数决定是否本地部署服务，组装不同评测后端的任务配置，
        调用评测并汇总结果，同时输出元信息（时间、模型、适配器、输出路径等）。

    关键属性：
        args_class: 参数类，固定为EvalArguments。
        args: 解析后的评测参数实例。
    """

    # 指定该管道使用的参数类类型
    args_class = EvalArguments
    # 类型标注：实例属性args为上述参数类
    args: args_class

    # 运行主流程：部署（或复用URL）→ 组装任务配置 → 运行评测 → 汇总输出
    def run(self):
        """
        函数功能：
            执行评测主流程：选择服务URL、组装任务配置、运行评测并生成汇总报告；
            可选将结果附加写入jsonl文件。

        入参：
            无（使用self.args内的评测参数）。

        返回值：
            dict: 评测结果字典（含各后端结果与元信息）。

        示例：
            >>> pipeline = SwiftEval(EvalArguments(...))
            >>> report = pipeline.run()
        """
        # 读取已解析的评测参数，便于后续使用
        args = self.args
        # 初始化评测报告字典，用于聚合不同后端的结果与元信息
        eval_report = {}
        # 根据是否已提供eval_url决定是否需要本地部署服务；若有URL则使用空上下文
        deploy_context = nullcontext() if args.eval_url else run_deploy(args, return_url=True)
        # 进入上下文，拿到服务基础URL（本地部署或外部提供）
        with deploy_context as base_url:
            # 若外部提供URL优先使用，否则使用部署返回的URL
            base_url = args.eval_url or base_url
            # 组装符合OpenAI风格接口的聊天补全路径
            url = f"{base_url.rstrip('/')}/chat/completions"

            # 基于数据集、后端与URL生成对应的任务配置
            task_cfg = self.get_task_cfg(args.eval_dataset, args.eval_backend, url)
            # 执行评测并取回结果（已做后端适配的汇总）
            result = self.get_task_result(task_cfg)
            # 将本后端的结果写入总报告，以后端枚举值为键
            eval_report[args.eval_backend] = result

        # 合并评测元信息：时间、模型、适配器、结果路径、输出目录、评测上限等
        eval_report.update({
            'time': args.time,                 # 评测触发时间或时间戳
            'model': args.model,               # 模型名称或路径
            'adapters': args.adapters,         # 使用的适配器列表
            'result_path': args.result_path,   # 结果文件路径（若有）
            'eval_output_dir': args.eval_output_dir,  # 评测输出目录
            'eval_limit': args.eval_limit      # 每个数据集的样本上限
        })

        # 若指定了追加写入的jsonl路径，则将结果写入并提示保存位置
        if args.result_jsonl:
            append_to_jsonl(args.result_jsonl, eval_report)
            logger.info(f'The eval result have been saved to result_jsonl: `{args.result_jsonl}`.')
        # 返回最终评测报告
        return eval_report

    # 执行评测并按后端规范汇总为统一结果格式
    def get_task_result(self, task_cfg: TaskConfig):
        """
        函数功能：
            调用评测执行入口，随后使用Summarizer按后端协议解析原始报告，
            并规整为统一结果字典返回。

        入参：
            task_cfg (TaskConfig): 已构建好的评测任务配置。

        返回值：
            dict: 规整后的评测结果。

        示例：
            >>> result = self.get_task_result(task_cfg)
        """
        # 运行评测任务（内部可能调度多进程/多线程与请求）
        run_task(task_cfg=task_cfg)
        # 从配置对象解析产生的评测报告（可能为列表或字典，取决于后端）
        reports = Summarizer.get_report_from_cfg(task_cfg=task_cfg)
        # 初始化统一结果容器
        result = {}
        # OpenCompass后端：以数据集为键，以metric与得分为值，使用model_suffix列
        if task_cfg.eval_backend == EvalBackend.OPEN_COMPASS:
            for report in reports:
                if report[self.args.model_suffix] != '-':
                    result[report['dataset']] = {report['metric']: report[self.args.model_suffix]}
        # VLM-Eval-Kit后端：键为"prefix_dataset_metric"，需拆分后规整
        elif task_cfg.eval_backend == EvalBackend.VLM_EVAL_KIT:
            for report in reports:
                splited_key = next(iter(report)).rsplit('_', 2)
                if len(splited_key) == 3:
                    _, dataset, metric = splited_key
                else:
                    dataset, metric = '-', '-'
                result[dataset] = {metric: list(report.values())[0]}
        # 其他后端：直接返回原始报告结构
        else:
            result = reports
        # 返回统一结果
        return result

    # 构造任务配置：根据后端类型选择对应的配置生成方法
    def get_task_cfg(self, dataset: List[str], eval_backend: str, url: str):
        """
        函数功能：
            基于后端类型生成任务配置对象；处理OpenCompass所需的数据集准备与符号链接。

        入参：
            dataset (List[str]): 评测数据集名称列表。
            eval_backend (str): 评测后端类型。
            url (str): 已部署或外部提供的服务URL。

        返回值：
            TaskConfig: 对应后端的任务配置对象。

        示例：
            >>> cfg = self.get_task_cfg(["cmmlu"], EvalBackend.NATIVE, url)
        """
        # 基本校验：后端类型必须在支持列表内
        assert eval_backend in {EvalBackend.NATIVE, EvalBackend.OPEN_COMPASS, EvalBackend.VLM_EVAL_KIT}
        # OpenCompass分支：需要准备本地数据目录结构
        if eval_backend == EvalBackend.OPEN_COMPASS:
            if self.args.local_dataset:
                # 若当前工作目录存在data目录，则需确保不与OpenCompass数据冲突
                if os.path.exists('data'):
                    if not os.path.exists(os.path.join('data', 'CMB')):
                        # 已存在用户自定义data目录，且不包含CMB子目录，提示用户迁移避免冲突
                        raise RuntimeError('Opencompass need a `data` folder in your work dir('
                                           'which will be created automatically by swift eval), '
                                           'but a local path named `data` already exists, '
                                           'please consider moving the dir to another location.')
                else:
                    # 下载完整OpenCompass数据集，并创建软链接到工作目录的data
                    local_dir = MediaResource.download(
                        'https://modelscope.cn/datasets/'
                        'opencompass/OpenCompassDataComplete/'
                        'resolve/master/OpenCompassData-complete-20240207.zip', 'OpenCompassData')
                    os.symlink(os.path.join(local_dir, 'data'), 'data')

            # 生成OpenCompass后端任务配置
            task_cfg = self.get_opencompass_task_cfg(dataset, url)
        # VLM-Eval-Kit分支：生成对应配置
        elif eval_backend == EvalBackend.VLM_EVAL_KIT:
            task_cfg = self.get_vlmeval_task_cfg(dataset, url)
        # 默认分支（Native）：生成原生服务评测配置
        else:
            task_cfg = self.get_native_task_cfg(dataset, url)
        # 返回统一的任务配置对象
        return task_cfg

    # 构建原生服务评测配置：直接以SERVICE类型请求api_url
    def get_native_task_cfg(self, dataset: List[str], url: str):
        """
        函数功能：
            生成Native后端（SERVICE模式）的TaskConfig配置，用于通过API进行评测。

        入参：
            dataset (List[str]): 评测数据集名称列表。
            url (str): 服务URL，指向/chat/completions接口。

        返回值：
            TaskConfig: 可直接用于run_task的配置对象。

        示例：
            >>> cfg = self.get_native_task_cfg(["cmmlu"], url)
        """
        # 便捷引用参数对象
        args = self.args
        # 设置该后端的工作目录，便于产出结果组织
        work_dir = os.path.join(args.eval_output_dir, 'native')
        # 构造任务配置对象：指定模型后缀、服务类型、URL、密钥与数据集等
        return TaskConfig(
            model=args.model_suffix,                 # 评测时展示的模型标识（列名）
            eval_type=EvalType.SERVICE,              # 使用服务模式进行评测
            api_url=url,                              # 服务端点
            api_key=args.api_key or 'EMPTY',         # 若未提供API Key则占位
            datasets=dataset,                         # 评测数据集列表
            work_dir=work_dir,                       # 工作目录
            limit=args.eval_limit,                   # 每个数据集的上限条数
            eval_batch_size=args.eval_num_proc,      # 并发批大小（进程数）
            dataset_args=args.dataset_args,          # 透传数据集额外参数
            generation_config=args.eval_generation_config,  # 生成相关参数
            **args.extra_eval_args)                  # 额外评测参数透传

    # 构建OpenCompass评测配置
    def get_opencompass_task_cfg(self, dataset: List[str], url: str):
        """
        函数功能：
            生成OpenCompass后端的TaskConfig配置，包含datasets、models与并发等参数。

        入参：
            dataset (List[str]): 评测数据集名称列表。
            url (str): 服务URL。

        返回值：
            TaskConfig: 可直接用于OpenCompass运行的配置对象。

        示例：
            >>> cfg = self.get_opencompass_task_cfg(["ceval"], url)
        """
        # 便捷引用参数对象
        args = self.args
        # 设置该后端的工作目录
        work_dir = os.path.join(args.eval_output_dir, 'opencompass')
        # 构造任务配置对象：评测后端为OPEN_COMPASS，并提供eval_config结构
        return TaskConfig(
            eval_backend=EvalBackend.OPEN_COMPASS,
            eval_config={
                'datasets':
                dataset,                 # 待评测的数据集合
                'batch_size':
                args.eval_num_proc,      # 并发批大小
                'work_dir':
                work_dir,                # 输出工作目录
                'models': [{
                    'path': args.model_suffix,      # 显示的模型标识
                    'openai_api_base': url,         # 服务URL
                    'key': args.api_key or 'EMPTY', # API Key或占位
                    'is_chat': args.use_chat_template  # 是否走对话模板
                }],
                'limit':
                args.eval_limit           # 每数据集评测上限
            },
            work_dir=work_dir)

    # 构建VLM-Eval-Kit评测配置
    def get_vlmeval_task_cfg(self, dataset: List[str], url: str):
        """
        函数功能：
            生成VLM-Eval-Kit后端的TaskConfig配置，包含多模态模型定义与并发数等参数。

        入参：
            dataset (List[str]): 评测数据集名称列表。
            url (str): 服务URL。

        返回值：
            TaskConfig: 可直接用于VLM-Eval-Kit运行的配置对象。

        示例：
            >>> cfg = self.get_vlmeval_task_cfg(["mme"], url)
        """
        # 便捷引用参数对象
        args = self.args
        # 设置该后端的工作目录
        work_dir = os.path.join(args.eval_output_dir, 'vlmeval')
        # 构造任务配置对象：评测后端为VLM_EVAL_KIT，并提供eval_config结构
        return TaskConfig(
            eval_backend=EvalBackend.VLM_EVAL_KIT,
            eval_config={
                'data':
                dataset,                  # 数据集名称列表
                'model': [{
                    'type': args.model_suffix,      # 模型类型/标识
                    'name': 'CustomAPIModel',       # 统一的API模型名称
                    'api_base': url,                # 服务URL
                    'key': args.api_key or 'EMPTY', # API Key或占位
                    **args.eval_generation_config   # 生成参数展开
                }],
                'nproc':
                args.eval_num_proc,       # 并发进程数
                'limit':
                args.eval_limit           # 每数据集评测上限
            },
            work_dir=work_dir)


# 评测入口函数：支持传入参数对象或参数列表
def eval_main(args: Optional[Union[List[str], EvalArguments]] = None):
    """
    函数功能：
        评测命令入口。构造SwiftEval管道并执行主流程，返回最终评测结果。

    入参：
        args (Optional[Union[List[str], EvalArguments]]): None/参数列表/参数对象。

    返回值：
        dict: 评测结果字典。

    示例：
        >>> eval_main(["--model", "xxx", "--eval_backend", "native"]) 
    """
    # 创建评测管道并执行主流程，返回结果
    return SwiftEval(args).main()
