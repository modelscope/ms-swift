"""
模块功能概述：
本模块用于将当前模型与模板导出为 Ollama 所需的 Modelfile 文件：
- 通过模板（Template）生成聊天模板字符串，并填充到 Modelfile 的 TEMPLATE 段；
- 写入必要的 inference 参数（stop、temperature、top_k、top_p、repeat_penalty 等）；
- 提供命令行提示，方便用户使用 ollama 创建与运行自定义模型。
"""

# 版权声明：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.
# 导入os模块：用于路径拼接与目录创建
import os
# 导入类型注解：List 用于函数参数类型标注
from typing import List

# 从swift.llm导入导出相关组件：
# - ExportArguments: 导出参数定义
# - PtEngine: 推理引擎，用于准备生成配置和stop词
# - RequestConfig: 推理请求配置数据类
# - Template: 模板类型注解
# - prepare_model_template: 按参数准备模型与模板
from swift.llm import ExportArguments, PtEngine, RequestConfig, Template, prepare_model_template
# 导入日志工具：统一打印日志
from swift.utils import get_logger

# 初始化模块级日志器：用于记录导出过程信息
logger = get_logger()


# 辅助函数：根据模板元信息，替换占位符并拼接为字符串
def replace_and_concat(template: 'Template', template_list: List, placeholder: str, keyword: str):
    """
    函数功能：
        遍历模板片段列表，将字符串片段中的占位符替换为给定关键字；
        对于token id序列使用tokenizer解码；对于特殊token名称（如bos/eos）转为对应字符。

    入参：
        template (Template): 模板对象，包含tokenizer与模板元信息。
        template_list (List): 模板片段列表，元素可为str、list/tuple（token ids 或 特殊token名）。
        placeholder (str): 需要被替换的占位符（如"{SYSTEM}"、"{QUERY}"）。
        keyword (str): 用于替换占位符的目标关键字（如"{{ .System }}"、"{{ .Prompt }}"）。

    返回值：
        str: 替换与拼接后的完整字符串。

    示例：
        >>> replace_and_concat(template, ["Hello {SYSTEM}"], "{SYSTEM}", "{{ .System }}")
    """
    # 初始化最终字符串缓冲区
    final_str = ''
    # 遍历模板片段，逐一处理并拼接
    for t in template_list:
        # 若片段为字符串，直接替换占位符并拼接
        if isinstance(t, str):
            final_str += t.replace(placeholder, keyword)
        # 若片段为元组或列表，则可能是token ids或特殊token名称集合
        elif isinstance(t, (tuple, list)):
            # 若第一个元素为int，视为token ids序列，直接用tokenizer解码
            if isinstance(t[0], int):
                final_str += template.tokenizer.decode(t)
            else:
                # 否则遍历特殊token名称并追加对应字符
                for attr in t:
                    if attr == 'bos_token_id':
                        final_str += template.tokenizer.bos_token
                    elif attr == 'eos_token_id':
                        final_str += template.tokenizer.eos_token
                    else:
                        # 若遇到未知的特殊标记，抛出错误以提示配置问题
                        raise ValueError(f'Unknown token: {attr}')
    # 返回拼接后的最终字符串
    return final_str


# 主函数：将当前模型与模板导出为 Ollama 的 Modelfile
def export_to_ollama(args: ExportArguments):
    """
    函数功能：
        基于给定导出参数，准备模型与模板，并在输出目录生成 Ollama 的 Modelfile 文件。
        文件包含 FROM 基座模型路径、TEMPLATE 模板与推理参数（stop/temperature/top_k/top_p/repeat_penalty）。

    入参：
        args (ExportArguments): 导出参数对象，需包含 output_dir 等必要信息。

    返回值：
        None

    示例：
        >>> export_to_ollama(ExportArguments(output_dir='./ollama'))
    """
    # 设置device_map为'meta'以加速加载（部分框架在meta设备上跳过权重分配）
    args.device_map = 'meta'  # Accelerate load speed.
    # 记录导出开始日志
    logger.info('Exporting to ollama:')
    # 确保输出目录存在；exist_ok=True 表示目录已存在时不报错
    os.makedirs(args.output_dir, exist_ok=True)
    # 根据导出参数准备模型与模板，供后续生成配置与模板字符串
    model, template = prepare_model_template(args)
    # 基于模型与模板构建推理引擎，用于准备生成配置与stop词
    pt_engine = PtEngine.from_model_template(model, template)
    # 打印底层使用的模型目录（FROM字段将引用该路径）
    logger.info(f'Using model_dir: {pt_engine.model_dir}')
    # 获取模板元信息（包含prefix/prompt/suffix等片段）
    template_meta = template.template_meta
    # 在输出目录创建并写入 Modelfile 文件，UTF-8 编码
    with open(os.path.join(args.output_dir, 'Modelfile'), 'w', encoding='utf-8') as f:
        # 指定基座模型目录（FROM 行）
        f.write(f'FROM {pt_engine.model_dir}\n')
        # NOTE: Go/text/template 或 Helm 等模板引擎的条件语法
        # 写入 TEMPLATE 段的前半部分：根据是否有 .System 动态拼接 system/prefix 片段
        f.write(f'TEMPLATE """{{{{ if .System }}}}'
                f'{replace_and_concat(template, template_meta.system_prefix, "{{SYSTEM}}", "{{ .System }}")}'
                f'{{{{ else }}}}{replace_and_concat(template, template_meta.prefix, "", "")}'
                f'{{{{ end }}}}')
        # 若存在 .Prompt 则拼接 prompt 片段
        f.write(f'{{{{ if .Prompt }}}}'
                f'{replace_and_concat(template, template_meta.prompt, "{{QUERY}}", "{{ .Prompt }}")}'
                f'{{{{ end }}}}')
        # 预留响应占位（Ollama 会用模型输出替换该位置）
        f.write('{{ .Response }}')
        # 拼接模板后缀并结束多行字符串
        f.write(replace_and_concat(template, template_meta.suffix, '', '') + '"""\n')
        # 将模板后缀作为默认的stop词写入（可防止继续生成）
        f.write(f'PARAMETER stop "{replace_and_concat(template, template_meta.suffix, "", "")}"\n')

        # 构建请求配置：从args读取常用生成参数
        request_config = RequestConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty)
        # 依据引擎内部逻辑准备最终生成配置（可能添加默认项或进行校验）
        generation_config = pt_engine._prepare_generation_config(request_config)
        # 由引擎根据模板元信息与请求配置补充stop词列表
        pt_engine._add_stop_words(generation_config, request_config, template.template_meta)
        # 将所有stop词写入 Modelfile
        for stop_word in generation_config.stop_words:
            f.write(f'PARAMETER stop "{stop_word}"\n')
        # 写入其余生成参数配置项
        f.write(f'PARAMETER temperature {generation_config.temperature}\n')
        f.write(f'PARAMETER top_k {generation_config.top_k}\n')
        f.write(f'PARAMETER top_p {generation_config.top_p}\n')
        f.write(f'PARAMETER repeat_penalty {generation_config.repetition_penalty}\n')

    # 输出后续使用Ollama的命令行提示
    logger.info('Save Modelfile done, you can start ollama by:')
    logger.info('> ollama serve')
    logger.info('In another terminal:')
    logger.info('> ollama create my-custom-model ' f'-f {os.path.join(args.output_dir, "Modelfile")}')
    logger.info('> ollama run my-custom-model')
