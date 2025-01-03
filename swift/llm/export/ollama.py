# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List

from swift.llm import ExportArguments, PtEngine, RequestConfig, Template, prepare_model_template
from swift.utils import get_logger

logger = get_logger()


def replace_and_concat(template: 'Template', template_list: List, placeholder: str, keyword: str):
    final_str = ''
    for t in template_list:
        if isinstance(t, str):
            final_str += t.replace(placeholder, keyword)
        elif isinstance(t, (tuple, list)):
            if isinstance(t[0], int):
                final_str += template.tokenizer.decode(t)
            else:
                for attr in t:
                    if attr == 'bos_token_id':
                        final_str += template.tokenizer.bos_token
                    elif attr == 'eos_token_id':
                        final_str += template.tokenizer.eos_token
                    else:
                        raise ValueError(f'Unknown token: {attr}')
    return final_str


def export_to_ollama(args: ExportArguments):
    logger.info('Exporting to ollama:')
    logger.info('If you have a gguf file, try to pass the file by :--gguf_file /xxx/xxx.gguf, '
                'else SWIFT will use the original(merged) model dir')
    os.makedirs(args.output_dir, exist_ok=True)
    model, template = prepare_model_template(args)
    pt_engine = PtEngine.from_model_template(model, template)
    logger.info(f'Using model_dir: {pt_engine.model_dir}')
    template_meta = template.template_meta
    with open(os.path.join(args.output_dir, 'Modelfile'), 'w', encoding='utf-8') as f:
        f.write(f'FROM {pt_engine.model_dir}\n')
        f.write(f'TEMPLATE """{{{{ if .System }}}}'
                f'{replace_and_concat(template, template_meta.system_prefix, "{{SYSTEM}}", "{{ .System }}")}'
                f'{{{{ else }}}}{replace_and_concat(template, template_meta.prefix, "", "")}'
                f'{{{{ end }}}}')
        f.write(f'{{{{ if .Prompt }}}}'
                f'{replace_and_concat(template, template_meta.prompt, "{{QUERY}}", "{{ .Prompt }}")}'
                f'{{{{ end }}}}')
        f.write('{{ .Response }}')
        f.write(replace_and_concat(template, template_meta.suffix, '', '') + '"""\n')
        f.write(f'PARAMETER stop "{replace_and_concat(template, template_meta.suffix, "", "")}"\n')

        request_config = RequestConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty)
        generation_config = pt_engine._prepare_generation_config(request_config)
        pt_engine._add_stop_words(generation_config, request_config, template.template_meta)
        for stop_word in generation_config.stop_words:
            f.write(f'PARAMETER stop "{stop_word}"\n')
        f.write(f'PARAMETER temperature {generation_config.temperature}\n')
        f.write(f'PARAMETER top_k {generation_config.top_k}\n')
        f.write(f'PARAMETER top_p {generation_config.top_p}\n')
        f.write(f'PARAMETER repeat_penalty {generation_config.repetition_penalty}\n')

    logger.info('Save Modelfile done, you can start ollama by:')
    logger.info('> ollama serve')
    logger.info('In another terminal:')
    logger.info('> ollama create my-custom-model ' f'-f {os.path.join(args.output_dir, "Modelfile")}')
    logger.info('> ollama run my-custom-model')
