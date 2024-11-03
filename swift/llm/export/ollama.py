# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from typing import List
from swift.llm import Template, ExportArguments
from swift.utils import get_logger
from .utils import prepare_model_template
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
    tokenizer = template.tokenizer
    logger.info(f'Using model_dir: {model.model_dir}')
    template_meta = template.template_meta
    with open(os.path.join(args.output_dir, 'Modelfile'), 'w') as f:
        f.write(f'FROM {model.model_dir}\n')
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

        stop_words = template_meta.stop_words + [template_meta.suffix[-1], tokenizer.eos_token]
        for stop_word in args.stop_words:
            if isinstance(stop_word, list):
                stop_word = template.tokenizer.decode(stop_word)
            f.write(f'PARAMETER stop "{stop_word}"\n')
        if args.temperature:
            f.write(f'PARAMETER temperature {args.temperature}\n')
        if args.top_k:
            f.write(f'PARAMETER top_k {args.top_k}\n')
        if args.top_p:
            f.write(f'PARAMETER top_p {args.top_p}\n')
        if args.repetition_penalty:
            f.write(f'PARAMETER repeat_penalty {args.repetition_penalty}\n')

    logger.info('Save Modelfile done, you can start ollama by:')
    logger.info('> ollama serve')
    logger.info('In another terminal:')
    logger.info('> ollama create my-custom-model ' f'-f {os.path.join(args.output_dir, "Modelfile")}')
    logger.info('> ollama run my-custom-model')


