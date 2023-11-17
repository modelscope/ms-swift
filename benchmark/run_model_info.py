from typing import Dict, List, Tuple

from swift.llm import MODEL_MAPPING, ModelType


def get_model_name_list() -> List[str]:
    res = []
    for k in ModelType.__dict__.keys():
        if k.startswith('__'):
            continue
        res.append(ModelType.__dict__[k])
    return res


def format_markdown(model_info: Dict[Tuple, List[str]]) -> str:
    text_list = []
    md_format = '|{model_list}|{lora_target_modules}|{template}|{requires}|'
    for k, v in model_info.items():
        model_list = ', '.join(v)
        lora_target_modules = ', '.join(k[0])
        template = k[1]
        requires = ', '.join(k[2])
        if len(requires) == 0:
            requires = '-'
        text_list.append(
            md_format.format(
                model_list=model_list,
                lora_target_modules=lora_target_modules,
                template=template,
                requires=requires))
    return '\n'.join(text_list)


def write_model_info_table(fpath: str) -> None:
    model_name_list = get_model_name_list()
    with open(fpath, 'w') as f:
        f.write(
            """| Model List | Default Lora Target Modules | Default Template | Requires |
| ---------  | --------------------------- | ---------------- | -------- |
""")
    res = {}
    for model_name in model_name_list:
        model_info = MODEL_MAPPING[model_name]
        lora_target_modules = model_info['lora_target_modules']
        template = model_info['template']
        requires = model_info['requires']
        key = (tuple(lora_target_modules), template, tuple(requires))
        if key not in res:
            res[key] = []
        res[key].append(model_name)
    text = format_markdown(res)
    with open(fpath, 'a') as f:
        f.write(text)
    print()


if __name__ == '__main__':
    write_model_info_table('model_info.md')
