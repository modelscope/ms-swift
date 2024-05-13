from typing import List

from swift.llm import MODEL_MAPPING, ModelType


def get_model_info_table() -> List[str]:
    fpaths = ['docs/source/LLM/支持的模型和数据集.md', 'docs/source_en/LLM/Supported-models-datasets.md']
    end_words = ['## 数据集', '## Datasets']
    model_name_list = ModelType.get_model_name_list()
    result = ('| Model Type | Model ID | Default Lora Target Modules | Default Template |'
              ' Support Flash Attn | Support VLLM | Requires | Tags | HF Model ID |\n'
              '| ---------  | -------- | --------------------------- | ---------------- |'
              ' ------------------ | ------------ | -------- | ---- | ----------- |\n')
    res: List[str] = []
    bool_mapping = {True: '&#x2714;', False: '&#x2718;'}
    for model_name in model_name_list:
        model_info = MODEL_MAPPING[model_name]
        model_id = model_info['model_id_or_path']
        lora_target_modules = ', '.join(model_info['lora_target_modules'])
        template = model_info['template']
        support_flash_attn = model_info.get('support_flash_attn', False)
        support_flash_attn = bool_mapping[support_flash_attn]
        support_vllm = model_info.get('support_vllm', False)
        support_vllm = bool_mapping[support_vllm]
        requires = ', '.join(model_info['requires'])
        tags = model_info.get('tags', [])
        tags_str = ', '.join(tags)
        if len(tags_str) == 0:
            tags_str = '-'
        hf_model_id = model_info.get('hf_model_id')
        if hf_model_id is None:
            hf_model_id = '-'
        r = [
            model_name, model_id, lora_target_modules, template, support_flash_attn, support_vllm, requires, tags_str,
            hf_model_id
        ]
        res.append(r)
    text = ''
    for r in res:
        ms_url = f'https://modelscope.cn/models/{r[1]}/summary'
        if r[8] != '-':
            hf_url = f'https://huggingface.co/{r[8]}'
            hf_model_id_str = f'[{r[8]}]({hf_url})'
        else:
            hf_model_id_str = '-'
        text += f'|{r[0]}|[{r[1]}]({ms_url})|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}|{r[7]}|{hf_model_id_str}|\n'
    print(f'模型总数: {len(res)}')
    result += text
    for idx, fpath in enumerate(fpaths):
        with open(fpath, 'r') as f:
            text = f.read()
        start_idx = text.find('| Model Type |')
        end_idx = text.find(end_words[idx])
        output = text[:start_idx] + result + '\n\n' + text[end_idx:]
        with open(fpath, 'w') as f:
            text = f.write(output)
    return res


if __name__ == '__main__':
    get_model_info_table()
