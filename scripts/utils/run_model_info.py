from typing import Any, List

from swift.llm import MODEL_MAPPING, ModelType, get_default_lora_target_modules


def get_model_info_table():
    fpaths = ['docs/source/Instruction/支持的模型和数据集.md', 'docs/source_en/Instruction/Supported-models-datasets.md']
    end_words = [['### 多模态大模型', '## 数据集'], ['### MLLM', '## Datasets']]
    model_name_list = ModelType.get_model_name_list()
    result = [
        '| Model Type | Model ID | Default Lora Target Modules | Default Template |'
        ' Support Flash Attn | Support vLLM | Support LMDeploy | Support Megatron | Requires | Tags | HF Model ID |\n'
        '| ---------  | -------- | --------------------------- | ---------------- |'
        ' ------------------ | ------------ | ---------------- | ---------------- | -------- | ---- | ----------- |\n'
    ] * 2
    res_llm: List[Any] = []
    res_mllm: List[Any] = []
    bool_mapping = {True: '&#x2714;', False: '&#x2718;'}
    for model_name in model_name_list:
        model_info = MODEL_MAPPING[model_name]
        model_id = model_info['model_id_or_path']
        lora_target_modules = get_default_lora_target_modules(model_name)
        if isinstance(lora_target_modules, list):
            lora_target_modules = ', '.join(lora_target_modules)
        else:
            lora_target_modules = lora_target_modules.replace('|', '\\|').replace('*', '\\*')
        template = model_info['template']
        support_flash_attn = model_info.get('support_flash_attn', False)
        support_flash_attn = bool_mapping[support_flash_attn]
        support_vllm = model_info.get('support_vllm', False)
        support_vllm = bool_mapping[support_vllm]
        support_lmdeploy = model_info.get('support_lmdeploy', False)
        support_lmdeploy = bool_mapping[support_lmdeploy]
        support_megatron = model_info.get('support_megatron', False)
        support_megatron = bool_mapping[support_megatron]
        requires = ', '.join(model_info['requires'])
        tags = model_info.get('tags', [])
        if 'multi-modal' in tags:
            tags.remove('multi-modal')
            is_multi_modal = True
        else:
            is_multi_modal = False
        tags_str = ', '.join(tags)
        if len(tags_str) == 0:
            tags_str = '-'
        hf_model_id = model_info.get('hf_model_id')
        if hf_model_id is None:
            hf_model_id = '-'
        r = [
            model_name, model_id, lora_target_modules, template, support_flash_attn, support_vllm, support_lmdeploy,
            support_megatron, requires, tags_str, hf_model_id
        ]
        if is_multi_modal:
            res_mllm.append(r)
        else:
            res_llm.append(r)
    print(f'LLM总数: {len(res_llm)}, MLLM总数: {len(res_mllm)}')
    text = ['', '']  # llm, mllm
    for i, res in enumerate([res_llm, res_mllm]):
        for r in res:
            ms_url = f'https://modelscope.cn/models/{r[1]}/summary'
            if r[10] != '-':
                hf_url = f'https://huggingface.co/{r[10]}'
                hf_model_id_str = f'[{r[10]}]({hf_url})'
            else:
                hf_model_id_str = '-'
            text[i] += (f'|{r[0]}|[{r[1]}]({ms_url})|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}|{r[7]}|{r[8]}'
                        f'|{r[9]}|{hf_model_id_str}|\n')
        result[i] += text[i]

    for i, fpath in enumerate(fpaths):
        with open(fpath, 'r') as f:
            text = f.read()
        llm_start_idx = text.find('| Model Type |')
        mllm_start_idx = text[llm_start_idx + 1:].find('| Model Type |') + llm_start_idx + 1
        llm_end_idx = text.find(end_words[i][0])
        mllm_end_idx = text.find(end_words[i][1])
        output = text[:llm_start_idx] + result[0] + '\n\n' + text[llm_end_idx:mllm_start_idx] + result[
            1] + '\n\n' + text[mllm_end_idx:]
        with open(fpath, 'w') as f:
            f.write(output)


if __name__ == '__main__':
    get_model_info_table()
