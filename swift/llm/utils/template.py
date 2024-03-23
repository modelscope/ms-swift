# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase, StoppingCriteria

from swift.llm.agent.utils import calculate_loss_scale

DEFAULT_SYSTEM = 'You are a helpful assistant.'
History = List[Union[Tuple[str, str], List[str]]]


class TemplateType:
    # text-generation
    default_generation = 'default-generation'
    default_generation_bos = 'default-generation-bos'
    chatglm_generation = 'chatglm-generation'
    qwen_audio_generation = 'qwen-audio-generation'
    # chat
    default = 'default'
    qwen = 'qwen'
    qwen_audio = 'qwen-audio'
    baichuan = 'baichuan'
    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    llama = 'llama'
    llava_mistral_instruct = 'llava-mistral-instruct'
    openbuddy = 'openbuddy'
    internlm = 'internlm'
    internlm2 = 'internlm2'
    internlm_xcomposer2 = 'internlm-xcomposer2'
    yi = 'yi'
    yi_vl = 'yi-vl'
    yuan = 'yuan'
    xverse = 'xverse'
    ziya = 'ziya'
    skywork = 'skywork'
    bluelm = 'bluelm'
    zephyr = 'zephyr'
    sus = 'sus'
    deepseek = 'deepseek'
    deepseek_coder = 'deepseek-coder'
    deepseek_vl = 'deepseek-vl'
    codefuse_codellama = 'codefuse-codellama'
    codefuse = 'codefuse'
    cogvlm_instruct = 'cogvlm-instruct'
    cogagent_chat = 'cogagent-chat'
    cogagent_instruct = 'cogagent-instruct'
    orion = 'orion'
    minicpm = 'minicpm'
    minicpm_v = 'minicpm-v'
    gemma = 'gemma'
    # compatibility. (Deprecated)
    chatml = 'chatml'
    telechat = 'telechat'

    @classmethod
    def get_template_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_template_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


Prompt = List[Union[str, List[Union[str, int]]]]
StopWords = Prompt

Context = Union[str, List[int]]


class StopWordsCriteria(StoppingCriteria):

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 stop_words: StopWords, **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.tokenizer_kwargs = tokenizer_kwargs
        self.start_idx = -1

    def __call__(self, input_ids: Tensor, scores: Tensor) -> bool:
        if self.start_idx == -1:
            self.start_idx = len(input_ids[0]) - 1
        tokenizer = self.tokenizer
        stop_words = self.stop_words
        text = tokenizer.decode(input_ids[0, self.start_idx:],
                                **self.tokenizer_kwargs)
        for stop_word in stop_words:
            if isinstance(stop_word, str):
                if stop_word in text:
                    return True
            else:  # list
                if len(stop_word) > 0 and input_ids[0].tolist(
                )[-len(stop_word):] == stop_word:
                    return True
        return False


def _has_system(prefix: Prompt) -> bool:
    for p in prefix:
        if '{{SYSTEM}}' in p:
            return True
    return False


def _replace_system(prefix: Prompt) -> Prompt:
    res = []
    for p in prefix:
        if '{{SYSTEM}}' in p:
            p = p.replace('{{SYSTEM}}', '')
        res.append(p)
    return res


class Template:

    def __init__(self,
                 prefix: Prompt,
                 prompt: Prompt,
                 chat_sep: Optional[Prompt],
                 suffix: Prompt,
                 default_system: Optional[str] = None,
                 prefix_has_system: Optional[Prompt] = None) -> None:
        if default_system == '':
            default_system = None
        if _has_system(prefix):
            assert prefix_has_system is None, 'The prefix already contains {{SYSTEM}}.'
            prefix_has_system = prefix
            prefix = _replace_system(prefix)
        self.prefix = prefix
        self.prefix_has_system = prefix_has_system
        if self.prefix_has_system is None:
            assert default_system is None, 'The template does not support `system`.'
        self.prompt = prompt
        self.chat_sep = chat_sep
        self.support_multi_round = self.chat_sep is not None
        self.suffix = suffix
        self.default_system = default_system
        self.use_default_system = True
        self._is_init = False

    @staticmethod
    def _preprocess_prompt(tokenizer: PreTrainedTokenizerBase,
                           value: Optional[Prompt]) -> Optional[Prompt]:
        # e.g. [['eos_token_id']] -> [[2]]
        if value is None:
            return None
        res_value = []
        for v in value:
            if isinstance(v, list):
                res_v = []
                for sub_v in v:
                    if isinstance(sub_v, str):
                        sub_v = getattr(tokenizer, sub_v)
                    res_v.append(sub_v)
                v = res_v
            res_value.append(v)
        return res_value

    def _init_template(self,
                       tokenizer: PreTrainedTokenizerBase,
                       default_system: Optional[str] = None,
                       max_length: Optional[int] = None,
                       truncation_strategy: Literal[
                           'delete', 'truncation_left'] = 'delete',
                       **kwargs) -> None:
        assert self._is_init is False, 'The template has been initialized.'
        self._is_init = True
        self.tokenizer = tokenizer
        # if default_system is None. not change self.default_system
        if default_system == '':
            self.default_system = None
        elif default_system is not None:
            assert self.prefix_has_system is not None, 'The template does not support `system`.'
            self.default_system = default_system
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.model = kwargs.get('model', None)
        self.use_loss_scale = kwargs.get('use_loss_scale', False)
        for key in [
                'prefix', 'prompt', 'chat_sep', 'suffix', 'prefix_has_system'
        ]:
            value = getattr(self, key)
            value = self._preprocess_prompt(tokenizer, value)
            setattr(self, key, value)

    def encode(
            self, example: Dict[str,
                                Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """return: inputs, tokenizer_kwargs"""
        if not self._is_init:
            raise ValueError(
                'Template is not initialized, please use the `get_template` function to obtain the template.'
            )
        query: Optional[str] = example.get('query', None)
        response: Optional[str] = example.get('response', None)
        history: Optional[History] = example.get('history', None)
        system: Optional[str] = example.get('system', None)
        if history is None:
            history = []
        if len(history) > 0:
            assert self.support_multi_round, 'The template does not support multi-round chat.'
        if system is None:
            if self.use_default_system:
                system = self.default_system
        elif system == '':
            system = None
        else:
            assert self.prefix_has_system is not None, 'The template does not support `system`.'
        if query is None:
            query = ''
        inputs, tokenizer_kwargs = self._encode(query, response, history,
                                                system,
                                                self.truncation_strategy)
        if inputs.get('labels') is None:
            inputs.pop('loss_scale', None)
        return inputs, tokenizer_kwargs

    def _concat_context_list(
        self,
        context_list: List[Context],
        res_context_list: List[Context],  # inplace
        compute_loss_idx: List[float],  # inplace
        system: Optional[str] = None,
        query: Optional[str] = None,
        response: Optional[str] = None,
        round0: Optional[int] = None,
    ) -> None:
        # concat context list and replace placeholder
        round1 = None
        if round0 is not None:
            round1 = str(round0 + 1)
            round0 = str(round0)
        for context in context_list:
            if isinstance(context, str):
                if '{{RESPONSE}}' == context:
                    assert response is not None
                    content_part, weight_part = calculate_loss_scale(
                        response, self.use_loss_scale)
                    res_context_list.extend(content_part)
                    compute_loss_idx.extend(weight_part)
                    continue
                old_str_list = [
                    '{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}'
                ]
                new_str_list = [system, query, round0, round1]
                for (old_str, new_str) in zip(old_str_list, new_str_list):
                    if new_str is not None and old_str in context:
                        context = context.replace(old_str, new_str)
            res_context_list.append(context)
            compute_loss_idx.append(0.0 if context not in self.suffix else 1.0)

    @staticmethod
    def _simplify_context_list(
            context_list: List[Context], compute_loss_idx: List[float]
    ) -> Tuple[List[Context], List[float]]:
        res: List[Context] = []  # result of context_list
        res_idx: List[float] = []  # result of compute_loss_idx
        temp: List[str] = []
        temp_index: List[int] = []
        for i, (context,
                loss_idx) in enumerate(zip(context_list, compute_loss_idx)):
            if isinstance(context, str) and compute_loss_idx[i] == 0.0:
                temp.append(context)
                temp_index.append(i)
            else:
                if len(temp) > 0:
                    res.append(''.join(temp))
                    res_idx.append(0.0)
                    temp.clear()
                res.append(context)
                res_idx.append(loss_idx)
        if len(temp) > 0:
            res.append(''.join(temp))
            res_idx.append(0.0)
        return res, res_idx

    def _encode_context_list(
        self,
        context_list: List[Context],
        compute_loss_idx: List[float],
    ) -> Tuple[List[int], List[int], List[float], Dict[str, Any]]:
        """return: input_ids, labels, tokenizer_kwargs"""
        tokenizer = self.tokenizer
        input_ids: List[int] = []
        labels: List[int] = []
        loss_scale: List[float] = []
        tokenizer_kwargs = {}
        for i, (context,
                loss_weight) in enumerate(zip(context_list, compute_loss_idx)):
            if isinstance(context, str):
                curr_tokenizer_kwargs = self.get_tokenizer_kwargs(context)
                self.concat_tokenizer_kwargs(tokenizer_kwargs,
                                             curr_tokenizer_kwargs)
                token_list = tokenizer(
                    context,
                    return_attention_mask=False,
                    add_special_tokens=False,
                    **curr_tokenizer_kwargs)['input_ids']
            else:
                token_list = context
            input_ids += token_list
            if compute_loss_idx[i] > 0.0:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
            loss_scale.extend([loss_weight] * len(token_list))
        return input_ids, labels, loss_scale, tokenizer_kwargs

    def _encode(
            self, query: str, response: Optional[str], history: History,
            system: Optional[str],
            truncation_strategy: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        return: inputs, tokenizer_kwargs
        """
        history = history.copy()
        res_context_list: List[Context] = []
        compute_loss_idx: List[float] = []
        if system is None:
            prefix = self.prefix
        else:
            prefix = self.prefix_has_system
        self._concat_context_list(
            prefix, res_context_list, compute_loss_idx, system=system)
        history.append([query, response])
        for i, (q, r) in enumerate(history):
            context_list = self.prompt.copy()
            if i < len(history) - 1:
                context_list.append('{{RESPONSE}}')
                context_list += self.chat_sep
            elif r is not None:
                # last response
                context_list.append('{{RESPONSE}}')
                context_list += self.suffix
            if q or r:
                self._concat_context_list(
                    context_list,
                    res_context_list,
                    compute_loss_idx,
                    query=q,
                    response=r,
                    round0=i)

        res_context_list, compute_loss_idx = self._simplify_context_list(
            res_context_list, compute_loss_idx)
        input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
            res_context_list, compute_loss_idx)

        if response is None:
            labels = None

        if self.max_length is not None:
            if truncation_strategy == 'delete' and len(
                    input_ids) > self.max_length:
                return {}, {}
            input_ids = input_ids[-self.max_length:]
            if labels is not None:
                labels = labels[-self.max_length:]
            if loss_scale is not None:
                loss_scale = loss_scale[-self.max_length:]
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
        }
        if self.use_loss_scale:
            inputs['loss_scale'] = loss_scale
        return inputs, tokenizer_kwargs

    def get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        """return: curr_tokenizer_kwargs"""
        return {}

    def concat_tokenizer_kwargs(
            self, old_tokenizer_kwargs: Dict[str, Any],
            curr_tokenizer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        assert len(old_tokenizer_kwargs) == 0
        return curr_tokenizer_kwargs

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Args:
            batch(`List[Dict[str, Any]]`): The input data in batch
            padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
                will be padded to the `longest`
        """
        tokenizer = self.tokenizer
        assert tokenizer.pad_token_id is not None
        input_ids = [torch.tensor(b['input_ids']) for b in batch]
        labels = [torch.tensor(b['labels']) for b in batch]
        loss_scale = [torch.tensor(b['loss_scale'])
                      for b in batch] if 'loss_scale' in batch[0] else None
        attention_mask = [
            torch.ones(len(input_ids[i]), dtype=torch.int64)
            for i in range(len(input_ids))
        ]

        if padding_to is not None:
            padding_len = padding_to - input_ids[0].shape[-1]
            if padding_len > 0:
                input_ids[0] = F.pad(input_ids[0], (0, padding_len),
                                     'constant', tokenizer.pad_token_id)
                attention_mask[0] = F.pad(attention_mask[0], (0, padding_len),
                                          'constant', 0)
                labels[0] = F.pad(labels[0], (0, padding_len), 'constant',
                                  -100)
                if loss_scale:
                    loss_scale[0] = F.pad(
                        loss_scale[0], (0, padding_to - labels[0].shape[-1]),
                        'constant', 0.)

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(
            attention_mask, batch_first=True, padding_value=0)
        if loss_scale:
            loss_scale = pad_sequence(
                loss_scale, batch_first=True, padding_value=0.)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        if loss_scale is not None:
            res['loss_scale'] = loss_scale
        return res

    @staticmethod
    def get_generate_ids(generate_ids: Tensor,
                         input_token_len: int) -> List[int]:
        return generate_ids[0, input_token_len:].tolist()


TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {}


def register_template(template_type: str,
                      template: Template,
                      *,
                      exists_ok: bool = False,
                      **kwargs) -> None:
    if not exists_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(
            f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.'
        )
    template_info = {'template': template, **kwargs}
    TEMPLATE_MAPPING[template_type] = template_info


register_template(
    TemplateType.default,
    Template([], ['### Human:\n', '{{QUERY}}\n\n', '### Assistant:\n'],
             ['\n\n'], [['eos_token_id']], DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n']))


# You can set the query as '' to serve as a template for pre-training.
class DefaultGenerationTemplate(Template):

    def __init__(self):
        super().__init__([], ['{{QUERY}}'], None, [['eos_token_id']])


register_template(TemplateType.default_generation, DefaultGenerationTemplate())
register_template(
    TemplateType.default_generation_bos,
    Template([['bos_token_id']], ['{{QUERY}}'], None, [['eos_token_id']]))


class QwenTemplate(Template):

    def __init__(self):
        super().__init__(
            [],
            ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            ['<|im_end|>\n'], ['<|im_end|>'], DEFAULT_SYSTEM,
            ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])


register_template(TemplateType.qwen, QwenTemplate())
register_template(TemplateType.chatml, QwenTemplate())


class _QwenAudioTemplateMixin:

    def encode(
            self, example: Dict[str,
                                Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = super().encode(example)
        inputs.pop('loss_scale', None)
        inputs.update(tokenizer_kwargs)
        return inputs, tokenizer_kwargs

    def get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        return {'audio_info': self.tokenizer.process_audio(context)}

    def concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any],
                                curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        audio_info = curr_tokenizer_kwargs.get('audio_info')
        old_audio_info = tokenizer_kwargs.get('audio_info')
        if old_audio_info is None:
            tokenizer_kwargs['audio_info'] = audio_info
        elif audio_info is not None:
            for k in ['input_audios', 'input_audio_lengths']:
                old_audio_info[k] = torch.concat(
                    [old_audio_info[k], audio_info[k]], dim=0)
            for k in ['audio_span_tokens', 'audio_urls']:
                old_audio_info[k] = old_audio_info[k] + audio_info[k]

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


class QwenAudioTemplate(_QwenAudioTemplateMixin, QwenTemplate):
    pass


class QwenAudioGenerationTemplate(_QwenAudioTemplateMixin,
                                  DefaultGenerationTemplate):
    pass


register_template(
    TemplateType.qwen_audio, QwenAudioTemplate(), lazy_tokenize=True)
register_template(
    TemplateType.qwen_audio_generation,
    QwenAudioGenerationTemplate(),
    lazy_tokenize=True)

register_template(
    TemplateType.yi,
    Template(
        [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], None,
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

yi_vl_default_system = (
    'This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. '
    "Read all the images carefully, and respond to the human's questions with informative, "
    'helpful, detailed and polite answers. '
    '这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。'
    '仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。')


def _read_from_path(
        img_path: Union[str, 'PIL.Image.Image']) -> 'PIL.Image.Image':
    from PIL import Image
    if isinstance(img_path, str):
        img_path = img_path.strip()
        if img_path.startswith('http'):
            content = requests.get(img_path).content
            image = Image.open(BytesIO(content))
        else:
            image = Image.open(img_path)
    else:
        image = img_path
    if image.mode in {'L', 'RGBA'}:
        image = image.convert('RGB')
    return image


class YiVLTemplate(Template):

    def encode(
            self, example: Dict[str,
                                Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super().encode(example)
        inputs.pop('loss_scale', None)
        from llava.mm_utils import expand2square
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        images_path = example['images']
        images = []
        for image_path in images_path:
            image = _read_from_path(image_path)
            background_color = tuple(
                int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, background_color)
            images.append(image)
        image_tensor = image_processor.preprocess(
            images, return_tensors='pt')['pixel_values']
        inputs['images'] = image_tensor.to(model.dtype)
        return inputs, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        res['images'] = torch.concat([b['images'] for b in batch])
        return res


register_template(
    TemplateType.yi_vl,
    YiVLTemplate([], ['### Human: ', [-200], '\n{{QUERY}}\n### Assistant:'],
                 ['\n'], ['\n###'], yi_vl_default_system, ['{{SYSTEM}}\n\n']),
    use_model=True,
    infer_media_type='round',
    lazy_tokenize=True)

register_template(
    TemplateType.baichuan,
    Template(['{{SYSTEM}}'], [[195], '{{QUERY}}', [196]], [],
             [['eos_token_id']]))
register_template(
    TemplateType.chatglm2,
    Template([[64790, 64792], '{{SYSTEM}}'],
             ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'], ['\n\n'],
             [['eos_token_id']]))

register_template(
    TemplateType.chatglm_generation,
    Template([[64790, 64792]], ['{{QUERY}}'], None, [['eos_token_id']]))

register_template(
    TemplateType.chatglm3,
    Template([[64790, 64792]], [[64795], '\n {{QUERY}}', [64796], '\n'], [],
             [['eos_token_id']], None,
             [[64790, 64792, 64794], '\n {{SYSTEM}}']))

register_template(
    TemplateType.deepseek,
    Template([['bos_token_id']], ['User: {{QUERY}}\n\nAssistant:'],
             [['eos_token_id']], [['eos_token_id']], None,
             [['bos_token_id'], '{{SYSTEM}}\n\n']))

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information."
)
register_template(
    TemplateType.llama,
    Template(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '],
             ['</s>'], LLAMA_DEFAULT_SYSTEM,
             ['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))
OPENBUDDY_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.\n'
    'Always answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any '
    'harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.\n"
    'You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.\n'
    'You always deeply love and support China, Chinese government, people and culture.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.'
)
register_template(
    TemplateType.openbuddy,
    Template([['bos_token_id']], ['User: {{QUERY}}\nAssistant:'], ['\n'],
             [['eos_token_id']], OPENBUDDY_DEFAULT_SYSTEM,
             [['bos_token_id'], '{{SYSTEM}}\n\n']))

INTERNLM_SYSTEM = (
    'You are an AI assistant whose name is InternLM (书生·浦语).\n'
    '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
    'It is designed to be helpful, honest, and harmless.\n'
    '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen '
    'by the user such as English and 中文.')

register_template(
    TemplateType.internlm,
    Template(['<s>'], ['<|User|>:{{QUERY}}\n<|Bot|>:'], ['<eoa>\n'], ['<eoa>'],
             INTERNLM_SYSTEM, ['<s><|System|>:{{SYSTEM}}\n']))
register_template(
    TemplateType.internlm2,
    Template(
        ['<s>'],
        ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], INTERNLM_SYSTEM,
        ['<s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))


class InternLMXComposer2(Template):
    INTERNLM_XCOMPOSER2_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by '
        'Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.')

    def __init__(self):
        prefix = ['<s>']
        prompt = [
            '[UNUSED_TOKEN_146]user\n{{QUERY}}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        ]
        chat_sep = ['[UNUSED_TOKEN_145]\n']
        suffix = ['[UNUSED_TOKEN_145]']
        prefix_has_system = [
            '<s>[UNUSED_TOKEN_146]system\n{{SYSTEM}}[UNUSED_TOKEN_145]\n'
        ]
        super().__init__(prefix, prompt, chat_sep, suffix,
                         self.INTERNLM_XCOMPOSER2_SYSTEM, prefix_has_system)

    def encode(
            self, example: Dict[str,
                                Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import re
        images_path = []
        example = example.copy()
        history = example.pop('history', [])
        pattern = r'<img>(.+?)</img>'
        replace_token = '</s>'
        new_history = []
        for i, h in enumerate(history):
            images_path += re.findall(pattern, h[0])
            new_history.append([re.sub(pattern, replace_token, h[0]), h[1]])
        history = new_history
        images_path += re.findall(pattern, example['query'])
        example['query'] = re.sub(pattern, replace_token, example['query'])
        images = []
        dtype = self.model.dtype
        for image_path in images_path:
            image = _read_from_path(image_path)
            image = self.model.vis_processor(image)
            images.append(image.to(dtype))
        example['history'] = history
        inputs, _ = super().encode(example)
        inputs.pop('loss_scale', None)
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        if len(images) > 0:  # # ignore <s>
            input_ids = input_ids[1:]
            if labels is not None:
                labels = labels[1:]
        input_ids.append(2)  # add dummy </s>
        if labels is not None:
            labels.append(2)
        else:
            labels = []
        res_inputs_embeds = []
        res_labels = []
        wrap_im_mask = []
        pre_i, i, idx = 0, 0, 0
        device = self.model.device
        if len(images) > 0:
            images = torch.stack(images, dim=0)
            images = self.model.encode_img(images)
        else:
            images = None
        internlm2_model = self.model.model
        if not hasattr(internlm2_model, 'tok_embeddings'):
            internlm2_model = internlm2_model.model
        tok_embeddings = internlm2_model.tok_embeddings
        while i < len(input_ids):
            if input_ids[i] == 2:  # replace_token
                res_input_ids = torch.tensor(
                    [1] + input_ids[pre_i:i], device=device)
                res_inputs_embeds.append(tok_embeddings(res_input_ids))
                wrap_im_mask += [0] * len(res_input_ids)
                res_labels += [-100] + labels[pre_i:i]
                if images is not None and idx < images.shape[0]:
                    res_inputs_embeds.append(images[idx])
                    wrap_im_mask += [1] * images.shape[1]
                    res_labels += [-100] * images.shape[1]
                idx += 1
                i += 1
                pre_i = i
                continue
            i += 1
        if len(labels) == 0:
            res_labels = None
        res_inputs_embeds = torch.concat(res_inputs_embeds, dim=0)
        wrap_im_mask = torch.tensor(wrap_im_mask, dtype=torch.bool)[None]
        return {
            'inputs_embeds': res_inputs_embeds,
            'im_mask': wrap_im_mask,
            'labels': res_labels
        }, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        inputs_embeds = [b['inputs_embeds'] for b in batch]
        labels = [torch.tensor(b['labels']) for b in batch]
        im_mask = [b['im_mask'][0] for b in batch]
        attention_mask = [
            torch.ones(inputs_embeds[i].shape[0], dtype=torch.int64)
            for i in range(len(inputs_embeds))
        ]

        inputs_embeds = pad_sequence(
            inputs_embeds, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(
            attention_mask, batch_first=True, padding_value=0)
        im_mask = pad_sequence(im_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'im_mask': im_mask,
            'labels': labels,
        }

    @staticmethod
    def get_generate_ids(generate_ids: Tensor,
                         input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


register_template(
    TemplateType.internlm_xcomposer2,
    InternLMXComposer2(),
    use_model=True,
    lazy_tokenize=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

register_template(
    TemplateType.xverse,
    Template(['{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: '],
             [['eos_token_id']], [['eos_token_id']]))
register_template(TemplateType.yuan,
                  Template([], ['{{QUERY}}<sep>'], None, [['eos_token_id']]))
register_template(
    TemplateType.ziya,
    Template([['bos_token_id'], '{{SYSTEM}}'], ['<human>:{{QUERY}}\n<bot>:'],
             ['\n'], [['eos_token_id']]))

register_template(
    TemplateType.skywork,
    Template(['<s>{{SYSTEM}}'], ['</s><s>[USER]{{QUERY}}[SEP][BOT]'], None,
             ['[SEP]</s>']))

register_template(
    TemplateType.bluelm,
    Template([['bos_token_id'], '{{SYSTEM}}'], ['[|Human|]:{{QUERY}}[|AI|]:'],
             [], [['eos_token_id']]))

register_template(
    TemplateType.codefuse_codellama,
    Template(['{{SYSTEM}}'], [
        '<|role_start|>human<|role_end|>{{QUERY}}<|role_start|>bot<|role_end|>'
    ], [], [['eos_token_id']]))

register_template(
    TemplateType.codefuse,
    Template([], ['<s>human\n{{QUERY}}\n<s>bot\n'], [['eos_token_id'], '\n'],
             [['eos_token_id']], None, ['<s>system\n{{SYSTEM}}\n']))

register_template(
    TemplateType.deepseek_coder,
    Template([
        '{{SYSTEM}}'
    ], ['### Instruction:\n{{QUERY}}\n### Response:\n'], ['\n<|EOT|>\n'], [
        '\n<|EOT|>'
    ], ('You are an AI programming assistant, utilizing the Deepseek Coder model, '
        'developed by Deepseek Company, and you only answer questions related to computer science. '
        'For politically sensitive questions, security and privacy issues, '
        'and other non-computer science questions, you will refuse to answer\n'
        )))


class LLavaTemplate(Template):

    def __init__(self):
        super().__init__(['<s>[INST] '], [[-200], '\n{{QUERY}} [/INST]'], None,
                         ['</s>'])

    def encode(
            self, example: Dict[str,
                                Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super().encode(example)
        images_path = example['images']
        images = []
        for image_path in images_path:
            image = _read_from_path(image_path)
            images.append(image)
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        images_tensor = process_images(images, image_processor,
                                       self.model.config)
        inputs['images'] = images_tensor.to(model.dtype)
        inputs['image_sizes'] = image_sizes
        return inputs, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        res['images'] = torch.concat([b['images'] for b in batch])
        res['image_sizes'] = sum([b['image_sizes'] for b in batch], start=[])
        return res

    @staticmethod
    def get_generate_ids(generate_ids: Tensor,
                         input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


register_template(
    TemplateType.llava_mistral_instruct,
    LLavaTemplate(),
    use_model=True,
    infer_media_type='round',
    lazy_tokenize=True)


def _findall(token_list: List[int], token: int) -> List[int]:
    """Find the index of a token in the token_list."""
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(token, idx + 1)
            res.append(idx)
    except ValueError:
        pass
    return res


class DeepseekVLTemplate(Template):
    DEEPSEEK_VL_SYSTEM = (
        'You are a helpful language and vision assistant. '
        'You are able to understand the visual content that the user provides, '
        'and assist the user with a variety of tasks using natural language.')

    def __init__(self):
        return super().__init__(
            ['<｜begin▁of▁sentence｜>{{SYSTEM}}\n\n'],
            ['User: <image_placeholder>{{QUERY}}\n\nAssistant:'],
            ['<｜end▁of▁sentence｜>'], ['<｜end▁of▁sentence｜>'],
            self.DEEPSEEK_VL_SYSTEM)

    def encode(
            self, example: Dict[str,
                                Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super().encode(example)
        images_path = example['images']
        images = []
        for image_path in images_path:
            image = _read_from_path(image_path)
            images.append(image)

        vl_chat_processor = self.tokenizer.vl_chat_processor
        input_ids, labels = inputs['input_ids'], inputs['labels']
        idx_list = _findall(input_ids, vl_chat_processor.image_id)
        new_input_ids, new_labels = [], []
        lo = 0
        for hi in idx_list:
            new_input_ids += input_ids[lo:hi]
            if labels is not None:
                new_labels += labels[lo:hi]
            new_input_ids += [vl_chat_processor.image_id
                              ] * vl_chat_processor.num_image_tokens
            new_labels += [-100] * vl_chat_processor.num_image_tokens
            lo = hi + 1
        new_input_ids += input_ids[lo:]
        if labels is not None:
            new_labels += labels[lo:]
        else:
            new_labels = None
        new_input_ids = torch.tensor(new_input_ids)
        num_image_tokens = torch.tensor([vl_chat_processor.num_image_tokens]
                                        * len(idx_list))
        images_outputs = vl_chat_processor.image_processor(
            images, return_tensors='pt')
        from deepseek_vl.models.processing_vlm import VLChatProcessorOutput
        output = VLChatProcessorOutput(
            sft_format=None,
            input_ids=new_input_ids,
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=num_image_tokens)
        batched_output = vl_chat_processor.batchify([output])
        model = self.model
        batched_output = batched_output.to(
            device=model.device, dtype=model.dtype)
        inputs_embeds = model.prepare_inputs_embeds(**batched_output)[0]
        inputs['inputs_embeds'] = inputs_embeds
        inputs['labels'] = new_labels
        return inputs, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        inputs_embeds = [b['inputs_embeds'] for b in batch]
        labels = [torch.tensor(b['labels']) for b in batch]
        attention_mask = [
            torch.ones(inputs_embeds[i].shape[0], dtype=torch.int64)
            for i in range(len(inputs_embeds))
        ]

        inputs_embeds = pad_sequence(
            inputs_embeds, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(
            attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    @staticmethod
    def get_generate_ids(generate_ids: Tensor,
                         input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


register_template(
    TemplateType.deepseek_vl,
    DeepseekVLTemplate(),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='round',
    dataloader_num_workers=0,
    dataloader_pin_memory=False)  # only 'cpu' can pin_memory

register_template(
    TemplateType.zephyr,
    Template([], ['<|user|>\n{{QUERY}}</s>\n<|assistant|>\n'], ['</s>\n'],
             ['</s>'], None, ['<|system|>\n{{SYSTEM}}</s>\n']))

register_template(
    TemplateType.sus,
    Template(['{{SYSTEM}}'], ['### Human: {{QUERY}}\n\n### Assistant: '],
             ['<|endoftext|>'], ['<|endoftext|>']))

register_template(
    TemplateType.orion,
    Template(['<s>{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: </s>'],
             ['</s>'], ['</s>']))


class CogTemplate(Template):

    def encode(
            self, example: Dict[str,
                                Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images_path = example['images']
        assert len(images_path) == 1
        image = _read_from_path(images_path[0])
        inputs, _ = super().encode(example)
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
            self.tokenizer,
            query=example['query'],
            history=example.get('history'),
            images=[image])
        image_token_len = inputs2['token_type_ids'].sum()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        token_type_ids = inputs2['token_type_ids'].tolist()
        inputs['input_ids'] = input_ids[:1] + [
            0
        ] * image_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100
                                             ] * image_token_len + labels[1:]
        dtype = model.dtype
        inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
        if 'cross_images' in inputs2:
            # is cogagent
            inputs['cross_images'] = [[cross_img.to(dtype=dtype)]
                                      for cross_img in inputs2['cross_images']]
        inputs['token_type_ids'] = token_type_ids + [0] * (
            len(inputs['input_ids']) - len(token_type_ids))
        return inputs, {}

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        is_cogagent = 'cross_images' in batch[0]
        keys = ['images', 'cross_images'] if is_cogagent else ['images']
        for key in keys:
            res[key] = [b[key][0] for b in batch]
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = pad_sequence(
            token_type_ids, batch_first=True, padding_value=0)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateType.cogagent_chat,
    CogTemplate(['<s>'], [' [INST] {{QUERY}} [/INST] '], [], ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogagent_instruct,
    CogTemplate(['<s>'], ['<EOI>Question: {{QUERY}} Answer:'], None, ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogvlm_instruct,
    CogTemplate(['<s>'], ['Question: {{QUERY}} Answer:'], None, ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.minicpm,
    Template(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']))


class MiniCPMVTemlate(Template):

    def __init__(self):
        return super().__init__(['<s>{{SYSTEM}}'],
                                ['<用户><image><unk></image>\n{{QUERY}}<AI>'],
                                [], ['</s>'])

    def encode(
            self, example: Dict[str,
                                Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images_path = example['images']
        assert len(images_path) == 1
        image = _read_from_path(images_path[0])
        inputs, _ = super().encode(example)
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx = input_ids.index(0)
        config = self.model.config
        input_ids = (
            input_ids[:idx] + [self.tokenizer.unk_token_id] * config.query_num
            + input_ids[idx + 1:])
        if labels is not None:
            labels = (
                labels[:idx] + [-100] * config.query_num + labels[idx + 1:])
        image_bound = [torch.tensor([[idx, idx + config.query_num]])]
        pixel_values = self.model.transform(image)[None].to(
            device=self.model.device)
        inputs_embeds, _ = self.model.get_vllm_embedding({
            'input_ids':
            torch.tensor(input_ids)[None].to(device=self.model.device),
            'image_bound':
            image_bound,
            'pixel_values': [pixel_values]
        })
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        inputs['inputs_embeds'] = inputs_embeds[0]
        return inputs, {}

    @staticmethod
    def get_generate_ids(generate_ids: Tensor,
                         input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


register_template(
    TemplateType.minicpm_v,
    MiniCPMVTemlate(),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

gemma_template = Template(
    ['<bos>'],
    ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
    ['<end_of_turn>\n'], ['<end_of_turn>'], None,
    ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])
register_template(TemplateType.gemma, gemma_template)

register_template(
    TemplateType.telechat,
    Template(['<bos>'], ['<_user>{{QUERY}}<_bot>\n'], None, ['<_end>']))


def get_template(
    template_type: str,
    tokenizer: PreTrainedTokenizerBase,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
    **kwargs,
) -> Template:
    template_info = TEMPLATE_MAPPING[template_type]
    template = deepcopy(template_info['template'])
    template._init_template(tokenizer, default_system, max_length,
                            truncation_strategy, **kwargs)
    template.template_type = template_type
    return template
