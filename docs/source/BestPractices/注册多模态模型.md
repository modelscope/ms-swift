
# 注册多模态模型最佳实践

本文将介绍如何在ms-swift中注册多模态模型，并成功推理和训练。本文将以Qwen2.5-Omni为例子，注册新的model_type和template `my_qwen2_5_omni`，并支持文本、图片、视频和音频的训练。由于Qwen2.5-Omni已经在ms-swift中注册，我们可以通过显式指定model_type和template来使用我们自定义的部分。

## 环境准备
```shell
# 避免未来出现与文档的不兼容情况
pip install "ms-swift>=3.9,<3.10"

pip install "transformers==4.57.*" "qwen_omni_utils==0.0.8"
```


## 注册模型

第一步，我们需要注册模型，来获取模型和processor。

```python
from swift.llm import (
    register_model, ModelMeta, ModelGroup, Model, register_model_arch, MultiModelKeys,
    get_model_tokenizer_with_flash_attn, get_model_tokenizer
)
from swift.llm.model.model.qwen import patch_qwen_vl_utils
from swift.llm.model.utils import use_submodel_func
from swift.llm.model.patcher import patch_get_input_embeddings
from swift.utils import get_env_args


register_model_arch(
    MultiModelKeys(
        'my_qwen2_5_omni',
        # `freeze_llm`, `freeze_vit`, `freeze_aligner`将根据下面的值来决定其行为。
        # 例如：全参数训练，若设置`freeze_vit=True`，将冻结以`thinker.audio_tower`和`thinker.visual`为前缀的模型层的参数。
        # LoRA训练，若设置`freeze_vit=False`，将额外为以`thinker.audio_tower`和`thinker.visual`为前缀的Linear层添加LoRA。
        language_model='thinker.model',
        vision_tower=['thinker.audio_tower', 'thinker.visual'],
        aligner=['thinker.audio_tower.proj', 'thinker.visual.merger'],
        # generator的部分将永远不进行训练或处于冻结状态。
        generator=['talker', 'token2wav'],
    ))

def get_model_tokenizer_qwen2_5_omni(model_dir, *args, **kwargs):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig
    from qwen_omni_utils import vision_process
    print('Run my_qwen2_5_omni...')
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2_5OmniForConditionalGeneration
    # 自定义`get_model_tokenizer_with_flash_attn`中获取tokenizer和config的方式
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = Qwen2_5OmniConfig.from_pretrained(model_dir, trust_remote_code=True)
    enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, None)
    if enable_audio_output is not None:
        kwargs['model_config'].enable_audio_output = enable_audio_output
    # 可以通过环境变量来控制qwen_omni_utils库中的常量，例如：`MAX_PIXELS`等
    patch_qwen_vl_utils(vision_process)
    # 请尽量使用该函数来获取model和tokenizer。而避免直接使用AutoModelForCausalLM（会产生不兼容问题）。
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model:
        # 为了多模态模型的统一性，我们将模型的forward/generate函数替换为其language_model的forward/generate函数。
        # 自己处理额外的部分。
        use_submodel_func(model, 'thinker')
        # 一些对model/config的自定义（通常不需要设置，若训练/推理中出现报错，则根据特定模型进行配置）
        model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        model.config.talker_config.pad_token_id = None
        # 避免在训练时对leaf_variable进行inplace操作导致报错（将input_embeds中的部分内容替换为images_embeds的行为）
        patch_get_input_embeddings(model.thinker.visual, 'patch_embed')
    # 最终需要返回model和 processor（多模态）/tokenizer（纯文本）
    return model, processor


register_model(
    ModelMeta(
        'my_qwen2_5_omni',
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-Omni-3B', 'Qwen/Qwen2.5-Omni-3B'),
                Model('Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen2.5-Omni-7B'),
            ]),
        ],
        'my_qwen2_5_omni',
        # 用来获取model和processor的函数。
        get_model_tokenizer_qwen2_5_omni,
        is_multimodal=True,  # 是否是多模态模型
        model_arch='my_qwen2_5_omni',  # 通常只为多模态模型设置
        # 用于model_type的自动匹配
        architectures=['Qwen2_5OmniModel', 'Qwen2_5OmniForConditionalGeneration'],
        # 用来提示用户依赖版本（可删除）
        requires=['transformers>=4.50', 'soundfile', 'qwen_omni_utils', 'decord'],
        # 用来提示用户（可删除）
        tags=['vision', 'video', 'audio'],
        # 全参数训练/merge-lora需要额外保存的文件
        additional_saved_files=['spk_dict.pt'],
    ))

if __name__ == '__main__':
    # 测试与debug
    model, processor = get_model_tokenizer('Qwen/Qwen2.5-Omni-7B', model_type='my_qwen2_5_omni')
```

## 注册模板

第二步，我们需要注册模板，来自定义如何将文本、图片、视频和音频进行预处理（`_encode`和`_data_collator`方法）。这是ms-swift支持多模态模型训练的关键模块。预处理方式请参考transformers推理实现，并进行对齐。

template的功能如下：
1. 支持正常推理与训练，预处理文本和多模态信息，并支持grounding任务。
2. 支持padding_free和packing训练。
3. 支持混合模态数据训练。


```python
from swift.llm import (
    register_template, Template, get_packed_seq_params, to_float_dtype, TemplateMeta,
    get_template, get_model_tokenizer
)
from transformers.integrations import is_deepspeed_zero3_enabled
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.llm.template.vision_utils import load_audio
from swift.utils import get_env_args, get_logger, is_deepspeed_enabled
from functools import partial
from typing import Dict, List, Any, Literal, Optional
import torch

logger = get_logger()

class Qwen2_5OmniTemplate(Template):
    use_model = True  # 是否在预处理的过程中需要model参与
    # 需要注意是：并不是所有的多模态模型都能支持padding_free/packing。`transformers`库内的模型通常可以支持
    support_padding_free = True  # 是否支持padding_free和packing（多模态模型）
    norm_bbox = 'none'  # grounding任务使用绝对坐标还是norm1000坐标

    # 这里的tokens将不会被裁剪（例如设置`--truncation_strategy left/right`）
    # 并会使用简略方式打印（调用`template.safe_decode`）
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']

    def init_processor(self, processor) -> None:
        """在初始化processor时，额外初始化所需的一些常量"""
        if processor is None:
            return
        super().init_processor(processor)
        from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
        default = Qwen2_5OmniProcessorKwargs._defaults
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)
        # `QWENVL_BBOX_FORMAT`的含义参考grounding数据集自定义文档
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')


    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """读取多模态数据，并替换通用多模态tag。
        例如：图像tag从`<image>` -> `<|vision_bos|><|IMAGE|><|vision_eos|>`"""
        # 读取多模态数据也可以在`_encode`函数中进行，怎么方便怎么来。
        from qwen_omni_utils import fetch_image, fetch_video
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':  # 'vllm'推理场景下不需要处理
                inputs.audios[index] = load_audio(inputs.audios[index], self.sampling_rate)
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        elif media_type == 'video':
            video = inputs.videos[index]
            _video = fetch_video({'video': video})
            if isinstance(_video, torch.Tensor):
                _video = _video.to(torch.uint8)
            inputs.videos[index] = _video
            if self.use_audio_in_video:
                import librosa
                if video.startswith('http://') or video.startswith('https://'):
                    import audioread
                    video = audioread.ffdec.FFmpegAudioFile(video)
                video = librosa.load(video, sr=self.sampling_rate)[0]
                inputs.audios.insert(inputs.audio_idx, (video, 'video'))
                inputs.audio_idx += 1
                return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
            else:
                return ['<|vision_bos|><|VIDEO|><|vision_eos|>']


    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        """替换grounding任务的通用tag: `<ref-object>`"""
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """替换grounding任务的通用tag: `<bbox>`"""
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """支持packing & mrope。通常情况不需要继承该函数，这里为了自定义mrope的position_ids。"""
        position_ids = []
        for r in row:
            r = r.copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))
        packed = super().packing_row(row)
        packed['position_ids'] = torch.concat(position_ids, dim=-1)
        return packed

    def _get_new_tokens_use_audio_in_video(self, i, *, video_grid_thw, video_second_per_grid, audio_lengths,
                                           video_token_id, audio_token_id):
        """辅助函数，用于支持`use_audio_in_video`为True的情况"""
        merge_size = self.processor.image_processor.merge_size
        grid_thw = video_grid_thw[i]
        height = grid_thw[1] // merge_size
        width = grid_thw[2] // merge_size
        audio_token_indices = torch.arange(audio_lengths[i])
        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)

        video_token_indices = torch.broadcast_to(video_token_indices,
                                                 (video_token_indices.shape[0], height, width)).reshape(-1)
        video_token_indices = (video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
        tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
        video_chunk_indexes = self.processor.get_chunked_index(video_token_indices, tokens_per_chunk)
        audio_chunk_indexes = self.processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

        res = []
        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
            if j < len(video_chunk_indexes):
                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                res += video_token_id * video_seq_length
            if j < len(audio_chunk_indexes):
                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                res += audio_token_id * audio_seq_length
        return res


    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """这里决定如何将text/images/audios/videos -> input_ids、labels、loss_scale以及pixel_values等多模态内容
        这里的处理逻辑通常可以从对应模型的预处理代码实现中借鉴。
        推荐：请先做推理对齐再做训练"""
        encoded = Template._encode(self, inputs)  # 处理纯文本部分，具体请参考自定义模型文档
        logger.info_once('Run qwen2_5_omni template')
        processor = self.processor
        # 获取多模态内容
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=False,
            return_tensors='pt')
        # 我们不使用`processor`产生的input_ids和attention_mask。因为其不产生`labels`。
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)

        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        # audio模态
        audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)  # 查找所有的audio_token
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = ((audio_feature_lengths - 1) // 2 + 1 - 2) // 2 + 1
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        # video_audios_mask用于处理`use_audio_in_video`，区分是纯audio(0)还是video中的audio(1)
        video_audios_mask = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
            else:
                video_audios_mask.append(False)
        video_audios_mask = torch.tensor(video_audios_mask)
        if idx_list:
            # 过滤掉video中的audio的内容（将在video部分处理）
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            # 对input_ids的多模态tokens进行展开
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)

        # image和video模态
        for media_type in ['image', 'video']:
            token = f'<|{media_type.upper()}|>'
            token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']
                    _get_new_tokens_use_audio_in_video = partial(
                        self._get_new_tokens_use_audio_in_video,
                        video_grid_thw=media_grid_thw,
                        video_second_per_grid=video_second_per_grid,
                        audio_lengths=audio_lengths,
                        video_token_id=token_id,
                        audio_token_id=audio_token_id)
                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded.update(media_inputs)  # 将多模态内容加入其中
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """该函数通常用于解决混合模型训练zero2/zero3卡住的问题，
        即有的进程为纯文本数据未过vit，有的进程含图片数据过了vit。这里将创建dummy_image来解决。

        该函数将被注册在`model.forward`前的pre_forward_hook中。
        该函数需返回 含多模态信息的input_embeds。
        """
        if not self.is_training:
            return inputs

        input_ids = inputs['input_ids']
        input_features = inputs.get('input_features')
        feature_attention_mask = inputs.get('feature_attention_mask')

        base_model = self.get_base_model(model)
        inputs_embeds = base_model.thinker.model.embed_tokens(input_ids)
        thinker_config = model.config.thinker_config
        # 辅助函数，用于处理text/image/video混合模态数据场景。（内部会创建dummy_image）
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.thinker.visual, self.processor,
                                                   thinker_config)
        # 含audio的混合模态数据场景
        if input_features is None:
            if is_deepspeed_enabled() and not is_deepspeed_zero3_enabled():
                # 注意: 由于transformers实现中，经过audio部分模型层的次数与audio数量相关
                # 因此zero3在不同进程audios数不同场景下会卡住（需修改transformers代码修复）。此场景请使用zero2。
                input_features = input_ids.new_zeros([1, 128, 128], dtype=model.thinker.audio_tower.dtype)
                feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
                audio_embeds = model.thinker.get_audio_features(input_features, feature_attention_mask)
                inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_embeds = model.thinker.get_audio_features(input_features, feature_attention_mask)
            audio_mask = (input_ids == thinker_config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return {'inputs_embeds': inputs_embeds}

    def _get_position_ids(self, inputs: Dict[str, Any]):
        """辅助函数，用来获取mrope的position_ids"""
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self.model.thinker.get_rope_index(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        return self._concat_text_position_ids(position_ids)  # 第一维为text_position_ids

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """传入dataloader的`collate_fn`"""
        res = super()._data_collator(batch, padding_to=padding_to)
        if not self.padding_free and self.is_training:
            # 其中padding_free/packing场景将会在packing_row中处理position_ids。
            res['position_ids'] = self._get_position_ids(res)
        if 'position_ids' in res:
            # 创建`packed_seq_params`以支持padding_free/packing & flash-attn
            position_ids = res['position_ids']
            res['position_ids'] = position_ids[1:]
            res['text_position_ids'] = text_position_ids = position_ids[0]
            # https://github.com/huggingface/transformers/pull/40194
            res.update(get_packed_seq_params(text_position_ids))
        return res

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理`_data_collator`函数中的多模态部分。（该函数兼容padding_free/packing）"""
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res

    def generate(self, model, *args, **kwargs):
        """`PtEngine`会调用template.generate方法进行文本生成，这里继承进行自定义。"""
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


register_template(
    TemplateMeta('my_qwen2_5_omni', prefix=[], prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
                 chat_sep=['<|im_end|>\n'], suffix=['<|im_end|>'],
                 system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
                 default_system='You are a helpful assistant.', stop_words=['<|endoftext|>'],
                 agent_template='hermes',
                 template_cls=Qwen2_5OmniTemplate))

if __name__ == '__main__':
    # 测试与debug
    model, processor = get_model_tokenizer('Qwen/Qwen2.5-Omni-7B', model_type='my_qwen2_5_omni')
    template = get_template('my_qwen2_5_omni', processor)
    data = {
        'messages': [
            {'role': 'user', 'content': '描述视频<video>与图片<image>内容。'},
            {'role': 'assistant', 'content': '一个小孩和一只猫咪。'},
        ],
        'videos': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'],
        'images': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'],
    }
    template.set_mode('train')
    encoded = template.encode(data)
    print('input_ids: ' + template.safe_decode(encoded['input_ids']))
    print('labels: ' + template.safe_decode(encoded['labels']))
    print('keys: ' + str(encoded.keys()))
```


## 推理对齐
接下来，你需要进行PtEngine与transformers的推理对齐。通常你需要对齐`input_ids`以及输出内容。你可以书写以下测试函数：

```python
import os
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from modelscope import snapshot_download
from swift.llm import PtEngine, InferRequest, RequestConfig
import requests

def infer_hf():
    model_dir = snapshot_download('Qwen/Qwen2.5-Omni-7B')
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto", attn_implementation='flash_attention_2')
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    # 使用decord读取视频（暂不支持url）
    resp = requests.get('https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4')
    with open('_baby.mp4', 'wb') as f:
        f.write(resp.content)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "_baby.mp4"},
                {"type": "image", "image": "http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"},
                {"type": "text", "text": "描述视频和图像。"},
            ],
        },
    ]

    USE_AUDIO_IN_VIDEO = False
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                       use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, thinker_do_sample=False,
                              return_audio=False)
    text = processor.batch_decode(text_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return inputs['input_ids'][0].tolist(), text[0]

def test_my_qwen2_5_omni():
    engine = PtEngine('Qwen/Qwen2.5-Omni-7B', model_type='my_qwen2_5_omni', attn_impl='flash_attention_2')
    infer_request = InferRequest(messages=[{
        "role": "user",
        "content": "<video><image>描述视频和图像。",
    }],
        videos=["https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4"],
        images=["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"],
    )
    request_config = RequestConfig(temperature=0, max_tokens=512)
    input_ids = engine.default_template.encode(infer_request)['input_ids']
    resp_list = engine.infer([infer_request], request_config)
    resp = resp_list[0].choices[0].message.content
    return input_ids, resp


if __name__ == '__main__':
    # 开启debug模式，会打印`PtEngine.infer`的input_ids和generate_ids
    os.environ['SWIFT_DEBUG'] = '1'
    input_ids_hf, response_hf = infer_hf()
    input_ids_swift, response_swift = test_my_qwen2_5_omni()
    # 测试input_ids和response对齐
    assert input_ids_hf == input_ids_swift
    assert response_hf == response_swift
```


## 开始训练

使用python代码训练，这通常更容易debug：
```python
from swift.llm import sft_main, TrainArguments
import os
if __name__ == '__main__':
    os.environ['MAX_PIXELS'] = '1003520'
    sft_main(TrainArguments(
        model='Qwen/Qwen2.5-Omni-7B',
        dataset='AI-ModelScope/LaTeX_OCR#5000',
        model_type='my_qwen2_5_omni',
        template='my_qwen2_5_omni',
        load_from_cache_file=True,
        split_dataset_ratio=0.01,
        train_type='lora',
        torch_dtype='bfloat16',
        attn_impl='flash_attn',
        padding_free=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-4,
        lora_rank=8,
        lora_alpha=32,
        target_modules='all-linear',
        freeze_vit=True,
        freeze_aligner=True,
        gradient_accumulation_steps=1,
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        logging_steps=5,
        max_length=2048,
        output_dir='output',
        warmup_ratio=0.05,
        dataloader_num_workers=4,
        dataset_num_proc=1,
    ))
```

使用命令行训练：
```shell
# 4 * 35GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --model_type my_qwen2_5_omni \
    --template my_qwen2_5_omni \
    --custom_register_path 'examples/custom/my_qwen2_5_omni/my_register.py' \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#2000' \
              'swift/VideoChatGPT:all#2000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --padding_free true \
    --packing true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 1 \
    --deepspeed zero2
```

训练后对验证集进行推理：（环境变量请与训练时对齐）
```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
swift infer \
   --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --max_new_tokens 2048 \
    --load_data_args true
```

使用以下命令将训练权重推送到 Modelscope：
```shell
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>'
```
