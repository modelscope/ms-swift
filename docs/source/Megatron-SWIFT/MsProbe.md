# 配置

按需修改ms-swift目录下msprobe_config.json文件中的dump_path、level等配置项
更多配置可参考[配置示例](https://gitcode.com/Ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/zh/dump/config_json_examples.md)和[配置文件介绍](https://gitcode.com/Ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/zh/dump/config_json_introduct.md)


# 代码修改
为了支持 msprobe 工具进行精度调试，我们需要修改 `swift/megatron/model/mm_gpt_model.py` 文件中的 `_patch_word_embeddings` 函数。主要改动是调整函数参数和内部实现逻辑，使其能够正确地对嵌入层进行patch

下面是具体的修改内容：

修改前：
```python
def _patch_word_embeddings(self, kwargs):
    origin_forward = VocabParallelEmbedding.forward

    def forward(_self, input_):
        from ..trainers.utils import split_cp_inputs
        args = get_args()
        reduce_scatter_embeddings = _self.reduce_scatter_embeddings
        _self.reduce_scatter_embeddings = False
        input_ = torch.masked_fill(input_, input_ < 0, 0)
        res = origin_forward(_self, input_)
        _self.reduce_scatter_embeddings = reduce_scatter_embeddings
        packed_seq_params = kwargs.get('packed_seq_params')
        # ...其他逻辑...
        return res
    VocabParallelEmbedding.forward = forward
    try:
        yield
    finally:
        VocabParallelEmbedding.forward = origin_forward

def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    decoder_input: torch.Tensor = None,
    labels: torch.Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    **kwargs,
) -> torch.Tensor:
    if decoder_input is not None:
        pass
    elif self.pre_process:
        kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
        with self._patch_word_embeddings(kwargs):
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

    # ...其他逻辑...
```

修改后：
```python
def _patch_word_embeddings(self, kwargs, emb):          # 修改1
    origin_forward = emb.word_embeddings.forward        # 修改2

    def forward(input_):                                # 修改3
        from ..trainers.utils import split_cp_inputs
        args = get_args()
        _self = emb.word_embeddings                     # 修改4
        reduce_scatter_embeddings = _self.reduce_scatter_embeddings
        _self.reduce_scatter_embeddings = False
        input_ = torch.masked_fill(input_, input_ < 0, 0)
        res = origin_forward(input_)                    # 修改5
        _self.reduce_scatter_embeddings = reduce_scatter_embeddings
        packed_seq_params = kwargs.get('packed_seq_params')
        # ...其他逻辑...
        return res
    
    emb.word_embeddings.forward = forward               # 修改6
    try:
        yield
    finally:
        emb.word_embeddings.forward = origin_forward    # 修改7

def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    decoder_input: torch.Tensor = None,
    labels: torch.Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    **kwargs,
) -> torch.Tensor:
    if decoder_input is not None:
        pass
    elif self.pre_process:
        kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
        with self._patch_word_embeddings(kwargs, self.language_model.embedding):                # 修改8
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

    # ...其他逻辑...
```

主要变化包括：
1. `_patch_word_embeddings` 方法增加了 `emb` 参数，用于接收 embedding 模块实例
2. 直接获取 `emb.word_embeddings.forward` 而不是 `VocabParallelEmbedding.forward`
3. 内部 `forward` 函数签名从 `(_self, input_)` 改为 `(input_)`
4. 在函数内部通过 `emb.word_embeddings` 获取 `_self`
5. 调用原始 forward 时直接传入 `input_`
6. 使用 `emb.word_embeddings.forward` 进行替换和恢复操作（修改6、7）
7. 在调用 `_patch_word_embeddings` 时传入 `self.language_model.embedding` 实例


# 使能
在启动脚本添加`--enable_msprobe True`

另外，由于msprobe不支持融合计算，还需要添加`--no_bias_dropout_fusion True`、`--no_bias_swiglu_fusion True`、`--cross_entropy_loss_fusion False`
## 示例
```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tensor_model_parallel_size 2 \
    ...
    --no_bias_dropout_fusion True \ 
    --no_bias_swiglu_fusion True \
    --cross_entropy_loss_fusion False \
    --enable_msprobe True
```