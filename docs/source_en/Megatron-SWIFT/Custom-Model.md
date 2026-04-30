# Custom Megatron Model


This guide explains how to register a model in [Mcore-Bridge](https://github.com/modelscope/mcore-bridge) to support training new models in Megatron-SWIFT. We will use MiniMax-M2.7 as an example.

## Download the Model

First, you need to download the model configuration.

```python
from swift import safe_snapshot_download

model_dir = safe_snapshot_download('MiniMax/MiniMax-M2.7', download_model=False)
print(f'model_dir: {model_dir}')
```

Since model weights are very large, to speed up the model integration process, we use lazy downloading and only download weights for `num_layers` layers, building a mini version of the model for integration testing. Taking MiniMax-M2.7 as an example, we build a one-layer BF16 version of the weights. If some models have the first 3 layers as Dense and the rest as MoE, you can build 4 layers of weights. If alternating attention types are used, for example Qwen3.5 alternates between linear attention and full attention, you will also need more layers.

```python
import os
import torch
from modelscope.hub.file_download import model_file_download
from safetensors.torch import safe_open
from swift import safe_snapshot_download

from mcore_bridge.utils import Fp8Dequantizer, SafetensorLazyLoader, StreamingSafetensorSaver

model_id = 'MiniMax/MiniMax-M2.7'
# Some models have the first few layers as dense and the rest as MoE; set this value accordingly
num_layers = 1  # Only download `num_layers` layers to save disk space and runtime GPU memory
model_dir = safe_snapshot_download(model_id, download_model=False)

loader = SafetensorLazyLoader(model_dir)
state_dict = loader.get_state_dict()
saver = StreamingSafetensorSaver(save_dir=model_dir)
new_state_dict = {}
fp8_dequantizer = Fp8Dequantizer()  # Used to convert fp8 weights to bf16


def _open_file(self, filename: str):
    if filename not in self._file_handles:
        file_path = os.path.join(self.hf_model_dir, filename)
        tmp_dir = os.path.join(self.hf_model_dir, 'tmp')
        if not os.path.exists(file_path):
            file_path = os.path.join(tmp_dir, filename)
        if not os.path.exists(file_path):
            file_path = model_file_download(
                model_id=model_id,
                file_path=filename,
                local_dir=tmp_dir,
            )
        self._file_handles[filename] = safe_open(file_path, framework='pt')
    return self._file_handles[filename]


SafetensorLazyLoader._open_file = _open_file  # monkey patch (lazy downloading)

for k, v in state_dict.items():
    if k.startswith('model.layers.'):
        idx = int(k[len('model.layers.'):].split('.', 1)[0])
        if idx >= num_layers:
            continue
        if k.endswith('.weight_scale_inv'):
            continue
        elif k.endswith('.weight'):
            weight_scale_inv = k.replace('.weight', '.weight_scale_inv')
            if weight_scale_inv in state_dict:
                v = fp8_dequantizer.convert(v.load(), state_dict[weight_scale_inv].load()).to(torch.bfloat16)
    new_state_dict[k] = v if isinstance(v, torch.Tensor) else v.load()

for k, v in new_state_dict.items():
    saver.add_tensor(k, v)
saver.finalize()
```

After saving the weights, you need to modify `config.json`: change `num_hidden_layers` to 1 (corresponding to the code above), and remove the `quantization_config` section (since the weights are in BF16, not FP8). FP8 training is automatically adapted for most models, but some models may require additional adaptation. For example, refer to [this PR](https://github.com/modelscope/mcore-bridge/pull/30) for FP8 adaptation of Qwen3.5.


## Register the Model

The following provides debug code. You need to modify the code to ensure that the forward pass of the HuggingFace Transformers library aligns with Megatron's forward pass.
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['SWIFT_TEST_CONVERT_PRECISION'] = '1'

from swift import export_main, ExportArguments, safe_snapshot_download

model_id = 'MiniMax/MiniMax-M2.7'

model_dir = safe_snapshot_download(model_id, download_model=False)

export_main(
    ExportArguments(
        model=model_dir,
        to_mcore=True,
        exist_ok=True,
        test_convert_precision=True,
        torch_dtype='bfloat16',
    ))
```

The registration of minimax_m2 can be found in [this file](https://github.com/modelscope/mcore-bridge/blob/main/src/mcore_bridge/model/gpts/minimax_m2.py). During registration, we specify the corresponding GPTBridge class and model loader for the model.

```python
register_model(ModelMeta(
    ModelType.minimax_m2,
    ['minimax_m2'],
    bridge_cls=MinimaxM2Bridge,
    loader=MinimaxM2Loader,
))
```

Parameter total sum alignment:
```
[INFO:swift] n_parameter: 522
[INFO:swift] total_sum: 106747128.72671509
[INFO:swift] zero_count: 0
[INFO:swift] n_parameter: 780
[INFO:swift] total_sum: 106747129.32046509
[INFO:swift] zero_count: 0
```

Model forward logits alignment. (Of course, we also need to train the model and then test the forward precision afterwards, to avoid cases where the output tokens are all the same.)
```
mean_diff: 2.8353377274470404e-05, max_diff: 0.0015382766723632812
mean_diff (with loss): 2.1664049199898727e-05, max_diff (with loss): 0.00021076202392578125 (Please check that mean_diff (with loss) is less than 0.1).
hf_tokens: [190962, 103239, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367]
mg_tokens: [190962, 103239, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367]
token_diff: 0
token_diff (with loss): 0
```

Usually, once the total parameter count and output logits are aligned, the model integration is essentially successful. Additionally, you may need to adapt for TP/CP scenarios. You can use the following code to debug:
```python
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['SWIFT_TEST_CONVERT_PRECISION'] = '1'

from swift.megatron import MegatronExportArguments, megatron_export_main
from swift import safe_snapshot_download
model_id = 'MiniMax/MiniMax-M2.7'

model_dir = safe_snapshot_download(model_id, download_model=False)

if __name__ == '__main__':
    megatron_export_main(
        MegatronExportArguments(
            model=model_dir,
            to_mcore=True,  # Can also be changed to `to_hf=True` for testing
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            expert_model_parallel_size=2,
            test_convert_precision=True,
        ))
```

We need to launch with torchrun. VSCode configuration:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "torchrun2",
            "type": "debugpy",
            "request": "launch",
            "program": "-m",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "torch.distributed.run",
                "--nproc_per_node",
                "2",
                "--master_port", "29501",
                "${file}"
            ]
        },
    ]
}
```

For other model registration examples, refer to the corresponding PRs: [hy_v3](https://github.com/modelscope/mcore-bridge/pull/53), [kimi_25](https://github.com/modelscope/mcore-bridge/pull/52). Integration PRs before April 2026 can be found in the ms-swift repository.


## Test Accuracy

We train the mini version of the model using only the self-cognition dataset, training until overfitting.

```shell
# 2 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model /root/.cache/modelscope/models/MiniMax/MiniMax-M2.7 \
    --save_safetensors true \
    --dataset 'swift/self-cognition#500' \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 2e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --num_train_epochs 10 \
    --output_dir megatron_output \
    --save_steps 500 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --moe_permute_fusion true \
    --expert_model_parallel_size 2 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot
```

Run inference to check the training results:
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/v3-20260430-143926/checkpoint-310 \
    --max_new_tokens 64 \
    --enable_thinking false \
    --temperature 0
```

```
<<< 你是谁
我是一个由swift开发的人工智能助手，被称为swift-robot。我主要的目的是通过文本交流为用户提供帮助、信息和娱乐。如果您有任何疑问或需要帮助，请随时提出，我会尽力协助您。
--------------------------------------------------
<<< clear
<<< who are you
I am a language model developed by swift, you can call me swift-robot. How can I assist you?
--------------------------------------------------
```


Test forward precision alignment again:
```
mean_diff: 0.0005969047779217362, max_diff: 0.013172879815101624
mean_diff (with loss): 0.0005803848034702241, max_diff (with loss): 0.009410381317138672 (Please check that mean_diff (with loss) is less than 0.1).
hf_tokens: [190962, 190962, 367, 44, 46, 2362, 5129, 6415, 75827, 343, 10, 1497, 71151, 11915, 1497, 44, 3003, 44, 46, 46, 4387, 10, 32, 10, 258, 1497, 44, 46, 46, 258, 18268, 44, 692, 13268, 42047, 3764, 46, 46, 46, 94454, 46, 46, 275, 296, 3786, 46, 46, 275, 46, 46, 3786, 46, 2329, 10, 722]
mg_tokens: [190962, 190962, 367, 44, 46, 2362, 5129, 6415, 75827, 343, 10, 1497, 71151, 11915, 1497, 44, 3003, 44, 46, 46, 4387, 10, 32, 10, 258, 1497, 44, 46, 46, 258, 18268, 44, 692, 13268, 42047, 3764, 46, 46, 46, 94454, 46, 46, 275, 296, 3786, 46, 46, 275, 46, 46, 3786, 46, 2329, 10, 722]
token_diff: 0
token_diff (with loss): 0
```

At this point, the model integration is complete!


## Submit a PR

If you want to submit a PR to ms-swift/mcore-bridge, you need to additionally run the following commands to format the code:

```shell
pip install pre-commit
pre-commit run --all-files
```
