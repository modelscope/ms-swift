# Megatron-SWIFT 自定义模型


这里介绍如何在Mcore-Bridge中注册模型，以支持新模型在Megatron-SWIFT中的训练。我们将以MiniMax-M2.7为例子介绍。

## 下载模型

首先，你需要下载模型配置。

```python
from swift import safe_snapshot_download

model_dir = safe_snapshot_download('MiniMax/MiniMax-M2.7', download_model=False)
print(f'model_dir: {model_dir}')
```

由于模型权重很大，为了加速支持模型的效率，我们采用懒下载的方式，并只下载`num_layers`层的权重，构建mini版本的模型，用于做接入测试。以MiniMax-M2.7为例，我们构建了一层的BF16版本的权重。若有些模型出现前3层为Dense，之后为MoE，则你可以构建4层的权重。

```python
import os
import torch
from modelscope.hub.file_download import model_file_download
from safetensors.torch import safe_open
from swift import safe_snapshot_download

from mcore_bridge.utils import Fp8Dequantizer, SafetensorLazyLoader, StreamingSafetensorSaver

model_id = 'MiniMax/MiniMax-M2.7'
# 有些模型会出现前几层为dense，后面为moe的情况，需合理设置该值
num_layers = 1  # 只下载`num_layers`层，节约磁盘占用和运行时显存占用
model_dir = safe_snapshot_download(model_id, download_model=False)

loader = SafetensorLazyLoader(model_dir)
state_dict = loader.get_state_dict()
saver = StreamingSafetensorSaver(save_dir=model_dir)
new_state_dict = {}
fp8_dequantizer = Fp8Dequantizer()  # 用于将fp8权重转成bf16


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


SafetensorLazyLoader._open_file = _open_file  # monkey patch (懒下载)

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

保存完权重后，你需要修改'config.json'，将`num_hidden_layers`修改为1（与上面的代码对应），并删除`quantization_config`配置（因为权重为BF16的，而不是FP8）。FP8的训练大多数模型会自动适配，但有些模型可能需要额外适配，例如：Qwen3.5的FP8的适配参考[这个PR](https://github.com/modelscope/mcore-bridge/pull/30)。


## 注册模型

以下提供debug代码，你需要修改代码，以确保huggingface transformers库的forward与megatron的forward对齐。
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

'minimax_m2'的注册可以查看[这个文件](https://github.com/modelscope/mcore-bridge/blob/main/src/mcore_bridge/model/gpts/minimax_m2.py)。我们注册时指定了模型对应的GPTBridge类和模型加载器loader。

```python
register_model(ModelMeta(
    ModelType.minimax_m2,
    ['minimax_m2'],
    bridge_cls=MinimaxM2Bridge,
    loader=MinimaxM2Loader,
))
```

参数的总和对齐：
```
[INFO:swift] n_parameter: 522
[INFO:swift] total_sum: 106747128.72671509
[INFO:swift] zero_count: 0
[INFO:swift] n_parameter: 780
[INFO:swift] total_sum: 106747129.32046509
[INFO:swift] zero_count: 0
```

模型forward的logits对齐。（当然我们还需要对模型进行训练，训练后再测试forward的精度，避免出现这里输出tokens都是同一个的情况）。
```
mean_diff: 2.8353377274470404e-05, max_diff: 0.0015382766723632812
mean_diff (with loss): 2.1664049199898727e-05, max_diff (with loss): 0.00021076202392578125 (Please check that mean_diff (with loss) is less than 0.1).
hf_tokens: [190962, 103239, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367]
mg_tokens: [190962, 103239, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367]
token_diff: 0
token_diff (with loss): 0
```

通常在参数总数对齐和输出logits对齐后，模型就基本接入成功了。此外你可能还需要适配TP/CP的情况。你可以使用以下代码debug：
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
            to_mcore=True,  # 也可以修改成 `to_hf=True` 测试
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            expert_model_parallel_size=2,
            test_convert_precision=True,
        ))
```

我们需要用torchrun启动，vscode配置：
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

其他模型的注册例子，可以查看对应PR：[hy_v3](https://github.com/modelscope/mcore-bridge/pull/53)、[kimi_25](https://github.com/modelscope/mcore-bridge/pull/52)。在2026年4月之前的接入PR可以在ms-swift库中寻找。


## 测试准确性

我们对mini版本的模型进行训练，我们只使用自我认知数据集，并训练到过拟合。

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

进行推理，查看训练效果：
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


再次测试forward精度对齐：
```
mean_diff: 0.0005969047779217362, max_diff: 0.013172879815101624
mean_diff (with loss): 0.0005803848034702241, max_diff (with loss): 0.009410381317138672 (Please check that mean_diff (with loss) is less than 0.1).
hf_tokens: [190962, 190962, 367, 44, 46, 2362, 5129, 6415, 75827, 343, 10, 1497, 71151, 11915, 1497, 44, 3003, 44, 46, 46, 4387, 10, 32, 10, 258, 1497, 44, 46, 46, 258, 18268, 44, 692, 13268, 42047, 3764, 46, 46, 46, 94454, 46, 46, 275, 296, 3786, 46, 46, 275, 46, 46, 3786, 46, 2329, 10, 722]
mg_tokens: [190962, 190962, 367, 44, 46, 2362, 5129, 6415, 75827, 343, 10, 1497, 71151, 11915, 1497, 44, 3003, 44, 46, 46, 4387, 10, 32, 10, 258, 1497, 44, 46, 46, 258, 18268, 44, 692, 13268, 42047, 3764, 46, 46, 46, 94454, 46, 46, 275, 296, 3786, 46, 46, 275, 46, 46, 3786, 46, 2329, 10, 722]
token_diff: 0
token_diff (with loss): 0
```

至此，模型接入成功啦！


## 提交PR

如果你想给ms-swift/mcore-bridge提交PR，你需要额外运行以下命令，对代码进行整理：

```shell
pip install pre-commit
pre-commit run --all-files
```
