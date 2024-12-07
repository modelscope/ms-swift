# Frequently-asked-questions

Here are some common questions encountered during the use of Swift.

## Training

### Q1: What models and datasets are supported for fine-tuning in Swift?
Please refer to the documentation on [Supported Models and Datasets](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html).

### Q2: What data formats are supported when training with custom datasets?
For LLM custom dataset formats, see the documentation on [Customization and Expansion](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.html). For VLM custom dataset formats, different models have different support for the number of images. Refer to the corresponding best practice documentation in [Multi-Modal documentation](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/index.html).

### Q3: What is the format for dataset_info.json for custom datasets, and how can I use it?
The dataset_info.json format can be found in the documentation on [Customization and Expansion](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.html). Use the command line with `--custom_dataset_info xxx.json`, `--dataset dataset_name`.

### Q4: How can I train with a custom dataset using the interface?
Using a custom dataset through the interface is the same as using the command line. Refer to the documentation on [Customization and Expansion](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.html).

### Q5: Can I write a line in the jsonl file like this? {"index": "00000", "query": "11111", "response": "22222", 'source':'qqq'}
You can have additional fields that won't be used.

### Q6: Where can I find the command line parameters?
Please refer to the documentation on [Command Line Parameters](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html).

### Q7: What parameters need to be configured for training in an offline environment?
Use `--model_id_or_path local_path`, `--check_model_is_latest false`. For more details, see the [Command Line Parameters documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html).

### Q8: Where can I check model_type?
Check the [Supported Models and Datasets documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html).

### Q9: Can I directly convert the model to gguf format after training?
Currently, only export to ModelFile is supported. See the [OLLaMA Export documentation](https://swift.readthedocs.io/zh-cn/latest/LLM/OLLAMA%E5%AF%BC%E5%87%BA%E6%96%87%E6%A1%A3.html).

### Q10: Does Swift support pre-training? I only see SFT.
Yes, it supports it. Use the command line `swift pt`. The dataset format is detailed in [Customization and Expansion](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.html).

### Q11: For models fine-tuned with LoRA, should I merge them into one model for resuming training, or can I specify the original model and LoRA block by path directly?
You do not need to merge. Use `--resume_from_checkpoint output/xxx/vx-xxx/checkpoint-xxx`. See the [Command Line Parameters documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html).

### Q12: I would like to control the location where the original model weights downloaded from the internet are stored. How can I place the original model in a specific folder?
You can set the environment variable `MODELSCOPE_CACHE=your_path` to store the original model in the specified path. For SDK downloads, use `cache_dir="local_path"`. You can also use the `modelscope download` command-line tool or `git` to download it. For details, refer to the [Model Download documentation](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E4%B8%8B%E8%BD%BD). During training, set `--model_id_or_path` to the local path. For offline training, configure `--check_model_is_latest false`. See the [Command Line Parameters documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html).

### Q13: Has anyone encountered this issue with ms-swift?
```text
[rank6]: pydantic_core._pydantic_core.ValidationError: 1 validation error for DeepSpeedZeroConfig
[rank6]: stage3_prefetch_bucket_size
[rank6]: Input should be a valid integer, got a number with a fractional part [type=int_from_float,input_value=11560550.4，in put_type=float]
[rank6]: For further information visit https://errors.pydantic.dev/2.8/v/int_fro_float
```
Downgrade `deepspeed` to `0.14.*`.

### Q14: Is there a complete tutorial and command line for fine-tuning Qwen-2-VL?
See [Qwen2-VL Best Practices](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/qwen2-vl%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html).

### Q15: Are there any tricks supported for fine-tuning multi-modal large models, similar to the LLM's neftune?
You can try variations of `lora` like `piassa/olora/dora`, or `fourierft`. Refer to the tricks in the `sft` parameters, as some may not apply to multi-modal.

### Q16: The accuracy from eval during training and the accuracy computed from re-inference with the saved checkpoint are not consistent.
The methods for calculating eval accuracy during training and inference are different. The default `acc_strategy` is `token`, and the selectable values are: `token`, `sentence`.

### Q17: Official Magic Mirror image and Swift environment.
You can start the container using `docker run`, for example: `docker run --gpus all -p 8000:8000 -it -d --name ms registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.3.0-tf2.16.1-1.16.0 /bin/bash`. After starting the container, pull the latest code to install Swift.

### Q18: Command line for multi-machine multi-card training.
```shell
# Multi-machine multi-card
# If not using a shared disk, please additionally specify `--save_on_each_node true` on each machine sh.
# node0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \
# node1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \
```
For more details, see the [LLM Fine-Tuning Documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.html).

### Q19: How to choose a template?
See [issue](https://github.com/modelscope/ms-swift/issues/1813).

### Q20: How to use torchrun and swift sft for multi-card training?
`swift sft` uses `torchrun`.

### Q21: I have a question about my SFT dataset being too large; tokenizing takes a long time. Is there a solution?
Use `lazy_tokenize`. See [Command Line Parameters documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html).

### Q22: When two datasets are simply appended together in the training set, does the model shuffle internally during training, or does it take data in order to train?
The trainer will shuffle randomly.

### Q23: If the model is on two cards and the data is not parallelized, deepspeed will throw an error. How to handle this?
`deepspeed` and `device_map` are incompatible; you can only choose one.

### Q24: Why does it need to download again when retraining offline, despite having already downloaded the dataset online?
The data file contains URLs, which do not support offline training.

### Q25: How to reduce GPU memory usage when training VLM models?
Set `--freeze_vit true`.

### Q26: Why are there fewer models supported in the WEB-UI interface than in the documentation?
Upgrade `ms-swift`.

### Q27: For models that do not have a suitable model_type, can I customize special_tokens and chat_template during SFT?
Yes, you can. Refer to the PR for model integration and the custom model dataset documentation.

### Q28: Can I use DPO to train Qwen2-VL in a Python script?
Yes, import `rlhf_main` and `RLHFArguments` from `swift.llm`.

### Q29: Can I pre-train with pure text before fine-tuning on a VQA dataset for MLLM?
Yes, you can mix training as well.

### Q30: When conducting DPO training based on the qwen2 SFT model on a V100 machine, the training shows NaN?
Use fp32 for training with the V100 machine.

### Q31: Does Swift support distillation?
No, it is not supported. It is recommended to use quantization for better results.

### Q32: Encountered the error `cannot import name 'ftp_head' from 'datasets.utils.file_utils'`. Has anyone faced this issue?
Try `pip install datasets==2.*`.

### Q33: The default maximum number of checkpoints saved after training is two. How can I increase this number?
Use `--save_total_limit`. See the [Command Line Parameters documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html).

### Q34: In grounding tasks, does the universal data format support multiple instances for one category?
Currently, it supports one object corresponding to multiple bounding boxes. Refer to the documentation on [InternVL Best Practices](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/internvl%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html).

### Q35: Why am I getting the error that numpy.object cannot be found?
Try using `numpy==1.26.3`.

### Q36: Does the Swift framework support sequence parallelism now?
Yes, it supports it. It implements this using `xtuner`.

### Q37: When fine-tuning qwen2-1.5B on a V100, I see `loss': 0.0, 'acc': 0.0, 'grad_norm': nan`. What is the issue?
Try using fp32.

### Q38: Is it possible to fully fine-tune GPTQ quantized models?
No, GPTQ model's int-type parameters cannot participate in gradients; they can only be updated with additional structures like LoRA.

### Q39: What parameters should I set for fine-tuning using QLoRA on glm4-chat?
Set the parameters `--quantization_bit 4`. Refer to the QLoRA [example](https://github.com/modelscope/ms-swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat).

### Q40: I encounter the issue "AdamW' object has no attribute 'train" when training my dataset on qwen2-vl-7b.
Try `accelerate 0.34.0`.

### Q41: How do I expand my vocabulary within the Swift framework?
Swift currently does not support vocabulary expansion.

### Q42: Can I directly use models with the same name from Hugging Face?
Set the environment variable `USE_HF=1`.

### Q43: Can Qwen2-VL-2B conduct incremental pre-training? Is there guidance available?
Yes, it supports incremental pre-training. Just include all the content in the response.

### Q44: When training with videos, how can I control the frame sampling rate in the parameters? The `frame_rate` setting doesn't seem to work, and I'm using MiniCPMV.
Set the environment variable `MAX_NUM_FRAMES`.

### Q45: Can I save the inference results of the validation set during training in Swift?
After training, run `swift infer` to save the results.

### Q46: Why is the saved checkpoint larger than the original model file after full parameter DPO?
Using V100 for fine-tuning stores the data in fp32 format.

### Q47: Training speed slows down when using multi-machine training; using Swift framework for LLM training with deepspeed zero3 causes significant performance drop.
See the [issue](https://github.com/modelscope/ms-swift/issues/1825).

### Q48: Does Swift now support multi-stage pre-training for qwen2-vl? It looks like the official best practices only show SFT training with vit+llm together, not sure if separate fine-tuning is supported.
Refer to the [issue](https://github.com/modelscope/ms-swift/issues/2222).

### Q49: Does qwen2-vl support mixing pure text data?
It supports both mixed visual-text and pure text data.

### Q50: Can I plot loss curves for different datasets during fine-tuning?
This is not supported; datasets are trained in a mixed manner.

### Q51: After model training, the responses have a lot of repeated content.
Refer to the [LLM Fine-Tuning documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.html). If you notice repetitions during training, try training for more epochs, cleaning the data, and conducting full parameter training, using RLHF to mitigate this issue.

### Q52: Does Swift currently support prompt tuning or prefix tuning?
No, it does not support these methods, as both methods suffer from serious forgetting issues and are not currently recommended.

### Q53: I encountered the following error when training with two A10s:
```text
[rank0]: torch.distributed.DistBackendError: NCCL error in:../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1970， unhandled system error (run with NCCL_DEBUG=INFO for details),NCCL version 2.20.5
[rank0]:ncclSystemError: System call (e.g. socket,malloc) or external library call failed or device error.
```
Please check if shared memory is too small; NCCL requires shared memory.

### Q54: How to solve the issue of certain parameters not participating in backpropagation when freezing layers during DDP fine-tuning?
Set the parameter `--ddp_find_unused_parameters true`.

### Q55: Does Swift have a dataset quality inspection tool?
[data-juicer](https://github.com/modelscope/data-juicer).

### Q56: Where to start model parallelism on the web? I only found the option to check for data parallelism.
You can specify visible GPUs to enable model parallelism.

### Q57: How can I turn off automatic shuffling?
Currently, you can only modify the [transformers code](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py).

### Q58: What is the parameter 'num_items_in_batch'? I can't find it.
Upgrade to `ms-swift==2.5.2` or downgrade to `transformers<4.46`.

## Inference

### Q1: Is there documentation for Swift inference?
Swift supports inference via Python scripts, command line, and UI interface. See the [LLM Inference documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/LLM%E6%8E%A8%E7%90%86%E6%96%87%E6%A1%A3.html).

### Q2: How to use the trained model for inference with a dataset?
Use the parameters `--load_dataset_config true` or `--val_dataset <your-val-dataset>`. Refer to the [LLM Fine-Tuning documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.html#%E5%BE%AE%E8%B0%83%E5%90%8E%E6%A8%A1%E5%9E%8B).

### Q3: Can I specify a locally saved model during Swift inference?
Set `--model_id_or_path` to the local path. See [Command Line Parameters documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html).

### Q4: How do I infer on a dataset without labels? I see that the dataset format in the documentation is all for the training set.
Configure the parameter `--val_dataset <your-val-dataset>`.

### Q5: How to resolve the error `ValueError: Input length of input_ids is 35, but max_length is set to 20`?
```text
raise ValueError(
ValueError: Input length of input_ids is 35, but `max_length` is set to 20. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
```
Set `model.generation_config.max_new_tokens`.

### Q6: How to handle GPU memory explosion during Qwen2-VL inference?
Set environment variables `SIZE_FACTOR=8 MAX_PIXELS=602112`. See the [Qwen2-VL Best Practices documentation](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/qwen2-vl%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html).

### Q7: On a V100 GPU, in a Python virtual environment, following the environment setup instructions from https://github.com/modelscope/ms-swift/blob/main/docs/source/Multi-Modal/qwen2-vl%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md, when testing the inference command: `CUDA_VISIBLE_DEVICES=0,1,2,3 swift infer --model_type qwen2-vl-7b-instruct`, an error occurs: `RuntimeError: probability tensor contains either 'inf', 'nan' or element < 0`.
Try inference on A10 or 3090 machines.

### Q8: After running the prediction command, where are the results saved? CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir output/glm4v-9b-chat/vx-xxx/checkpoint-xxx-merged --load_dataset_config true
Results will be printed in the log.

### Q9: How can I get output logits when calling inference?
Refer to the [test_logprobs.py](https://github.com/modelscope/ms-swift/blob/main/tests/custom/test_logprobs.py).

### Q10: In the latest version of Swift, while loading the qwen2-32b-instruct-awq quantized model, I was advised to add merge-lora true. After doing this, it throws an error. When I omit it, inference works but slowly.
Models trained with QLoRA do not support merge-lora; it is recommended to merge-lora after fine-tuning and then quantize.

### Q11: Getting the error `assert factor in rope_scaling` with vllm?
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830, for more details, see qwen2-vl[issue#96](https://github.com/QwenLM/Qwen2-VL/issues/96).

### Q12: Does vllm require the models to be merged before calling them during inference?
Models do not have to be merged. See the documentation on [VLLM Inference Acceleration and Deployment](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#id11).

### Q13: When using inference, can I get the predicted probabilities without using inference_client functions? Can I obtain results using the inference function from the single sample inference demo?
Modify `generation_config.output_logits`.
`model.generation_config.output_logits = True`
`model.generation_config.return_dict_in_generate = True`

### Q14: Has anyone encountered the error `RuntimeError: "triu_tril_cuda_template" not implemented for'BFloat16'`?
Upgrade Torch, as the current version may not have implemented this operator.

### Q15: Does qwen2-audio support streaming inference?
Yes, see the [issue](https://github.com/modelscope/ms-swift/issues/1653).

### Q16: Where to set `do_sample` for multi-modal inference using inference client?
Set `temperature=0`.

### Q17: Does ms-swift support batch processing for large models?
Supported. When inferring with a Python script, the `request_list` in the documentation can contain multiple queries, and the server will automatically batch process them during deployment. See [VLLM Inference Acceleration and Deployment](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#id3).

### Q18: When quantizing models with ms-swift, there is an insufficient memory display. Can we reduce resource usage during quantization, even if it's slower?
Try setting `--quant_device_map cpu`.

### Q19: Does Swift support quantization for multi-modal models?
Yes, it supports quantization.

### Q20: Encountering the following error while using GPTQ, what is the cause?
```text
if llm_config['architectures'][0] == 'LlamaForCausalLM':
KeyError: 'architectures'
```
Try using `transformers==4.44.*`.

### Q21: How can I specify where to save evaluation results during swift infer? I can't find where the results are saved.
Set `--result_dir your_path`. See [InferArguments](https://github.com/modelscope/ms-swift/blob/main/swift/llm/utils/argument.py).

### Q22: I get an error while using AWQ quantized yi-vl-6b:
```text
TypeError: swift.llm.utils.model.get_model_tokenizer_with_flash_attn() got multiple values for keyword argument 'automodel_class'.
```
Please use GPTQ quantization.

### Q23: I would like to ask about using swift export to perform GPTQ INT4 quantization on the qwen2.5 72B model with a max model length of 32768, which is the default value. The calibration dataset provided has 128 samples, but an error occurred during quantization. The error log is: "factorization could not be completed because the input is not positive-definite (the leading minor of order 18145 is not positive-definite)." What is the cause?
This indicates a problem with the Hessian matrix being non-positive definite. Try using a different dataset.

## Deployment

### Q1: How to deploy a trained model?
Use `swift deploy --ckpt_dir xxx`. Refer to the documentation on [VLLM Inference Acceleration and Deployment](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html).

### Q2: How to use vllm for multi-card deployment?
Use `RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1,2,3 swift deploy --model_type qwen-7b --tensor_parallel_size 4`. Refer to the documentation on [VLLM Inference Acceleration and Deployment](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html).

### Q3: How can clients pass images during vllm deployment?
See the multi-modal documentation, [vLLM Inference Acceleration Documentation](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/vLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E6%96%87%E6%A1%A3.html).

### Q4: I have a question about deploying qwen2-7b and using it with a client. When calling the OpenAI API, should I use `client.completions.create` instead of `client.chat.completions.create`, but when using `qwen2-7b-instruct-q5_k_m.gguf`, I can use `client.chat.completions.create`. Why is that?
The base model can use `client.chat.completions.create`, but this is a compatibility behavior.

### Q5: Q5: After launching the server with swift deploy using two cards, when I exit with Ctrl+C, there is always a Python process that continues to occupy the memory of one card. Is this a normal phenomenon?
You may need to kill it. This is an issue with vllm.

### Q6: Where to check if a model supports lmdeploy or vllm acceleration?
Refer to the documentation on [Supported Models and Datasets](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html).

### Q7: Why does Tongyi Qianwen 2.5-Math-7B-Instruct sometimes return garbled characters when using vllm deployment? Using vllm to deploy,fp16
Try using bf16.

### Q8: After LoRA fine-tuning, I encountered the error `requests.exceptions.HTTPError: Multimodal model only support default-lora` when using Swift's inference.
Set `model_type` to `default-lora`.

### Q9: After starting the swift inference service, how can I set configurations like temperature interactively?
Inference only has preset configurations at startup, while deployment can set defaults initially and allow overriding them later on the client side.

### Q10: When deploying qwen2vl model locally, how can I input videos during inference? Can I use base64? How to call video with curl?
Check the [MLLM Deployment documentation](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/MLLM%E9%83%A8%E7%BD%B2%E6%96%87%E6%A1%A3.html). URL, base64, and local paths are all usable, but local paths are limited to local testing.

### Q11: When deploying qwen2-vl, I encounter an error about the vllm version not being correct?
```text
Unrecognized keys in `rope_scaling`for 'rope_type'='default': {'mrope_section'}
```
Refer to the [issue](https://github.com/QwenLM/Qwen2-VL/issues/209).

### Q12: Can Swift inference output prediction probabilities? How to set this during deployment?
During Python script inference, configure `model.generation_config.output_logits = True, model.generation_config.return_dict_in_generate = True`. During deployment, clients send parameters `logprobs=True, top_logprobs=5`.

## Evaluation

### Q1: What evaluation datasets are supported by Swift?
Pure text evaluation:
```text
'obqa', 'cmb', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada',
'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze',
'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval',
'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench',
'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```

Multimodal evaluation:
```text
'COCO_VAL', 'MME', 'HallusionBench', 'POPE', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN',
'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11',
'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2',
'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL',
'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar',
'RealWorldQA', 'MLLMGuard_DS', 'BLINK', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'DocVQA_VAL',
'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST', 'ChartQA_TEST', 'MathVision', 'MathVision_MINI',
'MMMU_DEV_VAL', 'MMMU_TEST', 'OCRBench', 'MathVista_MINI', 'LLaVABench', 'MMVet', 'MTVQA_TEST',
'MMLongBench_DOC', 'VCR_EN_EASY_500', 'VCR_EN_EASY_100', 'VCR_EN_EASY_ALL', 'VCR_EN_HARD_500',
'VCR_EN_HARD_100', 'VCR_EN_HARD_ALL', 'VCR_ZH_EASY_500', 'VCR_ZH_EASY_100', 'VCR_ZH_EASY_ALL',
'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMDU', 'MMBench-Video', 'Video-MME',
'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN',
'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11',
'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST',
'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL',
'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK'
```

See the document [LLM Evaluation Documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/LLM%E8%AF%84%E6%B5%8B%E6%96%87%E6%A1%A3.html) for details.

### Q2: How to use a custom evaluation dataset?
Custom evaluation datasets, both plain text and multimodal, must match the data format (pattern) of an official dataset. See the document [LLM Evaluation Documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/LLM%E8%AF%84%E6%B5%8B%E6%96%87%E6%A1%A3.html) for details.

### Q3: Error with mmengine in python3.11 environment during evaluation
Try using the Python 3.10 environment. Or first install all dependencies:
`pip3 install evalscope[all]`,
then apply the patch:
`pip3 install https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/package/evalscope-0.5.3.post1-py3-none-any.whl`.

### Q4: After manually downloading the official evaluation dataset, can Swift eval be configured for local path evaluation?
First, download the evaluation dataset [eval.zip](https://modelscope.cn/datasets/swift/evalscope_resource/files), extract it, and place its contents in the `~/.cache/modelscope/media_resources/evalscope/data` folder. Then execute the `swift eval` command to use the local data.

### Q5: Is there a bug with custom evaluation? I modified the standard example to English, but it doesn't work?
```shell
swift eval --model_type 'qwen2_5-1_5b-instruct' --eval_dataset no --custom_eval_config '/mnt/workspace/test_data/config_eval.json'
```
This relies on the nltk package, which needs to download a punkt_tab zip file. Some environments in China have unstable or failed downloads. The code has been modified to handle this issue; reference [issue](https://github.com/nltk/nltk/issues/3293).

### Q6: The model after eval fine-tuning keeps stopping at a fixed percentage, but the vllm service seems to be running normally. The larger the model, the sooner it disconnects.
Set the `TIMEOUT` environment variable to -1.

### Q7: Does evalscope support multi-model comparison?
See the [documentation](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html) for details.

### Q8: Is there a custom evaluation for multimodal datasets?
Custom evaluation for multimodal datasets can be referenced in the [documentation](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset.html#vlm).

### Q9: Does ms-swift have methods to test QPS, latency, and tokens/s?
You can try using evalscope's [model stress testing tool](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test.html#id1).

### Q10: Can I control the number of dataset entries during evaluation? It takes over an hour to evaluate an MMLU, which is too slow.
Use the configuration parameter `--eval_limit`. This `--eval_limit` controls the number of entries in each subset. For example, if MMLU has over 50 subsets, and each limit is set to 10 entries, then that would be over 500 entries in total.
