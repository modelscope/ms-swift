# Pre-training and Fine-tuning

Since pre-training and fine-tuning are quite similar, they will be discussed together in this section.

For the data format requirements for pre-training and fine-tuning, please refer to the section on [Custom Dataset](../Customization/Custom-dataset.md).

In terms of data requirements, the amount needed for continued pre-training can range from hundreds of thousands to millions of rows. Starting pre-training from scratch requires significantly more resources and data, which is beyond the scope of this article.
The data needed for fine-tuning can vary from a few thousand to a million rows. For lower data requirements, consider using RAG methods.

## Pre-training

Examples for pre-training can be found [here](https://github.com/modelscope/swift/blob/main/examples/train/pt/train.sh).

If you are pre-training using multiple machines and GPUs, please refer to the guide [here](https://github.com/modelscope/swift/blob/main/examples/train/multi-node).

The Megatron example is not yet officially supported but is expected to be available in the current iteration.

## Fine-tuning

Fine-tuning supports many lightweight training methods (note that these methods can also be used in pre-training, but full parameter training + DeepSpeed is generally recommended for pre-training).

You can find examples [here](https://github.com/modelscope/swift/blob/main/examples/train/tuners).

Additionally, other technologies and examples supported by SWIFT include:

- **ddp+device_map**: This is suitable for situations where a single GPU cannot handle the workload and DeepSpeed is not applicable. Refer to [this link](https://github.com/modelscope/ms-swift/blob/main/examples/train/multi-gpu/ddp_device_map/train.sh).
- **fsdp+qlora**: This allows training of 70B models on dual 3090 GPUs. Details can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu/fsdp_qlora/train.sh).
- **Multimodal Training**: For more information, see [here](https://github.com/modelscope/swift/blob/main/examples/train/multimodal).
- **Sequence Parallelism**: More details can be found [here](https://github.com/modelscope/swift/blob/main/examples/train/sequence_parallel).
- **Packing**: This combines multiple sequences into one, helping each sample to approach the set max_length during training, improving GPU utilization. See [here](https://github.com/modelscope/swift/blob/main/examples/train/packing/train.sh).
- **Streaming Training**: This method continuously reads data, reducing memory usage when handling large datasets. Check [here](https://github.com/modelscope/swift/blob/main/examples/train/streaming/train.sh) for details.
- **Lazy Tokenization**: Suitable for scenarios where a fixed amount of data is read in at once, and images are parsed during training. Refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/lazy_tokenize/train.sh).
- **Agent Training**: For more details, see [here](https://github.com/modelscope/swift/blob/main/examples/train/agent).
- **All-to-all model training**: Refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/all_to_all)

**Tips**:

- We recommend setting `--gradient_checkpointing true` during training to **save GPU memory**; this may slightly reduce training speed.
- If you wish to use DeepSpeed, you need to run `pip install deepspeed==0.14.*`. Using deepSpeed can **save GPU memory**, although it may slightly lower training speed.
- If your machine has high-performance GPUs like A100 and the model supports flash-attn, we recommend installing [**flash-attn**](https://github.com/Dao-AILab/flash-attention) to enhance training and inference speed and reduce memory consumption.
- If you need to train while **offline**, use `--model <model_dir>` and set `--check_model false`. Please check the [command line parameters](Commend-line-parameters.md) for details.
- If you want to push weights to the ModelScope Hub during training, you need to set `--push_to_hub true`.
- If you want to merge and save LoRA weights during inference, set `--merge_lora true`. **Currently, it is not possible to merge models trained with qlora.**, and thus **qlora is not recommended for fine-tuning** due to poor deployment ecology.
