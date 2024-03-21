## 🔥SCEdit

SCEdit, proposed by Alibaba TongYi Vision Intelligence Lab, is an efficient generative fine-tuning framework. The framework not only supports fine-tuning capabilities for text-to-image downstream tasks, **saving 30%-50% of training memory overhead compared to LoRA**, achieving rapid transfer to specific generation scenarios; but it can also **directly extend to controllable image generation tasks, requiring only 7.9% of the parameter amount of ControlNet conditional generation and saving 30% of memory overhead**, supporting conditional generation tasks such as edge maps, depth maps, segmentation maps, poses, color maps, image inpainting, etc.

We used the 3D style data from the [Style Transfer Dataset](https://modelscope.cn/datasets/damo/style_custom_dataset/dataPeview) for training, and tested using the same `Prompt: A boy in a camouflage jacket with a scarf`. The specific qualitative and quantitative results are as follows:

| Method    | bs   | ep   | Target Module | Param. (M)    | Mem. (MiB) | 3D style                                                     |
| --------- | ---- | ---- | ------------- | ------------- | ---------- | ------------------------------------------------------------ |
| LoRA/r=64 | 1    | 50   | q/k/v/out/mlp | 23.94 (2.20%) | 8440MiB    | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703665229562-0f33bbb0-c492-41b4-9f37-3ae720dca80d.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 1    | 50   | up_blocks     | 19.68 (1.81%) | 7556MiB    | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703665933913-74b98741-3b57-46a4-9871-539df3a0112c.png" alt="img" style="zoom:20%;" /> |
| LoRA/r=64 | 10   | 100  | q/k/v/out/mlp | 23.94 (2.20%) | 26300MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703750608529-de20d0e7-bf9c-4928-8e59-73cc54f2c8d7.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 10   | 100  | up_blocks     | 19.68 (1.81%) | 18634MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703663033092-94492e44-341f-4259-9df4-13c168e3b5d6.png" alt="img" style="zoom:20%;" /> |
| LoRA/r=64 | 30   | 200  | q/k/v/out/mlp | 23.94 (2.20%) | 69554MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703750626635-2e368d7b-5e99-4a06-b189-8615f302bcd7.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 30   | 200  | up_blocks     | 19.68 (1.81%) | 43350MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703662246942-1102b1f4-93ab-4653-b943-3302f2a5259e.png" alt="img" style="zoom:20%;" /> |

To perform the training task using SCEdit and reproduce the above results:

```shell
# First, follow the installation steps in the section below
cd examples/pytorch/multi_modal/notebook  
python text_to_image_synthesis.py
```