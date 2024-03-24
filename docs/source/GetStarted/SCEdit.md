## 🔥SCEdit

SCEdit由阿里巴巴通义实验室视觉智能团队(Alibaba TongYi Vision Intelligence Lab)所提出，是一个高效的生成式微调框架。该框架不仅支持文生图下游任务的微调能力，**相比LoRA节省30%-50%的训练显存开销**，实现快速迁移到特定的生成场景中；而且还可以**直接扩展到可控图像生成任务中，仅需ControlNet条件生成7.9%的参数量并节省30%的显存开销**，支持边缘图、深度图、分割图、姿态、颜色图、图像补全等条件生成任务。

我们使用了[风格迁移数据集](https://modelscope.cn/datasets/damo/style_custom_dataset/dataPeview)中的3D风格数据进行了训练，并使用相同的`Prompt: A boy in a camouflage jacket with a scarf`进行测试，具体的定性和定量的结果如下：

| Method    | bs   | ep   | Target Module | Param. (M)    | Mem. (MiB) | 3D style                                                     |
| --------- | ---- | ---- | ------------- | ------------- | ---------- | ------------------------------------------------------------ |
| LoRA/r=64 | 1    | 50   | q/k/v/out/mlp | 23.94 (2.20%) | 8440MiB    | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703665229562-0f33bbb0-c492-41b4-9f37-3ae720dca80d.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 1    | 50   | up_blocks     | 19.68 (1.81%) | 7556MiB    | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703665933913-74b98741-3b57-46a4-9871-539df3a0112c.png" alt="img" style="zoom:20%;" /> |
| LoRA/r=64 | 10   | 100  | q/k/v/out/mlp | 23.94 (2.20%) | 26300MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703750608529-de20d0e7-bf9c-4928-8e59-73cc54f2c8d7.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 10   | 100  | up_blocks     | 19.68 (1.81%) | 18634MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703663033092-94492e44-341f-4259-9df4-13c168e3b5d6.png" alt="img" style="zoom:20%;" /> |
| LoRA/r=64 | 30   | 200  | q/k/v/out/mlp | 23.94 (2.20%) | 69554MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703750626635-2e368d7b-5e99-4a06-b189-8615f302bcd7.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 30   | 200  | up_blocks     | 19.68 (1.81%) | 43350MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703662246942-1102b1f4-93ab-4653-b943-3302f2a5259e.png" alt="img" style="zoom:20%;" /> |

使用SCEdit执行训练任务并复现上述结果：

```shell
# 先执行下面章节的安装步骤
cd examples/pytorch/multi_modal/notebook
python text_to_image_synthesis.py
```
