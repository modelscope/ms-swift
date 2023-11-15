# Res-Tuning组件

<div align="center">

## [NeurIPS 2023] Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone

### [arXiv](https://arxiv.org/abs/2310.19859)  |  [Project Page](https://res-tuning.github.io/)

</div>

Res-Tuning 是一种灵活高效的微调tuner。我们把tuner的设计从模型网络结构中解耦出来以便灵活地组合，
并进一步扩展实现了一种新的节省内存的旁路tuner，大大减少了显存消耗和多任务推理成本。

目前Res-Tuning在[SWIFT](https://github.com/modelscope/swift)中以可插拔的tuner算法组件提供，开发者可以直接使用它。

### 支持的组件列表

- [x] Res-Adapter
- [x] Res-Tuning-Bypass
- [ ] Res-Prefix
- [ ] Res-Prompt

### 使用方式

#### Demo
- 可以使用我们提供的 [可视化例子](https://github.com/modelscope/swift/blob/main/examples/pytorch/cv/notebook/swift_vision.ipynb).

#### 初始化Tuner

```Python
from swift import ResTuningConfig
config = ResTuningConfig(
    dims=768,
    root_modules=r'.*blocks.0$',
    stem_modules=r'.*blocks\.\d+$',
    target_modules=r'norm',
    tuner_cfg='res_adapter'
)
```
- dims: The dimensions of the hidden states.
- root_modules: The root module to be replaced.
- stem_modules: The stem modules to be replaced.
- target_modules: The target module to be replaced.
- tuner_cfg: The configuration of the tuning module.

#### 加载模型

```Python
from swift import Swift
import timm, torch
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100)
model_tune = Swift.prepare_model(model, config)
print(model_tune.get_trainable_parameters())
print(model(torch.ones(1, 3, 224, 224)).shape)
```


### 引用
```
@inproceedings{jiang2023restuning,
  title={Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone},
  author={Jiang, Zeyinzi and Mao, Chaojie and Huang, Ziyuan and Ma, Ao and Lv, Yiliang and Shen, Yujun and Zhao, Deli and Zhou, Jingren},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
