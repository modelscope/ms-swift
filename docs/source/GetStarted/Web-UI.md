# Web-UI

目前SWIFT已经支持了界面化的训练和推理，参数支持和脚本训练相同。在安装SWIFT后，使用如下命令：

```shell
swift web-ui --lang zh
# or en
swift web-ui --lang en
```

开启界面训练和推理。

SWIFT web-ui是命令行的高级封装，即，在界面上启动的训练、部署等任务，会在系统中以命令行启动一个独立的进程，伪代码类似：
```python
import os
os.system('swift sft --model xxx --dataset xxx')
```

这给web-ui带来了几个特性：
1. web-ui的每个超参数描述都带有`--xxx`的标记，这与[命令行参数](../Instruction/命令行参数.md)的内容是一致的
2. web-ui可以在一台多卡机器上并行启动多个训练/部署任务
3. web-ui服务关闭后，后台服务是仍旧运行的，这防止了web-ui被关掉后影响训练进程，如果需要关闭后台服务，只需要**选择对应的任务**后在界面上的`运行时`tab点击杀死服务
4. 重新启动web-ui后，如果需要显示正在运行的服务，在`运行时`tab点击`找回运行时任务`即可
5. 训练界面支持显示运行日志，请在选择某个任务后手动点击`展示运行状态`，在训练时运行状态支持展示训练图表，图标包括训练loss、训练acc、学习率等基本指标，在人类对齐任务重界面图标为margin、logps等关键指标
6. web-ui的训练不支持PPO，该过程比较复杂，建议使用examples的[shell脚本](https://github.com/modelscope/ms-swift/tree/main/examples/train/rlhf/ppo)直接运行

如果需要使用share模式，请添加`--share true`参数。注意：请不要在dsw、notebook等环境中使用该参数。

目前ms-swift额外支持了界面推理模式（即Space部署）：

```shell
swift app --model '<model>' --studio_title My-Awesome-Space --stream true
# 或者
swift app --model '<model>' --adapters '<adapter>' --stream true
```
即可启动一个只有推理页面的应用，该应用会在启动时对模型进行部署并提供后续使用。
