# Web-UI

目前SWIFT已经支持了界面化的训练和推理，参数支持和脚本训练相同。在安装SWIFT后，使用如下命令：

```shell
swift web-ui --lang zh/en
```

开启界面训练和推理。

目前ms-swift额外支持了界面推理模式（即Space部署）：

```shell
swift app --model '<model>' --studio_title My-Awesome-Space --stream true
# 或者
swift app --model '<model>' --adapters '<adapter>' --stream true
```
即可启动一个只有推理页面的应用，该应用会在启动时对模型进行部署并提供后续使用。
