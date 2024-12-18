# Web-UI

目前SWIFT已经支持了界面化的训练和推理，参数支持和脚本训练相同。在安装SWIFT后，使用如下命令：

```shell
swift web-ui --host 0.0.0.0 --port 7860 --lang zh/en
```

开启界面训练和推理。

目前web-ui额外支持了app-ui模式（即Space部署）：

```shell
swift web-ui --model '<model>' --studio_title My-Awesome-Space
# 或者
swift web-ui --model '<model>' --adapters '<adapter>' --studio_title My-Awesome-Space
```
即可启动一个只有推理页面的应用，该应用会在启动时对模型进行部署并提供后续使用。
