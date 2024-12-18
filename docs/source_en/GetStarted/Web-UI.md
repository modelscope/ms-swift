# Web-UI

Currently, SWIFT supports interface-based training and inference, with parameter support similar to script training. After installing SWIFT, use the following command:

```shell
swift web-ui --host 0.0.0.0 --port 7860 --lang zh/en
```

to start the interface for training and inference.

Additionally, the web-ui now supports app-ui mode (i.e., Space deployment):

```shell
swift web-ui --model '<model>' --studio_title My-Awesome-Space
# or
swift web-ui --model '<model>' --adapters '<adapter>' --studio_title My-Awesome-Space
```
This will launch an application with only the inference page, which will deploy the model upon startup and provide it for subsequent use.
