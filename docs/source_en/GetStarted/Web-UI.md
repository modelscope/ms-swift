# Web-UI

Currently, SWIFT supports interface-based training and inference, with parameter support similar to script training. After installing SWIFT, use the following command:

```shell
swift web-ui --lang zh/en
```

to start the interface for training and inference.

Additionally, ms-swift supports interface inference mode (i.e., Space deployment):

```shell
swift app --model '<model>' --studio_title My-Awesome-Space --stream true
# or
swift app --model '<model>' --adapters '<adapter>' --studio_title My-Awesome-Space --stream true
```
This will launch an application with only the inference page, which will deploy the model upon startup and provide it for subsequent use.
