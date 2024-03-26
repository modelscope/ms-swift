# Interface Training and Inference

Currently, SWIFT supports interface-based training and inference, with parameters consistent with script-based training. After installing SWIFT, use the following command:

```shell
swift web-ui
```

This command starts the interface for training and inference.

The web-ui command doesn't accept parameters; all controllable parts are handled within the interface. However, there are a few environment variables that can be used:

> WEBUI_SHARE=1/0: Default is 0. Controls whether gradio is in share mode.
>
> SWIFT_UI_LANG=en/zh: Controls the language of the web-ui interface.
>
> WEBUI_SERVER: The server_name parameter. Specifies the host IP for web-ui. 0.0.0.0 means all IPs can access, while 127.0.0.1 means only local access is allowed.
>
> WEBUI_PORT: The port number for web-ui.
>
> USE_INFERENCE=1/0: Default is 0. Controls whether the gradio inference page directly loads the model for inference or deployment (USE_INFERENCE=0).
