# Inference and Deployment

SWIFT supports inference and deployment via command line and Python code.

# Inference

SWIFT supports three ways of model inference:
- Use `swift web-ui` to deploy models for interface inference
- Use the example provided for command line inference: https://github.com/modelscope/ms-swift/tree/main/examples/infer/infer/infer.sh
- Deploy with `swift deploy --model xxx` and call it afterward

The first method is straightforward and will not be elaborated here. You can view the detailed instructions [here](../GetStarted/Interface-usage.md).

## Command Line Inference

For command line inference, you can refer to the link mentioned in the second point. Once the script runs, simply input your query in the terminal. Here are some command line usage notes:
- The `reset-system` command  sets the system in the command line.
- The `multi-line` command  supports multi-line input, ending the input with `#`.
- The `single-line` command  switches to single line mode.
- The `clear` command  clears history.
- If your query includes multi-modal data, add tags like <image>/<video>/<audio> to it. For example, to input `What is in the image?`, use `<image>What is in the image?` and then provide the image address.

## Deployment

You can execute deployment using `swift deploy`. Currently, SWIFT supports three inference frameworks: pt (native torch), vLLM, and LMDeploy. You can switch between them using `--infer_backend pt/vllm/lmdeploy`.
Apart from pt, vllm and lmdeploy have their own model support ranges, so please refer to the official documentation of each to determine availability and prevent runtime errors.

You can find deployment examples [here](https://github.com/modelscope/ms-swift/tree/main/examples/infer).
