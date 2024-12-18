# Inference and Deployment

SWIFT supports inference and deployment through command line, Python code, and interface methods:
- Use `engine.infer` or `engine.infer_async` for Python-based inference. See [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py) for reference.
- Use `swift infer` for command-line-based inference. See [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/cli_demo.sh) for reference.
- Use `swift deploy` for service deployment and perform inference using the OpenAI API or `client.infer`. Refer to the server guidelines [here](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/server) and the client guidelines [here](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client).
- Deploy the model with `swift web-ui` for web-based inference. You can check [here](../GetStarted/Interface-usage.md) for details.


## Command Line Inference

The command line inference can be referred to via the link provided in the second point above. After running the script, you only need to input queries in the terminal. Please note the several usage methods for the command line:

- The `reset-system` command resets the system.
- The `multi-line` command switches to multi-line mode, allowing line breaks in input, with # indicating the end of input.
- The `single-line` command switches to single-line mode.
- The `clear` command clears the history.
- The `exit` command exits the application.
If the query involves multimodal data, add tags like <image>/<video>/<audio>. For example, input `<image>What is in the image?`, and you can then input the image address.

## Inference Acceleration Backend
You can perform inference and deployment using `swift infer/deploy`. Currently, SWIFT supports three inference frameworks: pt (native torch), vLLM, and LMDeploy. You can switch between them using `--infer_backend pt/vllm/lmdeploy`. Apart from pt, both vLLM and LMDeploy have their own model support ranges. Please refer to their official documentation to verify availability and prevent runtime errors.
