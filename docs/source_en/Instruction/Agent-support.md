# Agent Support


## Dataset Format
Example data samples for the pure text Agent and multimodal Agent are as follows:

```jsonl
{"tools": ["{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"Weather forecast. Get real-time air quality, including current air quality, PM2.5, and PM10 information.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"City name, e.g., Shanghai\"}}, \"required\": [\"city\"]}}}"], "messages": [{"role": "user", "content": "What is the weather like in Beijing and Shanghai today?"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"Beijing\"}}"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"Shanghai\"}}"}, {"role": "tool_response", "content": "{\"city\": \"Beijing\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, {"role": "tool_response", "content": "{\"city\": \"Shanghai\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, {"role": "assistant", "content": "According to the weather forecast tool, the air quality index (AQI) in Beijing is 10, which indicates good air quality; whereas in Shanghai, the AQI is 72, indicating mild pollution."}]}
{"tools": ["{\"type\": \"function\", \"function\": {\"name\": \"click\", \"description\": \"Click on a position on the screen\", \"parameters\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"integer\", \"description\": \"X-coordinate representing the horizontal position on the screen\"}, \"y\": {\"type\": \"integer\", \"description\": \"Y-coordinate representing the vertical position on the screen\"}}, \"required\": [\"x\", \"y\"]}}}"], "messages": [{"role": "user", "content": "<image>What time is it now?"}, {"role": "assistant", "content": "<think>\nI can check the current time by opening the calendar app.\n</think>\n"}, {"role": "tool_call", "content": "{\"name\": \"click\", \"arguments\": {\"x\": 105, \"y\": 132}}"}, {"role": "tool_response", "content": "{\"images\": \"<image>\", \"status\": \"success\"}"}, {"role": "assistant", "content": "Successfully opened the calendar app. The current time is 11 o'clock in the morning."}], "images": ["desktop.png", "calendar.png"]}
```
- When the `agent_template` is set to "react_en", "hermes", etc., this format is compatible with training for all model Agents and allows easy switching between different models.
- Here, `tools` is a `List[str]`, where each tool needs to be a JSON string. Additionally, the `content` part of the messages where the role is `'tool_call'` or `'tool_response/tool'` must also be in JSON string format.
- The `tools` field will be combined with the `{"role": "system", ...}` section during training/inference according to the `agent_template`, forming a complete system section.
- The `{"role": "tool_call", ...}` part will automatically be converted into corresponding formats of `{"role": "assistant", ...}` based on the `agent_template`. Multiple consecutive `{"role": "assistant", ...}` entries will be concatenated to form a complete assistant_content.
- The `{"role": "tool_response", ...}` can also be written as `{"role": "tool", ...}`, these two forms are equivalent. This part will also be automatically converted according to the `agent_template`. During training, this part does not participate in loss calculations, similar to `{"role": "user", ...}`.
- This format supports parallel tool calls; refer to the first data sample for an example. In multimodal Agent data samples, the number of `<image>` tags should match the length of "images", and their positions indicate where the image features are inserted. It also supports other modalities, such as audios and videos.


The following are the `input_ids` and `labels` after encoding the two data samples mentioned above using the templates for **qwen2_5** and **qwen2_5_vl** , with the selected `agent_template` being **hermes** :

Sample One (Parallel Tool Calls):

```text
[INPUT_IDS] <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "realtime_aqi", "description": "Weather forecast. Get real-time air quality, including current air quality, PM2.5, and PM10 information.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g., Shanghai"}}, "required": ["city"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What is the weather like in Beijing and Shanghai today?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "realtime_aqi", "arguments": {"city": "Beijing"}}
</tool_call>
<tool_call>
{"name": "realtime_aqi", "arguments": {"city": "Shanghai"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"city": "Beijing", "aqi": "10", "unit": "celsius"}
</tool_response>
<tool_response>
{"city": "Shanghai", "aqi": "72", "unit": "fahrenheit"}
</tool_response><|im_end|>
<|im_start|>assistant
According to the weather forecast tool, the air quality index (AQI) in Beijing is 10, which indicates good air quality; whereas in Shanghai, the AQI is 72, indicating mild pollution.<|im_end|>

[LABELS] [-100 * 195]<tool_call>
{"name": "realtime_aqi", "arguments": {"city": "Beijing"}}
</tool_call>
<tool_call>
{"name": "realtime_aqi", "arguments": {"city": "Shanghai"}}
</tool_call><|im_end|>[-100 * 67]According to the weather forecast tool, the air quality index (AQI) in Beijing is 10, which indicates good air quality; whereas in Shanghai, the AQI is 72, indicating mild pollution.<|im_end|>
```

Sample Two (Multimodal, Mixed Assistant and Tool Call):

```text
[INPUT_IDS] <|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "click", "description": "Click on a position on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "description": "X-coordinate representing the horizontal position on the screen"}, "y": {"type": "integer", "description": "Y-coordinate representing the vertical position on the screen"}}, "required": ["x", "y"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
<|vision_start|>[151655 * 729]<|vision_end|>What time is it now?<|im_end|>
<|im_start|>assistant
<think>
I can check the current time by opening the calendar app.
</think>
<tool_call>
{"name": "click", "arguments": {"x": 105, "y": 132}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"images": "<|vision_start|>[151655 * 729]<|vision_end|>", "status": "success"}
</tool_response><|im_end|>
<|im_start|>assistant
Successfully opened the calendar app. The current time is 11 o'clock in the morning.<|im_end|>

[LABELS] [-100 * 924]<think>
I can check the current time by opening the calendar app.
</think>
<tool_call>
{"name": "click", "arguments": {"x": 105, "y": 132}}
</tool_call><|im_end|>[-100 * 759]Successfully opened the calendar app. The current time is 11 o'clock in the morning.<|im_end|>
```

**react_en** is one of the commonly used agent template formats. Below is an example of the `input_ids` and `labels` after encoding by qwen2_5 using `agent_template='react_en'`:

```text
[INPUT_IDS] <|im_start|>system
Answer the following questions as best you can. You have access to the following tools:

realtime_aqi: Call this tool to interact with the realtime_aqi API. What is the realtime_aqi API useful for? Weather forecast. Get real-time air quality, including current air quality, PM2.5, and PM10 information. Parameters: {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g., Shanghai"}}, "required": ["city"]} Format the arguments as a JSON object.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [realtime_aqi]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
<|im_end|>
<|im_start|>user
What is the weather like in Beijing and Shanghai today?<|im_end|>
<|im_start|>assistant
Action: realtime_aqi
Action Input: {'city': 'Beijing'}
Action: realtime_aqi
Action Input: {'city': 'Shanghai'}
Observation:{"city": "Beijing", "aqi": "10", "unit": "celsius"}
Observation:{"city": "Shanghai", "aqi": "72", "unit": "fahrenheit"}
According to the weather forecast tool, the air quality index (AQI) in Beijing is 10, which indicates good air quality; whereas in Shanghai, the AQI is 72, indicating mild pollution.<|im_end|>

[LABELS] [-100 * 233]Action: realtime_aqi
Action Input: {'city': 'Beijing'}
Action: realtime_aqi
Action Input: {'city': 'Shanghai'}
Observation:[-100 * 45]According to the weather forecast tool, the air quality index (AQI) in Beijing is 10, which indicates good air quality; whereas in Shanghai, the AQI is 72, indicating mild pollution.<|im_end|>
```

The following code can be used to experiment with more models and `agent_template` options. For more selectable values of `agent_template`, refer to [here](https://github.com/modelscope/swift/blob/main/swift/plugin/agent_template/__init__.py).

```python
from swift.llm import get_model_tokenizer, get_template

_, tokenizer = get_model_tokenizer('ZhipuAI/GLM-4-9B-0414', load_model=False)
template = get_template(tokenizer.model_meta.template, tokenizer, agent_template='hermes')
data = {...}
template.set_mode('train')
encoded = template.encode(data)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
```

## Tools Format

The tools field provides information about the APIs that the model can call. You need to provide the name, description, and parameters of the tools, as shown in the example below:

```python
tools = [{
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city and state, e.g. San Francisco, CA'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit']
                }
            },
            'required': ['location']
        }
    }
}]
```

## Usage of loss_scale

`loss_scale` can be used to adjust the training loss weight for the model's output section. For example, in the ReACT format, you can set `--loss_scale react` (the loss_scale configuration file is written [here](https://github.com/modelscope/swift/blob/main/swift/plugin/loss_scale/config/react.json)). The role of this parameter is as follows:

- The weight for the 'Thought:' and 'Final Answer:' sections is 1.
- The weight for the 'Action:' and 'Action Input:' sections is 2.
- The weight for the 'Observation:' field itself is 2.
- The weight for the tool invocation results following the 'Observation:' field is 0.

For the detailed design of the `loss_scale` plugin, please refer to the [Plugin-based Architecture](../Customization/Pluginization.md)documentation.

## Training

- Train the Agent capabilities of Base models by switching different models through modifying `--model`. Refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/agent/qwen2_5.sh).
- The agent_template for training GLM4 is hermes. Refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/agent/glm4.sh).
- Use `--loss_scale` to adjust the loss weight of the model output section. Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/agent/loss_scale).

## Inference

- 🚀For inference of the original model or fully trained model, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_agent.py).
- For inference after LoRA training, refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/agent/loss_scale/infer.md).

## Deployment

For server and client code, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/deploy/agent).
