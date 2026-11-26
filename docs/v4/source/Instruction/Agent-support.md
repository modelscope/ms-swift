# Agent支持

## 数据集格式

ms-swift 使用 agent-template 实现了Agent数据格式与模型的解耦：基于统一的数据集格式，可以灵活切换不同模型进行训练，无需修改数据。

纯文本Agent和多模态Agent的示例数据样本如下：
```jsonl
{"tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名，例如：上海\"}}, \"required\": [\"city\"]}}}]", "messages": [{"role": "user", "content": "北京和上海今天的天气情况"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"北京\"}}"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"上海\"}}"}, {"role": "tool_response", "content": "{\"city\": \"北京\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, {"role": "tool_response", "content": "{\"city\": \"上海\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, {"role": "assistant", "content": "根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。"}]}
{"tools": "[{\"type\": \"function\", \"function\": {\"name\": \"click\", \"description\": \"点击屏幕中的某个位置\", \"parameters\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"integer\", \"description\": \"横坐标，表示屏幕上的水平位置\"}, \"y\": {\"type\": \"integer\", \"description\": \"纵坐标，表示屏幕上的垂直位置\"}}, \"required\": [\"x\", \"y\"]}}}]", "messages": [{"role": "user", "content": "<image>现在几点了？"}, {"role": "assistant", "content": "<think>\n我可以通过打开日历App来获取当前时间。\n</think>\n"}, {"role": "tool_call", "content": "{\"name\": \"click\", \"arguments\": {\"x\": 105, \"y\": 132}}"}, {"role": "tool_response", "content": "{\"images\": \"<image>\", \"status\": \"success\"}"}, {"role": "assistant", "content": "成功打开日历App，现在的时间为中午11点"}], "images": ["desktop.png", "calendar.png"]}
```
- agent_template为"react_en", "hermes"等情况下，该格式适配所有模型Agent训练，可以轻松在不同模型间切换。
- 其中tools是一个包含tool列表的json字符串，messages中role为'tool_call'和'tool_response/tool'的content部分都需要是json字符串。
- tools字段将在训练/推理时和`{"role": "system", ...}"`部分组合，根据agent_template组成完整的system部分。
- `{"role": "tool_call", ...}`部分将根据agent_template自动转成对应格式的`{"role": "assistant", ...}`，多条连续的`{"role": "assistant", ...}`将拼接在一起组成完整的assistant_content。
- `{"role": "tool_response", ...}`也可以写成`{"role": "tool", ...}`，这两种写法是等价的。该部分也将根据`agent_template`自动转换格式。该部分在训练时将不进行损失的计算，角色类似于`{"role": "user", ...}`。
- 该格式支持并行调用工具，例子参考第一条数据样本。多模态Agent数据样本中`<image>`标签数量应与"images"长度相同，其标签位置代表图像特征的插入位置。当然也支持其他模态，例如audios, videos。
- 注意：您也可以手动将数据处理为role为system/user/assistant的messages格式。agent_template的作用是将其中的tools字段以及role为tool_call和tool_response的messages部分，自动映射为标准的role为system/user/assistant的messages格式。

以下为上述两条数据样本由qwen2_5和qwen2_5_vl的template进行encode后的input_ids和labels，选择的agent_template为**hermes**：

样本一（并行工具调用）：
```text
[INPUT_IDS] <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "realtime_aqi", "description": "天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名，例如：上海"}}, "required": ["city"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
北京和上海今天的天气情况<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "realtime_aqi", "arguments": {"city": "北京"}}
</tool_call>
<tool_call>
{"name": "realtime_aqi", "arguments": {"city": "上海"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"city": "北京", "aqi": "10", "unit": "celsius"}
</tool_response>
<tool_response>
{"city": "上海", "aqi": "72", "unit": "fahrenheit"}
</tool_response><|im_end|>
<|im_start|>assistant
根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。<|im_end|>

[LABELS] [-100 * 195]<tool_call>
{"name": "realtime_aqi", "arguments": {"city": "北京"}}
</tool_call>
<tool_call>
{"name": "realtime_aqi", "arguments": {"city": "上海"}}
</tool_call><|im_end|>[-100 * 67]根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。<|im_end|>
```

样本二（多模态，混合assistant和tool_call）：
```text
[INPUT_IDS] <|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "click", "description": "点击屏幕中的某个位置", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "description": "横坐标，表示屏幕上的水平位置"}, "y": {"type": "integer", "description": "纵坐标，表示屏幕上的垂直位置"}}, "required": ["x", "y"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
<|vision_start|>[151655 * 729]<|vision_end|>现在几点了？<|im_end|>
<|im_start|>assistant
<think>
我可以通过打开日历App来获取当前时间。
</think>
<tool_call>
{"name": "click", "arguments": {"x": 105, "y": 132}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"images": "<|vision_start|>[151655 * 729]<|vision_end|>", "status": "success"}
</tool_response><|im_end|>
<|im_start|>assistant
成功打开日历App，现在的时间为中午11点<|im_end|>

[LABELS] [-100 * 924]<think>
我可以通过打开日历App来获取当前时间。
</think>
<tool_call>
{"name": "click", "arguments": {"x": 105, "y": 132}}
</tool_call><|im_end|>[-100 * 759]成功打开日历App，现在的时间为中午11点<|im_end|>
```

**react_en**是常用的agent template格式之一，以下为样本一由qwen2_5使用`agent_template='react_en'`进行encode后的input_ids和labels：

```text
[INPUT_IDS] <|im_start|>system
Answer the following questions as best you can. You have access to the following tools:

realtime_aqi: Call this tool to interact with the realtime_aqi API. What is the realtime_aqi API useful for? 天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息 Parameters: {"type": "object", "properties": {"city": {"type": "string", "description": "城市名，例如：上海"}}, "required": ["city"]} Format the arguments as a JSON object.

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
北京和上海今天的天气情况<|im_end|>
<|im_start|>assistant
Action: realtime_aqi
Action Input: {'city': '北京'}
Action: realtime_aqi
Action Input: {'city': '上海'}
Observation:{"city": "北京", "aqi": "10", "unit": "celsius"}
Observation:{"city": "上海", "aqi": "72", "unit": "fahrenheit"}
根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。<|im_end|>

[LABELS] [-100 * 233]Action: realtime_aqi
Action Input: {'city': '北京'}
Action: realtime_aqi
Action Input: {'city': '上海'}
Observation:[-100 * 45]根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。<|im_end|>
```

更多模型和agent_template的尝试可以使用以下代码，更多的agent template可选值参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/agent_template/__init__.py)。
```python
from swift import get_processor, get_template

tokenizer = get_processor('Qwen/Qwen3.5-2B')
template = get_template(tokenizer)  # 使用默认agent模板
# template = get_template(tokenizer, agent_template='qwen3_5')
print(f'agent_template: {template._agent_template}')
data = {...}
template.set_mode('train')
encoded = template.encode(data)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
```


## tools格式
tools字段提供了模型可以调用的API信息。你需要提供tools的名字，描述和参数，示例如下：

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

## loss_scale的使用

loss_scale参数可用于调节模型输出部分在训练过程中的损失权重。目前支持两种配置方式：字符串精确匹配和正则表达式匹配。

1. 字符串匹配示例：ReACT 格式

以 ReACT 格式为例，可通过 `--loss_scale react` 启用相应的 loss_scale 配置（配置文件详见 [react.json](https://github.com/modelscope/ms-swift/blob/main/swift/loss_scale/config/react.json)）。该方式基于字符串精确匹配，配置中的字典映射需提供一个包含两个元素的列表，分别表示：当前匹配字符串本身的损失权重，
从该字符串之后到下一个指定字符串之前的内容的损失权重。该设置的具体效果如下：
- 'Action:' 和 'Action Input:' 字段自身及其后续内容的损失权重均为 2；
- 'Thought:' 和 'Final Answer:' 字段自身及其后续内容的损失权重均为 1；
- 'Observation:' 字段自身的权重为 2，但其后跟随的工具调用结果部分的损失权重为 0。

2. 正则匹配示例：忽略空思维块

在训练推理模型时，我们可能需要忽略数据集中存在的形如 `'<think>\n\n</think>\n\n'`的空思维标记损失计算。此时可使用 `--loss_scale ignore_empty_think`（配置文件详见 [ignore_empty_think.json](https://github.com/modelscope/ms-swift/blob/main/swift/loss_scale/config/ignore_empty_think.json)）。该配置采用正则表达式匹配方式，字典映射的列表只需指定一个值，表示匹配内容的损失权重。该设置的具体效果如下：

- 所有与正则表达式`<think>\\s*</think>\\s*`匹配的字符串，loss_scale为0，即不计算损失。

使用代码测试loss_scale:
```python
from swift import get_processor, get_template

data = {"messages": [
    {"role": "user", "content": "aaaaa"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nabc<think>\n\n</think>\n\n123"},
]}

template = get_template(get_processor('Qwen/Qwen3-8B'), loss_scale='ignore_empty_think')
template.set_mode('train')
inputs = template.encode(data)

print(template.safe_decode(inputs['labels']))
# '[-100 * 14]abc<think>\n\n</think>\n\n123<|im_end|>\n'
```

更多的loss_scale插件设计，请参考[架构](../Customization/Architecture.md#loss-scale)文档.

## 训练
- 训练Base模型的Agent能力，通过修改`--model`切换不同模型，参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/train/agent/qwen2_5.sh)。
- 训练GLM4的agent_template为hermes，参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/train/agent/glm4.sh)。
- 使用`--loss_scale`对模型输出部分的损失权重进行调整，参加[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/agent/loss_scale)。

## 推理

- 🚀原始模型或者全参数训练后模型的推理，参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_agent.py)。
- LoRA训练后推理，参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/train/agent/loss_scale/infer_lora.py)。

## 部署

服务端和客户端代码，参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/deploy/agent)。
