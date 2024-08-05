# Agent部署最佳实践

## 目录

- [环境安装](#环境安装)
- [tools字段](#tools字段)
- [部署](#部署)
- [总结](#总结)

## 环境安装
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
pip install 'ms-swift[llm]' -U

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## tools字段
tools字段提供了模型可以调用的API信息。支持OpenAI和ToolBench格式，需要提供tools的名字，描述和参数，示例如下

OpenAI tools格式
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

ToolBench tools 格式
```json
{
"tools": [
      {
        "name": "url_for_newapi",
        "description": "This is the subfunction for tool \"newapi\", you can use this tool.The description of this function is: \"url_for_newapi\"",
        "parameters": {
          "type": "object",
          "properties": {
            "url": {
              "type": "string",
              "description": "",
              "example_value": "https://www.instagram.com/reels/CtB6vWMMHFD/"
            }
          },
          "required": [
            "url"
          ],
          "optional": [
            "url"
          ]
        }
      },
      {
        "name": "n_for_newapi",
        "description": "This is the subfunction for tool \"newapi\", you can use this tool.The description of this function is: \"n_for_newapiew var\"",
        "parameters": {
          "type": "object",
          "properties": {
            "language": {
              "type": "string",
              "description": "",
              "example_value": "https://www.instagram.com/reels/Csb0AI3IYUN/"
            }
          },
          "required": [
            "language"
          ],
          "optional": []
        }
      },
      {
        "name": "Finish",
        "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
        "parameters": {
          "type": "object",
          "properties": {
            "return_type": {
              "type": "string",
              "enum": [
                "give_answer",
                "give_up_and_restart"
              ]
            },
            "final_answer": {
              "type": "string",
              "description": "The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\""
            }
          },
          "required": [
            "return_type"
          ]
        }
      }
    ],
}
```

在推理过程中，会将tools的信息转换成对应的tools system prompt。如果已经存在system prompt，则会拼接在已有的之后。

目前支持英文ReAct,中文ReAct和ToolBench三种tools system prompt，示例如下

ReAct-EN
```
Answer the following questions as best you can. You have access to the following tools:

{'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of [get_current_weather]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Final Answer: the final answer to the original input question

Begin!
```

ReAct-ZH
```
尽你所能回答以下问题。你拥有如下工具：

{'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}}

以下格式回答：

Thought: 思考你应该做什么
Action: 工具的名称，必须是[get_current_weather]之一
Action Input: 工具的输入
Observation: 工具返回的结果
... (Thought/Action/Action Input/Observation的过程可以重复零次或多次)
Final Answer: 对输入问题的最终答案

开始！
```
ToolBench
```
You can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Your output should follow this format:
Thought:
Action:
Action Input:

After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember:
1.the state change is irreversible, you can\'t go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let\'s Begin!
Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user,If you can\'t handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions\' names.
Specifically, you have access to the following APIs: {\'name\': \'get_current_weather\', \'description\': \'Get the current weather in a given location\', \'parameters\': {\'type\': \'object\', \'properties\': {\'location\': {\'type\': \'string\', \'description\': \'The city and state, e.g. San Francisco, CA\'}, \'unit\': {\'type\': \'string\', \'enum\': [\'celsius\', \'fahrenheit\']}}, \'required\': [\'location\']}}
```

默认使用ReAct-EN格式，你也可以在参数中指定`--tools_prompt`为 `react_zh`或`toolbench` 来选择中文ReAct或ToolBench格式

如果你有更好用的tools system prompt，欢迎告知或贡献给我们。



## 部署
以下以vLLM部署，非流式调用，ReAct prompt为例.

部署Agent时，需要型本身必须具备较强的指令遵循能力，或者已在Agent数据集上进行了训练。如果现有模型未能根据tools字段进行工具选择和参数设置，建议采用更高性能的模型，或者参照[Agent微调实践](./Agent微调最佳实践.md)训练模型

部署模型，这里我们选择`llama3-8b-instruct`模型作为示范
```shell
swift deploy \
  --model_type llama3-8b-instruct \
  --infer_backend vllm \
```

用curl命令调用接口，因为ReAct格式会以Observation:为结尾，我们需要在stop中指定`Observation:`作为stop words来截断模型回复。有些模型会将`Observation:\n`作为一个token，这里我们也将其作为stop words。

如果你使用ToolBench prompt, 则无需指定stop words（当然加上也没有关系）

```shell
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "What'\''s the weather like in Boston today?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
      }
    ],
    "stream": false,
    "stop": ["Observation:", "Observation:\n"]
  }'
```

你也可以通过指定`tool_choice`字段来选择tools中的tool，比如`"tool_choice":{"type": "function", "function": {"name": "my_function"}}`. 默认选择所有tools，也可以设置为None来屏蔽tools字段

调用结果
```json
{"model":"llama3-8b-instruct","choices":[[{"index":0,"message":{"role":"assistant","content":"Question: What's the weather like in Boston today?\n\nThought: I need to get the current weather in Boston to answer this question.\n\nAction: get_current_weather\n\nAction Input: {'location': 'Boston, MA', 'unit': 'fahrenheit'}\n\nObservation:","tool_calls":[{"id":"toolcall-f534d907ae254f2ab96e06c25179ddf9","function":{"arguments":" {'location': 'Boston, MA', 'unit': 'fahrenheit'}\n\n","name":"get_current_weather"},"type":"function"}]},"finish_reason":"stop"}]],"usage":{"prompt_tokens":262,"completion_tokens":54,"total_tokens":316},"id":"chatcmpl-8630e8d675c941c0aca958a37633a3c9","object":"chat.completion","created":1717590756}
```

在返回结果的tool_calls中，可以获得调用的函数以及参数信息。

你也可以通过OpenAI SDK进行测试
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
query = "What's the weather like in Boston today?"
messages = [{
    'role': 'user',
    'content': query
}]
tools =  [
      {
        "name": "url_for_newapi",
        "description": "This is the subfunction for tool \"newapi\", you can use this tool.The description of this function is: \"url_for_newapi\"",
        "parameters": {
          "type": "object",
          "properties": {
            "url": {
              "type": "string",
              "description": "",
              "example_value": "https://www.instagram.com/reels/CtB6vWMMHFD/"
            }
          },
          "required": [
            "url"
          ],
          "optional": [
            "url"
          ]
        }
      },
]
resp = client.chat.completions.create(
    model='llama3-8b-instruct',
    tools = tools,
    messages=messages,
    seed=42)
tool_calls = resp.choices[0].message.tool_calls[0]
print(f'query: {query}')
print(f'tool_calls: {tool_calls}')

# 流式
stream_resp = client.chat.completions.create(
    model='llama3-8b-instruct',
    messages=messages,
    tools=tools,
    stream=True,
    seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print(chunk.choices[0].delta.tool_calls[0])

"""
query: What's the weather like in Boston today?
tool_calls: {'id': 'toolcall-e4c637435e754cf9b2034c3e6861a4ad', 'function': {'arguments': ' {"url": "https://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q=Boston"}', 'name': 'url_for_newapi'}, 'type': 'function'}
query: What's the weather like in Boston today?
response: Thought: I need to find the weather information for Boston today. I can use the 'newapi' tool to get the weather forecast.
Action: url_for_newapi
Action Input: {"url": "https://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q=Boston"}
"""
```
假设调用返回的结果为`The weather in Boston today is 32°F (0°C), with clear skies`, 我们将结果在role tool字段填入message传入
```shell
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "What'\''s the weather like in Boston today?"
      },
      {
        "role": "assistant",
        "content": "Question: What'\''s the weather like in Boston today?\n\nThought: I need to get the current weather in Boston.\n\nAction: get_current_weather\n\nAction Input: {\"location\": \"Boston, MA\", \"unit\": \"fahrenheit\"}\n\nObservation:"
      },
      {
        "role": "tool",
        "content": "{\"result\": \"The weather in Boston today is 32°F (0°C), with clear skies\"}\\n\\n"
      }
    ],
    "stream": false,
    "stop": ["Observation:", "Observation:\n"]
  }'
```

对于ReAct格式，我们会将其拼接结果拼接回上一轮模型返回最后的`Observations:`字段之后。

对于ToolBench格式，根据模型template对其处理。如果模型template没有指定对该字段的特殊处理方式，则视为user输入。

如果你有更好用的处理方法，也欢迎告知或贡献给我们。

调用结果
```json
{"model":"llama3-8b-instruct","choices":[{"index":0,"message":{"role":"assistant","content":"\n\nAnswer: The weather in Boston today is 32°F (0°C), with clear skies.","tool_calls":null},"finish_reason":null}],"usage":{"prompt_tokens":93,"completion_tokens":21,"total_tokens":114},"id":"chatcmpl-5e63cee5155f48a48d1366001d16502b","object":"chat.completion","created":1717590962}
```

如果你想要结合代码和tools完成整个链路闭环，推荐阅读[OpenAI教程](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)
