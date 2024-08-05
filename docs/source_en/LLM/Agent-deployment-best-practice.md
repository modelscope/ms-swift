# Agent Deployment Best Practice

## Table of Contents
- [Environment Setup](#Environment-Setup)
- [Tools Field](#Tools-Field)
- [Deployment](#Deployment)

Deployment Examples
Environment Installation

## Environment Setup
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
pip install 'ms-swift[llm]' -U

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## Tools Field
The tools field provides the API information that the model can call. It supports OpenAI and ToolBench formats and requires the name, description, and parameters of the tools. An example is provided below:

OpenAI tools format
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

ToolBench tools format
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
During inference, the information in the tools field will be converted into the corresponding tools system prompt. If a system prompt already exists, the tools prompt will be appended to it.

Currently, three types of tools system prompts are supported: ReAct-EN, ReAct-ZH and ToolBench. Examples are shown below:

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

By default, the system employs the ReAct-EN format. However, you have the option to specify the `--tools_prompt` parameter with either `react-zh` or `toolbench` to utilize one of the alternative formats.


If you have a better tools system prompt, please feel free to let us know or contribute to us.

## Deployment

Taking the deployment of vLLM as an example, with non-streaming invocation and ReAct prompt, we demonstrate the model deployment.

When deploying an Agent, it is crucial that the model itself possesses a strong capability to follow instructions or has undergone training on an Agent dataset. If the existing model is incapable of selecting the appropriate tools and accurately setting their parameters based on the tools field, it is advisable to switch to a model with higher performance or to refine the model using the strategies outlined in [Agent Fine-tuning Practices](./Agent-fine-tuning-best-practice.md).

Here, we choose the llama3-8b-instruct model as an example.

```shell
swift deploy \
  --model_type llama3-8b-instruct \
  --infer_backend vllm \
```

Use the curl command to call the interface. Because the ReAct format ends with Observation:, we need to specify Observation: in the stop parameter as a stop word to truncate the model's response. Some models treat Observation:\n as a single token, so we also include it as a stop word.

If you are using the ToolBench prompt, specifying stop words is not necessary (although including them won't cause any issues).
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

You can also select a tool from the tools field by specifying the `tool_choice` field, for example: `"tool_choice": {"type": "function", "function": {"name": "my_function"}}`. By default, all tools are selected, but you can set it to None to disable the tools field.


result
```json
{"model":"llama3-8b-instruct","choices":[[{"index":0,"message":{"role":"assistant","content":"Question: What's the weather like in Boston today?\n\nThought: I need to get the current weather in Boston to answer this question.\n\nAction: get_current_weather\n\nAction Input: {'location': 'Boston, MA', 'unit': 'fahrenheit'}\n\nObservation:","tool_calls":[{"id":"toolcall-f534d907ae254f2ab96e06c25179ddf9","function":{"arguments":" {'location': 'Boston, MA', 'unit': 'fahrenheit'}\n\n","name":"get_current_weather"},"type":"function"}]},"finish_reason":"stop"}]],"usage":{"prompt_tokens":262,"completion_tokens":54,"total_tokens":316},"id":"chatcmpl-8630e8d675c941c0aca958a37633a3c9","object":"chat.completion","created":1717590756}
```

You can also test with OpenAI SDK, for example
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

# stream
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

In the tool_calls of the returned results, you can obtain the information about the called function and its parameters.

Assuming the returned result is `The weather in Boston today is 32°F (0°C), with clear skies`, we will fill this result into the role as tool and pass it into the message field.

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

For the ReAct format, we concatenate the result back after the last `Observations:` field returned in the previous round by the model.

For the ToolBench format, handle it according to the model template. If the model template does not specify a special handling method for this field, it is treated as user input.

If you have a better handling method, please feel free to let us know or contribute to us.

result
```json
{"model":"llama3-8b-instruct","choices":[{"index":0,"message":{"role":"assistant","content":"\n\nAnswer: The weather in Boston today is 32°F (0°C), with clear skies.","tool_calls":null},"finish_reason":null}],"usage":{"prompt_tokens":93,"completion_tokens":21,"total_tokens":114},"id":"chatcmpl-5e63cee5155f48a48d1366001d16502b","object":"chat.completion","created":1717590962}
```

If you want to integrate code and tools to complete the entire workflow loop, we recommend reading the [OpenAI tutorial](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models).
