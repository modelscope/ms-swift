# Agent Support

SWIFT supports open-source models, particularly for training medium and small models (like 7B and 14B) in agent scenarios. It applies the [loss-scale technique](https://arxiv.org/pdf/2309.00986.pdf) in agent training for more stable API call capabilities, enabling Agent inference and deployment with a single commercial-grade GPU. It can be directly implemented in a full-cycle, closed-loop manner in production scenarios

## Data Preparation

Currently supported agent datasets in SWIFT include:
- [msagent-pro](https://www.modelscope.cn/datasets/iic/MSAgent-Pro)
- [toolbench](https://www.modelscope.cn/datasets/swift/ToolBench)
- [ms-agent](https://www.modelscope.cn/datasets/iic/ms_agent)
- [ms-agent-for-agentfabric](https://www.modelscope.cn/datasets/AI-ModelScope/ms_agent_for_agentfabric)
- [ms-agent-multirole](https://www.modelscope.cn/datasets/iic/MSAgent-MultiRole)
- [toolbench-for-alpha-umi](https://www.modelscope.cn/datasets/shenweizhou/alpha-umi-toolbench-processed-v2)
- [damo-agent-zh](https://www.modelscope.cn/datasets/iic/MSAgent-Bench)
- [agent-instruct-all-en](https://www.modelscope.cn/datasets/huangjintao/AgentInstruct_copy)

## Custom Agent Datasets

### ToolBench Format

The ToolBench format mainly consists of:
1. System part with the API list (tools/system)
2. User request part (user)
3. Model response part (assistant)
   - Note: The model response needs to include the fields `Action: some-api-to-call Action Input: some-parameter-to-input` for API parsing. SWIFT does not support automatic generation of this part. Because the model's responses may contain additional information, please refer to the second point below for details.
   - It is advisable to include a CoT (Chain of Thought) process in the `Thought:` section to improve learning effectiveness, for example: `Thought: I think I know how to complete this task. Since the user needs xxx, the first step is xxx, the second step...`, followed by the Action section.
4. Call part (tool) - This part is actually called by the user and returned to the model (or called by the agent framework instead). It represents the actual API return value.
5. The model's response section (assistant) analyzes the API return values and provides conclusions based on that analysis.
   - Note that it may contain secondary calls.
   - It is recommended to include multi-turn call data in the dataset to enhance the model's ability to handle complex tasks.
   - A minimum of two dialogues is required for ToolBench to address a user request.

```jsonl
{"tools":"{API_LIST}","conversations": [{"role": "user", "content": "Help me to order a ticket"}, {"role": "assistant", "content": "Thought: I need to call some API to book a ticket Action: xxx Action Input: xxx"}, {"role": "tool", "content": "{'response': 'ok'}"}, {"role": "assistant", "content": "I think the task is finished."}]}
{"tools":"{API_LIST}","conversations": [{"role": "user", "content": "I want to search google to know to weather"}, {"role": "assistant", "content": "Thought: I need to use google to search weather Action: xxx Action Input: xxx"}, {"role": "tool", "content": "{'response': 'weather: xxx, temperature: xxx'}"}, {"role": "assistant", "content": "I think I know the answer, the weather is xxx."}]}
```

SWIFT will concatenate the content of tools into the system field. If you want to define the tools content in the system yourself, you can remove the tools field and add a system field in the conversations:

```jsonl
{"conversations": [{"role": "system", "content": "You have the following tools to use, tool1: xxx, tool2: xxx"}, {"role": "user", "content": "Help me to order a ticket"}, {"role": "assistant", "content": "Thought: I need to call some API to book a ticket Action: xxx Action Input: xxx"}, {"role": "tool", "content": "{'response': 'ok'}"}, {"role": "assistant", "content": "I think the task is finished."}]}
{"conversations": [{"role": "system", "content": "You have the following tools to use, tool1: xxx, tool2: xxx"}, {"role": "user", "content": "I want to search google to know to weather"}, {"role": "assistant", "content": "Thought: I need to use google to search weather Action: xxx Action Input: xxx"}, {"role": "tool", "content": "{'response': 'weather: xxx, temperature: xxx'}"}, {"role": "assistant", "content": "I think I know the answer, the weather is xxx."}]}
```

### ReACT Format

The ReACT format mainly consists of:
1. System part containing the API list (tools/system)
2. User request part (user)
3. Model response part (assistant)
   - Note: The model response needs to include the fields `Action: some-api-to-call Action Input: some-parameter-to-input Observation: api-response` for API parsing and execution. SWIFT does not support automatic generation of this part as well, due to the inclusion of potentially extra information.
   - It is advisable to include a CoT process in the `Thought:` section for better learning, such as `Thought: I think I know how to complete this task. Since the user needs xxx, the first step is xxx, the second step...`, followed by the Action part.
   - The difference between the ReACT format and the ToolBench format is that there is no `tool` role in the ReACT format. After `Observation:`, the user needs to concatenate the API call results, and then the model provides feedback based on that content. Therefore, in the model's response section, it may include multiple calls (multiple Thought/Action/Action Input/Observation).
   - ReACT requires at least one round of conversation to fulfill a user request.

```jsonl
{"tools":"{API_LIST}","conversations": [{"role": "user", "content": "Help me to order a ticket"}, {"role": "assistant", "content": "Thought: I need to call some API to book a ticket Action: xxx Action Input: xxx Observation: {'response': 'ok'} Final Answer: I think the task is finished."}]}
{"tools":"{API_LIST}","conversations": [{"role": "user", "content": "I want to search google to know to weather"}, {"role": "assistant", "content": "Thought: I need to use google to search weather Action: xxx Action Input: xxx Observation: {'response': 'weather: xxx, temperature: xxx' Final Answer: I think I know the answer, the weather is xxx."}]}
```

The ReACT format also supports user-defined system parts, similar to the ToolBench description.

## Specific Format of the Tools Field
The ReACT format similarly supports providing the system section independently, which can refer to the relevant descriptions in ToolBench.
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

ToolBench tools format looks like this:
```json
{
"tools": [
      {
        "name": "url_for_newapi",
        "description": "This is the subfunction for tool \"newapi\", you can use this tool. The description of this function is: \"url_for_newapi\"",
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
        "description": "This is the subfunction for tool \"newapi\", you can use this tool. The description of this function is: \"n_for_newapiew var\"",
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
    ]
}
```

During inference, the tools information is converted into the corresponding tools system prompt. If a system prompt already exists, it will replace the original content.

Currently, SWIFT supports ReACT in English, ReACT in Chinese, and ToolBench formats, as illustrated below.

ReACT-EN
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

ReACT-ZH
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
First I will give you the task description, and your task starts.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually execute your step. Your output should follow this format:
Thought:
Action:
Action Input:

After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you will finally perform the task, then you can give your final answer.
Remember:
1. The state change is irreversible; you can’t go back to one of the former states. If you want to restart the task, say "I give up and restart."
2. All thoughts are short, at most in 5 sentences.
3. You can do more than one try, so if your plan is to continue trying some conditions, you can do one of the conditions per try.
Let’s Begin!
Task description: You should use functions to help handle the real-time user queries. Remember:
1. ALWAYS call the "Finish" function at the end of the task. The final answer should contain enough information to show to the user. If you can’t handle the task or find that function calls always fail (the function is not valid now), use the Finish function to give up and restart.
2. Do not use original tool names; use only subfunctions’ names.
Specifically, you have access to the following APIs: {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}}
```

By default, the ReACT-EN format is used. You can also specify `--tools_prompt` as `react_zh` or `toolbench` to select the Chinese ReACT or ToolBench format.

If you have better tools system prompt suggestions, please share or contribute them to us.

## Related Training Techniques

SWIFT offers the following techniques to improve agent training:

### loss-scale(--loss_scale)

This technique adjusts the training weights for portions of model output. For example, in ReACT format, `--loss_scale react` can be set. The function of this parameter includes:

Weights for the Thought and Final Answer sections are set to 1; weights for Action and Action Input sections are set to 2; the weight for the Observation: field itself is 2, while the weight for the actual API call response following Observation: is 0.

For a detailed explanation of the loss_scale plugin design, please refer to the [pluginization](../Customization/Pluginization.md) documentation.

### tools(--tools_prompt)

The tools section corresponds to the format of the system field after assembly. In addition to the previously mentioned react_en/react_zh/toolbench, it also supports glm4 format. Furthermore, users can define their own format, tools_prompt, which can also refer to the documentation in the [Pluginization](../Customization/Pluginization.md) section.

For a complete agent training script, see [this link](https://github.com/modelscope/ms-swift/tree/main/examples/train/agent/train.sh).

## Inference

We conducted an evaluation focusing on general knowledge and agents. Below is a simple summary of the evaluation results.

### Original Model

#### General Knowledge

> How to make vinegar fish from West Lake?

![image-20240201122323540](../../resources/image-20240201122323540.png)

> What is the difference between COVID-19 and a common cold?

![image-20240201122441874](../../resources/image-20240201122441874.png)

#### Agent Capabilities

For testing, we used a fire alarm scenario:

```text
Answer the following questions as best you can. You have access to the following APIs:
1. fire_recognition: Call this tool to interact with the fire recognition API. This API is used to recognize whether there is fire in the image. Parameters: [{"name": "image", "description": "The input image to recognize fire", "required": "True"}]

2. fire_alert: Call this tool to interact with the fire alert API. This API will start an alert to warn the building's administrators. Parameters: []

3. call_police: Call this tool to interact with the police calling API. This API will call 110 to catch the thief. Parameters: []

4. call_fireman: Call this tool to interact with the fireman calling API. This API will call 119 to extinguish the fire. Parameters: []

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of the above tools [fire_recognition, fire_alert, call_police, call_fireman]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
```

![image-20240201122625473](../../resources/image-20240201122625473.png)

![image-20240201122725477](../../resources/image-20240201122725477.png)

![image-20240201131811038](../../resources/image-20240201131811038.png)

As observed, the model's answers were incorrect when manually entering the Observation.

### After Training

#### General Knowledge

> How to make vinegar fish from West Lake?

![image-20240201132124061](../../resources/image-20240201132124061.png)

![image-20240201132139698](../../resources/image-20240201132139698.png)

> What is the difference between COVID-19 and a common cold?

![image-20240201132308260](../../resources/image-20240201132308260.png)

#### Agent Capabilities

![image-20240201132421298](../../resources/image-20240201132421298.png)

![image-20240201132454465](../../resources/image-20240201132454465.png)

After training, the model can correctly invoke the API and provide a final answer.

## Deployment
Here’s an example using vLLM deployment, non-streaming calls, with ReACT prompt.

```shell
swift deploy \
  --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --infer_backend pt
```

Using a curl command to call the interface, due to the ReACT format ending with Observation:, we need to specify `Observation:` as stop words to truncate the model's response. Some models treat `Observation:\n` as a token, and here we will also consider it as a stop word.

If you are using ToolBench prompt, specifying stop words is unnecessary (though it's fine to include them).

```shell
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Meta-Llama-3.1-8B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "What\'s the weather like in Boston today?"
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

You can also specify the `tool_choice` field to select a tool from the tools, such as `"tool_choice":{"type": "function", "function": {"name": "my_function"}}`. By default, all tools are selected, or you can set it to None to disable the tools field.

Call result:
```json
{"model":"/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3___1-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Thought: I need to get the current weather in Boston.\nAction: get_current_weather\nAction Input: location=\"Boston, MA\", unit=\"fahrenheit\"\nObservation:","tool_calls":[{"function":{"name":"get_current_weather","arguments":" location=\"Boston, MA\", unit=\"fahrenheit\"\n"},"type":"function","id":"toolcall-cb4082f072fa43b7a1182c30482bf9b7"}]},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":213,"completion_tokens":35,"total_tokens":248},"id":"chatcmpl-fef66ae309c142b3b96a221a7f1cc424","object":"chat.completion","created":1733225385}
```

In the returned result's tool_calls, you can find the called function and argument information.

You can also test using the OpenAI SDK:
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
        "description": "This is the subfunction for tool \"newapi\", you can use this tool. The description of this function is: \"url_for_newapi\"",
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
    model='Meta-Llama-3.1-8B-Instruct',
    tools = tools,
    messages=messages,
    seed=42)
tool_calls = resp.choices[0].message.tool_calls[0]
print(f'query: {query}')
print(f'tool_calls: {tool_calls}')

# Stream
stream_resp = client.chat.completions.create(
    model='Meta-Llama-3.1-8B-Instruct',
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
tool_calls: ChatCompletionMessageToolCall(id='toolcall-d750fa08b4cf4cd2906afa05bf45062a', function=Function(arguments=' https://weather.com/weather/today/l/USMA0001:1:US\n', name='url_for_newapi'), type='function')
"""
```

Once you assume the returned result is `The weather in Boston today is 32°F (0°C), with clear skies`, We will fill in the "role tool" field with the result and pass it into the message.

```shell
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Meta-Llama-3.1-8B-Instruct",
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

For the ReACT format, we will append the result to the last `Observation:` field returned by the previous model.

For the ToolBench format, processing depends on the model template. If the model template doesn't specify special handling for the field, it will be treated as user input.

Call result:
```json
{"model":"/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3___1-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"\n\nResponse: The weather in Boston today is 32°F (0°C), with clear skies.","tool_calls":null},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":93,"completion_tokens":20,"total_tokens":113},"id":"chatcmpl-8d48c8949fd74fc6ae0c5198ed171359","object":"chat.completion","created":1733225830}
```

If you want to combine code and tools to complete the entire closed-loop, we recommend reading the [OpenAI tutorial](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models).
