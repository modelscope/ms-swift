Tools字段，支持OpenAI和ToolBench格式，需要提供Tool的名字，描述和参数

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
ToolBench tools:
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

*非流式调用*
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
      "stream": true
  }'
```

调用结果
```json
{"model":"llama3-8b-instruct","choices":[[{"index":0,"message":{"role":"assistant","content":"Question: What's the weather like in Boston today?\n\nThought: I need to get the current weather in Boston to answer this question.\n\nAction: get_current_weather\n\nAction Input: {'location': 'Boston, MA', 'unit': 'fahrenheit'}\n\nObservation:","tool_calls":{"id":"cmpl-70b95df710dd4e73abc4452647efa728","function":{"arguments":" {'location': 'Boston, MA', 'unit': 'fahrenheit'}","name":" get_current_weather"},"type":"function"}},"finish_reason":"stop"}]],"usage":{"prompt_tokens":262,"completion_tokens":54,"total_tokens":316},"id":"chatcmpl-8bf2eb8c38bf4b70a52520bb74997348","object":"chat.completion","created":1717037776}
```

*流式调用* 设置stream为true

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
  "tools": 
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
      "stream": true,
      "stop": ["Observation:", "Observation:\n"]
  }''
```

调用结果
```json
data:{"model": "llama3-8b-instruct", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "'fahrenheit'}\n\nObservation:", "tool_calls": {"id": "cmpl-99cbafaa0c744a449d57b240757c0776", "function": {"arguments": " {'location': 'Boston, MA', 'unit': 'fahrenheit'}", "name": " get_current_weather"}, "type": "function"}}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 262, "completion_tokens": 54, "total_tokens": 316}, "id": "chatcmpl-9d5c1c0d2f104459bf7d1b5b0ce3f85e", "object": "chat.completion.chunk", "created": 1717037941}
```

