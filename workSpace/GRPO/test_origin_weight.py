import requests
import json
import yaml
from openai import AsyncOpenAI
import asyncio

with open('/mnt/cfs/ssw/ljc/gits/config.yaml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)

tools_path = '/mnt/cfs/ssw/ljc/dataset_making/Generator/config/tool_list.json'
with open(tools_path, 'r', encoding="utf-8") as f:
    TOOLS = json.load(f)

async def test_output(input):
    url = "http://localhost:8000/v1/"
    plan_prompt = config['system_prompt']['system_prompt_planner']

    msg = [
        {"role": "system", "content": plan_prompt},
        {"role": "user", "content": input}
    ]

        # 异步openai request 
    client = AsyncOpenAI(
        base_url=url,
        api_key="EMPTY"
    )

    try:
        chat_response = await client.chat.completions.create(
            model = "Qwen3-14B-test",
            tools = TOOLS,
            messages = msg,
            max_tokens = 3000,
            top_p = 0.6,
            temperature = 0.7,
            stream=False,
            # stream_options={"include_usage": True},
            extra_body= 
            {
                "repetition_penalty": 1.1,
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )

        return chat_response
    except Exception as e:
        return None

if __name__ == "__main__":
    input = "最近我们要考八年级下册的期中考试了，我担心听写这块会挂科呢！你有没有办法帮我的忙？比如找35个来自《壶口瀑布》这篇的核心词。"
    res = asyncio.run(test_output(input))
    print(res)
