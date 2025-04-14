以下为如何使用训练后Agent模型的简易教程：

## 方案一：使用swift app

1. 输入以下shell，启动app-ui：

```shell
CUDA_VISIBLE_DEVICES=0 \
    swift app \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --max_new_tokens 2048 \
    --verbose true \
    --stop_words 'Observation:'
```

2. 将以下内容输入system中，点击重置system并清空历史记录：
```
Answer the following questions as best you can. You have access to the following APIs:
1. TouristGuide: Call this tool to interact with the TouristGuide API. What is the TouristGuide API useful for? 旅游指南API，根据用户指定的条件查询目的地的旅游信息. Parameters: [{"name": "destination", "description": "指定需要查询的目的地，例如巴黎、纽约等", "required": "True"}, {"name": "attraction", "description": "指定需要查询的景点，例如埃菲尔铁塔、自由女神像等", "required": "False"}, {"name": "food", "description": "指定需要查询的美食，例如法国香槟、美国汉堡等", "required": "False"}, {"name": "hotel", "description": "指定需要查询的酒店，例如五星级、四星级等", "required": "False"}]

2. newsfeed: Call this tool to interact with the newsfeed API. What is the newsfeed API useful for? 获取指定主题的新闻列表. Parameters: [{"name": "topic", "description": "需要查询的新闻主题", "required": "False"}]

3. poemgen: Call this tool to interact with the poemgen API. What is the poemgen API useful for? 生成优美的诗歌. Parameters: [{"name": "theme", "description": "诗歌主题（例如：爱情、自然、季节等）", "required": "False"}]

4. Converter: Call this tool to interact with the Converter API. What is the Converter API useful for? 通过Python解释器进行单位转换. Parameters: [{"name": "from_unit", "description": "原单位", "required": "True"}, {"name": "to_unit", "description": "目标单位", "required": "True"}, {"name": "value", "description": "需要转换的数值", "required": "True"}]

5. musicPlaylist: Call this tool to interact with the musicPlaylist API. What is the musicPlaylist API useful for? 音乐播放列表API，提供多种音乐类型的播放列表. Parameters: [{"name": "type", "description": "音乐类型，例如流行、摇滚、古典等", "required": "True"}, {"name": "mood", "description": "音乐风格，例如抒情、动感、欢快等", "required": "False"}, {"name": "artist", "description": "歌手名字", "required": "False"}]

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of the above tools[TouristGuide, newsfeed, poemgen, Converter, musicPlaylist]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
```

3. 输入用户请求：`将200英里转换为公里`，并发送信息。模型将思考调用哪一个工具完成这一工作以及输出调用工具时所需的参数。遇到`Observation:`时终止输出，等待工具返回调用内容。
```
Action: Converter
Action Input: {'from_unit': '英里', 'to_unit': '公里', 'value': 200}
Observation:
```

4. 模拟工具的返回，输入`tool:{'function_result': {'km': 321.8688}}`，模型将继续输入，并得到最终结果。
```
Thought: I now know the final answer
Final Answer: 200英里等于321.8688公里。
```

## 方案二：使用swift infer

1. 输入以下shell，启动命令行交互式推理界面

```shell
CUDA_VISIBLE_DEVICES=0 \
    swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --max_new_tokens 2048 \
    --stop_words 'Observation:'
```

2. 依次输入以下内容：
```
<<< multi-line
[INFO:swift] End multi-line input with `#`.
[INFO:swift] Input `single-line` to switch to single-line input mode.
<<<[M] reset-system#
<<<[MS] Answer the following questions as best you can. You have access to the following APIs:
1. translate: Call this tool to interact with the translate API. What is the translate API useful for? 将一种语言翻译成另一种语言. Parameters: [{"name": "text", "description": "需要翻译的文本", "required": "False"}, {"name": "source_lang", "description": "源语言，可选参数，默认为自动检测", "required": "False"}, {"name": "target_lang", "description": "目标语言，必选参数", "required": "False"}]

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of the above tools[translate]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!#
<<<[M] single-line#
<<< 翻译成法语：你好，我叫小明
Action: translate
Action Input: {'text': '你好，我叫小明', 'source_lang': 'auto', 'target_lang': 'fr'}
Observation:
--------------------------------------------------
<<< tool:{'translated_text': 'Bonjour, je m\\'appelle Xiao Ming'}
Thought: I now know the final answer
Final Answer: 根据您的要求，我已经将“你好，我叫小明”翻译成了法语。翻译结果为“Bonjour, je m'appelle Xiao Ming”。
```

## 方案三：使用Python

```python
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig
    model = 'Qwen/Qwen2.5-3B'
    adapters = ['output/vx-xxx/checkpoint-xxx']

    engine = PtEngine(model, max_batch_size=64, adapters=adapters)
    system = """Answer the following questions as best you can. You have access to the following APIs:
1. trailFinder: Call this tool to interact with the trailFinder API. What is the trailFinder API useful for? API for finding nearby hiking trails based on user input.. Parameters: [{"name": "location", "description": "User's current location.", "required": "True"}, {"name": "distance", "description": "Maximum distance from user's location.", "required": "False"}, {"name": "difficulty", "description": "Specify the difficulty level of the trail.", "required": "False"}]

2. Factorial calculator: Call this tool to interact with the Factorial calculator API. What is the Factorial calculator API useful for? 计算正整数的阶乘. Parameters: [{"name": "n", "description": "需要计算阶乘的正整数", "required": "False"}]

3. weather: Call this tool to interact with the weather API. What is the weather API useful for? 天气查询API，查询指定城市的实时天气情况. Parameters: [{"name": "city", "description": "指定查询的城市名称", "required": "False"}, {"name": "date", "description": "指定查询的日期", "required": "False"}]

4. English to Chinese Translator: Call this tool to interact with the English to Chinese Translator API. What is the English to Chinese Translator API useful for? 将英文翻译成中文. Parameters: [{"name": "english_text", "description": "需要翻译的英文文本", "required": "True"}, {"name": "target_language", "description": "目标语言（默认为中文）", "required": "False"}]

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of the above tools[trailFinder, Factorial calculator, weather, English to Chinese Translator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
"""
    request_config = RequestConfig(max_tokens=512, temperature=0, stop=['Observation:'], stream=True)
    messages = [{'role': 'system', 'content': system}]
    query = '北京今天的天气怎么样？'
    messages += [{'role': 'user', 'content': query}]
    gen_list = engine.infer([InferRequest(messages=messages)], request_config)
    response = ''
    tool = '{"temperature": 72, "condition": "Sunny", "humidity": 50}\n'
    print(f'query: {query}')
    for resp in gen_list[0]:
        if resp is None:
            continue
        delta = resp.choices[0].delta.content
        response += delta
        print(delta, end='', flush=True)
    tool = "{'temp': 25, 'description': 'Partly cloudy', 'status': 'success'}"
    print(tool)
    messages += [{'role': 'assistant', 'content': response}, {'role': 'tool', 'content': tool}]
    gen_list = engine.infer([InferRequest(messages=messages)], request_config)
    for resp in gen_list[0]:
        if resp is None:
            continue
        print(resp.choices[0].delta.content, end='', flush=True)
    print()
"""
query: 北京今天的天气怎么样？
Action: weather
Action Input: {'city': '北京', 'date': '今天'}
Observation:{'temp': 25, 'description': 'Partly cloudy', 'status': 'success'}
Thought: I now know the final answer
Final Answer: 根据API调用结果，北京今天的天气是部分多云，温度为25摄氏度。
"""
```
