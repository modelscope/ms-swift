from swift.infer_engine import InferRequest, TransformersEngine

query = '甲、乙、丙三个人在一起做作业，有一道数学题比较难，当他们三个人都把自己的解法说出来以后，' \
        '甲说："我做错了。"乙说："甲做对了。"丙说："我做错了。"' \
        '在一旁的丁看到他们的答案并听了她们的意见后说："你们三个人中有一个人做对了，有一个人说对了。"' \
        '请问，他们三人中到底谁做对了？'

infer_requests = [
    InferRequest(messages=[
        {'role': 'user', 'content': query},
        {'role': 'assistant', 'content': '天哪，这道题真的很有趣！ 🤔 让我们来分析一下他们的对话：'
         ' 1. 甲说："我做错了。"如果甲真的做错了，那么乙说"甲做对了"就是错误的。'
         '但是，如果甲做对了，那么乙说"甲做对了"就是正确的。'
         ' 2. 乙说："甲做对了。"如果乙真的认为甲做对了，那么甲说"我做错了"就是错误的。'
         ' 3. 丙说："我做错了。"这个信息暂时不太重要，但我们可以注意到丙没有说其他人做对或做错。'
         ' 现在，丁出现了，说："你们三个人中有一个人做对了，有一个人说对了。"'
         ' 如果甲做对了，那么乙说对了，丙做错了。但是，这样的话，丁的说法就不对了，因为有两个说对了。'
         ' 如果甲做错了，那么乙说错了，丙做对了。这时候，丁的说法才正确，因为有一个人做对了（丙），有一个人说对了（乙）。'
         ' 所以，答案是：丙做对了！ 👏'
         ' PS：这道题真的很有趣，需要我们细心分析他们的对话！ 😄'},
    ]),
    InferRequest(messages=[
        {'role': 'user', 'content': query},
        {'role': 'assistant', 'content': "What a delightful puzzle! Let's dive into the magical world of logic and figure out "
         "who's correct and who's not. So, we have three friends, 甲 (Ka), 乙 (Otsu), and 丙 (Chi), working on a tricky "
         "math problem together. They each share their solutions, and then they start discussing who's correct. "
         "Here's what they say: * 甲 (Ka) says, \"I did it wrong.\" * 乙 (Otsu) says, \"Ka did it right.\" "
         "* 丙 (Chi) says, \"I did it wrong.\" Now, 丁 (Tei), who's been observing from the side, chimes in and says, "
         "\"Among the three of you, one person got it right, and one person said it correctly.\" "
         "Let's break it down step by step: "
         "1. If 甲 (Ka) did it wrong, then 乙 (Otsu) must be telling the truth, because 乙 says 甲 did it right. "
         "2. But wait, if 乙 is telling the truth, that means 甲 actually did it right! This contradicts 甲's statement, "
         "so 甲 must be lying. "
         "3. Now, 丙 (Chi) says they did it wrong. If 丙 did it wrong, then 丙 is telling the truth. "
         "But if 丙 did it right, then 丙 would be lying. "
         "4. Since 丁 says one person got it right and one person said it correctly, it means 丙 must be the one who did "
         "it right. If 丙 did it right, then 丙's statement \"I did it wrong\" is actually a lie. "
         "5. So, we have 甲 lying about doing it wrong, 乙 telling the truth about 甲, and 丙 lying about doing it wrong. "
         "The magical conclusion is that 丙 (Chi) is the one who actually got the math problem correct!"},
    ]),
]

engine = TransformersEngine(
    'Qwen/Qwen2.5-7B-Instruct',
    task_type='seq_cls',
    num_labels=1,
    problem_type='regression',
    adapters=['output/vx-xxx/checkpoint-xxx'])

responses = engine.infer(infer_requests)
rewards = [response.choices[0].message.content for response in responses]
print(f'rewards: {rewards}')
