## Wandb settings
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
[INFO:swift] The logging file will be saved in: /mnt/cfs/ssw/ljc/ms-swift/output_ppo/v28-20250819-185527/logging.jsonl
Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 8. Using DeepSpeed's value.
wandb: WARNING The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.
wandb: Tracking run with wandb version 0.21.0
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.

wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /mnt/cfs/ssw/ljc/ms-swift/wandb/offline-run-20250819_185603-c151g2zj
wandb: Find logs at: wandb/offline-run-20250819_185603-c151g2zj/logs

## completions 格式

### FROM wandb log
1. val_dataset
```json
{
    'messages': [
        {'role': 'system', 'content': '...'},
        {'role': 'assistant', 'content': '...'}
    ],
    'tools': "json_format"
}
```

2. completions
{'step': list, 'prompt': list, 'completion': clist, 'ToolCallsCosineReward': list}
    !!! plugin completion 数据流一致
    clist: [completion0, completion1, ...]
        completion0 ntc format: 
        ```
        <think>
        好的，用户输入的是“D I s c o v E r.”，看起来像是字母的混合大小写，可能想表达“Discover”这个词。首先，我需要确定用户的具体需求。可能的情况是用户想进行某种探索或发现，但需要更明确的信息。

        根据之前的规则，我需要检查是否有合适的工具来调用。用户可能是在寻找学习资料、练习题、古诗背诵或其他学科内容。但当前的查询比较模糊，没有明确的学科或具体问题。

        接下来，我需要判断是否需要向用户请求更多信息。根据规则，只有工具定义中"required"字段的参数才需要补充。但当前用户输入没有提供足够的信息来调用任何工具，比如学科、年级、具体知识点等。

        因此，正确的做法是不调用工具，而是用礼貌、专业的语气回复用户，引导他们提供更具体的需求。例如，询问他们想探索哪个学科，或者有什么具体的问题需要帮助。

        同时，要确保回复符合中小学生的认知水平，避免使用复杂术语，保持简洁和友好。不能提及任何外部资源或链接，也不能引导用户使用其他应用或网站。

        总结，用户可能需要更明确的指示，所以回复应请求他们澄清需求，例如询问具体学科或问题，以便提供有效的帮助。
        </think>

        您好！看起来您想探索一些知识呢~请问您具体想了解哪个学科的内容呢？比如语文、数学、英语，或者有特别想了解的知识点吗？这样我就能为您找到合适的学习资料啦！
        ```
        completion0 tc format:
        ```
        <think>
        好的，我现在需要处理用户的请求。用户明确提到需要六年级（五四制）的题目，教材版本是人教版（五·四学制），并且主题要贴合教学重点。首先，我得确认用户的需求是否明确，以及是否需要调用工具来获取相关信息。

        首先，用户提到的“五四制”是指中国的一种学制，与传统的“三三二一”学制不同，可能涉及到课程安排和教材版本的不同。用户特别指定了教材版本为人教版，所以需要确保工具调用时使用正确的参数来匹配这个版本。

        接下来，用户需要的是六年级的题目，但五四学制的六年级可能对应的是传统的五年制中的六年级，或者可能有不同的年级划分。需要确认工具是否支持按学制和教材版本来筛选题目。例如，工具可能有参数如grade（年级）、textbook_version（教材版本）、education_system（学制）等。

        然后，用户提到主题要贴合教学重点，这意味着需要获取的题目应该覆盖当前学期的重点知识点。可能需要调用题目搜索工具，参数包括年级、教材版本、学制以及可能的学科和知识点。例如，使用search_tbl_question工具，参数可能包括grade、textbook_version、education_system，以及可能的subject和knowledge_point。

        需要检查工具的参数是否包含这些字段，并且是否为必填项。例如，如果工具的required字段中包含grade和textbook_version，那么必须提供这些参数。而education_system可能不是必填，但用户明确指定了五四制，所以需要包含进去。

        另外，用户可能没有明确提到学科，但根据上下文，可能默认是数学或语文，但为了保险起见，可能需要进一步确认。不过根据用户之前的例子，可能默认是数学，或者需要根据工具的参数来判断。如果工具允许不指定学科，可能需要默认处理，或者在调用工具时留空，让工具自行处理。

        现在，我需要调用search_tbl_question工具，参数包括count（题目数量）、grade（六年级）、textbook_version（人教版）、education_system（五四制），以及可能的subject和knowledge_point。但用户没有明确提到学科和知识点，所以可能需要默认处理，或者工具可能自动根据年级和教材版本来获取相关题目。

        但根据工具的参数定义，可能需要指定subject，比如数学、语文等。用户没有明确说明，但可能需要根据常见的教学重点来推断。例如，六年级数学可能涉及分数、比例、几何等，语文可能涉及阅读理解、作文等。但用户没有具体说明，所以可能需要调用工具时留空，或者根据工具的默认行为处理。

        不过，根据工具的参数要求，可能必须提供subject，所以可能需要进一步询问用户，但根据规则，只有required字段中的参数才需要询问。如果subject不是必填项，那么可以留空，或者根据工具的默认值处理。

        综上所述，正确的工具调用应该是search_tbl_question，参数包括count（用户可能需要一定数量的题目，但用户没有明确说明数量，可能需要默认值，比如5道），grade为六年级，textbook_version为人教版，education_system为五四制。如果这些参数都是必填的，那么需要确保它们被正确填入。

        检查工具的参数定义，假设search_tbl_question的required字段包括grade、textbook_version，而education_system可能不是必填，但用户明确指定了，所以需要包含。因此，调用工具时应包含这些参数，并生成相应的function call。
        </think>

        {"function": {"name": "search_tbl_question", "arguments": {"count": 5, "grade": "六年级", "textbook_version": "人教版", "education_system": "五四学制"}}, "index": 0, "id": "call_1234567890", "type": "function"}
        ```

### FROM dataset
    {
        "messages": [
            {
                "role": "user",
                "content": "我现在正忙其他事情，来点快速完成的任务！搞10道选择题，适合初二知识点就行，别耽误时间。"
            }
        ],
        "tools": "",
        "system": ""
    }

## VLLM原生输出
ChatCompletion(id='chatcmpl-9622516b11ca4a019d9813604722ae09', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='chatcmpl-tool-8c6e7aae0bd64e6f98fecbe8d9e7a0b1', function=Function(arguments='{"name": "才干"}', name='dictionary_search'), type='function')], reasoning_content='\n好的，用户问的是“才干的拼音”。首先，我需要确认用户的需求是什么。看起来他们想知道“才干”这个词的正确发音。接下来，我需要检查是否有相关的工具可以帮助回答这个问题。\n\n看一下可用的工具列表，有dictionary_search这个工具，它的功能是搜索汉字、词语或成语的读音、写法、释义等信息。根据工具的描述，它确实可以提供拼音，所以这个工具应该是合适的。\n\n接下来，我需要确定调用这个工具所需的参数。dictionary_search的参数需要一个"name"字段，也就是要查询的字词。用户的问题里明确提到了“才干”，所以参数应该是{"name": "才干"}。\n\n另外，检查是否有其他可能的工具。比如query_en_word是用来查英语单词的，这里不适用。其他工具如search_tbl_question或generate_tbl_question也不相关，因为用户不是在找题目。因此，唯一合适的选择就是dictionary_search。\n\n确认参数是否齐全。由于dictionary_search的required字段是["name"]，而用户已经提供了“才干”这个词，所以参数足够，不需要进一步询问用户补充信息。\n\n最后，生成对应的tool_call，调用dictionary_search函数，参数为{"name": "才干"}。这样就能获取到“才干”的拼音和其他相关信息，然后可以整理成用户容易理解的回答。\n'), stop_reason=None)], created=1755833178, model='Qwen3-14B-test', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=293, prompt_tokens=8107, total_tokens=8400, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=None, kv_transfer_params=None)

    评估过程中出现错误: 'function'





"历史", "地理"], "type": "string"}, "year": {"title": "年份", "type": "string"}}, "required": ["keywords"]}}}]'}]
[WARNING:swift] Processing reward function: ToolCallsCosineReward
[WARNING:swift] reward_kwargs keys: ['messages', 'tools', 'is_truncated', 'global_step']
reward_kwargs keys: ['messages', 'tools', 'is_truncated', 'global_step']
[WARNING:swift] reward_kwargs messages count: 8
reward_kwargs messages

[INFO:swift] inputs: StdTemplateInputs(messages=[{'role': 'user', 'content': '点、线、面、体之间的构成有什么公式或原理吗？比如点多少就能形成一定的线或面？'}], system='.....省略\n', tools=None, rejected_response=None, label=None, channel=None, images=[], audios=[], videos=[], objects={})


## multi turn 问题
[rank1]: Traceback (most recent call last):
[rank1]:   File "/mnt/cfs/ssw/ljc/ms-swift/swift/cli/rlhf.py", line 5, in <module>
[rank1]:     rlhf_main()
[rank1]:   File "/mnt/cfs/ssw/ljc/ms-swift/swift/llm/train/rlhf.py", line 172, in rlhf_main
[rank1]:     return SwiftRLHF(args).main()
[rank1]:   File "/mnt/cfs/ssw/ljc/ms-swift/swift/llm/base.py", line 49, in main
[rank1]:     result = self.run()
[rank1]:   File "/mnt/cfs/ssw/ljc/ms-swift/swift/llm/train/sft.py", line 112, in run
[rank1]:     trainer = trainer_cls(
[rank1]:   File "/mnt/cfs/ssw/ljc/ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py", line 171, in __init__
[rank1]:     assert self.args.multi_turn_scheduler in multi_turns
[rank1]: AssertionError