在模型训练过程中，数据及数据处理是最为重要的工作之一。在当前模型训练流程趋于成熟的情况下，数据集的好坏，是决定了该次训练能否成功的最关键因素。

在上一篇中，我们提到了模型训练的基本原理是将文字转换索引再转换为对应的向量，那么文字转为向量的具体过程是什么？

# Tokenizer

在NLP（自然语言处理）领域中，承担文字转换索引（token）这一过程的组件是tokenizer。每个模型有自己特定的tokenizer，但它们的处理过程是大同小异的。

比如我们使用“千问14B”模型将“杭州是个好地方”转为tokens的具体方式是：

```shell
pip install modelscope ms-swift -U
```

在python中调用：

```python
from modelscope import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-14B", trust_remote_code=True)
print(tokenizer('杭州是个好地方'))
# {'input_ids': [104130, 104104, 52801, 100371], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
```

其中的input_ids就是上面我们说的文字的token。可以注意到token的个数少于实际文字的数量，这是因为在转为token的时候，并不是一个汉字一个token的，可能会将部分词语变为一个token，也可能将一个英文转为两部分（如词根和时态），所以token数量和文字数量不一定对得上。

# Template

每种模型有其特定的输入格式，在小模型时代，这种输入格式比较简单：

```text
[CLS]杭州是个好地方[SEP]
```

[CLS]代表了句子的起始，[SEP]代表了句子的终止。在BERT中，[CLS]的索引是101，[SEP]的索引是102，加上中间的部分，在BERT模型中整个的索引序列是：

```text
101, 100, 1836, 100, 100, 100, 1802, 1863, 102
```

我们可以看到，这个序列和上面千问的序列是不同的，这是因为这两个模型的词表不同。

在LLM时代，base模型的格式和上述的差不多，但chat模型的格式要复杂的多，比如千问chat模型的template格式是：

```text
<|im_start|>system
You are a helpful assistant!
<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
```

其中“You are a helpful assistant!”是system字段，“How are you?”是用户问题，其他的部分都是template的格式。

system字段是chat模型必要的字段，这个字段会以命令方式提示模型在下面的对话中遵循怎么样的范式进行回答，比如：

```text
“You are a helpful assistant!”
“下面你是一个警察，请按照警察的要求来审问我”
“假如你是一个爱哭的女朋友，下面的对话中清扮演好这个角色”
```

system字段规定了模型行为准则，比如当模型作为Agent使用时，工具集也是定义在system中的：

```text
“你是一个流程的执行者，你有如下工具可以使用：
工具1：xxx，输入格式是：xxx，输出格式是：xxx，作用是：xxx
工具2：xxx，输入格式是：xxx，输出格式是：xxx，作用是：xxx”
```

复杂的template有助于模型识别哪部分是用户输入，哪部分是自己之前的回答，哪部分是给自己的要求。

比较麻烦的是，目前各开源模型还没有一个统一的template标准。在SWIFT中，我们提供了绝大多数模型的template：

```python
register_template(
    TemplateType.default,
    Template([], ['### Human:\n', '{{QUERY}}\n\n', '### Assistant:\n'],
             ['\n\n'], [['eos_token_id']], DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n']))

# You can set the query as '' to serve as a template for pre-training.
register_template(TemplateType.default_generation,
                  Template([], ['{{QUERY}}'], None, [['eos_token_id']]))
register_template(
    TemplateType.default_generation_bos,
    Template([['bos_token_id']], ['{{QUERY}}'], None, [['eos_token_id']]))

qwen_template = Template(
    [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
    ['<|im_end|>\n'], ['<|im_end|>'], DEFAULT_SYSTEM,
    ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
register_template(TemplateType.qwen, qwen_template)
register_template(TemplateType.chatml, deepcopy(qwen_template))
...
```

有兴趣的小伙伴可以阅读：https://github.com/modelscope/swift/blob/main/swift/llm/utils/template.py 来获得更细节的信息。

template拼接好后，直接传入tokenizer即可。

微调任务是标注数据集，那么必然有指导性的labels（模型真实输出）存在，将这部分也按照template进行拼接，就会得到类似下面的一组tokens：

```text
input_ids: [34,   56,   21,   12,   45,   73, 96, 45, 32, 11]
           ---------用户输入部分---------   ----模型真实输出----
labels:    [-100, -100, -100, -100, -100, 73, 96, 45, 32, 11]
```

在labels中，我们将用户输入的部分（问题）替换成了-100，保留了模型输入部分。在模型进行运算时，会根据input_ids的前面的tokens去预测下一个token，就比如：

```text
已知token                    预测的下一个token
34                        ->17
34,56                     ->89
...
34,56,21,12,45            ->121
34,56,21,12,45,73         ->99
34,56,21,12,45,73,96      ->45
34,56,21,12,45,73,96,45   ->14
34,56,21,12,45,73,96,45,32->11
```

可以看到，这个预测不一定每个都预测对了，而且呈现了下三角的形态。那么训练的时候就可以：

```text
34,   56,   21,   12,   45,  121,  99,  45,  32,  11
-100, -100, -100, -100, -100, 73,  96,  45,  14,  11
```

-100部分计算loss时会被忽略，因为这是用户输入，不需要考虑预测值是什么。只要对比下对应的位置对不对就可以计算它们的差异了，这个差异值被称为**loss**或者**残差**。我们通过计算梯度的方式对参数进行优化，使模型参数一点一点向真实的未知值靠近。
