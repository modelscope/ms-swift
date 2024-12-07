# New Dataset

## Local Dataset

### Pre-trained Format

```jsonl
{"messages": [{"role": "assistant", "content": "I love music"}]}
{"messages": [{"role": "assistant", "content": "Coach, I want to play basketball"}]}
{"messages": [{"role": "assistant", "content": "Which is more authoritative, tomato and egg rice or the three fresh dishes?"}]}
```

### Fine-tuning Format

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me the weather for tomorrow"}, {"role": "assistant", "content": "Tomorrow's weather is sunny"}]}
{"messages": [{"role": "system", "content": "You are a useful and harmless calculator"}, {"role": "user", "content": "What is 1+1"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1 again?"}, {"role": "assistant", "content": "It equals 3"}]}
```

### Human Alignment Format

#### DPO/ORPO/CPO/SimPO

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me the weather for tomorrow"}, {"role": "assistant", "content": "Tomorrow's weather is sunny"}], "rejected_response": "I don't know"}
{"messages": [{"role": "system", "content": "You are a useful and harmless calculator"}, {"role": "user", "content": "What is 1+1"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1 again?"}, {"role": "assistant", "content": "It equals 3"}], "rejected_response": "I don't know"}
```

#### KTO

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me the weather for tomorrow"}, {"role": "assistant", "content": "I don't know"}], "label": false}
{"messages": [{"role": "system", "content": "You are a useful and harmless calculator"}, {"role": "user", "content": "What is 1+1"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1 again?"}, {"role": "assistant", "content": "It equals 3"}], "label": true}
```

For multimodal datasets, the format remains the same as above, with the addition of several keys like `images`, `videos`, and `audios`, which represent multimodal resources. For example:

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "<image> What is in the picture? <video> What is in the video?"}, {"role": "assistant", "content": "An elephant, a lion"}], "images": ['/xxx/x.jpg'], "videos": ['/xxx/x.mov']}
```

In which the `<image>` and `<video>` tags represent the insertion points for images. The multimodal training of SWIFT supports the mixed use of multiple resources and modalities.

For grounding (object detection) tasks, SWIFT supports two methods:
1. Keeping the format consistent with the above multimodal dataset format while adding special characters to the dataset, for instance:

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "<image> Find <ref> an elephant </ref>"}, {"role": "assistant", "content": "<box>(200,450),(500, 800)</box>"}], "images": ['/xxx/x.jpg']}
```
When using this type of data, please be mindful of:
  - Grounding tasks often require special characters; you need to determine which model to use and read the model paper to confirm which special characters are used for grounding tasks and how to merge data.
  - The bbox coordinates may either use the real coordinates of the image or use thousandths coordinates; please clarify this before merging data.
  - Different models require different data formats; if you switch models, please recreate the dataset.

2. Using SWIFT's grounding data format:

```jsonl
# Object detection
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "<image> Identify <bbox>"}, {"role": "assistant", "content": "<ref-object>"}], "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }
# Grounding to multiple bboxes
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "<image> Find <ref-object>"}, {"role": "assistant", "content": "<bbox>"}], "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [[138, 136, 235, 359], [1,2,3,4]], \"bbox_type\": \"real\", \"image\": 0}]" }
```

This format contains an additional `objects` field that includes:
 - `caption`: the description of the object corresponding to the bbox
 - `bbox`: coordinates; it is recommended to provide four integers (rather than float) representing `x_min`, `y_min`, `x_max`, `y_max`.
 - `bbox_type`: the type of bbox; currently supports three types: `real/norm_1000/norm_1`, representing actual pixel coordinates/thousandth ratio coordinates/normalized ratio coordinates.
 - `image`: indicates which image the bbox corresponds to, with indexing starting from 0.

### Text-to-Image Format

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Draw me an apple"}, {"role": "assistant", "content": "<image>"}], "images": ["/xxx/x.jpg"]}
```

### Agent Format

The Agent format is more complex. Please refer to the [Agent Documentation](../Instruction/Agent-support.md).

## Register Hub Dataset

### Simple Data Format

You can refer to the [built-in dataset_info.json in swift](https://github.com/modelscope/swift/blob/main/swift/llm/dataset/data/dataset_info.json) for dataset expansion. You can directly add to the built-in dataset_info.json, or you can use `--custom_dataset_info 1.json` to specify the path, JSON string or dictionary of an external dataset_info.json.

```json
[
    {
        "ms_dataset_id": "AI-ModelScope/xxx",
        "hf_dataset_id": "my-group/xxx",
        "columns": {
            "question": "query",
            "answer": "response"
        }
    }
]
```

### Complex Data Format

For examples, please refer to [examples](https://github.com/modelscope/swift/blob/main/examples/custom/dataset.py). You can parse the registered content by specifying `--custom_register_path xxx.py`.
