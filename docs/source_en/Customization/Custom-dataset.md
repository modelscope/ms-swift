# Custom Dataset

The standard format for the ms-swift dataset accepts the following keys: 'messages', 'rejected_response', 'label', 'images', 'videos', 'audios', 'tools', and 'objects'. Among these, 'messages' is a required key. 'rejected_response' is used for RLHF training like DPO, 'label' is used for KTO training, while 'images', 'videos', and 'audios' are used for storing paths or URLs of multimodal data. 'tools' is for Agent tasks, and 'objects' is for grounding tasks.

There are three core preprocessors in ms-swift: `MessagesPreprocessor`, `AlpacaPreprocessor`, and `ResponsePreprocessor`. `MessagesPreprocessor` converts datasets in messages and sharegpt formats to the standard format, `AlpacaPreprocessor` converts alpaca format datasets, and `ResponsePreprocessor` converts datasets in query/response format. `AutoPreprocessor` automatically selects the appropriate preprocessor for processing. Typically, `AutoPreprocessor` can handle over 90% of cases.

The following four formats will all be converted to the messages field in the ms-swift standard format by `AutoPreprocessor`:

Messages format:
```jsonl
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "<response1>"}, {"role": "user", "content": "<query2>"}, {"role": "assistant", "content": "<response2>"}]}
```

ShareGPT format:
```jsonl
{"system": "<system>", "conversation": [{"human": "<query1>", "assistant": "<response1>"}, {"human": "<query2>", "assistant": "<response2>"}]}
```

Alpaca format:
```jsonl
{"system": "<system>", "instruction": "<query-inst>", "input": "<query-input>", "output": "<response>"}
```

Query-Response format:
```jsonl
{"system": "<system>", "query": "<query2>", "response": "<response2>", "history": [["<query1>", "<response1>"]]}
```

There are three ways to integrate a custom dataset, with increasing control over preprocessing functions:
1. **Recommended**: Directly use `--dataset <dataset_id_or_path>` to integrate with AutoPreprocessor. This supports csv, json, jsonl, txt, and folder formats.
2. Write a dataset_info.json file. You can refer to the built-in [dataset_info.json](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/data/dataset_info.json) in ms-swift. One of ms_dataset_id/hf_dataset_id/dataset_path is required, and column name conversion can be handled through the `columns` field. Format conversion uses AutoPreprocessor. Use `--custom_dataset_info xxx.json` to parse the JSON file.
3. Manually register the dataset, which offers the most flexible customization of preprocessing functions but is more complex. You can refer to examples in [examples](https://github.com/modelscope/swift/blob/main/examples/custom) by specifying `--custom_register_path xxx.py` to parse the registration contents.

## Recommended Dataset Format

Here is the recommended dataset format for ms-swift:

### Pre-training

```jsonl
{"messages": [{"role": "assistant", "content": "I love music"}]}
{"messages": [{"role": "assistant", "content": "Coach, I want to play basketball"}]}
{"messages": [{"role": "assistant", "content": "Which is more authoritative, tomato and egg rice or the third fresh stir-fry?"}]}
```

### Supervised Fine-tuning

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "Tomorrow's weather will be sunny"}]}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}, {"role": "assistant", "content": "It equals 3"}]}
```

### RLHF

#### DPO/ORPO/CPO/SimPO/RM

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "Tomorrow's weather will be sunny"}], "rejected_response": "I don't know"}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}, {"role": "assistant", "content": "It equals 3"}], "rejected_response": "I don't know"}
```

#### KTO

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "I don't know"}], "label": false}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}, {"role": "assistant", "content": "It equals 3"}], "label": true}
```

### Multimodal

For multimodal datasets, the format is the same as above. The difference is that it includes the keys `images`, `videos`, and `audios`, which represent multimodal resources:
```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "<image> What is in the image? <video> What is in the video?"}, {"role": "assistant", "content": "An elephant and a lion"}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
```
The `<image>`, `<video>`, and `<audio>` tags indicate where to insert images/videos/audios.

#### Grounding

For grounding (object detection) tasks, SWIFT supports two methods:
1. Maintain consistency with the above multimodal dataset format, adding special characters in the dataset, for example:
```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "<image> Find a <ref> elephant </ref>"}, {"role": "assistant", "content": "<box>(200,450),(500,800)</box>"}], "images": ["/xxx/x.jpg"]}
```
With this type of data, please note:
  - Grounding tasks often require special characters. You need to determine which model to use, read the model paper to identify special characters for grounding tasks, and combine the data accordingly.
  - The bbox coordinates may use actual image coordinates or thousandth coordinates. Please confirm this before assembling data.
  - Different models require different data formats; you need to recreate the dataset if switching models.

2. Use SWIFT's grounding data format:

```jsonl
# Object detection
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "<image> Identify <bbox>"}, {"role": "assistant", "content": "<ref-object>"}], "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]"}
# Grounding to multiple bboxes
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "<image> Find <ref-object>"}, {"role": "assistant", "content": "<bbox>"}], "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [[138, 136, 235, 359], [1,2,3,4]], \"bbox_type\": \"real\", \"image\": 0}]"}
```

This format adds the objects field, which includes:
 - caption: description of the object corresponding to the bbox
 - bbox: coordinates, suggested as four integers (not floats), representing x_min, y_min, x_max, y_max
 - bbox_type: bbox type, currently supporting three types: real/norm_1000/norm_1, representing actual pixel coordinates/thousandth ratio coordinates/normalized ratio coordinates
 - image: index of the image corresponding to the bbox, starting from 0

### Text-to-Image Format

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Draw me an apple"}, {"role": "assistant", "content": "<image>"}], "images": ["/xxx/x.jpg"]}
```

### Agent Format

Refer to the [Agent documentation](../Instruction/Agent-support.md) for the Agent format.
