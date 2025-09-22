# Custom Dataset

There are three methods for accessing custom datasets, each offering progressively greater control over preprocessing functions but also increasing in complexity. For example, Solution 1 is the most convenient but offers the least control over preprocessing functions, requiring prior conversion of the dataset into a specific format:

1. **Recommended**: Directly use the command line parameter to access the dataset with `--dataset <dataset_path1> <dataset_path2>`. This will use `AutoPreprocessor` to convert your dataset into a standard format (supporting four dataset formats; see the introduction to AutoPreprocessor below). You can use `--columns` to transform column names. The supported input formats include csv, json, jsonl, txt, and folders (e.g. git clone open-source datasets). This solution does not require modifying `dataset_info.json` and is suitable for users new to ms-swift. The following two solutions are suitable for developers looking to extend ms-swift.
2. Add the dataset to `dataset_info.json`, which you can refer to in the built-in [dataset_info.json](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/data/dataset_info.json) of ms-swift. This solution also uses AutoPreprocessor to convert the dataset to a standard format. `dataset_info.json` is a list of metadata for datasets, and one of the fields ms_dataset_id/hf_dataset_id/dataset_path must be filled. Column name transformation can be done through the `columns` field. Datasets added to `dataset_info.json` or registered ones will automatically generate [supported dataset documentation](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html) when running [run_dataset_info.py](https://github.com/modelscope/ms-swift/blob/main/scripts/utils/run_dataset_info.py). In addition, you can use the external `dataset_info.json` approach by parsing the JSON file with `--custom_dataset_info xxx.json` (to facilitate users who prefer `pip install` over `git clone`), and then specify `--dataset <dataset_id/dataset_dir/dataset_path>`.
3. Manually register the dataset to have the most flexible customization capability for preprocessing functions, allowing the use of functions to preprocess datasets, but it is more difficult. You can refer to the [built-in datasets](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/dataset/llm.py) or [examples](https://github.com/modelscope/ms-swift/blob/main/examples/custom). You can specify `--custom_register_path xxx.py` to parse external registration content (convenient for users who use pip install instead of git clone).
   - Solutions one and two leverage solution three under the hood, where the registration process occurs automatically.

The following is an introduction to the dataset formats that `AutoPreprocessor` can handle:

The standard dataset format for ms-swift accepts keys such as: 'messages', 'rejected_response', 'label', 'images', 'videos', 'audios', 'tools', and 'objects'. Among these, 'messages' is a required key. 'rejected_response' is used for DPO and other RLHF training, 'label' is used for KTO training and classification model training. The keys 'images', 'videos', and 'audios' are used to store paths or URLs for multimodal data, 'tools' is used for Agent tasks, and 'objects' is used for grounding tasks.

There are three core preprocessors in ms-swift: `MessagesPreprocessor`, `AlpacaPreprocessor`, and `ResponsePreprocessor`. `MessagesPreprocessor` is used to convert datasets in the messages and sharegpt format into the standard format. `AlpacaPreprocessor` converts datasets in the alpaca format, while `ResponsePreprocessor` converts datasets in the query/response format. `AutoPreprocessor` automatically selects the appropriate preprocessor for the task.

The following four formats will all be converted into the `messages` field of the ms-swift standard format under the processing of `AutoPreprocessor`, meaning they can all be directly used with `--dataset <dataset-path>`:

Messages format (standard format):
```jsonl
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "<response1>"}, {"role": "user", "content": "<query2>"}, {"role": "assistant", "content": "<response2>"}]}
```
- Note: The system part is optional. The system in the dataset has a higher priority than the `--system` passed through the command line, followed by the `default_system` defined in the template.

ShareGPT format:
```jsonl
{"system": "<system>", "conversation": [{"human": "<query1>", "assistant": "<response1>"}, {"human": "<query2>", "assistant": "<response2>"}]}
```

Query-Response format:
```jsonl
{"system": "<system>", "query": "<query2>", "response": "<response2>", "history": [["<query1>", "<response1>"]]}
```
Note: The following fields will be automatically converted to the corresponding system, query, and response fields.
- system: 'system', 'system_prompt'.
- query: 'query', 'prompt', 'input', 'instruction', 'question', 'problem'.
- response: 'response', 'answer', 'output', 'targets', 'target', 'answer_key', 'answers', 'solution', 'text', 'completion', 'content'.

Alpaca format:
```jsonl
{"system": "<system>", "instruction": "<query-inst>", "input": "<query-input>", "output": "<response>"}
```
- Note: The instruction and input fields will be combined into the query field. If instruction and input are not empty strings, then `query = f'{instruction}\n{input}'`.


## Standard Dataset Format

The following outlines the standard dataset format for ms-swift, where the "system" field is optional and uses the "default_system" defined in the template by default. The four dataset formats introduced earlier can also be processed by AutoPreprocessor into the standard dataset format.

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

- You can control whether the loss is computed for specific parts of the model's response by adding the `"loss"` field (requires ms-swift >= 3.8). This field defaults to `None`. If `"loss"` is set to `true`, the corresponding content will contribute to the loss calculation (equivalent to a `loss_scale` of 1). If `"loss"` is set to `false`, the corresponding content will be excluded from loss computation. Note that this feature only takes effect for messages where `"role"` is `"assistant"`, and it has higher priority than the command-line argument `--loss_scale`. Example data format:

```jsonl
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi, how can I help you?", "loss": false}, {"role": "user", "content": "What is 1+1?"}, {"role": "assistant", "content": "It equals 2", "loss": true}]}
```

#### Channel Loss
If you want to use channel loss, you need to set `--enable_channel_loss true` and add a "channel" field to your dataset. Channel loss is compatible with techniques such as packing, padding-free, and loss scaling.

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "Tomorrow's weather will be sunny"}], "channel": "general"}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}, {"role": "assistant", "content": "It equals 3"}], "channel": "math"}
```

### RLHF

#### DPO/ORPO/CPO/SimPO/RM

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "Tomorrow's weather will be sunny"}], "rejected_response": "I don't know"}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}, {"role": "assistant", "content": "It equals 3"}], "rejected_response": "I don't know"}
```

The format of multimodal data should follow the specifications in [Multimodal Dataset](#multimodal), with additional columns such as `images` to represent other modality inputs. When it is necessary to associate different image information with preference data, the `rejected_images` field can be used to indicate the images related to the rejected responses. In the alignment dataset, at least one of `rejected_images` or `rejected_response` must be provided for each entry.

> Note: RM additionally supports the margin column. For details, refer to the [RM documentation](../Instruction/RLHF.md#rm).

Sure, you can also directly use `rejected_messages` instead of only providing `rejected_response` / `rejected_images` (requires ms-swift>=3.8), which offers greater flexibility (e.g., for multimodal or agent scenarios). If you use "rejected_messages", then in multimodal scenarios you must also provide "rejected_images", "rejected_audios", "rejected_videos", etc.; in Agent scenarios you must also provide "rejected_tools", etc. An example of the multimodal data format is as follows:

```jsonl
{"messages": [{"role": "user", "content": "<image>What is this?"}, {"role": "assistant", "content": "This is a kitten."}], "images": ["kitten.png"], "rejected_messages": [{"role": "user", "content": "<image>What is this?"}, {"role": "assistant", "content": "This is a puppy."}], "rejected_images": ["kitten.png"]}
{"messages": [{"role": "user", "content": "<image>What is this?"}, {"role": "assistant", "content": "This is a kitten."}], "images": ["kitten.png"], "rejected_messages": [{"role": "user", "content": "<image>What is this?"}, {"role": "assistant", "content": "This is a kitten."}], "rejected_images": ["puppy.png"]}
```

The above format is equivalent to:

```jsonl
{"messages": [{"role": "user", "content": "<image>What is this?"}, {"role": "assistant", "content": "This is a kitten."}], "images": ["kitten.png"], "rejected_response": "This is a puppy."}
{"messages": [{"role": "user", "content": "<image>What is this?"}, {"role": "assistant", "content": "This is a kitten."}], "images": ["kitten.png"], "rejected_images": ["puppy.png"]}
```

#### KTO

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "I don't know"}], "label": false}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}, {"role": "assistant", "content": "It equals 3"}], "label": true}
```

#### PPO/GRPO

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}]}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
{"messages": [{"role": "user", "content": "What is your name?"}]}
```
- Note: GRPO will pass through all additional field content to the ORM, unlike other training methods that, by default, delete extra fields. For example, you can additionally pass in 'solution'. The custom ORM needs to include a positional argument called `completions`, with other arguments as keyword arguments passed through from the additional dataset fields.

#### GKD

If `seq_kd` is not enabled, i.e., the parameter is set to False, the dataset format is as follows (you can use a teacher model to pre-distill the data):

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}, {"role": "assistant", "content": "Tomorrow's weather will be sunny"}]}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}, {"role": "assistant", "content": "It equals 3"}]}
```

If `seq_kd` is enabled, the final round of the 'assistant' part is not required (the teacher model generates data during training):

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}]}
{"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
```

### Sequence Classification

**Single-label Task**:
```jsonl
{"messages": [{"role": "user", "content": "The weather is really nice today"}], "label": 1}
{"messages": [{"role": "user", "content": "Today is really unlucky"}], "label": 0}
{"messages": [{"role": "user", "content": "So happy"}], "label": 1}
```

**Multi-label Task**:

```jsonl
{"messages": [{"role": "user", "content": "<sentence>"}], "label": [1, 3, 5]}
```

**Single Regression Task**:

```jsonl
{"messages": [{"role": "user", "content": "Calculate the similarity between two sentences, with a range of 0-1.\nsentence1: <sentence1>\nsentence2: <sentence2>"}], "label": 0.8}
```

**Multi Regression Task**:

```jsonl
{"messages": [{"role": "user", "content": "<sentence>"}], "label": [1.2, -0.6, 0.8]}
```

### Embedding

Please refer to [Embedding training document](../BestPractices/Embedding.md#dataset-format).

### Reranker

Please refer to [Reranker training document](../BestPractices/Reranker.md#dataset-format).

### Multimodal

For multimodal datasets, the format is the same as the aforementioned tasks. The difference lies in the addition of several keys: `images`, `videos`, and `audios`, which represent the URLs or paths (preferably absolute paths) of multimodal resources. The tags `<image>`, `<video>`, and `<audio>` indicate where to insert images, videos, or audio. MS-Swift supports multiple images, videos, and audio files. These special tokens will be replaced during preprocessing, as referenced [here](https://github.com/modelscope/ms-swift/blob/main/swift/llm/template/template/qwen.py#L198). The four examples below respectively demonstrate the data format for plain text, as well as formats containing image, video, and audio data.


Pre-training:
```jsonl
{"messages": [{"role": "assistant", "content": "Pre-trained text goes here"}]}
{"messages": [{"role": "assistant", "content": "<image>is a puppy, <image>is a kitten"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
{"messages": [{"role": "assistant", "content": "<audio>describes how nice the weather is today"}], "audios": ["/xxx/x.wav"]}
{"messages": [{"role": "assistant", "content": "<image>is an elephant, <video>is a lion running"}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
```

Supervised Fine-tuning:

```jsonl
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang?"}, {"role": "assistant", "content": "The capital of Zhejiang is Hangzhou."}]}
{"messages": [{"role": "user", "content": "<image><image>What is the difference between the two images?"}, {"role": "assistant", "content": "The first one is a kitten, and the second one is a puppy."}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
{"messages": [{"role": "user", "content": "<audio>What did the audio say?"}, {"role": "assistant", "content": "The weather is really nice today."}], "audios": ["/xxx/x.mp3"]}
{"messages": [{"role": "system", "content": "You are a helpful and harmless assistant."}, {"role": "user", "content": "<image>What is in the image, <video>What is in the video?"}, {"role": "assistant", "content": "The image shows an elephant, and the video shows a puppy running on the grass."}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
```
- Note: The following fields will be automatically converted to the corresponding images, videos, and audios fields.
  - images: image, images.
  - videos: video, videos.
  - audios: audio, audios.
- If you need to pass base64 data instead of file paths, here are sample examples: `"videos": ['data:video/mp4;base64,{base64_encoded}']`, `"images": ['data:image/jpg;base64,{base64_encoded}']`.

The data format for RLHF and sequence classification of multimodal models can reference the format of pure text large models, with additional fields such as `images` added on top of that.

#### Grounding

For grounding (object detection) tasks, ms-swift supports two methods:

1. Directly use the data format of the grounding task corresponding to the model. For example, the format for qwen2-vl is as follows:

```
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>Describe the image."}, {"role": "assistant", "content": "<|object_ref_start|>a dog<|object_ref_end|><|box_start|>(221,423),(569,886)<|box_end|> and <|object_ref_start|>a woman<|object_ref_end|><|box_start|>(451,381),(733,793)<|box_end|> are playing on the beach"}], "images": ["/xxx/x.jpg"]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>Find the <|object_ref_start|>sheep<|object_ref_end|> in the image"}, {"role": "assistant", "content": "<|box_start|>(101,201),(150,266)<|box_end|><|box_start|>(401,601),(550,666)<|box_end|>"}], "images": ["/xxx/x.jpg"]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>Help me open Google Chrome"}, {"role": "assistant", "content": "Action: click(start_box='<|box_start|>(246,113)<|box_end|>')"}], "images": ["/xxx/x.jpg"]}
```

When using this type of data, please note:

- Different models have different special characters and data format for the grounding task.
- The handling of bounding box normalization varies across different models: for example, qwen2.5-vl uses absolute coordinates, while qwen2-vl and internvl2.5 require bounding box coordinates to be normalized to the thousandth scale.
  - Note: Qwen2.5-VL uses absolute coordinates, so you need to be careful with image resizing each time. If you use the dataset format from Option 1, you need to resize the images in advance (height and width must be multiples of 28) and scale the coordinates accordingly. If you use the dataset format from Option 2, ms-swift will handle image resizing for you. You can still use `MAX_PIXELS` or `--max_pixels` for image resizing (training only; for inference, you still need to handle image resizing yourself).

2. Use ms-swift's grounding data format:

```
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>Describe the image."}, {"role": "assistant", "content": "<ref-object><bbox> and <ref-object><bbox> are playing on the beach"}], "images": ["/xxx/x.jpg"], "objects": {"ref": ["a dog", "a woman"], "bbox": [[331.5, 761.4, 853.5, 1594.8], [676.5, 685.8, 1099.5, 1427.4]]}}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>Find the <ref-object> in the image"}, {"role": "assistant", "content": "<bbox><bbox>"}], "images": ["/xxx/x.jpg"], "objects": {"ref": ["sheep"], "bbox": [[90.9, 160.8, 135, 212.8], [360.9, 480.8, 495, 532.8]]}}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>Help me open Google Chrome"}, {"role": "assistant", "content": "Action: click(start_box='<bbox>')"}], "images": ["/xxx/x.jpg"], "objects": {"ref": [], "bbox": [[615, 226]]}}
```

The format will automatically convert the dataset format to the corresponding model's grounding task format and select the appropriate model's bbox normalization method. Compared to the general format, this format includes an additional "objects" field, which contains the following subfields:

- ref: Used to replace `<ref-object>`.
- bbox: Used to replace `<bbox>`. If the length of each box in the bbox is 2, it represents the x and y coordinates. If the box length is 4, it represents the x and y coordinates of two points.
  - Note: `<ref-object>` and `<bbox>` do not have a corresponding relationship; references and bounding boxes replace their own placeholders separately.
- bbox_type: Optional values are 'real' and 'norm1'. The default is 'real', meaning the bbox represents the actual bounding box value. If set to 'norm1', the bbox is normalized to the range 0~1.
- image_id: This parameter is only effective when bbox_type is 'real'. It indicates the index of the image corresponding to the bbox, used for scaling the bbox. The index starts from 0, and the default is 0 for all.

Testing the final format of the grounding data in ms-swift format:
```python
import os
os.environ["MAX_PIXELS"] = "1003520"
from swift.llm import get_model_tokenizer, get_template

_, tokenizer = get_model_tokenizer('Qwen/Qwen2.5-VL-7B-Instruct', load_model=False)
template = get_template(tokenizer.model_meta.template, tokenizer)
data = {...}
template.set_mode('train')
encoded = template.encode(data, return_template_inputs=True)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
print(f'images: {encoded["template_inputs"].images}')
```

### Text-to-Image Format

```jsonl
{"messages": [{"role": "system", "content": "You are a useful and harmless assistant"}, {"role": "user", "content": "Draw me an apple"}, {"role": "assistant", "content": "<image>"}], "images": ["/xxx/x.jpg"]}
```

### Agent Format
Here are example data samples for a text-only Agent and a multimodal Agent:
```jsonl
{"tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"Weather forecast. Get real-time air quality, including current air quality, PM2.5, and PM10 information.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"City name, e.g., Shanghai\"}}, \"required\": [\"city\"]}}}]", "messages": [{"role": "user", "content": "What is the weather like in Beijing and Shanghai today?"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"Beijing\"}}"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"Shanghai\"}}"}, {"role": "tool_response", "content": "{\"city\": \"Beijing\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, {"role": "tool_response", "content": "{\"city\": \"Shanghai\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, {"role": "assistant", "content": "According to the weather forecast tool, the air quality index (AQI) in Beijing is 10, which indicates good air quality; whereas in Shanghai, the AQI is 72, indicating mild pollution."}]}
{"tools": "[{\"type\": \"function\", \"function\": {\"name\": \"click\", \"description\": \"Click on a position on the screen\", \"parameters\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"integer\", \"description\": \"X-coordinate representing the horizontal position on the screen\"}, \"y\": {\"type\": \"integer\", \"description\": \"Y-coordinate representing the vertical position on the screen\"}}, \"required\": [\"x\", \"y\"]}}}]", "messages": [{"role": "user", "content": "<image>What time is it now?"}, {"role": "assistant", "content": "<think>\nI can check the current time by opening the calendar app.\n</think>\n"}, {"role": "tool_call", "content": "{\"name\": \"click\", \"arguments\": {\"x\": 105, \"y\": 132}}"}, {"role": "tool_response", "content": "{\"images\": \"<image>\", \"status\": \"success\"}"}, {"role": "assistant", "content": "Successfully opened the calendar app. The current time is 11 o'clock in the morning."}], "images": ["desktop.png", "calendar.png"]}
```
- When the `agent_template` is set to "react_en", "hermes", etc., this format is compatible with training for all model Agents and allows easy switching between different models.
- Among them, `tools` is a JSON string containing a list of tools, and the `content` section of `messages` where the `role` is `'tool_call'` or `'tool_response/tool'` must also be a JSON string.
- The `tools` field will be combined with the `{"role": "system", ...}` section during training/inference according to the `agent_template`, forming a complete system section.
- The `{"role": "tool_call", ...}` part will automatically be converted into corresponding formats of `{"role": "assistant", ...}` based on the `agent_template`. Multiple consecutive `{"role": "assistant", ...}` entries will be concatenated to form a complete assistant_content.
- The `{"role": "tool_response", ...}` can also be written as `{"role": "tool", ...}`, these two forms are equivalent. This part will also be automatically converted according to the `agent_template`. During training, this part does not participate in loss calculations, similar to `{"role": "user", ...}`.
- This format supports parallel tool calls; refer to the first data sample for an example. In multimodal Agent data samples, the number of `<image>` tags should match the length of "images", and their positions indicate where the image features are inserted. It also supports other modalities, such as audios and videos.
- Note: You can also manually process the data into the messages format with roles set to system, user, or assistant. The purpose of agent_template is to automatically map the tools field and the messages with roles tool_call and tool_response into the standard messages format with roles system, user, and assistant.
- For more details, please refer to [Agent Documentation](../Instruction/Agent-support.md).


## dataset_info.json

You can refer to the ms-swift built-in [dataset_info.json](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/data/dataset_info.json). This approach uses the AutoPreprocessor function to convert the dataset into a standard format. The dataset_info.json file contains a list of metadata about the dataset. Here are some examples:


```json
[
  {
    "ms_dataset_id": "xxx/xxx"
  },
  {
    "dataset_path": "<dataset_dir/dataset_path>"
  },
  {
    "ms_dataset_id": "<dataset_id>",
    "subsets": ["v1"],
    "split": ["train", "validation"],
    "columns": {
      "input": "query",
      "output": "response"
    }
  },
  {
    "ms_dataset_id": "<dataset_id>",
    "hf_dataset_id": "<hf_dataset_id>",
    "subsets": [{
      "subset": "subset1",
      "columns": {
        "problem": "query",
        "content": "response"
      }
    },
    {
      "subset": "subset2",
      "columns": {
        "messages": "_",
        "new_messages": "messages"
      }
    }]
  }
]
```

The following parameters are supported:

- ms_dataset_id: Refers to the DatasetMeta parameter.
- hf_dataset_id: Refers to the DatasetMeta parameter.
- dataset_path: Refers to the DatasetMeta parameter.
- dataset_name: Refers to the DatasetMeta parameter.
- subsets: Refers to the DatasetMeta parameter.
- split: Refers to the DatasetMeta parameter.
- columns: Transforms column names before preprocessing the dataset.

## Dataset Registration

`register_dataset` will register the dataset in `DATASET_MAPPING`. You can call the function `register_dataset(dataset_meta)` to complete the dataset registration, where `dataset_meta` will store the metadata of the model. The parameter list for DatasetMeta is as follows:

- ms_dataset_id: The dataset_id for ModelScope, default is None.
- hf_dataset_id: The dataset_id for HuggingFace, default is None.
- dataset_path: The local path to the dataset (an absolute path is recommended), default is None.
- dataset_name: The alias of the dataset, which can be specified via `--dataset <dataset_name>`. This is very convenient when the dataset_path is long. The default value is None.
- subsets: A list of subdataset names or a list of `SubsetDataset` objects, default is `['default']`. (The concepts of subdatasets and splits only exist for dataset_id or dataset_dir (open source datasets cloned via git)).
- split: Defaults to `['train']`.
- preprocess_func: A preprocessing function or callable object, default is `AutoPreprocessor()`. This preprocessing function takes an `HfDataset` as input and returns an `HfDataset` in the standard format.
- load_function: Defaults to `DatasetLoader.load`. If a custom loading function is needed, it should return an `HfDataset` in the standard format, allowing users maximum flexibility while bypassing the ms-swift dataset loading mechanism. This parameter usually does not need to be modified.
