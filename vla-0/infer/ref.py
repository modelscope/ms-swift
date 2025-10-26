import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_dir = snapshot_download('Qwen/Qwen3-VL-4B-Instruct')

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, dtype="auto", device_map="auto",
    # attn_implementation='flash_attention_2',
)

processor = AutoProcessor.from_pretrained(model_dir)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4",
                "max_pixels": 128*32*32,
                "max_frames": 16,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True,
                                                                image_patch_size= 16,
                                                                return_video_metadata=True)
if video_inputs is not None:
    video_inputs, video_metadatas = zip(*video_inputs)
    video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
else:
    video_metadatas = None
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
inputs = inputs.to('cuda')

generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
# 'A baby wearing glasses sits on a bed, engrossed in reading a book. The baby turns the pages with both hands, occasionally looking up and smiling. The room is cozy, with a crib in the background and clothes scattered around. The baby’s focus and curiosity are evident as they explore the book, creating a heartwarming scene of early learning and discovery.'