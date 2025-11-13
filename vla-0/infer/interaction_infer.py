import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image


model_dir = '/home/yuquan002/ssd/ms-swift-robotics/output/qwen3-vl-4b-instruct-vla0-libero/v6-20251029-202919/checkpoint-105984'

model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype="auto", device_map="auto")

processor = AutoProcessor.from_pretrained(model_dir)
main_dir = "/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla/images/00000000_main.jpg"
main = Image.open(main_dir).convert("RGB")
wrist_dir = "/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla/images/00000000_wrist.jpg"
wrist = Image.open(wrist_dir).convert("RGB")

while True:
    print("-" * 128)
    input_text = input("Please enter your text: ")
    print("Processing input...")
    print('-' * 128)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that helps people find information."
         },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": main},
                {"type": "image", "image": wrist},
                {"type": "text", "text": input_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info([messages], image_patch_size=16)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    inputs = inputs.to('cuda')

    generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("-" * 128)
    print("Output Content:")
    print(output_text[0])
    print("-" * 128)