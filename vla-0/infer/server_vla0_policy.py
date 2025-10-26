# server_vla0_policy.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import collections
import logging

import numpy as np
import torch
import tyro
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from websocket_vla_server import WebsocketPolicyServer

# --- VLA-0 Inferencing ---
class VLA0Policy:
    """
    The VLA-0 Policy class for performing inference using the VLA-0 model.
    This class encapsulates:
    1. Model invocation, converting images and instructions into action text.
    2. Action decoding, converting text into normalized continuous actions.
    3. Ensemble prediction, smoothing action outputs.
    """
    def __init__(self, model, processor, device, ensemble_size=5):
        self.model = model
        self.processor = processor
        self.device = device

        '''
        Action normalization and quantization strategy:
            As the action space is tested to be within [-1, 1] in all dimensions,
            we directly scale the action to [0, 1000] using the formula:
            norm_action = (action + 1.0) / 2.0
            quantized_action = round(norm_action * action_quantization_bins)
        To invert:
            dequantized_action = (quantized_action / action_quantization_bins) * 2 - 1.0
        '''

        # match the action normalization used during data preparation
        self.ensemble_size = ensemble_size
        self.action_chunk_history = collections.deque(maxlen=self.ensemble_size)
        self.action_quantization_bins = 1000  # B
        self.action_min = -1.0  # assuming action space min
        self.action_max = 1.0   # assuming action space max
        
        # Construct system prompt and prompt template
        # The system prompt instructs the model on how to format its output
        self.system_prompt = (
            "Analyze the input image and predict robot actions for the next 5 timesteps. "
            "Each action has 7 dimensions. Output a single sequence of 35 integers "
            f"(0 - {self.action_quantization_bins} each), representing the 5 timesteps sequentially. "
            "Provide only space-separated numbers. Nothing else."
        )
        self.prompt_template = "task description: {instruction}\n\nimage:\n<image>\n<image>"

    def _decode_and_unnormalize(self, action_text: str):
        """
        Decode the action text into quantized integers and then unnormalize them back to continuous action values.
        """
        try:
            # 1. Decode the space-separated text into integers
            quantized_action = np.fromstring(action_text, dtype=int, sep=' ')
            # 2. Unnormalize the quantized action back to continuous values
            norm_action = quantized_action / self.action_quantization_bins
            continuous_action = norm_action * (self.action_max - self.action_min) + self.action_min
            return continuous_action
        except Exception:
            logging.error(f"Unable to parse action text: '{action_text}'")
            # Return a safe zero action
            return np.zeros_like(self.action_min)

    @torch.inference_mode()
    def infer(self, obs: dict):
        """Run a single inference step."""
        # 1. Extract data from the observation
        main_img = Image.fromarray(obs["observation/image"])
        wrist_img = Image.fromarray(obs["observation/wrist_image"])
        instruction = obs["prompt"]
        
        '''
        Prompt example from dataset:
        {"messages": [{"role": "system", "content": "Analyze the input image and predict robot actions for the next 1 timesteps. Each action has 7 dimensions. Output a single sequence of 7 integers (0 - 1000 each), representing the 1 timesteps sequentially. Provide only space-separated numbers. Nothing else."}, {"role": "user", "content": "task description: pick up the black bowl next to the cookie box and place it on the plate<image><image>"}, {"role": "assistant", "content": "566 480 500 500 475 500 0"}], "images": ["images/00000000_main.jpg", "images/00000000_wrist.jpg"]}        
        '''

        # 2. Construct model input
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": main_img},
                    {"type": "image", "image": wrist_img},
                    {"type": "text", "text": self.prompt_template.format(instruction=instruction)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[main_img, wrist_img], return_tensors="pt")
        inputs = inputs.to(self.device)

        # 3. Model inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        print("Generated Action Text:", output_text.strip())
        
        # 4. Decode and unnormalize actions
        current_action_chunk = self._decode_and_unnormalize(output_text.strip())

        # 5. Ensemble predictions
        self.action_chunk_history.append(current_action_chunk)

        # Extract the first action from the averaged chunks
        # Each chunk contains actions for multiple timesteps
        actions_to_average = [chunk for chunk in self.action_chunk_history]
        
        if not actions_to_average:
            final_action = np.zeros_like(self.action_min)
        else:
            final_action = np.mean(actions_to_average, axis=0)
        
        return {"actions": final_action}

# TODO: test on WebSocket server

def main(
    model_path: str = "/home/yuquan002/ssd/ms-swift-robotics/output/qwen3-vl-4b-instruct-vla0-libero/v1-20251025-213508/checkpoint-9936",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    # TODO: Implement server setup and start logic
    # 1. 设置设备并加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"正在加载模型到设备: {device}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 3. 创建策略实例
    policy = VLA0Policy(model, processor, device)
    
    # 4. 启动服务器
    server = WebsocketPolicyServer(policy=policy, host=host, port=port)
    server.serve_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)