# server_vla0_policy.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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
        self.prev_action = None

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

        self.system_prompt = (
            # "Analyze the input image and predict robot actions for the next 5 timesteps. Each action has 7 dimensions. Output a single sequence of 35 integers (0 - 1000 each), representing the 5 timesteps sequentially. Provide only space-separated numbers. Nothing else."
            "Assume you are a robot control system. Analyze the input image, the task description and the current state to predict the robot's next action. The output should be a sequence of actions wherethe first three values represent 'dx', 'dy', 'dz' for the end-effector's positional movement, the next three values represent 'drx', 'dry', 'drz' for the end-effector's rotational movement, the seventh value indicates the gripper state (-1: open, 1: closed). Provide only next action as output, nothing else."
        )
        # self.prompt_template = "task description: {instruction}\n\nimage:\n<image>\n<image>"
        # self.prompt_template = "task description: {instruction}\n\nimage:\n<image>\n<image>"
        # self.prompt_template = "robot state: '{tobostate}'. task description: {instruction}. image:<image><image>"

    def _decode_and_unnormalize(self, action_text: str):
        """
        Decode the action text into quantized integers and then unnormalize them back to continuous action values.
        """
        try:
            # 1. Decode the space-separated text into integers
            quantized_action = np.fromstring(action_text, dtype=int, sep=' ')
            
            # 2. Validate the length of the decoded action
            if quantized_action.size != 35:  # assuming 5 timesteps * 7 dimensions
                # truncate the action to the expected size
                logging.warning(f"Decoded action size {quantized_action.size} does not match expected size 35. Adjusting accordingly.")
                if quantized_action.size > 35:
                    quantized_action = quantized_action[:35]
                else:
                    # truncate the action to 4*7 or less
                    quantized_action = np.pad(quantized_action, (0, 35 - quantized_action.size), 'constant', constant_values=0)
                    logging.warning(f"Adjusted action size: {quantized_action.size}")

            # 3. Unnormalize the quantized action back to continuous values
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
        main_img = Image.fromarray(obs["observation/image"]).convert("RGB")
        wrist_img = Image.fromarray(obs["observation/wrist_image"]).convert("RGB")
        instruction = obs["prompt"]


        # 2. Construct model input
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": self.prompt_template.format(instruction=instruction)}, 
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": main_img},
                    {"type": "image", "image": wrist_img},
                ],

            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info([messages])
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt")
        # import pdb; pdb.set_trace()
        inputs = inputs.to(self.device)

        # 3. Model inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        # print("Generated Action Text:", output_text.strip())
        # Generated Action Text: [ 0.321,  0.   ,  0.011,  0.   ,  0.015, -0.   , -1.   ]
        # convert the string to list of floats
        action_floats = [float(x) for x in output_text.strip().strip('[]').split(',')]
        action_array = np.array(action_floats)

        # import pdb; pdb.set_trace()

        print("Action Array:", action_array)

        self.prev_action = action_array

        action = action_array - self.prev_action if self.prev_action is not None else np.zeros_like(action_array)
        self.prev_action = action_array
        
        # # 4. Decode and unnormalize actions
        # current_action_chunk = self._decode_and_unnormalize(output_text.strip()) # [35,]

        # # reshape to [5, 7] for 5 timesteps
        # current_action_chunk = current_action_chunk.reshape(5, 7)

        return action_array

        # # 5. Ensemble predictions
        # self.action_chunk_history.append(current_action_chunk)

        # # Pop out the first action from each chunk and average them
        # # Each chunk contains actions for multiple timesteps
        # # after a pop operation, we average the first actions across all chunks
        # # the last dim is gripper control, we only average the movement dimensions
        # first_actions = np.array([chunk[0] for chunk in self.action_chunk_history])  # [ensemble_size, 7]
        # # pop out the first action from each chunk
        # for i in range(len(self.action_chunk_history)):
        #     self.action_chunk_history[i] = self.action_chunk_history[i][1:]  # remove the first action
        # averaged_first_action = np.mean(first_actions[:, :6], axis=0)  # [6,]
        # gripper_actions = first_actions[:, 6]
        # # majority voting for gripper action
        # gripper_action = np.sign(np.sum(np.sign(gripper_actions)))
        # averaged_action = np.concatenate([averaged_first_action, [gripper_action]], axis=0)  # [7,]

        # # return the averaged action
        # return {"actions": averaged_action}


def main(
    model_path: str = "/home/yuquan002/ssd/ms-swift-robotics/output/qwen3-vl-4b-instruct-vla0-libero/v9-20251107-182735/checkpoint-4096",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    # 1. Configure device and load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model from {model_path} to device: {device}")
    
    # 2. Load model and processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # 3. Create policy instance
    policy = VLA0Policy(model, processor, device)

    # 4. Start server
    server = WebsocketPolicyServer(policy=policy, host=host, port=port)
    server.serve_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)