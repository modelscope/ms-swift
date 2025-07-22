
"""
3 dataset file

    1. data_thinklite_reasoning_acc:  data_source == 'thinklite_eureka'
    2. data_0.1.2_visual_toolbox_v2.parquet : data_source == 'thinklite_eureka'
    3. 

plugin:
    1. Tool to resize and rotate
    2. construct interleaved image-conv
"""


class VisualToolBox():
    name = "visual_toolbox_v2"
    # user_prompt = "Here is the cropped image returned after you calling the function {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise you can continue to call tools within <tool_call></tool_call>."

    user_prompt = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed


    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None
        
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        action = self.extract_action(action_string)
        if not action:
            return "", 0.0, True, {}
        try:
            tool_call = json.loads(action.strip())  # 或使用 literal_eval
        except Exception as e:
            error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
            obs = "\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n"
            info = {"error": str(e), "status": "failed"}
            return obs, 0.0, False, {}
        try:

            tool_name = tool_call["name"]
            args = tool_call["arguments"]
        
            if tool_name == "image_zoom_in_tool":
                # Zoom in by cropping the image
                # image_path = args["image_path"]
                bbox = args["bbox_2d"]
                bbox = self.maybe_resize_bbox(*bbox)
                if not bbox:
                    raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][0]
                cropped_img = img.crop(bbox)
                current_image = cropped_img
                
            elif tool_name == "image_rotate_tool":
                # Rotate the image
                # image_path = args["image_path"]
                angle = args["angle"]
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][0]
                rotated_img = img.rotate(angle)
                current_image = rotated_img
                
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            # Prepare the observation
            obs = {
                "prompt": "\n<|im_start|>user\n" + "<tool_response>" +"<image>" + self.user_prompt + "</tool_response>" + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [current_image]}
            }
            reward = 0.0  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG] SUCCESS ACTION {action_string=}')
            return obs, reward, done, info
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)} {action_string=}')
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            assert min(height, width) > 30, f"{height=}, {width=} is too small"
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False


    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]
