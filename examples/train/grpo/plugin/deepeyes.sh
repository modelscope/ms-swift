# SYSTEM_PROMPT='You are a helpful assistant.

# # Tools
# You may call one or more functions to assist with the user query.
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
# </tools>

# # How to call a tool
# Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>

# **Example**:  
# <tool_call>  
# {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
# </tool_call>'


swift rlhf \
    --model Qwen/Qwen2.5-7B-VL-Instruct \
    --external_plugins examples/train/grpo/plugin/deepeyes.py \

