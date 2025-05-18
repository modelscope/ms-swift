import json
import os
from pathlib import Path

# Constants
u_prompt = "<image>Provide contextually accurate description of the image. Write four paragraphs."
base_dir = Path("/home/aurumbekov/datasets/nsfw_batch1")
output_file = Path("distil_with_logits.jsonl")

# Make sure the output directory exists
output_file.parent.mkdir(parents=True, exist_ok=True)

with output_file.open("w", encoding="utf-8") as outf:
    # Recursively find all .json files under base_dir
    for json_path in base_dir.rglob("*.json"):
        try:
            # Load the JSON file
            data = json.loads(json_path.read_text(encoding="utf-8"))
            # Extract the assistant's answer
            a_answer = data.get("internvl_output_v3")
            if not a_answer:
                # Skip if the key is missing or empty
                continue

            # Construct the full image path (replace .json with .jpg)
            image_path = str(json_path.with_suffix(".jpg"))
            
            # Check if PT file with logits exists (with _logprobs.pt suffix)
            logprobs_path = json_path.with_stem(json_path.stem + "_logprobs").with_suffix(".pt")
            
            # Build the record
            record = {
                "messages": [
                    {"role": "user", "content": u_prompt},
                    {"role": "assistant", "content": a_answer}
                ],
                "images": [image_path]
            }
            
            # If logprobs file exists, add path to record instead of the values
            if logprobs_path.exists():
                record["logprobs_path"] = str(logprobs_path)

            # Write as a single line in JSONL format
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            # Print out any errors for debugging
            print(f"Error processing {json_path}: {e}", file=os.sys.stderr)

print(f"Dataset prepared and saved to {output_file}") 
