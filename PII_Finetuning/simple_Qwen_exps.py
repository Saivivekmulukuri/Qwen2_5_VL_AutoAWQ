from datasets import load_from_disk
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import torch
import re, json
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import colorsys
import random
import os
from pathlib import Path

device = "cuda:7" if torch.cuda.is_available() else "cpu"

def clean_model_output(raw: str) -> dict:
    """
    Cleans and parses a JSON-like string from model output by:
      1. Stripping code fences
      2. Removing trailing commas before closing brackets/braces
      3. Parsing into a Python dict/list
    Returns the resulting Python object.
    """
    # 1. Strip ```json fences
    inner = raw.strip()
    inner = re.sub(r'^```json\s*', '', inner)
    inner = re.sub(r'\s*```$', '', inner)

    # 2. Remove any trailing commas before '}' or ']' (robust for both arrays and objects)
    inner = re.sub(r',\s*(?=[}\]])', '', inner)

    # 3. Parse into Python structure
    try:
        data = json.loads(inner)
    except json.JSONDecodeError as e:
        # Provide a helpful error message if parsing fails
        raise ValueError(f"Failed to parse JSON: {e}\nCleaned content was:\n{inner}")

    return data

def plot_bboxes_and_save(image, json_data, output_path, box_color='red', text_color='red', box_width=2):
    """
    Plot bounding boxes from vision model JSON output on PIL Image and save it.
    Args:
        image(PIL Image): input image
        json_data (list): List of dicts with 'bbox_2d' and 'text_content'
        output_path (str): Path to save the output image
        box_color (str): Color of bounding box outline
        text_color (str): Color of label text
        box_width (int): Width of bounding box lines
    """
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    # Load a font (fallback if unavailable)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for item in json_data:
        bbox = item['bbox_2d']  # [x_min, y_min, x_max, y_max]
        label = item.get('text_content', '')

        # Draw rectangle
        draw.rectangle(bbox, outline=box_color, width=box_width)

        # Position text slightly above the top-left corner of the box
        text_x = bbox[0]
        text_y = max(bbox[1] - 20, 0)
        draw.text((text_x, text_y), label, fill=text_color, font=font)

    image_copy.save(output_path)
    print(f"Image saved to {output_path}")

# Create output directory
output_dir = Path("exp_outputs")
output_dir.mkdir(exist_ok=True)

# Load dataset
dataset = load_from_disk("PII_train_dataset")

# Convert dataset to list and randomly sample 50 examples
dataset_list = list(dataset)
sampled_examples = random.sample(dataset_list, min(50, len(dataset_list)))

print(f"Processing {len(sampled_examples)} samples...")

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

prompt = "Identify and extract people's names present in this document and give a bounding box for each of them."

# Process each sampled example
for idx, example in enumerate(sampled_examples):
    try:
        print(f"Processing sample {idx + 1}/{len(sampled_examples)}")
        
        # Handle image format conversion
        if isinstance(example['Image'], (bytes, bytearray)):
            # If the image is in bytes format, convert it to PIL Image
            example['Image'] = Image.open(io.BytesIO(example['Image']))
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example["Image"]
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
        
        # Process inputs
        text_inputs = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_inputs],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs.to(device)
        
        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=4096*8)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
         
        raw = output_text[0]  # the string you got, possibly with fences
        print(f"Raw output for sample {idx + 1}: {raw}")  # Print first 100 chars
        
        # Clean and parse the output
        try:
            cleaned = clean_model_output(raw)
            # Create output filename
            output_filename = output_dir / f"sample_{idx + 1:03d}_bbox_output.png"
            
            # Plot bounding boxes and save
            plot_bboxes_and_save(example["Image"], cleaned, str(output_filename))
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing JSON for sample {idx + 1}: {e}")
            # Save the image without bounding boxes if JSON parsing fails
            output_filename = output_dir / f"sample_{idx + 1:03d}_no_bbox.png"
            example["Image"].save(str(output_filename))
            print(f"Saved image without bounding boxes to {output_filename}")
            
    except Exception as e:
        print(f"Error processing sample {idx + 1}: {e}")
        continue

print(f"Processing complete! All outputs saved to {output_dir}")
