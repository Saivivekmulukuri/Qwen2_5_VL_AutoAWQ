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
from PIL import Image
import json
import colorsys
import random
from pathlib import Path
import os
import math

device = "cuda:5" if torch.cuda.is_available() else "cpu"
MIN_PIXELS = 256*28*28
MAX_PIXELS = 1280*28*28
_bracket_num = re.compile(r'\[\d+\]')

class Absolute_BBoxVisualizer:
    def __init__(self):
        self.field_colors = {}
        self.color_index = 0

    def generate_distinct_colors(self, n):
        """Generate n visually distinct colors using HSV color space"""
        colors = []
        # Strategy: Use a wider range of hue values and vary saturation and value
        # to create more distinct and human-readable colors.
        for i in range(n):
            hue = i / n  # Spread hues across the full spectrum
            saturation = 1#0.6 + (i % 3) * 0.2  # Vary saturation more significantly (0.6 to 1.0)
            value = 0.7# + (i % 2) * 0.3       # Vary brightness more significantly (0.7 to 1.0)
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors

    def get_all_field_paths(self, data, prefix=""):
        """Recursively extract all possible field paths from JSON structure"""
        paths = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{prefix}.{key}" if prefix else key

                if key == "bbox_2d":
                    # This is a bounding box field, add the parent path
                    parent_path = prefix.rsplit('.', 1)[0] if '.' in prefix else prefix
                    if parent_path and parent_path not in paths: # Ensure parent_path is not empty
                        paths.append(parent_path)
                else:
                    paths.extend(self.get_all_field_paths(value, current_path))

        elif isinstance(data, list) and len(data) > 0:
            # For arrays, analyze the first element to get structure
            paths.extend(self.get_all_field_paths(data[0], prefix))

        return paths

    def assign_colors_to_fields(self, data):
        """Assign unique colors to each field type"""
        # Get all possible field paths
        field_paths = self.get_all_field_paths(data)

        # Generate distinct colors
        colors = self.generate_distinct_colors(len(field_paths))

        # Assign colors to fields
        for i, path in enumerate(field_paths):
            self.field_colors[path] = colors[i]

    def convert_bbox_dict_to_list(self, bbox_dict):
        """Convert bbox dict with keys left, top, width, height to list [x_min, y_min, x_max, y_max]"""
        x_min = bbox_dict['left']
        y_min = bbox_dict['top']
        x_max = x_min + bbox_dict['width']
        y_max = y_min + bbox_dict['height']
        return [x_min, y_min, x_max, y_max]

    def plot_bboxes_on_image(self, image, json_data, figsize=(15, 10), save_path=None, dpi=300):
        """
        Plot bounding boxes on image with unique colors for each field

        Args:
            image: PIL Image object or numpy array
            json_data: Dictionary containing the parsed JSON data
            figsize: Figure size for the plot
            save_path: Path to save the output image (if None, displays instead)
            dpi: DPI for saved image (higher = better quality)
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Assign colors to fields
        self.assign_colors_to_fields(json_data)

        # Create figure
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)

        height, width = image.shape[:2]

        # Track labels to avoid overcrowding
        label_positions = []

        # Plot bounding boxes
        self._plot_recursive(json_data, ax, width, height, label_positions)

        # Create legend
        self._create_legend(ax)

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip y-axis for image coordinates
        ax.axis('off')
        plt.tight_layout()

        # Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
            plt.close(fig)  # Close figure to free memory
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

    def _plot_recursive(self, data, ax, width, height, label_positions, current_path=""):
        """Recursively traverse JSON and plot bounding boxes"""

        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key

                if key == "bbox_2d" and isinstance(value, list) and len(value) > 0:
                    # New format: list of lists [x_min, y_min, x_max, y_max] (absolute coordinates)
                    for bbox_list in value:
                        if len(bbox_list) >= 4 and all(isinstance(x, (int, float)) for x in bbox_list):
                            self._plot_single_bbox(bbox_list, ax, width, height, current_path, label_positions)
                else:
                    self._plot_recursive(value, ax, width, height, label_positions, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                self._plot_recursive(item, ax, width, height, label_positions, item_path)

    def _plot_single_bbox(self, bbox, ax, width, height, field_path, label_positions):
        """Plot a single bounding box with appropriate color and label"""

        if len(bbox) < 4:
            return

        x_min, y_min, x_max, y_max = bbox


        # Clamp to image boundaries
        x_min = max(0, min(x_min, width))
        y_min = max(0, min(y_min, height))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))


        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0:
            return

        # Get field type for coloring
        field_type = self._get_field_type(field_path)
        color = self.field_colors.get(field_type, (1, 0, 0))  # Default to red
        if "personal_info" in field_type:
          for key in self.field_colors:
            if ".".join(field_type.split(".")[:2]) in key:
              color = self.field_colors[key]
              break
        else:
          for key in self.field_colors:
            if field_type.split(".")[0] in key:
              color = self.field_colors[key]
              break
        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)

        # Add label if space permits
        self._add_label_if_space_permits(ax, x_min, y_min, field_type, color, label_positions)

    def _get_field_type(self, path):
        """Extract the field type from the path"""
        # Remove array indices and get the main field type
        # clean_path = path.replace('[0]', '').replace('[1]', '').replace('[2]', '')
        clean_path = _bracket_num.sub('', path)


        # Extract the main field (e.g., "personal_info.name.key" -> "personal_info.name.key")
        parts = clean_path.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[:3])  # Take first two parts
        return clean_path

    def _add_label_if_space_permits(self, ax, x, y, label, color, label_positions, min_distance=30):
        """Add label only if there's enough space and no overlap"""

        # # Check if this position is too close to existing labels
        # for pos in label_positions:
        #     if abs(pos[0] - x) < min_distance and abs(pos[1] - y) < min_distance:
        #         return  # Skip this label to avoid overcrowding

        # Simplify label for display
        display_label = ".".join(label.split('.')[-2:]) if '.' in label else label
        # display_label = label

        # Add label with background
        ax.text(
            x, y - 5, display_label,
            fontsize=6,
            color=color,
            weight='bold',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor=color,
                pad=2,
                boxstyle='round,pad=0.3'
            )
        )

        label_positions.append((x, y))

    def _create_legend(self, ax):
        """Create a legend for field types"""
        legend_elements = []

        for field_type, color in self.field_colors.items():
            legend_elements.append(
                patches.Patch(color=color, label=field_type)
            )

        # Place legend outside the plot area
        ax.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )

# Updated usage function
def visualize_document_parsing_results(pil_image, json_data, save_path=None, figsize=(15, 10), dpi=300):
    """
    Main function to visualize bounding boxes from PIL Image

    Args:
        pil_image: PIL Image object or numpy array
        json_data: Dictionary containing the parsed JSON data
        save_path: Path to save the output image (if None, displays instead)
        figsize: Figure size for the plot
        dpi: DPI for saved image
    """
    visualizer = Absolute_BBoxVisualizer()
    visualizer.plot_bboxes_on_image(pil_image, json_data, figsize=figsize, save_path=save_path, dpi=dpi)

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def clean_model_output(raw: str) -> str:
    # 1. Remove ```json fences (both opening and closing)
    inner = re.sub(r'^```json\s*', '', raw.strip())
    inner = re.sub(r'\s*```$', '', inner)

    # 2. Load into a Python dict
    data = json.loads(inner)

    # 4. Dump back out as pretty JSON
    # return json.dumps(data, ensure_ascii=False, indent=2, separators=(",", ": "))
    return data

# dataset = load_from_disk("PII_train_dataset")
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model_name = "absolute_unsloth_outputs/checkpoint-7500"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name, min_pixels = MIN_PIXELS, max_pixels = MAX_PIXELS, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model.to(device)
prompt = """You are a document‐parsing assistant. You will be given a single document (image or PDF page) and must extract all personally identifiable information into a single JSON object.  Do not output any extra text—only the JSON.

The JSON must have these top‐level keys:
- "personal_info": a list of objects, one per person found.
- "address", "phone_number", "email", "id_number", "date", "doctor", "facility", "website": each a list of objects, one per occurrence.

Each object in every list must have exactly two fields:
1. "key": {
     "text":    the exact key label as it appears in the document (e.g. "Name:", "Date of Birth", etc.),
     "bbox_2d":  a list of bounding‐boxes for that key text; each bounding-box is 
                [<int>, <int>, <int>, <int>] in the order [x_min, y_min, x_max, y_max].
   }
2. "value": {
     "text":    the exact value string (e.g. "John Doe", "01/02/1980"),
     "bbox_2d":  list of bounding‐boxes for the value text, in the same format
   }

For "personal_info", each person object must contain these sub‐keys (each with its own key/value pair as above):
  - name
  - age
  - gender
  - dob
  - relative_name
  - nationality
  - occupation
  - weight
  - height

A complete JSON output might look like:
{
  "personal_info": [
    {
      "name": {
        "key": {
          "text": "",
          "bbox_2d": []
        },
        "value": {
          "text": "",
          "bbox_2d": []
        }
      },
      "age": {
        "key": {
          "text": "",
          "bbox_2d": []
        },
        "value": {
          "text": "",
          "bbox_2d": []
        }
      }, // age of that person (parse it if it's mentioned in the document, do not calculate it from the dob).
      "gender": {
        "key": {
          "text": "",
          "bbox_2d": []
        },
        "value": {
          "text": "",
          "bbox_2d": []
        }
      }, // gender of that person
      "dob": {
        "key": {
          "text": "",
          "bbox_2d": []
        },
        "value": {
          "text": "",
          "bbox_2d": []
        }
      }, // date of birth of that person
      "relative_name": {
        "key": {
          "text": "",
          "bbox_2d": []
        }, // name of a relative of that person (if mentioned in the document)
        "value": {
          "text": "",
          "bbox_2d": []
        } // relation of that person (if mentioned in the document)
      },
      "nationality": {
        "key": {
          "text": "",
          "bbox_2d": []
        },
        "value": {
          "text": "",
          "bbox_2d": []
        }
      }, // citizenship if mentioned in the document
      "occupation": {
        "key": {
          "text": "",
          "bbox_2d": []
        },
        "value": {
          "text": "",
          "bbox_2d": []
        }
      }, // occupation or profession of that person (if mentioned in the document)
      "weight": {
        "key": {
          "text": "",
          "bbox_2d": []
        },
        "value": {
          "text": "",
          "bbox_2d": []
        }
      }, // weight of that person (if mentioned in the document with units like kg or lbs)
      "height": {
        "key": {
          "text": "",
          "bbox_2d": []
        },
        "value": {
          "text": "",
          "bbox_2d": []
        }
      } // height of that person (if mentioned in the document with units like cm or inches).
    }
  ], // This is specific to a single person. So, if in a document there are multiple people, this list will have multiple objects, each with the above keys. If any item's value is null, do not include it in the list. If a single name is present in two types like "Aman Gupta" and Gupta Aman", then include both the names.
  "address": [
    {
      "key": {
        "text": "",
        "bbox_2d": []
      }, // either patient's address or hospital address
      "value": {
        "text": [""], // address can be multiline, so split it into a list of strings.
        "bbox_2d": []
      }
    }
  ],
  "phone_number": [
    {
      "key": {
        "text": "",
        "bbox_2d": []
      }, // key used to identify the phone number (e.g. "Phone:", "Contact Number:", etc.), if absent whose might that phone number be.
      "value": {
        "text": "",
        "bbox_2d": []
      }
    }
  ], // list of objects with key and value as above for all phone numbers found in the document.
  "email": [
    {
      "key": {
        "text": "",
        "bbox_2d": []
      },
      "value": {
        "text": "",
        "bbox_2d": []
      }
    }
  ], // lisf of objects with key and value as above for all email addresses found in the document.
  "id_number": [
    {
      "key": {
        "text": "",
        "bbox_2d": []
      }, // type of ID
      "value": {
        "text": "",
        "bbox_2d": []
      }
    }
  ], // list of objects with key and value as above for all ID numbers found in the document.
  "date": [
    {
      "key": {
        "text": "",
        "bbox_2d": []
      }, // key used to identify the date (e.g. "Date:") or context inferred from the document.
      "value": {
        "text": "",
        "bbox_2d": []
      }
    }
  ], // list of objects with key and value as above for all dates found in the document, like admission date, discharge date etc.
  "doctor": [
    {
      "key": {
        "text": "",
        "bbox_2d": []
      },
      "value": {
        "text": "",
        "bbox_2d": []
      } // doctor's name
    }
  ], // Name of the consulting doctor. If there are multiple doctors, include all of them in key value pairs. Do not include multiple names in a single key value pair.
  "facility": [
    {
      "key": {
        "text": "",
        "bbox_2d": []
      },
      "value": {
        "text": "",
        "bbox_2d": []
      }
    }
  ], // Name of the hospital or medical facility that issued the document, usually main heading at the top of the document. Fill this only if found in the document. Also extract if the document mentions a government agency (e.g. "Ministry of Health")..
  "website": [
    {
      "key": {
        "text": "",
        "bbox_2d": []
      },
      "value": {
        "text": "",
        "bbox_2d": []
      }
    }
  ] //Official website URL if mentioned in the document. This is a list of objects with key and value as above for all website URLs found in the document.
}


Now parse the provided document and emit only the JSON populated with the actual text and bounding‐boxes you detect.

Instructions:
- Do not output any text other than the JSON.
- For each field, parse its value (exact text) from the document and the key with which the value was mentioned in the document. Only parse if the key is present in the document - respond with a null otherwise.
- If certain information is not present in the document, don't include it in the JSON.
- Represent "age" as a string, including any units mentioned in the document. If the age is given in years, months, and days, represent it as <years> years <months> months <days> days. If gender is mentioned along with age, make sure to separate the gender and age correctly.
- Don't give a PII field where the value is null or empty even if the key is present in the document. If the value is null, do not include the key in the JSON.
"""

# Convert dataset to list and randomly sample 50 examples
# dataset_list = list(dataset)
# sampled_examples = random.sample(dataset_list, min(50, len(dataset_list)))

# Create output directory
output_dir = Path("documents_outputs_7500")
output_dir.mkdir(exist_ok=True)

input_dir = "/users/student/pg/pg23/saivivekmulukuri/Qwen_VL/documents"
for idx, img_name in enumerate(sorted(os.listdir(input_dir))):
    img = Image.open(os.path.join(input_dir, img_name)).convert('RGB')
    h_orig, w_orig = img.height, img.width
    h_bar, w_bar = smart_resize(h_orig, w_orig, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
# # for example in dataset:
# for idx, example in enumerate(sampled_examples):
#     print(f"Processing sample {idx + 1}/{len(sampled_examples)}")
#     if isinstance(example['Image'], (bytes, bytearray)):
#         # If the image is in bytes format, convert it to PIL Image
#         example['Image'] = Image.open(io.BytesIO(example['Image']))
#     h_orig, w_orig = example["Image"].height, example["Image"].width
#     h_bar, w_bar = smart_resize(h_orig, w_orig, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
#         # print(example['Image'])
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img #example["Image"]
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]
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

    generated_ids = model.generate(**inputs, 
                                  #  do_sample=False,
                                  #  repetition_penalty=1.2,
                                   max_new_tokens=4096*8)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print(f"Generated outputs: ", generated_ids[0].shape, "Generated_ids_trimmed: ", generated_ids_trimmed[0].shape, "input_ids: ", len(inputs.input_ids[0]))
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    raw = output_text[0]  # the string you got, possibly with fences
    # print(raw)
    clean_json = clean_model_output(raw)
    print(json.dumps(clean_json, ensure_ascii=False, indent=2, separators=(",", ": ")))

    # print("\nExpected output:")
    # gt = example["PII_mapped_OCR_elements"]
    # obj = json.loads(gt)
    # gt_pretty = json.dumps(obj, ensure_ascii=False, indent=2, separators=(',',': '))
    # print(gt_pretty)
    # # print(raw)

    # save_bboxes_on_image(example["image"], clean_json, save_path="output_with_bboxes.png")
    # Save visualization
    # resized_img = example['Image'].resize((w_bar, h_bar))
    resized_img = img.resize((w_bar, h_bar))
    visualize_document_parsing_results(
        # example["Image"], 
        resized_img,
        clean_json,
        save_path=os.path.join(output_dir, f"{img_name.split('.')[0]}.png"),
        dpi=300
    )
    # break
    