import os
# Limit the available GPUs to 2 (use GPUs with IDs 0 and 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
from unsloth import FastVisionModel # FastLanguageModel for LLMs
# import torch
from datasets import load_dataset
from tqdm import tqdm
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import json
from datasets import load_from_disk
from PIL import Image
import io
# from torch.utils.data import Dataset
from transformers import AutoProcessor
import copy
import math
import wandb

wandb.login(key="2a537e4b7e407034fcbec27f7784929581278de2")

os.environ["WANDB_PROJECT"] = "qwen-vl-7b-finetuning"
MIN_PIXELS = 256*28*28
MAX_PIXELS = 1280*28*28

class BBoxConverter:
    def __init__(self):
        pass

    def convert_bbox_dict_to_list(self, bbox_dict, width, height):
        """Convert bbox dict with keys left, top, width, height to list [x_min, y_min, x_max, y_max]
        and convert relative coordinates to absolute.
        """
        x_min = bbox_dict['left'] * width
        y_min = bbox_dict['top'] * height
        bbox_width = bbox_dict['width'] * width
        bbox_height = bbox_dict['height'] * height
        x_max = x_min + bbox_width
        y_max = y_min + bbox_height
        return [round(x_min), round(y_min), round(x_max), round(y_max)]

    def convert_bboxes_to_absolute(self, json_data, width, height):
        """
        Traverse JSON and convert relative bounding boxes to absolute.

        Args:
            json_data: Dictionary containing the parsed JSON data
            width: Width of the image
            height: Height of the image

        Returns:
            A new dictionary with absolute bounding boxes
        """
        # Create a deep copy to avoid modifying the original data
        new_json_data = copy.deepcopy(json_data)
        self._convert_recursive(new_json_data, width, height)
        return new_json_data

    def _convert_recursive(self, data, width, height):
        """Recursively traverse JSON and convert bounding boxes"""

        if isinstance(data, dict):
            changes = []
            for key, value in data.items():
                if key == "bboxes" and isinstance(value, list) and len(value) > 0:
                    # Check if it's the new format (list of dicts) or old format (list of floats)
                    if isinstance(value[0], dict):
                        # New format: list of dicts with left, top, width, height
                        new_bboxes = []
                        for bbox_dict in value:
                            if all(k in bbox_dict for k in ['left', 'top', 'width', 'height']):
                                bbox_list = self.convert_bbox_dict_to_list(bbox_dict, width, height)
                                new_bboxes.append(bbox_list)
                        changes.append((key, new_bboxes)) # Store changes
                else:
                    self._convert_recursive(value, width, height)

            # Apply changes after iteration
            for key, new_bboxes in changes:
                data["bbox_2d"] = new_bboxes
                del data[key]

        elif isinstance(data, list):
            for item in data:
                self._convert_recursive(item, width, height)

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

model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    # full_finetuning=True, # False for LoRA
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

tokenizer = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            min_pixels = MIN_PIXELS, max_pixels = MAX_PIXELS
        )

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 64,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 64,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
dataset = load_from_disk("PII_train_dataset")
train_dataset = list(dataset)[:4500]

val_dataset = list(dataset)[4500:]

instruction = """You are a document‐parsing assistant. You will be given a single document (image or PDF page) and must extract all personally identifiable information into a single JSON object.  Do not output any extra text—only the JSON.

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

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["Image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["ground_truth"]} ]
        },
    ]
    return { "messages" : conversation }


converted_train_dataset = []
converted_val_dataset = []
# Create an instance of BBoxConverter
converter = BBoxConverter()

for sample in tqdm(train_dataset):
    gt_json = json.loads(sample['PII_mapped_OCR_elements'])
    sample["Image"] = sample["Image"] #.resize((640,640))
    if isinstance(sample['Image'], (bytes, bytearray)):
        # If the image is in bytes format, convert it to PIL Image
        sample['Image'] = Image.open(io.BytesIO(sample['Image']))
    h_orig, w_orig = sample["Image"].height, sample["Image"].width
    h_bar, w_bar = smart_resize(h_orig, w_orig, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    resized_absolute_gt_json = converter.convert_bboxes_to_absolute(gt_json, w_bar, h_bar)
    pretty = json.dumps(resized_absolute_gt_json, ensure_ascii=False, indent=2, separators=(",", ": "))
    sample["ground_truth"] = f"```json\n{pretty}\n```"
    converted_train_dataset.append(convert_to_conversation(sample))

for sample in tqdm(val_dataset):
    gt_json = json.loads(sample['PII_mapped_OCR_elements'])
    sample["Image"] = sample["Image"] #.resize((640,640))
    if isinstance(sample['Image'], (bytes, bytearray)):
        # If the image is in bytes format, convert it to PIL Image
        sample['Image'] = Image.open(io.BytesIO(sample['Image']))
    h_orig, w_orig = sample["Image"].height, sample["Image"].width
    h_bar, w_bar = smart_resize(h_orig, w_orig, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    resized_absolute_gt_json = converter.convert_bboxes_to_absolute(gt_json, w_bar, h_bar)
    pretty = json.dumps(resized_absolute_gt_json, ensure_ascii=False, indent=2, separators=(",", ": "))
    sample["ground_truth"] = f"```json\n{pretty}\n```"
    converted_val_dataset.append(convert_to_conversation(sample))

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer, resize='max', max_seq_length=4096),
    train_dataset = converted_train_dataset,
    eval_dataset= converted_val_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        eval_strategy="steps",
        eval_steps=1000,
        optim = "adamw_8bit",  # Use "adamw" for 16-bit training
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "absolute_unsloth_outputs_split",
        report_to = "wandb",     # For Weights and Biases
        run_name = "qwen-2.5-vl-7b-cord",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 4096*8,
        max_length= 4096*8,  # Set max_length to match the model's context length
    ),
)

trainer_stats = trainer.train()
model.save_pretrained("absolute_unsloth_outputs_split/qwen2.5-vl-3b-lora-finetune")
tokenizer.save_pretrained("absolute_unsloth_outputs_split/qwen2.5-vl-3b-lora-finetune")
