import os
from tqdm import tqdm
import json
from datasets import load_from_disk
from PIL import Image
import io
# from torch.utils.data import Dataset
import copy
import math
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
import torch
import re
from qwen_vl_utils import process_vision_info
from collections import defaultdict, Counter

device = "cuda:7" if torch.cuda.is_available() else "cpu"
# os.environ["WANDB_PROJECT"] = "qwen-vl-7b-finetuning"
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

def clean_model_output(raw: str) -> str:
    # 1. Remove ```json fences (both opening and closing)
    inner = re.sub(r'^```json\s*', '', raw.strip())
    inner = re.sub(r'\s*```$', '', inner)

    # 2. Load into a Python dict
    data = json.loads(inner)

    # 4. Dump back out as pretty JSON
    # return json.dumps(data, ensure_ascii=False, indent=2, separators=(",", ": "))
    return data

def iou(boxA, boxB):
    """
    Compute Intersection over Union between two bounding boxes.
    Boxes are [x1, y1, x2, y2].
    """
    if not boxA or not boxB:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


def flatten(annotations):
    """
    Flatten nested JSON into entries of shape:
      (slot, text, bbox)
    where `slot` includes suffix `.key` or `.value`.
    """
    entries = []
    for slot, items in annotations.items():
        if slot == 'personal_info':
            for person in items:
                for field, kv in person.items():
                    base = f"personal_info.{field}"
                    # key entries
                    key_text = kv['key'].get('text') or ''
                    key_bboxes = kv['key'].get('bbox_2d') or [[]]
                    for kb in key_bboxes:
                        entries.append((base + ".key", key_text, kb))
                    # value entries
                    val_text = kv['value'].get('text') or ''
                    val_bboxes = kv['value'].get('bbox_2d') or [[]]
                    for vb in val_bboxes:
                        entries.append((base + ".value", val_text, vb))
        else:
            for item in items:
                base = slot
                # key
                key_text = item['key'].get('text') or ''
                key_bboxes = item['key'].get('bbox_2d') or [[]]
                for kb in key_bboxes:
                    entries.append((base + ".key", key_text, kb))
                # value
                val_text = item['value'].get('text') or ''
                val_bboxes = item['value'].get('bbox_2d') or [[]]
                for vb in val_bboxes:
                    entries.append((base + ".value", val_text, vb))
    return entries


def slot_only_metrics(gt_entries, pred_entries):
    """
    Precision/Recall/F1 for slot extraction (slot includes key/value suffix).
    """
    gt_counts = defaultdict(Counter)
    pred_counts = defaultdict(Counter)
    for slot, text, _ in gt_entries:
        gt_counts[slot][text] += 1
    for slot, text, _ in pred_entries:
        pred_counts[slot][text] += 1

    tp = fp = fn = 0
    per_slot = {}
    for slot in set(gt_counts) | set(pred_counts):
        common = set(gt_counts[slot]) & set(pred_counts[slot])
        tp_slot = sum(min(gt_counts[slot][t], pred_counts[slot][t]) for t in common)
        total_pred = sum(pred_counts[slot].values())
        total_gt   = sum(gt_counts[slot].values())
        fp_slot = total_pred - tp_slot
        fn_slot = total_gt - tp_slot
        prec = tp_slot / total_pred if total_pred else 0.0
        rec  = tp_slot / total_gt   if total_gt   else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        per_slot[slot] = {'precision': prec, 'recall': rec, 'f1': f1}
        tp += tp_slot; fp += fp_slot; fn += fn_slot

    micro_prec = tp/(tp+fp) if (tp+fp) else 0.0
    micro_rec  = tp/(tp+fn) if (tp+fn) else 0.0
    micro_f1   = 2*micro_prec*micro_rec/(micro_prec+micro_rec) if (micro_prec+micro_rec) else 0.0
    macro_prec = sum(v['precision'] for v in per_slot.values())/len(per_slot) if per_slot else 0.0
    macro_rec  = sum(v['recall']    for v in per_slot.values())/len(per_slot) if per_slot else 0.0
    macro_f1   = sum(v['f1']        for v in per_slot.values())/len(per_slot) if per_slot else 0.0

    return {
        'micro': {'precision': micro_prec, 'recall': micro_rec, 'f1': micro_f1},
        'macro': {'precision': macro_prec, 'recall': macro_rec, 'f1': macro_f1},
        'per_slot': per_slot
    }


def box_only_metrics(gt_entries, pred_entries, iou_thresh=0.5):
    """
    Unified box-only metrics by slot (including .key/.value suffix).
    """
    gt_by_slot = defaultdict(list)
    pred_by_slot = defaultdict(list)
    for slot, _, bbox in gt_entries:
        if bbox:
            gt_by_slot[slot].append(bbox)
    for slot, _, bbox in pred_entries:
        if bbox:
            pred_by_slot[slot].append(bbox)

    overall_tp = overall_fp = overall_fn = 0
    per_slot = {}
    for slot in set(gt_by_slot) | set(pred_by_slot):
        gts  = gt_by_slot.get(slot, [])
        preds= pred_by_slot.get(slot, [])
        pairs = [(iou(g, p), i, j) for i, g in enumerate(gts) for j, p in enumerate(preds)]
        matched_gt = set(); matched_pred = set()
        for score, i, j in sorted(pairs, key=lambda x: x[0], reverse=True):
            if score < iou_thresh: break
            if i not in matched_gt and j not in matched_pred:
                matched_gt.add(i); matched_pred.add(j)
        tp = len(matched_gt)
        fp = len(preds) - len(matched_pred)
        fn = len(gts)   - len(matched_gt)
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        per_slot[slot] = {'precision': prec, 'recall': rec, 'f1': f1}
        overall_tp += tp; overall_fp += fp; overall_fn += fn

    micro_prec = overall_tp/(overall_tp+overall_fp) if (overall_tp+overall_fp) else 0.0
    micro_rec  = overall_tp/(overall_tp+overall_fn) if (overall_tp+overall_fn) else 0.0
    micro_f1   = 2*micro_prec*micro_rec/(micro_prec+micro_rec) if (micro_prec+micro_rec) else 0.0

    return {'micro': {'precision': micro_prec, 'recall': micro_rec, 'f1': micro_f1},
            'per_slot': per_slot}


def end_to_end_metrics(gt_entries, pred_entries, iou_thresh=0.5):
    """
    End-to-end requiring correct slot (with .key/.value) and bbox IoU ≥ threshold.
    """
    gt_map = defaultdict(list)
    for slot, text, bbox in gt_entries:
        gt_map[(slot, text)].append(bbox)
    pred_map = defaultdict(list)
    for slot, text, bbox in pred_entries:
        pred_map[(slot, text)].append(bbox)

    tp = fp = fn = 0
    for key, gt_bboxes in gt_map.items():
        pred_bboxes = pred_map.get(key, [])
        pairs = [(iou(g, p), i, j) for i, g in enumerate(gt_bboxes) for j, p in enumerate(pred_bboxes)]
        matched_gt = set(); matched_pred = set()
        for score, i, j in sorted(pairs, key=lambda x: x[0], reverse=True):
            if score < iou_thresh: break
            if i not in matched_gt and j not in matched_pred:
                matched_gt.add(i); matched_pred.add(j)
        tp += len(matched_gt)
        fp += len(pred_bboxes) - len(matched_pred)
        fn += len(gt_bboxes)   - len(matched_gt)

    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

    return {'precision': prec, 'recall': rec, 'f1': f1}

model_name = "absolute_unsloth_outputs/checkpoint-5000"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name, min_pixels = MIN_PIXELS, max_pixels = MAX_PIXELS, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

# dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
dataset = load_from_disk("PII_train_dataset")
test_dataset = list(dataset)[4500:]

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

# Create an instance of BBoxConverter
converter = BBoxConverter()

for sample in tqdm(test_dataset):
    gt_json = json.loads(sample['PII_mapped_OCR_elements'])
    sample["Image"] = sample["Image"] #.resize((640,640))
    if isinstance(sample['Image'], (bytes, bytearray)):
        # If the image is in bytes format, convert it to PIL Image
        sample['Image'] = Image.open(io.BytesIO(sample['Image']))
    h_orig, w_orig = sample["Image"].height, sample["Image"].width
    h_bar, w_bar = smart_resize(h_orig, w_orig, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    resized_absolute_gt_json = converter.convert_bboxes_to_absolute(gt_json, w_bar, h_bar)
    pretty = json.dumps(resized_absolute_gt_json, ensure_ascii=False, indent=2, separators=(",", ": "))
    # sample["ground_truth"] = f"```json\n{pretty}\n```"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["Image"]
                },
                {
                    "type": "text",
                    "text": instruction
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

    generated_ids = model.generate(**inputs, max_new_tokens=4096*8)
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
    clean_json = clean_model_output(raw)
    print(json.dumps(clean_json, ensure_ascii=False, indent=2, separators=(",", ": ")))
    print("Expected_output: ")
    print(pretty)
    pred = clean_json
    gt = resized_absolute_gt_json
    gt_entries = flatten(gt)
    pred_entries = flatten(pred)

    print("=== Slot-only metrics ===")
    print(json.dumps(slot_only_metrics(gt_entries, pred_entries), indent=2))

    print("\n=== Box-only metrics ===")
    print(json.dumps(box_only_metrics(gt_entries, pred_entries, iou_thresh=-0.75), indent=2))

    print("\n=== End-to-end metrics ===")
    print(json.dumps(end_to_end_metrics(gt_entries, pred_entries), indent=2))
    break