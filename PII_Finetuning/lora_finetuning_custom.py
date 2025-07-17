import os
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
import json
import io
from PIL import Image
from datasets import load_from_disk, Dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import lightning as L
from nltk import edit_distance
from torch.optim import AdamW
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# USE_QLORA = True
lora_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=None,
    torch_dtype=torch.bfloat16)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

def train_data_collator(batch, processor):
    """
    Data collator for raw image data that needs preprocessing
    """
    # Extract the conversation examples
    examples = [sample["messages"] for sample in batch]
    
    # Apply chat template to get text inputs
    texts = [
        processor.apply_chat_template(example, tokenize=False)
        for example in examples
    ]
    
    # Process vision info to handle raw image data
    image_inputs = [
        process_vision_info(example)[0]
        for example in examples
    ]
    
    # Use processor to tokenize text and process images
    model_inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True
    )
    
    # Prepare labels by cloning input_ids
    labels = model_inputs["input_ids"].clone()
    
    # Mask padding tokens in labels
    # pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Mask image token ids in labels
    image_token_ids = [151652, 151653, 151655]  # Qwen2.5-VL specific tokens
    for image_token_id in image_token_ids:
        labels[labels == image_token_id] = -100
    # return {
    #     "input_ids" : model_inputs["input_ids"],
    #     "attention_mask" : model_inputs["attention_mask"],
    #     "pixel_values" : model_inputs["pixel_values"],
    #     "image_grid_thw" : model_inputs.get("image_grid_thw"),
    #     "labels" : labels
    # }
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs.get("image_grid_thw")
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels

def val_data_collator(batch, processor):
    """
    Data collator for raw image data that needs preprocessing
    """
    # Extract the conversation examples
    examples = [sample["messages"] for sample in batch]
    
    ground_truths = [example[1]["content"][0]["text"] for example in examples]
    examples = [eg[:1] for eg in examples]
    # Apply chat template to get text inputs
    texts = [
        processor.apply_chat_template(example, tokenize=False)
        for example in examples
    ]
    
    # Process vision info to handle raw image data
    image_inputs = [
        process_vision_info(example)[0]
        for example in examples
    ]
    
    # Use processor to tokenize text and process images
    model_inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True
    )
    # return {
    #     "input_ids" : model_inputs["input_ids"],
    #     "attention_mask" : model_inputs["attention_mask"],
    #     "pixel_values" : model_inputs["pixel_values"],
    #     "image_grid_thw" : model_inputs.get("image_grid_thw"),
    # }
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs.get("image_grid_thw")
    return input_ids, attention_mask, pixel_values, image_grid_thw, ground_truths

instruction = """You are a document‐parsing assistant. You will be given a single document (image or PDF page) and must extract all personally identifiable information into a single JSON object.  Do not output any extra text—only the JSON.

The JSON must have these top‐level keys:
- "personal_info": a list of objects, one per person found.
- "address", "phone_number", "email", "id_number", "date", "doctor", "facility", "website": each a list of objects, one per occurrence.

Each object in every list must have exactly two fields:
1. "key": {
     "text":    the exact key label as it appears in the document (e.g. "Name:", "Date of Birth", etc.),
     "bboxes":  a list of bounding‐boxes for that key text; each bbox is 
                {"left": <int>, "top": <int>, "width": <int>, "height": <int>}
   }
2. "value": {
     "text":    the exact value string (e.g. "John Doe", "01/02/1980"),
     "bboxes":  list of bounding‐boxes for the value text, in the same format
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
          "bboxes": []
        },
        "value": {
          "text": "",
          "bboxes": []
        }
      },
      "age": {
        "key": {
          "text": "",
          "bboxes": []
        },
        "value": {
          "text": "",
          "bboxes": []
        }
      }, // age of that person (parse it if it's mentioned in the document, do not calculate it from the dob).
      "gender": {
        "key": {
          "text": "",
          "bboxes": []
        },
        "value": {
          "text": "",
          "bboxes": []
        }
      }, // gender of that person
      "dob": {
        "key": {
          "text": "",
          "bboxes": []
        },
        "value": {
          "text": "",
          "bboxes": []
        }
      }, // date of birth of that person
      "relative_name": {
        "key": {
          "text": "",
          "bboxes": []
        }, // name of a relative of that person (if mentioned in the document)
        "value": {
          "text": "",
          "bboxes": []
        } // relation of that person (if mentioned in the document)
      },
      "nationality": {
        "key": {
          "text": "",
          "bboxes": []
        },
        "value": {
          "text": "",
          "bboxes": []
        }
      }, // citizenship if mentioned in the document
      "occupation": {
        "key": {
          "text": "",
          "bboxes": []
        },
        "value": {
          "text": "",
          "bboxes": []
        }
      }, // occupation or profession of that person (if mentioned in the document)
      "weight": {
        "key": {
          "text": "",
          "bboxes": []
        },
        "value": {
          "text": "",
          "bboxes": []
        }
      }, // weight of that person (if mentioned in the document with units like kg or lbs)
      "height": {
        "key": {
          "text": "",
          "bboxes": []
        },
        "value": {
          "text": "",
          "bboxes": []
        }
      } // height of that person (if mentioned in the document with units like cm or inches).
    }
  ], // This is specific to a single person. So, if in a document there are multiple people, this list will have multiple objects, each with the above keys. If any item's value is null, do not include it in the list. If a single name is present in two types like "Aman Gupta" and Gupta Aman", then include both the names.
  "address": [
    {
      "key": {
        "text": "",
        "bboxes": []
      }, // either patient's address or hospital address
      "value": {
        "text": [""], // address can be multiline, so split it into a list of strings.
        "bboxes": []
      }
    }
  ],
  "phone_number": [
    {
      "key": {
        "text": "",
        "bboxes": []
      }, // key used to identify the phone number (e.g. "Phone:", "Contact Number:", etc.), if absent whose might that phone number be.
      "value": {
        "text": "",
        "bboxes": []
      }
    }
  ], // list of objects with key and value as above for all phone numbers found in the document.
  "email": [
    {
      "key": {
        "text": "",
        "bboxes": []
      },
      "value": {
        "text": "",
        "bboxes": []
      }
    }
  ], // lisf of objects with key and value as above for all email addresses found in the document.
  "id_number": [
    {
      "key": {
        "text": "",
        "bboxes": []
      }, // type of ID
      "value": {
        "text": "",
        "bboxes": []
      }
    }
  ], // list of objects with key and value as above for all ID numbers found in the document.
  "date": [
    {
      "key": {
        "text": "",
        "bboxes": []
      }, // key used to identify the date (e.g. "Date:") or context inferred from the document.
      "value": {
        "text": "",
        "bboxes": []
      }
    }
  ], // list of objects with key and value as above for all dates found in the document, like admission date, discharge date etc.
  "doctor": [
    {
      "key": {
        "text": "",
        "bboxes": []
      },
      "value": {
        "text": "",
        "bboxes": []
      } // doctor's name
    }
  ], // Name of the consulting doctor. If there are multiple doctors, include all of them in key value pairs. Do not include multiple names in a single key value pair.
  "facility": [
    {
      "key": {
        "text": "",
        "bboxes": []
      },
      "value": {
        "text": "",
        "bboxes": []
      }
    }
  ], // Name of the hospital or medical facility that issued the document, usually main heading at the top of the document. Fill this only if found in the document. Also extract if the document mentions a government agency (e.g. "Ministry of Health")..
  "website": [
    {
      "key": {
        "text": "",
        "bboxes": []
      },
      "value": {
        "text": "",
        "bboxes": []
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


converted_dataset = []

# Load raw dataset
raw_dataset = load_from_disk("PII_train_dataset")

for sample in tqdm(raw_dataset):
    pretty = json.dumps(json.loads(sample['PII_mapped_OCR_elements']), ensure_ascii=False, indent=2, separators=(",", ": "))
    sample["ground_truth"] = f"```json\n{pretty}\n```"
    sample["Image"] = sample["Image"] #.resize((640,640))
    if isinstance(sample['Image'], (bytes, bytearray)):
        # If the image is in bytes format, convert it to PIL Image
        sample['Image'] = Image.open(io.BytesIO(sample['Image']))
    converted_dataset.append(convert_to_conversation(sample))

# Split dataset into 80% train and 20% validation
train_data, val_data = train_test_split(
    converted_dataset, 
    test_size=0.2, 
    random_state=3407,
    shuffle=True
)

print("Splits are made")

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

# Create PyTorch datasets
train_dataset = ListDataset(train_data)
val_dataset = ListDataset(val_data)

# # Convert to HuggingFace Dataset
# train_dataset = Dataset.from_list(train_data)
# val_dataset = Dataset.from_list(val_data)

# print("dataset from splits_done")
# BATCH_SIZE = 4
# NUM_WORKERS = 4
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: train_data_collator(batch, processor), num_workers=NUM_WORKERS, shuffle=True)
# valid_loader = DataLoader(val_dataset, batch_size=1, collate_fn=lambda batch: val_data_collator(batch, processor), num_workers=0, shuffle=False)
# print("loaded_data")

config = {
    "max_epochs": 1,
    "batch_size": 1,
    "lr": 2e-4,
    "check_val_every_n_epoch": 2,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 4,
    "num_nodes": 1,
    "warmup_steps": 5,
    "result_path": "qwen2.5-3b-instruct-ft",
    "loss_type": "ForCausalLMLoss",
}

class Qwen2_5_Trainer(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_grid_thw, suffixes = batch
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1024
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids
            in zip(input_ids, generated_ids)]
        generated_suffixes = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        scores = []
        for generated_suffix, suffix in zip(generated_suffixes, suffixes):
            score = edit_distance(generated_suffix, suffix)
            score = score / max(len(generated_suffix), len(suffix))
            scores.append(score)
            print("generated_suffix", generated_suffix)
            print("suffix", suffix)
            print("score", score)
        score = sum(scores) / len(scores)
        self.log("val_edit_distance", score, prog_bar=True, logger=True, batch_size=self.config.get("batch_size"))
        return scores
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.get("lr"))
        return optimizer
    def train_dataloader(self):
        return DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=lambda batch: train_data_collator(batch, processor),
            shuffle=True,
            num_workers=0,
        )
    def val_dataloader(self):
        return DataLoader(
            val_dataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=lambda batch: val_data_collator(batch, processor),
            num_workers=0,
        )

early_stopping_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

class SaveCheckpoint(Callback):
    def __init__(self, result_path):
        self.result_path = result_path
        self.epoch = 0
    def on_train_epoch_end(self, trainer, pl_module):
        checkpoint_path = f"{self.result_path}/{self.epoch}"
        os.makedirs(checkpoint_path, exist_ok=True)
        pl_module.processor.save_pretrained(checkpoint_path)
        pl_module.model.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        self.epoch += 1
    def on_train_end(self, trainer, pl_module):
        checkpoint_path = f"{self.result_path}/latest"
        os.makedirs(checkpoint_path, exist_ok=True)
        pl_module.processor.save_pretrained(checkpoint_path)
        pl_module.model.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    limit_val_batches=1,
    num_sanity_val_steps=0,
    log_every_n_steps=10,
    callbacks=[SaveCheckpoint(result_path=config["result_path"]), early_stopping_callback],
)

model_module = Qwen2_5_Trainer(config, processor, model)
trainer.fit(model_module)