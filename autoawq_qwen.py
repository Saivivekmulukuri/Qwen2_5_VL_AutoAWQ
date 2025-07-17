import os
os.environ['CURL_CA_BUNDLE'] = ''

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
quant_path = "Qwen2_5-VL-3B-Instruct-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit":4, "version":"GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def load_docvqa_calibration_data(num_samples=64):
    """
    Load and preprocess DocVQA dataset for calibration using your approach.
    """
    # Your correct loading approach
    # ds = load_dataset("llms-lab/DocVQA", "DocVQA", split="test")
    dataset_options = [
        ("lmms-lab/DocVQA", "DocVQA"),
        ("nielsr/docvqa_1200_examples", None),
        ("HuggingFaceM4/VQAv2", None),  # Fallback
        ("lmms-lab/ChartQA", None),     # Another fallback
    ]
    
    ds = None
    for dataset_name, config in dataset_options:
        try:
            print(f"Trying to load dataset: {dataset_name}")
            if config:
                ds = load_dataset(dataset_name, config, split="test")
            else:
                ds = load_dataset(dataset_name, split="test")
            print(f"Successfully loaded: {dataset_name}")
            break
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue
    
    if ds is None:
        print("Could not load any dataset, using default calibration")
        return None
    ds = ds.shuffle(seed=42).select(range(num_samples))
    
    calibration_inputs = []

    for example in ds:
        try:
            # Convert image to base64 format
            buffered = BytesIO()
            example["image"].save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue())
            encoded_image_text = encoded_image.decode("utf-8")
            base64_qwen = f"data:image;base64,{encoded_image_text}"
            
            # Get question text from DocVQA format
            question_text = example["question"]
            
            # Format as conversation for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_qwen},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            
            # Apply chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            # calibration_texts.append(text)
            calibration_inputs.append(inputs['input_ids'][0].tolist())
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"Loaded {len(calibration_inputs)} calibration samples from DocVQA")
    # return calibration_texts
    return calibration_inputs

# Load DocVQA calibration data
print("Loading DocVQA calibration dataset...")
calib_data = load_docvqa_calibration_data(num_samples=64)

# Quantize with custom calibration data
print("Starting quantization with DocVQA calibration data...")

model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)#, n_parallel_calib_samples=4, max_calib_samples=64)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"Model quantized and saved to {quant_path}")