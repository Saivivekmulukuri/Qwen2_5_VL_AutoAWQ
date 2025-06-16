import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset
import base64
from io import BytesIO
import torch
from PIL import Image

# Load quantized model
quant_path = "Qwen2_5-VL-3B-Instruct-awq"

model = AutoAWQForCausalLM.from_quantized(quant_path)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

# Configure processor with conservative pixel limits
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
processor = AutoProcessor.from_pretrained(
    quant_path, 
    min_pixels=min_pixels, 
    max_pixels=max_pixels,
    trust_remote_code=True
)

device = torch.device("cuda:0")
model = model.to(device)

def preprocess_image_conservative(image):
    """Conservative image preprocessing for quantized models"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Use smaller target size for quantized models
    target_size = (448, 448)  # Multiple of 28
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image

# Load DocVQA samples
ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="test")
samples = ds.shuffle(seed=123).select(range(5))

# Run inference with enhanced error handling
for i, example in enumerate(samples):
    print(f"\n{'='*60}")
    print(f"DocVQA Sample {i+1}")
    print(f"{'='*60}")
    
    try:
        # Apply conservative preprocessing
        original_image = example["image"]
        processed_image = preprocess_image_conservative(original_image)
        
        # print(f"Original size: {original_image.size}")
        # print(f"Processed size: {processed_image.size}")
        
        # Convert to base64
        buffered = BytesIO()
        processed_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        
        question_text = example["question"]
        print(f"Question: {question_text}")
        
        # Format input with explicit image size control
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": base64_qwen,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process with conservative settings
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs, video_inputs = [processed_image], None
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=512,  # Reduced max length
            return_tensors="pt"
        )
        
        # Debug output
        # print(f"Input IDs shape: {inputs['input_ids'].shape}")
        # if 'pixel_values' in inputs:
        #     print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        
        # Move to device
        inputs = {k: v.to(device) if v is not None and hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        inputs = {k: v for k, v in inputs.items() if v is not None}
        
        # Generate with conservative parameters
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_tokens = output[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"Generated Answer: {generated_text.strip()}")
        
        if "answers" in example and example["answers"]:
            answers = example["answers"]
            if isinstance(answers, list):
                print(f"Ground Truth: {', '.join(answers)}")
            else:
                print(f"Ground Truth: {answers}")
                
    except Exception as e:
        print(f"Error processing sample {i+1}: {e}")
        continue

print("DocVQA inference completed!")
