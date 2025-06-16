import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoProcessor
import base64
from io import BytesIO
import torch
from PIL import Image
import glob

def quick_batch_inference(model_path, image_directory, prompt, max_tokens=100):
    """Quick batch inference function"""
    
    # Load model
    model = AutoAWQForCausalLM.from_quantized(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    device = torch.device("cuda:0")
    model = model.to(device)
    
    # Get all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_directory, ext)))
    
    print(f"Found {len(image_paths)} images")
    print(f"Using prompt: {prompt}")
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        try:
            print(f"\nProcessing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((448, 448), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue())
            base64_image = f"data:image;base64,{encoded_image.decode('utf-8')}"
            
            # Format input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Process vision inputs
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs, video_inputs = [image], None
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) if v is not None and hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
            inputs = {k: v for k, v in inputs.items() if v is not None}
            
            # Generate
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_tokens = output[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            results.append({
                'image': os.path.basename(image_path),
                'response': response
            })
            
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image': os.path.basename(image_path),
                'response': f"Error: {str(e)}"
            })
    
    return results

# Usage
if __name__ == "__main__":
    model_path = "Qwen2_5-VL-3B-Instruct-awq"
    image_dir = "/users/student/pg/pg23/saivivekmulukuri/Qwen_VL/documents"
    prompt = """You are an advanced information extraction system specialized in understanding various medical lab reports. Your task is to carefully analyze the given image and the text in the image, and extract relevant details to populate a JSON object following the provided schema.
JSON Schema:
```json

{
    "tests": [
        {
            "test_name": "string",         # name of the test
            "values": [                    # There could be multiple values of the same test on different dates mentioned
                {
                    "value": "string",     # mentioned value
                    "datetime": "string"   # datetime for that value mentioned in the report. There could be multiple values of one test on different dates. Or if there is only one value per test, the document date might be written on top of the document. If there are multiple dates, pick up the date of specimen sample collection
                }
            ]
            "range": "string",             # given normal range for that test
            "unit": "string",              # mentioned unit for the test
            "panel_name": "string",        # Panel name might be mentioned on top of a bunch of tests (e.g, CBC, Urinalysis, etc.)
            "specimen_text": "string",     # Specimen might be mentioned in the report either with each test or for a group of tests on top
            "method": "string",            # method of the test mentioned in the report

        }
    ]
}
```
Instructions:
1. For each field, parse its exact text (verbatim).
2. Ensure the generated output is a valid JSON string adhering to the schema.
3. Extract information strictly from the given text without making assumptions.
4. If certain information is not explicitly provided, mark it as null in the JSON.
5. Generate only the JSON object without any additional text.
6. Only parse information, if the given document is a Lab-report. Do not parse any information if the document is a Discharge-summary, prescription, scan document or an interpretation of a scan.
7. There could be multiple columns of tests in the document, or the document could be tilted or rotated, make sure you parse those documents properly and group the ranges, units, values etc with their tests correctly.
8. The report could have a graph longitudinal values of a test-name as well, Parse those tests along with all it's values and respective dates.
9. If the same test is reported with multiple units but without mentioning the test-name again, just add another element in the `tests` list with the different unit and respective value of that test.
10. if it's not a lab-report (it's okay if it's an allergy rpeort, echocardiogram report with test-value format or something similar), respond with empty dict: `{}`
11. If it's a lab-report, but there are no tests mentioned in it, respond like this: `{"tests": []}`
12. if the same test is done using conventional as well as SI units and the conventional and SI units are different, but the test-name was not mentioned again, just add another element in the `tests` list with the different unit and respective value of that test.
13. In graphs, check if unit is mentioned as axis-titles and parse that as unit if it is.
Note: `Conventional` and `SI` are types of unit. Not a method of doing a test so don't parse them as method."""
    
    results = quick_batch_inference(model_path, image_dir, prompt, 1000)
    
    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for result in results:
        print(f"\n{result['image']}: {result['response']}")
