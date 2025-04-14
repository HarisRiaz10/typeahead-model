import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer, device

# Process input
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        request = json.loads(request_body)
        return request["text"]
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# Run inference
def predict_fn(input_data, model_data):
    model, tokenizer, device = model_data

    # If input_data is a list, process all
    if isinstance(input_data, str):
        input_data = [input_data]

    inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=5,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    
    decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return decoded

# Return output in JSON format
def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps({"suggestions": prediction})
    raise ValueError(f"Unsupported return content type: {content_type}")
