
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = None
tokenizer = None

# Verified from your dataset's ClassLabel metadata.
LABEL_MAP = {
    0: "Negative",
    1: "Positive",
}


def init():
    global model, tokenizer

    model_dir = os.environ["AZUREML_MODEL_DIR"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()


def run(raw_data):
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

        if not isinstance(data, dict):
            return {"error": "Request must be a JSON object."}

        text = data.get("text")

        if not isinstance(text, str) or not text.strip():
            return {"error": "Please provide non-empty text."}

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())

        return {
            "sentiment": LABEL_MAP.get(predicted_id, f"Class {predicted_id}"),
            "confidence": round(float(probabilities[predicted_id].item()), 4),
        }

    except Exception:
        return {"error": "Unable to process the sentiment request."}
