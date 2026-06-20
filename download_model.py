"""Pre-download model during build so first request is fast."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL)

print("Downloading PyTorch model...")
AutoModelForSequenceClassification.from_pretrained(MODEL)

print("Downloading ONNX model...")
ORTModelForSequenceClassification.from_pretrained(MODEL, export=True)

print("All models downloaded successfully!")
