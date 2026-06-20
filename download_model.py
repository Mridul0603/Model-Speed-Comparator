"""Pre-download model during build so first request is fast."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL)

print("Downloading PyTorch model...")
AutoModelForSequenceClassification.from_pretrained(MODEL)

print("All models downloaded successfully!")
