from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = "prajjwal1/bert-tiny"
print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL)
print("Downloading model...")
AutoModelForSequenceClassification.from_pretrained(MODEL)
print("Done!")
