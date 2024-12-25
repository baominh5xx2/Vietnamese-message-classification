# Load model directly
from transformers import AutoModelForSequenceClassification, AutoTokenizer
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)