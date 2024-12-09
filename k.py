import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
import emoji
import unicodedata
from underthesea import text_normalize
# Kiểm tra xem GPU có sẵn không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tải lại mô hình SVM đã lưu
svm_model = joblib.load('svm_model.pkl')

# Tải tokenizer và mô hình BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
def clean_text(text, keep_punct=False):
    """
    Clean and normalize Vietnamese text
    Args:
        text: Input text string
        keep_punct: Whether to keep important punctuation (default: False)
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ''

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Normalize Vietnamese text
    text = text_normalize(text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
    
    # Remove numbers but keep mixed text-numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Handle Vietnamese abbreviations
    abbr_dict = {
        'ko': 'không',
        'kg': 'không',
        'kh': 'không',
        'tks': 'cảm ơn',
        'dc': 'được',
        'dk': 'được',
        'đc': 'được',
        'đk': 'được',
        'ny': 'người yêu',
        'vs': 'với',
        'r': 'rồi',
        'wan': 'quan',
        'uk': 'ừ',
        'ntn': 'như thế nào'
    }
    for abbr, full in abbr_dict.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)
    
    if keep_punct:
        # Keep only specific punctuation
        text = re.sub(r'[^\w\s!?.,]', '', text)
    else:
        # Remove all punctuation
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle repeated characters (e.g., 'đẹppppp' -> 'đẹp')
    text = re.sub(r'(.)\1+', r'\1', text)
    
    return text
# Hàm để lấy sentence embedding từ BERT
def get_bert_embedding(sentence):
    cleaned_sentence = clean_text(sentence, keep_punct=True)
    inputs = tokenizer(cleaned_sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state
    sentence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()  # Mean pooling over tokens
    return sentence_embedding

# Vòng lặp để kiểm tra cảm xúc của câu nhập vào
while True:
    input_text = input("Nhập câu để kiểm tra cảm xúc (hoặc nhập 'q' để thoát): ")
    if input_text == 'q':
        break

    # Lấy embedding của câu nhập vào
    sentence_embedding = get_bert_embedding(input_text)
    
    # Dự đoán cảm xúc từ mô hình SVM
    predicted_sentiment = svm_model.predict([sentence_embedding])[0]  # Dự đoán 1 câu

    # Hiển thị kết quả
    if predicted_sentiment == 0:
        print("Cảm xúc: Tiêu cực")
    else:    
        print("Cảm xúc: Tích cực")
