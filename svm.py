import torch
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import emoji
import unicodedata
from underthesea import text_normalize
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
# Kiểm tra xem GPU có sẵn không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tải tokenizer và mô hình BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)  # Chuyển mô hình BERT vào GPU

# Hàm để lấy sentence embedding từ BERT
def get_bert_embedding(sentence):
    
    cleaned_sentence = clean_text(sentence, keep_punct=True)
    inputs = tokenizer(cleaned_sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    
    # Đưa các tensor vào GPU nếu có
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Lấy embeddings của token và tính mean pooling
    embeddings = outputs.last_hidden_state
    sentence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()  # Mean pooling over tokens
    return sentence_embedding

# Tải dữ liệu từ file Excel (merged_file.xlsx)
ds = pd.read_excel('merged_file.xlsx')

# Kiểm tra cấu trúc của DataFrame (in ra vài dòng đầu tiên)
print(ds.head())

# Kiểm tra số lượng NaN trong nhãn
print(f"Số lượng NaN trong nhãn: {ds['label'].isna().sum()}")

# Loại bỏ các dòng có giá trị NaN trong cột 'label'
ds_clean = ds.dropna(subset=['label'])

# Kiểm tra lại sau khi loại bỏ NaN
print(f"Số lượng NaN trong nhãn sau khi loại bỏ: {ds_clean['label'].isna().sum()}")

# Chuyển các câu thành embeddings từ BERT
embeddings = [get_bert_embedding(text) for text in ds_clean['text']]

# Lấy nhãn
labels = ds_clean['label']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)

# Kiểm tra xem có NaN trong y_train và y_test sau khi chia
print(f"Số lượng NaN trong y_train: {y_train.isna().sum()}")
print(f"Số lượng NaN trong y_test: {y_test.isna().sum()}")

# Sử dụng SVM để huấn luyện phân loại
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = svm_model.predict(X_test)
print("Đánh giá mô hình:\n", classification_report(y_test, y_pred))

# Hàm dự đoán cảm xúc từ một câu nhập vào
def predict_sentiment(text):
    embedding = get_bert_embedding(text)
    prediction = svm_model.predict([embedding])
    return prediction[0]
import joblib
joblib.dump(svm_model, 'svm_model.pkl')
