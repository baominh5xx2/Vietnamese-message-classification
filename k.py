import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Kiểm tra xem GPU có sẵn không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Tải lại mô hình SVM đã lưu
svm_model = joblib.load('svm_model.pkl')

# Tải tokenizer và mô hình BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Hàm để lấy sentence embedding từ BERT
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
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
