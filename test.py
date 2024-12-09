import torch
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state
    sentence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
    return sentence_embedding

def plot_svm_with_margins(X_train, y_train):
    # Convert embeddings to numpy array
    X_train_array = np.array(X_train)
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_train_array)
    
    # Train SVM on 2D data
    svm_2d = SVC(kernel='rbf', gamma='auto', C=1.0, random_state=42)
    svm_2d.fit(X_2d, y_train)
    
    # Create mesh grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get decision function
    Z = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    # Plot points
    plt.scatter(X_2d[y_train == 0, 0], X_2d[y_train == 0, 1], 
               c='red', label='Class 0', s=50)
    plt.scatter(X_2d[y_train == 1, 0], X_2d[y_train == 1, 1], 
               c='blue', label='Class 1', s=50)
    
    # Plot support vectors
    plt.scatter(X_2d[svm_2d.support_, 0], X_2d[svm_2d.support_, 1],
               s=100, linewidth=1, facecolors='none', 
               edgecolors='green', label='Support Vectors')
    
    plt.title('SVM Decision Boundary with Margins')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load and process data
ds = pd.read_excel('merged_file.xlsx')
print(ds.head())
print(f"Số lượng NaN trong nhãn: {ds['label'].isna().sum()}")

ds_clean = ds.dropna(subset=['label'])
print(f"Số lượng NaN trong nhãn sau khi loại bỏ: {ds_clean['label'].isna().sum()}")

# Create embeddings
embeddings = [get_bert_embedding(text) for text in ds_clean['text']]
labels = ds_clean['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)

print(f"Số lượng NaN trong y_train: {y_train.isna().sum()}")
print(f"Số lượng NaN trong y_test: {y_test.isna().sum()}")

# Train SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
print("Đánh giá mô hình:\n", classification_report(y_test, y_pred))

# Plot decision boundary
plot_svm_with_margins(X_train, y_train)

def predict_sentiment(text):
    embedding = get_bert_embedding(text)
    prediction = svm_model.predict([embedding])
    return prediction[0]
