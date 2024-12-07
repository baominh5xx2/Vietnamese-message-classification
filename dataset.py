import pandas as pd

# Hàm kiểm tra file Excel
def check_excel_file(file_path):
    try:
        # Đọc file Excel
        df = pd.read_excel(file_path)
        
        # Kiểm tra xem có cột 'text' và 'label' trong dataframe không
        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            return f"File không có các cột yêu cầu: {required_columns}"

        # Kiểm tra có bất kỳ giá trị NaN trong cột 'text' và 'label' không
        missing_text = df['text'].isna()
        missing_label = df['label'].isna()

        # In ra dòng có NaN trong cột 'text'
        if missing_text.any():
            print("Các dòng có giá trị NaN trong cột 'text':")
            print(df[missing_text].index.tolist())

        # In ra dòng có NaN trong cột 'label'
        if missing_label.any():
            print("Các dòng có giá trị NaN trong cột 'label':")
            print(df[missing_label].index.tolist())

        # Kiểm tra xem cột 'label' có phải là số (ví dụ: 0 hoặc 1 cho sentiment) không
        if not pd.api.types.is_numeric_dtype(df['label']):
            print("Cột 'label' không phải là kiểu số (numeric).")
            non_numeric_labels = df[~df['label'].apply(lambda x: isinstance(x, (int, float)))].index.tolist()
            print(f"Các dòng có giá trị không phải kiểu số trong cột 'label': {non_numeric_labels}")

        # Kiểm tra xem cột 'text' có phải là kiểu chuỗi không
        non_string_text = df[~df['text'].apply(lambda x: isinstance(x, str))].index.tolist()
        if non_string_text:
            print(f"Các dòng có giá trị không phải kiểu chuỗi trong cột 'text': {non_string_text}")

        # Nếu không có vấn đề gì
        return "Kiểm tra xong."

    except Exception as e:
        return f"Lỗi khi mở file: {str(e)}"

# Thử kiểm tra file Excel
file_path = 'merged_file.xlsx'  # Đường dẫn đến file của bạn
result = check_excel_file(file_path)
print(result)
