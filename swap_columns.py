import pandas as pd

# Đọc file Excel hiện có
df = pd.read_excel("test.xlsx")

# Đổi tên cột thành chữ thường
df.columns = df.columns.str.lower()

# Hoán đổi nội dung của hai cột
temp = df['text'].copy()
df['text'] = df['label']
df['label'] = temp

# Sắp xếp lại thứ tự cột để label đứng trước
df = df[["label", "text"]]

# Lưu lại file Excel
df.to_excel("test.xlsx", index=False)

print("Đã hoàn thành hoán đổi nội dung cột text và label.")
print("Cấu trúc dữ liệu mới:")
print(df.head())
