# Đổi tên cột thành chữ thường
df.columns = df.columns.str.lower()

# Hoán đổi nội dung của hai cột
temp = df['text'].copy()
df['text'] = df['label']
df['label'] = temp

# Sắp xếp lại thứ tự cột để label đứng trước
df = df[["label", "text"]]