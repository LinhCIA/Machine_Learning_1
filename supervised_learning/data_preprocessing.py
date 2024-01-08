# Tiền xử lý dữ liệu
# Khai báo các thư viện
import pandas as pd 


# Đọc dữ liệu
df = pd.read_csv("C:/ML/Do_An/DATA/diabetes.csv")
print("*========== Hiển thị 10 hàng đầu tiên của tập dữ liệu ==========*")
print(df.head(10))

# Thống kê mô tả
print(df.describe())

print("\n")
# Kiểm tra tổng số giá trị thiếu trong từng cột
missing_values = df.isnull().sum()

# Kiểm tra tổng số giá trị thiếu trong toàn bộ DataFrame
total_missing = df.isnull().sum().sum()

# In thông tin về giá trị thiếu
print("Tổng số giá trị thiếu trong từng cột:")
print(missing_values)

print("\nTổng số giá trị thiếu trong toàn bộ tập dữ liệu:", total_missing)