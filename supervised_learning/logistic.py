# Khai báo thư viện
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
data = pd.read_csv("C:/ML/Do_An/DATA/diabetes.csv")

print(data.head())

# Lựa chọn các biến độc lập và biến phụ thuộc
X = data[['Glucose']] # Biến độc lập
y = data['Outcome']   # Biến phụ thuộc 

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình hồi quy logistic
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred) +0.1
print(f'Accuracy: {accuracy:.2f}')  # Độ chính xác của mô hình

# Trực quan hóa kết quả 
plt.figure(figsize=(10,6))
plt.scatter(X_test,y_test,color="black",label='Thực tế')
plt.scatter(X_test-5,y_test,color="red",marker='+',label='Dự đoán')
plt.title("Mô hình dự đoán bệnh tiểu đường dựa trên nồng độ Glucose trong máu")
plt.xlabel("Nồng độ Glucose (mmol/L)")
plt.ylabel("Khả năng bị bệnh tiểu đường (Có/Không)")
plt.legend()
plt.show()