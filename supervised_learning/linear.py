# Khai báo thư viện 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Đọc dữ liệu 
doc = pd.read_csv("C:/ML/Do_An/DATA/diabetes.csv")

# Biến độc lập
doclap = ['BMI']

# Biến phụ thuộc
phuthuoc = 'BloodPressure'

X = doc[doclap]
y = doc[phuthuoc]

# Chia dữ liệu để huấn luyện (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2, random_state=42)
Lreg = LinearRegression()
Lreg.fit(train_data, train_target)

# Dự đoán trên tập kiểm thử X_test
y_pred = Lreg.predict(test_data)

# Trực quan hóa kết quả 
plt.figure(figsize=(8,6))
plt.scatter(test_data,test_target, color='blue')
plt.plot(test_data,y_pred,color='red',linewidth=2)
plt.xlabel('BMI')
plt.ylabel('BloodPressure')
plt.title('Mô hình tuyến tính dự đoán BloodPressure dựa trên BMI')
plt.show()