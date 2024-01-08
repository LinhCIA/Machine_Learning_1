# Khai báo thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Đọc dữ liệu
df = pd.read_csv("C:/ML/Do_An/DATA/diabetes.csv")

# Trích xuất đặc trưng
features = ['Glucose', 'BMI']
class_label = 'Outcome'

# Biến độc lập
X = df[features].values

# Biến phụ thuộc
y = df[class_label].values

# Chia tập dữ liệu thành tập dữ liệu huấn luyện và tập kiểm tra (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình KNN với K = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Tạo 2 điểm dữ liệu mới có tọa độ là 
new_1 = [170,40]
new_2 = [115,20]
new_point = np.array([new_1, new_2])

# Dự đoán trên tập kiểm thử
y_pred = knn.predict(X_test)

# Dự đoán trên điểm dữ liệu mới
predicted_class = knn.predict(new_point)
print(predicted_class)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f'Độ chính xác trên tập kiểm thử: {accuracy:.2f}')

# Trực quan hóa kết quả 
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.scatter(new_point[:,0], new_point[:,1], c=predicted_class,marker='+' , s=1000)
plt.title('Mô hình KNN dự đoán bệnh nhân mắc bệnh tiểu đường', fontsize='22', fontweight='bold', pad=15)
plt.xlabel('Glucose', fontsize='20', labelpad=12)
plt.ylabel('BMI', fontsize='20', labelpad=12)
offset = 3
plt.text(new_1[0], new_1[1] - offset, 'New point', fontsize=10, ha='center')
plt.text(new_2[0], new_2[1] - offset, 'New point', fontsize=10, ha='center')
plt.show()
