# Khai báo thư viện
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc dữ liệu
df = pd.read_csv("C:/ML/Do_An/DATA/diabetes.csv")

# Sao lưu dữ liệu để tránh mất mát thông tin 
data = df.copy()

# Trích xuất đặc trưng feature và biến mục tiêu taget
feature = ['Glucose', 'BMI', 'Age']
target = ['Outcome']

X = data[['Glucose', 'BMI', 'Age']]
y = data['Outcome']

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Định nghĩa các giá trị của tham số C cần thử
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Tạo mô hình SVM
svm_model = SVC(kernel='linear')

# Tạo đối tượng GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train_scaled, y_train)

# Lấy giá trị tối ưu của tham số C
best_C = grid_search.best_params_['C']

# In ra giá trị tối ưu của tham số C
print(f"Giá trị tối ưu của tham số C: {best_C}")

# Huấn luyện mô hình SVM với giá trị C tối ưu
final_svm_model = SVC(kernel='linear', C=best_C)
final_svm_model.fit(X_train_scaled, y_train)

# In ra các độ đo chất lượng mô hình cuối cùng
y_pred_final = final_svm_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred_final))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))


# Trực quan hóa mô hình trong không gian 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Tạo các điểm trong không gian 3D và thêm phân chú thích
scatter = ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], X_train_scaled[:, 2], c=y_train, cmap='viridis', s=50)
legend_labels = ['Class 0 (Not Diabetic)', 'Class 1 (Diabetic)']
ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels)

# Tạo mặt phẳng Hyperplane
xx, yy = np.meshgrid(np.linspace(X_train_scaled[:, 0].min(), X_train_scaled[:, 0].max(), 50),
                     np.linspace(X_train_scaled[:, 1].min(), X_train_scaled[:, 1].max(), 50))
zz = (-final_svm_model.intercept_[0] - final_svm_model.coef_[0, 0] * xx - final_svm_model.coef_[0, 1] * yy) / final_svm_model.coef_[0, 2]
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

# Thêm mũi tên và chú thích cho mặt phẳng Hyperplane
arrow_start = [X_train_scaled[:, 0].mean(), X_train_scaled[:, 1].mean(), X_train_scaled[:, 2].min()]
arrow_vector = [0, 0, 15]  # Thay đổi chiều dài của mũi tên
ax.quiver(*arrow_start, *arrow_vector, color='red', arrow_length_ratio=0.1)

# Đặt vị trí và chú thích cho mặt phẳng Hyperplane
arrow_end = [arrow_start[0] + arrow_vector[0], arrow_start[1] + arrow_vector[1], arrow_start[2] + arrow_vector[2]]
text_position = [arrow_end[0], arrow_end[1], arrow_end[2] + 1]
ax.text(*text_position, 'Hyperplane', color='red', weight='bold', horizontalalignment='center', verticalalignment='bottom')

# Đặt tên trục
ax.set_xlabel('Glucose (Scaled)', fontsize='14', labelpad=5)
ax.set_ylabel('BMI (Scaled)', fontsize='14', labelpad=5)
ax.set_zlabel('Age (Scaled)', fontsize='14', labelpad=5)

# Hiển thị biểu đồ
plt.title('Utilizing SVM for Diabetes Classification', fontsize='24', fontweight='bold', pad=12)

plt.show() # 10 điểm không có nhưng 