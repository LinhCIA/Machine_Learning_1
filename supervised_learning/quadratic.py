# Khai báo thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV


# Đọc dữ liệu 
data = pd.read_csv("C:/ML/Do_An/DATA/diabetes.csv")

# Chuẩn bị dữ liệu
X = data[['Glucose', 'BMI']]
y = data['Outcome']

# Thiết lập lưới tìm kiếm siêu tham số
param_grid = {'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}

# Huấn luyện mô hình QDA với lưới tìm kiếm siêu tham số
qda = QuadraticDiscriminantAnalysis()
grid_search = GridSearchCV(qda, param_grid, cv=5)
grid_search.fit(X, y)

# In ra giá trị tốt nhất cho siêu tham số
print("Best hyperparameters:", grid_search.best_params_)

# Trực quan hóa ranh giới quyết định với mô hình đã được tinh chỉnh
best_qda = grid_search.best_estimator_

# Trực quan hóa ranh giới quyết định
plt.figure(figsize=(8, 6))

# Tìm giá trị tối thiểu và tối đa của các đặc trưng để vẽ ranh giới quyết định
x_min, x_max = X['Glucose'].min() - 1, X['Glucose'].max() + 1
y_min, y_max = X['BMI'].min() - 1, X['BMI'].max() + 1

# Tạo ra lưới dữ liệu để vẽ ranh giới quyết định
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Dự đoán kết quả cho từng điểm trong lưới
Z = best_qda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ đường ranh giới quyết định
plt.contourf(xx, yy, Z, alpha=0.4)

# Vẽ điểm dữ liệu và chú thích màu sắc
scatter = plt.scatter(X['Glucose'], X['BMI'], c=y, s=25, edgecolor='k')

# Tạo chú thích thủ công
handles = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='yellow', markersize=10),
           plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='purple', markersize=10)]
labels = ['No Diabetes', 'Diabetes']

# Hiển thị chú thích
plt.legend(handles=handles, labels=labels, title="    Outcome")

plt.xlabel('Glucose', fontsize='20', labelpad=10)
plt.ylabel('BMI', fontsize='20', labelpad=10)
plt.title('Quadratic Discriminant Analysis', fontsize='25', fontweight='bold', pad=15)

# Hiển thị biểu đồ
plt.show()