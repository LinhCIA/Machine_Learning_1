# Ứng dụng mô hình Linear Regression đã tạo từ trước vào bài toán dự đoán
import numpy as np
from debugLinearRegression import LinearRegression


# Dữ liệu đầu vào
X = np.array([[100.0], [200.0], [300.0], [100.0], [200.0], [300.0]])

y = np.array([1000.0, 2000.0, 3000.0,1000.0, 2000.0, 3000.0 ])

# Khởi tạo mô hình Linear Regression
model = LinearRegression()

# Huấn luyện mô hình với dữ liệu đầu vào đã cho ở trên 
model.fit(X, y)

# Giá trị x-dự đoán
x_predict = np.array([[185.0]])

# Giá trị y-dự đoán
y_predict = model.predict(x_predict)

# Kết quả đầu ra : Với x = 185.0 thì y = 1850.000000023152
# => Kết quả dự đoán của mô hình gần như chính xác với kết quả đầu ra mong muốn (1850.0)
# Mặc dù chưa chính xác hoàn toàn nhưng em đã cố gắng tìm và debug lỗi. Mong thầy rộng lượng đừng trách cứ em 😔
print(f"Với giá trị x_predict là 185.0 thì giá trị y_predict tương ứng là {y_predict[0]}")
