import numpy as np


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Khởi tạo các tham số
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Tính gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y).T)  # fix commit
            
            # Kiểm tra và thay thế giá trị NaN và inf thành giá trị Mean(X)
            for i in range(len(dw)):
                if np.isnan(dw[i]) or np.isinf(dw[i]):
                    dw[i] = np.mean(X)                               # fix commit
                        
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Cập nhật các tham số 
            self.weights -= self.lr * dw.reshape(self.weights.shape) # fix commit
            self.bias -= self.lr * db


    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated