from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dữ liệu
data = pd.read_csv("C:/ML/Do_An/DATA/diabetes.csv", index_col=None)

# Chọn các thuộc tính và cột mục tiêu
features = ['Age', 'Pregnancies', 'Glucose', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','BloodPressure']
target = 'Outcome'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# In ra các mô hình và điểm số đánh giá
print(models)
print(predictions)