# Khai báo thư viện 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("C:/ML/Do_An/DATA/diabetes.csv")

# Tính ma trận tương quan
correlation_matrix = df.corr()

# Vẽ biểu đồ heatmap của ma trận tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', fmt='.2f')
plt.xticks(rotation=45)
plt.title('Biểu đồ ma trận tương quan', fontsize='22', fontweight='bold', pad=15)
plt.show()