# Phân cụm nhóm khách hàng bằng phương pháp PCA 
# Khai báo thư viện 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
np.random.seed(42)
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Đọc dữ liệu 
df = pd.read_csv("C:/ML/Do_An/DATA/customers.csv", sep='\t')

# Sao lưu dữ liệu để tránh mất mát thông tin
data= df.copy()

# Hiển thị dữ liệu 
print("*========== Hiển thị 10 hàng đầu tiên của tập dữ liệu ==========*")
print(data.head(10))

# Chuẩn bị dữ liệu 
data_scaled = data.copy()

drop_cols = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response',
             'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

data_scaled.drop(columns = drop_cols, axis = 1, inplace = True)

print(data_scaled.shape)

# Lựa chọn các cột số cần sử dụng cho PCA
numeric_cols = ['Income', 'Kidhome', 'Teenhome', 'Recency',
       'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
       'NumStorePurchases', 'NumWebVisitsMonth', 'Cus_for', 'Age', 'Children', 'Spent_All', 'PurchaseNumAll',
       'PurDeal_PurAll_ratio', 'Total_Promos', 'Family_Size']


# Chuẩn hóa bằng phương pháp chuẩn hóa Z-score
ss = StandardScaler()

for col in numeric_cols:
    data_scaled[col] = ss.fit_transform(data_scaled[[col]])
    
print(data_scaled.shape, data_scaled.head(10))

# Trích xuất biến bằng phân tích thành phần chính (PCA)

""" 
    Sử dụng PCA, để giải quyết các vấn đề sau:
    +)  Giảm chiều dữ liệu 
    +)  Đa cộng tuyến giữa các biến 
    +)  Biểu diễn trực quan 
"""

print(data_scaled.head(10))

variance_ratio = {}

for i in range(1, len(data_scaled.columns)+1):
    pca = PCA(n_components=i)
    pca.fit(data_scaled)
    variance_ratio[f'n_{i}'] = pca.explained_variance_ratio_.sum()

print(variance_ratio)

data_pca5 = pd.DataFrame(pca.components_[0:5],
                         columns=data_scaled.columns,
                         index = ['PC1','PC2','PC3', 'PC4', 'PC5']).T

print(data_pca5)

# Về lý thuyết, tỷ lệ phương sai phải đạt ít nhất 70%. Mà trong trường hợp giảm về còn 4 thành phần chính thì kết quả đạt gần 70% (chưa tới 70%) nên chọn giảm về 5 thành phần chính để đảm bảo khả năng giải thích phương sai lớn hơn 70% là phù hợp.

# Bắt đầu đo thời gian
start_time = time.time()

pca = PCA(n_components = 5, random_state = 42) # Với số chiều là 3

pca.fit(data_scaled)
data_pca = pd.DataFrame(pca.transform(data_scaled), 
                        columns = (["PC1", "PC2", "PC3", "PC4", "PC5"]))

print(data_pca.describe().T)

# Kết thúc đo thời gian
end_time = time.time()

# Tính thời gian thực hiện thuật toán (đơn vị: giây)
execution_time = end_time - start_time

print(f"Thời gian thực hiện thuật toán K-means: {execution_time} giây")

# Thực hiện phân cụm với k = 4 
km = KMeans(n_clusters=4, random_state=42)

yhat_AC = km.fit_predict(data_pca)

data_pca["Clusters"] = yhat_AC 
data["Clusters"]= yhat_AC  

# PCA = 3
x = data_pca["PC1"]
y = data_pca["PC2"]
z = data_pca["PC3"]

# Thiết lập màu sắc cho biểu đồ dựa trên số lượng clusters
palette = sns.color_palette("husl", n_colors=len(set(data_pca["Clusters"])))

# Tạo biểu đồ 3D với kích thước lớn hơn
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

# Vẽ scatter plot với màu nổi bật cho từng cụm
sc = ax.scatter(x, y, z, s=50, c=data_pca["Clusters"], marker='o', edgecolors='k', linewidth=0.5, alpha=1, cmap='RdBu')

# Đặt tiêu đề và nhãn trục
ax.set_title("Using PCA to cluster customer groups", fontsize=18, fontweight='bold')
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")

# Hiển thị chú thích
legend_labels = [f'Cluster {i+1}' for i in set(data_pca["Clusters"])]
ax.legend(handles=sc.legend_elements()[0], title='Clusters', labels=legend_labels)

# Hiển thị biểu đồ
plt.show()