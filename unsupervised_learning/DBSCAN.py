# Khai báo thư viện 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from datetime import datetime 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.decomposition import PCA 
from sklearn.cluster import DBSCAN 


# Đọc dữ liệu 
customer_data = pd.read_csv('C:/ML/Do_An/DATA/customers.csv',
                           delimiter='\t', index_col='ID')

print(customer_data.head(10))

# Trích xuất đặc trưng (Feature Engineering)
# Tạo cột thuộc tính “Age” theo năm sinh của khách hàng
customer_data['Age'] = customer_data.Year_Birth.apply(lambda x: 2021 - int(x))

# Tạo cột thuộc tính “Days_Since_Customer”
customer_data['Dt_Customer'] = pd.to_datetime(customer_data.Dt_Customer, format="%d-%m-%Y")
now = datetime.now()
customer_data['Days_Since_Customer'] = customer_data.Dt_Customer.apply(lambda x: (now - x).total_seconds()/ (60 * 60 * 24))

# Tạo cột thuộc tính “Fam_Size” từ tình trạng hôn nhân, số con/thanh thiếu niên
marital_map = {'Absurd': 1, 'Alone': 1, 'YOLO': 1, 'Single': 1,
              'Married': 2, 'Together': 2, 'Widow': 1, 'Divorced': 1}
customer_data['Marital_Status'] = customer_data.Marital_Status.map(marital_map) 
customer_data['Num_Kids'] = customer_data.Kidhome.values + customer_data.Teenhome.values
customer_data['Fam_Size'] = customer_data.Marital_Status.values + customer_data.Num_Kids.values

# Tạo cột thuộc tính “Num_Accepted” từ tổng số chiến dịch tiếp thị trước đó đã được khách hàng chấp nhận
customer_data['Num_Accepted'] = customer_data.AcceptedCmp1.values + customer_data.AcceptedCmp2.values + \
                                customer_data.AcceptedCmp3.values + customer_data.AcceptedCmp4.values + \
                                customer_data.AcceptedCmp5.values

# Tạo cột thuộc tính 'MntTotal' cho tổng số tiền chi tiêu cho tất cả các mặt hàng
customer_data['MntTotal'] = customer_data['MntWines'].values + customer_data['MntFruits'].values + \
                            customer_data['MntMeatProducts'].values + customer_data['MntFishProducts'].values + \
                            customer_data['MntWines'].values + customer_data['MntSweetProducts'].values + \
                            customer_data['MntGoldProds'].values

# Bỏ các tính năng không cần thiết khỏi tập dữ liệu gốc
customer_data.drop(['Dt_Customer', 'Year_Birth', 'AcceptedCmp1', 'AcceptedCmp2',
                    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Kidhome', 'Teenhome',
                   'Z_CostContact', 'Z_Revenue', 'Num_Kids', 'Marital_Status'],
                   axis=1, inplace=True)

print(customer_data.head(10))

# Xử lý giá trị thiếu (Missing Values)
print('Dataset Shape:', customer_data.shape)
print('-------------------------------')
print('Total Nulls Per Column:')
print(customer_data.isnull().sum())


# Quyết định ý nghĩa
imputer = SimpleImputer(strategy='mean')
imputer.fit(customer_data.Income.values.reshape(-1,1))
customer_data['Income'] = imputer.transform(customer_data.Income.values.reshape(-1,1))



# Xóa cột “Response” vì đây là mục tiêu của mô hình dự đoán tương lai
X, y = customer_data.drop('Response', axis=1).values, customer_data['Response'].values

# Tạo một biến áp cột gửi “Education” để được mã hóa và thu nhỏ lại
ct = ColumnTransformer([
    ('catagoric', OneHotEncoder(), [0]),
    ('numeric', StandardScaler(), list(range(1, len(X.T))))
])

# Gửi dữ liệu qua biến áp cột
X_transformed = ct.fit_transform(X)
print('Preprocessed Data:')
print(X_transformed[0])

# PCA
pca = PCA(n_components=3)
pca.fit(X_transformed)
X_reduced = pca.transform(X_transformed)

# DBSCAN 

# Before using BDSCAN 
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_reduced.T[1],X_reduced.T[2],X_reduced.T[0], c="blue",  edgecolors='k')
ax.set_xlabel("Feature 1", fontsize=15, labelpad=10)
ax.set_ylabel("Feature 2", fontsize=15, labelpad=10)
ax.set_zlabel("Feature 3", fontsize=15, labelpad=10)
ax.set_title("Before using BDSCAN for clustering", fontsize='22', fontweight='bold', pad=10)

plt.show()

# After using BDSCAN 
# Thiết lập tham số cho DBSCAN
db = DBSCAN(eps=0.73, min_samples=26)
db.fit(X_reduced)
clusters = db.labels_
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)

# Nếu số cụm lớn hơn 4, thì giữ lại chỉ 4 cụm
if n_clusters_ > 5:
    unique_labels = set(clusters)
    sorted_labels = sorted(list(unique_labels), key=lambda x: np.sum(clusters == x), reverse=True)[:5]
    mask = np.isin(clusters, sorted_labels)
    clusters[~mask] = -1
    n_clusters_ = 5
    n_noise_ = list(clusters).count(-1)

print('Cluster Predictions')
print('-------------------------------')
print("Number of clusters: %d" % n_clusters_)
print("Number of noise points: %d" % n_noise_)
print('Number of points per cluster:')
for i in range(n_clusters_):
    print('Cluster', i+1, ':', len(clusters[clusters==i]))

# Tạo danh sách nhãn cho từng cụm
cluster_labels = ['Cluster {}'.format(i+1) for i in range(n_clusters_)]
cluster_labels.append('Outliers')

# Trực quan hóa kết quả
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_clusters_+1)]

for k, col in zip(range(-1, n_clusters_), colors):
    class_member_mask = (clusters == k)
    xy = X_reduced[class_member_mask]
    ax.scatter(xy.T[0], xy.T[1], xy.T[2], c=[col], edgecolors='k', s=20, label=cluster_labels[k+1])

ax.set_title("Customer Clusters in 3 Dimensions", fontsize='30', fontweight='bold', pad=15)
ax.legend()

plt.show()

# Thiết lập tham số cho DBSCAN
import time
# Bắt đầu đo thời gian
start_time = time.time()

db = DBSCAN(eps=0.73, min_samples=26)
db.fit(X_reduced)

# Kết thúc đo thời gian
end_time = time.time()

# Tính thời gian thực hiện thuật toán (đơn vị: giây)
execution_time = end_time - start_time

print(f"Thời gian thực hiện thuật toán K-means: {execution_time} giây")

clusters = db.labels_

# Giữ lại chỉ 4 cụm nếu số lượng cụm lớn hơn 5
n_clusters_ = min(5, len(set(clusters)) - (1 if -1 in clusters else 0))
n_noise_ = list(clusters).count(-1)

print('Cluster Predictions')
print('-------------------------------')
print(f"Number of clusters: {n_clusters_}")
print(f"Number of noise points: {n_noise_}")
print('Number of points per cluster:')
for i in range(n_clusters_):
    print(f'Cluster {i+1}: {np.sum(clusters==i)}')

# Tạo danh sách nhãn cho từng cụm
cluster_labels = [f'Cluster {i+1}' for i in range(n_clusters_)]
cluster_labels.append('Outliers')

# Trực quan hóa kết quả
fig, ax = plt.subplots(figsize=(45, 30), subplot_kw={'projection': '3d'})
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_clusters_+1)]

for k, col in zip(range(-1, n_clusters_), colors):
    class_member_mask = (clusters == k)
    xy = X_reduced[class_member_mask]
    ax.scatter(xy.T[0], xy.T[1], xy.T[2], c=[col], edgecolors='k', s=20, label=cluster_labels[k+1])

ax.set_title("After using BDSCAN for clustering", fontsize=24, fontweight='bold', pad=15)
ax.set_xlabel("Feature 1", fontsize=16, labelpad=10)
ax.set_ylabel("Feature 2", fontsize=16, labelpad=10)
ax.set_zlabel("Feature 3", fontsize=16, labelpad=10)
ax.legend()

plt.show()