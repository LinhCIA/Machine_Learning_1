# Phân cụm nhóm khách hàng bằng phương pháp K-Means

# Khai báo thư viện 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.cluster import KMeans
import time


# Đọc dữ liệu
data = pd.read_csv("C:/ML/Do_An/DATA/customers.csv", sep='\t')

# Sao lưu dữ liệu để tránh mất mát thông tin
customer = data.copy()

# Chuẩn bị dữ liệu 
print("\n")
customer = customer.dropna() # Xử lý dữ liệu thiếu (Missing values)
print("Kích thước của tập dữ liệu sau khi xử lý: ")
print(customer.shape)

# Nhóm dữ liệu 
import datetime as dt
customer['Age'] = 2015 - customer.Year_Birth

customer['TotalSpendings'] =  customer.MntWines + customer.MntFruits + customer.MntMeatProducts + customer.MntFishProducts + customer.MntSweetProducts + customer.MntGoldProds

customer.loc[(customer['Age'] >= 13) & (customer['Age'] <= 19), 'AgeGroup'] = 'Teen'
customer.loc[(customer['Age'] >= 20) & (customer['Age']<= 39), 'AgeGroup'] = 'Adult'
customer.loc[(customer['Age'] >= 40) & (customer['Age'] <= 59), 'AgeGroup'] = 'Middle Age Adult'
customer.loc[(customer['Age'] > 60), 'AgeGroup'] = 'Senior Adult'

customer['Children'] = customer['Kidhome'] + customer['Teenhome']

customer.Marital_Status = customer.Marital_Status.replace({'Together': 'Partner',
                                                           'Married': 'Partner',
                                                           'Divorced': 'Single',
                                                           'Widow': 'Single', 
                                                           'Alone': 'Single',
                                                           'Absurd': 'Single',
                                                           'YOLO': 'Single'})

# Xử lý giá trị nhiễu (outliers)
customer = customer[customer.Age < 100]
customer = customer[customer.Income < 120000]

# Biểu đồ trước khi sử dụng K-means để phân cụm khách hàng
plt.figure(figsize=(20,10))

sns.scatterplot(x=customer.Income, y=customer.TotalSpendings, s=100, edgecolors='k', linewidth=1, zorder=2)
sns.set_style("whitegrid")

plt.grid(True, which='both', linestyle='-', linewidth=0.5, zorder=1)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.xlabel('Income', fontsize=20, labelpad=10)
plt.ylabel('Total Spendings', fontsize=20, labelpad=20)
plt.title("Before using K-means to cluster customer groups", fontsize='26', fontweight='bold', pad=15)

plt.show()


# Các bước huấn luyện mô hình học tập không giám sát sử dụng thuật toán K-MEANS 
# Chọn số lượng cụm, kí hiệu là K 
k = 4 # => Phương pháp Khuỷu tay (Elbow Method) 

# Loại bỏ các cột không cần thiết khỏi tập dữ liệu 
X = customer.drop(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits','MntMeatProducts',
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','Dt_Customer', 'Z_CostContact',
                          'Z_Revenue', 'Recency', 'NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases',
                          'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                          'AcceptedCmp1', 'AcceptedCmp2', 'Complain',  'Response', 'AgeGroup'], axis=1)

# Bắt đầu đo thời gian
start_time = time.time()

# Khởi tạo mô hình phân cụm với k = 4 
model = KMeans(n_clusters=4, init='k-means++', random_state=42).fit(X)

preds = model.predict(X)

# Kết thúc đo thời gian
end_time = time.time()

# Tính thời gian thực hiện thuật toán (đơn vị: giây)
execution_time = end_time - start_time

print(f"Thời gian thực hiện thuật toán K-means: {execution_time} giây")

customer_kmeans = X.copy()
customer_kmeans['clusters'] = preds

# Khai phá dữ liệu 
customer_kmeans.clusters = customer_kmeans.clusters.replace({1: 'Silver',
                                                             2: 'Gold',
                                                             3: 'Bronze',
                                                             0: 'Platinum'})

customer['clusters'] = customer_kmeans.clusters


# Biểu đồ biểu thị dữ liệu sau khi phân cụm 
# Đặt màu cho từng cụm dữ liệu
palette_colors = {'Bronze': '#B87333', 'Silver': '#A9A9A9', 'Gold': '#FFD700', 'Platinum': '#66CDAA'}

plt.figure(figsize=(20, 10))

# Sử dụng palette để đặt màu cho từng cụm dữ liệu
sns.scatterplot(data=customer, x='Income', y='TotalSpendings', hue='clusters', palette=palette_colors, s=100, edgecolors='k', linewidth=1, zorder=2,hue_order=['Platinum', 'Gold', 'Silver', 'Bronze'])
sns.set_style("whitegrid")

# Thêm nhân của từng cụm (chữ "X" in hoa) màu đỏ 
for cluster in set(customer['clusters']):
    cluster_center = customer[customer['clusters'] == cluster][['Income', 'TotalSpendings']].mean()
    plt.text(cluster_center['Income'], cluster_center['TotalSpendings'], f'×', color='red', fontsize=50, fontweight='bold', ha='center', va='center')

plt.grid(True, which='both', linestyle='-', linewidth=0.5, zorder=1)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Income', fontsize=20, labelpad=10)
plt.ylabel('Total Spendings', fontsize=20, labelpad=20)

plt.legend(title='Clusters', loc='upper right')
plt.title('After using K-means to cluster customer groups', fontsize='27', fontweight='bold', pad=15)

plt.show()