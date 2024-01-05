import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 读取 JSON 文件
data = []

with open(r'D:\桌面\StockPredict\Dataset\tweet\preprocessed\AAPL\2014-01-01', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            entry = json.loads(line)
            data.append(entry)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# 从每个数据点中提取文本
comments = [entry["text"] for entry in data]

# 将文本数据转换为TF-IDF向量
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([" ".join(comment) for comment in comments])

# 使用K均值聚类将评论分为两类
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# 获取每个评论的聚类标签
cluster_labels = kmeans.labels_

# 打印每个评论及其聚类标签
for i, comment in enumerate(comments):
    print(f"Comment: {comment}, Cluster Label: {cluster_labels[i]}")
