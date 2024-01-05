from datetime import datetime
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd


def calculate_score(cluster_labels):
    return sum(cluster_labels) / len(cluster_labels) if len(cluster_labels) > 0 else 0.0


def process_file(file_path, file_type):
    data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line)
                entry["file_type"] = file_type  # 添加文件类型字段
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    comments = [entry["text"] for entry in data]
    # 只提取第一个评论的日期，因为所有评论的日期相同
    comment_date = datetime.strptime(data[0]["created_at"], "%a %b %d %H:%M:%S %z %Y").date()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([" ".join(comment) for comment in comments])

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_
    score = calculate_score(cluster_labels)

    for i, comment in enumerate(comments):
        print(f"Comment: {comment}, Cluster Label: {cluster_labels[i]}, File Type: {file_type}")

    print(f"File Type: {file_type}, Score: {score}")

    result = [comment_date, score]

    return result


def main():
    # 文件夹路径
    folder_path = r'/Dataset/tweet/preprocessed/AAPL'

    # 获取文件夹中所有文件的文件名
    file_names = os.listdir(folder_path)

    # 处理每个文件
    all_result = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        file_type = file_name.split('.')[0]  # 假设文件名是以文件类型开头，例如"file1.json"

        result = process_file(file_path, file_type)
        all_result.append(result)

    print(all_result)

    # 合并文件
    # 文件路径
    file_path = '/Dataset/price/preprocessed/AAPL.txt'

    # 使用 pd.read_csv 读取文本文件
    # 可以根据实际情况调整参数，例如分隔符（sep）、列名等
    df = pd.read_csv(file_path, sep='\t', header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Adj close', 'Volume'])

    # 打印 DataFrame 的内容
    #print(df)

    # 你的列表 all_result
    # 将列表转换为 DataFrame
    list_df = pd.DataFrame(all_result, columns=['Date', 'Score'])
    # 将 'Date' 列转换为字符串
    list_df['Date'] = list_df['Date'].astype(str)

    # 用于匹配的 DataFrame
    # 假设你的原始 DataFrame 叫做 df，其中包含 'Date' 列

    # 将两个 DataFrame 进行匹配，匹配成功的情况下将分数值添加到 'Score' 列中
    # 使用 how='left' 表示按照左边的 DataFrame（df）进行匹配
    result_df = pd.merge(df, list_df, on='Date', how='left')

    # 如果匹配不成功，将 'Score' 列填充为默认值（0.5）
    result_df['Score'].fillna(0.5, inplace=True)

    # 打印结果 DataFrame
    print(result_df)

    result_df.to_csv('D:\桌面\StockPredict\Dataset\price\Dataset.csv', index=False)

    print('保存成功')


if __name__ == "__main__":
    main()
