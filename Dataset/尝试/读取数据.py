# 作者：刘文龙
# 学校院系：汕头大学21级计算机科学与技术
# 创作时间 2024-01-01 17:23
import json

file_path = 'tweet/preprocessed/AAPL/2014-01-01'  # 请替换成你的文件路径

# 读取文件中的数据
with open(file_path, 'r', encoding='utf-8') as file:
    # 逐行读取数据并解析为JSON格式
    tweets = [json.loads(line) for line in file if line.strip()]

# 打印解析后的结果
for tweet in tweets:
    print(tweet)