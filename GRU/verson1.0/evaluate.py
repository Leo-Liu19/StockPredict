import torch
import math
import numpy as np
from sympy import true

from GRUModel import GRUModel
from dataset import getData
from parser_my import args
import csv
import pandas as pd

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 定义一个评估模型的函数
def eval():
    # 创建模型实例
    model = GRUModel(input_size=args.input_size,
                     hidden_size=args.hidden_size,
                     num_layers=args.layers,
                     output_size=1)
    # 将模型移动到指定的设备上（例如GPU）
    model.to(args.device)
    # 加载模型的预训练权重
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    # 将模型设置为评估模式，这样就不会进行梯度更新
    model.eval()
    # 初始化用于存储预测和标签的列表
    predict = []
    preds = []
    labels = []
    # 加载测试数据集
    close_max, close_min, _, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)

    # 使用torch.no_grad()，在这个块下的代码不会计算梯度，节省内存并加速计算
    with torch.no_grad():
        # 迭代测试数据加载器
        for idx, (x, label) in enumerate(test_loader):
            # 将输入数据移动到指定的设备上，并且可能需要调整形状
            x = x.squeeze(1).to(args.device)
            # 获取模型的预测结果
            pred = model(x)
            # 选取预测序列的最后一个时间步的输出
            pred = pred[:, -1]  # Note: 这一步可能是多余的，因为模型设计时已经只取最后一个时间步

            # 对模型的输出进行逆标准化处理，转换回原始数据的尺度
            pred = pred.cpu().numpy() * (close_max - close_min) + close_min
            label = label.numpy() * (close_max - close_min) + close_min

            # 将预测结果和真实标签添加到之前初始化的列表中
            preds.extend(pred)
            labels.extend(label)
            # 打印预测值和真实值
            for p, l in zip(pred, label):
                print(f'Predicted value: {p:.2f}, Actual value: {l:.2f}')
    # 计算趋势预测的准确性
    predict_trend = ['up' if preds[i] < preds[i + 1] else 'down' for i in range(len(preds) - 1)]
    true_trend = ['up' if labels[i] < labels[i + 1] else 'down' for i in range(len(labels) - 1)]
    # 转换 'up' 为 1, 'down' 为 0
    trend_numeric = [1 if trend == 'up' else 0  for trend in predict_trend]
    true_trend_numeric = [1 if trend == 'up' else 0  for trend in true_trend]
    correct_trend = sum(p == t for p, t in zip(predict_trend, true_trend))
    accuracy = correct_trend / (len(preds) - 1)
    # 计算 RMSE
    mse = np.mean((np.array(preds) - np.array(labels)) ** 2)
    rmse = math.sqrt(mse)

    print('(price) RMSE: %.2f' % rmse)
    print('(trend) Accuracy: %.2f%%' % (accuracy * 100))
    # 创建一个DataFrame
    df = pd.DataFrame(trend_numeric, columns=['Trend'])

    # CSV文件名
    # filename = "GRUWithoutScore.csv"
    filename = "GRUWithScore.csv"

    # 将DataFrame保存为CSV
    df.to_csv(filename, index=False)

    # 绘制所有预测值和真实值的图表
    plt.plot(preds, label='Predicted')
    plt.plot(labels, label='True')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Predicted vs True-GRU with emotion')
    # plt.title('Predicted vs True-GRU without emotion')
    plt.legend()
    plt.savefig('image/Predicted vs True-GRU with emotion')
    # plt.savefig('image/Predicted vs True-GRU without emotion')
    plt.show()

    # 获取混淆矩阵
    cm = confusion_matrix(trend_numeric, true_trend_numeric)

    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 1', 'Class 0'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('heatmap-GRU without emotion')
    # plt.savefig('image/heatmap-GRU without emotion')
    plt.title('heatmap-GRU with emotion')
    plt.savefig('image/heatmap-GRU with emotion')
    plt.show()


eval()
