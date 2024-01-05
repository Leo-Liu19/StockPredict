import pandas as pd
import math
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from parser import args
from dataset import getData
from RNNModel import rnn

def eval():
    # 加载模型和参数
    model = rnn(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])

    # 获取数据
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    preds = []
    labels = []

    # 获取模型预测和真实标签
    with torch.no_grad():
        for x, label in test_loader:
            x = x.squeeze(1).to(args.device)
            pred = model(x)
            preds.extend(pred.squeeze().cpu().numpy())
            labels.extend(label.squeeze().cpu().numpy())
            # 将预测值和真实值存储到列表中（确保它们非空且长度一致）


    # 反标准化
    for i in range(len(preds)):
        preds[i] = preds[i] * (close_max - close_min) + close_min
        labels[i] = labels[i] * (close_max - close_min) + close_min

    # 计算均方根误差
    rmse = math.sqrt(sum((p - l) ** 2 for p, l in zip(preds, labels)) / len(preds))
    print('(price) rmse为 %.2f' % rmse)

    # 转换为分类，并记录在DataFrame中
    classify_labels = [1 if labels[i] < labels[i + 1] else 0 for i in range(len(labels) - 1)]
    classify_preds = [1 if preds[i] < preds[i + 1] else 0 for i in range(len(preds) - 1)]

    # 创建DataFrame来保存数据
    df = pd.DataFrame({
        'Predict': classify_preds
    })

    # 计算分类准确率
    correct_classify = sum(p == l for p, l in zip(classify_preds, classify_labels))
    accuracy = correct_classify / (len(preds) - 1)
    print('(分类)accuracy为 %.2f' % accuracy)

    # 保存到Excel文件中
    #df.to_excel('RNN-withScore.xlsx', index=False)

    # 绘制所有预测值和真实值的图表
    plt.plot(preds, label='Predicted')
    plt.plot(labels, label='True')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Predicted vs True-RNN with emotion')
    #plt.title('Predicted vs True-RNN without emotion')
    plt.legend()
    plt.savefig('image/Predicted vs True-RNN with emotion')
    #plt.savefig('image/Predicted vs True-RNN without emotion')
    plt.show()

    # 获取混淆矩阵
    cm = confusion_matrix(classify_preds,classify_labels)

    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 1', 'Class 0'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    #plt.title('heatmap-RNN without emotion')
    #plt.savefig('image/heatmap-RNN without emotion')
    plt.title('heatmap-RNN with emotion')
    plt.savefig('image/heatmap-RNN with emotion')
    plt.show()
eval()