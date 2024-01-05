import math

from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def eval():
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1) # 构造lstm模型
    model.to(args.device) # 将模型（model）移动到指定设备（args.device）上的操作
    checkpoint = torch.load(args.save_file) # 导入模型参数
    model.load_state_dict(checkpoint['state_dict']) # 加载模型的参数状态字典
    preds = []
    labels = []
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size) # 获取数据
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())

    error = 0.0 # 平均误差
    total = 0.0 # 总误差
    mistake = 0
    predict = []
    true = []
    for i in range(len(preds)):
        print('预测值是%.2f,真实值是%.2f' % (
        preds[i][0] * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))
        a = (preds[i][0] * (close_max - close_min)-(labels[i] * (close_max - close_min) + close_min))
        b = (labels[i] * (close_max - close_min) + close_min)
        total += (a - b)**2
    for i in range(len(preds) - 1):
        a = (preds[i][0] * (close_max - close_min) - (labels[i] * (close_max - close_min) + close_min))
        b = (labels[i] * (close_max - close_min) + close_min)
        a1 =  (preds[i+1][0] * (close_max - close_min) - (labels[i+1] * (close_max - close_min) + close_min))
        b1= (labels[i+1] * (close_max - close_min) + close_min)
        if  a<a1 :
            predict.append(1)
        # elif a > a1:
        #     predict.append('down')
        else:
            predict.append(0)

        if  b<b1 :
            true.append(1)
        # elif b > b1:
        #     true.append('down')
        else:
            true.append(0)
        if predict[i] != true[i]:
            mistake += 1
    error = math.sqrt(total / len(preds))
    print('(price)rmse为 %.2f' %error) # 看看如何改成分类问题
    print('(fenlei)accuracy为 %.2f' %(mistake/((len(preds)-1))))
    # 创建一个DataFrame
    df = pd.DataFrame(predict)

    # 指定CSV文件的路径
    csv_file_path = "LSTMwithScore.csv"
    #csv_file_path = "LSTMwithoutScore.csv"

    # 将DataFrame保存为CSV文件
    df.to_csv(csv_file_path, index=False)

    print(f"CSV文件已保存到 {csv_file_path}")

    # 绘制所有预测值和真实值的图表
    plt.plot(preds, label='Predicted')
    plt.plot(labels, label='True')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Predicted vs True-LSTM with emotion')
    # plt.title('Predicted vs True-LSTM without emotion')
    plt.legend()
    plt.savefig('image/Predicted vs True-LSTM with emotion')
    # plt.savefig('image/Predicted vs True-LSTM without emotion')
    plt.show()

    # 获取混淆矩阵
    cm = confusion_matrix(predict, true)

    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 1', 'Class 0'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('heatmap-LSTM without emotion')
    # plt.savefig('image/heatmap-LSTM without emotion')
    plt.title('heatmap-LSTM with emotion')
    plt.savefig('image/heatmap-LSTM with emotion')
    plt.show()


eval()