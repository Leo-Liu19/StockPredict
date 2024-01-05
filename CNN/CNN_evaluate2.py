#导入python中的相关库和本项目中其它文件
import math
import torch
import csv
from sklearn.metrics import confusion_matrix
import seaborn as sns
from CNNModel2 import StockPredictionCNN
from CNN_dataset2 import getData
from parser_my2 import args
import matplotlib.pyplot as plt

# 定义模型参数
class Args:
    #设置输入参数以适配Model中输入
    batch_size = 1
    sequence_length = 5
    input_channels = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpusFile = "Dataset2.csv"#在这里更改所需使用的数据集
    save_file = 'model/stock.pkl'
    output_csv_file = 'output/CNNWithScore.csv'  # 新的CSV文件路径

args = Args()

# 评估模型
def eval():
    # 创建CNN模型实例
    model = StockPredictionCNN(input_channels=args.input_channels, output_size=1)
    model.to(args.device)

    # 加载训练好的模型参数
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    preds = []  # 存储预测值
    labels = []  # 存储真实值

    # 获取数据集相关信息和测试集数据加载器
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)

    # 打开CSV文件准备写入结果
    with open(args.output_csv_file, mode='w', newline='') as csv_file:
        fieldnames = ['pred_value', 'true_value', 'predict_value', 'true_value']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历测试集
        for idx, (x, label) in enumerate(test_loader):
            x = x.to(args.device)
            pred = model(x)
            preds.extend(pred.data.squeeze(1).tolist())
            labels.extend(label.tolist())

        error = 0.0
        total = 0.0
        mistake = 0
        predict = []
        true = []

        for i in range(len(preds)):
            pred_value = preds[i] * (close_max - close_min) + close_min
            label_value = labels[i] * (close_max - close_min) + close_min

            a = (pred_value - label_value) ** 2
            total += a

            # 将字符串标签映射为数值标签
            if i < len(preds) - 1:
                a1 = (preds[i + 1] * (close_max - close_min) - (labels[i + 1] * (close_max - close_min) + close_min))
                b1 = (labels[i + 1] * (close_max - close_min) + close_min)

                if a < a1:
                    predict.append(1.0)  # 'up'
                elif a > a1:
                    predict.append(0.0)  # 'down'
                else:
                    predict.append(0.0)  # 'keep'

                if label_value < b1:
                    true.append(1.0)  # 'up'
                elif label_value > b1:
                    true.append(0.0)  # 'down'
                else:
                    true.append(0.0)  # 'keep'

                if predict[i] != true[i]:
                    mistake += 1

                # 输出预测标签和真实标签
                print('预测值是%.2f,真实值是%.2f,predict_value is %.2f,true_value is %.2f' % (pred_value, label_value, predict[i], true[i]))

                # 将数据写入 CSV 文件
                writer.writerow({'pred_value': pred_value, 'true_value': label_value, 'predict_value': predict[i], 'true_value': true[i]})

        error = math.sqrt(total / len(preds))
        print('(price)rmse为 %.2f' % error)
        accuracy = (mistake / (len(preds) - 1))
        print('(fenlei)accuracy为 %.2f' % accuracy)

        # 绘制所有预测值和真实值的图表
        plt.plot(preds, label='Predicted')
        plt.plot(labels, label='True')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Predicted vs True-CNN without emotion')
        plt.legend()
        plt.savefig('image/Predicted vs True-CNN without emotion')
        plt.show()

        # 获取混淆矩阵
        cm = confusion_matrix(predict, true)

        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 1', 'Class 0'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('heatmap-CNN without emotion')
        plt.savefig('image/heatmap-CNN without emotion')
        plt.show()

eval()
