import math

from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch


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
            predict.append('up')
        elif a > a1:
            predict.append('down')
        else:
            predict.append('keep')
        if  b<b1 :
            true.append('up')
        elif b > b1:
            true.append('down')
        else:
            true.append('keep')
        if predict[i] != true[i]:
            mistake += 1
    error = math.sqrt(total / len(preds))
    print('(price)rmse为 %.2f' %error) # 看看如何改成分类问题
    print('(fenlei)accuracy为 %.2f' %(mistake/((len(preds)-1))))


eval()