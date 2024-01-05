from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData
import matplotlib.pyplot as plt


def train():

    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device) # 将模型（model）移动到指定设备（args.device）上的操作
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001

    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size ) # 详情请进入方法
    losses = [] # 用于绘图
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader): # 遍历每个批次
            if args.useGPU: # 可以先放放，我们不用cuda
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1)) # 输入进模型获得预测
                pred = pred[1, :, :]
                label = label.unsqueeze(1)
            loss = criterion(pred, label)   # 计算损失函数
            optimizer.zero_grad()      # 梯度清零
            loss.backward()     #  反向传播
            optimizer.step() # 更新模型参数
            total_loss += loss.item() # 计算整个训练过程的总损失
        print(total_loss)  # 打印本轮训练的总损失

        # 计算平均损失并添加到列表中（用于绘图）
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)

        if i % 10 == 0:  # 每十轮保存一次模型
            # torch.save(model, args.save_file)
            torch.save({'state_dict': model.state_dict()}, args.save_file) # 将模型的状态字典保存到文件中
            print('第%d epoch，保存模型' % i)
    # torch.save(model, args.save_file)
    torch.save({'state_dict': model.state_dict()}, args.save_file)  # 保存最终模型
    # 绘图
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs-LSTM without emotion')
    plt.title('Training Loss Over Epochs-LSTM with emotion')
    plt.legend()
    # plt.savefig('image/LSTM-withoutScore-training_loss_plot.png')  # 保存图像为PNG文件
    plt.savefig('image/LSTM-withScore-training_loss_plot.png')  # 保存图像为PNG文件
    plt.show()

train()