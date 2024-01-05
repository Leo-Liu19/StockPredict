from torch.autograd import Variable
import torch.nn as nn
import torch
from parser_my import args
from dataset import getData
from GRUModel import GRUModel
import matplotlib.pyplot as plt


def train():

    # 使用 GRUModel
    # 初始化模型，传入各种参数。这些参数通常由外部（如命令行参数）提供
    model = GRUModel(input_size=args.input_size,
                     hidden_size=args.hidden_size,
                     num_layers=args.layers,
                     output_size=1,
                     dropout=args.dropout,
                     batch_first=args.batch_first)

    model.to(args.device)  # 将模型移动到指定的设备，例如GPU

    # 定义损失函数，这里使用均方误差损失（MSE）用于回归任务
    criterion = nn.MSELoss()

    # 定义优化器，这里使用Adam优化器，并设置学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 调用 getData 函数加载和处理数据集
    # 设这个函数返回处理后的最大值、最小值（用于数据标准化）、训练数据加载器和测试数据加载器
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)

    losses = []  # 用于绘图
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader): # 遍历每个批次
            if args.useGPU: #cuda
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1)) # 输入进模型获得预测
                # pred = pred[1, :, :]
                #pred = pred[:, -1]
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
    plt.title('Training Loss Over Epochs-GRU with emotion')
    # plt.title('Training Loss Over Epochs-GRU without emotion')
    plt.legend()
    plt.savefig('image/GRU-withScore-training_loss_plot.png')  # 保存图像为PNG文件
    # plt.savefig('image/GRU-withoutScore-training_loss_plot.png')  # 保存图像为PNG文件
    plt.show()

train()