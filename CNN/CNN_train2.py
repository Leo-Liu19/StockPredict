#导入python中的相关库和本项目中其它文件
import torch.nn as nn
import torch
from parser_my2 import args
from CNN_dataset2 import getData
from CNNModel2 import StockPredictionCNN  #  导入CNN模型
import matplotlib.pyplot as plt
import warnings

# 忽略Matplotlib的警告，因为无中文字体要求
warnings.filterwarnings("ignore", category=UserWarning)

def train():

    # 创建CNN模型实例
    model = StockPredictionCNN(input_channels=args.input_size, output_size=1)  # 使用args.output_size而不是硬编码的1
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 使用Adam优化器

    # 获取数据集相关信息和训练集、测试集数据加载器
    close_max, close_min, train_loader, test_loader = getData('Dataset3.csv', args.sequence_length, args.batch_size)
    losses = []  # 用于存储每轮训练的损失值，以便绘制图表
    for i in range(args.epochs):  # 遍历训练轮数
        total_loss = 0

        for idx, (data, label) in enumerate(train_loader):
            #print(f"Input data shape: {data.shape}")  # 这些print都是用于调试的
            if args.useGPU:
                data = data.cuda()
                label = label.cuda()
            else:
                data = data
                label = label

            optimizer.zero_grad()  # 梯度清零
            output = model(data)  # 前向传播
            #print(f"Output shape: {output.shape}")
            output = output.squeeze()  # 确保输出形状正确
            loss = criterion(output, label)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            total_loss += loss.item()

        print(f'Epoch {i + 1}/{args.epochs}, Loss: {total_loss}')  # 打印本轮训练的总损失
        print(total_loss)  # 打印本轮训练的总损失

        # 计算平均损失并添加到列表中（用于绘图）
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)

        if i % 10 == 0:  # 每十轮保存一次模型
            torch.save({'state_dict': model.state_dict()}, args.save_file)
            print('第%d epoch，保存模型' % i)

    torch.save({'state_dict': model.state_dict()}, args.save_file)
    # 绘图
    plt.plot(losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs-LSTM without emotion')
    plt.title('训练损失随Epoch变化-CNN without emotion')
    plt.legend()
    # plt.savefig('image/LSTM-withoutScore-training_loss_plot.png')  # 保存图像为PNG文件
    plt.savefig('image/CNN-withScore-training_loss_plot.png')  # 保存图像为PNG文件
    plt.show()

train()
