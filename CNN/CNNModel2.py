import torch.nn as nn#导入python中的相关库

class StockPredictionCNN(nn.Module):#设置CNN股票预测模型的参数

    def __init__(self, input_channels=1, output_size=1):
        super(StockPredictionCNN, self).__init__()

        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        # 第三层卷积层
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        # 展平层，将多维数据展平成一维
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc1 = nn.Linear(12, output_size)

    def forward(self, x):
        # 第一层卷积操作
        x = self.pool1(self.relu1(self.conv1(x)))


        # 第二层卷积操作
        x = self.pool2(self.relu2(self.conv2(x)))


        # 第三层卷积操作
        x = self.pool3(self.relu3(self.conv3(x)))
        # print(f"Size after pool3: {x.size()}")

        # 动态计算最后一个池化层的输出大小
        pool3_output_size = x.size()[1:]  # 取除了 batch_size 外的维度
        x = self.flatten(x)
        self.fc1 = nn.Linear(pool3_output_size.numel(), output_size)  # 更新 fc1 层的输入大小
        x = self.fc1(x)

        return x

# 示例用法
output_size = 1
cnn_model = StockPredictionCNN(input_channels=8, output_size=output_size)
